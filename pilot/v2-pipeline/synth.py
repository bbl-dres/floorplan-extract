"""Style-randomised renderer v2: a floor record -> plan image, label mask, style tags and extra targets.

    r = Renderer([(DATA / "floors-train.pkl", 0.9), (DATA / "floors-ifc.pkl", 0.1)])
    img, lab, tags = r.sample(rng)                          # as in v1
    img, lab, tags, tg = r.sample(rng, targets=True)        # plus tg = {"interior", "boundary", "void", "text"}

Each sample is a square crop of one floor (Swiss Dwellings, or an IFC storey from ifc_prepare.py) at s = PX_PER_M x k
pixels per metre. Half the samples draw k from the v1 range 0.8-1.25, half log-uniformly from 0.6-1.6: the pipeline
resamples sheets to 50 px/m when the scale is known, but scale guesses from title blocks, reduced prints or door
widths can be off by a factor of 1.5 or more, and the scale-proposal stage runs the segmenter at candidate scales.
Image and label map are drawn at the same resolution, so labels are never resampled.

Labels (common.CLASSES): background, wall, door, window, column, stairs. Extra targets (common.HEADS), uint8 0/1 maps:
  interior  inside the building outline: walls, openings, columns and all indoor areas (rooms, shafts, voids);
            outdoor areas (balcony, loggia, terrace, patio, garden, light well) and everything outside are 0
  boundary  outlines of all area polygons, 3 px wide, including open-plan boundaries that have no wall
  void      voids: Swiss Dwellings VOID and AIR areas (Luftraum, stairwell voids)
  text      boxes of all rendered text: stamps, dimensions, axis labels, title blocks, legends, annotations
Panels (title block, legend, key plan) set all labels and targets beneath them to 0 except their text boxes.

Styles (v1): walls solid, outlined, hatched, grey or coloured (thick and thin walls may differ); doors with swing, leaf
only or as gaps; windows as glass lines or gaps; stairs with treads; fixtures; room stamps (DE/FR/IT, areas, numbers,
AOIDs) in print, hand, calligraphic or Fraktur lettering; dimension chains, axes; scan defects.
New in v2 (all labelled background unless stated):
  hatched regions that are not walls: terraces and balconies, floor finishes in rooms, roof slopes, terrain, paving,
  neighbouring buildings and section snippets outside the building; ceiling ornament (cornices, coffers, rosettes,
  garlands); window reveals and wall niches (cut out of the wall label); pilasters (wall label) and plinth lines;
  casement windows with sash swings (window label); Swiss material hatching per wall thickness (concrete, masonry,
  sand-lime, stone, drywall, insulation band); column grids with axis lines and bubbles (column label), and floors
  and crops with columns or stairs oversampled; voids drawn with a diagonal cross; colour zones (fire compartments),
  rubber stamps, red-pen revisions and hand annotations; title blocks, legends, key plans and a second drawing on
  the same sheet (labelled like the main drawing); scan defects: folds, shadows, bleed-through, ink fading,
  perspective, stronger resolution loss, dithered 1-bit scans, toner streaks.
Everything random comes from the numpy Generator passed in, so a seed reproduces a sample in any process.
"""
import pickle
import zlib
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import shapely
from PIL import Image, ImageDraw, ImageFont

from common import BG, COLUMN, DOOR, FONTS, HEADS, PX_PER_M, STAIRS, WALL, WINDOW

VERSION = "2.0"
OUTDOOR = {"BALCONY", "LOGGIA", "TERRACE", "PATIO", "GARDEN", "OUTDOOR_VOID", "LIGHTWELL"}
VOIDS = {"VOID", "AIR"}
BOUNDARY_PX = 3
IGNORE = 255                                            # target value: unknown, not trained on

NAMES = {
    "office": ["Büro", "Bureau", "Ufficio", "Sitzungszimmer", "Salle de réunion", "Sala riunioni", "Besprechung",
               "Empfang", "Réception", "Archiv", "Archives", "Archivio", "Kopierraum", "Server", "Labor", "Bibliothek",
               "Grossraumbüro", "Direktion", "Sekretariat", "Cafeteria"],
    "ROOM": ["Zimmer", "Raum", "Chambre", "Camera", "Kammer", "Stube", "Saal", "Salon", "Studio", "Arbeiten"],
    "BEDROOM": ["Schlafen", "Schlafzimmer", "Chambre", "Camera da letto", "Gast"],
    "LIVING": ["Wohnen", "Wohnzimmer", "Séjour", "Soggiorno", "Essen", "Salle à manger", "Pranzo", "Salon"],
    "KITCHEN": ["Küche", "Cuisine", "Cucina", "Teeküche", "Office"],
    "BATHROOM": ["Bad", "WC", "WC D", "WC H", "WC IV", "Dusche", "Salle de bain", "Bagno", "Toilette", "Sanitär", "Lavabo"],
    "CORRIDOR": ["Korridor", "Gang", "Flur", "Couloir", "Corridoio", "Halle", "Vorplatz", "Entrée", "Eingang", "Foyer", "Hall", "Dégagement"],
    "STAIRCASE": ["Treppe", "Treppenhaus", "Escalier", "Scala", "Tr.H."],
    "STOREROOM": ["Abstellraum", "Reduit", "Réduit", "Ripostiglio", "Lager", "Dépôt", "Deposito", "Keller", "Estrich", "Putzraum"],
    "SHAFT": ["Lift", "Ascenseur", "Schacht", "Gaine", "Steigzone", "Install.", "Elektro", "Technik"],
    "OUTSIDE": ["Balkon", "Loggia", "Terrasse", "Balcon", "Sitzplatz"],
    "VOID": ["Luftraum", "LR", "Luftraum über EG", "Leerraum", "Vide", "Vide sur rez", "Vuoto", "Spazio vuoto"],
}
TYPE_GROUP = {"BEDROOM": "BEDROOM", "LIVING_ROOM": "LIVING", "LIVING_DINING": "LIVING", "DINING": "LIVING",
              "KITCHEN": "KITCHEN", "KITCHEN_DINING": "KITCHEN", "BATHROOM": "BATHROOM", "SANITARY_ROOMS": "BATHROOM",
              "CORRIDOR": "CORRIDOR", "CORRIDORS_AND_HALLS": "CORRIDOR", "LOBBY": "CORRIDOR", "FOYER": "CORRIDOR",
              "STAIRCASE": "STAIRCASE", "STOREROOM": "STOREROOM", "BASEMENT_COMPARTMENT": "STOREROOM",
              "SHAFT": "SHAFT", "ELEVATOR": "SHAFT", "BALCONY": "OUTSIDE", "LOGGIA": "OUTSIDE", "TERRACE": "OUTSIDE",
              "VOID": "VOID", "AIR": "VOID"}
PASTELS = [(255, 236, 179), (200, 230, 201), (187, 222, 251), (248, 187, 208), (225, 190, 231), (255, 204, 188),
           (220, 237, 200), (178, 235, 242), (255, 249, 196), (215, 204, 200)]
WALL_COLOURS = [(200, 40, 40), (230, 120, 40), (150, 90, 60), (90, 90, 200), (240, 170, 170), (60, 140, 60)]
ZONE_COLOURS = [(230, 40, 40), (40, 160, 60), (40, 90, 220), (240, 190, 0), (240, 120, 20), (150, 60, 200)]
PEN_COLOURS = [(30, 50, 170), (200, 30, 30), (30, 30, 30), (85, 85, 85)]             # blue, red, black, pencil
STAMP_TEXT = [["GENEHMIGT"], ["BEWILLIGT", "Baupolizei"], ["KOPIE"], ["ARCHIV"], ["APPROUVÉ"], ["VISTO"], ["ANNULLIERT"],
              ["ERSETZT DURCH", "Plan Nr. {n}"], ["Eingang", "{d}"], ["GEPRÜFT", "{d}"], ["PLAN ÉCHU"], ["Bestand"],
              ["Bauamt", "{d}"], ["Service des bâtiments", "{d}"], ["Ufficio tecnico"], ["Nachgeführt", "{d}"]]
ANNOTATIONS = ["neu", "Abbruch", "prüfen!", "Türe versetzt", "nicht ausgeführt", "Wand neu 12 cm", "à vérifier",
               "da demolire", "OK", "siehe Detail", "Höhe 2.40", "Massaufnahme", "Fenster ersetzt", "? m²", "zu",
               "Lüftung", "à démolir", "nuovo", "verputzt", "Kamin", "Brüstung 90", "UK Sturz 2.10", "geändert", "x"]
TITLE_ROWS = [["Grundriss Erdgeschoss", "Grundriss 1. Obergeschoss", "Grundriss 2. OG", "Untergeschoss", "Dachgeschoss",
               "Plan du rez-de-chaussée", "1er étage", "Pianta piano terreno", "Primo piano"],
              ["Massstab 1:100", "Mst. 1:50", "Echelle 1:100", "Scala 1:50", "M 1:200", "1:100"],
              ["Plan Nr. {n}", "Plan-Nr. {n}", "No {n}", "Index {i}", "Format A1", "Format A3"],
              ["Datum {d}", "{d}", "gez. {a}", "dess. {a}", "gepr. {a}", "rev. {d}"],
              ["Umbau und Sanierung", "Verwaltungsgebäude", "Bestandesaufnahme", "Transformation", "Ristrutturazione",
               "Projekt", "Ausführungsplan", "Baueingabe", "Mise à l'enquête"]]
LEGEND = ["Beton", "Stahlbeton", "Mauerwerk", "Backstein", "Kalksandstein", "Wärmedämmung", "Leichtbauwand", "Naturstein",
          "Bestehend", "Neu", "Abbruch", "Béton", "Maçonnerie", "Isolation", "Calcestruzzo", "Muratura", "Isolamento",
          "Brandabschnitt EI 60", "Fluchtweg", "Plattenbelag", "Parkett", "Terrain"]
MATERIALS = {   # per wall-thickness class: Swiss material conventions (SIA 400 style)
    "thick": ["concrete_grey", "concrete_black", "masonry", "concrete_dots", "sandlime"],
    "medium": ["masonry", "sandlime", "concrete_grey", "outline", "masonry"],
    "thin": ["outline", "drywall_cross", "light_grey", "outline"],
}
HIST_MATERIALS = {"thick": ["stone", "masonry", "concrete_black"], "medium": ["masonry", "stone", "outline"],
                  "thin": ["outline", "light_grey"]}


# ---------- geometry helpers ----------

def polys(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    return [p for part in getattr(g, "geoms", []) for p in polys(part)]


def rings(p):
    return [np.asarray(p.exterior.coords)] + [np.asarray(r.coords) for r in p.interiors]


def fx(a):
    """Pixel coordinates -> fixed point (4 fractional bits) for sub-pixel OpenCV drawing."""
    return np.round(np.asarray(a) * 16).astype(np.int32)


def ring_coords(g, offset=None):
    """All rings (exteriors and holes) of the polygons in g as fixed-point int32 arrays, in one vectorised pass."""
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":                         # common case: plain attribute access is faster
        rs = [np.asarray(g.exterior.coords)] + [np.asarray(r.coords) for r in g.interiors]
        return [fx(r - offset if offset is not None else r) for r in rs]
    parts = shapely.get_parts(g)
    t = shapely.get_type_id(parts)
    if (t >= 4).any():                                   # nested multi-geometries or collections
        parts = np.array(polys(g), dtype=object)
    else:
        parts = parts[t == 3]
    if not len(parts):
        return []
    coords, idx = shapely.get_coordinates(shapely.get_rings(parts), return_index=True)
    if offset is not None:
        coords = coords - offset
    return np.split(fx(coords), np.flatnonzero(np.diff(idx)) + 1)


def fill(img, g, colour, offset=None):
    rs = [r for r in ring_coords(g, offset) if len(r) >= 3]
    if rs:
        cv2.fillPoly(img, rs, colour, lineType=cv2.LINE_AA if img.ndim == 3 else cv2.LINE_8, shift=4)


def stroke(img, g, colour, w):
    rs = [r for r in ring_coords(g) if len(r) >= 2]
    if rs:
        cv2.polylines(img, rs, True, colour, max(1, int(round(w))), cv2.LINE_AA if img.ndim == 3 else cv2.LINE_8, shift=4)


def local_mask(g, n, pad=1):
    """Raster of g restricted to its bounding box in an n x n image: (bool mask, (y slice, x slice)) or None."""
    if g is None or g.is_empty:
        return None
    x0, y0, x1, y1 = g.bounds
    X0, Y0 = max(int(np.floor(x0)) - pad, 0), max(int(np.floor(y0)) - pad, 0)
    X1, Y1 = min(int(np.ceil(x1)) + pad + 1, n), min(int(np.ceil(y1)) + pad + 1, n)
    if X1 <= X0 or Y1 <= Y0:
        return None
    m = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
    fill(m, g, 1, offset=np.array([X0, Y0], float))
    return m > 0, (slice(Y0, Y1), slice(X0, X1))


def line(img, a, b, colour, w):
    cv2.line(img, tuple(fx(a)), tuple(fx(b)), colour, max(1, int(round(w))), cv2.LINE_AA, shift=4)


def polyline(img, pts, colour, w, closed=False):
    cv2.polylines(img, [fx(pts)], closed, colour, max(1, int(round(w))), cv2.LINE_AA, shift=4)


def dashed(img, pts, colour, w, pattern=(6, 4), closed=False, phase=0.0):
    """Polyline drawn with a dash pattern (on, off, on, off, ... in px): resampled at 1 px, drawn in one call."""
    pts = np.asarray(pts, float)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    if len(pts) < 2:
        return
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    total, period = cum[-1], float(sum(pattern))
    if total <= 1:
        return
    ts = np.append(np.arange(0.0, total, 1.0), total)
    bounds = np.cumsum(pattern)
    on = np.searchsorted(bounds, (ts + phase) % period, side="right") % 2 == 0
    P = fx(np.stack([np.interp(ts, cum, pts[:, 0]), np.interp(ts, cum, pts[:, 1])], 1))
    edges = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
    runs = [r for r, o in zip(np.split(P, edges), np.split(on, edges)) if o[0] and len(r) >= 2]
    if runs:
        cv2.polylines(img, runs, False, colour, max(1, int(round(w))), cv2.LINE_AA, shift=4)


def rect_axes(p):
    """Minimum rotated rectangle of a polygon: (centre, unit vector along the long side, length, thickness)."""
    c = np.asarray(p.minimum_rotated_rectangle.exterior.coords)[:4]
    e0, e1 = c[1] - c[0], c[2] - c[1]
    if np.hypot(*e0) < np.hypot(*e1):
        e0, e1 = e1, e0
    L, T = np.hypot(*e0), np.hypot(*e1)
    return c.mean(axis=0), e0 / max(L, 1e-9), L, T


def at(m, q):
    """Value of a raster at a point, 0 outside."""
    x, y = int(round(q[0])), int(round(q[1]))
    return m[y, x] if 0 <= x < m.shape[1] and 0 <= y < m.shape[0] else 0


def stable_index(key, n):
    """Deterministic index for a string; Python's hash() differs between processes."""
    return zlib.crc32(str(key).encode("utf-8")) % n


def clip_line(q, d, n, margin=0.0):
    """Segment of the line q + t d inside the square [-margin, n + margin]^2, or None."""
    lo, hi = -np.inf, np.inf
    for k in range(2):
        if abs(d[k]) < 1e-9:
            if not -margin <= q[k] <= n + margin:
                return None
            continue
        t0, t1 = (-margin - q[k]) / d[k], (n + margin - q[k]) / d[k]
        lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
    return (q + lo * d, q + hi * d) if hi > lo else None


@lru_cache(maxsize=256)
def font(path, size):
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=128)
def _lines_tex(n, angle, spacing, width):
    t = np.zeros((n, n), np.uint8)
    a = np.radians(angle)
    d = np.array([np.cos(a), np.sin(a)])
    nrm = np.array([-d[1], d[0]])
    for k in np.arange(-1.5 * n, 1.5 * n, spacing):
        o = n / 2 + nrm * k
        cv2.line(t, tuple(fx(o - d * 2 * n)), tuple(fx(o + d * 2 * n)), 1, width, cv2.LINE_8, shift=4)
    t.setflags(write=False)
    return t


def lines_tex(n, angle, spacing, width=1, phase=0):
    """Boolean texture of parallel lines (cached per angle in whole degrees and spacing in whole pixels)."""
    t = _lines_tex(n, int(round(angle)) % 180, max(2, int(round(spacing))), int(width))
    return np.roll(t, int(phase) % max(2, int(round(spacing))), axis=0) > 0 if phase else t > 0


@lru_cache(maxsize=8)
def _stone_tex(n, variant):
    """Natural-stone hatch: lines at varying angle and spacing with gaps (a few fixed variants, rolled per use)."""
    r = np.random.default_rng(7919 + variant)
    t = np.zeros((n, n), np.uint8)
    ang = np.radians((45, 135)[variant % 2] + r.normal(0, 6))
    d = np.array([np.cos(ang), np.sin(ang)])
    nrm = np.array([-d[1], d[0]])
    segs, k = [], -0.8 * n
    while k < 0.8 * n:
        o = n / 2 + nrm * k
        for a in np.arange(-0.8 * n, 0.8 * n, 40):
            if r.random() < 0.75:
                segs.append(fx([o + d * a, o + d * (a + r.uniform(15, 40))]))
        k += r.uniform(3, 9)
    cv2.polylines(t, segs, False, 1, 1, cv2.LINE_8, shift=4)
    t.setflags(write=False)
    return t


@lru_cache(maxsize=4096)
def _text_tile(s_txt, fnt, size, angle):
    f = font(fnt, size)
    l, t, r, bt = f.getbbox(s_txt)
    if r - l <= 0 or bt - t <= 0:
        return None
    tile = Image.new("L", (r - l + 4, bt - t + 4), 0)
    ImageDraw.Draw(tile).text((2 - l, 2 - t), s_txt, fill=255, font=f)
    if angle:
        tile = tile.rotate(angle, expand=True, resample=Image.NEAREST if angle % 90 == 0 else Image.BILINEAR)
    m = np.asarray(tile, np.float32) / 255
    m.setflags(write=False)
    return m


@lru_cache(maxsize=4)
def _grid(n):
    gx, gy = np.meshgrid(np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32))
    gx.setflags(write=False)
    gy.setflags(write=False)
    return gx, gy


@lru_cache(maxsize=16)
def _zigzag_tex(n, period, amp, vertical):
    t = np.zeros((n + 2 * amp, n + 2 * amp), np.uint8)
    xs = np.arange(0, n + 2 * amp + period, period / 2)
    for y0 in np.arange(0, n + 2 * amp, amp):
        ys = y0 + amp * (np.arange(len(xs)) % 2)
        cv2.polylines(t, [fx(np.c_[xs, ys])], False, 1, 1, cv2.LINE_8, shift=4)
    t = t[amp:amp + n, amp:amp + n]
    t = np.ascontiguousarray(t.T if vertical else t)
    t.setflags(write=False)
    return t


# ---------- renderer ----------

class Renderer:
    def __init__(self, floors_path, size=512, scale=(0.6, 1.6), scale_core=0.5, column_floors=0.0, focus=()):
        """floors_path: one floors pickle, or a list of (path, weight) or (path, weight, {"include"|"exclude": sites})
        to mix sources, e.g. Swiss Dwellings and IFC storeys (site = IFC project).
        scale: range of the scale factor around PX_PER_M; scale_core: share of samples drawn from the v1 range 0.8-1.25.
        column_floors: share of samples drawn from floors that have columns (columns are rare in Swiss Dwellings).
        focus: (element, probability) pairs: centre the crop on a random column or stair of the floor.
        The defaults sample floors and crops uniformly; train.py switches the oversampling on."""
        specs = [(floors_path, 1.0)] if isinstance(floors_path, (str, Path)) else list(floors_path)
        self.sources, weights = [], []
        for spec in specs:
            blobs = pickle.loads(Path(spec[0]).read_bytes())
            sel = spec[2] if len(spec) > 2 and spec[2] else None
            if sel:
                inc, exc = set(sel.get("include", ())), set(sel.get("exclude", ()))
                blobs = [b for b in blobs if (lambda site: (not inc or site in inc) and site not in exc)(
                    pickle.loads(zlib.decompress(b)).get("site"))]
            if blobs and spec[1] > 0:
                self.sources.append(blobs)
                weights.append(float(spec[1]))
        assert self.sources, f"no floors in {floors_path}"
        w = np.array(weights, float)
        self.weights = w / w.sum()
        self.floors = [f for src in self.sources for f in src]
        self.fonts = [str(f) for f in sorted(Path(FONTS).glob("*.ttf"))]
        assert self.fonts, f"no fonts in {FONTS} (set V2_FONTS)"
        self.hand = [f for f in self.fonts if any(k in f for k in ("Architects", "Caveat", "Kalam"))]
        self.calli = [f for f in self.fonts if any(k in f for k in ("GreatVibes", "Pinyon", "Unifraktur", "IMFe"))]
        self.print = [f for f in self.fonts if f not in self.hand and f not in self.calli]
        self.size = size
        self.scale, self.scale_core, self.column_floors = tuple(scale), float(scale_core), float(column_floors)
        self.focus = dict(focus or ())
        self._text_mask = None
        self._elements = set()
        self._footprints = {}

    def floor(self, i):
        return pickle.loads(zlib.decompress(self.floors[i]))

    # ----- style -----
    def style(self, rng, **force):
        """Random graphical style. Keyword arguments override single choices after drawing them (previews, tests)."""
        era = rng.choice(["cad", "historical", "colour", "mixed"], p=[0.4, 0.25, 0.15, 0.2])
        hist = era == "historical"
        st = {"era": era}
        if era == "cad":
            st["walls"] = rng.choice(["outline", "hatch", "grey", "solid", "material"], p=[0.3, 0.2, 0.15, 0.1, 0.25])
        elif hist:
            st["walls"] = rng.choice(["solid", "hatch", "outline", "material"], p=[0.55, 0.15, 0.15, 0.15])
        elif era == "colour":
            st["walls"] = "colour"
        else:
            st["walls"] = "split"                       # thick and thin walls drawn differently
        st["thick"], st["thin"] = rng.choice(["solid", "hatch", "grey", "colour", "material"]), rng.choice(["outline", "hatch", "grey"])
        st["pen"] = float(rng.choice([1, 1.5, 2, 3, 4], p=[0.2, 0.25, 0.25, 0.2, 0.1]))
        st["thin_pen"] = float(rng.choice([1, 1, 1.5]))
        st["doors"] = rng.choice(["arc", "leaf", "gap", "arc_dashed", "sliding"],
                                 p=[0.5, 0.1, 0.25, 0.1, 0.05] if not hist else [0.2, 0.1, 0.65, 0.05, 0.0])
        st["windows"] = rng.choice(["three", "two", "rect", "glass", "gap", "casement", "casement_dashed"],
                                   p=[0.25, 0.15, 0.15, 0.12, 0.13, 0.15, 0.05] if not hist else [0.12, 0.2, 0.05, 0.05, 0.4, 0.13, 0.05])
        st["room_fill"] = rng.random() < (0.85 if era == "colour" else 0.08)
        st["patterns"] = rng.random() < 0.2
        st["text"] = rng.random() < 0.85
        st["font"] = rng.choice(self.calli if hist and rng.random() < 0.6 else
                                self.hand if rng.random() < 0.3 else self.print)
        st["dims"] = rng.random() < 0.5
        st["axes"] = rng.random() < 0.2
        st["features"] = rng.random() < 0.85
        st["ink"] = tuple(int(v) for v in (rng.choice([0, 20, 40]),) * 3) if rng.random() < 0.8 else \
            tuple(int(v) for v in rng.choice([(70, 40, 20), (90, 30, 20), (30, 30, 80)]))   # sepia, red-brown, blue
        st["wall_colour"] = tuple(int(v) for v in WALL_COLOURS[rng.integers(len(WALL_COLOURS))])
        st["grey"] = int(rng.integers(90, 200))
        # v2
        mats = HIST_MATERIALS if hist else MATERIALS
        st["materials"] = {k: str(rng.choice(v)) for k, v in mats.items()}
        st["insulation"] = bool(not hist and rng.random() < 0.6)
        st["column_style"] = str(rng.choice(["solid", "x", "hatch", "outline", "grey"], p=[0.35, 0.2, 0.2, 0.15, 0.1]))
        st["outdoor_hatch"] = bool(rng.random() < 0.45)        # terraces, balconies
        st["room_hatch"] = float(rng.choice([0, 0, 0.1, 0.3]))  # share of rooms with a floor-finish hatch
        st["context"] = bool(rng.random() < 0.4)              # roofs, terrain, neighbours, paving outside
        st["ornament"] = bool(rng.random() < (0.4 if hist else 0.1))
        st["reveals"] = bool(rng.random() < (0.5 if hist else 0.25))
        st["niches"] = bool(rng.random() < (0.35 if hist else 0.12))
        st["pilasters"] = bool(rng.random() < (0.3 if hist else 0.08))
        st["column_grid"] = bool(rng.random() < 0.15)
        st["void_cross"] = bool(rng.random() < 0.7)
        st["zones"] = bool(rng.random() < 0.08)
        st["rubber_stamp"] = bool(rng.random() < 0.08)
        st["red_pen"] = bool(rng.random() < 0.08)
        st["annotations"] = bool(rng.random() < (0.2 if hist else 0.1))
        st["title_block"] = bool(rng.random() < 0.08)
        st["legend"] = bool(rng.random() < 0.06)
        st["keyplan"] = bool(rng.random() < 0.04)
        st["multi"] = bool(rng.random() < 0.07)               # a second drawing on the same sheet
        st.update(force)
        return st

    # ----- sample -----
    def sample(self, rng, index=None, targets=False, style=None):
        """Render one sample: (img, lab, tags), or (img, lab, tags, targets) with targets=True. The image and labels do
        not depend on `targets`. style: optional overrides of style choices (previews; 'defects' forces scan defects).
        A floor with broken geometry is skipped for another one."""
        for attempt in range(10):
            try:
                img, lab, tags, tg = self._sample(rng, index if attempt == 0 else None, style)
                return (img, lab, tags, tg) if targets else (img, lab, tags)
            except shapely.errors.GEOSException:
                continue
        raise RuntimeError("no renderable floor found")

    def scale_factor(self, rng):
        lo, hi = self.scale
        if rng.random() < self.scale_core:
            return float(rng.uniform(max(lo, 0.8), min(hi, 1.25)))
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    def _pick(self, rng, index=None, s=None):
        """Floor record, pixels per metre and crop centre (floor metres)."""
        n = self.size
        want_columns = index is None and rng.random() < self.column_floors
        for attempt in range(40):
            if index is not None:
                f = self.floor(index)
            else:
                src = self.sources[int(rng.choice(len(self.sources), p=self.weights))]
                f = pickle.loads(zlib.decompress(src[int(rng.integers(len(src)))]))
                if want_columns and f["columns"].is_empty and attempt < 25:
                    continue
            sc = PX_PER_M * self.scale_factor(rng) if s is None else s
            half = n / sc / 2
            x0, y0, x1, y1 = f["walls"].bounds
            target = self._focus(f, rng)
            for _ in range(10):
                c = rng.uniform([x0, y0], [x1, y1]) if target is None else target + rng.uniform(-0.6, 0.6, 2) * half
                win = shapely.box(*(c - half), *(c + half))
                if f["walls"].intersection(win).area > 0.02 * (2 * half) ** 2:
                    return f, sc, c
                target = None
        return f, sc, c

    def _focus(self, f, rng):
        r = rng.random()
        pc, ps = self.focus.get("columns", 0.0), self.focus.get("stairs", 0.0)
        if r < pc and not f["columns"].is_empty:
            cs = polys(f["columns"])
            return np.asarray(cs[int(rng.integers(len(cs)))].centroid.coords[0])
        if pc <= r < pc + ps and f["stairs"]:
            return np.asarray(f["stairs"][int(rng.integers(len(f["stairs"])))].centroid.coords[0])
        return None

    def _project(self, f, s, c, theta):
        n = self.size
        t = np.radians(theta)
        A = s * np.array([[np.cos(t), -np.sin(t)], [-np.sin(t), -np.cos(t)]])
        b = n / 2 - A @ c
        r = n / s / 2 * 1.5
        clip = lambda g: shapely.clip_by_rect(g, c[0] - r, c[1] - r, c[0] + r, c[1] + r)
        px = lambda g: shapely.transform(clip(g), lambda xy: xy @ A.T + b)
        g = {k: px(f[k]) for k in ("walls", "doors", "windows", "columns", "railings")}
        stairs = [px(p) for p in f["stairs"]]
        feats = [(k, px(p)) for k, p in f["features"]]
        areas = [(k, px(p), p.area) for k, p in f["areas"]]
        if "licence" in f:                               # IFC storey: spaces are often incomplete, voids not modelled
            hull, ok_hull, ok_spaces = self.footprint(f)
            g["ignore"] = {"void"} | (set() if ok_spaces else {"boundary"}) | (set() if ok_hull else {"interior"})
            if ok_hull:
                M = np.hstack([A / hull[1], (A @ hull[2] + b)[:, None]])
                g["hull"] = cv2.warpAffine(hull[0], M, (self.size, self.size), flags=cv2.INTER_LINEAR) > 127
        return g, stairs, feats, areas

    def footprint(self, f, r=10.0, close=1.0, pad=2.0):
        """Building footprint for floors with incomplete geometry (IFC storeys): walls, openings, columns and indoor
        spaces of the whole floor rasterised at r px/m, gaps up to `close` m closed, holes filled. Returns
        ((mask * 255, r, origin), footprint plausible, spaces cover the rooms). Plausible means the footprint covers at
        least half the convex hull of the walls (not a skeleton of inner walls without facades). Cached per floor."""
        key = f.get("floor_id")
        if key in self._footprints:
            return self._footprints[key]
        x0, y0, x1, y1 = f["walls"].bounds
        o = np.array([x0 - pad, y0 - pad])
        W, H = int((x1 - x0 + 2 * pad) * r) + 1, int((y1 - y0 + 2 * pad) * r) + 1
        to = lambda g: shapely.transform(g, lambda xy: (xy - o) * r)
        walls = np.zeros((H, W), np.uint8)
        for k in ("walls", "doors", "windows", "columns"):
            fill(walls, to(f[k]), 1)
        sp = np.zeros((H, W), np.uint8)
        for k_, p in f["areas"]:
            if k_ not in OUTDOOR:
                fill(sp, to(p), 1)
        pts = cv2.findNonZero(walls)
        hull_area = cv2.contourArea(cv2.convexHull(pts)) if pts is not None and len(pts) >= 3 else 0.0
        for c in (close, 2.5 * close):                   # facades with wide gaps (curtain walls not cut): second try
            k = int(c * r) | 1
            m = cv2.morphologyEx(walls | sp, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            outside = np.zeros((H + 2, W + 2), np.uint8)
            cv2.floodFill(m.copy(), outside, (0, 0), 1, flags=4 | cv2.FLOODFILL_MASK_ONLY | (1 << 8))
            m = (outside[1:-1, 1:-1] == 0).astype(np.uint8)
            ok_hull = hull_area > 0 and m.sum() > 0.5 * hull_area
            if ok_hull:
                break
        inner = m.astype(bool) & ~walls.astype(bool)
        ok_spaces = ok_hull and (sp.astype(bool) & inner).sum() > 0.7 * inner.sum()
        out = ((m * 255).astype(np.uint8), r, o), bool(ok_hull), bool(ok_spaces)
        self._footprints[key] = out
        return out

    def _theta(self, rng):
        theta = float(rng.choice([0, 90, 180, 270]) + np.clip(rng.normal(0, 1.0), -3, 3))
        if rng.random() < 0.04:
            theta = float(rng.uniform(0, 360))
        return theta

    def _sample(self, rng, index=None, style=None):
        f, s, c = self._pick(rng, index)
        st = self.style(rng, **(style or {}))
        st["theta"] = self._theta(rng)
        g, stairs, feats, areas = self._project(f, s, c, st["theta"])
        img, lab, tg = self._draw(st, s, g, stairs, feats, areas, f.get("public", 0), rng)
        elements = set(self._elements)
        if st["multi"]:
            self._second_drawing(img, lab, tg, st, s, rng)
            elements |= self._elements | {"multi"}
        self._elements = elements
        img, lab = self.degrade(img, lab, st, rng, targets=tg)
        tags = {k: str(v) for k, v in st.items() if k in ("era", "walls", "doors", "windows")}
        tags.update(source="ifc" if "licence" in f else "sd", floor=str(f.get("floor_id", "")), scale=f"{s / PX_PER_M:.2f}",
                    elements=",".join(sorted(self._elements)))
        return img, lab, tags, tg

    def draw(self, st, s, g, stairs, feats, areas, public, rng, targets=None):
        """Draw one crop (pixel geometry) in style st: (img, lab). Pass a dict as `targets` to receive the extra
        target maps (common.HEADS) as well."""
        img, lab, tg = self._draw(st, s, g, stairs, feats, areas, public, rng)
        if targets is not None:
            targets.update(tg)
        return img, lab

    def _draw(self, st, s, g, stairs, feats, areas, public, rng):
        n = self.size
        self._elements = set()
        paper = st.get("paper") or ((255, 255, 255) if rng.random() < 0.7 else tuple(int(v) for v in rng.integers(225, 256, 3)))
        self._paper = paper
        img = np.full((n, n, 3), paper, np.uint8)
        self._text_mask = np.zeros((n, n), np.uint8)
        ink, pen, thin = st["ink"], st["pen"], st["thin_pen"]
        g = dict(g)
        indoor = [p for k, p, _ in areas if k not in OUTDOOR and k not in VOIDS]
        outdoor = [p for k, p, _ in areas if k in OUTDOOR]
        voids = [p for k, p, _ in areas if k in VOIDS]

        # targets that follow from the floor geometry
        interior = self.interior(g, indoor, voids, outdoor, s)
        boundary = np.zeros((n, n), np.uint8)
        for _, p, _ in areas:
            stroke(boundary, p, 1, BOUNDARY_PX)
        void = np.zeros((n, n), np.uint8)
        for p in voids:
            fill(void, p, 1)

        # geometry details: column grid, window reveals, niches, pilasters
        grid = bool(st.get("column_grid") or (public > 0.5 and rng.random() < 0.15))
        occ = wall_m = None
        if grid or st.get("niches") or st.get("pilasters"):
            occ = np.zeros((n, n), np.uint8)             # doors, windows, stairs: keep details away from them
            for p in [g["doors"], g["windows"], *stairs]:
                fill(occ, p, 1)
            wall_m = np.zeros((n, n), np.uint8)
            fill(wall_m, g["walls"], 1)
        grid_lines = None
        if grid:
            cols, grid_lines = self.column_grid(st, s, interior, occ, rng)
            if cols:
                g["columns"] = shapely.union_all([g["columns"], *cols])
                self._elements.add("column_grid")
        carve, frames, inner = [], {}, {}
        if st.get("reveals") or str(st["windows"]).startswith("casement"):
            for i, p in enumerate(polys(g["windows"])):
                c, u, L, T = rect_axes(p)
                inner[i] = self.inner_side(c, np.array([-u[1], u[0]]), T, s, interior)
        if st.get("reveals"):
            carve += self.reveals(g, s, inner, frames, rng)
            if frames:
                self._elements.add("reveals")
        if st.get("niches"):
            niches = self.niches(indoor, wall_m, occ, s, rng)
            carve += niches
            if niches:
                self._elements.add("niches")
        added, plinths = ([], [])
        if st.get("pilasters"):
            added, plinths = self.pilasters(g, indoor, interior, wall_m, occ, s, rng)
        walls_lab = g["walls"]
        if carve:
            walls_lab = shapely.difference(walls_lab, shapely.union_all(carve), grid_size=0.01)
        if added:
            walls_lab = shapely.union_all([walls_lab, *added], grid_size=0.01)
            for p in added:
                fill(interior, p, 1)
            self._elements.add("pilasters")

        # labels (later classes paint over earlier ones)
        lab = np.zeros((n, n), np.uint8)
        for p in stairs:
            fill(lab, p, STAIRS)
        fill(lab, walls_lab, WALL)
        fill(lab, g["columns"], COLUMN)
        fill(lab, g["windows"], WINDOW)
        fill(lab, g["doors"], DOOR)

        openings = shapely.union_all([g["doors"], g["windows"]])
        walls = shapely.difference(walls_lab, openings, grid_size=0.01) if not openings.is_empty else walls_lab

        # outside the building: roofs, terrain, neighbours, paving
        if st.get("context"):
            self.context(img, st, s, interior, rng)
        # room fills, floor patterns and finishes
        for k, p, _ in areas:
            if st["room_fill"]:
                fill(img, p, PASTELS[stable_index(k, len(PASTELS))] if rng.random() < 0.7 else PASTELS[rng.integers(len(PASTELS))])
            if st["patterns"] and (k in ("BATHROOM", "KITCHEN") or rng.random() < 0.08):
                self.pattern(img, p, rng, s)
            if k in OUTDOOR and st.get("outdoor_hatch") and rng.random() < 0.8:
                self.region_pattern(img, p, st, s, rng, outdoor=True)
                self._elements.add("outdoor_hatch")
            elif k not in OUTDOOR and k not in VOIDS and rng.random() < st.get("room_hatch", 0):
                self.region_pattern(img, p, st, s, rng, outdoor=False)
                self._elements.add("room_hatch")
        # ceiling ornament
        if st.get("ornament"):
            big = [p for p in indoor if p.area > 8 * s * s]
            for j in rng.permutation(len(big))[:4]:
                self.ornament(img, big[j], st, s, rng)
            if big:
                self._elements.add("ornament")
        # fixtures
        if st["features"]:
            for k, p in feats:
                self.fixture(img, k, p, ink, thin, rng)
        # stairs: outline, treads, walking line
        for p in stairs:
            self.stair(img, p, ink, thin, s, rng)
        stroke(img, g["railings"], ink, thin)
        # voids: diagonal cross
        if st.get("void_cross"):
            for p in voids:
                self.void_cross(img, p, ink, thin)
        # walls
        self.walls(img, walls, st, s, interior, rng)
        # columns
        cs = st.get("column_style", "solid")
        for p in polys(g["columns"]):
            if cs == "solid":
                fill(img, p, ink)
            elif cs == "grey":
                fill(img, p, (st["grey"],) * 3)
                stroke(img, p, ink, thin)
            else:
                stroke(img, p, ink, pen)
                if cs == "x":
                    q = np.asarray(p.minimum_rotated_rectangle.exterior.coords)[:4]
                    line(img, q[0], q[2], ink, thin)
                    line(img, q[1], q[3], ink, thin)
                elif cs == "hatch":
                    self.hatch(img, p, ink, rng, 4)
        # openings
        for p in polys(g["doors"]):
            self.door(img, p, st, rng)
        for i, p in enumerate(polys(g["windows"])):
            self.window(img, p, st, s, rng, frames.get(i), inner.get(i, 0))
        if st["windows"].startswith("casement"):
            self._elements.add("casement")
        for a, b in plinths:
            line(img, a, b, ink, 1)
        # text, dimensions, axes
        if st["text"]:
            for k, p, area in areas:
                if p.is_empty or p.area < (0.8 * s) ** 2:
                    continue
                self.stamp(img, k, p, area, st, s, public, rng)
        if st["dims"]:
            for _ in range(rng.integers(1, 4)):
                self.dimension(img, st, s, rng)
        if st["axes"]:
            self.axes(img, st, rng)
        if grid_lines is not None:
            self.grid_axes(img, st, s, grid_lines, rng)
        # overlays
        if st.get("zones") and indoor:
            self.zones(img, indoor, s, rng)
        if st.get("red_pen"):
            self.red_pen(img, st, s, rng)
        if st.get("annotations"):
            self.annotations(img, st, s, rng)
        if st.get("rubber_stamp"):
            self.rubber_stamp(img, st, s, rng)
        inside = interior.copy()
        tmap = {"interior": interior, "boundary": boundary, "void": void}
        for k in g.get("ignore", ()):                    # unknown for this floor: 255, ignored by the loss
            tmap[k] = np.full((n, n), IGNORE, np.uint8)
        interior, boundary, void = tmap["interior"], tmap["boundary"], tmap["void"]
        maps = [lab, interior, boundary, void]
        if st.get("title_block"):
            self.title_block(img, maps, st, s, rng)
        if st.get("legend"):
            self.legend(img, maps, st, s, rng)
        if st.get("keyplan"):
            self.keyplan(img, maps, st, s, inside, rng)
        tg = {"interior": interior, "boundary": boundary, "void": void, "text": self._text_mask}
        self._text_mask = None
        assert list(tg) == list(HEADS)
        return img, lab, tg

    def _second_drawing(self, img, lab, tg, st, s, rng):
        """Another floor drawn next to this one on the same sheet, in the same style and scale, separated by a gutter."""
        n = self.size
        f2, _, c2 = self._pick(rng, s=s)
        st2 = dict(st, multi=False, title_block=False, legend=False, keyplan=False, paper=self._paper)
        st2["theta"] = float(np.round(st["theta"] / 90) * 90 + np.clip(rng.normal(0, 1.0), -3, 3))
        elements = set(self._elements)
        g2, stairs2, feats2, areas2 = self._project(f2, s, c2, st2["theta"])
        img2, lab2, tg2 = self._draw(st2, s, g2, stairs2, feats2, areas2, f2.get("public", 0), rng)
        self._elements |= elements
        k = int(rng.uniform(0.3, 0.6) * n)
        gutter = int(rng.integers(8, 40))
        own = np.zeros((n, n), bool)                     # pixels taken from the second drawing
        own[:, :k] = True
        side = int(rng.integers(4))
        own = np.rot90(own, side)
        gut = np.zeros((n, n), bool)
        gut[:, max(k - gutter // 2, 0):k + gutter // 2] = True
        gut = np.rot90(gut, side)
        img[own] = img2[own]
        lab[own] = lab2[own]
        for key in tg:
            tg[key][own] = tg2[key][own]
        img[gut] = self._paper
        lab[gut] = 0
        for key in tg:
            tg[key][gut] = 0
        if rng.random() < 0.4:                           # frame line along the gutter
            ys, xs = np.nonzero(gut)
            q = np.array([xs.mean(), ys.mean()])
            d = np.array([0.0, 1.0]) if side % 2 == 0 else np.array([1.0, 0.0])
            seg = clip_line(q, d, n)
            if seg is not None:
                line(img, *seg, st["ink"], 1)

    # ----- targets -----
    def interior(self, g, indoor, voids, outdoor=(), s=PX_PER_M):
        """Inside the building outline: walls, openings, columns and indoor areas, small raster gaps closed, enclosed
        regions that are not outdoor areas (courtyards, light wells) filled. With g["hull"] (IFC storeys, see
        footprint()) the footprint replaces the room areas, which are incomplete there."""
        n = self.size
        m = np.zeros((n, n), np.uint8)
        hull = g.get("hull")
        for p in ([] if hull is not None else indoor) + list(voids):
            fill(m, p, 1)
        for k in ("walls", "doors", "windows", "columns"):
            fill(m, g[k], 1)
        if hull is not None:                             # eroded a little: the wall raster gives the exact outer face
            e = max(3, int(0.2 * s) | 1)
            m |= cv2.erode(hull.astype(np.uint8), np.ones((e, e), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))   # raster gaps between areas and walls
        k, cc, stats, _ = cv2.connectedComponentsWithStats((m == 0).astype(np.uint8), connectivity=4)
        x, y, w, h, area = stats.T
        hole = (x > 0) & (y > 0) & (x + w < n) & (y + h < n)
        hole[0] = False
        if hole.any() and outdoor:                       # keep courtyards and light wells (outdoor areas) outside
            o = np.zeros((n, n), np.uint8)
            for p in outdoor:
                fill(o, p, 1)
            share = np.bincount(cc.ravel(), weights=o.ravel(), minlength=k) / np.maximum(area, 1)
            hole &= share < 0.3
        m[hole[cc]] = 1
        return m

    # ----- geometry details -----
    def inner_side(self, c, v, T, s, interior):
        """+1 or -1: the side of an opening (along v) that is inside the building; 0 if both or neither side is."""
        a, b = at(interior, c + v * (T / 2 + 0.25 * s)), at(interior, c - v * (T / 2 + 0.25 * s))
        return 1 if a and not b else -1 if b and not a else 0

    def reveals(self, g, s, inner, frames, rng):
        """Splayed or straight window reveals cut out of the wall on the room side; the window moves to the outer part."""
        carve = []
        for i, p in enumerate(polys(g["windows"])):
            c, u, L, T = rect_axes(p)
            if T < 0.2 * s or L < 0.4 * s or not inner.get(i) or rng.random() < 0.2:
                continue
            vin = np.array([-u[1], u[0]]) * inner[i]
            d = T * rng.uniform(0.35, 0.65)
            e = s * rng.uniform(0.04, 0.18)
            face = c + vin * T / 2
            back = u * (L / 2 if rng.random() < 0.6 else L / 2 + e)          # splayed or straight
            quad = [face - u * (L / 2 + e) + vin, face + u * (L / 2 + e) + vin, face - vin * d + back, face - vin * d - back]
            carve.append(shapely.Polygon(quad))
            frames[i] = (vin, d)
        return carve

    def _room_edges(self, rooms, s, min_len, min_area):
        out = []
        for p in rooms:
            if p.area < min_area * s * s:
                continue
            for q in polys(p):
                pts = np.asarray(q.exterior.coords)
                for a, b in zip(pts[:-1], pts[1:]):
                    if np.hypot(*(b - a)) > min_len * s:
                        out.append((a, b))
        return out

    def niches(self, indoor, wall_m, occ, s, rng):
        """Recesses in thick walls on the room side (cupboards, radiator niches), labelled background."""
        edges = self._room_edges(indoor, s, 1.2, 4)
        carve = []
        for j in rng.permutation(len(edges))[:int(rng.integers(1, 6))]:
            a, b = edges[j]
            L = np.hypot(*(b - a))
            u = (b - a) / L
            nrm = np.array([-u[1], u[0]])
            m = (a + b) / 2
            side = 1 if at(wall_m, m + nrm * 0.1 * s) else -1 if at(wall_m, m - nrm * 0.1 * s) else 0
            if not side:
                continue
            d = nrm * side
            T = next((t for t in range(1, int(1.0 * s)) if not at(wall_m, m + d * t)), int(1.0 * s))
            if T < 0.25 * s:
                continue
            depth, wd = rng.uniform(0.3, 0.6) * T, min(rng.uniform(0.5, 1.4) * s, L - 0.3 * s)
            t0 = rng.uniform(0.15 * s, max(L - wd - 0.15 * s, 0.15 * s + 1))
            q0 = a + u * t0
            rect = [q0 - d, q0 + u * wd - d, q0 + u * wd + d * depth, q0 + d * depth]
            if any(at(occ, q) for q in rect + [np.mean(rect, axis=0)]):
                continue
            carve.append(shapely.Polygon(rect))
        return carve

    def pilasters(self, g, indoor, interior, wall_m, occ, s, rng):
        """Pilasters on long room walls and facades (labelled wall) and plinth lines along facades (background)."""
        width, depth, spacing = rng.uniform(0.25, 0.5) * s, rng.uniform(0.05, 0.15) * s, rng.uniform(1.8, 4.0) * s
        edges = [(a, b, "in") for a, b in self._room_edges(indoor, s, 2.5, 12)]
        for p in polys(g["walls"]):
            pts = np.asarray(p.exterior.coords)
            edges += [(a, b, "out") for a, b in zip(pts[:-1], pts[1:]) if np.hypot(*(b - a)) > 2.5 * s]
        added, plinths = [], []
        for j in rng.permutation(len(edges))[:6]:
            a, b, where = edges[j]
            L = np.hypot(*(b - a))
            u = (b - a) / L
            nrm = np.array([-u[1], u[0]])
            m = (a + b) / 2
            if where == "in":
                side = -1 if at(wall_m, m + nrm * 0.1 * s) else 1 if at(wall_m, m - nrm * 0.1 * s) else 0
            else:
                side = 1 if not at(interior, m + nrm * 0.2 * s) else -1 if not at(interior, m - nrm * 0.2 * s) else 0
            if not side:
                continue
            d = nrm * side                               # direction the pilaster stands out
            for t in np.arange(rng.uniform(0.3, 1.0) * spacing, L - width - 0.2 * s, spacing):
                q0 = a + u * t
                rect = [q0 - d, q0 + u * width - d, q0 + u * width + d * depth, q0 + d * depth]
                if not any(at(occ, q) for q in rect + [np.mean(rect, axis=0)]):
                    added.append(shapely.Polygon(rect))
            if where == "out" and rng.random() < 0.6:
                off = depth + rng.uniform(0.03, 0.1) * s
                plinths.append((a + d * off, b + d * off))
        return added, plinths

    def column_grid(self, st, s, interior, occ, rng):
        """A regular column grid inside the building (square, round or steel profiles), with its axis lines."""
        n = self.size
        t = np.radians(st.get("theta", 0.0))
        d1, d2 = np.array([np.cos(t), -np.sin(t)]), np.array([-np.sin(t), -np.cos(t)])
        sp1 = rng.uniform(4.0, 8.5) * s
        sp2 = sp1 if rng.random() < 0.5 else rng.uniform(4.0, 8.5) * s
        o = np.array([n / 2, n / 2]) + d1 * rng.uniform(0, sp1) + d2 * rng.uniform(0, sp2)
        a = rng.uniform(0.25, 0.55) * s
        shape = rng.choice(["square", "round", "steel"], p=[0.6, 0.3, 0.1])
        k1, k2 = int(n / sp1) + 2, int(n / sp2) + 2
        cols = []
        for i in range(-k1, k1 + 1):
            for j in range(-k2, k2 + 1):
                q = o + i * sp1 * d1 + j * sp2 * d2
                if not (-a < q[0] < n + a and -a < q[1] < n + a) or not at(interior, q) or rng.random() < 0.1:
                    continue
                x0, y0 = int(max(q[0] - a, 0)), int(max(q[1] - a, 0))
                if occ[y0:int(q[1] + a) + 1, x0:int(q[0] + a) + 1].any():
                    continue
                if shape == "round":
                    cols.append(shapely.Point(q).buffer(a / 2, quad_segs=6))
                    continue
                if shape == "square":
                    loc = [(-a / 2, -a / 2), (a / 2, -a / 2), (a / 2, a / 2), (-a / 2, a / 2)]
                else:                                    # HEA-like I profile
                    f_, w_ = 0.15 * a, 0.1 * a
                    loc = [(-a / 2, -a / 2), (a / 2, -a / 2), (a / 2, -a / 2 + f_), (w_ / 2, -a / 2 + f_), (w_ / 2, a / 2 - f_),
                           (a / 2, a / 2 - f_), (a / 2, a / 2), (-a / 2, a / 2), (-a / 2, a / 2 - f_), (-w_ / 2, a / 2 - f_),
                           (-w_ / 2, -a / 2 + f_), (-a / 2, -a / 2 + f_)]
                cols.append(shapely.Polygon([q + x * d1 + y * d2 for x, y in loc]))
        lines = ([(o + i * sp1 * d1, d2) for i in range(-k1, k1 + 1)], [(o + j * sp2 * d2, d1) for j in range(-k2, k2 + 1)])
        return cols, lines

    # ----- elements -----
    def walls(self, img, walls, st, s, interior, rng):
        kind = st["walls"]
        if kind == "split":
            thick, thin = [], []
            for p in polys(walls):
                (thick if 2 * p.area / max(p.length, 1e-6) > 0.2 * s else thin).append(p)
            self.wall_style(img, shapely.MultiPolygon(thick) if thick else None, st["thick"], st, s, interior, rng)
            self.wall_style(img, shapely.MultiPolygon(thin) if thin else None, st["thin"], st, s, interior, rng)
        else:
            self.wall_style(img, walls, kind, st, s, interior, rng)

    def wall_style(self, img, g, kind, st, s, interior, rng):
        if g is None or g.is_empty:
            return
        ink, pen = st["ink"], st["pen"]
        if kind == "solid":
            fill(img, g, ink)
        elif kind == "grey":
            fill(img, g, (st["grey"],) * 3)
            stroke(img, g, ink, pen)
        elif kind == "colour":
            fill(img, g, st["wall_colour"])
            if rng.random() < 0.7:
                stroke(img, g, ink, pen)
        elif kind == "hatch":
            self.hatch(img, g, ink, rng, int(rng.integers(3, 9)))
            stroke(img, g, ink, pen)
        elif kind == "material":
            self.wall_material(img, g, st, s, interior, rng)
            self._elements.add("material")
        else:
            stroke(img, g, ink, pen)

    def wall_material(self, img, g, st, s, interior, rng):
        """Swiss material conventions by local wall thickness: concrete fill, masonry or sand-lime hatch, stone, drywall,
        and an insulation band (zig-zag) along the outer face of thick exterior walls."""
        n, ink = self.size, st["ink"]
        W = np.zeros((n, n), np.uint8)
        fill(W, g, 1)
        if not W.any():
            return
        D = cv2.distanceTransform(W, cv2.DIST_L2, 3)
        k = max(3, int(0.5 * s) | 1)                     # local thickness: largest inscribed half-width nearby
        thick = 2 * cv2.dilate(D, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        Wb = W > 0
        classes = {"thin": Wb & (thick < 0.13 * s), "thick": Wb & (thick >= 0.27 * s)}
        classes["medium"] = Wb & ~classes["thin"] & ~classes["thick"]
        mats = st.get("materials", {"thin": "outline", "medium": "masonry", "thick": "concrete_grey"})
        for cls, m in classes.items():
            if m.any():
                self.material(img, m, mats[cls], st, rng)
        if st.get("insulation"):
            out = ~Wb & (interior == 0)
            if out.any() and classes["thick"].any():
                dO = cv2.distanceTransform((~out).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                band = classes["thick"] & (dO <= np.maximum(0.33 * thick, 3))
                if band.any():                           # zig-zag across the band, running along the facade
                    ys, xs = np.nonzero(band)
                    sx = cv2.Sobel(dO, cv2.CV_32F, 1, 0, ksize=3)[ys, xs]
                    sy = cv2.Sobel(dO, cv2.CV_32F, 0, 1, ksize=3)[ys, xs]
                    along = np.where(np.abs(sx) > np.abs(sy), ys, xs).astype(np.float32)   # normal horizontal: runs along y
                    bw = float(np.median(np.maximum(0.33 * thick[ys, xs], 3)))            # one band width per sample
                    t = along / np.float32(bw * rng.uniform(1.4, 2.2))
                    tri = 2 * np.abs(t - np.floor(t + 0.5))                                # triangle wave 0..1
                    on = np.abs(dO[ys, xs] - 1 - np.float32(bw - 2) * tri) < 0.7
                    img[band] = self._paper
                    img[ys[on], xs[on]] = ink
                    inner = band & ~cv2.erode(band.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (dO > 1.5)
                    img[inner] = ink
                    self._elements.add("insulation")
        stroke(img, g, ink, st["pen"])

    def material(self, img, m, mat, st, rng, box=None):
        """Fill mask m (within box, a pair of slices, or the whole image) with a material pattern."""
        n, ink = self.size, st["ink"]
        box = box or (slice(0, n), slice(0, n))
        sub = img[box]
        if mat == "concrete_grey":
            sub[m] = (st["grey"],) * 3
        elif mat == "concrete_black":
            sub[m] = ink
        elif mat == "light_grey":
            sub[m] = (int(rng.integers(200, 235)),) * 3
        elif mat == "concrete_dots":
            sub[m] = (int(rng.integers(190, 225)),) * 3
            sub[m & (rng.random(m.shape, dtype=np.float32) < 0.08)] = ink
        elif mat == "masonry":
            sub[m & lines_tex(n, 45 + 90 * int(rng.integers(2)), rng.integers(3, 6))[box]] = ink
        elif mat == "sandlime":
            sp = int(rng.integers(6, 10))
            sub[m & (lines_tex(n, 45, sp) | lines_tex(n, 45, sp, phase=2))[box]] = ink
        elif mat == "drywall_cross":
            sp = int(rng.integers(3, 5))
            sub[m & (lines_tex(n, 45, sp) | lines_tex(n, 135, sp))[box]] = ink
        elif mat == "stone":
            tex = np.roll(_stone_tex(n, int(rng.integers(4))), int(rng.integers(n)), axis=int(rng.integers(2)))
            sub[m & (tex[box] > 0)] = ink

    def hatch(self, img, g, ink, rng, spacing):
        lm = local_mask(g, self.size)
        if lm is not None:
            m, box = lm
            img[box][m & lines_tex(self.size, rng.choice([45, -45, 30, 60]), spacing)[box]] = ink

    def pattern(self, img, p, rng, s):
        lm = local_mask(p, self.size)
        step = max(4, int(s * rng.uniform(0.15, 0.35)))
        both = rng.random() < 0.7
        grey = int(rng.integers(120, 200))
        if lm is None:
            return
        m, (ys, xs) = lm
        grid = np.zeros(m.shape, bool)
        grid[(-ys.start) % step::step, :] = True
        if both:
            grid[:, (-xs.start) % step::step] = True
        img[ys, xs][m & grid] = grey

    def region_pattern(self, img, g, st, s, rng, outdoor, mask=None):
        """Hatch, tiles, decking, stipple or parquet in a region (terrace, room finish, outside context)."""
        n = self.size
        if mask is None:
            lm = local_mask(g, n)
            if lm is None:
                return
            mask, box = lm
        else:
            box = (slice(0, n), slice(0, n))
        if not mask.any():
            return
        th = st.get("theta", 0.0)
        colour = st["ink"] if rng.random() < (0.6 if outdoor else 0.4) else (int(rng.integers(90, 190)),) * 3
        kinds = ["hatch", "hatch", "cross", "tiles", "deck", "dots"] if outdoor else ["hatch", "cross", "tiles", "parquet", "dots"]
        kind = rng.choice(kinds)
        lo = 3 if outdoor else 4                         # terraces as dense as wall hatches; floor finishes a little finer
        if kind == "hatch":
            tex = lines_tex(n, rng.choice([45, 135, 30, 60, -th, 90 - th]), rng.integers(lo, 10))[box]
        elif kind == "cross":
            sp = rng.integers(lo + 1, 11)
            tex = (lines_tex(n, 45, sp) | lines_tex(n, 135, sp))[box]
        elif kind == "tiles":
            sp = max(4, s * rng.uniform(0.2, 0.6))
            tex = (lines_tex(n, -th, sp) | lines_tex(n, 90 - th, sp))[box]
        elif kind == "deck":
            tex = lines_tex(n, -th + 90 * int(rng.integers(2)), max(3, s * rng.uniform(0.1, 0.2)))[box]
        elif kind == "parquet":
            sp = max(3, s * rng.uniform(0.07, 0.15))
            ang = -th + 90 * int(rng.integers(2))
            tex = (lines_tex(n, ang, sp) | (lines_tex(n, ang + 90, sp * rng.integers(4, 8)) & lines_tex(n, ang, sp * 2, width=2)))[box]
        else:
            tex = rng.random(mask.shape, dtype=np.float32) < rng.uniform(0.02, 0.1)
        img[box][mask & tex] = colour

    def context(self, img, st, s, interior, rng):
        """Hatched and stippled regions outside the building: roof slopes, terrain, neighbours, paving, section snippets."""
        n, ink = self.size, st["ink"]
        ext = interior == 0
        if ext.mean() < 0.05:
            return
        t = np.radians(st.get("theta", 0.0))
        d1, d2 = np.array([np.cos(t), -np.sin(t)]), np.array([-np.sin(t), -np.cos(t)])
        ys, xs = np.nonzero(ext)
        clear = cv2.erode(ext.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0       # a little away from the facade
        for _ in range(int(rng.integers(1, 4))):
            kind = rng.choice(["roof", "ground", "neighbour", "paving", "section"], p=[0.3, 0.25, 0.2, 0.15, 0.1])
            j = int(rng.integers(len(xs)))
            q = np.array([xs[j], ys[j]], float)
            a, b = rng.uniform(1.5, 7) * s, rng.uniform(1.5, 7) * s
            quad = [q - d1 * a / 2 - d2 * b / 2, q + d1 * a / 2 - d2 * b / 2, q + d1 * a / 2 + d2 * b / 2, q - d1 * a / 2 + d2 * b / 2]
            m = np.zeros((n, n), np.uint8)
            cv2.fillPoly(m, [fx(quad)], 1, cv2.LINE_8, shift=4)
            if kind == "roof":                             # roof slope: covering lines, eaves, hips and ridge
                m = (m > 0) & ext
                ax = d1 if rng.random() < 0.5 else d2
                other = d2 if ax is d1 else d1
                ang = np.degrees(np.arctan2(ax[1], ax[0]))
                layer = np.zeros((n, n), np.uint8)
                cv2.polylines(layer, [fx(quad)], True, 1, 1, cv2.LINE_8, shift=4)
                c0 = np.mean(quad, axis=0)
                ridge = [c0 - other * min(a, b) * 0.25, c0 + other * min(a, b) * 0.25]
                for corner in quad:
                    end = min(ridge, key=lambda r_: np.hypot(*(r_ - corner)))
                    cv2.line(layer, tuple(fx(corner)), tuple(fx(end)), 1, 1, cv2.LINE_8, shift=4)
                cv2.line(layer, tuple(fx(ridge[0])), tuple(fx(ridge[1])), 1, 1, cv2.LINE_8, shift=4)
                img[(m & lines_tex(n, ang, rng.integers(3, 9))) | ((layer > 0) & ext)] = ink
            elif kind == "ground":
                if rng.random() < 0.5:                     # band around the building
                    near = cv2.distanceTransform(ext.astype(np.uint8), cv2.DIST_L2, 3) < rng.uniform(0.5, 2.5) * s
                    m = near & clear
                else:
                    m = (m > 0) & clear
                if rng.random() < 0.5:
                    img[m & (rng.random((n, n), dtype=np.float32) < rng.uniform(0.03, 0.12))] = ink
                else:
                    img[m & lines_tex(n, 45, rng.integers(5, 10)) & (rng.random((n, n), dtype=np.float32) < 0.85)] = ink
            elif kind == "neighbour":
                m = (m > 0) & clear
                r_ = rng.random()
                if r_ < 0.4:
                    img[m & lines_tex(n, rng.choice([45, 135]), rng.integers(3, 7))] = ink
                elif r_ < 0.75:
                    img[m] = (int(rng.integers(150, 225)),) * 3
                else:
                    img[m] = ink
                cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnt, -1, ink, 1, cv2.LINE_AA)
            elif kind == "paving":
                m = (m > 0) & clear
                self.region_pattern(img, None, st, s, rng, outdoor=True, mask=m)
            else:                                          # section snippet: slab band over hatched ground
                h = rng.uniform(0.2, 0.35) * s
                band = [q - d1 * a / 2, q + d1 * a / 2, q + d1 * a / 2 + d2 * h, q - d1 * a / 2 + d2 * h]
                bm = np.zeros((n, n), np.uint8)
                cv2.fillPoly(bm, [fx(band)], 1, cv2.LINE_8, shift=4)
                bm = (bm > 0) & clear
                img[bm] = ink if rng.random() < 0.5 else (st["grey"],) * 3
                gm = np.zeros((n, n), np.uint8)
                ground = [band[3], band[2], band[2] + d2 * b * 0.5, band[3] + d2 * b * 0.5]
                cv2.fillPoly(gm, [fx(ground)], 1, cv2.LINE_8, shift=4)
                gm = (gm > 0) & clear & ~bm
                img[gm & lines_tex(n, 45, rng.integers(5, 9)) & (rng.random((n, n), dtype=np.float32) < 0.8)] = ink
            self._elements.add("context")

    def ornament(self, img, p, st, s, rng):
        """Ceiling ornament drawn inside a room: cornice lines, coffers, corner motifs, rosette, garland."""
        ink = st["ink"] if rng.random() < 0.7 else (int(rng.integers(80, 160)),) * 3
        w = 1
        pattern = (6, 4) if rng.random() < 0.4 else None
        draw = (lambda pts: dashed(img, pts, ink, w, pattern, closed=True)) if pattern else \
            (lambda pts: polyline(img, pts, ink, w, closed=True))
        insets = sorted(rng.uniform(0.06, 0.5, int(rng.integers(1, 4))) * s)
        ring = None
        for d in insets:
            q = p.buffer(-d, join_style="mitre", mitre_limit=3)
            for part in polys(q):
                draw(np.asarray(part.exterior.coords))
                ring = part
        if ring is None:
            return
        c, u, L, T = rect_axes(ring)
        v = np.array([-u[1], u[0]])
        r = rng.random()
        if r < 0.3:                                        # coffers: grid of inset rectangles
            k1, k2 = int(rng.integers(2, 5)), int(rng.integers(2, 4))
            for i in range(k1):
                for j in range(k2):
                    cc = c + u * (i - (k1 - 1) / 2) * L / k1 + v * (j - (k2 - 1) / 2) * T / k2
                    hu, hv = u * L / k1 * 0.38, v * T / k2 * 0.38
                    polyline(img, [cc - hu - hv, cc + hu - hv, cc + hu + hv, cc - hu + hv], ink, w, closed=True)
        elif r < 0.75:                                     # rosette / medallion
            R = min(L, T) * rng.uniform(0.12, 0.3)
            ang = np.degrees(np.arctan2(u[1], u[0]))
            ecc = rng.uniform(1.0, 1.8)
            for f_ in sorted(rng.uniform(0.3, 1.0, int(rng.integers(1, 4)))):
                cv2.ellipse(img, tuple(fx(c)), (int(16 * R * f_ * ecc), int(16 * R * f_)), ang, 0, 360, ink, w, cv2.LINE_AA, shift=4)
            for a in np.linspace(0, 2 * np.pi, int(rng.integers(8, 17)), endpoint=False):
                e = u * np.cos(a) * ecc + v * np.sin(a)
                line(img, c + e * R * 0.35, c + e * R, ink, w)
        else:                                              # garland: wave along the inner ring
            pts = np.asarray(ring.exterior.coords)
            seg = np.diff(pts, axis=0)
            Ls = np.hypot(*seg.T)
            out = []
            for a, sg, l_ in zip(pts[:-1], seg, Ls):
                if l_ < 4:
                    continue
                t = np.linspace(0, 1, max(int(l_ / 2), 2))
                nrm = np.array([-sg[1], sg[0]]) / l_
                amp, per = rng.uniform(2, 5), rng.uniform(8, 20)
                out.append(a + np.outer(t, sg) + np.outer(amp * np.sin(t * l_ / per * 2 * np.pi) - amp - 2, nrm))
            for o in out:
                polyline(img, o, ink, w)
        if rng.random() < 0.5:                             # corner motifs: quarter circles
            R = min(L, T) * rng.uniform(0.06, 0.15)
            for corner in np.asarray(ring.minimum_rotated_rectangle.exterior.coords)[:4]:
                cv2.circle(img, tuple(fx(corner)), int(16 * R), ink, w, cv2.LINE_AA, shift=4)

    def void_cross(self, img, p, ink, w):
        for q in polys(p):
            a = np.asarray(q.minimum_rotated_rectangle.exterior.coords)[:4]
            for x0, x1 in ((a[0], a[2]), (a[1], a[3])):
                seg = shapely.LineString([x0, x1]).intersection(q)
                for part in getattr(seg, "geoms", [seg]):
                    if part.geom_type == "LineString" and not part.is_empty:
                        line(img, *np.asarray(part.coords)[[0, -1]], ink, w)

    def fixture(self, img, kind, p, ink, w, rng):
        for q in polys(p):
            stroke(img, q, ink, w)
            c, u, L, T = rect_axes(q)
            if kind in ("TOILET", "SINK", "BATHTUB", "SHOWER") and L > 3:
                ax = (int(L * 0.35), int(T * 0.35)) if kind != "BATHTUB" else (int(L * 0.42), int(T * 0.38))
                ang = np.degrees(np.arctan2(u[1], u[0]))
                cv2.ellipse(img, tuple(int(x) for x in c), (max(ax[0], 1), max(ax[1], 1)), ang, 0, 360, ink, 1, cv2.LINE_AA)
            elif kind == "ELEVATOR" or (kind == "SHAFT"):
                a = np.asarray(q.minimum_rotated_rectangle.exterior.coords)[:4]
                line(img, a[0], a[2], ink, w)
                line(img, a[1], a[3], ink, w)
            elif kind == "KITCHEN" and L > 10:
                for k in range(int(rng.integers(0, 3))):
                    cv2.circle(img, tuple(int(x) for x in c + u * (k - 0.5) * T * 0.6), max(1, int(T * 0.2)), ink, 1, cv2.LINE_AA)

    def stair(self, img, p, ink, w, s, rng):
        for q in polys(p):
            stroke(img, q, ink, w)
            c, u, L, T = rect_axes(q)
            v = np.array([-u[1], u[0]])
            step = s * rng.uniform(0.25, 0.32)
            if L < step * 2:
                continue
            for k in np.arange(-L / 2 + step, L / 2, step):          # treads across the run direction
                a, b = c + u * k - v * T, c + u * k + v * T
                seg = shapely.LineString([a, b]).intersection(q)
                for part in getattr(seg, "geoms", [seg]):
                    if part.geom_type == "LineString" and not part.is_empty:
                        line(img, *np.asarray(part.coords)[[0, -1]], ink, w)
            if rng.random() < 0.5:                                    # walking line with arrow
                a, b = c - u * L * 0.45, c + u * L * 0.45
                line(img, a, b, ink, w)
                for sgn in (1, -1):
                    line(img, b, b - u * T * 0.15 + sgn * v * T * 0.1, ink, w)

    def door(self, img, p, st, rng):
        ink, w = st["ink"], st["thin_pen"]
        kind = st["doors"]
        c, u, L, T = rect_axes(p)
        if kind == "gap" or L < 4:
            return
        v = np.array([-u[1], u[0]]) * (1 if rng.random() < 0.5 else -1)
        face = c + v * T / 2
        if kind == "sliding":
            line(img, c - u * L / 2 + v * T * 0.15, c + u * L * 0.1 + v * T * 0.15, ink, w)
            line(img, c - u * L * 0.1 - v * T * 0.15, c + u * L / 2 - v * T * 0.15, ink, w)
            return
        leaves = [(face - u * L / 2, u, L)] if L < 1.5 * PX_PER_M or rng.random() < 0.4 else \
            [(face - u * L / 2, u, L / 2), (face + u * L / 2, -u, L / 2)]
        if rng.random() < 0.5:
            leaves = [(face + u * L / 2, -u, L)] if len(leaves) == 1 else leaves
        for hinge, along, r in leaves:
            tip = hinge + v * r
            line(img, hinge, tip, ink, w + (1 if rng.random() < 0.3 else 0))
            if kind in ("arc", "arc_dashed"):
                ang = np.linspace(0, np.pi / 2, 24)
                pts = hinge + np.outer(np.cos(ang), v * r) + np.outer(np.sin(ang), along * r)
                if kind == "arc":
                    polyline(img, pts, ink, 1)
                else:
                    for i in range(0, len(pts) - 1, 2):
                        line(img, pts[i], pts[i + 1], ink, 1)

    def window(self, img, p, st, s, rng, frame=None, inner=0):
        ink, w = st["ink"], st["thin_pen"]
        kind = st["windows"]
        c, u, L, T = rect_axes(p)
        v = np.array([-u[1], u[0]])
        if frame is not None:                              # reveal: the window sits in the outer part of the wall
            vin, d = frame
            c, T = c - vin * d / 2, max(T - d, 2.0)
        a, b = c - u * L / 2, c + u * L / 2
        if kind == "gap":
            return
        if kind.startswith("casement"):
            for o in (-0.2, 0.2):                          # frame
                line(img, a + v * T * o, b + v * T * o, ink, w)
            line(img, a - v * T * 0.2, a + v * T * 0.2, ink, w)
            line(img, b - v * T * 0.2, b + v * T * 0.2, ink, w)
            vin = v * (inner or (1 if rng.random() < 0.5 else -1))
            face = c + vin * T / 2
            sashes = [(face - u * L / 2, u, L)] if L < 0.8 * s else [(face - u * L / 2, u, L / 2), (face + u * L / 2, -u, L / 2)]
            for hinge, along, r in sashes:
                phi = np.radians(rng.choice([90, 90, rng.uniform(55, 90)]))
                t = np.linspace(0, phi, 16)
                pts = hinge + np.outer(np.cos(t), along * r) + np.outer(np.sin(t), vin * r)
                line(img, hinge, pts[-1], ink, 1)
                if kind == "casement":
                    polyline(img, pts, ink, 1)
                else:
                    dashed(img, pts, ink, 1, (3, 3))
            return
        offs = {"three": [-0.5, 0, 0.5], "two": [-0.5, 0.5], "glass": [-0.1, 0.1], "rect": [-0.5, 0, 0.5]}[kind]
        for o in offs:
            line(img, a + v * T * o, b + v * T * o, ink, w)
        if kind in ("rect", "glass"):
            line(img, a - v * T / 2, a + v * T / 2, ink, w)
            line(img, b - v * T / 2, b + v * T / 2, ink, w)

    def text(self, img, xy, s_txt, height, fnt, ink, angle=0, align="centre", alpha=1.0):
        """Draw a string centred (or left-aligned) at xy; its box goes into the text target."""
        angle = int(round(angle))
        size = max(6, int(height))
        m = _text_tile(str(s_txt), fnt, size if size < 12 else size & ~1, angle)   # even sizes: more cache hits
        if m is None:
            return
        h, w = m.shape
        x = int(xy[0] - w / 2) if align == "centre" else int(xy[0])
        y = int(xy[1] - h / 2)
        X0, Y0, X1, Y1 = max(x, 0), max(y, 0), min(x + w, self.size), min(y + h, self.size)
        if X1 <= X0 or Y1 <= Y0:
            return
        mm = m[Y0 - y:Y1 - y, X0 - x:X1 - x, None] * alpha
        reg = img[Y0:Y1, X0:X1].astype(np.float32)
        img[Y0:Y1, X0:X1] = (reg * (1 - mm) + np.array(ink, np.float32) * mm).astype(np.uint8)
        if self._text_mask is not None:
            ys, xs = np.nonzero(m > 0.25)
            if len(xs) < 2:
                return
            if angle % 90 == 0:
                self._text_mask[max(y + ys.min() - 1, 0):max(y + ys.max() + 2, 0), max(x + xs.min() - 1, 0):max(x + xs.max() + 2, 0)] = 1
            else:
                box = cv2.boxPoints(cv2.minAreaRect(np.c_[xs + x, ys + y].astype(np.float32)))
                cv2.fillPoly(self._text_mask, [np.round(box).astype(np.int32)], 1)

    def stamp(self, img, kind, p, area, st, s, public, rng):
        grp = TYPE_GROUP.get(kind, "office" if (public > 0.5 or rng.random() < 0.4) and kind == "ROOM" else "ROOM")
        name = rng.choice(NAMES.get(grp, NAMES["ROOM"]))
        if kind.startswith("NAME:") and rng.random() < 0.8:      # real room names from IFC models
            name = kind[5:][:40]
        lines = [name]
        if rng.random() < 0.6:
            a = area * rng.uniform(0.97, 1.03)
            lines.append(rng.choice([f"{a:.2f} m²", f"{a:.1f} m2", f"{a:.2f}".replace(".", ","), f"F = {a:.1f} m²"]))
        if rng.random() < 0.4:
            lines.insert(0, rng.choice([f"{rng.integers(1, 9)}.{rng.integers(1, 40):02d}", f"{rng.integers(100, 999)}",
                                        f"{rng.integers(1000, 9999)}.{rng.choice(['AA', 'AB', 'BA'])}.{rng.integers(0, 6):02d}.{rng.integers(1, 120):03d}"]))
        h = s * rng.uniform(0.15, 0.45)
        pt = np.asarray(p.representative_point().coords[0]) + rng.normal(0, s * 0.3, 2)
        angle = 90 if rng.random() < 0.1 else 0
        for i, txt in enumerate(lines):
            off = (i - (len(lines) - 1) / 2) * h * 1.3
            xy = pt + (np.array([0, off]) if not angle else np.array([off, 0]))
            self.text(img, xy, txt, h if i == 0 else h * 0.8, st["font"], st["ink"], angle)

    def dimension(self, img, st, s, rng):
        n, ink = self.size, st["ink"]
        horiz = rng.random() < 0.5
        k = rng.uniform(0.05, 0.95) * n
        a, b = sorted(rng.uniform(0, n, 2))
        if b - a < 0.15 * n:
            return
        p0, p1 = (np.array([a, k]), np.array([b, k])) if horiz else (np.array([k, a]), np.array([k, b]))
        line(img, p0, p1, ink, 1)
        ticks = np.sort(np.concatenate([[0, 1], rng.uniform(0, 1, rng.integers(0, 4))]))
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        nrm = np.array([-d[1], d[0]])
        tick = rng.choice(["slash", "dot", "arrow"])
        for t_ in ticks:
            q = p0 + (p1 - p0) * t_
            if tick == "slash":
                line(img, q - (d + nrm) * 4, q + (d + nrm) * 4, ink, 1)
            elif tick == "dot":
                cv2.circle(img, tuple(int(x) for x in q), 2, ink, -1, cv2.LINE_AA)
            line(img, q - nrm * 6, q + nrm * 6, ink, 1)
        for t0, t1 in zip(ticks[:-1], ticks[1:]):
            mid = p0 + (p1 - p0) * (t0 + t1) / 2 - nrm * s * 0.2
            length = np.linalg.norm(p1 - p0) * (t1 - t0) / s
            if length > 0.4:
                self.text(img, mid, f"{length:.2f}", s * 0.18, st["font"], ink, 0 if horiz else 90)

    def axes(self, img, st, rng):
        n, ink = self.size, st["ink"]
        for _ in range(rng.integers(1, 3)):
            horiz, k = rng.random() < 0.5, rng.uniform(0, n)
            pts = np.arange(0, n, 18)
            for i, q in enumerate(pts):
                a, b = (np.array([q, k]), np.array([q + (11 if i % 2 == 0 else 2), k])) if horiz else \
                    (np.array([k, q]), np.array([k, q + (11 if i % 2 == 0 else 2)]))
                line(img, a, b, ink, 1)

    def grid_axes(self, img, st, s, grid_lines, rng):
        """Dash-dot axis lines of the column grid with labelled bubbles (letters one way, numbers the other)."""
        n = self.size
        colour = st["ink"] if rng.random() < 0.7 else (int(rng.integers(90, 160)),) * 3 if rng.random() < 0.5 else (190, 40, 40)
        bubbles = rng.random() < 0.85
        r = rng.uniform(8, 15)
        fnt = st["font"] if rng.random() < 0.5 else self.print[int(rng.integers(len(self.print)))]
        start = int(rng.integers(0, 6))
        for which, group in enumerate(grid_lines):
            for i, (q, d) in enumerate(group):
                seg = clip_line(q, d, n)
                if seg is None:
                    continue
                a, b = seg
                dashed(img, [a, b], colour, 1, (14, 3, 2, 3))
                if not bubbles:
                    continue
                label = chr(ord("A") + (start + i) % 26) if which == 0 else str(start + i + 1)
                for end, other in ((a, b), (b, a)) if rng.random() < 0.3 else ((a, b),):
                    dd = (other - end) / max(np.linalg.norm(other - end), 1e-6)
                    cc = end + dd * (r + rng.uniform(4, 30))
                    cv2.circle(img, tuple(fx(cc)), int(16 * r), self._paper, -1, cv2.LINE_AA, shift=4)
                    cv2.circle(img, tuple(fx(cc)), int(16 * r), colour, 1, cv2.LINE_AA, shift=4)
                    self.text(img, cc, label, r * 1.1, fnt, colour)

    # ----- overlays -----
    def zones(self, img, indoor, s, rng):
        """Colour zones (fire compartments): translucent fill or colour hatch, thick boundary line."""
        n = self.size
        order = rng.permutation(len(indoor))
        groups = np.array_split(order, int(rng.integers(1, 4)))
        for grp in groups:
            if not len(grp):
                continue
            m = np.zeros((n, n), np.uint8)
            for j in grp[:int(rng.integers(1, len(grp) + 1))]:
                fill(m, indoor[j], 1)
            m = cv2.dilate(m, np.ones((5, 5), np.uint8))
            if not m.any():
                continue
            col = np.array(ZONE_COLOURS[int(rng.integers(len(ZONE_COLOURS)))], np.float32)
            mb = m > 0
            if rng.random() < 0.7:
                alpha = rng.uniform(0.15, 0.4)
                img[mb] = (img[mb] * (1 - alpha) + col * alpha).astype(np.uint8)
            else:
                img[mb & lines_tex(n, rng.choice([45, 135]), rng.integers(5, 12))] = col.astype(np.uint8)
            cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnt, -1, tuple(int(v) for v in col), int(rng.integers(2, 7)), cv2.LINE_AA)
        self._elements.add("zones")

    def wobbly(self, img, pts, colour, w, rng, amp=1.0):
        """Hand-drawn polyline: resampled with smooth jitter."""
        pts = np.asarray(pts, float)
        cum = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
        if cum[-1] < 2:
            return
        t = np.linspace(0, cum[-1], max(int(cum[-1] / 3), 3))
        q = np.c_[np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])]
        jit = np.cumsum(rng.normal(0, 0.3 * amp, q.shape), axis=0)
        jit -= np.linspace(0, 1, len(q))[:, None] * jit[-1]
        polyline(img, q + jit, colour, w)

    def red_pen(self, img, st, s, rng):
        """Revision clouds, delta markers, cross-outs and correction lines in red."""
        n = self.size
        red = tuple(int(v) for v in rng.choice([(210, 30, 30), (230, 60, 40), (190, 20, 60)]))
        for _ in range(int(rng.integers(1, 3))):
            c = rng.uniform(0.1, 0.9, 2) * n
            a, b = rng.uniform(0.8, 3.5, 2) * s
            box = np.array([c + [-a, -b], c + [a, -b], c + [a, b], c + [-a, b], c + [-a, -b]]) / 1
            r = rng.uniform(5, 12)
            for p0, p1 in zip(box[:-1], box[1:]):          # scallops along each side, bulging outwards
                L = np.linalg.norm(p1 - p0)
                k = max(int(L / (2 * r)), 1)
                d = (p1 - p0) / L
                ang = np.degrees(np.arctan2(d[1], d[0]))
                for i in range(k):
                    m = p0 + d * (i + 0.5) * L / k
                    cv2.ellipse(img, tuple(fx(m)), (int(16 * L / k / 2), int(16 * r * 0.8)), ang, 180, 360, red, 2, cv2.LINE_AA, shift=4)
            if rng.random() < 0.6:                         # revision delta with index
                q = c + np.array([a, -b]) + rng.uniform(5, 20, 2)
                tri = np.array([q + [0, -12], q + [11, 8], q + [-11, 8]])
                polyline(img, tri, red, 2, closed=True)
                self.text(img, q + [0, 2], str(rng.choice(list("123ABC"))), 11, self.print[0], red)
        for _ in range(int(rng.integers(0, 3))):           # cross-outs and correction lines
            c = rng.uniform(0.1, 0.9, 2) * n
            r = rng.uniform(0.3, 1.2) * s
            if rng.random() < 0.5:
                self.wobbly(img, [c - r, c + r], red, 2, rng)
                self.wobbly(img, [c + [-r, r], c + [r, -r]], red, 2, rng)
            else:
                d = np.array([1.0, 0.0]) if rng.random() < 0.5 else np.array([0.0, 1.0])
                self.wobbly(img, [c - d * r * 2, c + d * r * 2], red, int(rng.integers(2, 4)), rng)
        self._elements.add("red_pen")

    def annotations(self, img, st, s, rng):
        """Hand-written notes with leader arrows or circles, in pen or pencil."""
        n = self.size
        colour = tuple(int(v) for v in PEN_COLOURS[int(rng.integers(len(PEN_COLOURS)))])
        fnt = self.hand[int(rng.integers(len(self.hand)))] if self.hand else st["font"]
        for _ in range(int(rng.integers(1, 5))):
            q = rng.uniform(0.05, 0.95, 2) * n
            txt = str(rng.choice(ANNOTATIONS))
            self.text(img, q, txt, s * rng.uniform(0.2, 0.45), fnt, colour, angle=float(rng.uniform(-15, 15)), alpha=rng.uniform(0.85, 1))
            r = rng.random()
            if r < 0.4:                                    # leader with arrow head
                tip = q + rng.normal(0, 1.2 * s, 2)
                self.wobbly(img, [q + (tip - q) * 0.25, tip], colour, 1, rng)
                d = (tip - q) / max(np.linalg.norm(tip - q), 1e-6)
                nrm = np.array([-d[1], d[0]])
                line(img, tip, tip - d * 8 + nrm * 4, colour, 1)
                line(img, tip, tip - d * 8 - nrm * 4, colour, 1)
            elif r < 0.6:                                  # circled spot
                c = q + rng.normal(0, s, 2)
                t = np.linspace(0, 2.2 * np.pi, 40)
                R = rng.uniform(0.3, 1.0) * s
                self.wobbly(img, c + np.c_[np.cos(t) * R * 1.3, np.sin(t) * R], colour, 1, rng)
        self._elements.add("annotations")

    def rubber_stamp(self, img, st, s, rng):
        """An inked rubber stamp (rectangle or circle with text), rotated, with patchy ink."""
        n = self.size
        colour = np.array(rng.choice([(200, 30, 30), (40, 60, 170), (120, 40, 150), (40, 40, 40)]), np.float32)
        words = [w.format(n=f"{rng.integers(100, 999)}-{rng.integers(10, 99)}", d=f"{rng.integers(1, 29)}.{rng.integers(1, 13)}.{rng.integers(1950, 2025)}")
                 for w in STAMP_TEXT[int(rng.integers(len(STAMP_TEXT)))]]
        fnt = self.print[int(rng.integers(len(self.print)))]
        h = rng.uniform(14, 26)
        tw = int(max(font(fnt, int(h)).getlength(w) for w in words)) + 20
        th = int(len(words) * h * 1.3) + 16
        circle = rng.random() < 0.35
        S = max(tw, th) + 8 if circle else None
        tile = np.zeros((S or th, S or tw), np.uint8)
        tbox = np.zeros_like(tile)
        Ht, Wt = tile.shape
        if circle:
            cv2.circle(tile, (Wt // 2, Ht // 2), Wt // 2 - 2, 255, 2, cv2.LINE_AA)
            cv2.circle(tile, (Wt // 2, Ht // 2), Wt // 2 - 6, 255, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(tile, (1, 1), (Wt - 2, Ht - 2), 255, 2, cv2.LINE_AA)
        for i, w in enumerate(words):
            y = Ht / 2 + (i - (len(words) - 1) / 2) * h * 1.3
            pil = Image.fromarray(tile)
            f = font(fnt, int(h))
            l, t, r, b = f.getbbox(w)
            x0, y0 = Wt / 2 - (r - l) / 2 - l, y - (b - t) / 2 - t
            ImageDraw.Draw(pil).text((x0, y0), w, fill=255, font=f)
            tile = np.asarray(pil).copy()
            tbox[int(y0 + t):int(y0 + b) + 1, int(x0 + l):int(x0 + r) + 1] = 1
        ang = rng.uniform(-25, 25)
        c = rng.uniform(0.15, 0.85, 2) * n
        M = cv2.getRotationMatrix2D((Wt / 2, Ht / 2), ang, 1.0)
        M[:, 2] += c - [Wt / 2, Ht / 2]
        m = cv2.warpAffine(tile, M, (n, n), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
        m *= (rng.random((n, n), dtype=np.float32) > rng.uniform(0.05, 0.3)) * rng.uniform(0.6, 1.0)
        img[:] = (img * (1 - m[..., None]) + colour * m[..., None]).astype(np.uint8)
        self._text_mask |= cv2.warpAffine(tbox, M, (n, n), flags=cv2.INTER_NEAREST)
        self._elements.add("rubber_stamp")

    # ----- panels -----
    def _panel(self, img, maps, rng, w, h):
        """A rectangle anchored near a corner (may run off the crop), cleared to paper with all labels set to 0."""
        n = self.size
        x0 = int(n - w + rng.integers(-n // 10, w // 3)) if rng.random() < 0.65 else int(-rng.integers(-n // 10, w // 3))
        y0 = int(n - h + rng.integers(-n // 10, h // 3)) if rng.random() < 0.65 else int(-rng.integers(-n // 10, h // 3))
        X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, n), min(y0 + h, n)
        if X1 - X0 < 8 or Y1 - Y0 < 8:
            return None
        img[Y0:Y1, X0:X1] = self._paper
        for m in maps + [self._text_mask]:
            m[Y0:Y1, X0:X1] = 0
        return x0, y0, w, h

    def title_block(self, img, maps, st, s, rng):
        n, ink = self.size, st["ink"]
        w, h = int(rng.uniform(0.4, 0.85) * n), int(rng.uniform(0.18, 0.45) * n)
        r = self._panel(img, maps, rng, w, h)
        if r is None:
            return
        x0, y0, w, h = r
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), ink, int(rng.integers(1, 4)))
        k = int(rng.integers(3, 7))
        hs = rng.dirichlet(np.ones(k) * 3) * h
        fnt = st["font"] if rng.random() < 0.4 else self.print[int(rng.integers(len(self.print)))]
        y = float(y0)
        for i, rh in enumerate(hs):
            if i:
                cv2.line(img, (x0, int(y)), (x0 + w, int(y)), ink, 1)
            cols = [x0, x0 + w] if rng.random() < 0.5 else [x0, int(x0 + rng.uniform(0.3, 0.7) * w), x0 + w]
            for c0, c1 in zip(cols[:-1], cols[1:]):
                if c0 > x0:
                    cv2.line(img, (c0, int(y)), (c0, int(y + rh)), ink, 1)
                row = TITLE_ROWS[int(rng.integers(len(TITLE_ROWS)))]
                txt = str(rng.choice(row)).format(n=f"{rng.integers(100, 9999)}-{rng.integers(1, 99):02d}", i=rng.choice(list("ABCDE")),
                                                  d=f"{rng.integers(1, 29):02d}.{rng.integers(1, 13):02d}.{rng.integers(1930, 2025)}",
                                                  a="".join(rng.choice(list("ABCDEFGHKLMNPRSTW"), 2)))
                self.text(img, (c0 + 6, y + rh / 2), txt, min(rh * rng.uniform(0.35, 0.6), 0.6 * s), fnt, ink, align="left")
            y += rh
        if rng.random() < 0.4:                             # north arrow
            self.north_arrow(img, np.array([x0 + w - 25.0, y0 + 30.0]), ink, fnt)
        self._elements.add("title_block")

    def north_arrow(self, img, c, ink, fnt):
        cv2.circle(img, tuple(fx(c)), 16 * 14, ink, 1, cv2.LINE_AA, shift=4)
        cv2.fillPoly(img, [fx([c + [0, -14], c + [6, 10], c + [0, 5], c + [-6, 10]])], ink, cv2.LINE_AA, shift=4)
        self.text(img, c + [0, -24], "N", 12, fnt, ink)

    def legend(self, img, maps, st, s, rng):
        n, ink = self.size, st["ink"]
        k = int(rng.integers(3, 8))
        rh = rng.uniform(16, 30)
        w, h = int(rng.uniform(0.3, 0.55) * n), int(k * rh + 20)
        r = self._panel(img, maps, rng, w, h)
        if r is None:
            return
        x0, y0, w, h = r
        if rng.random() < 0.6:
            cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), ink, 1)
        fnt = st["font"] if rng.random() < 0.4 else self.print[int(rng.integers(len(self.print)))]
        sw = rng.uniform(1.5, 3.0) * rh
        for i in range(k):
            yc = y0 + 10 + (i + 0.5) * rh
            q = [x0 + 10, yc - rh * 0.35, x0 + 10 + sw, yc + rh * 0.35]
            box = shapely.box(*q)
            mat = rng.choice(["concrete_grey", "concrete_black", "masonry", "sandlime", "drywall_cross", "stone", "concrete_dots",
                              "light_grey", "colour", "zigzag"])
            lm = local_mask(box, n)
            if lm is not None:
                m, sl = lm
                if mat == "colour":
                    img[sl][m] = ZONE_COLOURS[int(rng.integers(len(ZONE_COLOURS)))]
                elif mat == "zigzag":
                    img[sl][m & (_zigzag_tex(n, 5, 4, False)[sl] > 0)] = ink
                else:
                    self.material(img, m, mat, st, rng, box=sl)
            stroke(img, box, ink, 1)
            self.text(img, (q[2] + 8, yc), str(rng.choice(LEGEND)), rh * 0.55, fnt, ink, align="left")
        self._elements.add("legend")

    def keyplan(self, img, maps, st, s, interior, rng):
        """Small-scale key plan (Situation): the building footprint filled, with a north arrow and caption."""
        n, ink = self.size, st["ink"]
        w = h = int(rng.uniform(0.25, 0.4) * n)
        r = self._panel(img, maps, rng, w, h)
        if r is None:
            return
        x0, y0, w, h = r
        inner = int(w * 0.7)
        small = cv2.resize(interior * 255, (inner, inner), interpolation=cv2.INTER_AREA) > 127
        if not small.any():
            small = np.zeros((inner, inner), bool)
            small[inner // 4:3 * inner // 4, inner // 5:4 * inner // 5] = True
        tile = np.zeros((n, n), bool)
        X, Y = x0 + (w - inner) // 2, y0 + (h - inner) // 2
        xs0, ys0 = max(X, 0), max(Y, 0)
        xs1, ys1 = min(X + inner, n), min(Y + inner, n)
        if xs1 <= xs0 or ys1 <= ys0:
            return
        tile[ys0:ys1, xs0:xs1] = small[ys0 - Y:ys1 - Y, xs0 - X:xs1 - X]
        img[tile] = ink if rng.random() < 0.5 else (st["grey"],) * 3
        cv2.rectangle(img, (x0 + 4, y0 + 4), (x0 + w - 4, y0 + h - 4), ink, 1)
        fnt = self.print[int(rng.integers(len(self.print)))]
        self.text(img, (x0 + w / 2, y0 + h - 12), str(rng.choice(["Situation 1:500", "Situation", "Plan de situation", "Planimetria 1:1000"])), 11, fnt, ink)
        self.north_arrow(img, np.array([x0 + w - 22.0, y0 + 24.0]), ink, fnt)
        self._elements.add("keyplan")

    # ----- scan and print defects -----
    def degrade(self, img, lab, st, rng, targets=None):
        """Scan and print defects. Geometric ones (wobble, fold, perspective) move labels and targets (a dict, updated in
        place) with the image; the others change the image only. Grey images are processed in one channel."""
        n = self.size
        hist = st["era"] == "historical"
        el = self._elements
        force = set(st.get("defects", ()))                     # previews: defects to apply in any case
        packed = None
        if targets:                                              # 2 bits per head: 0, 1 or 2 = ignore
            packed = np.zeros((n, n), np.uint8)
            for i, k in enumerate(HEADS):
                t = targets[k]
                packed |= (np.where(t == IGNORE, 2, t > 0).astype(np.uint8) << (2 * i))

        def field(k, scale):                                     # smooth random field in [-scale, scale]
            f = cv2.resize(rng.standard_normal((k, k), dtype=np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
            return f * np.float32(scale / max(float(np.abs(f).max()), 1e-6))

        def up(f):                                               # coarse grid (cell centres of an n/16 grid) -> n x n
            return cv2.resize(np.ascontiguousarray(f, np.float32), (n, n), interpolation=cv2.INTER_LINEAR)

        cc = (np.arange(n // 16, dtype=np.float32) + 0.5) * 16 - 0.5
        cx, cy = np.meshgrid(cc, cc)

        def remap(mx, my, border=None):
            nonlocal img, lab, packed
            mx, my = np.ascontiguousarray(mx, np.float32), np.ascontiguousarray(my, np.float32)
            mode = cv2.BORDER_REPLICATE if border is None else cv2.BORDER_CONSTANT
            img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=mode, borderValue=border or 0)
            lab = cv2.remap(lab, mx, my, cv2.INTER_NEAREST, borderMode=mode, borderValue=0)
            if packed is not None:
                packed = cv2.remap(packed, mx, my, cv2.INTER_NEAREST, borderMode=mode, borderValue=0)

        gx, gy = _grid(n)
        if rng.random() < 0.2:                                   # hand-drawn wobble
            amp = rng.uniform(1, 2.5)
            remap(gx + field(n // 24, amp), gy + field(n // 24, amp))
        fold = None
        if rng.random() < (0.25 if hist else 0.1) or "fold" in force:  # fold: small step across a line, shading later
            ang = rng.uniform(0, np.pi)
            nv = np.array([np.cos(ang), np.sin(ang)], np.float32)
            q = rng.uniform(0.2, 0.8, 2) * n
            fold = gx * nv[0] + gy * nv[1] - np.float32(q @ nv)
            shift = np.float32(rng.uniform(0.25, 1.0)) * (1 + np.tanh(fold / np.float32(3)))
            remap(gx - shift * nv[0], gy - shift * nv[1])
            el.add("fold")
        if rng.random() < 0.08 or "perspective" in force:       # photographed sheet: perspective
            j = n * rng.uniform(0.01, 0.05)
            src = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
            dst = (src + rng.uniform(-j, j, (4, 2))).astype(np.float32)
            Minv = cv2.getPerspectiveTransform(dst, src)
            pts = cv2.perspectiveTransform(np.stack([cx, cy], -1).reshape(-1, 1, 2), Minv).reshape(len(cc), len(cc), 2)
            table = tuple(int(v) for v in ((rng.integers(40, 120),) * 3 if rng.random() < 0.5 else (250,) * 3))
            remap(up(pts[..., 0]), up(pts[..., 1]), border=table)
            el.add("perspective")
        if rng.random() < 0.25:                                  # heavier or lighter ink
            k = np.ones((2, 2), np.uint8)
            img = cv2.erode(img, k) if rng.random() < 0.6 else cv2.dilate(img, k)
        grey = (rng.random() < 0.5 or hist) and "colour" not in force
        x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None] if grey else img
        if rng.random() < (0.3 if hist else 0.12) or "fade" in force:  # ink fading, broken strokes
            f = 0.6 + field(8, 0.4) if rng.random() < 0.5 else 0.25 + 0.75 * (field(8, 0.5) + 0.5)
            out = 255 - (255 - x.astype(np.float32)) * np.clip(f, 0.1, 1)[..., None]
            if rng.random() < 0.5:
                drop = cv2.GaussianBlur(rng.random((n, n), dtype=np.float32), (0, 0), 1.0) > rng.uniform(0.6, 0.68)
                out[drop] += (255 - out[drop]) * np.float32(rng.uniform(0.5, 1.0))
            x = np.clip(out, 0, 255).astype(np.uint8)
            el.add("fade")
        if rng.random() < (0.2 if hist else 0.08) or "bleed" in force:  # bleed-through from the back of the sheet
            g = x[..., 0] if grey else cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            g = np.roll(g[:, ::-1], int(rng.integers(n)), axis=int(rng.integers(2)))
            g = cv2.resize(cv2.GaussianBlur(cv2.resize(g, (n // 2, n // 2), interpolation=cv2.INTER_AREA), (0, 0),
                                            rng.uniform(0.5, 1.25)), (n, n))
            c = np.float32(rng.uniform(0.06, 0.18))
            x = (x * (1 - c * (1 - g.astype(np.float32) / 255))[..., None]).astype(np.uint8)
            el.add("bleed")
        if rng.random() < 0.3 or "lowres" in force:             # lost resolution
            f = rng.uniform(1.5, 4.0)
            small = cv2.resize(x, (int(n / f), int(n / f)), interpolation=cv2.INTER_AREA)
            x = cv2.resize(small, (n, n), interpolation=cv2.INTER_LINEAR if rng.random() < 0.8 else cv2.INTER_NEAREST)
            x = x[..., None] if x.ndim == 2 else x
        if rng.random() < 0.5:
            x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.2))
            x = x[..., None] if x.ndim == 2 else x
        out = x.astype(np.float32)
        if rng.random() < 0.3:                                   # yellowed paper, uneven light
            if out.shape[2] == 1:
                out = np.repeat(out, 3, axis=2)
            gx_ = np.linspace(rng.uniform(0.85, 1), rng.uniform(0.85, 1), n, dtype=np.float32)
            out *= gx_[None, :, None] * np.array([1, rng.uniform(0.95, 1), rng.uniform(0.85, 1)], np.float32)
        if rng.random() < 0.12 or "shadow" in force:            # shadow from an edge or a corner
            side = int(rng.integers(4))
            d = [cx, cy, n - 1 - cx, n - 1 - cy][side]
            if rng.random() < 0.4:
                d = np.minimum(d, [cy, cx, n - 1 - cy, n - 1 - cx][side])
            out *= up(1 - rng.uniform(0.2, 0.6) * np.exp(-d / rng.uniform(20, 150)))[..., None]
            el.add("shadow")
        if fold is not None:                                     # fold shading: dark crease, lighter on one side
            a, w = np.float32(rng.uniform(0.15, 0.4)), np.float32(rng.uniform(1.5, 4))
            out *= (1 - a / (1 + (fold / w) ** 2) - np.float32(rng.uniform(0, 0.08)) * (fold > 0))[..., None]
        if rng.random() < 0.5:
            out += rng.standard_normal((n, n, 1), dtype=np.float32) * np.float32(rng.uniform(2, 12))
        if rng.random() < 0.15:                                  # speckle
            m = rng.random((n, n), dtype=np.float32) < rng.uniform(0.0005, 0.003)
            out[m] = rng.uniform(0, 80)
        if rng.random() < 0.05 or "streaks" in force:           # photocopier toner streaks
            ax = int(rng.integers(2))
            for c_ in rng.integers(0, n, int(rng.integers(1, 6))):
                w_ = int(rng.integers(1, 4))
                sl = (slice(None), slice(c_, c_ + w_)) if ax == 0 else (slice(c_, c_ + w_), slice(None))
                out[sl] *= np.float32(rng.uniform(0.5, 0.85))
            el.add("streaks")
        x = np.clip(out, 0, 255).astype(np.uint8)
        if rng.random() < (0.5 if hist else 0.15) or "dither" in force:  # 1-bit scan, sometimes dithered
            g = (x[..., 0] if x.shape[2] == 1 else cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)).astype(np.float32)
            thr = rng.uniform(110, 190)
            if rng.random() < 0.3 or "dither" in force:
                bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16 - 0.5
                g = g + np.tile(bayer, (n // 4 + 1, n // 4 + 1))[:n, :n] * np.float32(rng.uniform(80, 200))
                el.add("dither")
            x = np.where(g < thr, 0, 255).astype(np.uint8)[..., None]
        img = cv2.cvtColor(x[..., 0], cv2.COLOR_GRAY2RGB) if x.shape[2] == 1 else np.ascontiguousarray(x)
        if rng.random() < 0.3:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(20, 90))])
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if targets:
            for i, k in enumerate(HEADS):
                t = (packed >> (2 * i)) & 3
                t[t == 2] = IGNORE
                targets[k] = t
        return img, lab
