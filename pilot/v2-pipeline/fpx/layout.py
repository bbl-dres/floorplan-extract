"""Stage 1b, sheet layout and masking: a rule baseline without training (docs/pipeline.md 1b).

    from fpx import inputs, layout
    pkg = inputs.load("sheet.pdf")[0]
    res = layout.analyse(pkg, texts=ocr_items)          # texts: OCR items in raster px (optional; native text is used)
    res["regions"], res["drawings"], res["sheet"]

Regions (class, polygon in raster px and paper mm, confidence with a reason, source, links) and drawings (region,
kind, title, storey, caption, scale note, mask) of one sheet package or raster. The rules, in order:

- background: the tint around a photographed or scanned sheet (large colour areas connected to the image border) and
  scan edges (thin dark bands along the border) are not content;
- frame: the outermost long, thin, unbroken lines near the border with a free margin inside them (ISO 5457 cut edge
  and frame; the outer walls of a tightly cropped plan have walls attached and openings, so they are no frame); three
  sides, or two opposite ones; everything outside the innermost frame line is ignored;
- title block: a ruled table near a corner, bottom right first (SIA 400, CADexchange 4.3): thin full-width rows (three
  or more), a column divider or a fourth row, closed on at least two sides, words inside (not just numbers: a band of
  dimension chains looks like a table too), little other ink; it may be rotated. Its scale notes become the sheet
  default only; a small cluster right above it is a key plan;
- legend: a column of at least three swatches (filled or outlined rectangles) with read text beside them, standing in
  free space; or a legend keyword (Legende, Légende, Legend, Zeichenerklärung ...) heading a text block or line; or a
  numbered list of four or more lines outside the drawings;
- scale bars: labelled bars (fpx.scale.find_scale_bars, also "10 0 10") and checkered bars with equal divisions in free
  space; north arrows: an "N" (Nord, North) next to a compact symbol (cheap candidates only);
- drawings: the content left after removing the frame, title block, legend, scale bars, north arrows and text boxes,
  closed over cfg.layout_gap_mm (dimension chains stay with their plan, separate drawings stay apart). Clusters that
  interleave (one half inside the other's bounding box) are joined, and so are open fragments (little area against
  their convex hull: walls of a plan whose door openings are drawn as gaps) with their nearest neighbour within
  cfg.layout_fragment_reach gaps. Clusters below cfg.layout_drawing_min_mm, or below 4 % (cfg.layout_minor_share
  squared) of the largest drawing's area, are joined to a drawing nearby (detached chains, labels) or kept as "other";
  clusters made mostly of text are text blocks. Each polygon follows the closed ink (L- and U-shaped plans stay L and
  U), plus text at its edge, minus the title block and legend;
- captions: text lines naming a drawing kind (Grundriss, Erdgeschoss, OG, UG, Schnitt, Ansicht, Situation, Plan,
  étage, rez, piano ...), a storey or a scale, assigned to their nearest drawing within cfg.layout_caption_mm (more for
  large drawings and large lettering; anywhere on a sheet with one drawing), scored by those words, a scale note,
  size, overlap and distance; they give the drawing's title, kind, storey and its own scale note;
- text blocks outside drawings are notes (large ones the sheet title); scale notes outside captions and title blocks
  become scale-note regions and sheet defaults;
- from the package where present: DWG/DXF viewports become drawings with their exact scale and clip polygon; a
  model-space render gives every drawing the exact model scale;
- masks: each drawing's polygon plus cfg.layout_mask_margin_mm (dimension text and chain ends), minus other drawings
  and the title block, legend, key plan, notes and scale notes (viewports: the clip boundary, no margin).

Millimetre thresholds need an exact resolution (a render or a scan header); otherwise they become shares of the sheet
(cfg.layout_gap_share ...). Without text strings (no OCR, no native text) captions, legend keywords and title-block
notes cannot be read; the geometry rules still run. Limits (see the curated check): no unruled cartouches (historical
titles become notes or captions), no revision tables, stamps or colour keys; a site plan cut by streets wider than the
gap, or two halves of a building joined by faint lines, split into several drawings; two plans touching each other,
or a sheet whose frame is broken, stay one; plans drawn with openings as gaps and without sheet context (renders) may
stay in pieces.
"""
import math
import re

import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon, box as sbox

from .config import DEFAULT
from . import scale as fscale

CLASSES = ("drawing", "title block", "legend", "scale bar", "scale note", "north arrow", "notes", "revision table",
           "frame", "key plan", "stamp", "colour key", "caption", "other")
KINDS = ("floor plan", "section", "elevation", "detail", "site plan", "key plan")
COLOURS = {"frame": (150, 150, 150), "title block": (230, 120, 0), "legend": (0, 160, 160), "scale bar": (200, 0, 200),
           "scale note": (200, 0, 120), "north arrow": (120, 0, 220), "notes": (120, 120, 0), "key plan": (220, 0, 0),
           "caption": (0, 120, 255), "drawing": (0, 170, 0), "other": (90, 90, 90)}

# drawing kind by caption words (checked in this order: "plan de situation" is a site plan, not a floor plan)
KIND_WORDS = [
    ("site plan", r"situation|lageplan|site\s*plan|planimetri|katasterplan|übersichtsplan|ubersichtsplan|umgebung"),
    ("section", r"schnitt|coupe|sezione|\bsection|querschnitt"),
    ("elevation", r"ansicht|fassade|fa[cç]ade|elevation|[ée]l[ée]vation|prospetto|aufriss|\bc[oô]t[ée]\s+(du|de|nord|sud|est|ouest)"),
    ("detail", r"\bdetail|d[ée]tail|dettaglio"),
    ("floor plan", r"grundri|erdgescho|obergescho|untergescho|dachgescho|kellergescho|zwischengescho|gescho[sß]|"
                   r"stockwerk|\bstock\b|etage|[ée]tage|rez[- ]de[- ]chauss|\brez\b|sous[- ]sol|souterrain|parterre|"
                   r"\bpiano\b|pianta|pianterreno|floor\s*plan|ground\s*floor|\bfloor\b|ground\s*plan|\bplan\b|\brzut\b|"
                   r"plattegrond|\bplanta\b|\bog\b|\beg\b|\bug\b|\bdg\b"),
]
LEGEND_WORDS = r"legende|l[ée]gende|legend|zeichenerkl|erkl[äa]rung|leggenda|signaturen|explication|key\s*:|экспликац"
NTS_WORDS = r"nicht\s*ma[sß]+st|not\s*to\s*scale|\bn\.?\s*t\.?\s*s\.?\b|sans\s*[ée]chelle|non\s*in\s*scala"
ORD = {"erst": 1, "zweit": 2, "dritt": 3, "viert": 4, "f[üu]nft": 5, "premi": 1, "deuxi": 2, "second": 2, "troisi": 3,
       "quatri": 4, "primo": 1, "secondo": 2, "terzo": 3, "quarto": 4, "first": 1, "third": 3, "fourth": 4}
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}


# ---------------------------------------------------------------------------------------------------------------------
# Caption parsing

def drawing_kind(text):
    """Drawing kind named in a caption, or None."""
    s = text.lower()
    for kind, pat in KIND_WORDS:
        if re.search(pat, s):
            return kind
    return None


def parse_storey(text):
    """Storey named in a caption -> 'EG', 'n. OG', 'n. UG', 'DG' or None (DE/FR/IT/EN, Roman numerals, ordinals)."""
    s = " " + text.lower().replace("ß", "ss") + " "
    up = r"(?:og|obergeschoss\w*|ober-geschoss|stock\w*|etage|[ée]tage|floor|piano|pi[eę]tro)"
    if re.search(r"\b(?:eg|erdgeschoss\w*|erdgescho|parterre|hochparterre|rez[- ]de[- ]chauss\w*|rez\b|piano\s*terr\w*|"
                 r"pianterreno|ground\s*floor|parter)\b", s):
        return "EG"
    m = re.search(r"\b(\d{1,2})\s*\.?\s*(?:ug|untergeschoss\w*|kellergeschoss\w*|sous[- ]sol|seminterrato|basement)\b", s)
    if m:
        return f"{m.group(1)}. UG"
    if re.search(r"\b(?:ug|untergeschoss\w*|kellergeschoss\w*|keller|sous[- ]sol|seminterrato|basement|souterrain)\b", s):
        return "UG"
    if re.search(r"\b(?:dg|dachgeschoss\w*|dachstock|attika|attique|combles|sottotetto|attic)\b", s):
        return "DG"
    m = re.search(rf"\b(\d{{1,2}})\s*(?:\.|er|e|ème|eme|st|nd|rd|th|°)?\s*{up}\b", s)
    if m:
        return f"{int(m.group(1))}. OG"
    m = re.search(rf"\b([ivx]{{1,4}})\s*\.?\s*{up}\b", s)
    if m and m.group(1) in ROMAN:
        return f"{ROMAN[m.group(1)]}. OG"
    for w, n in ORD.items():
        if re.search(rf"\b{w}\w*\s+{up}", s):
            return f"{n}. OG"
    if re.search(r"\b(?:og|obergescho\w*|ober-geschoss\w*)\b", s):
        return "OG"                                      # an upper floor without its number
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Small helpers

def _box_poly(b):
    return sbox(*b)


def _poly_px(poly):
    """Shapely polygon -> [[x, y], ...] (exterior, rounded)."""
    if poly is None or poly.is_empty:
        return []
    if poly.geom_type != "Polygon":
        poly = max(shapely.get_parts(poly), key=lambda p: p.area)
    return [[round(float(x), 1), round(float(y), 1)] for x, y in list(poly.exterior.coords)[:-1]]


def _lines_of(texts, join=1.0):
    """Text items -> lines: items on one baseline (centres within half a height) closer than join x height are joined
    left to right. Returns [{"text", "box", "height", "items"}] (horizontal items; vertical ones stay single)."""
    items = sorted([t for t in texts if t.get("text", "").strip()], key=lambda t: (t["box"][1], t["box"][0]))
    lines = []
    for t in items:
        x0, y0, x1, y1 = t["box"]
        h = max(y1 - y0, 1.0) if t.get("angle", 0) == 0 else max(x1 - x0, 1.0)
        placed = False
        if t.get("angle", 0) == 0:
            for ln in lines:
                if ln["angle"] != 0:
                    continue
                lx0, ly0, lx1, ly1 = ln["box"]
                lh = ln["height"]
                if abs((y0 + y1) / 2 - (ly0 + ly1) / 2) < 0.5 * max(h, lh) and 0.5 < h / lh < 2 and \
                        max(x0 - lx1, lx0 - x1) < join * max(h, lh):
                    ln["items"].append(t)
                    ln["box"] = (min(lx0, x0), min(ly0, y0), max(lx1, x1), max(ly1, y1))
                    ln["height"] = max(lh, h)
                    placed = True
                    break
        if not placed:
            lines.append({"box": tuple(t["box"]), "height": h, "items": [t], "angle": t.get("angle", 0)})
    for ln in lines:
        ln["items"].sort(key=lambda t: t["box"][0] if ln["angle"] == 0 else t["box"][1])
        ln["text"] = " ".join(t["text"].strip() for t in ln["items"])
    return lines


def _runs(v):
    d = np.diff(np.r_[0, v.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _fill_holes(mask):
    m = mask.astype(np.uint8)
    h, w = m.shape
    ff = np.pad(m, 1, constant_values=0).copy()
    cv2.floodFill(ff, None, (0, 0), 2)
    return (ff[1:-1, 1:-1] != 2)


def _rect_dist(a, b):
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


# ---------------------------------------------------------------------------------------------------------------------
# The analysis

class _Sheet:
    """Working copy of the sheet: reduced raster, ink, content, text boxes, scale of the thresholds."""

    def __init__(self, img, dpi, texts, cfg):
        self.cfg = cfg
        self.H, self.W = img.shape[:2]
        self.f = min(1.0, cfg.layout_max_side / max(self.H, self.W))
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        self.ink_full = fscale.binarise(grey)                       # full resolution: scale cues and bar search
        if self.f < 1:
            size = (max(1, int(round(self.W * self.f))), max(1, int(round(self.H * self.f))))
            self.ink = cv2.resize(self.ink_full.astype(np.uint8) * 255, size, interpolation=cv2.INTER_AREA) > 20
            rgb = cv2.resize(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), size, interpolation=cv2.INTER_AREA)
        else:
            self.ink = self.ink_full.copy()
            rgb = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        paper = np.median(rgb.reshape(-1, 3)[:: max(1, rgb.shape[0] * rgb.shape[1] // 200000)], axis=0)
        self.paper = paper
        dist = np.abs(rgb.astype(np.int16) - paper.astype(np.int16)).max(2)
        self.colour = (dist > 45) & ~self.ink                        # fills and tints that differ from the paper
        self.h, self.w = self.ink.shape
        # the background around a photographed or scanned sheet: large tints connected to the image border
        n, lab, st, _ = cv2.connectedComponentsWithStats(self.colour.astype(np.uint8), connectivity=8)
        edge = np.zeros(n, bool)
        edge[np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])] = True
        edge[0] = False
        edge &= st[:, 4] > 0.01 * self.h * self.w
        self.colour &= ~edge[lab]
        # scan edges: thin dark bands along the image border, not connected to the drawing
        n, lab, st, _ = cv2.connectedComponentsWithStats(self.ink.astype(np.uint8), connectivity=8)
        drop = np.zeros(n, bool)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            touch_v = x <= 0.01 * self.w or x + w >= 0.99 * self.w
            touch_h = y <= 0.01 * self.h or y + h >= 0.99 * self.h
            if (touch_v and w <= 0.03 * self.w and h >= 0.2 * self.h) or (touch_h and h <= 0.03 * self.h and w >= 0.2 * self.w):
                drop[i] = True
        self.scan_edges = drop[lab]
        self.ink = self.ink & ~self.scan_edges
        self.content = self.ink | self.colour
        self.mm = None if not dpi else dpi / 25.4 * self.f          # working pixels per paper millimetre
        self.texts = [t for t in (texts or []) if t.get("box") is not None]
        self.tw = [tuple(v * self.f for v in t["box"]) for t in self.texts]       # boxes in working px
        hs = [min(b[2] - b[0], b[3] - b[1]) for b, t in zip(self.tw, self.texts) if t.get("text", "").strip()]
        self.text_h = float(np.median(hs)) if hs else None
        L = max(self.h, self.w)
        if self.mm:
            self.gap = cfg.layout_gap_mm * self.mm
            self.min_side = cfg.layout_drawing_min_mm * self.mm
            self.unit = self.mm                                       # working px per (nominal) millimetre
        else:
            self.gap = cfg.layout_gap_share * L
            self.min_side = cfg.layout_drawing_min_share * L
            self.unit = self.gap / cfg.layout_gap_mm
        self.gap = max(self.gap, 3.0)
        self.caption = cfg.layout_caption_mm * self.unit
        self.margin = cfg.layout_mask_margin_mm * self.unit

    def full(self, b):
        return tuple(v / self.f for v in b)


def analyse(sheet, texts=None, cfg=DEFAULT, dpi=None, viewports=None, model_scale=None, paper_mm_per_px=None):
    """Regions and drawings of one sheet. sheet: a fpx.inputs.SheetPackage or an RGB/grey image. texts: OCR items
    {text, box (raster px), height, angle} to add to the package's native text. Returns
    {"regions": [...], "drawings": [...], "sheet": {...}, "flags": [...]} with polygons in raster px (and paper mm when
    the resolution is known)."""
    if hasattr(sheet, "img"):
        pkg = sheet
        img = pkg.img
        if dpi is None:                                  # millimetre thresholds only with an exact resolution
            dpi = pkg.dpi if pkg.dpi_trusted else None
        viewports = pkg.viewports if viewports is None else viewports
        model_scale = pkg.model_scale if model_scale is None else model_scale
        native = [t for t in pkg.text if not t.get("invisible")]
    else:
        pkg, img, native = None, sheet, []
    all_texts = _merge_texts(native, texts or [])
    S = _Sheet(img, dpi, all_texts, cfg)
    regions, flags = [], []

    def add(cls, poly_w, conf, reason, source="rule", **extra):
        """poly_w: shapely polygon in working px."""
        poly = shapely.affinity.scale(poly_w, 1 / S.f, 1 / S.f, origin=(0, 0)) if S.f != 1 else poly_w
        r = {"id": f"r{len(regions) + 1}", "class": cls, "polygon_px": _poly_px(poly),
             "bbox_px": [round(float(v), 1) for v in poly.bounds], "confidence": conf, "reason": reason,
             "source": source, "links": {}, **extra}
        r["_w"] = poly_w
        regions.append(r)
        return r

    removed = np.zeros((S.h, S.w), bool)                 # ink that belongs to non-drawing regions

    # --- frame
    frame, frame_lines = _frame(S)
    inner = np.ones((S.h, S.w), bool)
    if frame is not None:
        fx0, fy0, fx1, fy1 = frame["box"]
        add("frame", sbox(fx0, fy0, fx1, fy1), "high" if frame["sides"] == 4 else "medium",
            f"long lines on {frame['sides']} sides near the border ({frame['note']})")
        inner[:] = False
        ix0, iy0, ix1, iy1 = frame["inner"]                  # content lies inside the innermost frame line
        inner[int(iy0):int(iy1) + 1, int(ix0):int(ix1) + 1] = True
        removed |= frame_lines
    area = (S.w * S.h) if frame is None else (frame["box"][2] - frame["box"][0]) * (frame["box"][3] - frame["box"][1])
    content_box = frame["box"] if frame is not None else (0, 0, S.w, S.h)

    # --- text boxes (working px)
    text_mask = _text_mask(S)

    # --- title block (ruled table near a corner) and key plan
    tb = _title_block(S, removed, text_mask, content_box, area)
    tb_region = None
    if tb is not None:
        tb_region = add("title block", sbox(*tb["box"]), tb["confidence"], tb["reason"])
        removed[int(tb["box"][1]):int(tb["box"][3]) + 1, int(tb["box"][0]):int(tb["box"][2]) + 1] = True

    # --- scale bars (labelled: fpx.scale.find_scale_bars; checkered bars)
    bars = []
    for b in fscale.find_scale_bars(S.texts, S.ink_full, cfg):
        bw = tuple(v * S.f for v in b["box"])
        bars.append(add("scale bar", sbox(*bw).buffer(2), "medium", f"labelled scale bar: {b['evidence']}",
                        px_per_m=b["px_per_m"]))
    for bw in _checkered_bars(S, text_mask):
        if not any(_rect_dist(bw, r["_w"].bounds) < 2 for r in bars):
            bars.append(add("scale bar", sbox(*bw).buffer(2), "low", "checkered bar (alternating filled segments)"))
    for r in bars:
        x0, y0, x1, y1 = (int(v) for v in r["_w"].bounds)
        removed[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = True

    # --- legend
    for lg in _legends(S, text_mask, removed):
        r = add("legend", sbox(*lg["box"]), lg["confidence"], lg["reason"])
        x0, y0, x1, y1 = (int(v) for v in lg["box"])
        removed[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = True

    # --- north arrow candidates
    for na in _north_arrows(S, text_mask):
        add("north arrow", sbox(*na["box"]), "low", na["reason"])
        x0, y0, x1, y1 = (int(v) for v in na["box"])
        removed[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = True

    # --- drawings from the package (DWG/DXF viewports), then ink clusters
    vp_drawings = []
    for vp in viewports or []:
        clip = Polygon(vp["clip_px"])
        if clip.is_empty or clip.area <= 0:
            continue
        cw = shapely.affinity.scale(clip, S.f, S.f, origin=(0, 0)) if S.f != 1 else clip
        r = add("drawing", cw, "high", f"layout viewport {vp['id']} (clip {vp.get('clip_kind', 'rectangle')})",
                source="dwg-viewport", viewport=vp["id"])
        vp_drawings.append((r, vp))
    clusters = _drawing_clusters(S, removed | ~inner, text_mask)
    rule_drawings = []
    for c in clusters:
        poly = c["poly"]
        if any(poly.intersection(r["_w"]).area > 0.3 * poly.area for r, _ in vp_drawings):
            continue                                     # the viewport already gives this drawing
        for other in regions:
            if other["class"] in ("title block", "legend"):
                poly = poly.difference(other["_w"].buffer(1))
        if poly.is_empty:
            continue
        poly = max(shapely.get_parts(poly), key=lambda p: p.area) if poly.geom_type != "Polygon" else poly
        rule_drawings.append((c, poly))
    # key plan: a small cluster in or right above the title block; minor clusters next to a much larger drawing
    # (symbols, stamps, archive labels, a stray arrow) are kept as "other" regions, not drawings
    drawings_w = []
    big_area = max((p.area for c, p in rule_drawings if not c.get("small")), default=0)
    majors, minors = [], []
    for c, poly in rule_drawings:
        if tb is not None and _is_key_plan(poly, tb["box"], [p for c2, p in rule_drawings if not c2.get("small")], S):
            kp = add("key plan", poly, "low", "small closed outline in or above the title block")
            if tb_region is not None:
                kp["links"]["title_block"] = tb_region["id"]
                tb_region["links"].setdefault("key_plans", []).append(kp["id"])
            continue
        (minors if c.get("small") or poly.area < cfg.layout_minor_share ** 2 * big_area else majors).append([c, poly])
    reach = max(3 * S.gap, S.caption)
    for c, poly in minors:
        # detached dimension chains, labels and symbols next to a drawing belong to it (the scale cues need the
        # chains); far from every drawing they are kept as "other"
        near = min(majors, key=lambda m: m[1].distance(poly), default=None)
        if near is not None and near[1].distance(poly) <= reach:
            d = near[1].distance(poly) / 2 + 1
            u = shapely.union_all([near[1].buffer(d, join_style="mitre"), poly.buffer(d, join_style="mitre")]).buffer(-d, join_style="mitre")
            u = max(shapely.get_parts(u), key=lambda p: p.area) if u.geom_type != "Polygon" else u
            if not u.is_empty:
                near[1] = Polygon(u.exterior)
                near[0] = dict(near[0], parts=near[0].get("parts", 0) + 1)
            continue
        if not c.get("small"):
            add("other", poly, "low", f"small ink cluster ({poly.area / max(big_area, 1):.1%} of the largest drawing's "
                                      "area) away from the drawings: a symbol, stamp or label, or a small detail")
    for c, poly in majors:
        why = c["reason"] + (f"; {c['parts']} small detached parts joined (dimension chains, labels)" if c.get("parts") else "")
        drawings_w.append(add("drawing", poly, c["confidence"], why, source="rule"))
    drawings_w = [r for r, _ in vp_drawings] + drawings_w

    # --- text lines: captions, title-block notes, legend text, notes
    lines = _lines_of(S.texts)
    for ln in lines:
        ln["box_w"] = tuple(v * S.f for v in ln["box"])
        ln["h_w"] = ln["height"] * S.f
    captions = _captions(S, lines, drawings_w, regions)
    sheet_notes = []
    if tb_region is not None:
        tbp = tb_region["_w"]
        tb_lines = [ln for ln in lines if tbp.contains(shapely.Point(_centre(ln["box_w"])))]
        tb_region["text"] = [ln["text"] for ln in tb_lines][:60]
        for ln in tb_lines:
            for n, kw in fscale.parse_scale_note(ln["text"]):
                sheet_notes.append({"scale": n, "text": ln["text"], "keyword": kw})
        if sheet_notes:
            tb_region["scale_notes"] = sorted({s["scale"] for s in sheet_notes})
    used = set()
    for d, cap in captions.items():
        for ln in cap["lines"]:
            used.add(id(ln))
    # notes and loose scale notes: text lines outside drawings, title block, legends and captions
    occupied = [r["_w"] for r in regions if r["class"] in ("title block", "legend", "scale bar", "north arrow", "key plan")]
    loose = [ln for ln in lines if id(ln) not in used
             and not any(p.contains(shapely.Point(_centre(ln["box_w"]))) for p in occupied)
             and not any(r["_w"].buffer(-0.5 * ln["h_w"]).contains(shapely.Point(_centre(ln["box_w"]))) for r in drawings_w)]
    for blk in _blocks(loose):
        bx = blk["box_w"]
        bp = sbox(*bx)
        if any(r["_w"].intersection(bp).area > 0.3 * bp.area for r in drawings_w):
            continue                                     # text inside a drawing (labels, OCR of hatching)
        notes_n = [n for ln in blk["lines"] for n, _ in fscale.parse_scale_note(ln["text"])]
        txt = " / ".join(ln["text"] for ln in blk["lines"])
        if notes_n and len(txt) <= 40:
            r = add("scale note", sbox(*bx).buffer(1), "medium", f"scale note outside the drawings: {txt[:60]!r}",
                    scales=sorted(set(notes_n)), text=txt[:200])
            sheet_notes += [{"scale": n, "text": txt, "keyword": True, "loose": True} for n in notes_n]
            continue
        if len(blk["lines"]) >= 2 or len(txt) >= 12 or blk["height"] >= 1.8 * (S.text_h or blk["height"]):
            big = S.text_h and blk["height"] >= 1.8 * S.text_h and len(blk["lines"]) <= 2
            add("notes", sbox(*bx).buffer(1), "low", "sheet title (large text outside the drawings)" if big else
                f"text block outside the drawings ({len(blk['lines'])} lines)", text=txt[:300],
                role="sheet title" if big else "notes")

    # --- drawings: caption, kind, storey, own scale note, mask
    drawings = []
    blockers = [r for r in regions if r["class"] in ("title block", "legend", "key plan", "notes", "scale note")]
    for k, r in enumerate(drawings_w, 1):
        cap = captions.get(r["id"])
        vp = next((v for rr, v in vp_drawings if rr is r), None)
        title = cap["text"] if cap else None
        kind = (drawing_kind(title) if title else None)
        storey = parse_storey(title) if title else None
        notes = [n for ln in (cap["lines"] if cap else []) for n, _ in fscale.parse_scale_note(ln["text"])]
        nts = bool(title and re.search(NTS_WORDS, title.lower()))
        cap_region = None
        if cap:
            cap_region = add("caption", sbox(*cap["box_w"]).buffer(1), "medium" if cap["score"] >= 3 else "low",
                             f"caption of {r['id']}: {cap['why']}", text=title)
            cap_region["links"]["drawing"] = r["id"]
            r["links"]["caption"] = cap_region["id"]
        conf_kind = "medium" if kind else "low"
        if kind is None:
            kind = "floor plan"
            kind_reason = "no drawing word in a caption: assumed floor plan (to be confirmed)"
        else:
            kind_reason = f"caption word: {title[:60]!r}"
        own = sorted(set(notes))
        scale_note = {"scale": own[0] if len(own) == 1 else None, "all": own, "source": "caption"} if own else \
            {"scale": None, "all": [], "source": None}
        if vp is not None and vp.get("scale"):
            scale_note = {"scale": vp["scale"], "all": [vp["scale"]], "source": "viewport",
                          "caption_notes": own}
        poly_w = r["_w"]
        margin = 0.0 if vp is not None else S.margin
        mask = poly_w.buffer(margin, join_style="mitre") if margin else poly_w
        for other in drawings_w:
            if other is not r:
                mask = mask.difference(other["_w"])
        for b in blockers:
            if b is not cap_region and not b["_w"].within(poly_w):
                mask = mask.difference(b["_w"])
        mask = mask.intersection(sbox(0, 0, S.w, S.h))
        mask_full = shapely.affinity.scale(mask, 1 / S.f, 1 / S.f, origin=(0, 0)) if S.f != 1 else mask
        d = {"id": f"d{k}", "region": r["id"], "kind": kind, "kind_confidence": conf_kind, "kind_reason": kind_reason,
             "title": title, "storey": storey, "caption": None if cap is None else
             {"region": cap_region["id"], "text": title, "box_px": [round(v / S.f, 1) for v in cap["box_w"]],
              "score": cap["score"], "why": cap["why"]},
             "scale_note": scale_note, "not_to_scale": nts, "polygon_px": r["polygon_px"], "bbox_px": r["bbox_px"],
             "source": r["source"], "confidence": r["confidence"],
             "mask": {"polygon_px": _poly_px(mask_full) if not mask_full.is_empty else [],
                      "parts": [_poly_px(p) for p in shapely.get_parts(mask_full)] if mask_full.geom_type != "Polygon" else None,
                      "margin_px": round(margin / S.f, 1), "margin_mm": cfg.layout_mask_margin_mm if margin else 0.0,
                      "reason": "drawing polygon plus a margin for dimension chains, minus other drawings and regions"
                                if margin else "viewport clip boundary (exact)"}}
        if vp is not None:
            d["viewport"] = {k2: v2 for k2, v2 in vp.items() if k2 != "model_to_px"}
        if model_scale:
            d["model_scale"] = model_scale
        d["_mask_full"] = mask_full
        drawings.append(d)
    if tb_region is not None and sheet_notes:
        for d in drawings:
            d["sheet_scale_notes"] = sorted({s["scale"] for s in sheet_notes})
    all_scales = sorted({n for d in drawings for n in d["scale_note"]["all"]} | {s["scale"] for s in sheet_notes})
    if len(all_scales) > 1:
        flags.append({"check": "scale", "severity": "low", "element": "sheet",
                      "message": f"several scales on the sheet: {', '.join(f'1:{n}' for n in all_scales)}"})
    if not drawings:
        flags.append({"check": "layout", "severity": "high", "element": "sheet", "message": "no drawing found on the sheet"})

    # paper millimetres and clean-up
    pxmm = None if not dpi else dpi / 25.4
    for r in regions:
        r.pop("_w", None)
        r["polygon_mm"] = None if not pxmm else [[round(x / pxmm, 2), round(y / pxmm, 2)] for x, y in r["polygon_px"]]
    for d in drawings:
        d["polygon_mm"] = None if not pxmm else [[round(x / pxmm, 2), round(y / pxmm, 2)] for x, y in d["polygon_px"]]
    sheet_info = {"raster_px": [S.W, S.H], "work_scale": round(S.f, 5), "dpi": dpi,
                  "thresholds_px": {"gap": round(S.gap / S.f, 1), "min_drawing": round(S.min_side / S.f, 1),
                                    "caption": round(S.caption / S.f, 1), "margin": round(S.margin / S.f, 1),
                                    "basis": "paper millimetres" if S.mm else "shares of the sheet (resolution unknown)"},
                  "text_items": len(S.texts), "median_text_px": None if S.text_h is None else round(S.text_h / S.f, 1),
                  "sheet_scale_notes": sorted({s["scale"] for s in sheet_notes}),
                  "title_block": None if tb_region is None else tb_region["id"]}
    return {"regions": regions, "drawings": drawings, "sheet": sheet_info, "flags": flags}


def _centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _merge_texts(native, ocr):
    """Native text runs plus OCR items that no native run covers (by 30 % of the OCR box)."""
    out = [dict(t) for t in native]
    nb = [t["box"] for t in native if t.get("text")]
    for t in ocr:
        b = t["box"]
        a = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
        if any(max(0, min(b[2], n[2]) - max(b[0], n[0])) * max(0, min(b[3], n[3]) - max(b[1], n[1])) > 0.3 * a for n in nb):
            continue
        out.append(dict(t))
    return out


# --- frame ---

def _frame(S):
    """Outermost long horizontal and vertical lines in the border bands -> frame box (working px) and the mask of all
    long lines in those bands (cut edge and frame), or (None, empty mask)."""
    ink = S.ink.astype(np.uint8)
    share = S.cfg.layout_frame_min_share
    kh = max(10, int(0.25 * S.w))
    kv = max(10, int(0.25 * S.h))
    # long lines, tolerant of a slight tilt (scans): opened on the ink thickened by a pixel across the line
    hl = (cv2.morphologyEx(cv2.dilate(ink, np.ones((3, 1), np.uint8)), cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1))) > 0) & (ink > 0)
    vl = (cv2.morphologyEx(cv2.dilate(ink, np.ones((1, 3), np.uint8)), cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv))) > 0) & (ink > 0)
    rows = _thin(np.flatnonzero(hl.sum(1) >= share * S.w), max(3, 0.006 * S.h))
    cols = _thin(np.flatnonzero(vl.sum(0) >= share * S.h), max(3, 0.006 * S.w))
    band_h, band_w = 0.2 * S.h, 0.2 * S.w
    other = S.ink & ~hl & ~vl

    def free(idx, axis, inward):
        """Lines with a free margin inside them: a frame keeps its distance from the drawing, while the outer wall of
        a tightly cropped plan has walls attached on its inner side. Keeps the lines up to the innermost free one."""
        side = S.h if axis == 0 else S.w
        d = max(3, int(0.015 * side))
        ok = []
        for i in (sorted(idx) if inward > 0 else sorted(idx, reverse=True)):
            a, b = (i + 2, i + 2 + d) if inward > 0 else (i - 1 - d, i - 1)
            a, b = max(0, a), max(0, b)
            band = other[a:b] if axis == 0 else other[:, a:b]
            touched = band.any(axis=0).mean() if axis == 0 else band.any(axis=1).mean()
            if touched < 0.3:
                ok.append(i)
        if not ok:
            return np.asarray([], int)
        first = ok[0]                                    # the outermost free line and the nested ones near it
        return np.asarray([i for i in idx if abs(i - first) <= 0.05 * side and (i - first) * inward >= -1], int)

    top = free(rows[rows < band_h], 0, 1)
    bot = free(rows[rows > S.h - band_h], 0, -1)
    lef = free(cols[cols < band_w], 1, 1)
    rig = free(cols[cols > S.w - band_w], 1, -1)
    # a frame line runs unbroken between its corners; the outer walls of a tightly cropped plan have openings
    for _ in range(2):
        x0 = int(lef.min()) if len(lef) else 0
        x1 = int(rig.max()) if len(rig) else S.w - 1
        y0 = int(top.min()) if len(top) else 0
        y1 = int(bot.max()) if len(bot) else S.h - 1
        cov_h = lambda rs: max((hl[r, x0:x1 + 1].mean() for r in rs), default=0.0)
        cov_v = lambda cs: max((vl[y0:y1 + 1, c].mean() for c in cs), default=0.0)
        top = top if cov_h(top) >= 0.85 else top[:0]
        bot = bot if cov_h(bot) >= 0.85 else bot[:0]
        lef = lef if cov_v(lef) >= 0.85 else lef[:0]
        rig = rig if cov_v(rig) >= 0.85 else rig[:0]
    sides = sum(len(v) > 0 for v in (top, bot, lef, rig))
    lines = np.zeros_like(S.ink)
    opposite = (len(top) and len(bot)) or (len(lef) and len(rig))
    if sides < 3 and not (sides == 2 and opposite):  # two opposite sides: the others lie on the image border
        return None, lines
    y0 = int(top.min()) if len(top) else 0
    y1 = int(bot.max()) if len(bot) else S.h - 1
    x0 = int(lef.min()) if len(lef) else 0
    x1 = int(rig.max()) if len(rig) else S.w - 1
    if (x1 - x0) < 0.5 * S.w or (y1 - y0) < 0.5 * S.h:
        return None, lines
    for r in np.r_[top, bot]:
        lines[max(0, r - 1):r + 2] |= hl[max(0, r - 1):r + 2]
    for c in np.r_[lef, rig]:
        lines[:, max(0, c - 1):c + 2] |= vl[:, max(0, c - 1):c + 2]
    nested = (len(top) and np.ptp(top) > 3) or (len(bot) and np.ptp(bot) > 3)
    note = "cut edge and frame" if nested else "one line per side"
    # the content area is inside the innermost frame line
    iy0 = int(top.max()) if len(top) else 0
    iy1 = int(bot.min()) if len(bot) else S.h - 1
    ix0 = int(lef.max()) if len(lef) else 0
    ix1 = int(rig.min()) if len(rig) else S.w - 1
    return {"box": (x0, y0, x1, y1), "inner": (ix0, iy0, ix1, iy1), "sides": sides, "note": note}, lines


def _thin(idx, max_run):
    """Rows (or columns) of long lines, keeping only runs no thicker than max_run: frame lines are thin; a plan's
    solid outer wall that happens to run along the border is not a frame."""
    if not len(idx):
        return idx
    keep = []
    start = 0
    for k in range(1, len(idx) + 1):
        if k == len(idx) or idx[k] != idx[k - 1] + 1:
            if idx[k - 1] - idx[start] + 1 <= max_run:
                keep += list(idx[start:k])
            start = k
    return np.asarray(keep, int)


def _busy(S, box, pad, text_mask):
    """Share of content pixels in a ring of width pad around box (text excluded): legends and scale bars sit in free
    space, columns and furniture inside a plan do not."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    X0, Y0 = max(0, x0 - int(pad)), max(0, y0 - int(pad))
    X1, Y1 = min(S.w, x1 + int(pad)), min(S.h, y1 + int(pad))
    ring = np.ones((Y1 - Y0, X1 - X0), bool)
    ring[max(0, y0 - Y0):max(0, y1 - Y0), max(0, x0 - X0):max(0, x1 - X0)] = False
    sub = S.content[Y0:Y1, X0:X1] & ~text_mask[Y0:Y1, X0:X1] & ring
    return float(sub.sum() / max(ring.sum(), 1))


# --- text ---

def _text_mask(S):
    """Union of the text boxes (working px), without boxes taller than 4 x the median text height (OCR artefacts over
    hatching), plus outlined-text boxes."""
    m = np.zeros((S.h, S.w), np.uint8)
    for b, t in zip(S.tw, S.texts):
        bh = min(b[2] - b[0], b[3] - b[1])
        if S.text_h and bh > 4 * S.text_h:
            continue
        p = 0.1 * bh
        cv2.rectangle(m, (int(b[0] - p), int(b[1] - p)), (int(math.ceil(b[2] + p)), int(math.ceil(b[3] + p))), 1, -1)
    return m > 0


# --- title block ---

def _title_block(S, removed, text_mask, content_box, area):
    """Best ruled table near a corner: candidates are components of long horizontal and vertical line pieces (at least
    ~12 mm on paper or 1.5 % of the sheet), with at least three rows spanning half its width, a vertical divider or a
    fourth row, text inside, and little other ink; scored by the SIA 400 corner prior (bottom right first)."""
    ink = (S.ink & ~removed).astype(np.uint8)
    k = int(max(8, min(12 * S.unit, 0.03 * max(S.w, S.h))))
    hl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    vl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(6, k // 2))))
    lines = cv2.dilate(hl | vl, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    cx0, cy0, cx1, cy1 = content_box
    cw, ch = cx1 - cx0, cy1 - cy0
    best = None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < 0.08 * cw or h < 0.025 * ch or w * h > S.cfg.layout_title_block_max_share * area:
            continue
        if w < 4 * k / 3 or h < 2 * max(3, S.text_h or 3):
            continue
        sub_h = hl[y:y + h, x:x + w] > 0
        sub_v = vl[y:y + h, x:x + w] > 0
        row_hits = np.flatnonzero(sub_h.sum(1) >= 0.5 * w)
        rows = len(_runs(np.isin(np.arange(h), row_hits))) if len(row_hits) else 0
        col_hits = np.flatnonzero(sub_v.sum(0) >= 0.3 * h)
        cols = len(_runs(np.isin(np.arange(w), col_hits))) if len(col_hits) else 0
        if rows < 3 or (cols < 2 and rows < 4):
            continue
        # ruled with thin lines (walls of a plan are thick) and enclosed by them on at least two sides
        thick = [b - a for a, b in _runs(np.isin(np.arange(h), row_hits))]
        if np.median(thick) > max(3.0, 0.004 * max(S.w, S.h)):
            continue
        sides = int(sub_h[:max(2, h // 20)].sum(1).max() >= 0.9 * w) + int(sub_h[-max(2, h // 20):].sum(1).max() >= 0.9 * w) \
            + int(sub_v[:, :max(2, w // 20)].sum(0).max() >= 0.9 * h) + int(sub_v[:, -max(2, w // 20):].sum(0).max() >= 0.9 * h)
        if sides < 2:
            continue
        # text inside, little other ink
        tb = (x, y, x + w, y + h)
        inside = [t for b, t in zip(S.tw, S.texts) if tb[0] <= (b[0] + b[2]) / 2 <= tb[2] and tb[1] <= (b[1] + b[3]) / 2 <= tb[3]]
        texts_in = len(inside)
        if texts_in >= 3:
            # a title block holds words (it may be rotated); a band of dimension chains holds numbers
            words = sum(1 for t in inside if sum(ch.isalpha() for ch in t.get("text", "")) >= 2) / texts_in
            if words < 0.3:
                continue
        other = (S.ink[y:y + h, x:x + w] & ~(lines[y:y + h, x:x + w] > 0) & ~text_mask[y:y + h, x:x + w]).mean()
        chars = _char_count(S.ink[y:y + h, x:x + w] & ~(lines[y:y + h, x:x + w] > 0), S)
        if texts_in < 2 and chars < 8:
            continue
        if other > 0.2:
            continue
        # SIA 400 / CADexchange prior: bottom right first
        near = 0.06
        right = (cx1 - (x + w)) < near * cw
        left = (x - cx0) < near * cw
        bottom = (cy1 - (y + h)) < near * ch
        topp = (y - cy0) < near * ch
        prior = 1.0 if (right and bottom) else 0.7 if (left and bottom) else 0.6 if (right and topp) else \
            0.5 if (left and topp) else 0.45 if (bottom or right) else 0.2
        score = prior * (0.5 + min(rows, 8) / 8) * (1.0 if texts_in >= 3 or chars >= 20 else 0.7)
        why = (f"ruled table, {rows} rows and {cols} columns, {texts_in} text items, "
               f"{'bottom right' if right and bottom else 'bottom left' if left and bottom else 'top right' if right and topp else 'top left' if left and topp else 'edge' if bottom or right else 'inside the sheet'}"
               f" (corner prior {prior:g})")
        if best is None or score > best["score"]:
            best = {"box": (x, y, x + w, y + h), "score": score, "reason": why,
                    "confidence": "high" if score >= 0.9 else "medium" if score >= 0.5 else "low"}
    if best is not None and best["score"] < 0.35:
        return None
    return best


def _char_count(ink, S):
    """Number of character-like components (compact, text height) in a crop."""
    if ink.size == 0 or not ink.any():
        return 0
    n, _, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    w, h, a = st[1:, 2].astype(float), st[1:, 3].astype(float), st[1:, 4].astype(float)
    hmin = 2 if S.text_h is None else 0.4 * S.text_h
    hmax = 40 if S.text_h is None else 2.5 * S.text_h
    ok = (h >= hmin) & (h <= hmax) & (w <= 1.5 * h) & (a >= 0.1 * w * h)
    return int(ok.sum())


def _is_key_plan(poly, tb_box, all_polys, S):
    """A small drawing cluster inside the title block or right above it (SIA 400: key plan in the fields above)."""
    x0, y0, x1, y1 = tb_box
    tw = x1 - x0
    px0, py0, px1, py1 = poly.bounds
    big = max((max(p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1]) for p in all_polys), default=0)
    side = max(px1 - px0, py1 - py0)
    if side > 0.35 * big and len(all_polys) > 1:
        return False
    if side > 1.2 * tw:
        return False
    inside = px0 >= x0 - 2 and px1 <= x1 + 2 and py0 >= y0 - 2 and py1 <= y1 + 2
    above = px0 >= x0 - 0.1 * tw and px1 <= x1 + 0.1 * tw and py1 <= y0 + 2 and y0 - py1 < max(y1 - y0, side)
    return bool(inside or above) and len(all_polys) > 1


# --- scale bars and north arrows ---

def _checkered_bars(S, text_mask):
    """Checkered scale bars: thin, long components (length >= 6 x height) whose columns alternate between filled
    (ink across the bar) and hollow (ink only on the edges) at least three times."""
    ink = (S.ink & ~text_mask).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    out = []
    hmax = 0.03 * max(S.w, S.h) if not S.mm else 8 * S.mm
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h < 3 or h > hmax or w < 6 * h or w < 0.04 * max(S.w, S.h):
            continue
        sub = lab[y:y + h, x:x + w] == i
        fill = sub[1:-1].mean(0) if h > 2 else sub.mean(0)
        filled = fill >= 0.8
        hollow = fill <= 0.45
        fr = [r for r in _runs(filled) if r[1] - r[0] >= max(2, 0.5 * h)]
        hr = [r for r in _runs(hollow) if r[1] - r[0] >= max(2, 0.5 * h)]
        if len(fr) < 2 or len(hr) < 2 or len(fr) + len(hr) < 5:
            continue
        lens = np.array([b - a for a, b in fr + hr], float)
        if np.mean(np.abs(lens / np.median(lens) - 1) <= 0.35) < 0.6:   # most divisions of a bar are equal
            continue
        bx = (float(x), float(y), float(x + w), float(y + h))
        if _busy(S, bx, 2 * h, text_mask) > 0.08:                  # a bar stands in free space, not inside a plan
            continue
        out.append(bx)
    return out


def _north_arrows(S, text_mask):
    """Cheap candidates: a text "N" (or Nord, North, Norden) with a compact non-text symbol beside it."""
    out = []
    ink = (S.ink & ~text_mask).astype(np.uint8)
    for b, t in zip(S.tw, S.texts):
        s = t.get("text", "").strip().lower().strip(".")
        if s not in ("n", "nord", "north", "norden", "nord.", "n."):
            continue
        h = max(b[3] - b[1], 2)
        x0, y0 = int(max(0, b[0] - 4 * h)), int(max(0, b[1] - 4 * h))
        x1, y1 = int(min(S.w, b[2] + 4 * h)), int(min(S.h, b[3] + 4 * h))
        sub = ink[y0:y1, x0:x1]
        n, lab, st, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        for i in range(1, n):
            x, y, w, hh, a = st[i]
            side = max(w, hh)
            if 1.5 * h <= side <= 8 * h and 0.3 < w / max(hh, 1) < 3.5:
                bx = (min(b[0], x0 + x), min(b[1], y0 + y), max(b[2], x0 + x + w), max(b[3], y0 + y + hh))
                out.append({"box": bx, "reason": f"text {t['text']!r} beside a compact symbol"})
                break
    return out


# --- legend ---

def _legends(S, text_mask, removed):
    out = []
    lines = _lines_of(S.texts)
    for ln in lines:
        ln["box_w"] = tuple(v * S.f for v in ln["box"])
        ln["h_w"] = max(ln["height"] * S.f, 1.0)
    # (a) swatch columns
    cont = (S.content & ~text_mask & ~removed).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cont, connectivity=8)
    th = S.text_h or 0.01 * max(S.w, S.h)
    sw = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (0.6 * th <= h <= 4 * th and 0.6 * th <= w <= 6 * th):
            continue
        sub = lab[y:y + h, x:x + w] == i
        filled = _fill_holes(sub).mean()
        if filled < 0.85:                                # a swatch is a rectangle, filled or outlined
            continue
        sw.append((x, y, x + w, y + h))
    sw.sort(key=lambda b: (round(b[0] / max(th, 1)), b[1]))
    used = set()
    for i, a in enumerate(sw):
        if i in used:
            continue
        col = [i]
        for j in range(i + 1, len(sw)):
            b, last = sw[j], sw[col[-1]]
            wa, wb = last[2] - last[0], b[2] - b[0]
            if abs(b[0] - last[0]) <= 0.5 * max(wa, wb) and 0.6 < wb / max(wa, 1) < 1.6 and 0 < b[1] - last[3] <= 3 * (last[3] - last[1]):
                col.append(j)
        if len(col) < 3:
            continue
        rows = []
        for j in col:
            b = sw[j]
            bh = b[3] - b[1]
            right = [ln for ln in lines if ln["angle"] == 0 and b[2] <= ln["box_w"][0] <= b[2] + 4 * max(bh, b[2] - b[0])
                     and min(ln["box_w"][3], b[3]) - max(ln["box_w"][1], b[1]) > 0.3 * min(bh, ln["h_w"])]
            if right:
                rows.append((b, right[0]))
            elif _char_count(S.ink[int(b[1]):int(b[3]) + 1, int(b[2]):int(b[2] + 6 * bh)], S) >= 2:
                rows.append((b, None))
        if len(rows) < 3 or (S.texts and sum(r[1] is not None for r in rows) < 2):
            continue                                     # with text available, the swatches need read text beside them
        boxes = [r[0] for r in rows] + [r[1]["box_w"] for r in rows if r[1] is not None]
        bx = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        sw_side = float(np.median([b[3] - b[1] for b, _ in rows]))
        if _busy(S, bx, 1.5 * sw_side, text_mask) > 0.1:
            continue                                     # a legend stands apart; columns and seats inside a plan do not
        used.update(col)
        title = [ln for ln in lines if re.search(LEGEND_WORDS, ln["text"].lower()) and _rect_dist(ln["box_w"], bx) < 3 * th]
        for ln in title:
            bx = (min(bx[0], ln["box_w"][0]), min(bx[1], ln["box_w"][1]), max(bx[2], ln["box_w"][2]), max(bx[3], ln["box_w"][3]))
        out.append({"box": bx, "confidence": "medium",
                    "reason": f"{len(rows)} swatches in a column with text beside them" + (" and a legend title" if title else "")})
    # (b) legend keyword with the text lines below it; (c) numbered lists
    for ln in lines:
        low = ln["text"].lower().strip()
        heading = re.match(rf"[\W\d]*(?:{LEGEND_WORDS})", low) is not None    # "Legende: 1 Eingang; 2 Halle ..."
        if not re.search(LEGEND_WORDS, low) or (len(low) > 40 and not heading):
            continue
        if any(_rect_dist(ln["box_w"], o["box"]) < 1 for o in out):
            continue
        blk = _block_below(ln, lines)
        if len(blk) >= 2 or (heading and len(low) > 25):
            boxes = [b["box_w"] for b in blk]
            bx = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
            out.append({"box": bx, "confidence": "medium", "reason": f"legend keyword {ln['text'][:30]!r} with {len(blk) - 1} lines"})
    for blk in _blocks([ln for ln in lines if not any(_rect_dist(ln["box_w"], o["box"]) < 1 for o in out)]):
        numbered = [ln for ln in blk["lines"] if re.match(r"^\s*(\d{1,3}|[A-Za-z])\s*[.):]?\s+\S", ln["text"])]
        if len(blk["lines"]) >= 4 and len(numbered) >= 0.7 * len(blk["lines"]):
            out.append({"box": blk["box_w"], "confidence": "low",
                        "reason": f"numbered list of {len(blk['lines'])} lines (a key to the numbers in the drawing)",
                        "numbered": True})
    return out


def _block_below(head, lines):
    """The heading line plus the lines stacked below it (left edges aligned within 3 heights, gaps under 1.5 heights)."""
    blk = [head]
    cur = head
    for ln in sorted(lines, key=lambda l: l["box_w"][1]):
        if ln is head or ln["box_w"][1] <= cur["box_w"][1]:
            continue
        h = max(cur["h_w"], ln["h_w"])
        if ln["box_w"][1] - cur["box_w"][3] <= 1.5 * h and abs(ln["box_w"][0] - head["box_w"][0]) <= 3 * h:
            blk.append(ln)
            cur = ln
    return blk


def _blocks(lines):
    """Text lines -> blocks of lines stacked closely (vertical gap < 1.2 heights, overlapping in x)."""
    lines = sorted(lines, key=lambda l: (l["box_w"][1], l["box_w"][0]))
    blocks = []
    for ln in lines:
        b = ln["box_w"]
        h = ln["h_w"]
        for blk in blocks:
            bb = blk["box_w"]
            if 0 <= b[1] - bb[3] <= 1.2 * max(h, blk["height"]) or (b[1] < bb[3] and b[3] > bb[1]):
                if min(b[2], bb[2]) - max(b[0], bb[0]) > -0.5 * h and abs(h / blk["height"] - 1) < 0.6:
                    blk["lines"].append(ln)
                    blk["box_w"] = (min(bb[0], b[0]), min(bb[1], b[1]), max(bb[2], b[2]), max(bb[3], b[3]))
                    blk["height"] = max(blk["height"], h)
                    break
        else:
            blocks.append({"lines": [ln], "box_w": b, "height": h})
    return blocks


# --- drawings ---

def _drawing_clusters(S, removed, text_mask):
    """Ink clusters (working px): content minus removed regions and text, specks dropped, closed over S.gap; clusters
    with a side of at least S.min_side and mostly non-text ink. Returns [{"poly", "confidence", "reason"}]."""
    base = S.content & ~removed
    with_text = base.copy()
    base &= ~text_mask
    # specks: dust, stipple, scan noise
    n, lab, st, _ = cv2.connectedComponentsWithStats(base.astype(np.uint8), connectivity=8)
    small = max(2.0, 0.0015 * max(S.w, S.h))
    keep = np.zeros(n, bool)
    keep[1:] = np.maximum(st[1:, 2], st[1:, 3]) >= small
    base = keep[lab]
    if not base.any():
        return []
    r = max(1, int(round(S.gap / 2)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    grown = cv2.dilate(base.astype(np.uint8), ker)
    closed = cv2.erode(grown, ker) > 0
    closed = _fill_holes(closed | base)
    n, lab, st, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if max(w, h) - 2 * r < 0.3 * S.min_side:         # far too small even to be part of a drawing
            continue
        sl = (slice(y, y + h), slice(x, x + w))
        comp = lab[sl] == i
        ink_px = int((base[sl] & comp).sum())
        if ink_px < 20:
            continue
        txt_px = int((with_text[sl] & text_mask[sl] & comp).sum())
        tshare = txt_px / max(txt_px + ink_px, 1)
        if tshare >= S.cfg.layout_text_share:
            continue
        m = closed[sl] & comp
        if not m.any():
            continue
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in cnts:
            if len(c) < 3:
                continue
            eps = max(1.0, 0.25 * r)
            c = cv2.approxPolyDP(c, eps, True)
            if len(c) >= 3:
                p = Polygon(c[:, 0, :].astype(float) + 0.5 + [x, y]).buffer(0)
                if not p.is_empty:
                    polys.append(p)
        if not polys:
            continue
        poly = shapely.union_all(polys)
        poly = max(shapely.get_parts(poly), key=lambda p: p.area) if poly.geom_type != "Polygon" else poly
        poly = Polygon(poly.exterior)
        # text right at the cluster's edge (dimension numbers, labels on the outside) belongs to it; captions sit
        # further away and stay out
        near = [sbox(*b) for b, t in zip(S.tw, S.texts) if t.get("text", "").strip() and
                poly.distance(sbox(*b)) <= 0.35 * S.gap and not poly.contains(sbox(*b))]
        if near:
            u = shapely.union_all([poly] + [b.buffer(0.2 * r) for b in near]).buffer(0.5 * r).buffer(-0.5 * r)
            if u.geom_type != "Polygon":
                u = max(shapely.get_parts(u), key=lambda p: p.area)
            poly = Polygon(u.exterior).simplify(0.25 * r)
        density = ink_px / max(poly.area, 1)
        conf = "medium" if max(w, h) >= 2 * S.min_side and tshare < 0.4 else "low"
        out.append({"poly": poly, "confidence": conf, "ink": ink_px,
                    "reason": f"ink cluster {w / S.f:.0f} x {h / S.f:.0f} px (closed over {2 * r / S.f:.0f} px), "
                              f"{tshare:.0%} text, ink density {density:.2f}"})
    # merge clusters that interleave: one lies half or more inside the other's bounding box (wall pieces of a plan
    # whose door openings are drawn as gaps, a wing inside the courtyard of a U); repeated until nothing changes.
    # Separate drawings side by side do not share their bounding boxes.
    out.sort(key=lambda c: -c["poly"].area)
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(out):
            for j in range(i + 1, len(out)):
                b = out[j]
                small, big = (a, b) if a["poly"].area < b["poly"].area else (b, a)
                if small["poly"].intersection(big["poly"].envelope).area < 0.5 * small["poly"].area:
                    continue
                d = small["poly"].distance(big["poly"]) / 2 + 1
                u = shapely.union_all([big["poly"].buffer(d, join_style="mitre"),
                                       small["poly"].buffer(d, join_style="mitre")]).buffer(-d, join_style="mitre")
                u = max(shapely.get_parts(u), key=lambda p: p.area) if u.geom_type != "Polygon" else u
                big["poly"] = Polygon(u.exterior)
                big["ink"] = big["ink"] + small["ink"]
                big["n"] = big.get("n", 1) + small.get("n", 1)
                out.pop(out.index(small))
                changed = True
                break
            if changed:
                break
    # wall fragments of a plan whose openings are drawn as gaps: open, thin pieces (little area against their convex
    # hull) join their nearest neighbour within cfg.layout_fragment_reach gaps; closed drawings never join each other
    def solidity(c):
        return c["poly"].area / max(c["poly"].convex_hull.area, 1.0)

    while True:
        best = None
        for a in out:
            if solidity(a) >= S.cfg.layout_fragment_solidity:
                continue
            for b in out:
                if b is not a:
                    d = a["poly"].distance(b["poly"])
                    if d <= S.cfg.layout_fragment_reach * S.gap and (best is None or d < best[0]):
                        best = (d, a, b)
        if best is None:
            break
        d, a, b = best
        e = d / 2 + 1
        u = shapely.union_all([a["poly"].buffer(e, join_style="mitre"), b["poly"].buffer(e, join_style="mitre")]).buffer(-e, join_style="mitre")
        u = max(shapely.get_parts(u), key=lambda p: p.area) if u.geom_type != "Polygon" else u
        b["poly"] = Polygon(u.exterior)
        b["ink"] += a["ink"]
        b["n"] = b.get("n", 1) + a.get("n", 1)
        b["fragments"] = b.get("fragments", 0) + 1
        out.remove(a)
    for c in out:
        if c.get("n", 1) > 1:
            c["reason"] += f"; {c['n']} clusters joined (interleaved or open wall fragments)"
        b = c["poly"].bounds
        c["small"] = max(b[2] - b[0], b[3] - b[1]) < S.min_side   # only joined to a drawing nearby, never one itself
    return out


# --- captions ---

def _captions(S, lines, drawings, regions):
    """Best caption per drawing: text lines (and the lines stacked with them) within S.caption of the drawing
    polygon, outside the title block, scored by drawing words, a scale note, size and position."""
    if not drawings:
        return {}
    blocked = [r["_w"] for r in regions if r["class"] in ("title block", "legend")]
    th = S.text_h or 1.0
    # a sheet with one drawing: its title may stand anywhere outside the other regions (distance still costs)
    reach = max(S.w, S.h) if len(drawings) == 1 else S.caption
    cands = []
    for ln in lines:
        b = ln["box_w"]
        c = shapely.Point(_centre(b))
        if any(p.contains(c) for p in blocked):
            continue
        text = ln["text"]
        kind = drawing_kind(text)
        notes = fscale.parse_scale_note(text)
        storey = parse_storey(text)
        if not (kind or notes or storey):
            continue
        lb = _box_poly(b)
        nearest = min(drawings, key=lambda d: d["_w"].distance(lb))   # a caption belongs to its nearest drawing
        for d in [nearest]:
            poly = d["_w"]
            dist = poly.exterior.distance(lb) if not poly.contains(lb) else 0.0
            inside = poly.contains(c)
            pb = poly.bounds
            reach_d = max(reach, 0.2 * min(pb[2] - pb[0], pb[3] - pb[1]), 4 * ln["h_w"])   # larger drawings, larger titles
            if dist > reach_d:
                continue
            if inside:
                # inside the polygon only near its top or bottom edge (titles merged into the ink cluster)
                y0, y1 = poly.bounds[1], poly.bounds[3]
                if min(b[1] - y0, y1 - b[3]) > max(S.caption, 3 * ln["h_w"]):
                    continue
            score = 0.0
            why = []
            if kind:
                score += 2.0 if kind != "floor plan" or re.search(r"grundri|gescho|[ée]tage|rez|piano|floor|plan", text.lower()) else 1.0
                why.append(f"word for a {kind}")
            if notes:
                score += 2.0
                why.append(f"scale 1:{notes[0][0]}")
            if storey:
                score += 1.0
                why.append(f"storey {storey}")
            if ln["h_w"] >= 1.3 * th:
                score += 0.5
                why.append("large lettering")
            dx0, dx1 = poly.bounds[0], poly.bounds[2]
            if min(b[2], dx1) - max(b[0], dx0) > 0:
                score += 0.5
            score -= 0.5 * dist / max(reach_d, 1)
            cands.append((score, d["id"], ln, "; ".join(why)))
    best = {}
    for score, did, ln, why in sorted(cands, key=lambda c: -c[0]):
        if score < 1.5:
            continue
        # a line serves one drawing: the best-scoring pairing wins
        if any(ln is b2["lines"][0] for b2 in best.values()):
            continue
        if did in best:
            continue
        group = [ln]
        for other in lines:                              # lines stacked with it (title + scale on the next line)
            if other is ln:
                continue
            ob = other["box_w"]
            if abs(ob[1] - ln["box_w"][3]) <= 1.2 * max(ln["h_w"], other["h_w"]) or abs(ln["box_w"][1] - ob[3]) <= 1.2 * max(ln["h_w"], other["h_w"]):
                if min(ob[2], ln["box_w"][2]) - max(ob[0], ln["box_w"][0]) > 0 and \
                        (fscale.parse_scale_note(other["text"]) or drawing_kind(other["text"]) or parse_storey(other["text"])):
                    group.append(other)
        group.sort(key=lambda l: l["box_w"][1])
        bx = (min(l["box_w"][0] for l in group), min(l["box_w"][1] for l in group), max(l["box_w"][2] for l in group),
              max(l["box_w"][3] for l in group))
        best[did] = {"lines": group, "text": " / ".join(l["text"] for l in group), "box_w": bx,
                     "score": round(score, 2), "why": why}
    return best


# ---------------------------------------------------------------------------------------------------------------------
# Masks and overlays

def drawing_mask(drawing, shape):
    """Boolean raster mask (H, W) of a drawing's mask polygon(s) at full resolution."""
    m = np.zeros(shape[:2], np.uint8)
    parts = drawing["mask"].get("parts") or [drawing["mask"]["polygon_px"]]
    for p in parts:
        if len(p) >= 3:
            cv2.fillPoly(m, [np.round(np.asarray(p) * 4).astype(np.int32)], 1, shift=2)
    return m > 0


def overlay(img, result, max_side=1600, labels=True):
    """Regions coloured (translucent fill and outline), drawings outlined thick with their id, kind, storey and scale
    note, captions boxed; returned as an RGB image reduced to max_side."""
    H, W = img.shape[:2]
    f = min(1.0, max_side / max(H, W))
    base = cv2.resize(img, (max(1, int(W * f)), max(1, int(H * f))), interpolation=cv2.INTER_AREA) if f < 1 else img.copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    out = base.copy()
    fill = base.copy()
    for r in result["regions"]:
        if r["class"] == "drawing" or len(r["polygon_px"]) < 3:
            continue
        pts = np.round(np.asarray(r["polygon_px"]) * f).astype(np.int32)
        col = COLOURS.get(r["class"], (90, 90, 90))
        if r["class"] != "frame":
            cv2.fillPoly(fill, [pts], col)
    out = cv2.addWeighted(fill, 0.3, out, 0.7, 0)
    for r in result["regions"]:
        if r["class"] == "drawing" or len(r["polygon_px"]) < 3:
            continue
        pts = np.round(np.asarray(r["polygon_px"]) * f).astype(np.int32)
        col = COLOURS.get(r["class"], (90, 90, 90))
        cv2.polylines(out, [pts], True, col, 2)
        if labels and r["class"] not in ("frame",):
            x, y = pts[:, 0].min(), pts[:, 1].min()
            cv2.putText(out, r["class"], (int(x) + 2, int(y) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    for d in result["drawings"]:
        if len(d["polygon_px"]) < 3:
            continue
        pts = np.round(np.asarray(d["polygon_px"]) * f).astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 170, 0), 3)
        mp = d["mask"].get("parts") or [d["mask"]["polygon_px"]]
        for p in mp:
            if len(p) >= 3:
                cv2.polylines(out, [np.round(np.asarray(p) * f).astype(np.int32)], True, (0, 220, 120), 1)
        if labels:
            sc = d["scale_note"]["scale"]
            lab = f"{d['id']} {d['kind']}" + (f" {d['storey']}" if d["storey"] else "") + (f" 1:{sc}" if sc else "")
            x, y = pts[:, 0].min(), pts[:, 1].min()
            cv2.putText(out, lab, (int(x) + 3, int(y) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(out, lab, (int(x) + 3, int(y) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 0), 1, cv2.LINE_AA)
    return out
