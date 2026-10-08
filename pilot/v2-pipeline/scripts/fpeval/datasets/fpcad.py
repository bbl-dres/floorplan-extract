"""FloorPlanCAD blocks (fpeval.datasets): SVG primitives, the release's semantic ids and their grouping into the
paper's categories, the per-block scale check, rendering at the working resolution, sampling along primitives and
the filled ground-truth masks. The protocol (votes, metrics, report) is fpeval.fpcad; fpcad_eval.py runs it.

FloorPlanCAD is CC BY-NC 4.0 (annotations; drawings stay with their owners): evaluation only, outputs stay in the
gitignored data folder. Never train or select a deployed model on it.
"""
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from common import REPO
from fpx.config import DEFAULT
from fpx.model import BG, CLASSES, COLUMN, DOOR, STAIRS, WALL, WINDOW

FPCAD = REPO / "data/benchmark/floorplancad"
PX_PER_M = DEFAULT.px_per_m
UNITS_PER_M = 10.0                      # SVG units per metre in the release
PPU = PX_PER_M / UNITS_PER_M            # 5 px per SVG unit at the working resolution
LINE_PX = 1.25                          # pen width in working pixels: 0.25 mm at 1:100
SAMPLE_PX = 0.5                         # sample spacing along a primitive (px); start, middle and end always sampled
MAX_SAMPLES = 4000                      # per primitive
MAX_SIDE_PX = 6000                      # rescaled blocks larger than this are skipped
K = len(CLASSES)

INK_LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"
NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
CMD = re.compile(r"([MmLlHhVvAaCcQqSsTtZz])([^MmLlHhVvAaCcQqSsTtZz]*)")

# release semantic ids (SOURCE.md); 0 = unlabelled, 36 = unlabelled primitive on a CAD column layer
ID_NAMES = {0: "unlabelled", 1: "wall", 2: "curtain wall", 3: "single door", 4: "double door", 5: "sliding door",
            6: "folding door", 7: "revolving door", 8: "rolling door", 9: "window", 10: "bay window",
            11: "blind window", 12: "opening symbol", **{i: f"id {i} (furniture, appliance, sanitary)" for i in range(13, 30)},
            30: "stairs", 31: "elevator", 32: "escalator", 33: "railing", 34: "row chairs", 35: "parking spot",
            36: "unlabelled, column layer"}
COLUMN_LAYER = 36
N_GT = 37
CATS = ["Door", "Window", "Stair", "Wall"]                      # the paper's Table 3 order, restricted to our classes
PRED_CAT = {DOOR: "Door", WINDOW: "Window", STAIRS: "Stair", WALL: "Wall"}     # column, background: other
_DOOR, _WIN = list(range(3, 9)), list(range(9, 13))
GROUPINGS = {                                                   # "ignore": release ids left out of every count
    "paper": {"Door": _DOOR, "Window": _WIN, "Stair": [30], "Wall": [1], "ignore": [2]},
    "curtain-as-other": {"Door": _DOOR, "Window": _WIN, "Stair": [30], "Wall": [1], "ignore": []},
    "wall+curtain": {"Door": _DOOR, "Window": _WIN, "Stair": [30], "Wall": [1, 2], "ignore": []},
    "curtain-as-window": {"Door": _DOOR, "Window": [2] + _WIN, "Stair": [30], "Wall": [1], "ignore": []},
    # diagnostic only: also leave out unlabelled primitives on CAD column layers (columns are unlabelled in the release
    # and our segmenter calls most of them wall)
    "columns-ignored": {"Door": _DOOR, "Window": _WIN, "Stair": [30], "Wall": [1], "ignore": [2, COLUMN_LAYER]},
}
KINDS = ["line", "arc", "circle", "ellipse", "curve"]
KIND_GT = [1, 2] + _DOOR + _WIN + [30]                          # labelled ids whose recall is split by primitive kind
COLUMN_LAYER_RE = re.compile(r"COLU|COLS|(^|[^A-Z])COL([^A-Z]|$)|柱")
NOT_COLUMN_RE = re.compile(r"柱网|轴|AXIS|DIM|尺寸|标注|TEXT|文字|PUB_")

# Scale cues (SVG units at 10 units/m), calibrated on the whole test split. Double-door leaves (fire doors of 2 x 0.45 m)
# and wall modes (plaster and finish lines 2-4 cm off the wall face) gave false alarms on blocks checked by eye at
# 10 units/m, so only single doors decide and the wall cue can only veto.
DOOR_ARC_IDS = (3,)                     # single doors: swing radius = leaf width
REF_LEAF = 8.73                         # median single-door leaf of the 1,087 blocks with 200 mm walls (P5-P95 5.7-10.7)
MIN_ARCS = 2
DOOR_LOW, DOOR_HIGH = 0.45, 2.2         # leaf / REF_LEAF outside: off scale (single doors under 0.39 m or over 1.9 m)
DOOR_SANE = (0.1, 4.0)                  # outside: stray arcs labelled as doors, cue ignored
REF_WALL = 2.0                          # 200 mm, the mode in 85 % of the drawings
WALL_VETO = (0.6, 1.6)                  # a low door cue needs wall mode / REF_WALL < 0.6, a high one > 1.6
WALL_SUSPECT = 0.35                     # wall mode under 7 cm and no door cue: flagged "suspect", not rescaled
MIN_WALL_LEN = 10.0                     # units of overlapping parallel wall lines needed for the wall cue


# ----- SVG parsing -----

def _affine(transform):
    """2 x 3 matrix of an SVG transform list (rotate, translate, scale, matrix)."""
    m = np.eye(3)
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", transform or ""):
        v = [float(x) for x in NUM.findall(args)]
        if name == "rotate":
            a = math.radians(v[0])
            r = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            if len(v) == 3:
                t = np.array([[1, 0, v[1]], [0, 1, v[2]], [0, 0, 1]])
                r = t @ r @ np.linalg.inv(t)
            m = m @ r
        elif name == "translate":
            m = m @ np.array([[1, 0, v[0]], [0, 1, v[1] if len(v) > 1 else 0], [0, 0, 1]])
        elif name == "scale":
            m = m @ np.diag([v[0], v[1] if len(v) > 1 else v[0], 1])
        elif name == "matrix":
            m = m @ np.array([[v[0], v[2], v[4]], [v[1], v[3], v[5]], [0, 0, 1]])
    return m[:2]


def arc_points(p0, rx, ry, phi, large, sweep, p1):
    """SVG elliptical arc (endpoint parameterisation, SVG 1.1 F.6.5) as a polyline with steps of at most 0.5 units.
    Returns (points, radius) with the radii scaled up if they are too small to reach the end point."""
    (x1, y1), (x2, y2) = p0, p1
    rx, ry = abs(rx), abs(ry)
    if rx == 0 or ry == 0 or (x1 == x2 and y1 == y2):
        return np.array([p0, p1], float), 0.0
    c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    xp, yp = c * dx + s * dy, -s * dx + c * dy
    lam = xp ** 2 / rx ** 2 + yp ** 2 / ry ** 2
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num = rx ** 2 * ry ** 2 - rx ** 2 * yp ** 2 - ry ** 2 * xp ** 2
    den = rx ** 2 * yp ** 2 + ry ** 2 * xp ** 2
    co = math.sqrt(max(0.0, num / den)) if den > 0 else 0.0
    if bool(large) == bool(sweep):
        co = -co
    cxp, cyp = co * rx * yp / ry, -co * ry * xp / rx
    cx, cy = c * cxp - s * cyp + (x1 + x2) / 2, s * cxp + c * cyp + (y1 + y2) / 2
    ang = lambda ux, uy, vx, vy: math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
    t1 = ang(1, 0, (xp - cxp) / rx, (yp - cyp) / ry)
    dt = ang((xp - cxp) / rx, (yp - cyp) / ry, (-xp - cxp) / rx, (-yp - cyp) / ry)
    if not sweep and dt > 0:
        dt -= 2 * math.pi
    elif sweep and dt < 0:
        dt += 2 * math.pi
    n = int(min(max(4, math.ceil(abs(dt) * max(rx, ry) * 2)), 4000))
    t = t1 + dt * np.linspace(0, 1, n + 1)
    pts = np.c_[cx + rx * c * np.cos(t) - ry * s * np.sin(t), cy + rx * s * np.cos(t) + ry * c * np.sin(t)]
    pts[0], pts[-1] = p0, p1
    return pts, max(rx, ry)


def path_points(d):
    """Polyline of an SVG path (M, L, H, V, A, Z; curves by their control polygon, absent from the release).
    One path element is one primitive, as in the dataset. Returns (points, kind, arc radius)."""
    pts, cur, start, kind, radius = [], np.zeros(2), np.zeros(2), "line", 0.0
    for cmd, args in CMD.findall(d):
        v = [float(x) for x in NUM.findall(args)]
        rel = cmd.islower()
        C = cmd.upper()
        if C == "Z":
            pts.append(start.copy())
            cur = start.copy()
            continue
        step = {"M": 2, "L": 2, "H": 1, "V": 1, "A": 7, "C": 6, "Q": 4, "S": 4, "T": 2}[C]
        for i in range(0, len(v) - step + 1, step):
            a = v[i:i + step]
            if C in ("M", "L", "T"):                        # pairs after the first of an M are implicit L
                p = np.array(a) + (cur if rel else 0)
                if C == "M" and i == 0:
                    start = p.copy()                        # a second subpath would be joined (none in the release)
                pts.append(p.copy())
                cur = p
            elif C == "H":
                cur = np.array([a[0] + (cur[0] if rel else 0), cur[1]])
                pts.append(cur.copy())
            elif C == "V":
                cur = np.array([cur[0], a[0] + (cur[1] if rel else 0)])
                pts.append(cur.copy())
            elif C == "A":
                p = np.array(a[5:7]) + (cur if rel else 0)
                arc, r = arc_points(cur.copy(), a[0], a[1], a[2], a[3], a[4], p)
                pts.extend(arc[1:] if pts else arc)
                kind, radius = "arc", max(radius, r)
                cur = p
            else:                                           # C, Q, S: control polygon
                q = np.array(a).reshape(-1, 2) + (cur if rel else 0)
                pts.extend(q)
                kind = "curve"
                cur = q[-1]
    if not pts:
        return None, kind, radius
    return np.asarray(pts, float), kind, radius


def ellipse_points(cx, cy, rx, ry, transform=None):
    n = int(min(max(16, math.ceil(2 * math.pi * max(rx, ry) * 2)), 4000))
    t = np.linspace(0, 2 * math.pi, n + 1)
    pts = np.c_[cx + rx * np.cos(t), cy + ry * np.sin(t)]
    if transform:
        m = _affine(transform)
        pts = pts @ m[:, :2].T + m[:, 2]
    return pts


def is_column_layer(name):
    tail = (name or "").split("$")[-1].upper()
    return bool(COLUMN_LAYER_RE.search(tail)) and not NOT_COLUMN_RE.search(tail)


def parse_svg(data):
    """Primitives of one block (bytes or path): list of dicts with kind, sid (release semantic id, 0 = none),
    gt (sid, or COLUMN_LAYER for unlabelled primitives on a column layer), iid, layer, pts (polyline, SVG units),
    length (SVG units) and radius (arcs)."""
    if isinstance(data, (str, Path)):
        data = Path(data).read_bytes()
    root = ET.fromstring(data)
    prims = []

    def walk(node, layer):
        for e in node:
            tag = e.tag.rsplit("}", 1)[-1]
            if tag == "g":
                walk(e, e.get(INK_LABEL) or e.get("id") or layer)
                continue
            radius = 0.0
            if tag == "path":
                pts, kind, radius = path_points(e.get("d", ""))
            elif tag == "circle":
                r = float(e.get("r", 0))
                pts, kind = ellipse_points(float(e.get("cx", 0)), float(e.get("cy", 0)), r, r, e.get("transform")), "circle"
            elif tag == "ellipse":
                pts, kind = ellipse_points(float(e.get("cx", 0)), float(e.get("cy", 0)), float(e.get("rx", 0)),
                                           float(e.get("ry", 0)), e.get("transform")), "ellipse"
            else:
                continue
            if pts is None:
                continue
            if tag == "path" and e.get("transform"):
                m = _affine(e.get("transform"))
                pts = pts @ m[:, :2].T + m[:, 2]
            sid = int(float(e.get("semantic-id") or 0))
            gt = sid if sid else (COLUMN_LAYER if is_column_layer(layer) else 0)
            iid = int(float(e.get("instance-id") or -1))
            length = float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0
            prims.append({"kind": kind, "sid": sid, "gt": gt, "iid": iid, "layer": layer, "pts": pts,
                          "length": length, "radius": radius})

    walk(root, "")
    return prims



# ----- scale check -----

def wall_mode(prims):
    """Mode of the gap between labelled parallel wall lines (SVG units, bins of 0.05) and the overlap length behind it."""
    segs = np.array([np.r_[p["pts"][0], p["pts"][-1]] for p in prims if p["sid"] == 1 and p["kind"] == "line"
                     and len(p["pts"]) == 2]).reshape(-1, 4)
    if len(segs) < 2:
        return None, 0.0
    a, b = segs[:, :2], segs[:, 2:]
    L = np.hypot(*(b - a).T)
    keep = np.argsort(-L)[:1500]
    keep = keep[L[keep] > 0.5]
    a, b, L = a[keep], b[keep], L[keep]
    if len(a) < 2:
        return None, 0.0
    u = (b - a) / L[:, None]
    nrm = np.c_[-u[:, 1], u[:, 0]]
    par = np.abs(u @ u.T) > math.cos(math.radians(2))
    mid = (a + b) / 2
    dist = np.abs(np.einsum("ijk,ik->ij", mid[None] - a[:, None], nrm))
    ta = np.einsum("ijk,ik->ij", a[None] - a[:, None], u)
    tb = np.einsum("ijk,ik->ij", b[None] - a[:, None], u)
    ov = np.clip(np.minimum(np.maximum(ta, tb), L[:, None]) - np.maximum(np.minimum(ta, tb), 0), 0, None)
    ok = par & (dist > 0.1) & (dist < 10) & (ov > 0.5 * np.minimum(L[:, None], L[None, :]))
    np.fill_diagonal(ok, False)
    dd = np.where(ok, dist, np.inf)
    j = dd.argmin(1)
    i = np.where(np.isfinite(dd[np.arange(len(a)), j]))[0]
    if not len(i):
        return None, 0.0
    hist = np.bincount(np.minimum((dd[i, j[i]] / 0.05).astype(int), 199), weights=ov[i, j[i]], minlength=200)
    return (int(hist.argmax()) + 0.5) * 0.05, float(hist.sum())


def scale_check(prims):
    """Scale factor f of a block (1 = 10 units/m; 0.2 = a plan drawn 5x too small) and how it was decided.
    status: ok, off (rescale by f), conflict (doors off, walls normal: kept), suspect (walls only, under 7 cm: kept),
    unverified (no cue: 10 units/m assumed)."""
    radii = [p["radius"] for p in prims if p["sid"] in DOOR_ARC_IDS and p["kind"] == "arc" and p["radius"] > 0]
    out = {"door_arcs": len(radii), "door_leaf": round(float(np.median(radii)), 3) if radii else None}
    wm, wlen = wall_mode(prims)
    fw = wm / REF_WALL if wm is not None and wlen >= MIN_WALL_LEN else None
    out.update({"wall_mode": round(wm, 3) if wm is not None else None, "wall_pair_len": round(wlen, 1)})
    fd = float(np.median(radii)) / REF_LEAF if len(radii) >= MIN_ARCS else None
    if fd is not None and not DOOR_SANE[0] <= fd <= DOOR_SANE[1]:
        out["door_cue_ignored"] = round(fd, 4)
        fd = None
    if fd is not None:
        if DOOR_LOW <= fd <= DOOR_HIGH:
            out.update({"cue": "doors", "factor": 1.0, "status": "ok"})
        elif fw is not None and (fw >= WALL_VETO[0] if fd < DOOR_LOW else fw <= WALL_VETO[1]):  # walls not off alike
            out.update({"cue": "doors", "factor": 1.0, "status": "conflict", "door_factor": round(fd, 4)})
        else:
            out.update({"cue": "doors", "factor": round(fd, 4), "status": "off"})
    elif fw is not None:
        out.update({"cue": "walls", "factor": 1.0, "status": "suspect" if fw < WALL_SUSPECT else "ok"})
    else:
        out.update({"cue": "none", "factor": 1.0, "status": "unverified"})
    return out



# ----- rendering, ground truth masks, sampling -----

def render(data, ppu=PPU, line_px=LINE_PX):
    """Black-on-white RGB raster of an SVG block at ppu px per unit with every stroke line_px pixels wide."""
    import pymupdf
    s = data.decode("utf-8") if isinstance(data, bytes) else data
    s = re.sub(r'stroke="[^"]*"', 'stroke="#000000"', s)
    s = re.sub(r'stroke-width="[^"]*"', f'stroke-width="{line_px / ppu:.6g}"', s)
    s = re.sub(r'fill="(?!none)[^"]*"', 'fill="#000000"', s)
    s = re.sub(r"background-color:\s*#?\w+;?", "", s)
    doc = pymupdf.open(stream=s.encode("utf-8"), filetype="svg")
    pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(ppu, ppu), colorspace=pymupdf.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].copy()
    doc.close()
    return img


def sample_pixels(prims, ppu, shape):
    """(primitive index, row, col) of the samples of every primitive: n + 1 points at equal arc-length steps of at
    most SAMPLE_PX along the path, n even and at least 2 (start, middle and end included). SVG point (x, y) lies in
    pixel (floor(y ppu), floor(x ppu)) as rasterised by MuPDF (pixel i covers [i, i + 1) in device space)."""
    idx, rows, cols = [], [], []
    H, W = shape
    for k, p in enumerate(prims):
        pts = p["pts"]
        seg = np.hypot(*np.diff(pts, axis=0).T) if len(pts) > 1 else np.zeros(0)
        L = seg.sum()
        if L <= 0:
            xy = pts[:1]
        else:
            keep = np.r_[True, seg > 0]
            pts, cum = pts[keep], np.r_[0, np.cumsum(seg[seg > 0])]
            n = int(min(max(2, math.ceil(L * ppu / SAMPLE_PX)), MAX_SAMPLES))
            n += n % 2
            t = np.linspace(0, L, n + 1)
            xy = np.c_[np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])]
        c = np.clip(np.floor(xy[:, 0] * ppu).astype(int), 0, W - 1)
        r = np.clip(np.floor(xy[:, 1] * ppu).astype(int), 0, H - 1)
        idx.append(np.full(len(xy), k))
        rows.append(r)
        cols.append(c)
    if not idx:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(idx), np.concatenate(rows), np.concatenate(cols)


def _poly_px(pts, ppu):
    return np.round((pts * ppu - 0.5) * 16).astype(np.int32)        # pixel-centre coordinates, 4 fractional bits


def _disk(d_px):
    r = max(1, int(round(d_px / 2)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


IGNORE_PX = 255


def gt_mask(prims, shape, ppu, grouping="paper"):
    """Filled six-class label mask (like the earlier sample's _label.png) for the given grouping; ignored ids
    (curtain wall in "paper") are closed like windows and set to IGNORE_PX where nothing else is painted."""
    g = GROUPINGS[grouping]
    ppm = PX_PER_M                                          # rescaled blocks are at 50 px/m too

    def lines(ids):
        t = np.zeros(shape, np.uint8)
        polys = [_poly_px(p["pts"], ppu) for p in prims if p["gt"] in ids and len(p["pts"]) > 1]
        if polys:
            cv2.polylines(t, polys, False, 1, 1, cv2.LINE_8, shift=4)
        return t

    def hulls(ids):
        t = np.zeros(shape, np.uint8)
        groups = {}
        for k, p in enumerate(prims):
            if p["gt"] in ids:
                groups.setdefault((p["sid"], p["iid"]) if p["iid"] >= 0 else ("p", k), []).append(p["pts"])
        for pts in groups.values():
            q = _poly_px(np.concatenate(pts), ppu)
            cv2.fillConvexPoly(t, cv2.convexHull(q), 1, cv2.LINE_8, shift=4)
            cv2.polylines(t, [q], False, 1, 1, cv2.LINE_8, shift=4)
        return t > 0

    m = np.zeros(shape, np.uint8)
    wall = cv2.morphologyEx(lines(g["Wall"]), cv2.MORPH_CLOSE, _disk(0.45 * ppm)) > 0
    m[wall] = WALL
    col = lines([COLUMN_LAYER])
    if col.any():
        n, comp, stats, _ = cv2.connectedComponentsWithStats((col == 0).astype(np.uint8), connectivity=4)
        small = stats[:, cv2.CC_STAT_AREA] <= 2.5 * ppm ** 2
        touches = np.zeros(n, bool)
        touches[np.unique(np.r_[comp[0], comp[-1], comp[:, 0], comp[:, -1]])] = True
        cells = small[comp] & ~touches[comp] & (col == 0)
        blob = (col > 0) | cells
        n2, comp2, stats2, _ = cv2.connectedComponentsWithStats(blob.astype(np.uint8), connectivity=8)
        has_cell = np.zeros(n2, bool)
        has_cell[np.unique(comp2[cells])] = True
        area = stats2[:, cv2.CC_STAT_AREA]
        ok = has_cell & (area >= 0.02 * ppm ** 2) & (area <= 2.5 * ppm ** 2)
        ok[0] = False
        m[ok[comp2]] = COLUMN
    if g["ignore"]:
        ign = cv2.morphologyEx(lines(g["ignore"]), cv2.MORPH_CLOSE, _disk(0.30 * ppm)) > 0
        m[ign & (m == BG)] = IGNORE_PX
    win = cv2.morphologyEx(lines(g["Window"]), cv2.MORPH_CLOSE, _disk(0.30 * ppm)) > 0
    m[win] = WINDOW
    m[hulls(g["Stair"])] = STAIRS
    m[hulls(g["Door"])] = DOOR
    return m


def scale_decision(prims, policy):
    """(ppu, scale record); ppu None = skip."""
    sc = scale_check(prims)
    f = sc["factor"]
    if sc["status"] == "off":
        if policy == "skip":
            sc["action"] = "skipped"
            return None, sc
        if policy == "rescale":
            if 100 * PPU / f > MAX_SIDE_PX or f > 4:
                sc["action"] = "skipped (size)"
                return None, sc
            sc["action"] = "rescaled"
            return PPU / f, sc
    sc["action"] = "none"
    return PPU, sc
