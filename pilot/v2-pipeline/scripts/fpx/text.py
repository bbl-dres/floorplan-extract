"""Stage 2, text layer: native PDF text, OCR on rasters, and a first role for each text item.

OCR (ocr()) runs the detector per tile of the upright image, plus a detection-only pass on the image rotated by 90
degrees and reduced (cfg.ocr_rotated_scale): the detector finds vertical dimension strings as horizontal text there,
while on the upright image it misses a third of them. Every box is then recognised once, in one batched call; a box
taller than wide holds vertical text and is recognised rotated clockwise and counter-clockwise, the better score
wins (vertical numbers are misread without the rotation). The earlier whole-image OCR at 90 degrees (every tile
detected, classified and recognised twice) is gone. The engine's own confidence filter is not applied;
cfg.ocr_min_conf is the one threshold."""
import re

import cv2
import numpy as np
import shapely

from .config import DEFAULT

FIXTURE_WORDS = {"KACHELOFEN", "KACHEL", "OFEN", "CHEMINEE", "KAMIN", "SPIEGEL"}
VOID_NAMES = {"LUFTRAUM", "LUFT", "VIDE", "VIDESURSALLE", "VUOTO"}    # a stamp with this name marks a void, not a room
AOID_RE = re.compile(r"\b\d{4}\.[A-Za-z0-9]{1,4}\.\d{2}\.\d{3}\b")      # WWWW.GG.EE.RRR as plan-check accepts it (fpx.conformance)
TILE_MIN_INK = 300                                     # px: a tile with less ink than one small word holds no text
TALL = 1.5                                             # a detection box at least this much taller than wide is vertical text
TALL_SURE = 0.95                                       # a vertical word of one or two letters needs this confidence (see recognise)


def merge_chars(chars):
    """Native PDF characters -> words/lines: same direction and baseline, gaps below 0.6 x font size."""
    items = []
    for d in {c[6] for c in chars}:
        cs = [c for c in chars if c[6] == d]
        horiz = abs(d[0]) > 0.5
        key = (lambda c: (c[2] + c[4]) / 2) if horiz else (lambda c: (c[1] + c[3]) / 2)
        pos = (lambda c: c[1]) if horiz else (lambda c: -c[4])
        cs.sort(key=lambda c: (round(key(c) / max(c[5] * 0.5, 1)), pos(c)))
        cur = []
        for c in cs:
            if cur and (abs(key(c) - key(cur[-1])) > 0.4 * c[5] or pos(c) - (cur[-1][3] if horiz else -cur[-1][2]) > 0.6 * c[5]):
                items.append(cur)
                cur = []
            cur.append(c)
        if cur:
            items.append(cur)
    out = []
    for it in items:
        xs0, ys0, xs1, ys1 = zip(*[(c[1], c[2], c[3], c[4]) for c in it])
        out.append({"text": "".join(c[0] for c in it), "box": (min(xs0), min(ys0), max(xs1), max(ys1)),
                    "conf": 1.0, "source": "pdf", "angle": 0 if abs(it[0][6][0]) > 0.5 else 90, "height": it[0][5]})
    return out


def dedupe(found):
    """Drop boxes that another, better-scored box covers by half or more (tile overlaps, the two rotations)."""
    if not found:
        return []
    order = sorted(range(len(found)), key=lambda i: -found[i]["conf"] * len(found[i]["text"]))
    boxes = [shapely.box(*f["box"]) for f in found]
    tree = shapely.STRtree(boxes)
    kept, keep = set(), []
    for i in order:
        b = boxes[i]
        clash = False
        for j in tree.query(b, predicate="intersects"):
            if int(j) in kept and b.intersection(boxes[j]).area >= 0.5 * min(b.area, boxes[j].area):
                clash = True
                break
        if not clash:
            kept.add(i)
            keep.append(found[i])
    return keep


# ---------- OCR: detection per tile, batched recognition, tall boxes rotated ----------

def _starts(n, tile, overlap):
    """Tile start positions along one axis: a step of tile - overlap, the last tile aligned with the end so that no
    tile is a narrow strip (the detector scales a strip up to 736 px on its short side, at a cost)."""
    if n <= tile:
        return [0]
    s = list(range(0, n - tile, tile - overlap))
    return s + [n - tile]


def _tiles(img, tile, overlap):
    H, W = img.shape[:2]
    for y in _starts(H, tile, overlap):
        for x in _starts(W, tile, overlap):
            yield x, y, img[y:y + tile, x:x + tile]


def _aabb(q):
    """Axis-aligned quad (clockwise from the top-left) of any four points."""
    x0, y0 = q.min(0)
    x1, y1 = q.max(0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)


def _detect_tiles(img, engine, tile, overlap, min_ink=TILE_MIN_INK):
    quads = []
    for x, y, t in _tiles(img, tile, overlap):
        if (t < 128).sum() < min_ink:
            continue
        r = engine.text_det(cv2.cvtColor(t, cv2.COLOR_GRAY2BGR))
        if r.boxes is None:
            continue
        for q in r.boxes:
            quads.append(np.asarray(q, np.float64) + [x, y])
    return quads


def detect(img, engine, tile=1280, overlap=200, rotated=0.0):
    """Text detection quads (n x 4 x 2, image pixels) of a grey image, tile by tile; tiles without ink are skipped.
    rotated > 0: a second detection pass on the image rotated by 90 degrees and reduced by this factor, keeping the
    boxes that are wide there (vertical text of the original; its horizontal text the upright pass already has)."""
    quads = _detect_tiles(img, engine, tile, overlap)
    if rotated:
        H = img.shape[0]
        rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if rotated != 1:
            rot = cv2.resize(rot, None, fx=rotated, fy=rotated, interpolation=cv2.INTER_AREA)
        for q in _detect_tiles(rot, engine, tile, overlap, TILE_MIN_INK * rotated * rotated):
            q = q / rotated
            w, h = quad_size(q)
            if h >= TALL * w:
                continue
            quads.append(_aabb(np.c_[q[:, 1], H - 1 - q[:, 0]]))      # rotated (u, v) -> original (x, y)
    return quads


def dedupe_quads(quads):
    """Indices of the detection quads to keep: a quad that a larger one covers by half or more of the smaller area is
    the same word seen again on an overlapping tile (often cut at the tile border)."""
    if not quads:
        return []
    boxes = [shapely.box(q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()) for q in quads]
    tree = shapely.STRtree(boxes)
    kept, keep = set(), []
    for i in sorted(range(len(quads)), key=lambda i: -boxes[i].area):
        b = boxes[i]
        if any(int(j) in kept and b.intersection(boxes[j]).area >= 0.5 * min(b.area, boxes[j].area)
               for j in tree.query(b, predicate="intersects")):
            continue
        kept.add(i)
        keep.append(i)
    return keep


def quad_size(q):
    """Width and height of a detection quad (points clockwise from the top-left)."""
    w = max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3]))
    h = max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2]))
    return float(w), float(h)


def crop_quad(img, q):
    """Perspective crop of a quad (as PP-OCR crops its boxes), upright as detected: tall crops are not rotated here."""
    w, h = quad_size(q)
    w, h = max(int(round(w)), 1), max(int(round(h)), 1)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(np.asarray(q, np.float32), dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)


def recognise(img, quads, engine):
    """Batched recognition of the quads' crops (grey image). A tall quad is recognised rotated clockwise and
    counter-clockwise (vertical dimensions read upward or downward) and keeps the better score. Returns items with
    the text, the axis-aligned box, the recogniser's confidence, the angle (0 or 90) and the character height."""
    if not quads:
        return []
    from rapidocr.ch_ppocr_rec import TextRecInput
    crops, jobs = [], []
    for i, q in enumerate(quads):
        w, h = quad_size(q)
        c = crop_quad(img, q)
        if h >= TALL * w:
            crops.append(cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE))             # text reading upward (the usual case)
            crops.append(cv2.rotate(c, cv2.ROTATE_90_COUNTERCLOCKWISE))
            jobs += [(i, 90), (i, 90)]
        else:
            crops.append(c)
            jobs.append((i, 0))
    r = engine.text_rec(TextRecInput(img=[cv2.cvtColor(c, cv2.COLOR_GRAY2BGR) for c in crops]))
    best = {}
    for (i, angle), txt, sc in zip(jobs, r.txts or (), r.scores or ()):
        if txt and txt.strip() and (i not in best or sc > best[i][1]):
            best[i] = (txt, float(sc), angle)
    # a short vertical word read with less than TALL_SURE is a fixture symbol more often than text (a toilet bowl
    # reads "LO" at 0.89): vertical text that counts is a number, three characters or more, or read with certainty
    best = {i: v for i, v in best.items() if v[2] == 0 or v[1] >= TALL_SURE or len(v[0].strip()) >= 3
            or re.fullmatch(r"[\d.,' ]+", v[0].strip())}
    out = []
    for i, (txt, sc, angle) in best.items():
        q = quads[i]
        x0, y0 = q.min(0)
        x1, y1 = q.max(0)
        out.append({"text": txt.replace("–", "-").replace("—", "-").replace(" -", "-"),
                    "box": (float(x0), float(y0), float(x1), float(y1)), "conf": sc, "source": "ocr",
                    "angle": angle, "height": float(min(x1 - x0, y1 - y0))})
    return out


def _ocr_whole(img, engine, tile, overlap, min_conf):
    """OCR through the engine's one-call interface (an engine without separate detector and recogniser): the upright
    image and the image rotated by 90 degrees, tile by tile."""
    found = []
    for rot in (0, 90):
        im = img if rot == 0 else cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        for x, y, t in _tiles(im, tile, overlap):
            if (t < 128).sum() < TILE_MIN_INK:
                continue
            bgr = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            try:
                r = engine(bgr, text_score=min_conf) if min_conf is not None else engine(bgr)
            except TypeError:                              # an engine without the text_score keyword
                r = engine(bgr)
            if r.boxes is None:
                continue
            for box, txt, sc in zip(r.boxes, r.txts, r.scores):
                b = np.asarray(box) + [x, y]
                if rot == 90:                              # rotated (u, v) -> original (x, y) = (v, H_orig - 1 - u)
                    b = np.c_[b[:, 1], img.shape[0] - 1 - b[:, 0]]
                x0, y0 = b.min(0)
                x1, y1 = b.max(0)
                found.append({"text": txt.replace("–", "-").replace("—", "-").replace(" -", "-"),
                              "box": (float(x0), float(y0), float(x1), float(y1)), "conf": float(sc), "source": "ocr",
                              "angle": 0 if x1 - x0 >= y1 - y0 else 90, "height": float(min(x1 - x0, y1 - y0))})
    return found


def ocr(img, engine, tile=1280, overlap=200, min_conf=None, rotated=DEFAULT.ocr_rotated_scale):
    """RapidOCR (PP-OCR models, ONNX) on a grey image: detection on tiles of the upright image and of the rotated,
    reduced image (rotated: the factor, 0 for none), one batched recognition of all boxes, tall boxes recognised
    rotated both ways; boxes in input pixels. Items below min_conf are dropped (the engine's own threshold is
    bypassed, so the configured one is the effective one). An engine without a detector and recogniser of its own
    (engine.text_det, engine.text_rec) is called whole, upright and rotated."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if hasattr(engine, "text_det") and hasattr(engine, "text_rec"):
        quads = detect(img, engine, tile, overlap, rotated)
        quads = [quads[i] for i in dedupe_quads(quads)]
        found = recognise(img, quads, engine)
    else:
        found = _ocr_whole(img, engine, tile, overlap, min_conf)
    if min_conf is not None:
        found = [f for f in found if f["conf"] >= min_conf]
    return dedupe([f for f in found if f["text"].strip()])


def role(s):
    """First role of a text item; numbers are split into areas, room numbers and dimensions in stage 7, and void
    names (also hyphenated over two lines) are recognised on the clustered stamp (attributes.cluster_stamps)."""
    up = s.upper()
    if up in FIXTURE_WORDS or up.rstrip("-") in FIXTURE_WORDS:
        return "fixture label"
    if re.sub(r"[^A-ZÄÖÜ]", "", up) in VOID_NAMES:
        return "void label"
    if AOID_RE.search(s):
        return "room stamp"
    if re.fullmatch(r"[\d.,\s]+(m\s*[2²?'°]|qm|m\^2)?", s, re.I):
        return "number"
    if sum(ch.isalpha() for ch in s) >= 2:
        return "room stamp"
    return "other"


def text_layer(sheet, engine, cfg=DEFAULT, items=None):
    """Native text of the sheet plus OCR items, each with a first role. items: OCR items already in working pixels
    (pipeline.run_document reads the sheet once and hands every drawing its share through sheet.meta["ocr_items"]);
    without them the sheet's OCR image is read here."""
    out = merge_chars(sheet.native_text)
    if items is None:
        items = sheet.meta.pop("ocr_items", None)
    if items is None and sheet.ocr_img is not None and engine is not None:
        f = cfg.px_per_m / cfg.ocr_px_per_m
        items = []
        for t in ocr(sheet.ocr_img, engine, cfg.ocr_tile, cfg.ocr_overlap, cfg.ocr_min_conf, cfg.ocr_rotated_scale):
            x0, y0, x1, y1 = t["box"]
            items.append(dict(t, box=(x0 * f, y0 * f, x1 * f, y1 * f), height=t["height"] * f))
    native = [shapely.box(*n["box"]) for n in out]
    for t in items or []:
        if t["conf"] < cfg.ocr_min_conf or len(t["text"].strip()) < 1:
            continue
        b = shapely.box(*t["box"])
        if any(b.intersection(n).area > 0.3 * b.area for n in native):
            continue                                   # already in the native text layer
        out.append(dict(t))
    mask = getattr(sheet, "drawing_mask", None)
    if mask is not None:                                   # stage 1b: text in the title block or notes is no room text
        H, W = mask.shape
        out = [t for t in out if mask[min(max(int((t["box"][1] + t["box"][3]) / 2), 0), H - 1), min(max(int((t["box"][0] + t["box"][2]) / 2), 0), W - 1)]]
    for t in out:
        t["role"] = role(t["text"].strip())
    sheet.text = out
    return out
