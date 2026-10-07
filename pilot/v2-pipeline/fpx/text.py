"""Stage 2, text layer: native PDF text, OCR on rasters, and a first role for each text item."""
import re

import cv2
import numpy as np
import shapely

from .config import DEFAULT

FIXTURE_WORDS = {"KACHELOFEN", "KACHEL", "OFEN", "CHEMINEE", "KAMIN", "SPIEGEL"}
VOID_WORDS = {"LUFTRAUM", "LUFT", "RAUM", "VIDE"}
AOID_RE = re.compile(r"\b\d{4}\.[A-Z0-9]{2}\.\d{2}\.\d{3}\b")


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


def ocr(img, engine, tile=1280, overlap=200):
    """RapidOCR (PP-OCR models, ONNX) on tiles, upright and rotated by 90 degrees; boxes in input pixels."""
    found = []
    for rot in (0, 90):
        im = img if rot == 0 else cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        H, W = im.shape[:2]
        for y in range(0, max(H - overlap, 1), tile - overlap):
            for x in range(0, max(W - overlap, 1), tile - overlap):
                t = im[y:y + tile, x:x + tile]
                if (t < 128).mean() < 0.002:
                    continue
                r = engine(cv2.cvtColor(t, cv2.COLOR_GRAY2BGR))
                if r.boxes is None:
                    continue
                for box, txt, sc in zip(r.boxes, r.txts, r.scores):
                    b = np.asarray(box) + [x, y]
                    if rot == 90:                      # rotated (u, v) -> original (x, y) = (v, H_orig - 1 - u)
                        b = np.c_[b[:, 1], img.shape[0] - 1 - b[:, 0]]
                    x0, y0 = b.min(0)
                    x1, y1 = b.max(0)
                    found.append({"text": txt.replace("–", "-").replace("—", "-").replace(" -", "-"),
                                  "box": (x0, y0, x1, y1), "conf": float(sc), "source": "ocr",
                                  "angle": 0 if x1 - x0 >= y1 - y0 else 90, "height": float(min(x1 - x0, y1 - y0))})
    keep = []                                          # drop duplicates from tile overlaps and the two rotations
    for f in sorted(found, key=lambda f: -f["conf"] * len(f["text"])):
        b = shapely.box(*f["box"])
        if all(b.intersection(shapely.box(*k["box"])).area < 0.5 * min(b.area, shapely.box(*k["box"]).area) for k in keep):
            keep.append(f)
    return keep


def role(s):
    """First role of a text item; numbers are split into areas and dimensions in stage 7."""
    up = s.upper()
    if up in FIXTURE_WORDS or up.rstrip("-") in FIXTURE_WORDS:
        return "fixture label"
    if up.rstrip("-") in VOID_WORDS:
        return "void label"
    if AOID_RE.search(s):
        return "room stamp"
    if re.fullmatch(r"[\d.,\s]+(m2|m²)?", s, re.I):
        return "number"
    if sum(ch.isalpha() for ch in s) >= 2:
        return "room stamp"
    return "other"


def text_layer(sheet, engine, cfg=DEFAULT):
    items = merge_chars(sheet.native_text)
    if sheet.ocr_img is not None and engine is not None:
        f = cfg.px_per_m / cfg.ocr_px_per_m
        for t in ocr(sheet.ocr_img, engine, cfg.ocr_tile, cfg.ocr_overlap):
            if t["conf"] < cfg.ocr_min_conf or len(t["text"].strip()) < 1:
                continue
            x0, y0, x1, y1 = t["box"]
            t.update(box=(x0 * f, y0 * f, x1 * f, y1 * f), height=t["height"] * f)
            b = shapely.box(*t["box"])
            if any(b.intersection(shapely.box(*n["box"])).area > 0.3 * b.area for n in items if n["source"] == "pdf"):
                continue                               # already in the native text layer
            items.append(t)
    for t in items:
        t["role"] = role(t["text"].strip())
    sheet.text = items
    return items
