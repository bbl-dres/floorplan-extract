"""Raster helpers shared by the benchmarks: image loading, resampling to the working resolution with a white pad,
painting polygons into label maps, the class palette and overlays."""
from collections import namedtuple

import cv2
import numpy as np
from PIL import Image
from shapely import affinity

from fpx.config import DEFAULT

Image.MAX_IMAGE_PIXELS = None                   # benchmark sheets and scans run to 100 Mpx
# background white, wall dark grey, door red, window blue, column green, stairs magenta (fpx.model.CLASSES order)
PALETTE = np.array([[255, 255, 255], [40, 40, 40], [230, 40, 40], [40, 120, 230], [40, 170, 40], [200, 40, 200]], np.uint8)
PASSAGE_COLOUR = (255, 140, 0)
Working = namedtuple("Working", "img ocr_img tf f pad")


def rgb(path):
    """An image file as (H, W, 3) uint8 RGB; transparency is composited on white."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("RGB"))


def resized(img, f):
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)


def to_working(img, px_per_m, cfg=DEFAULT, pad_m=1.0, ocr=False):
    """A native image at px_per_m -> Working(img at cfg.px_per_m with pad_m of white around it, the grey OCR image at
    cfg.ocr_px_per_m with the same pad or None, tf: shapely transform native px -> working px, f: the factor,
    pad: the pad in working px). The pad keeps rooms off the sheet border (crops end at the outer wall face)."""
    f = cfg.px_per_m / px_per_m
    pad = int(round(pad_m * cfg.px_per_m))
    work = cv2.copyMakeBorder(resized(img, f), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    ocr_img = None
    if ocr:
        po = int(round(pad_m * cfg.ocr_px_per_m))
        grey = cv2.cvtColor(resized(img, cfg.ocr_px_per_m / px_per_m), cv2.COLOR_RGB2GRAY)
        ocr_img = cv2.copyMakeBorder(grey, po, po, po, po, cv2.BORDER_CONSTANT, value=255)
    tf = lambda g: affinity.translate(affinity.scale(g, f, f, origin=(0, 0)), pad, pad)
    return Working(work, ocr_img, tf, f, pad)


def mask_to_working(mask, shape, pad):
    """A native boolean mask resampled (nearest) into the padded working frame of shape (H, W)."""
    H, W = shape
    small = cv2.resize(mask.astype(np.uint8), (W - 2 * pad, H - 2 * pad), interpolation=cv2.INTER_NEAREST) > 0
    return np.pad(small, pad)


def polygons(g):
    """Polygons of a geometry (Polygon, MultiPolygon or collection), empty ones dropped."""
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    return [p for part in getattr(g, "geoms", []) for p in polygons(part)]


def paint(lab, geoms, value):
    """Fill geometries (working px, holes respected) into a label map, exact to 1/16 px."""
    for g in geoms:
        rings = [np.round(np.asarray(r.coords) * 16).astype(np.int32)
                 for p in polygons(g) for r in (p.exterior, *p.interiors) if len(r.coords) >= 3]
        if rings:
            cv2.fillPoly(lab, rings, value, shift=4)


def tint(img, label, alpha=0.65):
    """The sheet with the class palette blended over the labelled pixels."""
    out = img.copy()
    m = label > 0
    out[m] = ((1 - alpha) * out[m] + alpha * PALETTE[label][m]).astype(np.uint8)
    return out


def overlay(sheet, label=None, names=True, seed=1):
    """Segmenter classes over the sheet, plus room outlines (one colour per room, with id and name), and passages."""
    img = tint(sheet.img, sheet.label if label is None else label)
    rng = np.random.default_rng(seed)
    for r in sheet.rooms:
        c = tuple(int(v) for v in rng.integers(60, 220, 3))
        cv2.polylines(img, [np.asarray(r["poly"].exterior.coords, np.int32)], True, c, 2)
        if names:
            x, y = r["poly"].representative_point().coords[0]
            cv2.putText(img, f"{r['id']} {r.get('name') or ''}"[:28], (int(x) - 30, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    for o in sheet.openings:
        if o["kind"] == "passage":
            a, b = o["line"]
            cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), PASSAGE_COLOUR, 3)
    return img
