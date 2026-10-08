"""One module per dataset. Each gives the plan in its native pixels and the reference in the working frame:

    native(id) -> Native(img, px_per_m, scale_method, polys, masks)
        img           (H, W, 3) uint8 RGB
        px_per_m      the known scale (None: unknown, to be proposed from the sheet's cues)
        scale_method  how it is known
        polys         the reference as polygons per class in native pixels: {walls, doors, windows, columns, stairs:
                      [polygon], rooms: [(type, polygon)], ignore: [polygon] (areas that are not scored)}, or None
        masks         or as boolean masks {walls, doors, windows, interior, columns, stairs}, or None
    reference(native, work, cfg) -> Reference
        the dict fpeval.oracle.reference() returns, in the working frame of work (fpeval.raster.Working):
        rooms [(type, polygon)], ignore [polygon], edges {frozenset of two room indices}, open_edges, openings
        [(kind, (x, y))], gf, gf_with_outdoor, label (class map), unscored [class names without a reference];
        a key missing or None is not scored (fpeval.score)
    build(id, ctx) -> (Sheet at the working resolution, Reference, info, cache key)    one harness item
    items(ctx) -> iterator of (id, build thunk)                                        all harness items

Sources: CubiCasa5K (CC BY-NC-SA), CVC-FP (CC BY-NC), the WAFFLE benchmark (per-image Commons licences), Swiss plans
on Wikimedia Commons (public domain), FloorPlanCAD (CC BY-NC, its own protocol in fpeval.fpcad), the Landgut Lohn
sheets (BBL, local only), rendered Swiss Dwellings floors (CC BY), and the curated collection (data/curated) that
draws on all of them. Every public dataset is benchmark data: outputs stay in the gitignored data folder.
"""
import hashlib
from collections import namedtuple

import cv2
import numpy as np
import shapely

from common import DATA, REPO
from fpeval.raster import mask_to_working, paint
from fpx.geometry import mask_polys
from fpx.model import COLUMN, DOOR, STAIRS, WALL, WINDOW

BENCH = REPO / "data/benchmark"
CACHE = DATA / "harness/cache"                  # segmenter outputs and proposed scales, keyed by model and input
OPENING_MIN_PX = 4                              # a door or window blob of fewer working pixels is mask noise, not a symbol
Native = namedtuple("Native", "img px_per_m scale_method polys masks")
PAINT_ORDER = (("stairs", STAIRS), ("walls", WALL), ("columns", COLUMN), ("windows", WINDOW), ("doors", DOOR))   # later classes win, as the renderer paints


def file_sha1(path, n=None):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def reference_from_polys(polys, tf, shape):
    """Reference from native polygons per class: typed rooms, ignored areas, door and window centres, and the label
    map painted from the class polygons (rooms without a type are typed None)."""
    lab = np.zeros(shape, np.uint8)
    for key, value in PAINT_ORDER:
        paint(lab, [tf(p) for p in polys.get(key, [])], value)
    rooms = [(r[0], tf(r[1])) if isinstance(r, tuple) else (None, tf(r)) for r in polys.get("rooms", [])]
    openings = [(kind, tuple(tf(p).centroid.coords[0])) for key, kind in (("doors", "door"), ("windows", "window"))
                for p in polys.get(key, [])]
    return {"rooms": rooms, "ignore": [tf(p) for p in polys.get("ignore", [])], "openings": openings, "label": lab}


def reference_from_masks(masks, shape, pad, cfg):
    """Reference from native boolean masks: the label map, opening centres from the blobs of the door and window
    masks, and the GF from the interior plus walls and openings when an interior mask is given. Returns (reference,
    the masks in the working frame)."""
    small = {k: mask_to_working(v, shape, pad) for k, v in masks.items()}
    lab = np.zeros(shape, np.uint8)
    for key, value in PAINT_ORDER:
        if key in small:
            lab[small[key]] = value
    openings = []
    for key, kind in (("doors", "door"), ("windows", "window")):
        if key in small:
            n, cc, stats, cents = cv2.connectedComponentsWithStats(small[key].astype(np.uint8), connectivity=8)
            openings += [(kind, tuple(cents[i])) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= OPENING_MIN_PX]
    ref = {"openings": openings, "label": lab}
    if "interior" in small:
        building = small["interior"].copy()
        for key in ("walls", "doors", "windows"):
            if key in small:
                building |= small[key]
        ref["gf"] = shapely.union_all(mask_polys(building, cfg.px2(1.0)))
    return ref, small


def reference(nat, work, cfg, unscored=()):
    """Reference of a Native in the working frame of work: rooms and ignored areas from its polygons, the label map
    and openings from its masks when it has them (exact), else painted from its polygons."""
    shape = work.img.shape[:2]
    ref = {"unscored": list(unscored)}
    if nat.masks:
        ref.update(reference_from_masks(nat.masks, shape, work.pad, cfg)[0])
    if nat.polys:
        rp = reference_from_polys(nat.polys, work.tf, shape)
        if "rooms" in nat.polys:
            ref["rooms"], ref["ignore"] = rp["rooms"], rp["ignore"]
        if not nat.masks and any(k in nat.polys for k, _ in PAINT_ORDER):
            ref["label"], ref["openings"] = rp["label"], rp["openings"]
    return ref
