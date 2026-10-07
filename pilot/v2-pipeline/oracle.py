"""Oracle segmentation and reference of a Swiss Dwellings floor: the floor rasterised into a perfect label map, so that
stages 3b-9 (wall graph to QA) can be tested and tuned without the segmenter, and the floor's own geometry as the
reference (rooms with their area types, door and window centres, door connections, GF). The harness (harness.py)
uses the same frame and reference for the rendered floors.

    from oracle import floors, oracle_sheet, match
    for rec in floors(DATA / "floors-test.pkl", 20):
        sheet, ref = oracle_sheet(rec)
        pipeline.run(sheet, stages=pipeline.AFTER_SEGMENTATION)
        print(match([r["poly"] for r in sheet.rooms + sheet.fragments], [p for _, p in ref]))

Swiss Dwellings is CC BY 4.0; the floors pickles come from sd_prepare.py.
"""
import pickle
import zlib

import cv2
import numpy as np
import shapely

from fpx import DEFAULT, Sheet
from fpx.model import CLASSES, COLUMN, DOOR, STAIRS, WALL, WINDOW
from metrics import iou, match_rooms  # noqa: F401 (iou re-exported)

OUTDOOR = {"BALCONY", "LOGGIA", "TERRACE", "PATIO", "GARDEN", "OUTDOOR_VOID", "LIGHTWELL"}
NOT_ROOMS = OUTDOOR | {"VOID"}                 # area types that are not rooms bounded by walls
REF_MIN_ROOM = 0.8                             # m²: smaller reference areas are not scored (the pipeline's default room_min_area);
                                               # fixed, so that sweeping room_min_area does not move the reference
DOOR_TOUCH = 0.05                              # m: a door polygon connects the areas within this distance (they share its faces)
OPEN_TOUCH = 0.02                              # m: areas this close have no wall between them (walls are thicker)
OPEN_MIN = 0.5                                 # m: shortest shared boundary of an open connection (corners touching do not count)


def floors(path, n=None):
    """Floor records from a floors pickle (sd_prepare.py or ifc_prepare.py), the first n or all."""
    blobs = pickle.loads(open(path, "rb").read())
    for b in blobs[:n]:
        yield pickle.loads(zlib.decompress(b))


def polys(g):
    """Polygons of a geometry (Polygon, MultiPolygon or collection), empty ones dropped."""
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    return [p for part in getattr(g, "geoms", []) for p in polys(part)]


def _fill(img, g, value):
    rings = [np.round(np.asarray(r.coords) * 16).astype(np.int32)
             for p in polys(g) for r in (p.exterior, *p.interiors) if len(r.coords) >= 3]
    if rings:
        cv2.fillPoly(img, rings, value, shift=4)


def frame(rec, cfg=DEFAULT, pad_m=1.0):
    """Working-pixel frame of a floor at cfg.px_per_m: (transform plan metres -> pixels (y down), (H, W), origin in
    plan metres of pixel (0, 0)). The extent covers walls, openings, railings, stairs and areas plus pad_m."""
    m = cfg.px_per_m
    geoms = [rec[k] for k in ("walls", "doors", "windows", "columns", "railings")] + list(rec["stairs"]) + [p for _, p in rec["areas"]]
    b = np.array([g.bounds for g in geoms if g is not None and not g.is_empty])
    x0, y0, x1, y1 = (*b[:, :2].min(0), *b[:, 2:].max(0))
    W, H = int(np.ceil((x1 - x0 + 2 * pad_m) * m)), int(np.ceil((y1 - y0 + 2 * pad_m) * m))
    tf = lambda g: shapely.transform(g, lambda xy: np.c_[(xy[:, 0] - x0 + pad_m) * m, (y1 - xy[:, 1] + pad_m) * m])
    return tf, (H, W), (x0 - pad_m, y1 + pad_m)


def labels(rec, tf, shape):
    """Perfect label map, painted in the renderer's order (stairs, walls, columns, windows, doors: later classes win)."""
    lab = np.zeros(shape, np.uint8)
    for p in rec["stairs"]:
        _fill(lab, tf(p), STAIRS)
    for key, value in (("walls", WALL), ("columns", COLUMN), ("windows", WINDOW), ("doors", DOOR)):
        _fill(lab, tf(rec[key]), value)
    return lab


def union(geoms):
    """Union that survives invalid input (some floors have self-touching rings): repaired first, then snapped."""
    gs = [shapely.make_valid(g) for g in geoms if g is not None and not g.is_empty]
    try:
        return shapely.union_all(gs)
    except shapely.errors.GEOSException:
        return shapely.union_all(gs, grid_size=1e-3)


def reference(rec, tf, min_room=REF_MIN_ROOM, touch=DOOR_TOUCH):
    """The floor's own geometry as the reference, in the frame of tf:
    rooms     [(area type, polygon)]: areas that are not outdoor or void and at least min_room (m²)
    ignore    [polygon]: the other areas (outdoor, void, small); predictions on them do not count against precision
    edges     {frozenset of two room indices}: a door polygon touches both rooms (within touch m)
    open_edges  rooms that touch directly over at least OPEN_MIN (no wall between them: open plan or a wall gap
              without a door); not door connections, so the harness neither credits nor penalises them
    openings  [(kind, (x, y))]: door and window centres
    gf        union of walls, columns, openings, stairs and the room and void areas, gaps under 2 x touch closed
    gf_with_outdoor   the same with the outdoor areas"""
    rooms, ignore = [], []
    for k, p in rec["areas"]:
        if p.is_empty:
            continue
        (rooms if k not in NOT_ROOMS and p.area >= min_room else ignore).append((k, p))
    edges, open_edges = set(), set()
    if rooms:
        tree = shapely.STRtree([p for _, p in rooms])
        for d in polys(rec["doors"]):
            zone = d.buffer(touch)
            near = sorted(tree.query(zone, predicate="intersects"), key=lambda i: -zone.intersection(rooms[i][1]).area)
            if len(near) >= 2:
                edges.add(frozenset(int(i) for i in near[:2]))
        for i, (_, a) in enumerate(rooms):               # open connections: areas that touch without a wall between them
            for j in tree.query(a.buffer(OPEN_TOUCH), predicate="intersects"):
                e = frozenset((i, int(j)))
                if j > i and e not in edges and a.boundary.intersection(rooms[j][1].buffer(OPEN_TOUCH)).length >= OPEN_MIN:
                    open_edges.add(e)
    openings = [(kind, tuple(np.asarray(tf(p).centroid.coords[0]))) for key, kind in (("doors", "door"), ("windows", "window"))
                for p in polys(rec[key])]
    solid = [rec[k] for k in ("walls", "columns", "doors", "windows")] + list(rec["stairs"])
    inside = [p for k, p in rec["areas"] if k not in OUTDOOR]
    close = lambda gs: union(gs).buffer(touch, join_style="mitre").buffer(-touch, join_style="mitre")
    gf = close(solid + inside)
    gf_out = close(solid + [p for _, p in rec["areas"]])
    return {"rooms": [(k, tf(p)) for k, p in rooms], "ignore": [tf(p) for _, p in ignore], "edges": edges,
            "open_edges": open_edges, "openings": openings, "gf": tf(gf), "gf_with_outdoor": tf(gf_out)}


def oracle_sheet(rec, cfg=DEFAULT, pad_m=1.0, min_room=None):
    """Floor record -> (sheet with label and prob set, reference rooms [(type, polygon in working px)]).

    The image shows walls and columns in black on white, enough for the stages that look at it (sealing)."""
    m = cfg.px_per_m
    min_room = cfg.room_min_area if min_room is None else min_room
    tf, (H, W), origin = frame(rec, cfg, pad_m)
    lab = labels(rec, tf, (H, W))
    img = np.full((H, W, 3), 255, np.uint8)
    img[np.isin(lab, (WALL, COLUMN))] = 0
    sheet = Sheet(rec["floor_id"], f"Swiss Dwellings floor {rec['floor_id']} (oracle labels)", "Swiss Dwellings v3.0.0",
                  "oracle labels", img, origin, {"value": f"{m} px/m", "method": "oracle"}, px_per_m=m)
    sheet.label = lab
    sheet.prob = (np.arange(len(CLASSES))[:, None, None] == lab[None]).astype(np.uint8)   # one-hot; uint8 keeps big floors small
    ref = [(k, tf(p)) for k, p in rec["areas"] if k not in NOT_ROOMS and p.area >= min_room]
    return sheet, ref


match = match_rooms                            # one-to-one room matching (moved to metrics.py)
