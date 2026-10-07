"""CubiCasa5K reader: one plan's first floor as polygons by class, its scale, and rasterised labels.

CubiCasa5K is CC BY-NC-SA 4.0 (non-commercial): used here as a benchmark only (data/benchmark/cubicasa5k).
SVG coordinates are pixels of F1_scaled.png. The scale comes from the room dimension labels ("3.72 m x 1.86 m")
and the dimension marks drawn for the same room.
"""
import re
import xml.etree.ElementTree as ET

import numpy as np
import shapely
from shapely.geometry import Polygon

from common import COLUMN, DOOR, REPO, STAIRS, WALL, WINDOW

ROOT = REPO / "data/benchmark/cubicasa5k"
NS = "{http://www.w3.org/2000/svg}"
METRIC = re.compile(r"([\d.]+)\s*m\s*x\s*([\d.]+)\s*m")


def matrix(e):
    m = re.search(r"matrix\(([^)]+)\)", e.get("transform", ""))
    if not m:
        return np.eye(3)
    a, b, c, d, tx, ty = (float(v) for v in re.split(r"[,\s]+", m.group(1).strip()))
    return np.array([[a, c, tx], [b, d, ty], [0, 0, 1]])


def polygon(e, T):
    pts = np.array([[float(v) for v in xy.split(",")] for xy in e.get("points").split()])
    if len(pts) < 3:
        return None
    pts = (T @ np.c_[pts, np.ones(len(pts))].T).T[:, :2]
    p = Polygon(pts).buffer(0)
    return p if p.area > 0 else None


def first_polygon(g, T):
    """The outline of an element is the first polygon directly inside its group (T includes the group's transform)."""
    p = g.find(NS + "polygon")
    return polygon(p, T) if p is not None else None


def load(plan):
    """plan: e.g. 'high_quality/7523'. Returns polygons per class, rooms, scale (px per m) and the image path.
    All floors are read: on multi-floor plans F1_scaled.png and F2_scaled.png are the same sheet showing every floor."""
    root = ET.parse(ROOT / plan / "model.svg").getroot()
    floors = [g for g in root.iter(NS + "g") if g.get("class") == "Floor"]
    out = {k: [] for k in ("walls", "doors", "windows", "columns", "stairs", "railings")}
    rooms, scales = [], []

    def visit(g, T):
        T = T @ matrix(g)
        cls = g.get("class", "")
        kind = cls.split(" ")[0]
        if kind in ("Wall", "Door", "Window", "Column", "Railing"):
            p = first_polygon(g, T)
            if p is not None:
                out[{"Wall": "walls", "Door": "doors", "Window": "windows", "Column": "columns", "Railing": "railings"}[kind]].append(p)
        elif kind == "Stairs":
            ps = [polygon(p, T) for p in g.iter(NS + "polygon")]
            ps = [p for p in ps if p is not None]
            if ps:
                out["stairs"].append(shapely.union_all(ps))
            return
        elif kind == "Space":
            p = first_polygon(g, T)
            if p is not None:
                rooms.append({"type": cls[6:].strip() or "Undefined", "poly": p})
        elif kind == "Dimension":
            marks = [matrix(m)[:2, 2] for m in g.iter(NS + "g") if m.get("class") == "DimensionMark"]
            texts = " ".join("".join(t.itertext()) for t in g.iter(NS + "text"))
            m = METRIC.search(texts)
            if marks and m and float(m.group(1)) > 0.5 and float(m.group(2)) > 0.5:
                w_px, h_px = max(x for x, _ in marks), max(y for _, y in marks)
                scales.extend([w_px / float(m.group(1)), h_px / float(m.group(2))])
            return
        for ch in g:
            if ch.tag == NS + "g":
                visit(ch, T)

    for floor in floors:
        visit(floor, np.eye(3))
    return {**out, "rooms": rooms, "px_per_m": float(np.median(scales)) if scales else None,
            "image": ROOT / plan / "F1_scaled.png"}


def rasterise(plan_data, shape, factor):
    """Label mask in the pilot's classes at the image resolution scaled by factor (stairs < wall < column < window < door)."""
    import cv2
    lab = np.zeros(shape, np.uint8)

    def fill(polys, value):
        for p in polys:
            for q in getattr(p, "geoms", [p]):
                if q.geom_type == "Polygon":
                    rings = [np.round(np.asarray(r.coords) * factor * 16).astype(np.int32) for r in [q.exterior, *q.interiors]]
                    cv2.fillPoly(lab, rings, value, shift=4)
    for key, value in (("stairs", STAIRS), ("walls", WALL), ("columns", COLUMN), ("windows", WINDOW), ("doors", DOOR)):
        fill(plan_data[key], value)
    return lab
