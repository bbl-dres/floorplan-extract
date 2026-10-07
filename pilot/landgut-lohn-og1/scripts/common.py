"""Shared paths and reference data for the Landgut Lohn 1. OG pilot.

All plan files and derived images live in ../data (gitignored, non-public).
"""
import json
from pathlib import Path

from shapely.geometry import shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INPUTS = DATA / "inputs"
REFERENCE = DATA / "reference"
OUT = DATA / "out"
OUT.mkdir(parents=True, exist_ok=True)

PDF_2005 = INPUTS / "2051-AA-1. OG 2005-136979-ARCH-1-OG-100.pdf"
SCAN_0056 = INPUTS / "2051-AA-1. OG-54467-0056.tif"
SCAN_0066 = INPUTS / "2051-AA-1. OG-54477-0066.tif"

_topo = json.loads((REFERENCE / "plan_topology_og1-main.json").read_text(encoding="utf-8"))
CAL = json.loads((REFERENCE / "plan_calibration.json").read_text(encoding="utf-8"))["sheets"]["og1"]

# Reference rooms: (name, area printed on the room stamp, polygon in plan metres)
ROOMS = [(r["name"], r["labelArea"], shape(r["geometry"])) for r in _topo["rooms"]]
WALLS = [shape(w["geometry"]) for w in _topo["walls"]]
OPENINGS = [(o["gap"], shape(o["geometry"])) for o in _topo["openings"]]
BUILDING = unary_union([g for *_, g in ROOMS] + WALLS).buffer(0.05).buffer(-0.05)


def main_component(mask):
    """Keep the largest connected part of a binary mask and everything inside its outer contour."""
    import cv2
    import numpy as np
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return mask
    big = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cnts, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    footprint = np.zeros_like(mask)
    cv2.drawContours(footprint, cnts, -1, 1, -1)
    return (mask & footprint).astype(np.uint8)


def mask_to_geometry(mask, to_plan, min_area=0.02):
    """Vectorise a binary mask (outer contours with holes) into a shapely geometry in plan metres."""
    import cv2
    from shapely.geometry import Polygon
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for i, c in enumerate(cnts):
        if hier[0][i][3] != -1 or len(c) < 3:
            continue
        holes, j = [], hier[0][i][2]
        while j != -1:
            if len(cnts[j]) >= 3:
                holes.append([to_plan(x, y) for x, y in cnts[j][:, 0, :]])
            j = hier[0][j][0]
        p = Polygon([to_plan(x, y) for x, y in c[:, 0, :]], holes).buffer(0)
        if p.area >= min_area:
            polys.append(p)
    return unary_union(polys)


def iou(a, b):
    union = a.union(b).area
    return a.intersection(b).area / union if union else 0.0


def best_match(candidates, ref):
    """candidates: iterable of (key, polygon). Returns (key, polygon, IoU) with the highest IoU."""
    return max(((k, p, iou(p, ref)) for k, p in candidates), key=lambda t: t[2], default=(None, None, 0.0))
