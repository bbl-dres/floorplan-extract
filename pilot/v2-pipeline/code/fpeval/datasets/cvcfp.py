"""CVC-FP: scanned plans with SVG ground truth (walls, doors, windows, rooms, separation lines, parking).

CC BY-NC, evaluation only (data/benchmark/cvc-fp). The scale comes from the reference door symbols. Door labels
cover the swing, so door centres are taken on the side of the swing that lies in the wall opening, and door pixels
are not scored. Rooms next to a "Separation" line (open plan) are typed "room (separation)"; "Parking" areas are
not scored.
"""
import re

import numpy as np
import shapely
from shapely.geometry import Point

from fpeval.cli import Skip
from fpeval.datasets import BENCH, Native, reference_from_polys
from fpeval.raster import polygons, rgb, to_working
from fpx.model import Sheet

FOLDER = BENCH / "cvc-fp/ImagesGT"
DOOR_M = 0.85                                   # the drawn door symbols: leaf length (the swing's radius) of a standard door
SEP_M = 0.1                                     # a room within this of a separation line is open plan next to it (the line is drawn on the room's edge)
JAMB_M = 0.15                                   # a side of the door symbol spans the wall opening when both ends lie within this of a wall


def ids():
    return sorted({p.name.split("_gt_")[0] for p in FOLDER.glob("*_gt_*.svg")})


def paths(n):
    return next(FOLDER.glob(f"{n}_gt_*.svg")), next(p for p in (FOLDER / f"{n}.png", FOLDER / f"{n}.jpg") if p.exists())


def image_path(n):
    return paths(n)[1]


def group(n):
    if n.isdigit():
        return "numbered"
    if n.startswith("image"):
        return "image"
    if n.startswith("II"):
        return "II"
    if n.startswith("I"):
        return "I (sommaire)"
    return "p"


def parse(svg):
    """CVC-FP ground truth: polygons per class (native pixels)."""
    text = svg.read_text(encoding="utf-8", errors="replace")
    out = {}
    for cls, pts in re.findall(r'<polygon class="([^"]+)"[^>]*points="([^"]+)"', text):
        xy = np.array([float(v) for v in re.split(r"[ ,]+", pts.strip()) if v]).reshape(-1, 2)
        if len(xy) >= 3:
            p = shapely.Polygon(xy).buffer(0)
            if p.area > 0:
                out.setdefault(cls, []).extend(polygons(p))
    return out


def scale(doors):
    """px per m from the longest side of the door symbols' rectangles (the leaf) = DOOR_M."""
    edges = [np.diff(np.asarray(d.minimum_rotated_rectangle.exterior.coords)[:3], axis=0) for d in doors]
    return float(np.median([np.hypot(e[:, 0], e[:, 1]).max() for e in edges])) / DOOR_M


def door_in_wall(door, walls, tol):
    """CVC-FP door symbols cover the swing: the door itself is the side of the swing's rectangle that spans the wall
    opening (both ends on a wall, its middle off the wall). Falls back to the centroid."""
    c = np.asarray(door.minimum_rotated_rectangle.exterior.coords)[:4]
    best = None
    for k in range(4):
        a, b = c[k], c[(k + 1) % 4]
        ends = max(walls.distance(shapely.Point(a)), walls.distance(shapely.Point(b)))
        mid = walls.distance(shapely.Point((a + b) / 2))
        if ends <= tol and (best is None or mid > best[0]):
            best = (mid, (a + b) / 2)
    return best[1] if best is not None else np.asarray(door.centroid.coords[0])


def native(n):
    svg, img_path = paths(n)
    gt = parse(svg)
    doors = gt.get("Door", [])
    ppm = scale(doors) if doors else None
    walls = shapely.union_all(gt.get("Wall", [])) if gt.get("Wall") else shapely.Polygon()
    sep = shapely.union_all(gt["Separation"]).buffer(SEP_M * ppm) if gt.get("Separation") and ppm else None
    polys = {"walls": gt.get("Wall", []), "doors": doors, "windows": gt.get("Window", []),
             "rooms": [("room (separation)" if sep is not None and r.intersects(sep) else "room", r) for r in gt.get("Room", [])],
             "ignore": gt.get("Parking", []),
             "door_centres": [door_in_wall(d, walls, JAMB_M * ppm) for d in doors] if ppm else []}
    method = f"from {len(doors)} reference door symbols ({DOOR_M} m)" if ppm else "unknown: no reference door symbols"
    return Native(rgb(img_path), ppm, method, polys, None)


def reference(nat, work, cfg):
    """Rooms and parking from the polygons; walls and windows in the label map (doors unscored: the label covers the
    swing); door centres on the side of the swing in the wall opening."""
    ref = reference_from_polys({k: nat.polys[k] for k in ("walls", "windows", "rooms", "ignore")}, work.tf, work.img.shape[:2])
    ref["openings"] = [("door", tuple(work.tf(Point(c)).coords[0])) for c in nat.polys["door_centres"]] + ref["openings"]
    ref["unscored"] = ["door"]
    return ref


def build(n, ctx):
    nat = native(n)
    if nat.px_per_m is None:
        raise Skip("no reference doors: scale unknown")
    svg, img_path = paths(n)
    work = to_working(nat.img, nat.px_per_m, ctx.cfg, ocr=ctx.take_ocr())
    H, W = work.img.shape[:2]
    sheet = Sheet(f"cvcfp_{n}", f"CVC-FP {n}", img_path.name, "raster (public benchmark)", work.img, (0.0, 0.0),
                  {"value": f"{nat.px_per_m:.1f} px/m", "method": nat.scale_method}, ocr_img=work.ocr_img, px_per_m=ctx.cfg.px_per_m)
    info = {"group": group(n), "size": [W, H], "px_per_m_native": nat.px_per_m, "ref_doors": len(nat.polys["doors"])}
    return sheet, reference(nat, work, ctx.cfg), info, [n, img_path.stat().st_size, round(nat.px_per_m, 4)]


def items(ctx):
    for n in ids()[:ctx.args.n]:
        yield n, lambda n=n: build(n, ctx)
