"""Geometry helpers shared by the stages: masks to polygons, component filtering, dominant direction."""
import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon


def to_plan(sheet, g):
    """Shapely geometry in working pixels -> plan metres."""
    return shapely.transform(g, lambda xy: sheet.to_plan(xy))


def mask_polys(mask, min_px=1, edges=True, simplify=0.0):
    """Binary mask -> list of shapely polygons (with holes) in pixel coordinates.

    Contours run through the centres of the boundary pixels, so a traced polygon is half a pixel too small on every
    side (a 4 x 5 m room at 50 px/m loses 0.9 % of its area). With edges=True the polygon is offset outward by half a
    pixel (mitred corners), so that it follows the pixel edges and adjacent regions share their boundary. simplify
    (px) runs Douglas-Peucker on the pixel-centre contour first: the steps of a tilted edge oscillate around the chord
    by half a pixel, so 1 px removes them without an area bias."""
    cnts, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for i, c in enumerate(cnts):
        if hier[0][i][3] != -1 or len(c) < 3:
            continue
        holes, j = [], hier[0][i][2]
        while j != -1:
            if len(cnts[j]) >= 3:
                holes.append(cnts[j][:, 0, :] + 0.5)
            j = hier[0][j][0]
        p = Polygon(c[:, 0, :] + 0.5, holes).buffer(0)
        if simplify and not p.is_empty:
            p = p.simplify(simplify)
        if edges and not p.is_empty:
            p = p.buffer(0.5, join_style="mitre")
        out += [q for q in getattr(p, "geoms", [p]) if q.geom_type == "Polygon" and q.area >= min_px]
    return out


def clean(mask, min_px):
    """Drop 8-connected components smaller than min_px pixels."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[lab]


def touches_border(m):
    return bool(m[:2].any() or m[-2:].any() or m[:, :2].any() or m[:, -2:].any())


def dominant_angle(segments):
    """Length-weighted mode of wall directions modulo 90 degrees (image frame)."""
    ang, w = [], []
    for s in segments:
        c = np.asarray(s["line"].coords)
        d = c[1:] - c[:-1]
        ang += list(np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 90)
        w += list(np.hypot(d[:, 0], d[:, 1]))
    if not ang:
        return 0.0
    hist, edges = np.histogram(ang, bins=90, range=(0, 90), weights=w)
    k = int(np.argmax(hist + np.roll(hist, 1) + np.roll(hist, -1)))
    centre = edges[k] + 0.5
    # the length-weighted mean of the directions within the mode's three bins, not the bin centre: a plan drawn at
    # exactly 0 degrees would otherwise be snapped to 0.5 and every long wall rotated by half a degree
    ang, w = np.asarray(ang), np.asarray(w)
    rel = (ang - centre + 45) % 90 - 45
    sel = np.abs(rel) <= 1.5
    return float((centre + np.average(rel[sel], weights=w[sel])) % 90) if sel.any() and w[sel].sum() > 0 else float(centre)
