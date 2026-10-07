"""Stage 6, rooms: free space between walls and closed openings, stair split, sealing, voids and the GF outline."""
import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon

from .config import DEFAULT
from .geometry import mask_polys, touches_border
from .model import DOOR, STAIRS, WINDOW
from .openings import rough_building


def seal(mask, gray, cfg=DEFAULT):
    """Split a region that leaks to the sheet border through an unclosed opening: cores left after an erosion by
    seal_erosion (wider than a window) that stay clear of the border are grown back with a watershed; the rest is
    outside."""
    r = int(cfg.px(cfg.seal_erosion))
    seeds = cv2.erode(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n, sl = cv2.connectedComponents(seeds, connectivity=4)
    markers = np.zeros(mask.shape, np.int32)
    keep = [j for j in range(1, n) if not touches_border(sl == j) and (sl == j).sum() >= cfg.px2(cfg.seal_min_core)]
    for q, j in enumerate(keep, 1):
        markers[sl == j] = q
    outside = len(keep) + 1
    markers[~mask] = outside
    markers[(sl > 0) & (markers == 0)] = outside
    if not keep:
        return []
    ws = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
    return [(ws == q) & mask for q in range(1, len(keep) + 1)]


def rooms(sheet, cfg=DEFAULT):
    lab = sheet.label
    min_room = cfg.px2(cfg.room_min_area)
    simplify = cfg.px(cfg.poly_simplify)
    barrier = (sheet.wall_mask | sheet.column_mask | (lab == DOOR) | (lab == WINDOW)).astype(np.uint8)
    for o in sheet.openings:                           # close passages with a virtual wall
        if o["kind"] == "passage":
            a, b = o["line"]
            cv2.line(barrier, tuple(int(v) for v in a), tuple(int(v) for v in b), 1, 3)
    k = int(cfg.px(cfg.slit_close)) | 1                # close slits between walls and partly labelled openings
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    building = rough_building(barrier | (lab == STAIRS), cfg)
    free = building & (barrier == 0)
    n, cc = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    # a stair open to a hall (no wall between them) becomes its own room, split along the stair outline
    hull, nxt = sheet.stair_hull, n
    split = cfg.px2(cfg.stair_split_min_area)
    for i in range(1, n):
        m = cc == i
        inside, outside = (m & hull).sum(), (m & ~hull).sum()
        if inside > split and outside > split:
            cc[m & hull] = nxt
            nxt += 1
    # regions that leak to the sheet border through an unclosed opening are sealed, not dropped
    gray = cv2.cvtColor(sheet.img, cv2.COLOR_RGB2GRAY)
    for i in range(1, nxt):
        m = cc == i
        if m.sum() >= min_room and touches_border(m):
            cc[m] = 0
            for part in seal(m, gray, cfg):
                cc[part] = nxt
                nxt += 1
    # the remainder of a split region may fall apart (e.g. hall and room joined only through the stair): relabel
    for i in range(1, nxt):
        k, parts = cv2.connectedComponents((cc == i).astype(np.uint8), connectivity=4)
        for j in range(2, k):
            cc[parts == j] = nxt
            nxt += 1
    out = []
    for i in range(1, nxt):
        m = cc == i
        if m.sum() < min_room:
            continue
        ps = mask_polys(m, min_room)
        if not ps:
            continue
        p = max(ps, key=lambda q: q.area).simplify(simplify)
        if touches_border(m):
            continue                                   # touches the sheet border: outside the building
        if p.area < min_room:
            continue                                   # the polygon itself must reach the minimum (plan-check POLY_004)
        stair_share = (m & hull).sum() / m.sum()
        out.append({"id": f"r{len(out):03d}", "poly": p, "px_label": i, "stair": stair_share > cfg.stair_room_share})
    sheet.room_label = cc
    sheet.rooms = out
    holes = (hull & free & ~sheet.stair_mask).astype(np.uint8)
    hn, hcc = cv2.connectedComponents(holes, connectivity=4)
    for t in sheet.text:
        if t["role"] != "void label":
            continue
        x, y = int((t["box"][0] + t["box"][2]) / 2), int((t["box"][1] + t["box"][3]) / 2)
        i = hcc[min(y, hcc.shape[0] - 1), min(x, hcc.shape[1] - 1)]
        if i and not any(v["px"] == i for v in sheet.voids):
            ps = mask_polys(hcc == i, cfg.px2(cfg.void_min_area))
            if ps:
                sheet.voids.append({"px": int(i), "poly": max(ps, key=lambda q: q.area).simplify(simplify), "label": t["text"]})
    # GF: outer contour of rooms, walls, openings and stairs, closed over small gaps
    solid = (barrier | (lab == STAIRS) | np.isin(cc, [r["px_label"] for r in out])).astype(np.uint8)
    g = int(cfg.px(cfg.gf_close))
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((g, g), np.uint8))
    solid = ndimage.binary_fill_holes(solid) & building
    gf = mask_polys(solid, cfg.px2(cfg.gf_min_area))
    sheet.gf = max(gf, key=lambda q: q.area).simplify(simplify) if gf else Polygon()
    # CAD-Richtlinie Kap. 5.9: voids over 5 m² are cut out of the GF (a hole here; the DXF writes the GF as one
    # continuous polyline with the hole cut out through a zero-width bridge, fpx.export.keyhole)
    for v in sheet.voids:
        v["gf_deducted"] = cfg.m2(v["poly"].area) > cfg.void_gf_deduction
        if v["gf_deducted"]:
            gf = sheet.gf.difference(v["poly"])
            sheet.gf = max(getattr(gf, "geoms", [gf]), key=lambda q: q.area)
    return out
