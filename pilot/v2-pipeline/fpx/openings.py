"""Stage 4, openings: doors and windows from the segmenter, open passages from wall gaps, host walls."""
import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon

from .config import DEFAULT
from .geometry import clean, dominant_angle
from .model import DOOR, STAIRS, WALL, WINDOW


def rough_building(barrier, cfg=DEFAULT):
    """Largest region enclosed by the barrier after closing facade gaps (doors, windows) of up to building_close."""
    k = int(cfg.px(cfg.building_close)) | 1
    closed = cv2.morphologyEx(barrier.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    filled = ndimage.binary_fill_holes(closed)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(filled.astype(np.uint8), connectivity=8)
    return lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if n > 1 else filled


def openings(sheet, cfg=DEFAULT):
    m = cfg.m
    lab = sheet.label
    found = []
    near_wall = cv2.dilate(sheet.wall_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    for cls, kind in ((DOOR, "door"), (WINDOW, "window")):
        n, cc, stats, _ = cv2.connectedComponentsWithStats(clean(lab == cls, cfg.px2(cfg.opening_min_area)).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if not (near_wall & (cc == i)).any():      # an opening sits in a wall; isolated pixels are noise
                lab[cc == i] = 0
                continue
            ys, xs = np.nonzero(cc == i)
            rect = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
            (cx, cy), (w, h), ang = rect
            L, T = max(w, h), min(w, h)
            if L < cfg.px(cfg.opening_min_length):
                continue
            a = np.radians(ang if w >= h else ang + 90)
            found.append({"kind": kind, "centre": np.array([cx, cy]), "along": np.array([np.cos(a), np.sin(a)]),
                          "width": L / m, "depth": T / m, "poly": Polygon(cv2.boxPoints(rect)),
                          "score": float(sheet.prob[cls][cc == i].mean()), "source": "segmenter"})
    barrier = sheet.wall_mask | sheet.column_mask | (lab == DOOR) | (lab == WINDOW)
    sheet.building_rough = rough_building(barrier | (lab == STAIRS), cfg)
    found += passages(sheet, barrier, cfg)
    # host wall, exterior or interior, and plausibility of the type
    segs = sheet.wall_segments
    for i, o in enumerate(found):
        o["id"] = f"o{i:03d}"
        c = Point(o["centre"])
        host = min(segs, key=lambda s: s["line"].distance(c), default=None)
        o["host"] = host["id"] if host is not None and host["line"].distance(c) < cfg.px(cfg.opening_host_max_dist) else None
        nrm = np.array([-o["along"][1], o["along"][0]])
        off = (o["depth"] / 2 + cfg.opening_side_offset) * m
        sides = [o["centre"] + s * nrm * off for s in (1, -1)]
        inside = [bool(sheet.building_rough[int(np.clip(p[1], 0, lab.shape[0] - 1)), int(np.clip(p[0], 0, lab.shape[1] - 1))]) for p in sides]
        o["exterior"] = not all(inside)                 # refined in stage 8 with the GF outline
        o["sides_px"] = sides
    sheet.openings = found
    return found


def passages(sheet, barrier, cfg=DEFAULT):
    """Open passages: a wall end facing another wall within passage_min..passage_max, with no door or window in
    between. Ends that are jambs of a detected opening are skipped, and the ray must stay inside the building."""
    G, lab, wall = sheet.wall_graph, sheet.label, sheet.wall_mask
    H, W = lab.shape
    at = lambda p: (int(np.clip(p[1], 0, H - 1)), int(np.clip(p[0], 0, W - 1)))
    opening_near = cv2.dilate(np.isin(lab, (DOOR, WINDOW)).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    nw, wcc, wstats, _ = cv2.connectedComponentsWithStats(wall.astype(np.uint8), connectivity=8)
    big = np.zeros(nw, bool)
    big[1:] = wstats[1:, cv2.CC_STAT_AREA] >= cfg.px2(cfg.passage_min_wall_area)   # ornaments and stubs are not walls to bridge
    dom = dominant_angle(sheet.wall_segments)
    tol = cfg.passage_max_angle
    lo, hi = cfg.px(cfg.passage_min), cfg.px(cfg.passage_max)
    out = []
    for n, d in G.nodes(data=True):
        if G.degree(n) != 1:
            continue
        (_, _, e), = [(u, v, dd) for u, v, dd in G.edges(n, data=True)]
        path, p_end = e["path"], d["xy"]
        if np.linalg.norm(path[0] - p_end) > np.linalg.norm(path[-1] - p_end):
            path = path[::-1]
        back = min(len(path) - 1, int(cfg.px(cfg.passage_back)))
        if back < 10:
            continue
        direction = path[0] - path[back]
        direction /= max(np.linalg.norm(direction), 1e-6)
        if not big[wcc[at(p_end)]] or (np.degrees(np.arctan2(direction[1], direction[0])) - dom + tol) % 90 > 2 * tol:
            continue
        pos, steps = p_end.astype(float).copy(), 0
        while steps < cfg.px(cfg.passage_exit_max) and wall[at(pos)]:   # leave the own wall through wall pixels only
            pos += direction
            steps += 1
        if opening_near[at(pos)]:                             # the wall ends at a door or window: a jamb
            continue
        start, free = pos.copy(), 0
        while free < hi:
            y, x = at(pos)
            if not sheet.building_rough[y, x]:
                free = None
                break
            if barrier[y, x]:
                break
            pos += direction
            free += 1
        if free is None or not lo <= free < hi or lab[at(pos)] != WALL or not big[wcc[at(pos)]]:
            continue
        mid = (start + pos) / 2
        if any(np.linalg.norm(o["centre"] - mid) < cfg.px(cfg.passage_dedupe) for o in out):
            continue
        out.append({"kind": "passage", "centre": mid, "along": direction.copy(), "width": free / cfg.m,
                    "depth": max(float(e.get("thickness", 0.2)), 0.1), "line": (start, pos.copy()),
                    "poly": LineString([start, pos]).buffer(1.5, cap_style="flat"), "score": 0.5, "source": "wall gap"})
    return out
