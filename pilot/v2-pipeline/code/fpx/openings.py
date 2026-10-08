"""Stage 4, openings: doors and windows from the segmenter, open passages between facing wall ends, host walls."""
import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon

from .config import DEFAULT
from .geometry import clean, dominant_angle
from .model import DOOR, STAIRS, WALL, WINDOW
from .walls import end_direction


def rough_building(barrier, cfg=DEFAULT, anchors=None):
    """Regions enclosed by the barrier after closing facade gaps (doors, windows) of up to building_close: the largest
    one, plus every other enclosed region holding an anchor pixel (door or window labels: a second wing or building on
    the sheet has doors, a legend box or a stray symbol has none)."""
    k = int(cfg.px(cfg.building_close)) | 1
    closed = cv2.morphologyEx(barrier.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    filled = ndimage.binary_fill_holes(closed)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(filled.astype(np.uint8), connectivity=8)
    if n <= 1:
        return filled
    keep = np.zeros(n, bool)
    keep[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))] = True
    if anchors is not None:
        keep |= np.bincount(lab[anchors], minlength=n) > 0
        keep[0] = False
    return keep[lab]


def host_wall(o, segs, cfg=DEFAULT):
    """The wall segment an opening sits in. Centre lines stop at the jambs, so the opening's centre lies off the end
    of its host by half its width: each segment's centre line is extended by opening_host_extension beyond its ends,
    and the host is the segment whose extended line passes closest to the centre, within its thickness (at least
    4 px) and running along the opening within opening_host_angle. Falls back to the nearest segment within
    opening_host_max_dist."""
    c = Point(o["centre"])
    ext = cfg.px(cfg.opening_host_extension) + cfg.px(o["width"]) / 2
    cos_min = np.cos(np.radians(cfg.opening_host_angle))
    best, best_d = None, None
    for s in segs:
        line = s["line"]
        if line.distance(c) > ext + cfg.px(s["thickness"]):
            continue
        xy = np.asarray(line.coords)
        d0, d1 = xy[1] - xy[0], xy[-1] - xy[-2]
        d0, d1 = d0 / max(np.linalg.norm(d0), 1e-6), d1 / max(np.linalg.norm(d1), 1e-6)
        extended = LineString(np.vstack([xy[0] - d0 * ext, xy, xy[-1] + d1 * ext]))
        dist = extended.distance(c)
        if dist > max(cfg.px(s["thickness"]) / 2 + 2, 4):
            continue
        near = np.asarray(extended.interpolate(extended.project(c)).coords[0])
        t = near - np.asarray(extended.interpolate(max(extended.project(c) - 3, 0)).coords[0])
        if np.linalg.norm(t) < 1e-6:
            t = d1
        t /= np.linalg.norm(t)
        if abs(float(np.dot(t, o["along"]))) < cos_min:
            continue
        if best_d is None or dist < best_d:
            best, best_d = s, dist
    if best is None:
        near = min(segs, key=lambda s: s["line"].distance(c), default=None)
        if near is not None and near["line"].distance(c) < cfg.px(cfg.opening_host_max_dist):
            best = near
    return best["id"] if best is not None else None


def openings(sheet, cfg=DEFAULT):
    m = cfg.m
    lab = sheet.label
    found = []
    near_wall = cv2.dilate(sheet.wall_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    for cls, kind in ((DOOR, "door"), (WINDOW, "window")):
        n, cc, stats, _ = cv2.connectedComponentsWithStats(clean(lab == cls, cfg.px2(cfg.opening_min_area)).astype(np.uint8), connectivity=8)
        touching = np.bincount(cc[near_wall], minlength=n) > 0
        score = ndimage.mean(sheet.prob[cls], cc, index=np.arange(n)) if n > 1 else []
        for i, sl in enumerate(ndimage.find_objects(cc), 1):
            if sl is None:
                continue
            if not touching[i]:                        # an opening sits in a wall; isolated pixels are noise
                lab[sl][cc[sl] == i] = 0
                continue
            ys, xs = np.nonzero(cc[sl] == i)
            rect = cv2.minAreaRect(np.c_[xs + sl[1].start, ys + sl[0].start].astype(np.float32))
            (cx, cy), (w, h), ang = rect
            L, T = max(w, h), min(w, h)
            if L < cfg.px(cfg.opening_min_length):
                continue
            a = np.radians(ang if w >= h else ang + 90)
            found.append({"kind": kind, "centre": np.array([cx, cy]), "along": np.array([np.cos(a), np.sin(a)]),
                          "width": L / m, "depth": T / m, "poly": Polygon(cv2.boxPoints(rect)),
                          "score": float(score[i]), "source": "segmenter"})
    barrier = sheet.wall_mask | sheet.column_mask | (lab == DOOR) | (lab == WINDOW)
    sheet.building_rough = rough_building(barrier | (lab == STAIRS), cfg, anchors=np.isin(lab, (DOOR, WINDOW)))
    found += passages(sheet, barrier, cfg)
    # host wall, exterior or interior, and plausibility of the type
    segs = sheet.wall_segments
    for i, o in enumerate(found):
        o["id"] = f"o{i:03d}"
        o["host"] = host_wall(o, segs, cfg)
        nrm = np.array([-o["along"][1], o["along"][0]])
        off = (o["depth"] / 2 + cfg.opening_side_offset) * m
        sides = [o["centre"] + s * nrm * off for s in (1, -1)]
        inside = [bool(sheet.building_rough[int(np.clip(p[1], 0, lab.shape[0] - 1)), int(np.clip(p[0], 0, lab.shape[1] - 1))]) for p in sides]
        o["exterior"] = not all(inside)                 # refined in stage 8 with the GF outline
        o["sides_px"] = sides
    sheet.openings = found
    return found


def passages(sheet, barrier, cfg=DEFAULT):
    """Open passages: two wall ends facing each other across passage_min..passage_max of free space, with no door or
    window in between. A wall end facing the side of another wall is a corridor corner, not a passage. Ends that are
    jambs of a detected opening are skipped, and the ray must stay inside the building."""
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
    ends = []                                            # (end point, outward direction, thickness) of every wall end
    for n, d in G.nodes(data=True):
        if G.degree(n) != 1:
            continue
        direction, _ = end_direction(G, n, cfg.px(cfg.passage_back))
        if direction is None or (np.degrees(np.arctan2(direction[1], direction[0])) - dom + tol) % 90 > 2 * tol:
            continue
        (_, _, e), = [(u, v, dd) for u, v, dd in G.edges(n, data=True)]
        ends.append((d["xy"].astype(float), direction, float(e.get("thickness", 0.2))))
    out = []
    reach = cfg.px(cfg.passage_dedupe)
    for p_end, direction, thickness in ends:
        if not big[wcc[at(p_end)]]:
            continue
        pos, steps = p_end.copy(), 0
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
        # the other side must be a wall end pointing back, not the flank of a wall
        facing = [q for q, dq, _ in ends if np.linalg.norm(q - pos) < reach + thickness * cfg.m and float(np.dot(dq, direction)) < -0.9]
        if not facing:
            continue
        mid = (start + pos) / 2
        if any(np.linalg.norm(o["centre"] - mid) < reach for o in out):
            continue
        out.append({"kind": "passage", "centre": mid, "along": direction.copy(), "width": free / cfg.m,
                    "depth": max(thickness, 0.1), "line": (start, pos.copy()),
                    "poly": LineString([start, pos]).buffer(1.5, cap_style="flat"), "score": 0.5, "source": "wall gap"})
    return out
