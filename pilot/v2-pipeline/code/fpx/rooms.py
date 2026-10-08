"""Stage 6, rooms: free space between walls and closed openings, stair rooms where stamps demand them, sealing, voids
and the GF outline."""
import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon

from .attributes import cluster_stamps, usage
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


def regions(cc):
    """(label, slice, mask within the slice) of every labelled region, without a full-image comparison per label."""
    for i, sl in enumerate(ndimage.find_objects(cc), 1):
        if sl is not None:
            yield i, sl, cc[sl] == i


def at_border(sl, m, shape):
    """touches_border() of a region given by its slice and mask within it."""
    H, W = shape
    ys, xs = sl
    return bool((ys.start < 2 and m[:2 - ys.start].any()) or (ys.stop > H - 2 and m[H - 2 - ys.start:].any())
                or (xs.start < 2 and m[:, :2 - xs.start].any()) or (xs.stop > W - 2 and m[:, W - 2 - xs.start:].any()))


def stair_rooms(cc, nxt, hull, stamps, barrier, cfg=DEFAULT):
    """Cut a stair room out of a region along the stair outline where the stamps demand one: a stair stamp inside the
    outline, or stamps on both sides. Otherwise a stair is an object inside its room (a stair open to a hall belongs to
    the hall), so that neither a staircase nor a corridor is cut in two by the outline. Without a stamp on the stair
    itself (scans whose lettering OCR cannot read) a stair is cut only where it is open to a hall: the rest of the region is
    stair_hall_ratio times the outline part and the outline is walled on stair_enclosure of its boundary. Both parts
    must reach stair_split_min_area. Returns the next free label."""
    split = cfg.px2(cfg.stair_split_min_area)
    pts = [(int((s["box"][1] + s["box"][3]) / 2), int((s["box"][0] + s["box"][2]) / 2), usage(s["name"] or "") == "stair")
           for s in stamps]
    for i, sl, m in list(regions(cc)):
        h = hull[sl]
        part, rest = m & h, m & ~h
        if part.sum() <= split or rest.sum() <= split:
            continue
        inside = outside = stair = False
        for y, x, is_stair in pts:
            yi, xi = y - sl[0].start, x - sl[1].start
            if 0 <= yi < m.shape[0] and 0 <= xi < m.shape[1] and m[yi, xi]:
                inside, outside = inside or h[yi, xi], outside or not h[yi, xi]
                stair = stair or (is_stair and h[yi, xi])
        cut = stair or (inside and outside)
        if not cut and not inside and rest.sum() >= cfg.stair_hall_ratio * part.sum():   # no stamp on the stair itself
            ring = cv2.dilate(part.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            ring &= ~part
            walled = (ring & (barrier[sl] > 0)).sum() / max(ring.sum(), 1)
            cut = walled >= cfg.stair_enclosure
        if cut:
            cc[sl][part] = nxt
            nxt += 1
    return nxt


def voids(sheet, cc, out, hull, free_raw, cfg=DEFAULT):
    """Voids from void labels (Luftraum, Vide): a label inside a stair outline marks the stair eye, the free space
    between the flights (kind "stair eye"); a label elsewhere marks the whole free region it stands in as an air space
    over the storey below (kind "air space"), which is then no room. Measured on the free space of the unclosed
    barrier within the outline."""
    simplify = cfg.px(cfg.poly_simplify)
    holes = (hull & free_raw & ~sheet.stair_mask).astype(np.uint8)
    hn, hcc = cv2.connectedComponents(holes, connectivity=4)
    by_label = {r["px_label"]: r for r in out}
    for t in sheet.text:
        if t["role"] != "void label":
            continue
        x, y = int((t["box"][0] + t["box"][2]) / 2), int((t["box"][1] + t["box"][3]) / 2)
        y, x = min(y, cc.shape[0] - 1), min(x, cc.shape[1] - 1)
        i = int(hcc[y, x])
        if i and hull[y, x]:
            if any(v["px"] == i and v["kind"] == "stair eye" for v in sheet.voids):
                continue
            ps = mask_polys(hcc == i, cfg.px2(cfg.void_min_area), simplify=cfg.px(cfg.contour_simplify))
            if ps:
                sheet.voids.append({"id": f"v{len(sheet.voids):03d}", "kind": "stair eye", "px": i, "label": t["text"],
                                    "poly": max(ps, key=lambda q: q.area).simplify(simplify)})
            continue
        r = by_label.get(int(cc[y, x]))                # a region labelled as a void is an air space, not a room
        if r is not None and not any(v["px"] == -r["px_label"] for v in sheet.voids):
            out.remove(r)
            sheet.voids.append({"id": f"v{len(sheet.voids):03d}", "kind": "air space", "px": -r["px_label"],
                                "label": t["text"], "poly": r["poly"]})
    sheet.rooms = out
    return sheet.voids


def separations(sheet, cc, nxt, cfg=DEFAULT):
    """Open-plan areas from the v2 boundary head, which outlines every area also where no wall stands: inside a free
    region, pixels at separation_prob or more cut it into parts; the cut is kept when at least two parts reach
    separation_min_area, and the cut pixels and small parts join the nearest part. Where the sheet has stamps they
    arbitrate: a cut is kept only when two of its parts carry a stamp (a room without a stamp is no room to BBL).
    Returns the next free label."""
    bp = sheet.boundary_prob
    if bp is None or cfg.separation_prob >= 1.0:
        return nxt
    min_part = cfg.px2(cfg.separation_min_area)
    pts = [(int((s["box"][1] + s["box"][3]) / 2), int((s["box"][0] + s["box"][2]) / 2)) for s in sheet.stamps]
    for i, sl, m in list(regions(cc)):
        if m.sum() < 2 * min_part:
            continue
        sep = m & (bp[sl] >= cfg.separation_prob)
        if not sep.any():
            continue
        k, parts = cv2.connectedComponents((m & ~sep).astype(np.uint8), connectivity=4)
        sizes = np.bincount(parts.ravel(), minlength=k)
        big = [j for j in range(1, k) if sizes[j] >= min_part]
        if len(big) < 2:
            continue
        if pts:                                         # stamps arbitrate: on a stamped sheet a cut needs a stamp in two parts
            stamped = set()                             # (Landgut S1: the head cut the Vorplatz and the stair against their single stamps)
            for y, x in pts:
                yi, xi = y - sl[0].start, x - sl[1].start
                if 0 <= yi < m.shape[0] and 0 <= xi < m.shape[1] and parts[yi, xi] in big:
                    stamped.add(int(parts[yi, xi]))
            if len(stamped) < 2:
                continue
        keep = np.isin(parts, big)
        iy, ix = ndimage.distance_transform_edt(~keep, return_distances=False, return_indices=True)
        owner = parts[iy, ix]
        for j in big[1:]:                                # the first part keeps the region's label
            cc[sl][m & (owner == j)] = nxt
            nxt += 1
    return nxt


def enclosed(cc, barrier, cfg=DEFAULT):
    """Drop regions whose boundary is mostly not wall, door or window (a blob the interior head put on empty paper)."""
    for i, sl, m in list(regions(cc)):
        ring = (cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~m
        if ring.any() and (ring & (barrier[sl] > 0)).sum() / ring.sum() < cfg.building_enclosure:
            cc[sl][m] = 0
    return cc


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
    barrier_raw = barrier.copy()                       # voids are measured on the unclosed barrier (the closing eats 0.15 m per side)
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    building = rough_building(barrier | (lab == STAIRS), cfg, anchors=np.isin(lab, (DOOR, WINDOW)))
    if cfg.building_from_interior and sheet.interior_prob is not None:     # experiment: the v2 interior head
        k2 = int(cfg.px(cfg.building_close)) | 1
        head = cv2.morphologyEx((sheet.interior_prob > 0.5).astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2)))
        building = building | (ndimage.binary_fill_holes(head) > 0)
    free = building & (barrier == 0)
    n, cc = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    if cfg.building_from_interior and sheet.interior_prob is not None:
        enclosed(cc, barrier, cfg)
    sheet.stamps = cluster_stamps(sheet.text)          # stage 7 assigns them to rooms; here they decide stair rooms
    hull = sheet.stair_hull
    nxt = stair_rooms(cc, n, hull, sheet.stamps, barrier, cfg)
    # regions that leak to the sheet border through an unclosed opening are sealed, not dropped
    gray = cv2.cvtColor(sheet.img, cv2.COLOR_RGB2GRAY)
    for i, sl, m in list(regions(cc)):
        if m.sum() >= min_room and at_border(sl, m, cc.shape):
            full = cc == i
            cc[full] = 0
            for part in seal(full, gray, cfg):
                cc[part] = nxt
                nxt += 1
    # the remainder of a split region may fall apart (e.g. hall and room joined only through the stair): relabel
    for i, sl, m in list(regions(cc)):
        k, parts = cv2.connectedComponents(m.astype(np.uint8), connectivity=4)
        for j in range(2, k):
            cc[sl][parts == j] = nxt
            nxt += 1
    nxt = separations(sheet, cc, nxt, cfg)             # open-plan areas (model v2 boundary head; off by default)
    out = []
    for i, sl, m in regions(cc):
        if m.sum() < min_room or at_border(sl, m, cc.shape):
            continue                                   # too small, or touches the sheet border: outside the building
        ps = mask_polys(m, min_room, simplify=cfg.px(cfg.contour_simplify))
        if not ps:
            continue
        p = max(ps, key=lambda q: q.area).simplify(simplify)
        if p.area < min_room:
            continue                                   # the polygon itself must reach the minimum (plan-check POLY_004)
        p = Polygon(np.asarray(p.exterior.coords) + (sl[1].start, sl[0].start),
                    [np.asarray(h.coords) + (sl[1].start, sl[0].start) for h in p.interiors])
        stair_share = (m & hull[sl]).sum() / m.sum()
        out.append({"id": f"r{len(out):03d}", "poly": p, "px_label": i, "stair": stair_share > cfg.stair_room_share,
                    "stair_flights": []})
    sheet.room_label = cc
    sheet.rooms = out
    by_label = {r["px_label"]: r for r in out}          # stair flights are objects of the room they lie in
    for k, flight in enumerate(sheet.stairs):
        x, y = flight.representative_point().coords[0]
        r = by_label.get(int(cc[min(int(y), cc.shape[0] - 1), min(int(x), cc.shape[1] - 1)]))
        if r is not None:
            r["stair_flights"].append(k)
    voids(sheet, cc, out, hull, building & (barrier_raw == 0), cfg)
    # GF: outer contour of rooms, walls, openings and stairs, closed over small gaps
    solid = (barrier | (lab == STAIRS) | np.isin(cc, [r["px_label"] for r in out])).astype(np.uint8)
    g = int(cfg.px(cfg.gf_close))
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((g, g), np.uint8))
    solid = ndimage.binary_fill_holes(solid) & building
    gf = mask_polys(solid, cfg.px2(cfg.gf_min_area))
    sheet.gf = max(gf, key=lambda q: q.area).simplify(simplify) if gf else Polygon()
    # CAD-Richtlinie Kap. 5.9 (plan-check GPOLY): stair eyes over 5 m² and air spaces are cut out of the GF (a hole
    # here; the DXF writes the GF as one continuous polyline with the hole cut out through a zero-width bridge,
    # fpx.export.keyhole). The wording "Treppenaugen > 5 m², Lufträume" is read as: the threshold applies to stair eyes
    for v in sheet.voids:
        v["gf_deducted"] = v["kind"] == "air space" or cfg.m2(v["poly"].area) > cfg.void_gf_deduction
        if v["gf_deducted"]:
            gf = sheet.gf.difference(v["poly"])
            sheet.gf = max(getattr(gf, "geoms", [gf]), key=lambda q: q.area)
    return out
