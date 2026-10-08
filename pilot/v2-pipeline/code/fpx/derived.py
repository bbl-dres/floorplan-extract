"""Stage 8, derived outputs: net and gross room areas, wall kind and construction, GF and AGF, exterior doors and
windows, EBF proposal and room connectivity."""
import cv2
import networkx as nx
import numpy as np
from scipy import ndimage
from shapely.geometry import Point, Polygon

from .config import DEFAULT
from .geometry import mask_polys
from .model import DOOR, WINDOW

CONNECTING = ("door", "exterior door", "passage")   # opening kinds that connect two spaces; windows and interior openings do not


def gross_partition(sheet, cfg=DEFAULT):
    """Gross room polygons: every wall, column, door and window pixel goes to the nearest room, so that neighbouring
    rooms meet at the wall centre line and an exterior wall is shared between the room and the outside (its outer
    half belongs to nobody). Free space that is not a room (outside, border regions) competes like a room. Sets
    poly_gross on every room and fragment (working pixels) and sheet.gross_label (the owner per pixel, 0 = nobody)."""
    lab = sheet.label
    band = sheet.wall_mask | sheet.column_mask | (lab == DOOR) | (lab == WINDOW)
    regions = sheet.rooms + sheet.fragments
    labels = {r["px_label"] for r in regions}
    owner = np.where(np.isin(sheet.room_label, list(labels)), sheet.room_label, 0).astype(np.int32)
    outside = int(sheet.room_label.max()) + 1
    owner[(owner == 0) & ~band] = outside
    ys, xs = np.nonzero(band)
    if len(ys):                                         # the distance transform runs on the drawing's box only
        y0, y1 = max(ys.min() - 2, 0), min(ys.max() + 3, lab.shape[0])
        x0, x1 = max(xs.min() - 2, 0), min(xs.max() + 3, lab.shape[1])
        sub, b = owner[y0:y1, x0:x1], band[y0:y1, x0:x1]
        iy, ix = ndimage.distance_transform_edt(b, return_distances=False, return_indices=True)
        sub[b] = sub[iy, ix][b]
    owner[owner == outside] = 0
    sheet.gross_label = owner
    by_label = {r["px_label"]: r for r in regions}
    for i, sl in enumerate(ndimage.find_objects(owner), 1):
        r = by_label.get(i)
        if sl is None or r is None:
            continue
        ps = mask_polys(owner[sl] == i, 1, simplify=cfg.px(cfg.contour_simplify))
        if ps:
            p = max(ps, key=lambda q: q.area).simplify(cfg.px(cfg.poly_simplify))
            r["poly_gross"] = Polygon(np.asarray(p.exterior.coords) + (sl[1].start, sl[0].start),
                                      [np.asarray(h.coords) + (sl[1].start, sl[0].start) for h in p.interiors])
    for r in regions:
        r.setdefault("poly_gross", r["poly"])


def wall_kind(sheet, cfg=DEFAULT):
    """Exterior or interior per wall segment, from what lies on either side of its centre line (the gross partition:
    a room label, or 0 for the outside and free-standing pieces); unknown when neither side is a room."""
    H, W = sheet.gross_label.shape
    for s in sheet.wall_segments:
        line, half = s["line"], cfg.px(s["thickness"]) / 2 + 3
        n = max(3, min(9, int(line.length / cfg.px(0.5))))
        rooms_a = rooms_b = outside = 0
        for k in range(n):
            d = (k + 0.5) / n
            p = np.asarray(line.interpolate(d, normalized=True).coords[0])
            q = np.asarray(line.interpolate(min(d + 0.01, 1.0), normalized=True).coords[0])
            t = q - p
            t /= max(np.linalg.norm(t), 1e-6)
            nrm = np.array([-t[1], t[0]])
            sides = []
            for sgn in (1, -1):
                x, y = p + sgn * nrm * half
                sides.append(int(sheet.gross_label[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))]))
            rooms_a += sides[0] > 0
            rooms_b += sides[1] > 0
            outside += (sides[0] > 0) != (sides[1] > 0)
        if rooms_a + rooms_b == 0:
            s["kind"] = "unknown"
        else:
            s["kind"] = "exterior" if outside >= n / 2 else "interior"
    return sheet.wall_segments


# ---------------------------------------------------------------------------------------------------------------------
# Massive or lightweight walls (a heuristic; the DXF hatches massive walls on A_SCHRAFFUR)

def _construction(t, fill, cfg):
    if t >= cfg.massive_wall_min_thickness:
        return "massive", f"heuristic: {t:.2f} m thick (massive from {cfg.massive_wall_min_thickness} m)"
    if fill is not None and fill >= cfg.solid_fill_share and t >= cfg.massive_solid_min_thickness:
        return "massive", f"heuristic: drawn solid ({fill:.0%} ink in the wall core) and {t:.2f} m thick"
    seen = "core too thin to see a fill" if fill is None else f"{fill:.0%} ink in the wall core, not drawn solid"
    return "lightweight", f"heuristic: {t:.2f} m thick (massive from {cfg.massive_wall_min_thickness} m), {seen}"


def wall_construction(sheet, cfg=DEFAULT):
    """Massive or lightweight per wall segment. A heuristic: the guideline hatches massive walls only (A_SCHRAFFUR),
    and a raster plan rarely says what a wall is made of.

    Every wall pixel belongs to the nearest centre line in its wall piece, so each segment owns part of the wall mask.
    A segment is massive if (1) it is at least massive_wall_min_thickness thick (thickness from the wall graph), or (2)
    its core, the pixels at least two pixels inside the wall faces, is drawn solid (solid_fill_share or more dark ink)
    and it is at least massive_solid_min_thickness thick: on plans that fill walls, poché marks masonry and concrete.
    Otherwise lightweight. Wall pieces without a centre line use their largest inscribed width.

    Sets construction, construction_basis and fill_share on each wall segment; returns the massive wall polygons
    (working pixels) and a summary, also kept as sheet.massive_walls."""
    wall = sheet.wall_mask
    segs = {s["id"]: s for s in sheet.wall_segments}
    if wall is None or not wall.any():
        sheet.massive_walls = ([], {"method": "heuristic", "massive_segments": 0, "lightweight_segments": 0})
        return sheet.massive_walls
    dark = cv2.cvtColor(sheet.img, cv2.COLOR_RGB2GRAY) < 128
    depth = ndimage.distance_transform_edt(wall)
    core = depth >= 2.5
    skel, order = np.zeros(wall.shape, np.int32), []
    for s in sheet.wall_segments:                          # every wall pixel belongs to the nearest segment's centre line
        order.append(s)
        pts = np.round(np.asarray(s["line"].coords)).astype(np.int32)
        cv2.polylines(skel, [pts], False, len(order), 1)
    skel[~wall] = 0
    _, comp = cv2.connectedComponents(wall.astype(np.uint8), connectivity=8)
    owner, massive, loose = np.zeros(wall.shape, np.int32), np.zeros(wall.shape, bool), 0
    for c, sl in enumerate(ndimage.find_objects(comp), 1):
        if sl is None:
            continue
        mc = comp[sl] == c
        sk = np.where(mc, skel[sl], 0)
        if sk.any():
            _, (iy, ix) = ndimage.distance_transform_edt(sk == 0, return_indices=True)
            owner[sl][mc] = sk[iy, ix][mc]
        else:                                          # a wall piece without centre line: classified on its own
            cm = core[sl] & mc
            fill = float(dark[sl][cm].mean()) if cm.sum() >= 10 else None
            if _construction(2 * float(depth[sl][mc].max()) / cfg.m, fill, cfg)[0] == "massive":
                massive[sl] |= mc
                loose += 1
    n = len(order) + 1
    core_n = np.bincount(owner[core], minlength=n)
    core_dark = np.bincount(owner[core], weights=dark[core], minlength=n)
    heavy = np.zeros(n, bool)
    for k, s in enumerate(order, 1):
        fill = float(core_dark[k] / core_n[k]) if core_n[k] >= 10 else None
        s["construction"], s["construction_basis"] = _construction(s["thickness"], fill, cfg)
        s["fill_share"] = None if fill is None else round(fill, 2)
        heavy[k] = s["construction"] == "massive"
    massive |= heavy[owner] & wall
    polys = [p.simplify(cfg.px(cfg.poly_simplify)) for p in mask_polys(massive, cfg.px2(cfg.wall_min_area))]
    summary = {"method": "heuristic", "rule": f"massive if at least {cfg.massive_wall_min_thickness} m thick, or drawn solid "
                                             f"(>= {cfg.solid_fill_share:.0%} ink in the wall core) and at least "
                                             f"{cfg.massive_solid_min_thickness} m thick; otherwise lightweight",
               "massive_segments": int(heavy.sum()), "lightweight_segments": len(order) - int(heavy.sum()),
               "massive_pieces_without_centre_line": loose,
               "massive_area": round(cfg.m2(float(massive.sum())), 2), "wall_area": round(cfg.m2(float(wall.sum())), 2),
               "note": "thickness and fill are measured on the sheet; the wall material is not known. Check before use."}
    sheet.massive_walls = (polys, summary)
    return sheet.massive_walls


# ---------------------------------------------------------------------------------------------------------------------

def derived(sheet, cfg=DEFAULT):
    m = cfg.m
    area = lambda g: cfg.m2(g.area)
    # The GF polygon keeps balconies, loggias and terraces: BBL's 2005 reference polygon of Landgut Lohn includes the
    # balcony (SIA 416 would count it as AGF). Their area is reported apart as agf_area; open question in the review
    outdoor = [r for r in sheet.rooms if r["usage"] == "outdoor"]
    sheet.gf_area = area(sheet.gf)
    sheet.agf_area = round(sum(area(r["poly"]) for r in outdoor), 2)
    gross_partition(sheet, cfg)
    for r in sheet.rooms + sheet.fragments:             # three areas side by side: net, gross (stage 8) and stamp (stage 7)
        voids = sum(area(v["poly"]) for v in r.get("voids", []))
        r["area_polygon"] = round(area(r["poly"]), 2)
        r["area_net"] = round(area(r["poly"]) - voids, 2)
        r["area_gross"] = round(area(r["poly_gross"]) - voids, 2)
    wall_kind(sheet, cfg)
    wall_construction(sheet, cfg)
    # exterior = one side outside the floor outline (the outline without its voids: a probe in a stair eye is inside)
    inner = Polygon(sheet.gf.exterior).buffer(-cfg.px(cfg.gf_inner_buffer)) if not sheet.gf.is_empty else Polygon()
    lo, hi = cfg.interior_window_door_width
    for o in sheet.openings:
        nrm = np.array([-o["along"][1], o["along"][0]])
        pts = [Point(o["centre"] + s * nrm * (o["depth"] / 2 + cfg.exterior_probe) * m) for s in (1, -1)]
        o["exterior"] = not all(inner.contains(q) for q in pts)
        if o["kind"] == "window" and not o["exterior"]:
            if lo < hi and lo <= o["width"] <= hi:     # the swing of a door and a casement window look alike: in an
                o["kind"], o["confidence"] = "door", "low"   # interior wall at door width it is a door (CubiCasa: hardly any interior windows)
                o["flag"] = "drawn like a window in an interior wall, at door width: treated as a door"
            else:
                o["kind"], o["flag"] = "interior opening", "drawn like a window in an interior wall: treated as an opening"
        if o["kind"] == "door" and o["exterior"]:
            o["kind"] = "exterior door"
    ebf_excluded = [r for r in sheet.rooms if r["usage"] == "outdoor"]
    sheet.ebf_proposal = {"area": round(sheet.gf_area, 2), "excluded": [r["id"] for r in ebf_excluded],
                          "status": "proposal, to be confirmed"}
    H, W = sheet.room_label.shape
    pid = {r["px_label"]: r["id"] for r in sheet.rooms}

    def probe(p):
        """Room at a probe point: the centre pixel first, else the most frequent room label in a 7 x 7 window."""
        x, y = int(np.clip(p[0], 0, W - 1)), int(np.clip(p[1], 0, H - 1))
        v = int(sheet.room_label[y, x])
        if v in pid:
            return pid[v]
        lbl = sheet.room_label[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        vals = [int(u) for u in np.unique(lbl) if u in pid]
        if vals:
            return pid[max(vals, key=lambda u: (lbl == u).sum())]
        return "outside" if not sheet.building_rough[y, x] else None

    edges = []
    for o in sheet.openings:
        if o["kind"] == "window":
            continue
        nrm = np.array([-o["along"][1], o["along"][0]])
        ends = None
        for axis, half in ((nrm, o["depth"] / 2), (o["along"], o["width"] / 2)):
            for off in cfg.connect_probe:
                e = [probe(o["centre"] + s * axis * (half + off) * m) for s in (1, -1)]
                if all(e) and e[0] != e[1]:
                    ends = e
                    break
            if ends:
                break
        ends = ends or [probe(p) for p in o["sides_px"]]
        o["connects"] = ends
        if o["kind"] in CONNECTING and all(ends) and ends[0] != ends[1]:
            edges.append((ends[0], ends[1], o["id"], o["kind"], o.get("confidence", "medium")))
    g = nx.Graph()
    g.add_nodes_from(r["id"] for r in sheet.rooms)
    g.add_edges_from((a, b, {"opening": o, "type": k, "confidence": c}) for a, b, o, k, c in edges)
    sheet.connectivity = g
    return g
