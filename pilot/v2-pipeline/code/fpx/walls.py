"""Stage 3b, walls and the wall graph: clean masks, skeleton, gaps bridged where the sheet shows a wall, centre lines
with thickness."""
import cv2
import networkx as nx
import numpy as np
from scipy import ndimage
from shapely.geometry import LineString

from shapely.geometry import Point
from shapely.ops import unary_union

from .config import DEFAULT
from .geometry import clean, dominant_angle, mask_polys
from .model import COLUMN, DOOR, STAIRS, WALL, WINDOW


def skeleton_graph(skel):
    """Skeleton pixels -> graph whose nodes are junction clusters and end points; edges carry pixel paths."""
    pts = [tuple(p) for p in np.argwhere(skel)]
    on = set(pts)
    nb = lambda p: [(p[0] + dy, p[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx) and (p[0] + dy, p[1] + dx) in on]
    deg = {p: len(nb(p)) for p in pts}
    junction = {p for p in pts if deg[p] >= 3}
    node_of, G = {}, nx.MultiGraph()
    for p in pts:                                       # junction clusters become single nodes
        if p in junction and p not in node_of:
            stack, comp = [p], []
            node_of[p] = len(G)
            while stack:
                q = stack.pop()
                comp.append(q)
                for r in nb(q):
                    if r in junction and r not in node_of:
                        node_of[r] = node_of[p]
                        stack.append(r)
            G.add_node(node_of[p], xy=np.mean(comp, 0)[::-1], kind="junction")
    for p in pts:
        if deg[p] <= 1 and p not in node_of:
            node_of[p] = len(G)
            G.add_node(node_of[p], xy=np.array(p[::-1], float), kind="end")
    seen = set()
    for p, n in list(node_of.items()):
        for q in nb(p):
            if q in node_of and node_of[q] == n:
                continue
            path, prev, cur = [p, q], p, q
            while cur not in node_of:
                nxt = [r for r in nb(cur) if r != prev and r not in path[-3:]]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
            if cur not in node_of:
                continue
            key = frozenset([(p, q), (path[-1], path[-2])])
            if key in seen:
                continue
            seen.add(key)
            G.add_edge(n, node_of[cur], path=np.array([r[::-1] for r in path], float))
    return G


def wall_graph(wall, dist, cfg=DEFAULT):
    """Skeleton graph of the wall mask with short spurs pruned (skeleton artefacts at wall ends)."""
    from skimage.morphology import skeletonize           # scikit-image loads only here (3 s)
    G = skeleton_graph(skeletonize(wall))
    for _ in range(2):
        for u, v, k, d in list(G.edges(keys=True, data=True)):
            if (G.degree(u) == 1 or G.degree(v) == 1) and len(d["path"]) < 1.5 * max(dist[int(d["path"][0][1]), int(d["path"][0][0])], 4):
                if G.degree(u) + G.degree(v) > 2:
                    G.remove_edge(u, v, k)
        G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    contract_degree2(G)
    return G


def contract_degree2(G):
    """Merge the two edges of every degree-2 node (a junction whose spur was pruned) into one, so that a straight wall
    is one segment; loops through a single node are left alone."""
    for n in [n for n in list(G.nodes) if G.degree(n) == 2 and G.nodes[n]["kind"] == "junction"]:
        edges = list(G.edges(n, keys=True, data=True))
        if len(edges) != 2:
            continue
        (_, u, ku, du), (_, v, kv, dv) = edges
        if u == n or v == n or u == v:
            continue
        pu, pv = du["path"], dv["path"]
        xy = G.nodes[n]["xy"]
        pu = pu if np.linalg.norm(pu[-1] - xy) <= np.linalg.norm(pu[0] - xy) else pu[::-1]    # ... -> n
        pv = pv if np.linalg.norm(pv[0] - xy) <= np.linalg.norm(pv[-1] - xy) else pv[::-1]    # n -> ...
        G.remove_edge(n, u, ku)
        G.remove_edge(n, v, kv)
        G.remove_node(n)
        G.add_edge(u, v, path=np.vstack([pu, pv[1:]]))


def end_direction(G, n, back_px):
    """Unit direction in which the wall ending at degree-1 node n points outward, measured over back_px of its path."""
    (_, _, e), = [(u, v, dd) for u, v, dd in G.edges(n, data=True)]
    path, p_end = e["path"], G.nodes[n]["xy"]
    if np.linalg.norm(path[0] - p_end) > np.linalg.norm(path[-1] - p_end):
        path = path[::-1]
    back = min(len(path) - 1, int(back_px))
    if back < 4:
        return None, path
    d = path[0] - path[back]
    return d / max(np.linalg.norm(d), 1e-6), path


def bridge_gaps(sheet, G, wall, dist, cfg=DEFAULT):
    """Missed wall pieces: from each wall end cast a ray along the wall across free pixels to the next wall within
    gap_bridge_max, and accept it when the sheet shows a wall along it: a band of the wall's own half thickness on
    either side of the ray holds ink at gap_bridge_ink of its length or more (outlined, hatched and solid walls all
    leave ink in the band, an open passage leaves none, a door only its two jambs). Rays that meet a door or window
    label stop: that gap is an opening. Returns [(start, end, half thickness px)] in working pixels."""
    lab = sheet.label
    H, W = lab.shape
    ink = cv2.cvtColor(sheet.img, cv2.COLOR_RGB2GRAY) < 128
    opening = np.isin(lab, (DOOR, WINDOW))
    at = lambda p: (int(np.clip(p[1], 0, H - 1)), int(np.clip(p[0], 0, W - 1)))
    lo, hi = cfg.px(cfg.gap_bridge_min), cfg.px(cfg.gap_bridge_max)
    out = []
    for n in [n for n in G.nodes if G.degree(n) == 1]:
        direction, path = end_direction(G, n, cfg.px(cfg.passage_back))
        if direction is None:
            continue
        p_end = G.nodes[n]["xy"]
        half = max(float(dist[at(p_end)]), 1.5)
        pos, steps = p_end.astype(float).copy(), 0
        while steps < cfg.px(cfg.passage_exit_max) and wall[at(pos)]:    # leave the own wall body
            pos += direction
            steps += 1
        if wall[at(pos)]:
            continue
        start, free = pos.copy(), 0
        while free < hi:
            y, x = at(pos)
            if opening[y, x] or not (0 < x < W - 1 and 0 < y < H - 1):
                free = None
                break
            if wall[y, x]:
                break
            pos += direction
            free += 1
        if free is None or not lo <= free < hi or not wall[at(pos)]:
            continue
        nrm = np.array([-direction[1], direction[0]])
        t = np.arange(free)
        offs = np.arange(-int(half), int(half) + 1)
        px = start[0] + np.outer(t, direction[0]) + np.outer(np.ones_like(t), offs * nrm[0])
        py = start[1] + np.outer(t, direction[1]) + np.outer(np.ones_like(t), offs * nrm[1])
        band = ink[np.clip(py.round().astype(int), 0, H - 1), np.clip(px.round().astype(int), 0, W - 1)]
        if band.any(1).mean() < cfg.gap_bridge_ink:
            continue
        mid = (start + pos) / 2
        if any(np.linalg.norm(mid - (a + b) / 2) < h + 2 for a, b, h in out):     # the same gap seen from its other end
            continue
        out.append((start, pos.copy(), half))
    return out


def reject_pieces(sheet, wall, cfg=DEFAULT):
    """Wall-labelled pieces that are no wall, with the reason: the thin outline of a fixture next to its label
    (Kachelofen, Cheminée), a railing or stringer surrounded by stair treads, an isolated single stroke (dimension
    line, hatching, leader), lettering read as wall (model v2+ text head). Returns (wall mask, rejects)."""
    lab = sheet.label
    m = cfg.m
    rejects = []
    simplify = cfg.px(cfg.poly_simplify)

    def record(mask, reason):
        for p in mask_polys(mask, 1, simplify=cfg.px(cfg.contour_simplify)):
            rejects.append({"reason": reason, "area": round(cfg.m2(p.area), 2), "poly": p.simplify(simplify)})

    boxes = [t["box"] for t in sheet.text if t.get("role") == "fixture label"]
    if boxes:                                              # fixture outlines: thin wall pixels around the label
        near = np.zeros(wall.shape, np.uint8)
        r = int(cfg.px(cfg.fixture_reach))
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(near, (int(x0) - r, int(y0) - r), (int(x1) + r, int(y1) + r), 1, -1)
        thin = (2 * ndimage.distance_transform_edt(wall) - 1) < cfg.px(cfg.fixture_outline_max)
        removed = wall & (near > 0) & thin
        if removed.any():
            record(removed, "fixture outline")
            wall = wall & ~removed
    n, cc, stats, _ = cv2.connectedComponentsWithStats(wall.astype(np.uint8), connectivity=8)
    if n <= 1:
        return wall, rejects
    dist = ndimage.distance_transform_edt(wall)
    typical = (2 * float(np.median(dist[wall])) - 1) / m if wall.any() else 0.0      # the sheet's own wall thickness
    stair = cv2.dilate((lab == STAIRS).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    text_p = sheet.text_prob
    keep = np.ones(n, bool)
    H, W = wall.shape
    for i, sl in enumerate(ndimage.find_objects(cc), 1):
        if sl is None:
            continue
        ys, xs = sl
        sl2 = (slice(max(ys.start - 3, 0), min(ys.stop + 3, H)), slice(max(xs.start - 3, 0), min(xs.stop + 3, W)))
        m_ = cc[sl2] == i
        th = (2 * float(np.median(dist[sl2][m_])) - 1) / m
        ring = (cv2.dilate(m_.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~m_
        other = bool((ring & wall[sl2] & (cc[sl2] != i)).any())
        stair_share = float((ring & stair[sl2]).sum() / max(ring.sum(), 1))
        text_share = float(text_p[sl2][m_].mean()) if text_p is not None else 0.0
        if text_share >= cfg.wall_text_prob:
            reason = "text"
        elif not other and stair_share >= cfg.stair_ring_share and th < cfg.railing_max_thickness:
            reason = "stair railing"
        elif not other and th < min(cfg.wall_min_thickness, typical / 2) and stats[i, cv2.CC_STAT_AREA] < cfg.px2(cfg.stroke_max_area):
            reason = "thin stroke"
        else:
            continue
        keep[i] = False
        full = np.zeros(wall.shape, bool)
        full[sl2] = m_
        record(full, reason)
    keep[0] = False
    return keep[cc], rejects


def railing_edges(G, lab, dist, cfg=DEFAULT):
    """Graph edges with stair treads on both sides of the centre line for most of their length: railings, stringers
    and tread lines read as wall, also where they join a real wall. Returns the edges (u, v, key)."""
    stair = cv2.dilate((lab == STAIRS).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    H, W = lab.shape
    out = []
    for u, v, k, d in G.edges(keys=True, data=True):
        path = d["path"]
        if len(path) < 6:
            continue
        step = max(1, len(path) // 12)
        hits = tot = 0
        for i in range(step, len(path) - step, step):
            t = path[i + step] - path[i - step]
            n = np.linalg.norm(t)
            if n < 1e-6:
                continue
            nrm = np.array([-t[1], t[0]]) / n
            off = float(dist[int(path[i][1]), int(path[i][0])]) + 3
            sides = [stair[int(np.clip(path[i][1] + s * nrm[1] * off, 0, H - 1)), int(np.clip(path[i][0] + s * nrm[0] * off, 0, W - 1))] for s in (1, -1)]
            hits += all(sides)
            tot += 1
        if tot and hits / tot >= cfg.stair_ring_share:
            th = 2 * float(np.median(dist[path[:, 1].astype(int), path[:, 0].astype(int)])) - 1
            if th / cfg.m < cfg.railing_max_thickness:
                out.append((u, v, k))
    return out


def _snap(angle, dom, tol):
    """Angle snapped to the nearest dominant direction (dom + k * 90) when within tol degrees, else unchanged."""
    best = min((dom + k * 90 for k in range(-2, 4)), key=lambda c: abs(((angle - c) + 90) % 180 - 90))
    return best if abs(((angle - best) + 90) % 180 - 90) <= tol else angle


def regularise(G, dist, cfg=DEFAULT):
    """Straight wall segments with thickness from the skeleton graph: every edge path is simplified at a share of its
    wall's thickness, cut at the vertices, each piece snapped to the plan's dominant directions when within
    wall_snap_angle, junction stubs shorter than the thickness dropped, and the end pieces extended to the junction
    they belong to. Returns [{"id", "line", "thickness", "angle", "nodes"}] in working pixels."""
    m = cfg.m
    dom = dominant_angle([{"line": LineString(d["path"])} for _, _, d in G.edges(data=True) if len(d["path"]) > 1])
    segs = []
    for u, v, d in G.edges(data=True):
        path = d["path"]
        if len(path) < 2:
            continue
        th_px = max(2 * float(np.median(dist[path[:, 1].astype(int), path[:, 0].astype(int)])) - 1, 1.0)
        tol = max(cfg.px(cfg.line_simplify), th_px * cfg.wall_simplify_share)
        pts = np.asarray(LineString(path).simplify(tol).coords)
        pieces = []
        for a, b in zip(pts[:-1], pts[1:]):
            L = float(np.linalg.norm(b - a))
            if L < 1e-6:
                continue
            # a short piece of a thick wall has an uncertain direction (the skeleton zigzags through niches and jambs):
            # its snap tolerance grows with atan(thickness / length)
            tol_deg = max(cfg.wall_snap_angle, float(np.degrees(np.arctan2(th_px, L))))
            ang = _snap(float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))), dom, tol_deg)
            if pieces and abs(((pieces[-1][2] - ang) + 90) % 180 - 90) < 1.0:   # collinear with the piece before: extend it
                pieces[-1][1], pieces[-1][3] = b, pieces[-1][3] + L
            else:
                pieces.append([a.copy(), b.copy(), ang, L])
        long = [p for p in pieces if p[3] >= max(th_px, 3.0)]                 # junction stubs are shorter than the wall is thick
        pieces = long or pieces[:1]
        if not pieces:
            continue
        ends = (G.nodes[u]["xy"], G.nodes[v]["xy"])
        first = ends[0] if np.linalg.norm(ends[0] - path[0]) <= np.linalg.norm(ends[1] - path[0]) else ends[1]
        last = ends[1] if first is ends[0] else ends[0]
        for k, (a, b, ang, L) in enumerate(pieces):
            mid = (a + b) / 2
            dv = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
            a2, b2 = mid - dv * L / 2, mid + dv * L / 2
            if k == 0:                                     # reach the junction: the foot of the node on the piece's line
                a2 = mid + dv * float(np.dot(first - mid, dv))
            if k == len(pieces) - 1:
                b2 = mid + dv * float(np.dot(last - mid, dv))
            if np.linalg.norm(b2 - a2) < 1.0:
                continue
            segs.append({"id": f"w{len(segs):03d}", "line": LineString([a2, b2]), "thickness": round(th_px / m, 3),
                         "angle": round(ang % 180, 1), "nodes": (u, v)})
    return segs


def segment_polys(G, segs, cfg=DEFAULT):
    """Wall polygons of the regularised segments: each centre line buffered by half its thickness (flat ends), plus
    a square of the thickest incident wall at every junction so that corners close."""
    parts = [s["line"].buffer(cfg.px(s["thickness"]) / 2, cap_style="flat", join_style="mitre") for s in segs]
    by_node = {}
    for s in segs:
        for n in s["nodes"]:
            by_node[n] = max(by_node.get(n, 0.0), s["thickness"])
    for n, t in by_node.items():
        if G.degree(n) >= 2:
            x, y = G.nodes[n]["xy"]
            parts.append(Point(x, y).buffer(cfg.px(t) / 2, cap_style="square"))
    u = unary_union(parts)
    return [p.simplify(cfg.px(cfg.poly_simplify)) for p in getattr(u, "geoms", [u])
            if p.geom_type == "Polygon" and p.area >= cfg.px2(cfg.wall_min_area)]


def walls(sheet, cfg=DEFAULT):
    m = cfg.m
    lab = sheet.label
    wall = clean(lab == WALL, cfg.px2(cfg.wall_min_area))
    column = clean(lab == COLUMN, cfg.px2(cfg.column_min_area))
    wall, rejects = reject_pieces(sheet, wall, cfg)
    sheet.wall_rejects = rejects
    lab[(lab == WALL) & ~wall] = 0                       # rejected pieces are background from here on
    dist = ndimage.distance_transform_edt(wall)
    G = wall_graph(wall, dist, cfg)
    rails = railing_edges(G, lab, dist, cfg)
    if rails:                                           # erase them from the mask and read the graph again
        w8 = wall.astype(np.uint8)
        erased = np.zeros(wall.shape, np.uint8)
        for u, v, k in rails:
            path = G.edges[u, v, k]["path"]
            th = int(2 * float(np.median(dist[path[:, 1].astype(int), path[:, 0].astype(int)])) + 2)
            cv2.polylines(erased, [np.round(path).astype(np.int32)], False, 1, max(th, 3))
        erased = (erased > 0) & wall
        for p in mask_polys(erased, 1, simplify=cfg.px(cfg.contour_simplify)):
            sheet.wall_rejects.append({"reason": "stair railing", "area": round(cfg.m2(p.area), 2), "poly": p.simplify(cfg.px(cfg.poly_simplify))})
        wall = wall & ~erased
        lab[erased] = 0
        dist = ndimage.distance_transform_edt(wall)
        G = wall_graph(wall, dist, cfg)
    bridges = bridge_gaps(sheet, G, wall, dist, cfg)
    if bridges:                                         # draw the missed pieces and read the graph again
        w8 = wall.astype(np.uint8)
        for a, b, half in bridges:
            cv2.line(w8, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b), 1, max(int(2 * half), 3))
        wall = (w8 > 0) & ~np.isin(lab, (DOOR, WINDOW, COLUMN))     # a bridge never covers an opening or a column
        lab[wall] = WALL
        dist = ndimage.distance_transform_edt(wall)
        G = wall_graph(wall, dist, cfg)
    sheet.wall_mask, sheet.column_mask = wall, column
    sheet.wall_bridges = [{"line": LineString([a, b]), "length": round(float(np.linalg.norm(b - a)) / m, 2),
                           "thickness": round(2 * half / m, 3)} for a, b, half in bridges]
    segs = [s for s in regularise(G, dist, cfg) if s["line"].length >= cfg.px(cfg.wall_segment_min_length)]
    sheet.wall_graph, sheet.wall_segments = G, segs
    sheet.wall_polys_raw = [p.simplify(cfg.px(cfg.poly_simplify)) for p in mask_polys(wall, cfg.px2(cfg.wall_min_area))]
    sheet.wall_polys = segment_polys(G, segs, cfg) or sheet.wall_polys_raw
    return segs
