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
        elif not other and th < min(cfg.wall_min_thickness, typical / 2) and                 np.hypot(ys.stop - ys.start, xs.stop - xs.start) / m <= cfg.stroke_max_length:
            reason = "thin stroke"                        # a longer thin isolated piece is a single-line partition wall
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
    """Angle snapped to the nearest dominant direction (dom + k * 90, on the full circle so the piece keeps pointing
    the way it runs) when within tol degrees, else unchanged."""
    diff = lambda a, b: abs(((a - b) + 180) % 360 - 180)
    best = min((dom + k * 90 for k in range(-2, 4)), key=lambda c: diff(angle, c))
    return best if diff(angle, best) <= tol else angle


def path_thickness(path, mask, dist, n=25):
    """Thickness of a wall along a skeleton path in pixels and the offset of the wall's centre from the path along the
    path's left normal: the median width of the wall mask's cross-section at up to n points, measured along the normal
    of the local path direction and capped by the inscribed disc there (2 x EDT + 1: a normal that runs into a
    crossing wall at a junction must not count), and the median of half the difference between the two sides (the
    skeleton of a wall of even width sits half a pixel off its centre). Points inside an opening band (no wall pixel)
    are skipped; with fewer than three measurable points the inscribed disc decides and the offset is 0. (2 x EDT - 1
    alone is a pixel thin on even widths.) Returns (thickness, offset)."""
    H, W = mask.shape
    idx = np.linspace(2, len(path) - 3, min(n, max(len(path) - 4, 1))).astype(int) if len(path) >= 5 else np.arange(len(path))
    widths, discs, offsets = [], [], []
    for i in idx:
        xi, yi = int(round(path[i][0])), int(round(path[i][1]))
        if not (0 <= xi < W and 0 <= yi < H) or not mask[yi, xi]:
            continue
        disc = 2 * float(dist[yi, xi]) + 1
        discs.append(disc)
        if i < 2 or i + 2 >= len(path):
            continue
        t = path[i + 2] - path[i - 2]
        L = np.hypot(*t)
        if L < 1e-6:
            continue
        nrm = np.array([-t[1], t[0]]) / L
        side = {}
        for sgn in (1, -1):
            side[sgn] = 0
            for s in range(1, 200):
                x, y = path[i] + sgn * nrm * s
                xj, yj = int(round(x)), int(round(y))
                if not (0 <= xj < W and 0 <= yj < H) or not mask[yj, xj]:
                    break
                side[sgn] += 1
        w = 1 + side[1] + side[-1]
        if w <= disc:                                      # a clean cross-section: its centre is trusted too
            offsets.append((side[1] - side[-1]) / 2)
        widths.append(min(float(w), disc))
    if len(widths) >= 3:
        return float(np.median(widths)), (float(np.median(offsets)) if len(offsets) >= 3 else 0.0)
    if discs:
        return max(float(np.median(discs)) - 2, 1.0), 0.0
    return max(2 * float(np.median(dist[path[:, 1].astype(int), path[:, 0].astype(int)])) - 1, 1.0), 0.0


def refit(a, b, mask, th_px):
    """Fit a straight piece to the wall mask: cross-sections of the mask at samples along the piece give the median
    centre offset along the left normal and the median width; samples off the mask (an opening band, a gap) or with
    a side that runs away (a crossing wall at a junction) are skipped. Returns (shift vector, width) in pixels, or
    (zero, th_px) with fewer than three usable samples."""
    H, W = mask.shape
    d = b - a
    L = float(np.hypot(*d))
    if L < 3:
        return np.zeros(2), th_px
    d /= L
    nrm = np.array([-d[1], d[0]])
    cap = int(3 * th_px + 2)
    offs, widths = [], []
    for f in np.linspace(0.08, 0.92, max(5, min(40, int(L / 4)))):
        p = a + d * f * L
        xi, yi = int(round(p[0])), int(round(p[1]))
        if not (0 <= xi < W and 0 <= yi < H) or not mask[yi, xi]:
            continue
        side = {}
        for sgn in (1, -1):
            side[sgn] = 0
            for s in range(1, cap + 1):
                x, y = p + sgn * nrm * s
                xj, yj = int(round(x)), int(round(y))
                if not (0 <= xj < W and 0 <= yj < H) or not mask[yj, xj]:
                    break
                side[sgn] += 1
        if side[1] >= cap or side[-1] >= cap:
            continue
        offs.append((side[1] - side[-1]) / 2)
        widths.append(1 + side[1] + side[-1])
    if len(offs) < 3:
        return np.zeros(2), th_px
    return float(np.median(offs)) * nrm, float(np.median(widths))


def regularise(G, dist, cfg=DEFAULT, mask=None, mask_dist=None):
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
        if mask is not None:
            th_px, offset = path_thickness(path, mask, mask_dist if mask_dist is not None else dist)
        else:
            th_px, offset = max(2 * float(np.median(dist[path[:, 1].astype(int), path[:, 0].astype(int)])) - 1, 1.0), 0.0
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
            # the piece is refitted to the mask's cross-sections (the skeleton of a wall of even width sits half a
            # pixel off its centre, and jambs and niches pull it); skeleton coordinates index pixels while the
            # polygons of the pipeline follow pixel edges, so the line also moves to the pixel centres (+0.5)
            shift, width = (refit(a2, b2, mask, th_px) if mask is not None else (offset * np.array([-dv[1], dv[0]]), th_px))
            width = min(width, 1.5 * th_px)                # a cross-section through a blob is not the wall's width
            segs.append({"id": f"w{len(segs):03d}", "line": LineString([a2 + shift + 0.5, b2 + shift + 0.5]), "thickness": round(width / m, 3),
                         "angle": round(ang % 180, 1), "nodes": (u, v)})
    return segs


def opening_bands(lab, cfg=DEFAULT):
    """Door and window blobs as rectangles no deeper than opening_axis_max_depth (a door label that includes its swing
    is cut back to the wall band), added to the wall mask so that the wall axis runs on through the openings."""
    op = clean(np.isin(lab, (DOOR, WINDOW)), cfg.px2(cfg.opening_min_area))
    out = np.zeros(lab.shape, np.uint8)
    cap = cfg.px(cfg.opening_axis_max_depth)
    if not op.any() or cap <= 0:                              # 0 turns the rule off: the axis stops at the jambs
        return out > 0
    n, cl, st, _ = cv2.connectedComponentsWithStats(op.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        ys, xs = np.nonzero(cl[y:y + h, x:x + w] == i)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(np.column_stack([xs + x, ys + y]).astype(np.float32))
        rw, rh = max(rw, 1.0), max(rh, 1.0)
        if min(rw, rh) > cap:
            rw, rh = (cap, rh) if rw < rh else (rw, cap)
        cv2.fillPoly(out, [np.round(cv2.boxPoints(((cx, cy), (rw, rh), ang))).astype(np.int32)], 1)
    return out > 0


def opening_intervals(line, near, lab, cfg=DEFAULT):
    """Door and window intervals along a segment's centre line: runs of opening pixels (near: the door and window
    masks dilated by one pixel) at least half opening_min_length long, as (from, to, kind) in pixels from the line's
    start; the kind is the label that dominates the run."""
    a, b = np.asarray(line.coords[0]), np.asarray(line.coords[-1])
    n = max(int(line.length) + 1, 2)
    pts = a + (b - a) * np.linspace(0.0, 1.0, n)[:, None]
    H, W = lab.shape
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    hit = {k: m[yi, xi] for k, m in near.items()}
    v = hit["door"] | hit["window"]
    step = line.length / (n - 1)
    min_run = cfg.px(cfg.opening_min_length) / 2
    runs, k = [], 0
    while k < n:
        if not v[k]:
            k += 1
            continue
        j = k
        while j < n and v[j]:
            j += 1
        if (j - k) * step >= min_run:
            nd, nw = int(hit["door"][k:j].sum()), int(hit["window"][k:j].sum())
            runs.append((k * step, (j - 1) * step, "door" if nd >= nw else "window"))
        k = j
    return runs


def segment_polys(G, segs, cfg=DEFAULT):
    """Wall polygons of the regularised segments: each centre line, without its opening intervals, buffered by half
    its thickness (flat ends); at a junction every incident segment's end is extended by half the thickness of the
    thickest other wall there, so that corners close without a square that would stick out of a thin wall."""
    by_node = {}
    for s in segs:
        for n in s["nodes"]:
            by_node.setdefault(n, []).append(s)
    parts = []
    for s in segs:
        line, half = s["line"], cfg.px(s["thickness"]) / 2
        a, b = np.asarray(line.coords[0]), np.asarray(line.coords[-1])
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        d = (b - a) / L
        ext = [0.0, 0.0]
        for n in s["nodes"]:
            others = [o for o in by_node.get(n, []) if o is not s]
            if G.degree(n) < 2 or not others:
                continue
            xy = G.nodes[n]["xy"] + 0.5                   # pixel-edge coordinates, as the segment lines
            e = max(cfg.px(o["thickness"]) for o in others) / 2
            for k, end in enumerate((a, b)):
                if np.linalg.norm(end - xy) <= half + 3:
                    ext[k] = max(ext[k], e)
        t, pieces = 0.0, []
        for t0, t1, _ in s.get("_openings_px", []):
            if t0 - t >= 1.0:
                pieces.append((t, t0))
            t = max(t, t1)
        if L - t >= 1.0:
            pieces.append((t, L))
        for t0, t1 in pieces:
            p0 = a + d * (t0 - (ext[0] if t0 <= 0.0 else 0.0))
            p1 = a + d * (t1 + (ext[1] if t1 >= L else 0.0))
            parts.append(LineString([p0, p1]).buffer(half, cap_style="flat", join_style="mitre"))
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
    # the axis runs on through doors and windows: an opening is an interval on its wall, not the wall's end, so the
    # piers between the windows of a facade are one wall and the exterior wall closes around the building
    bands = opening_bands(lab, cfg)
    axis, dist_wall = wall, dist
    if bands.any():
        axis = wall | bands
        dist = ndimage.distance_transform_edt(axis)
        G = wall_graph(axis, dist, cfg)
    near = {k: cv2.dilate((lab == c).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0 for k, c in (("door", DOOR), ("window", WINDOW))}
    segs = []
    for s in regularise(G, dist, cfg, mask=wall, mask_dist=dist_wall):     # thickness from the wall mask, not the opening bands
        if s["line"].length < cfg.px(cfg.wall_segment_min_length):
            continue
        runs = opening_intervals(s["line"], near, lab, cfg) if bands.any() else []
        s["_openings_px"] = runs
        s["openings"] = [{"kind": k, "from": round(t0 / m, 2), "to": round(t1 / m, 2)} for t0, t1, k in runs]
        segs.append(s)
    sheet.wall_graph, sheet.wall_segments = G, segs
    sheet.wall_polys_raw = [p.simplify(cfg.px(cfg.poly_simplify)) for p in mask_polys(wall, cfg.px2(cfg.wall_min_area))]
    sheet.wall_polys = segment_polys(G, segs, cfg) or sheet.wall_polys_raw
    return segs
