"""Stage 3b, walls and the wall graph: clean masks, skeleton, centre lines with thickness."""
import networkx as nx
import numpy as np
from scipy import ndimage
from shapely.geometry import LineString
from skimage.morphology import skeletonize

from .config import DEFAULT
from .geometry import clean, mask_polys
from .model import COLUMN, WALL


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


def walls(sheet, cfg=DEFAULT):
    m = cfg.m
    lab = sheet.label
    wall = clean(lab == WALL, cfg.px2(cfg.wall_min_area))
    column = clean(lab == COLUMN, cfg.px2(cfg.column_min_area))
    sheet.wall_mask, sheet.column_mask = wall, column
    dist = ndimage.distance_transform_edt(wall)
    G = skeleton_graph(skeletonize(wall))
    for _ in range(2):                                  # prune short spurs (skeleton artefacts at wall ends)
        for u, v, k, d in list(G.edges(keys=True, data=True)):
            if (G.degree(u) == 1 or G.degree(v) == 1) and len(d["path"]) < 1.5 * max(dist[int(d["path"][0][1]), int(d["path"][0][0])], 4):
                if G.degree(u) + G.degree(v) > 2:
                    G.remove_edge(u, v, k)
        G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    segs = []
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        path = d["path"]
        line = LineString(path).simplify(cfg.px(cfg.line_simplify)) if len(path) > 1 else None
        if line is None or line.length < cfg.px(cfg.wall_segment_min_length):
            continue
        th = 2 * float(np.median(dist[path[:, 1].astype(int), path[:, 0].astype(int)])) / m
        d["thickness"], d["line"], d["id"] = th, line, f"w{i:03d}"
        segs.append({"id": d["id"], "line": line, "thickness": round(th, 3)})
    sheet.wall_graph, sheet.wall_segments = G, segs
    sheet.wall_polys = [p.simplify(cfg.px(cfg.poly_simplify)) for p in mask_polys(wall, cfg.px2(cfg.wall_min_area))]
    return segs
