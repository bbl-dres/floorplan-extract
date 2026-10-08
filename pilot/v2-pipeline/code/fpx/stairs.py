"""Stage 5, stairs and voids: stair flights and one outline per stair (voids are found in stage 6)."""
import cv2
import networkx as nx
import numpy as np
import shapely

from .config import DEFAULT
from .geometry import clean, mask_polys
from .model import STAIRS


def stairs(sheet, cfg=DEFAULT):
    m = clean(sheet.label == STAIRS, cfg.px2(cfg.stair_min_area))
    polys_ = mask_polys(m, cfg.px2(cfg.stair_min_area))
    sheet.stairs = [p.simplify(cfg.px(cfg.line_simplify)) for p in polys_]
    # one outline per stair: convex hull of flights closer than stair_group_dist (encloses landings and the stairwell)
    groups = nx.Graph()
    groups.add_nodes_from(range(len(polys_)))
    groups.add_edges_from((i, j) for i in range(len(polys_)) for j in range(i + 1, len(polys_))
                          if polys_[i].distance(polys_[j]) < cfg.px(cfg.stair_group_dist))
    hull = np.zeros(m.shape, np.uint8)
    for comp in nx.connected_components(groups):
        h = shapely.union_all([polys_[i] for i in comp]).convex_hull
        cv2.fillPoly(hull, [np.asarray(h.exterior.coords, np.int32)], 1)
    sheet.stair_mask, sheet.stair_hull = m, (hull > 0) & ~sheet.wall_mask
    sheet.voids = []                                   # found in stage 6 from void labels (no void class in the segmenter)
    return sheet.stairs
