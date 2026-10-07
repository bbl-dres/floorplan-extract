"""Stage 8, derived outputs: GF area, exterior openings, EBF proposal and room connectivity."""
import networkx as nx
import numpy as np
from shapely.geometry import Point

from .config import DEFAULT


def derived(sheet, cfg=DEFAULT):
    m = cfg.m
    area = lambda g: cfg.m2(g.area)
    sheet.gf_area = area(sheet.gf)
    inner = sheet.gf.buffer(-cfg.px(cfg.gf_inner_buffer))
    for o in sheet.openings:                          # exterior = one side outside the GF outline
        nrm = np.array([-o["along"][1], o["along"][0]])
        pts = [Point(o["centre"] + s * nrm * (o["depth"] / 2 + cfg.exterior_probe) * m) for s in (1, -1)]
        o["exterior"] = not all(inner.contains(q) for q in pts)
        if o["kind"] == "window" and not o["exterior"]:
            o["kind"], o["flag"] = "interior opening", "drawn like a window in an interior wall: treated as an opening"
        if o["kind"] == "door" and o["exterior"]:
            o["kind"] = "exterior door"
    ebf_excluded = [r for r in sheet.rooms if r["usage"] == "outdoor"]
    sheet.ebf_proposal = {"area": round(sheet.gf_area - sum(area(r["poly"]) for r in ebf_excluded), 2),
                          "excluded": [r["id"] for r in ebf_excluded], "status": "proposal, to be confirmed"}
    H, W = sheet.room_label.shape
    pid = {r["px_label"]: r["id"] for r in sheet.rooms}

    def probe(p):
        x, y = int(np.clip(p[0], 0, W - 1)), int(np.clip(p[1], 0, H - 1))
        lbl = sheet.room_label[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        vals = [pid[v] for v in np.unique(lbl) if v in pid]
        return vals[0] if vals else ("outside" if not sheet.building_rough[y, x] else None)

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
        if all(ends) and ends[0] != ends[1]:
            edges.append((ends[0], ends[1], o["id"]))
    g = nx.Graph()
    g.add_nodes_from(r["id"] for r in sheet.rooms)
    g.add_edges_from((a, b, {"opening": o}) for a, b, o in edges)
    sheet.connectivity = g
    return g
