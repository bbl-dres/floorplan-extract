"""Score the pipeline output (data/out/<sheet>.json) against the curated 2005 reference of Landgut Lohn 1. OG.

    python evaluate.py [--out data/out-v2]   # folder of run_pipeline.py output (default data/out)

S1 shares the reference frame (calibrated sheet). The scans are registered by translation only: their GF
outline is centred on the reference building, then refined on the wall overlap within +-0.5 m. The scans show
an earlier state of the building, so for them only unchanged elements (outline, exterior openings, some rooms)
are meaningful.

Rooms are matched one to one (Hungarian on IoU, metrics.match_rooms), so a merged room counts for one reference
only; the best-IoU figures of the earlier evaluation are kept under rooms.legacy_best_iou. Connections are mapped
through a one-to-one matching at IoU >= 0.3 (the scans show an earlier state of some rooms).
"""
import difflib
import json
import re
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Point, shape
from shapely.ops import unary_union

import metrics
from common import DATA, V1

OUT = DATA / "out"
_ref = json.loads((V1 / "data/reference/plan_topology_og1-main.json").read_text(encoding="utf-8"))
REF_ROOMS = [(r["name"], r["labelArea"], shape(r["geometry"])) for r in _ref["rooms"]]
REF_WALLS = unary_union([shape(w["geometry"]) for w in _ref["walls"]])
REF_OPENINGS = [shape(o["geometry"]) for o in _ref["openings"]]
# S2 shows an earlier state with other room names: the 11 calligraphic labels as read in pilot v1
HISTORICAL_S2 = ["Toilette", "Diensten-Zimmer", "Antichambre", "Herren-Zimmer", "Halle", "Vestibule", "Bad", "Zimmer",
                 "Zimmer", "Boudoir", "Schlafzimmer"]
BUILDING = unary_union([g for *_, g in REF_ROOMS] + [REF_WALLS]).buffer(0.05).buffer(-0.05)


def iou(a, b):
    u = a.union(b).area
    return a.intersection(b).area / u if u else 0.0


def norm(s):
    return re.sub(r"[^a-zäöü]", "", (s or "").lower())


def ref_edge(poly):
    """The two reference rooms on either side of an opening: try both rectangle axes (thick walls, narrow doors)."""
    c = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)[:4]
    ctr = c.mean(0)
    for e in (c[2] - c[1], c[1] - c[0]):
        ln = max(np.hypot(*e), 1e-9)
        for off in (0.3, 0.6):
            a, b = (ref_room_at(Point(ctr + s * e / ln * (ln / 2 + off))) for s in (1, -1))
            if a and b and a != b:
                return frozenset((a, b))
    return None


def ref_room_at(p):
    return next((name for name, _, g in REF_ROOMS if g.buffer(0.05).contains(p)), None)


def register(d):
    """Translation that aligns a scan's output with the reference (scans only)."""
    gf = shape(d["floor"]["gf"])
    walls = unary_union([shape(w) for w in d["wall_polygons"]])
    dx0 = np.array(BUILDING.centroid.coords[0]) - np.array(gf.centroid.coords[0])
    best = max(((iou(shapely.affinity.translate(walls, *(dx0 + [ex, ey])), REF_WALLS), ex, ey)
                for ex in np.arange(-0.5, 0.51, 0.05) for ey in np.arange(-0.5, 0.51, 0.05)))
    return dx0 + best[1:]


def evaluate(sid):
    d = json.loads((OUT / f"{sid}.json").read_text(encoding="utf-8"))
    shift = np.zeros(2) if sid == "s1" else register(d)
    T = lambda g: shapely.affinity.translate(shape(g), *shift)
    rooms = [(r, T(r["geometry"])) for r in d["rooms"]]
    walls = unary_union([T(w) for w in d["wall_polygons"]])
    gf = T(d["floor"]["gf"])
    voids = [T(v["geometry"]) for v in d["voids"] if v["gf_deducted"]]
    gf_gross = unary_union([gf] + voids)
    res = {"sheet": sid, "shift_m": [round(float(v), 2) for v in shift]}
    res["walls_iou"] = round(iou(walls, REF_WALLS), 3)
    res["walls_area"] = [round(walls.area, 1), round(REF_WALLS.area, 1)]
    res["gf_iou"] = round(iou(gf_gross, BUILDING), 3)
    res["gf_area_error_pct"] = round((gf_gross.area - BUILDING.area) / BUILDING.area * 100, 1)
    # rooms: one-to-one matching; each reference room gets the prediction assigned to it (also below IoU 0.5)
    ref_geoms = [g for *_, g in REF_ROOMS]
    mr = metrics.match_rooms([pg for _, pg in rooms], ref_geoms, 0.5)
    per_room = []
    for j, (name, stamp, g) in enumerate(REF_ROOMS):
        i = mr["ref_pred"][j]
        r, pg = rooms[i] if i is not None else (None, None)
        v = mr["ref_iou"][j]
        per_room.append({"ref": name, "iou": round(v, 3), "pred": r["id"] if r else None,
                         "pred_name": r["name"] if r else None,
                         "name_ok": bool(r and r["name"] and (difflib.SequenceMatcher(None, norm(r["name"]), norm(name)).ratio() > 0.75
                                                             or (len(norm(r["name"])) >= 4 and norm(name).startswith(norm(r["name"]))))),
                         "stamp_read": r["area_stamp"] if r else None, "stamp_ok": bool(r and r["area_stamp"] == stamp),
                         "area_error_pct": round((r["area"] - stamp) / stamp * 100, 1) if v >= 0.5 else None,
                         "area_error_m2": round(r["area"] - stamp, 2) if v >= 0.5 else None,
                         "geometry_area_error_pct": round((pg.area - g.area) / g.area * 100, 1) if v >= 0.5 else None,
                         "best_iou_legacy": round(mr["best_iou"][j], 3)})
    matched = [p for p in per_room if p["iou"] >= 0.5]
    med = lambda xs: round(float(np.median(xs)), 2) if len(xs) else None
    res["rooms"] = {"reference": len(REF_ROOMS), "predicted": len(rooms), "matched_iou50": len(matched),
                    "mean_iou": round(float(np.mean([p["iou"] for p in per_room])), 3),
                    "median_abs_area_error_pct": round(float(np.median([abs(p["area_error_pct"]) for p in matched])), 1) if matched else None,
                    "names_ok": sum(p["name_ok"] for p in per_room), "stamp_areas_ok": sum(p["stamp_ok"] for p in per_room),
                    "per_room": per_room,
                    "matching": "one-to-one (Hungarian on IoU), IoU >= 0.5",
                    "recall": round(mr["recall"], 3), "precision": round(mr["precision"], 3) if mr["precision"] is not None else None,
                    "mean_iou_matched": round(mr["mean_iou"], 3) if mr["mean_iou"] is not None else None,
                    "median_area_error_pct": med([p["area_error_pct"] for p in matched]),
                    "median_abs_area_error_m2": med([abs(p["area_error_m2"]) for p in matched]),
                    "median_geometry_area_error_pct": med([p["geometry_area_error_pct"] for p in matched]),
                    "median_abs_geometry_area_error_pct": med([abs(p["geometry_area_error_pct"]) for p in matched]),
                    "legacy_best_iou": {"matched_iou50": int(sum(b >= 0.5 for b in mr["best_iou"])),
                                        "mean_iou": round(float(np.mean(mr["best_iou"])), 3)}}
    # openings: matched by centre distance < 0.6 m
    pred_o = [(o, T(o["geometry"])) for o in d["openings"]]
    ext_ref = [BUILDING.exterior.distance(o.centroid) < 0.6 for o in REF_OPENINGS]
    used, hits = set(), []
    for i, o in enumerate(REF_OPENINGS):
        cands = [(o.centroid.distance(g.centroid), j) for j, (_, g) in enumerate(pred_o) if j not in used]
        dist, j = min(cands, default=(9, None))
        hits.append(j if dist < 0.6 else None)
        if dist < 0.6:
            used.add(j)
    res["openings"] = {"reference": len(REF_OPENINGS), "predicted": len(pred_o),
                       "recall": round(sum(h is not None for h in hits) / len(REF_OPENINGS), 3),
                       "precision": round(len(used) / max(len(pred_o), 1), 3),
                       "exterior_recall": f"{sum(h is not None for h, e in zip(hits, ext_ref) if e)}/{sum(ext_ref)}",
                       "interior_recall": f"{sum(h is not None for h, e in zip(hits, ext_ref) if not e)}/{len(ext_ref) - sum(ext_ref)}",
                       "by_kind": {k: sum(o["kind"] == k for o, _ in pred_o) for k in ("door", "exterior door", "window", "passage")}}
    # room connections: reference edges from interior openings, predicted edges mapped to reference rooms
    ref_edges = set()
    for o, e in zip(REF_OPENINGS, ext_ref):
        edge = None if e else ref_edge(o)
        if edge:
            ref_edges.add(edge)
    def edges_via(to_ref):
        return {frozenset((to_ref[c["a"]], to_ref[c["b"]])) for c in d["connectivity"]
                if c["a"] in to_ref and c["b"] in to_ref and to_ref[c["a"]] != to_ref[c["b"]]}

    M = metrics.iou_matrix([g for _, g in rooms], ref_geoms)          # one-to-one mapping of rooms at IoU >= 0.3
    to_ref = {rooms[i][0]["id"]: REF_ROOMS[j][0] for i, j in metrics.assign(M, 0.3) if M[i, j] >= 0.3}
    pred_edges = edges_via(to_ref)
    tp = len(ref_edges & pred_edges)
    p_, r_ = tp / max(len(pred_edges), 1), tp / max(len(ref_edges), 1)
    legacy = {rooms[i][0]["id"]: REF_ROOMS[int(np.argmax(M[i]))][0] for i in range(len(rooms)) if M.shape[1] and M[i].max() >= 0.3}
    lp = edges_via(legacy)                                             # earlier mapping: each room to its best reference
    res["connectivity"] = {"reference_edges": len(ref_edges), "predicted_edges": len(pred_edges), "correct": tp,
                           "f1": round(2 * p_ * r_ / max(p_ + r_, 1e-9), 3),
                           "missing": sorted(" - ".join(sorted(e)) for e in ref_edges - pred_edges),
                           "extra": sorted(" - ".join(sorted(e)) for e in pred_edges - ref_edges),
                           "mapping": "one-to-one (Hungarian on IoU), IoU >= 0.3",
                           "legacy_best_iou": {"predicted_edges": len(lp), "correct": len(ref_edges & lp)}}
    if sid == "s2":
        def hits(names):
            left, n = [norm(h) for h in HISTORICAL_S2], 0
            for x in names:
                if norm(x) in left:
                    left.remove(norm(x))
                    n += 1
            return n
        raw = [x for r in d["rooms"] if r.get("name_raw") for x in r["name_raw"].split(" / ")]
        cor = [x for r in d["rooms"] if r["name"] for x in r["name"].split(" / ")]
        sim = [max(difflib.SequenceMatcher(None, norm(x), norm(h)).ratio() for x in raw) if raw else 0 for h in HISTORICAL_S2]
        res["historical_names"] = {"labels": len(HISTORICAL_S2), "raw_ocr_exact": hits(raw), "corrected_exact": hits(cor),
                                   "raw_ocr_mean_similarity": round(float(np.mean(sim)), 2), "raw": raw, "corrected": cor}
    res["qa_issues"] = len(d["qa"])
    res["confidence"] = {c: sum(r["confidence"] == c for r in d["rooms"]) for c in ("high", "medium", "low")}
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT), help="folder with s1.json … (writes evaluation.json there)")
    OUT = Path(ap.parse_args().out)
    results = []
    for sid in ("s1", "s2", "s3"):
        if not (OUT / f"{sid}.json").exists():
            continue
        r = evaluate(sid)
        results.append(r)
        ro, op, co = r["rooms"], r["openings"], r["connectivity"]
        print(f"{sid}: walls IoU {r['walls_iou']} ({r['walls_area'][0]} vs {r['walls_area'][1]} m²) | GF IoU {r['gf_iou']}, "
              f"area {r['gf_area_error_pct']:+}% | rooms {ro['matched_iou50']}/{ro['reference']} IoU>=0.5, mean IoU {ro['mean_iou']}, "
              f"median |area err| {ro['median_abs_area_error_pct']}%, names {ro['names_ok']}/15, stamp areas {ro['stamp_areas_ok']}/15 | "
              f"openings recall {op['recall']} (ext {op['exterior_recall']}, int {op['interior_recall']}), precision {op['precision']} | "
              f"connections F1 {co['f1']} ({co['correct']}/{co['reference_edges']}) | shift {r['shift_m']}")
        lg = ro["legacy_best_iou"]
        print(f"     one-to-one: recall {ro['recall']}, precision {ro['precision']}, mean IoU of matches {ro['mean_iou_matched']}, "
              f"median area error vs stamp {ro['median_area_error_pct']}% (|m²| {ro['median_abs_area_error_m2']}), "
              f"vs reference polygon {ro['median_geometry_area_error_pct']}% | best-IoU (old): {lg['matched_iou50']}/{ro['reference']}, "
              f"mean IoU {lg['mean_iou']} | connections with the old mapping {co['legacy_best_iou']['correct']}/{co['reference_edges']} "
              f"of {co['legacy_best_iou']['predicted_edges']}")
        for p in ro["per_room"]:
            print(f"     {p['ref']:20s} IoU {p['iou']:.2f}  pred {p['pred']} {p['pred_name']!r:28s} stamp {p['stamp_read']}  err {p['area_error_pct']}")
        if "historical_names" in r:
            h = r["historical_names"]
            print(f"     historical names: raw OCR exact {h['raw_ocr_exact']}/11 (mean similarity {h['raw_ocr_mean_similarity']}), "
                  f"after dictionary correction {h['corrected_exact']}/11")
        if sid == "s1":
            print("     missing connections:", co["missing"])
            print("     extra connections:", co["extra"])
    (OUT / "evaluation.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
