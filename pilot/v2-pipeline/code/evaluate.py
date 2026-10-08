"""Score the pipeline output (data/out/<sheet>.json) against the curated 2005 reference of Landgut Lohn 1. OG.

    python evaluate.py [--out data/out-v2]   # folder of run_pipeline.py output (default data/out)

S1 shares the reference frame (calibrated sheet). The scans are registered by translation only: their GF
outline is centred on the reference building, then refined on the wall overlap within +-0.5 m. The scans show
an earlier state of the building, so for them only unchanged elements (outline, exterior openings, some rooms)
are meaningful.

Scored by fpeval.score, the one protocol of every benchmark, on the output in plan metres: rooms one to one
(Hungarian on IoU at 0.5), openings by centre within 0.5 m one to one regardless of kind (the reference has no
kinds; before: nearest within 0.6 m, greedily), connections mapped through the room matching (before: a separate
one-to-one mapping at IoU >= 0.3 for the scans, kept under connectivity.mapping_iou30), GF IoU and area error. The
best-IoU figures of the earlier evaluation stay under legacy_best_iou, so older evaluation.json files read the same.
"""
import difflib
import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import shapely
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from common import DATA, V1
from fpeval import metrics
from fpeval.cli import utf8_stdout
from fpeval.score import ROOM_IOU, score
from fpx import DEFAULT

OUT = DATA / "out"
METRES = replace(DEFAULT, px_per_m=1)                    # the output is in plan metres: one "pixel" per metre
MAPPING_IOU30 = 0.3                                      # the earlier connectivity mapping: scans show an earlier state of some rooms
_ref = json.loads((V1 / "data/reference/plan_topology_og1-main.json").read_text(encoding="utf-8"))
REF_ROOMS = [(r["name"], r["labelArea"], shape(r["geometry"])) for r in _ref["rooms"]]
REF_WALLS = unary_union([shape(w["geometry"]) for w in _ref["walls"]])
REF_OPENINGS = [shape(o["geometry"]) for o in _ref["openings"]]
# S2 shows an earlier state with other room names: the 11 calligraphic labels as read in pilot v1
HISTORICAL_S2 = ["Toilette", "Diensten-Zimmer", "Antichambre", "Herren-Zimmer", "Halle", "Vestibule", "Bad", "Zimmer",
                 "Zimmer", "Boudoir", "Schlafzimmer"]
BUILDING = unary_union([g for *_, g in REF_ROOMS] + [REF_WALLS]).buffer(0.05).buffer(-0.05)


def iou(a, b):
    return metrics.iou(a, b)


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


def reference():
    """The 2005 reference as a Reference in plan metres: rooms typed by name, openings without a kind, the door
    connections between rooms, the building outline."""
    ext = [BUILDING.exterior.distance(o.centroid) < 0.6 for o in REF_OPENINGS]
    names = [n for n, _, _ in REF_ROOMS]
    edges = {frozenset(names.index(n) for n in e) for o, e_ in zip(REF_OPENINGS, ext) if not e_ and (e := ref_edge(o))}
    return {"rooms": [(n, g) for n, _, g in REF_ROOMS], "ignore": [], "edges": edges, "open_edges": set(),
            "openings": [("opening", tuple(o.centroid.coords[0])) for o in REF_OPENINGS], "gf": BUILDING}, ext


def prediction(d, shift):
    """The output JSON, shifted, as the sheet-like object fpeval.score reads (rooms, openings, connectivity, GF, voids)."""
    T = lambda g: shapely.affinity.translate(shape(g), *shift)
    rooms = [{"id": r["id"], "poly": T(r["geometry"]), "names": [r["name"]] if r["name"] else [], "area_stamp": r["area_stamp"], "json": r}
             for r in d["rooms"]]
    openings = [{"id": o["id"], "kind": o["kind"], "centre": np.asarray(T(o["geometry"]).centroid.coords[0]), "poly": T(o["geometry"])}
                for o in d["openings"] if not shape(o["geometry"]).is_empty]
    g = nx.Graph()
    g.add_nodes_from(r["id"] for r in rooms)
    g.add_edges_from((c["a"], c["b"], {"opening": c["opening"]}) for c in d["connectivity"])
    return SimpleNamespace(rooms=rooms, fragments=[], openings=openings, connectivity=g, gf=T(d["floor"]["gf"]),
                           voids=[{"poly": T(v["geometry"]), "gf_deducted": v["gf_deducted"]} for v in d["voids"]],
                           walls=unary_union([T(w) for w in d["wall_polygons"]]), text=d.get("text", []),
                           wall_mask=None, seg_label=None, label=None, building_rough=None)


def evaluate(sid):
    d = json.loads((OUT / f"{sid}.json").read_text(encoding="utf-8"))
    shift = np.zeros(2) if sid == "s1" else register(d)
    pred = prediction(d, shift)
    ref, ext_ref = reference()
    row = score(pred, ref, METRES)
    rooms, op, co = row["rooms"], row["openings"], row["connectivity"]
    by_id = {r["id"]: r for r in pred.rooms}
    res = {"sheet": sid, "shift_m": [round(float(v), 2) for v in shift]}
    res["walls_iou"] = round(iou(pred.walls, REF_WALLS), 3)
    res["walls_area"] = [round(pred.walls.area, 1), round(REF_WALLS.area, 1)]
    res["gf_iou"] = round(row["gf"]["iou"], 3)
    res["gf_area_error_pct"] = round(row["gf"]["area_err_pct"], 1)
    # rooms: one-to-one matching; each reference room gets the prediction assigned to it (also below IoU 0.5)
    per_room = []
    for (name, stamp, g), (pid, v, best) in zip(REF_ROOMS, rooms["ref_assigned"]):
        r = by_id[pid]["json"] if pid else None
        pg = by_id[pid]["poly"] if pid else None
        hit = v >= ROOM_IOU
        per_room.append({"ref": name, "iou": round(v, 3), "pred": pid, "pred_name": r["name"] if r else None,
                         "name_ok": bool(r and r["name"] and (difflib.SequenceMatcher(None, norm(r["name"]), norm(name)).ratio() > 0.75
                                                             or (len(norm(r["name"])) >= 4 and norm(name).startswith(norm(r["name"]))))),
                         "stamp_read": r["area_stamp"] if r else None, "stamp_ok": bool(r and r["area_stamp"] == stamp),
                         "area_error_pct": round((r["area"] - stamp) / stamp * 100, 1) if hit else None,
                         "area_error_m2": round(r["area"] - stamp, 2) if hit else None,
                         "geometry_area_error_pct": round((pg.area - g.area) / g.area * 100, 1) if hit else None,
                         "best_iou_legacy": round(best, 3)})
    matched = [p for p in per_room if p["iou"] >= ROOM_IOU]
    med = lambda xs: round(float(np.median(xs)), 2) if len(xs) else None
    res["rooms"] = {"reference": rooms["ref"], "predicted": len(pred.rooms), "matched_iou50": rooms["matched"],
                    "mean_iou": round(rooms["mean_iou_ref"], 3),
                    "median_abs_area_error_pct": round(float(np.median([abs(p["area_error_pct"]) for p in matched])), 1) if matched else None,
                    "names_ok": sum(p["name_ok"] for p in per_room), "stamp_areas_ok": sum(p["stamp_ok"] for p in per_room),
                    "per_room": per_room,
                    "matching": "one-to-one (Hungarian on IoU), IoU >= 0.5",
                    "recall": round(rooms["recall"], 3), "precision": round(rooms["precision"], 3) if rooms["precision"] is not None else None,
                    "mean_iou_matched": round(rooms["mean_iou"], 3) if rooms["mean_iou"] is not None else None,
                    "median_area_error_pct": med([p["area_error_pct"] for p in matched]),
                    "median_abs_area_error_m2": med([abs(p["area_error_m2"]) for p in matched]),
                    "median_geometry_area_error_pct": med([p["geometry_area_error_pct"] for p in matched]),
                    "median_abs_geometry_area_error_pct": med([abs(p["geometry_area_error_pct"]) for p in matched]),
                    "missed": rooms["missed"],
                    "legacy_best_iou": {"matched_iou50": int(sum(a[2] >= ROOM_IOU for a in rooms["ref_assigned"])),
                                        "mean_iou": round(float(np.mean([a[2] for a in rooms["ref_assigned"]])), 3)}}
    # openings: matched by centre, one to one, regardless of kind (the reference has none)
    hits = [None] * len(REF_OPENINGS)
    for i, j in op["pairs_any"]:                        # i indexes the scored (door/window) predictions, j the reference
        hits[j] = i
    res["openings"] = {"reference": op["ref_any"], "predicted": op["pred_any"],
                       "recall": round(op["found_any_kind"] / max(op["ref_any"], 1), 3),
                       "precision": round(op["found_any_kind"] / max(op["pred_any"], 1), 3),
                       "exterior_recall": f"{sum(h is not None for h, e in zip(hits, ext_ref) if e)}/{sum(ext_ref)}",
                       "interior_recall": f"{sum(h is not None for h, e in zip(hits, ext_ref) if not e)}/{len(ext_ref) - sum(ext_ref)}",
                       "by_kind": {k: sum(o["kind"] == k for o in pred.openings) for k in ("door", "exterior door", "window", "passage")},
                       "matching": "one-to-one by centre within 0.5 m, any kind"}
    # room connections: reference edges from interior openings, predicted edges mapped to reference rooms
    names = [n for n, _, _ in REF_ROOMS]
    ref_edges = {frozenset(names[i] for i in e) for e in ref["edges"]}

    def edges_via(to_ref):
        return {frozenset((to_ref[c["a"]], to_ref[c["b"]])) for c in d["connectivity"]
                if c["a"] in to_ref and c["b"] in to_ref and to_ref[c["a"]] != to_ref[c["b"]]}

    to_ref = {pid: name for name, (pid, v, _) in zip(names, rooms["ref_assigned"]) if pid and v >= ROOM_IOU}
    pred_edges = edges_via(to_ref)
    tp = len(ref_edges & pred_edges)
    assert tp == co["correct"] and len(pred_edges) == co["pred_mapped"], "the named edges and the scorer's counts must agree"
    rates = metrics.edge_rates(co)
    M = metrics.iou_matrix([r["poly"] for r in pred.rooms], [g for *_, g in REF_ROOMS])
    at30 = {pred.rooms[i]["id"]: names[j] for i, j in metrics.assign(M, MAPPING_IOU30) if M[i, j] >= MAPPING_IOU30}
    e30 = edges_via(at30)
    legacy = {pred.rooms[i]["id"]: names[int(np.argmax(M[i]))] for i in range(len(pred.rooms)) if M.shape[1] and M[i].max() >= MAPPING_IOU30}
    lp = edges_via(legacy)                                             # earlier mapping: each room to its best reference
    # predicted_edges counts every predicted connection, also those at a room without a match (the harness rule;
    # before: only connections between matched rooms, now predicted_edges_mapped and f1_matched)
    res["connectivity"] = {"reference_edges": co["ref"], "predicted_edges": co["pred"], "correct": co["correct"],
                           "f1": round(rates["f1"] or 0.0, 3),
                           "predicted_edges_mapped": co["pred_mapped"], "f1_matched": round(rates["f1_matched"] or 0.0, 3),
                           "missing": sorted(" - ".join(sorted(e)) for e in ref_edges - pred_edges),
                           "extra": sorted(" - ".join(sorted(e)) for e in pred_edges - ref_edges),
                           "mapping": "through the room matching (one-to-one, IoU >= 0.5)",
                           "mapping_iou30": {"predicted_edges": len(e30), "correct": len(ref_edges & e30),
                                             "f1": round(metrics.f1(len(ref_edges & e30) / max(len(e30), 1), len(ref_edges & e30) / max(len(ref_edges), 1)), 3)},
                           "legacy_best_iou": {"predicted_edges": len(lp), "correct": len(ref_edges & lp)}}
    if sid == "s2":
        def hits_of(items):
            left, n = [norm(h) for h in HISTORICAL_S2], 0
            for x in items:
                if norm(x) in left:
                    left.remove(norm(x))
                    n += 1
            return n
        raw = [x for r in d["rooms"] if r.get("name_raw") for x in r["name_raw"].split(" / ")]
        cor = [x for r in d["rooms"] if r["name"] for x in r["name"].split(" / ")]
        sim = [max(difflib.SequenceMatcher(None, norm(x), norm(h)).ratio() for x in raw) if raw else 0 for h in HISTORICAL_S2]
        res["historical_names"] = {"labels": len(HISTORICAL_S2), "raw_ocr_exact": hits_of(raw), "corrected_exact": hits_of(cor),
                                   "raw_ocr_mean_similarity": round(float(np.mean(sim)), 2), "raw": raw, "corrected": cor}
    res["qa_issues"] = len(d["qa"])
    res["confidence"] = {c: sum(r["confidence"] == c for r in d["rooms"]) for c in ("high", "medium", "low")}
    res["harness_row"] = {k: v for k, v in row.items() if k != "rooms"} | {"rooms": {k: v for k, v in rooms.items() if k not in ("ref_rooms", "ref_assigned", "pred_best_iou", "missed")}}
    return res


def main(argv=None):
    import argparse
    global OUT
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT), help="folder with s1.json … (writes evaluation.json there)")
    OUT = Path(ap.parse_args(argv).out)
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
              f"mean IoU {lg['mean_iou']} | connections mapped at IoU 0.3 {co['mapping_iou30']['correct']}/{co['reference_edges']} "
              f"of {co['mapping_iou30']['predicted_edges']}")
        for p in ro["per_room"]:
            print(f"     {p['ref']:20s} IoU {p['iou']:.2f}  pred {p['pred']} {p['pred_name']!r:28s} stamp {p['stamp_read']}  err {p['area_error_pct']}")
        if "historical_names" in r:
            h = r["historical_names"]
            print(f"     historical names: raw OCR exact {h['raw_ocr_exact']}/11 (mean similarity {h['raw_ocr_mean_similarity']}), "
                  f"after dictionary correction {h['corrected_exact']}/11")
        if sid == "s1":
            print("     missing connections:", co["missing"])
            print("     extra connections:", co["extra"])
    from fpeval.cli import jsonable
    (OUT / "evaluation.json").write_text(json.dumps(jsonable(results), indent=1, ensure_ascii=False), encoding="utf-8")
    return results


if __name__ == "__main__":
    main()
