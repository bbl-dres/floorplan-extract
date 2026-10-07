"""Package sheet images, pipeline output and evaluation for viewer.html into data/out/viewer-data.js (gitignored).

    python export_viewer.py [folder ...]   # after run_pipeline.py and evaluate.py, bench.py or curated.py run

Images are embedded as data URIs, so the viewer works when opened from disk. Scans are shifted by the
registration found in evaluate.py, so the 2005 reference lines up with every sheet.

Benchmark sheets (*_bench.json) come from data/bench, data/curated-test and any folder given. Sheets of the curated set
(data/curated/plans.json, plus plans.local.json when present) are grouped by the plan's category and carry its tags,
licence, attribution, scale record, reference and `why`. A sheet is matched to its plan by `curated_id` in the bench
JSON, by source 'curated' with sid = plan id, or by its dataset sample id (cubicasa high_quality/8380, commons c02, ...).
When the same plan or sheet was run more than once, the newest output is kept.
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, shape

from common import DATA
from evaluate import REF_OPENINGS, REF_ROOMS, REF_WALLS, iou

OUT = DATA / "out"
BENCH = DATA / "bench"
BENCH_DIRS = [BENCH, DATA / "curated-test"]
LANDGUT = "Landgut Lohn, 1. OG (BBL)"
GROUPS = {"cubicasa": "CubiCasa5K (benchmark, CC BY-NC-SA)", "cvcfp": "CVC-FP (benchmark, CC BY-NC)",
          "waffle": "WAFFLE benchmark (Wikimedia Commons)", "commons": "Swiss plans on Wikimedia Commons (public domain)"}
TAG_AXES = ("category", "drawing_type", "input", "wall_style", "era", "building_type", "content", "language", "challenges",
            "use", "source")


def rings(g, shift, tol=0.01):
    """[[exterior, hole, ...], ...] of a (Multi)Polygon in plan metres, shifted."""
    g = affinity.translate(shape(g) if isinstance(g, dict) else g, *shift).simplify(tol)
    ring = lambda r: [[round(x, 3), round(y, 3)] for x, y in r.coords[:-1]]
    return [[ring(p.exterior)] + [ring(h) for h in p.interiors if abs(Polygon(h).area) > 0.02]
            for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and p.area > 0.01]


def point(g, shift):
    p = affinity.translate(shape(g), *shift).representative_point()
    return [round(p.x, 3), round(p.y, 3)]


def review(r):
    """Review flag of a small unlabelled region kept as a room (fpx.qa.small_region_review), shortened for the viewer."""
    rv = r.get("review")
    if not rv:
        return None
    return {"guess": rv.get("guess"), "width": rv.get("width"), "reason": rv.get("reason")}


def sheet_entry(sid, ev, folder=OUT, group=LANDGUT):
    d = json.loads((folder / f"{sid}.json").read_text(encoding="utf-8"))
    shift = np.array(ev["shift_m"])
    centre = {}
    rooms = []
    for r in d["rooms"]:
        g = affinity.translate(shape(r["geometry"]), *shift)
        best = max(REF_ROOMS, key=lambda rr: iou(g, rr[2]))
        centre[r["id"]] = point(r["geometry"], shift)
        rooms.append({"id": r["id"], "name": r["name"] or "(no stamp)", "raw": r.get("name_raw"), "usage": r["usage"],
                      "area": r["area"], "areaGross": r["area_gross"], "stamp": r["area_stamp"], "conf": r["confidence"],
                      "reasons": r["reasons"], "neighbours": r["neighbours"], "polygons": rings(r["geometry"], shift),
                      "label": centre[r["id"]], "iou": round(iou(g, best[2]), 2), "ref": best[0] if iou(g, best[2]) > 0 else None,
                      "smallRegion": bool(r.get("small_region")), "review": review(r)})
    openings = [{"id": o["id"], "kind": o["kind"], "width": o["width"], "exterior": o["exterior"], "connects": o["connects"],
                 "polygons": rings(o["geometry"], shift), "c": point(o["geometry"], shift)} for o in d["openings"]
                if not shape(o["geometry"]).is_empty]
    oc = {o["id"]: o["c"] for o in openings}
    graph = [[centre[c["a"]], oc[c["opening"]], centre[c["b"]]] for c in d["connectivity"]
             if c["a"] in centre and c["b"] in centre and c["opening"] in oc]
    tri = d["sheet"]["triage"]
    image = "data:image/jpeg;base64," + base64.b64encode((folder / f"{sid}_sheet.jpg").read_bytes()).decode()
    x0, y0, x1, y1 = d["sheet"]["extent"]
    # small unlabelled regions: rooms flagged for review (fpx.qa); older outputs list them only as fragments
    small = d.get("fragments")
    if small is None:
        small = [r["geometry"] for r in d["rooms"] if r.get("small_region")]
    return {
        "id": sid, "title": d["sheet"]["title"], "group": group, "image": image,
        "extent": [round(x0 + shift[0], 3), round(y0 + shift[1], 3), round(x1 + shift[0], 3), round(y1 + shift[1], 3)],
        "triage": {"Input class": tri["input_class"], "Graphical style (detected)": tri["graphical_style"],
                   "Scale": f"{d['sheet']['scale']['value']} – {d['sheet']['scale']['method']}", "Text layer": tri["text_layer"]},
        "rooms": rooms, "openings": openings, "graph": graph,
        "walls": {"polygons": [p for w in d["wall_polygons"] for p in rings(w, shift)], "iou": ev["walls_iou"],
                  "area": ev["walls_area"][0], "refArea": ev["walls_area"][1]},
        "stairs": [p for s in d["stairs"] for p in rings(s, shift)],
        "voids": [{"label": v["label"], "area": v["area"], "deducted": v["gf_deducted"], "polygons": rings(v["geometry"], shift)}
                  for v in d["voids"]],
        "gf": {"polygons": rings(d["floor"]["gf"], shift), "area": d["floor"]["gf_area"], "iou": ev["gf_iou"]},
        "fragments": [p for f in small for p in rings(f, shift)],
        "summary": ev,
    }


def bench_entry(path, b=None, group=None, plan=None):
    """A public benchmark sheet: no registration, its own reference (if any) and scores; plan: its curated entry."""
    folder = path.parent
    b = b or json.loads(path.read_text(encoding="utf-8"))
    rooms = [(r, shape(r["geometry"])) for r in json.loads((folder / f"{b['id']}.json").read_text(encoding="utf-8"))["rooms"]]
    ref_rooms = [shape(g) for g in b["reference"].get("rooms", [])]
    ev = {"shift_m": [0, 0], "walls_iou": b["scores"].get("wall_iou", 0), "walls_area": [0, 0], "gf_iou": 0, "gf_area_error_pct": 0,
          "rooms": {"per_room": []}, "openings": {}, "connectivity": {}}
    e = sheet_entry(b["id"], ev, folder, group or GROUPS.get(b["source"], b["source"]))
    for r, (src, g) in zip(e["rooms"], rooms):              # accuracy against this sheet's own reference rooms
        r["iou"] = round(max((iou(g, q) for q in ref_rooms), default=0.0), 2)
        r["ref"] = "reference room" if r["iou"] > 0 else None
    e["reference"] = {"rooms": [p for g in ref_rooms for p in rings(g, (0, 0))],
                      "walls": [p for g in b["reference"].get("walls", []) for p in rings(shape(g), (0, 0))]}
    e["summary"] = {"kind": "bench", "scale": f"{b['px_per_m']:.1f} px/m – {b['scale_method']}", "scores": b["scores"],
                    "search": b.get("scale_search")}
    if plan:
        e["title"] = plan["title"]
        e["curated"] = curated_info(plan)
    return e


# ---------- curated set ----------

def curated_index():
    """(plans by id, plans by (bench source, sample id), category order); empty when the manifest is missing."""
    try:
        import curated
        entries = curated.plans()
        cats = list(curated.read()["meta"]["vocabularies"]["category"])
    except (OSError, ValueError, KeyError) as ex:
        print(f"no curated manifest ({type(ex).__name__}: {ex}); benchmark sheets keep their dataset groups")
        return {}, {}, []
    ids = {e["id"]: e for e in entries}
    keys = {k: e for e in entries if (k := curated.dataset_key(e))}
    return ids, keys, cats


def match(b, ids, keys):
    cid = b.get("curated_id") or (b.get("sid") if b.get("source") == "curated" else None)
    if cid in ids:
        return ids[cid]
    return keys.get((b.get("source"), b.get("sid")))


def curated_info(e):
    """What the viewer shows of a curated plan: description, licence, scale, reference and tags (one list per axis)."""
    return {"id": e["id"], "title": e["title"], "why": e["why"], "source": e["source"], "origin": e["origin"],
            "licence": e["licence"], "use": e["use"], "attribution": e["attribution"], "scale": e["scale"],
            "reference": e["reference"], "local": bool(e.get("_local")),
            "tags": {k: (e[k] if isinstance(e[k], list) else [e[k]]) for k in TAG_AXES}}


def bench_sheets(folders):
    """Bench outputs of all folders, curated plans first (category order, then manifest order), then the rest by
    dataset; the newest output per plan or sheet id."""
    ids, keys, cats = curated_index()
    rank = {pid: i for i, pid in enumerate(ids)}
    found = {}
    for folder in folders:
        for path in sorted(folder.glob("*_bench.json")) if folder.exists() else []:
            b = json.loads(path.read_text(encoding="utf-8"))
            plan = match(b, ids, keys)
            key = ("plan", plan["id"]) if plan else ("sheet", b["id"])
            m = path.stat().st_mtime
            if key not in found or m > found[key][0]:
                found[key] = (m, path, b, plan)
    order = {k: i for i, k in enumerate(GROUPS)}

    def sort_key(item):
        _, path, b, plan = item
        if plan:
            return 0, cats.index(plan["category"]) if plan["category"] in cats else len(cats), rank[plan["id"]], ""
        return 1, order.get(b["source"], 9), 0, path.name
    out = []
    for _, path, b, plan in sorted(found.values(), key=sort_key):
        group = f"Curated: {plan['category']}" if plan else None
        try:
            out.append(bench_entry(path, b, group, plan))
        except (OSError, KeyError, ValueError) as ex:      # an incomplete run: skip the sheet, keep the export going
            print(f"skipped {path.name}: {type(ex).__name__}: {ex}")
    return out


if __name__ == "__main__":
    data = {"reference": {"rooms": [{"name": n, "polygons": rings(g, (0, 0))} for n, _, g in REF_ROOMS],
                          "walls": rings(REF_WALLS, (0, 0)),
                          "openings": [p for o in REF_OPENINGS for p in rings(o, (0, 0))]},
            "sheets": []}
    if (OUT / "evaluation.json").exists():
        evals = {e["sheet"]: e for e in json.loads((OUT / "evaluation.json").read_text(encoding="utf-8"))}
        data["sheets"] = [sheet_entry(s, evals[s]) for s in ("s1", "s2", "s3") if s in evals]
    else:
        print(f"no {OUT / 'evaluation.json'}: Landgut Lohn sheets skipped (run run_pipeline.py and evaluate.py)")
    data["sheets"] += bench_sheets(BENCH_DIRS + [Path(a) for a in sys.argv[1:]])
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "viewer-data.js"
    target.write_text("window.VIEWER_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    n = sum("curated" in s for s in data["sheets"])
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB): {len(data['sheets'])} sheets, {n} from the curated set")
