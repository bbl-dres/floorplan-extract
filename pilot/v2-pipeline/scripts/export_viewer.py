"""Package sheet images, pipeline output and evaluation for viewer.html into viewer-data/ (tracked, so the viewer runs on
GitHub Pages) or, with --local, into data/out/viewer-data/ (gitignored).

    python export_viewer.py            # the core curated plans (plans.json "core": true, all publicly showable) and the
                                       # Landgut Lohn sheets, every run folder as a version -> pilot/v2-pipeline/viewer-data/
    python export_viewer.py --local    # the same plus the benchmark-only plans -> data/out/viewer-data/;
                                       # open viewer.html#data=local (served over HTTP; file:// blocks fetch)

Layout of the output folder: index.js (window.VIEWER_INDEX: one light record per sheet and version, the Landgut
reference), sheets/<plan>--<version>.json (rooms, walls, openings, ... of one version, fetched on demand) and
img/<plan>.jpg (one sheet image per plan). Files of plans or versions that no longer exist are removed.

A version is a run folder: out (v1), out-v2b (v2), out-v3 (v3), curated-vN (vN), curated-test (test); a folder named
curated-model-<name> or out-model-<name> holds the output of a benchmark model and is shown as a version of kind
"benchmark model", never as the default (the latest pipeline version is). Scans are shifted by the registration found
in evaluate.py, so the 2005 reference lines up with every sheet. A sheet is matched to its plan by `curated_id` in the
bench JSON, by source 'curated' with sid = plan id, or by its dataset sample id.
"""
import argparse
import datetime as dt
import json
import re
import shutil
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, shape

from common import DATA, PILOT
from fpeval.metrics import iou

try:
    from evaluate import REF_OPENINGS, REF_ROOMS, REF_WALLS
except FileNotFoundError:
    REF_OPENINGS, REF_ROOMS, REF_WALLS = [], [], None

OUT = DATA / "out"
VIEW = PILOT / "viewer-data"                               # tracked: the public viewer on GitHub Pages
LOCAL_VIEW = OUT / "viewer-data"                          # gitignored: with the benchmark-only plans
CURATED_DIRS = [DATA / "curated-test", DATA / "curated-v1", DATA / "curated-v2", DATA / "curated-v3"]
LANDGUT_DIRS = [OUT, DATA / "out-v2b", DATA / "out-v3"]   # v1, the v2 run after the code review (out-v2 was v2 before it) and v3
LANDGUT = "Landgut Lohn, 1. OG (BBL)"
GROUPS = {"cubicasa": "CubiCasa5K (benchmark, CC BY-NC-SA)", "cvcfp": "CVC-FP (benchmark, CC BY-NC)",
          "waffle": "WAFFLE benchmark (Wikimedia Commons)", "commons": "Swiss plans on Wikimedia Commons (public domain)"}
TAG_AXES = ("category", "drawing_type", "input", "wall_style", "era", "building_type", "content", "language", "challenges",
            "use", "source")
LIGHT = ("id", "title", "group", "plan", "version", "run", "curated", "summary", "counts", "image", "file", "extent", "triage")


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


def version_of(folder):
    """{"label", "rank", "kind"} of a run folder (see the module docstring)."""
    name = folder.name
    if m := re.fullmatch(r"out(?:-v(\d+)\w*)?", name):
        return {"label": f"v{m[1] or 1}", "rank": int(m[1] or 1), "kind": "pipeline"}
    if m := re.fullmatch(r"curated-v(\d+)\w*", name):
        return {"label": f"v{m[1]}", "rank": int(m[1]), "kind": "pipeline"}
    if name == "curated-test":
        return {"label": "test", "rank": 0, "kind": "pipeline"}
    if m := re.fullmatch(r"(?:curated|out)-model-(.+)", name):
        return {"label": m[1], "rank": 100, "kind": "benchmark model"}
    return {"label": name, "rank": 99, "kind": "other"}


def sheet_entry(sid, ev, folder=OUT, group=LANDGUT):
    """The full record of one sheet and version, and the bytes of its sheet image."""
    path = folder / f"{sid}.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    shift = np.array(ev["shift_m"])
    if ev.get("walls_iou_raw") is not None:                 # since the wall focus: walls_iou scores the regularised walls; versions
        ev = {**ev, "walls_iou": ev["walls_iou_raw"], "walls_iou_regularised": ev["walls_iou"]}   # compare on the mask, as before
    centre = {}
    rooms = []
    for r in d["rooms"]:
        g = affinity.translate(shape(r["geometry"]), *shift)
        best = max(REF_ROOMS, key=lambda rr: iou(g, rr[2])) if REF_ROOMS else None
        centre[r["id"]] = point(r["geometry"], shift)
        gross = r.get("geometry_gross")
        rooms.append({"id": r["id"], "name": r["name"] or "(no stamp)", "raw": r.get("name_raw"), "usage": r["usage"],
                      "area": r["area"], "areaNet": r.get("area_net", r["area"]), "areaGross": r.get("area_gross"),
                      "areaPolygon": r.get("area_polygon"), "areaBasis": r.get("area_basis"), "deviationPct": r.get("area_deviation_pct"),
                      "stamp": r["area_stamp"], "number": r.get("number"), "aoid": r.get("aoid"), "conf": r["confidence"],
                      "stairFlights": r.get("stair_flights", []), "reasons": r["reasons"], "neighbours": r["neighbours"],
                      "polygons": rings(r["geometry"], shift), "geometryGross": rings(gross, shift) if gross else None,
                      "label": centre[r["id"]], "iou": round(iou(g, best[2]), 2) if best else 0.0,
                      "ref": best[0] if best and iou(g, best[2]) > 0 else None,
                      "smallRegion": bool(r.get("small_region")), "review": review(r)})
    openings = [{"id": o["id"], "kind": o["kind"], "width": o["width"], "exterior": o["exterior"], "connects": o["connects"],
                 "polygons": rings(o["geometry"], shift), "c": point(o["geometry"], shift)} for o in d["openings"]
                if not shape(o["geometry"]).is_empty]
    oc = {o["id"]: o["c"] for o in openings}
    graph = [[centre[c["a"]], oc[c["opening"]], centre[c["b"]]] for c in d["connectivity"]
             if c["a"] in centre and c["b"] in centre and c["opening"] in oc]
    tri = d["sheet"]["triage"]
    image = (folder / f"{sid}_sheet.jpg").read_bytes()
    x0, y0, x1, y1 = d["sheet"]["extent"]
    small = d.get("fragments")                           # small unlabelled regions flagged for review; older outputs: fragments only
    if small is None:
        small = [r["geometry"] for r in d["rooms"] if r.get("small_region")]
    qa = d.get("qa", [])
    entry = {
        "id": sid, "title": d["sheet"]["title"], "group": group,
        "extent": [round(x0 + shift[0], 3), round(y0 + shift[1], 3), round(x1 + shift[0], 3), round(y1 + shift[1], 3)],
        "triage": {"Input class": tri["input_class"], "Graphical style (detected)": tri["graphical_style"],
                   "Scale": f"{d['sheet']['scale']['value']} – {d['sheet']['scale']['method']}", "Text layer": tri["text_layer"]},
        "rooms": rooms, "openings": openings, "graph": graph,
        "walls": {"polygons": [p for w in d["wall_polygons"] for p in rings(w, shift)], "iou": ev["walls_iou"],
                  "area": ev["walls_area"][0], "refArea": ev["walls_area"][1]},
        "stairs": [p for s in d["stairs"] for p in rings(s, shift)],
        "voids": [{"id": v.get("id"), "kind": v.get("kind"), "label": v["label"], "area": v["area"], "deducted": v["gf_deducted"],
                   "polygons": rings(v["geometry"], shift)} for v in d["voids"]],
        "gf": {"polygons": rings(d["floor"]["gf"], shift), "area": d["floor"]["gf_area"], "iou": ev["gf_iou"]},
        "agfArea": d["floor"].get("agf_area"), "sumNet": d["floor"].get("sum_net"), "sumGross": d["floor"].get("sum_gross"),
        "wallCount": len(d.get("walls", [])),
        "wallBridges": [{"line": [[round(x + shift[0], 3), round(y + shift[1], 3)] for x, y in w["centre_line"]["coordinates"]],
                         "thickness": w["thickness"]} for w in d.get("wall_bridges", [])],
        "separations": [[[round(x + shift[0], 3), round(y + shift[1], 3)] for x, y in l["centre_line"]["coordinates"]]
                        for l in d.get("separations", [])],
        "structure": [{"kind": s["kind"], "polygons": rings(s["geometry"], shift)} for s in d.get("structure", [])],
        "qa": qa,
        "run": folder.name,
        "version": {**version_of(folder), "date": dt.date.fromtimestamp(path.stat().st_mtime).isoformat()},
        "fragments": [p for f in small for p in rings(f, shift)],
        "summary": ev,
        "counts": {"rooms": len(rooms), "openings": len(openings), "qa": sum(1 for q in qa if q.get("severity"))},
    }
    return entry, image


def bench_entry(path, b=None, group=None, plan=None):
    """A curated or benchmark sheet: no registration, its own reference (if any) and scores; plan: its curated entry."""
    folder = path.parent
    b = b or json.loads(path.read_text(encoding="utf-8"))
    rooms = [(r, shape(r["geometry"])) for r in json.loads((folder / f"{b['id']}.json").read_text(encoding="utf-8"))["rooms"]]
    ref_rooms = [shape(g) for g in b["reference"].get("rooms", [])]
    scores = b["scores"]
    wall_iou = (scores.get("wall_rates") or {}).get("iou")             # the harness row (fpeval.score); older outputs: wall_iou
    ev = {"shift_m": [0, 0], "walls_iou": scores.get("wall_iou", 0) if wall_iou is None else round(wall_iou, 3),
          "walls_area": [0, 0], "gf_iou": 0, "gf_area_error_pct": 0,
          "rooms": {"per_room": []}, "openings": {}, "connectivity": {}}
    e, image = sheet_entry(b["id"], ev, folder, group or GROUPS.get(b["source"], b["source"]))
    for r, (src, g) in zip(e["rooms"], rooms):              # accuracy against this sheet's own reference rooms
        r["iou"] = round(max((iou(g, q) for q in ref_rooms), default=0.0), 2)
        r["ref"] = "reference room" if r["iou"] > 0 else None
    e["reference"] = {"rooms": [p for g in ref_rooms for p in rings(g, (0, 0))],
                      "walls": [p for g in b["reference"].get("walls", []) for p in rings(shape(g), (0, 0))]}
    shown = {k: v for k, v in scores.items() if not isinstance(v, (dict, list))}      # the flat figures; the harness row stays in the JSON
    e["summary"] = {"kind": "bench", "scale": f"{b['px_per_m']:.1f} px/m – {b['scale_method']}", "scores": shown,
                    "search": b.get("scale_search")}
    if plan:
        e["title"] = plan["title"]
        e["curated"] = curated_info(plan)
    return e, image


# ---------- curated set ----------

def curated_index():
    """(plans by id, plans by (bench source, sample id), category order); empty when the manifest is missing."""
    try:
        from fpeval.datasets import curated
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
            "reference": e["reference"], "local": bool(e.get("_local")), "core": bool(e.get("core")),
            "tags": {k: (e[k] if isinstance(e[k], list) else [e[k]]) for k in TAG_AXES}}


def wanted(plan, local):
    """Core plans always; with --local also the benchmark-only ones and the plans of plans.local.json."""
    if plan is None:
        return False
    if local:
        return True
    return bool(plan.get("core")) and plan.get("use") != "benchmark only"


def bench_sheets(folders, local=False):
    """Bench outputs of all folders for the wanted plans, in category order then manifest order, every run folder as a
    version; the newest output per plan and folder."""
    ids, keys, cats = curated_index()
    rank = {pid: i for i, pid in enumerate(ids)}
    found = {}
    for folder in folders:
        for path in sorted(folder.glob("*_bench.json")) if folder.exists() else []:
            b = json.loads(path.read_text(encoding="utf-8"))
            plan = match(b, ids, keys)
            if not wanted(plan, local):
                continue
            key = (plan["id"], folder.name)
            m = path.stat().st_mtime
            if key not in found or m > found[key][0]:
                found[key] = (m, path, b, plan)

    def sort_key(item):
        _, path, b, plan = item
        return cats.index(plan["category"]) if plan["category"] in cats else len(cats), rank[plan["id"]], version_of(path.parent)["rank"]
    out = []
    for _, path, b, plan in sorted(found.values(), key=sort_key):
        try:
            e, image = bench_entry(path, b, f"Curated: {plan['category']}", plan)
        except (OSError, KeyError, ValueError) as ex:      # an incomplete run: skip the sheet, keep the export going
            print(f"skipped {path.name}: {type(ex).__name__}: {ex}")
            continue
        e["plan"] = plan["id"]
        out.append((e, image))
    return out


def landgut_sheets():
    out = []
    for folder in LANDGUT_DIRS:                        # every run folder with an evaluation: the viewer shows them as versions
        if (folder / "evaluation.json").exists():
            evals = {e["sheet"]: e for e in json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))}
            for s in ("s1", "s2", "s3"):
                if s in evals:
                    e, image = sheet_entry(s, evals[s], folder)
                    e["plan"] = s
                    out.append((e, image))
        elif folder == OUT:
            print(f"no {OUT / 'evaluation.json'}: Landgut Lohn sheets skipped (run run_pipeline.py and evaluate.py)")
    return out


def slug(t):
    return re.sub(r"[^a-z0-9]+", "-", str(t).lower()).strip("-")


def write(sheets, target):
    """index.js, one JSON per sheet and version, one image per plan; stale files of earlier exports removed."""
    (target / "sheets").mkdir(parents=True, exist_ok=True)
    (target / "img").mkdir(parents=True, exist_ok=True)
    keep = set()
    index = []
    images = {}
    for e, image in sheets:
        name = f"{slug(e['plan'])}--{slug(e['version']['label'])}.json"
        e["file"] = f"sheets/{name}"
        e["image"] = f"img/{slug(e['plan'])}.jpg"
        if e["plan"] not in images or e["version"]["rank"] > images[e["plan"]][0]:
            images[e["plan"]] = (e["version"]["rank"], image)
        (target / e["file"]).write_text(json.dumps(e, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        keep.add(e["file"])
        index.append({k: e[k] for k in LIGHT if k in e})
    for plan, (_, image) in images.items():
        p = f"img/{slug(plan)}.jpg"
        (target / p).write_bytes(image)
        keep.add(p)
    for sub in ("sheets", "img"):
        for f in (target / sub).iterdir():
            if f"{sub}/{f.name}" not in keep:
                f.unlink()
    data = {"generated": dt.date.today().isoformat(), "sheets": index,
            "reference": {"rooms": [{"name": n, "polygons": rings(g, (0, 0))} for n, _, g in REF_ROOMS],
                          "walls": rings(REF_WALLS, (0, 0)) if REF_WALLS is not None else [],
                          "openings": [p for o in REF_OPENINGS for p in rings(o, (0, 0))]}}
    (target / "index.js").write_text("window.VIEWER_INDEX = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    plans = {e["plan"] for e, _ in sheets}
    print(f"wrote {target} ({size / 1e6:.1f} MB): {len(sheets)} sheet versions of {len(plans)} plans, {len(images)} images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--local", action="store_true", help=f"include the benchmark-only plans; write to {LOCAL_VIEW}")
    ap.add_argument("folders", nargs="*", help="more run folders with *_bench.json outputs (a version each)")
    args = ap.parse_args()
    folders = CURATED_DIRS + [Path(a) for a in args.folders]
    sheets = landgut_sheets() + bench_sheets(folders, local=args.local)
    legacy = OUT / "viewer-data.js"
    if legacy.exists():
        legacy.unlink()                                    # the single-file format of earlier exports
    write(sheets, LOCAL_VIEW if args.local else VIEW)
