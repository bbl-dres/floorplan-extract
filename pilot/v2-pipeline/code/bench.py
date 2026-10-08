"""Run pipeline stages 0-10 on sample plans from public datasets, for the viewer and a first look at unseen styles.

    python bench.py                 # all samples below
    python bench.py waffle commons  # only these sources
    python bench.py scale-eval cubicasa:60 cvcfp:40 commons landgut [--doors]   # scale cues against known scales

Sources (fpeval.datasets) and scale (the segmenter needs cfg.px_per_m):
- CubiCasa5K (CC BY-NC-SA): 100 px/m, known from the room dimension labels; reference walls, openings, rooms.
- CVC-FP (CC BY-NC): scale from the reference door symbols (0.85 m leaves); reference walls, openings, rooms.
- WAFFLE benchmark (per-image Commons licences): reference walls, doors, windows and interior as masks.
- Commons Swiss plans (public domain, CC0): no reference.
Sheets without a known scale take the consensus of the scale cues (fpx.scale: scale note, dimension strings, scale
bar, door widths; stamp areas after a first pass). Sheets with a known scale keep it, and the cues are checked
against it. All cues end up in the sheet's scale record (exported JSON). Sheets with a reference are scored by
fpeval.score (the harness row under scores, next to the keys bench wrote before: wall/door/window pixel IoU and the
best-IoU room figures). All of these are benchmark data: outputs stay in data/bench (gitignored).
"""
import json
import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import shapely
from PIL import Image

from common import DATA, MODEL
from fpeval import metrics
from fpeval.cli import cli, jsonable, record_failure, setup, utf8_stdout
from fpeval.datasets import cubicasa, commons, cvcfp, waffle
from fpeval.raster import rgb, to_working
from fpeval.score import score
from fpx import DEFAULT, Sheet, pipeline, scale
from fpx.geometry import mask_polys, to_plan

OUT = DATA / "bench"
SCALE_OUT = OUT / "scale_eval"
SOURCES = {"cubicasa": cubicasa, "cvcfp": cvcfp, "waffle": waffle, "commons": commons}
SAMPLES = {
    "cubicasa": ["high_quality/7523", "high_quality_architectural/2085", "colorful/11260", "high_quality_architectural/2530"],
    "cvcfp": ["1", "image014", "Ib_CT0601_sommaire", "IIa_GC0801", "p2"],
    "waffle": ["HouseFlrPlan", "Grundriss_Hall_house_105b", "Rzut_budynku", "Tearoom_layout", "Typical_Dogtrot_Floorplan",
               "Categories_of_Aile_Richelieu", "Complesso_del_Pio_Monte_della_Misericordia", "Hofburg_Vienna_plan"],
    "commons": commons.ids(),
}
SECOND_PASS = 0.05                              # rerun once if the stamp-area cue moves an estimated scale this much (log ratio)


def image_path(source, sid):
    return SOURCES[source].image_path(sid)


def meta_dpi(path):
    """dpi in the image file's metadata (unverified: renditions often carry a default)."""
    d = Image.open(path).info.get("dpi")
    return float(d[0]) if d and d[0] and d[0] > 1 else None


# ---------- stage 9: scale ----------

def ocr_cached(key, img, engine, cache_dir=None, dpi=None):
    """Pre-pass OCR (fpx.scale.read_text), cached as JSON per sheet and parameter set."""
    cfg = DEFAULT
    cache_dir = cache_dir or SCALE_OUT / "ocr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"c{cfg.scale_ocr_char_px:g}-{cfg.scale_ocr_small_px:g}-{cfg.scale_ocr_enlarge_px:g}_t{cfg.scale_ocr_tile}"
           f"_o{cfg.scale_ocr_overlap}_m{cfg.scale_ocr_max_side}")
    path = cache_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', key)}__{tag}.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        for t in d["items"]:
            t["box"] = tuple(t["box"])
        return d["items"], {**d["info"], "cached": True}
    items, info = scale.read_text(img, engine, cfg, dpi=dpi)
    path.write_text(json.dumps({"items": items, "info": info}, ensure_ascii=False), encoding="utf-8")
    return items, info


def scale_prepass(key, img, engine, model=None, dpi=None, dpi_source=None, meta=None, native_text=(), cache_dir=None):
    """All scale cues on the native image (OCR cached); door search only when a model is given."""
    trusted = dpi if dpi_source in ("render", "scan") else None
    texts, info = ([], None) if engine is None else ocr_cached(key, img, engine, cache_dir, trusted)
    r = scale.prepass(img, model=model, dpi=dpi, dpi_source=dpi_source, native_text=native_text, meta_dpi=meta,
                      texts=texts)
    r["ocr"] = info
    if info:
        r["seconds"]["ocr"] = info["seconds"]
    return r


def scale_record(pre, px_per_m, method):
    """Sheet.scale: the value used, how it was obtained, and every cue with the consensus (and its flags)."""
    return {"value": f"{px_per_m:.1f} px/m", "method": method, "native_px_per_m": round(float(px_per_m), 3),
            **scale.record(pre)}


# ---------- run ----------

def one_pass(source, sid, img, px_per_m, scale_rec, model, engine, cfg):
    work = to_working(img, px_per_m, cfg, ocr=True)
    sheet = Sheet(f"{source}_{sid.replace('/', '_')}", f"{source}: {sid}", sid, "raster (public benchmark)", work.img, (0.0, 0.0),
                  scale_rec, ocr_img=work.ocr_img, px_per_m=cfg.px_per_m)
    pipeline.run(sheet, model=model, engine=engine, cfg=cfg, stages=pipeline.STAGES[:-1])      # all but export
    return sheet, work


def legacy_scores(row):
    """The keys bench wrote before the shared scorer: per-class pixel IoU of the segmenter and the best-IoU room
    figures (each reference room's best prediction, counted at IoU >= 0.5)."""
    out = {}
    if row.get("classes") is not None:
        iou = metrics.class_iou(row["classes"], row["class_names"])
        out.update({f"{c}_iou": round(iou[c] or 0.0, 3) for c in ("wall", "door", "window")})
    rooms = row.get("rooms")
    if rooms:
        best = [a[2] for a in rooms["ref_assigned"]]
        out.update(rooms_ref=rooms["ref"], rooms_pred=rooms["pred"], rooms_matched=int(sum(b >= 0.5 for b in best)),
                   rooms_mean_iou=round(float(np.mean(best)), 3) if best else None)
    return out


def viewer_reference(nat, work, sheet, cfg):
    """The reference polygons per class in plan metres, for the viewer (walls, doors, windows, rooms)."""
    P = lambda g: shapely.geometry.mapping(shapely.set_precision(to_plan(sheet, g), 0.001))
    out = {}
    if nat.polys:
        for k in ("walls", "doors", "windows"):
            if k in nat.polys:
                out[k] = [P(work.tf(p)) for p in nat.polys[k]]
        if "rooms" in nat.polys:
            out["rooms"] = [P(work.tf(p)) for _, p in nat.polys["rooms"]]
    if nat.masks:
        from fpeval.raster import mask_to_working
        shape = work.img.shape[:2]
        for k in ("walls", "doors", "windows"):
            if k in nat.masks:
                out[k] = [P(p) for p in mask_polys(mask_to_working(nat.masks[k], shape, work.pad), 4)]
    return out


def run(source, sid, model, engine, cfg=DEFAULT, dataset=None, out_dir=None):
    """One sample through the scale cues, stages 0-10 and the shared scorer; writes <sheet id>.json/.dxf, the sheet
    image and <sheet id>_bench.json to out_dir. dataset: the module (or object) with native(), reference() and
    image_path() for the source (default: fpeval.datasets.<source>)."""
    dataset = dataset or SOURCES[source]
    out_dir = Path(out_dir or OUT)
    nat = dataset.native(sid)
    img, px_per_m, scale_method = nat.img, nat.px_per_m, nat.scale_method
    t0 = time.time()
    known = px_per_m is not None
    pre = scale_prepass(f"{source}_{sid}", img, engine, None if known else model, meta=meta_dpi(dataset.image_path(sid)))
    con = pre["consensus"]
    search = pre["cues"].get("door_widths", {}).get("rows")
    if known:
        scale.compare_known(con, px_per_m, scale_method)
    else:
        if not con["px_per_m"]:
            raise ValueError("no scale cue: set the scale by hand")
        px_per_m = con["px_per_m"]
        scale_method = (f"consensus of scale cues ({con['confidence']} confidence: "
                        f"{', '.join(a['cue'] for a in con['agreeing']) or 'none'} agree"
                        f"{'; disagreeing: ' + ', '.join(d['cue'] for d in con['disagreeing']) if con['disagreeing'] else ''}); to be confirmed")
    sheet, work = one_pass(source, sid, img, px_per_m, scale_record(pre, px_per_m, scale_method), model, engine, cfg)
    # stamp areas after the first pass: a further cue; an estimated scale that it moves by more than 5 % is rerun once
    stamp = scale.stamp_area_cue(sheet.scale.get("stamp_area_cue"), px_per_m)
    if stamp["candidates"]:
        con = scale.add_cue(pre, "stamp_areas", stamp)
        if known:
            scale.compare_known(con, px_per_m, scale_method)
        elif con["px_per_m"] and abs(math.log(con["px_per_m"] / px_per_m)) > SECOND_PASS:
            first = px_per_m
            px_per_m = con["px_per_m"]
            scale_method += f"; second pass after stamp areas (first pass at {first:.1f} px/m)"
            sheet, work = one_pass(source, sid, img, px_per_m, scale_record(pre, px_per_m, scale_method), model, engine, cfg)
            con["second_pass_from"] = round(first, 3)
        sheet.scale.update(scale_record(pre, px_per_m, scale_method))
    sheet.qa.extend(con["flags"])                      # scale disagreements are QA issues, never resolved silently
    pipeline.run(sheet, stages=("export",), out_dir=out_dir, cfg=cfg)
    cv2.imwrite(str(out_dir / f"{sheet.id}_sheet.jpg"), cv2.cvtColor(sheet.img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])

    scores, reference = {}, {}
    if nat.polys or nat.masks:                         # the harness row (fpeval.score) next to the old keys
        ref = dataset.reference(nat, work, cfg)
        row = score(sheet, ref, cfg, sheet.seg_label)
        scores = {**legacy_scores(row), **row}
        reference = viewer_reference(nat, work, sheet, cfg)
    entry = {"id": sheet.id, "source": source, "sid": sid, "px_per_m": px_per_m, "scale_method": scale_method,
             "scale_search": search, "scale_consensus": con, "scores": jsonable(scores), "seconds": round(time.time() - t0),
             "reference": reference}
    (out_dir / f"{sheet.id}_bench.json").write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
    return entry, sheet


# ---------- scale evaluation (review step 6b) ----------

def eval_sheets(spec):
    """(source, sid) pairs for 'cubicasa:60', 'cvcfp:40', 'commons', 'landgut'."""
    out = []
    for s in spec:
        name, _, n = s.partition(":")
        n = int(n) if n else None
        if name == "cubicasa":                           # test split in its order, two thirds architectural (as in the split)
            ids = cubicasa.ids("test")
            n = n or 60
            arch = 2 * n // 3
            take = {"high_quality_architectural": arch, "high_quality": (n - arch) // 2, "colorful": n - arch - (n - arch) // 2}
            for style, k in take.items():
                out += [("cubicasa", i) for i in ids if i.startswith(style + "/")][:k]
        elif name == "cvcfp":
            ids = cvcfp.ids()
            step = max(1, len(ids) // (n or len(ids)))
            out += [("cvcfp", i) for i in ids[::step][:n or len(ids)]]
        elif name == "commons":
            out += [("commons", i) for i in SAMPLES["commons"][:n]]
        elif name == "landgut":
            out += [("landgut", i) for i in ("s1", "s2", "s3")]
    return out


AREA_RE = re.compile(r"(\d{1,3})\s*[.,]\s*(\d{1,2})\s*m\s*[²2*]?", re.I)


def stamp_reference(rooms, texts):
    """CVC-FP reference independent of the cues: printed room areas ("12,56 m²", read by the pre-pass OCR) against the
    ground-truth room polygons (px²) they fall into; median over rooms with exactly one area stamp (at least 3)."""
    from shapely.geometry import Point
    ratios = []
    for poly in rooms:
        areas = [float(f"{m.group(1)}.{m.group(2)}") for t in texts if (m := AREA_RE.fullmatch(t["text"].strip())
                 or AREA_RE.search(t["text"]) if re.search(r"m\s*[²2*]", t["text"]) else None)
                 and poly.contains(Point((t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2))]
        if len(areas) == 1 and areas[0] >= 2:
            ratios.append(math.sqrt(poly.area / areas[0]))
    if len(ratios) < 3:
        return None, len(ratios), None
    r = np.array(ratios)
    return float(np.median(r)), len(r), float(np.median(np.abs(r / np.median(r) - 1)))


def eval_one(source, sid, engine, model, door_cue=None):
    """Cues and consensus for one sheet against its known scale (native px per m), without running the pipeline."""
    native_text, dpi, dpi_source, meta, truth, truth_note, cache = (), None, None, None, None, None, None
    if source == "landgut":                             # BBL plans: local only; outputs stay in the gitignored data folder
        from fpeval.datasets import landgut
        d = landgut.native(sid)
        img, native_text, dpi, dpi_source = d["img"], d["text"], d["dpi"], d["dpi_source"]
        truth, truth_note = d["px_per_m"], d["method"]
        cache = DATA / "scale_eval_landgut" / "ocr"
    elif source == "cubicasa":
        img = rgb(image_path(source, sid))
        truth, truth_note = cubicasa.load(sid)["px_per_m"], "CubiCasa SVG dimension marks"
        if truth is None:                                # no metric labels in the SVG: the dataset's scaled images are 100 px/m
            truth, truth_note = 100.0, "CubiCasa F1_scaled.png convention (100 px/m)"
    elif source == "cvcfp":
        nat = cvcfp.native(sid)
        img, truth, truth_note = nat.img, nat.px_per_m, nat.scale_method
        cvc_rooms = [p for _, p in nat.polys["rooms"]]
    else:
        img = rgb(image_path(source, sid))
    if source != "landgut":
        meta = meta_dpi(image_path(source, sid))
    t = time.time()
    pre = scale_prepass(f"{source}_{sid}", img, engine, model, dpi, dpi_source, meta, native_text, cache)
    if door_cue is not None:                            # a door search saved by an earlier --doors run
        scale.add_cue(pre, "door_widths", door_cue)
        pre["seconds"]["door_widths"] = door_cue.get("seconds", 0)
    con = pre["consensus"]
    door_ref = None
    if source == "cvcfp":                               # door symbols (0.85 m) are a rough reference: prefer the room areas
        door_ref = truth
        ref, n, spread = stamp_reference(cvc_rooms, pre["texts"])
        if ref:
            truth, truth_note = ref, f"printed room areas vs ground-truth room polygons ({n} rooms, MAD {spread * 100:.1f} %)"
    row = {"source": source, "sid": sid, "shape": list(img.shape[:2]), "truth": truth, "truth_note": truth_note,
           "door_reference": door_ref,
           "consensus": con, "seconds": {**pre["seconds"], "wall": round(time.time() - t, 1)}, "ocr": pre["ocr"],
           "cues": {}}
    for k, cue in pre["cues"].items():
        best = max(cue["candidates"], key=lambda c: c["weight"]) if cue["candidates"] else None
        row["cues"][k] = {"best": best, "candidates": cue["candidates"],
                          **{kk: vv for kk, vv in cue.items() if kk != "candidates"}}
    return row


def rel_err(v, truth):
    return None if not v or not truth else v / truth - 1


def eval_report(rows):
    """Coverage and accuracy per cue and for the consensus, per source."""
    lines = []
    cues = ["scale_note", "dimension_strings", "dimension_strings (precise)", "scale_bar", "door_widths", "consensus",
            "consensus (high/medium)"]

    def value(r, cue):
        if cue.startswith("consensus"):
            c = r["consensus"]
            return c["px_per_m"] if cue == "consensus" or c["confidence"] in ("high", "medium") else None
        best = r["cues"].get(cue.split(" ")[0], {}).get("best") or {}
        if cue.endswith("(precise)") and not best.get("precise"):
            return None
        return best.get("px_per_m")

    fails = []
    for source in sorted({r["source"] for r in rows}):
        rs = [r for r in rows if r["source"] == source]
        lines.append(f"\n{source}: {len(rs)} sheets (truth known for {sum(r['truth'] is not None for r in rs)})")
        lines.append(f"  {'cue':28s} {'coverage':>9s} {'med |err|':>9s} {'<=2%':>6s} {'<=5%':>6s} {'<=10%':>6s} {'>20%':>6s}")
        for cue in cues:
            vals = [(value(r, cue), r["truth"]) for r in rs]
            got = [v for v, _ in vals if v]
            errs = [abs(rel_err(v, t)) for v, t in vals if v and t]
            if not any(cue.split(" ")[0] in r["cues"] for r in rs) and not cue.startswith("consensus"):
                continue
            for r in rs:
                v = value(r, cue)
                if v and r["truth"] and abs(rel_err(v, r["truth"])) > 0.05 and "(" not in cue:
                    ev = r["consensus"] if cue == "consensus" else (r["cues"][cue]["best"] or {})
                    fails.append(f"  {source} {r['sid']} {cue}: {v:.1f} vs {r['truth']:.1f} ({rel_err(v, r['truth']) * 100:+.0f}%) "
                                 + (ev.get("evidence", "") if cue != "consensus" else
                                    f"{ev['confidence']}; agree {[a['cue'] for a in ev['agreeing']]}")[:150])
            cue = f"{cue:28s}"
            cov = f"{len(got)}/{len(vals)}"
            if errs:
                e = np.array(errs)
                lines.append(f"  {cue} {cov:>9s} {np.median(e) * 100:8.1f}% {np.mean(e <= .02) * 100:5.0f}% "
                             f"{np.mean(e <= .05) * 100:5.0f}% {np.mean(e <= .1) * 100:5.0f}% {np.mean(e > .2) * 100:5.0f}%")
            else:
                lines.append(f"  {cue} {cov:>9s}")
        confs = {}
        for r in rs:
            c = r["consensus"]["confidence"]
            e = rel_err(r["consensus"]["px_per_m"], r["truth"])
            confs.setdefault(c, []).append(e)
        for c in ("high", "medium", "low", "none"):
            if c in confs:
                es = [abs(e) for e in confs[c] if e is not None]
                lines.append(f"    consensus {c:6s}: {len(confs[c]):3d} sheets"
                             + (f", within 5%: {sum(e <= .05 for e in es)}/{len(es)}, median |err| {np.median(es) * 100:.1f}%" if es else ""))
        if any(r.get("door_reference") for r in rs):     # CVC-FP: the second, rougher reference
            for name, key in (("room-area reference", "truth"), ("door-symbol reference", "door_reference")):
                es = [abs(rel_err(r["consensus"]["px_per_m"], r[key])) for r in rs
                      if r["consensus"]["px_per_m"] and r.get(key) and r["consensus"]["confidence"] in ("high", "medium")
                      and (key == "door_reference" or r["truth"] != r["door_reference"])]
                if es:
                    lines.append(f"    consensus (high/medium) vs {name}: {len(es)} sheets, median |err| "
                                 f"{np.median(es) * 100:.1f}%, within 5%: {sum(e <= .05 for e in es)}")
            both = [(r["truth"], r["door_reference"]) for r in rs if r.get("door_reference") and r["truth"] != r["door_reference"]]
            if both:
                d = [t / dr - 1 for t, dr in both]
                lines.append(f"    room-area / door-symbol reference: median {np.median(d) * 100:+.1f}%, "
                             f"5-95 % {np.percentile(d, 5) * 100:+.1f}..{np.percentile(d, 95) * 100:+.1f}% ({len(d)} sheets)")
        flagged = sum(bool(r["consensus"]["flags"]) for r in rs)
        wrong = [r for r in rs if r["truth"] and r["consensus"]["px_per_m"] and abs(rel_err(r["consensus"]["px_per_m"], r["truth"])) > .05]
        lines.append(f"    flagged: {flagged}/{len(rs)}; wrong by >5%: {len(wrong)}, of which flagged or low: "
                     f"{sum(bool(r['consensus']['flags']) or r['consensus']['confidence'] == 'low' for r in wrong)}")
        secs = [r["seconds"].get("ocr", 0) or 0 for r in rs]
        lines.append(f"    seconds per sheet: OCR median {np.median(secs):.1f}, cues median "
                     f"{np.median([r['seconds'].get('text_cues', 0) for r in rs]):.2f}"
                     + (f", door search median {np.median([r['seconds'].get('door_widths', 0) for r in rs]):.1f}"
                        if any('door_widths' in r['seconds'] for r in rs) else ""))
    if fails:
        lines += ["\nerrors above 5 %:"] + fails
    return "\n".join(lines)


def scale_eval(args):
    import torch
    from fpx.segment import load_model
    doors = "--doors" in args
    reuse = "--reuse-doors" in args                     # recompute the text cues, take the door cue from a --doors run
    spec = [a for a in args if not a.startswith("--")]
    SCALE_OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    model = load_model(MODEL) if doors else None
    engine = pipeline.ocr_engine()
    rows, failed = [], []
    for source, sid in eval_sheets(spec):
        out_dir = DATA / "scale_eval_landgut" if source == "landgut" else SCALE_OUT
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{source}_{sid.replace('/', '_')}{'_doors' if doors or reuse else ''}.json"
        door_cue = None
        if reuse:
            if not path.exists():
                continue
            saved = json.loads(path.read_text(encoding="utf-8"))
            rows_ = saved["cues"]["door_widths"].get("rows")
            cands = ([scale.door_choice(rows_)[2]] if rows_ else saved["cues"]["door_widths"]["candidates"])
            door_cue = {"candidates": cands, "rows": rows_, "seconds": saved["seconds"].get("door_widths", 0)}
        try:
            row = eval_one(source, sid, engine, model, door_cue)
        except Exception as ex:                          # keep going; the failures are listed at the end
            record_failure(failed, f"{source} {sid}", ex)
            continue
        path.write_text(json.dumps(row, ensure_ascii=False, default=float), encoding="utf-8")
        rows.append(row)
        c = row["consensus"]
        e = rel_err(c["px_per_m"], row["truth"])
        print(f"{source:9s} {sid:36s} truth {row['truth'] or 0:8.2f}  consensus {c['px_per_m'] or 0:8.2f} "
              f"({c['confidence']:6s}) err {'' if e is None else f'{e * 100:+6.1f}%':>7s}  "
              + "  ".join(f"{k[:4]}={(v['best'] or {}).get('px_per_m', '-')}" for k, v in row["cues"].items())
              + f"  {row['seconds'].get('wall')} s", flush=True)
    report = eval_report(rows)
    print(report)
    if failed:
        print(f"{len(failed)} sheets failed: " + ", ".join(f"{f['id']} ({f['error']})" for f in failed))
    tag = "_".join(a.replace(":", "") for a in spec) + ("_doors" if doors or reuse else "")
    pub = [r for r in rows if r["source"] != "landgut"]
    if pub:
        (SCALE_OUT / f"report_{tag}.txt").write_text(eval_report(pub), encoding="utf-8")
    return rows


def main(argv=None):
    from fpx.segment import load_model
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["scale-eval"]:
        utf8_stdout()
        scale_eval(argv[1:])
        return
    ap = cli(__doc__.split("\n")[0], out=OUT, threads=12)
    ap.add_argument("sources", nargs="*", help="only these sources (default: all)")
    a = ap.parse_args(argv)
    cfg = setup(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(a.sources)
    model = load_model(a.model)
    engine = pipeline.ocr_engine()
    failed = []
    for source, ids in SAMPLES.items():
        if only and source not in only:
            continue
        for sid in ids:
            try:
                e, sheet = run(source, sid, model, engine, cfg, out_dir=out)
            except Exception as ex:                      # keep going; the failures are listed at the end
                record_failure(failed, f"{source} {sid}", ex)
                continue
            s = e["scores"]
            shown = {k: s[k] for k in ("wall_iou", "door_iou", "window_iou", "rooms_matched", "rooms_ref") if k in s}
            print(f"{e['id']:55s} {e['px_per_m']:7.1f} px/m  {len(sheet.rooms):3d} rooms  {shown}  {e['seconds']} s | {e['scale_method'][:70]}", flush=True)
    if failed:
        (out / "failed.json").write_text(json.dumps(failed, indent=1), encoding="utf-8")
        print(f"{len(failed)} sheets failed (see {out / 'failed.json'})")


if __name__ == "__main__":
    main()
