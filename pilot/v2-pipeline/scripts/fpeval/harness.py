"""The evaluation harness on public data (review step 1c): scores the whole pipeline (segmenter and post-processing)
and the post-processing alone, so that thresholds are swept and frozen here instead of on BBL sheets.

    python harness.py oracle [--n N]                    # Swiss Dwellings test floors with perfect labels: post-processing alone
    python harness.py render [--n N] [--seed S] [--ocr [K]]   # the same floors rendered whole in random styles: full pipeline
    python harness.py cvcfp [--n N]                     # all CVC-FP scans, scale from the reference doors
    python harness.py waffle [--n N] [--scale proposed|reference]   # WAFFLE benchmark images (masks only)

Options: --config cfg.json (fpx.Config.from_file, for sweeps), --out file, --ids a,b (only these sheets), --model path,
--threads 8, --no-cache. Writes data/harness/<mode>.json (arguments, configuration, summary, groups, one row per
sheet) and prints a summary table.

Modes (one dataset module each, fpeval.oracle and fpeval.datasets.render/cvcfp/waffle):
- oracle: data/floors-test.pkl rasterised into perfect labels; stages walls..qa. Floors are taken in a fixed shuffled
  order (oracle.ORDER_SEED), the same in render mode, so --n N gives the same N floors spread over sites in both.
- render: each test floor drawn whole by synth.Renderer at exactly px_per_m, stored in data/harness/renders/seed<S>/
  and reused; then triage, preprocessing, segmentation and stages walls..qa. OCR is off (about 30 s per sheet); --ocr K
  turns the text stage on for the first K sheets. Floors wider than --max-side pixels are skipped (memory).
- cvcfp: data/benchmark/cvc-fp (CC BY-NC, evaluation only), scale from the reference door symbols.
- waffle: data/benchmark/waffle (per-image Commons licences; masks only), scale proposed from detected door widths
  or, with --scale reference, from the reference door masks where possible.

Metrics (fpeval.score): rooms one-to-one at IoU >= 0.5 (rooms and flagged fragments are predictions), walls from the
cleaned wall mask (pixel IoU, boundary F at 2 px, centre lines within 5 cm), class pixel IoU of the segmenter,
openings by centre within 0.5 m, connectivity mapped through the room matching, GF gross (with deducted voids).
Segmenter outputs are cached in data/harness/cache/<mode>, keyed by model, input, renderer and pre-segmentation
settings, so that a threshold sweep reruns only the post-processing; the v2 head maps are cached with them.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from common import DATA, MODEL
from fpeval import metrics, oracle
from fpeval.cli import Skip, jsonable, record_failure, utf8_stdout
from fpeval.datasets import CACHE, file_sha1
from fpeval.datasets import cvcfp, render, waffle
from fpeval.score import deskew_reference, score
from fpx import DEFAULT, Config, pipeline
from fpx.model import HEADS

OUT = DATA / "harness"
PRE = ("triage", "preprocess")
POST = ("walls", "openings", "stairs", "rooms", "attributes", "derived", "qa")
PRE_SEG_FIELDS = ("px_per_m", "seg_", "skew_", "deskew_", "solid_ink_", "flat_scan_")   # settings that change the segmenter input or output
SAVE_EVERY = 10                                 # write partial results every this many sheets
HEAD_FIELDS = tuple(f"{h}_prob" for h in HEADS) # v2 head maps on the Sheet, cached next to label and prob
ITEMS = {"oracle": oracle.items, "render": render.items, "cvcfp": cvcfp.items, "waffle": waffle.items}
GROUPS = {"render": ("era", "wall_style", "door_style", "window_style"), "cvcfp": ("group",), "waffle": ()}


class Context:
    """What a benchmark run holds: the configuration, the model and its hash, the OCR engine and how many sheets
    still get OCR, the renderer, and the segmenter cache folder (none for oracle labels or --no-cache)."""

    def __init__(self, args, mode=None):
        self.args = args
        self.mode = mode or args.mode
        self.cfg = Config.from_file(args.config) if getattr(args, "config", None) else DEFAULT
        self.model = self.engine = self.renderer = None
        self.model_sha = self.synth_sha = None
        self.ocr_left = 0 if self.mode == "oracle" else int(getattr(args, "ocr", 0) or 0)
        self.cache = None if getattr(args, "no_cache", False) or self.mode == "oracle" else CACHE / self.mode
        if self.mode != "oracle":
            import torch
            from fpx.segment import load_model
            torch.set_num_threads(args.threads)
            path = Path(args.model)
            self.model = load_model(path)
            self.model_sha = file_sha1(path, 12)
        if self.ocr_left:
            self.engine = pipeline.ocr_engine()
        pre = {f: v for f, v in self.cfg.to_dict().items() if f.startswith(PRE_SEG_FIELDS)}
        self.pre_key = json.dumps(pre, sort_keys=True)

    def take_ocr(self):
        """Whether the next sheet gets the text stage (--ocr K: the first K sheets)."""
        if self.engine is not None and self.ocr_left > 0:
            self.ocr_left -= 1
            return True
        return False

    def cache_path(self, sid, key):
        h = hashlib.sha1(json.dumps([self.model_sha, self.pre_key, key], sort_keys=True).encode()).hexdigest()[:12]
        return self.cache / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', sid)}-{h}.npz"


def run_sheet(sheet, ref, ctx, key):
    """Stages up to segmentation (segmenter output from the cache when present), then walls..qa. Returns the
    reference (moved if the sheet was deskewed), the segmenter label and the seconds per step."""
    cfg, times = ctx.cfg, {}
    if ctx.mode == "oracle":
        seg = sheet.label.copy()
        times.update(pipeline.run(sheet, cfg=cfg, stages=POST))
        return ref, seg, times
    stages = PRE + (("text",) if sheet.ocr_img is not None and ctx.engine is not None else ())
    times.update(pipeline.run(sheet, engine=ctx.engine, cfg=cfg, stages=stages))
    if "deskew" in sheet.meta:
        ref = deskew_reference(ref, sheet.meta["deskew"]["matrix"], sheet.img.shape[:2])
    path = ctx.cache_path(sheet.id, key) if ctx.cache is not None else None
    if path is not None and path.exists():
        z = np.load(path)
        prob = z["prob"]
        sheet.label, sheet.prob = z["label"], (prob.astype(np.float32) / 255 if prob.dtype == np.uint8 else prob.astype(np.float32))
        sheet.seg_label = sheet.label.copy()
        for name in HEAD_FIELDS:                        # head maps of the v2 segmenter; older cache files have none
            if name in z.files:
                setattr(sheet, name, z[name].astype(np.float32))
        times["segment"] = 0.0
        times["cached"] = True
    else:
        times.update(pipeline.run(sheet, model=ctx.model, cfg=cfg, stages=("segment",)))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            heads = {name: getattr(sheet, name).astype(np.float16) for name in HEAD_FIELDS if getattr(sheet, name, None) is not None}
            np.savez_compressed(path, label=sheet.label, prob=sheet.prob.astype(np.float16), **heads)   # float16: scores stay as computed
    seg = sheet.label.copy()                            # stage 4 edits the label (isolated opening blobs)
    times.update(pipeline.run(sheet, cfg=cfg, stages=POST))
    return ref, seg, times


# ---------- report ----------

def fmt(v, spec=".2f"):
    return "-" if v is None else format(v, spec)


def table(named):
    """One line per summary: rooms, walls, openings, connections, GF."""
    head = (f"{'':24s} {'sheets':>6s} | {'rooms R':>7s} {'P':>5s} {'mIoU':>5s} {'err%':>6s} {'|err%|':>6s} {'|m2|':>5s} | "
            f"{'wall IoU':>8s} {'BF':>5s} {'CL-P':>5s} {'CL-R':>5s} | {'door P':>6s} {'R':>5s} {'win P':>5s} {'R':>5s} {'d>w':>5s} {'w>d':>5s} | "
            f"{'conn F1':>7s} {'F1m':>5s} | {'GF IoU':>6s} {'err%':>6s}")
    lines = [head, "-" * len(head)]
    for name, s in named:
        ro, wa, op, co, gf = (s.get(k) or {} for k in ("rooms", "walls", "openings", "connectivity", "gf"))
        lines.append(f"{name[:24]:24s} {s['sheets']:6d} | {fmt(ro.get('recall')):>7s} {fmt(ro.get('precision')):>5s} {fmt(ro.get('mean_iou')):>5s} "
                     f"{fmt(ro.get('area_err_pct_median'), '+.1f'):>6s} {fmt(ro.get('area_err_pct_abs_median'), '.1f'):>6s} "
                     f"{fmt(ro.get('area_err_m2_abs_median')):>5s} | {fmt(wa.get('iou')):>8s} {fmt(wa.get('boundary_f')):>5s} "
                     f"{fmt(wa.get('centreline_p')):>5s} {fmt(wa.get('centreline_r')):>5s} | "
                     f"{fmt((op.get('door') or {}).get('precision')):>6s} {fmt((op.get('door') or {}).get('recall')):>5s} "
                     f"{fmt((op.get('window') or {}).get('precision')):>5s} {fmt((op.get('window') or {}).get('recall')):>5s} "
                     f"{fmt(op.get('door_as_window')):>5s} {fmt(op.get('window_as_door')):>5s} | "
                     f"{fmt(co.get('f1')):>7s} {fmt(co.get('f1_matched')):>5s} | {fmt(gf.get('iou_median')):>6s} "
                     f"{fmt(gf.get('area_err_pct_median'), '+.1f'):>6s}")
    return "\n".join(lines)


def type_table(types, top=14):
    lines = [f"{'area type':26s} {'ref':>5s} {'found':>5s} {'recall':>6s} | missed: {'merged':>6s} {'oversz':>6s} {'split':>5s} "
             f"{'missing':>7s} {'outside':>7s} | merged with"]
    for t, d in list(types.items())[:top]:
        lines.append(f"{str(t)[:26]:26s} {d['ref']:5d} {d['matched']:5d} {fmt(d['recall']):>6s} |         {d['merged']:6d} {d['oversized']:6d} "
                     f"{d['split']:5d} {d['missing']:7d} {d['outside building']:7d} | " + ", ".join(f"{k} {v}" for k, v in d["merged_with"].items()))
    return "\n".join(lines)


def progress(n, row):
    """One console line per sheet."""
    ro, wa, op = row.get("rooms") or {}, row.get("wall_rates") or {}, row.get("openings") or {}
    print(f"{n:4d} {str(row['id']):>34s} {row['size'][0]:5d}x{row['size'][1]:<5d} "
          f"rooms {ro.get('matched', '-')}/{ro.get('ref', '-')} (pred {ro.get('pred', '-')}) "
          f"wall IoU {fmt(wa.get('iou'))} BF {fmt(wa.get('boundary_f'))} | "
          f"doors {(op.get('door') or {}).get('matched', '-')}/{(op.get('door') or {}).get('ref', '-')} "
          f"windows {(op.get('window') or {}).get('matched', '-')}/{(op.get('window') or {}).get('ref', '-')} "
          f"| GF IoU {fmt((row.get('gf') or {}).get('iou'))} | {row['seconds']['total']:.1f} s"
          + (" (cached)" if row["seconds"].get("cached") else ""), flush=True)


def write(args, ctx, rows, skipped, runtime, partial=False):
    """Summarise and write data/harness/<mode>.json (also during the run, marked partial)."""
    ok = [r for r in rows if "error" not in r]
    groups = {g: {v: metrics.summarise([r for r in ok if str(r.get(g)) == v]) for v in sorted({str(r.get(g)) for r in ok})}
              for g in GROUPS.get(args.mode, ())}
    seconds = [r["seconds"]["total"] for r in ok]
    result = {"mode": args.mode, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "partial": partial,
              "args": {k: v for k, v in vars(args).items()},
              "config_overrides": {k: v for k, v in ctx.cfg.to_dict().items() if DEFAULT.to_dict().get(k) != v},
              "model_sha1": ctx.model_sha, "synth_sha1": sorted({str(r["synth_sha1"]) for r in ok if r.get("synth_sha1")}),
              "runtime_s": round(runtime, 1), "seconds_per_sheet_median": metrics.med(seconds),
              "seconds_segment_median": metrics.med([r["seconds"].get("segment") for r in ok if "segment" in r["seconds"]]),
              "skipped": skipped, "failed": [{"id": r["id"], "error": r["error"]} for r in rows if "error" in r],
              "summary": metrics.summarise(rows), "groups": groups,
              "by_area_type": metrics.by_type(ok) if args.mode in ("oracle", "render", "cvcfp") else {}, "rows": rows}
    out = Path(args.out) if args.out else OUT / f"{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(result), ensure_ascii=False), encoding="utf-8")
    return result, out


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=sorted(ITEMS))
    ap.add_argument("--n", type=int, default=None, help="first n sheets (default: all); Swiss Dwellings floors in a fixed shuffled order, the same for oracle and render")
    ap.add_argument("--ids", default=None, help="comma-separated sheet ids (floor ids, CVC-FP or WAFFLE names)")
    ap.add_argument("--config", default=None, help="JSON thresholds (fpx.Config.from_file)")
    ap.add_argument("--seed", type=int, default=0, help="render: style seed (with the floor index)")
    ap.add_argument("--ocr", type=int, nargs="?", const=5, default=0, help="run the text stage (OCR) on the first K sheets (default 5)")
    ap.add_argument("--scale", choices=("proposed", "reference"), default="proposed", help="waffle: scale source")
    ap.add_argument("--ref-min-room", type=float, default=oracle.REF_MIN_ROOM, help="oracle/render: smallest scored reference room (m²)")
    ap.add_argument("--max-side", type=int, default=4500, help="render/waffle: skip sheets with a longer side (px)")
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true", help="do not read or write cached segmenter outputs")
    ap.add_argument("--rerender", action="store_true", help="render: draw the floors again instead of using the stored renders")
    ap.add_argument("--out", default=None, help="output JSON (default data/harness/<mode>.json)")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.mode == "oracle":
        args.ocr = 0
    utf8_stdout()
    ctx = Context(args)
    only = set(args.ids.split(",")) if args.ids else None
    rows, skipped, t0 = [], [], time.time()
    for sid, build in ITEMS[args.mode](ctx):
        if only and sid not in only and f"sd{sid}" not in only:
            continue
        t = time.time()
        try:
            sheet, ref, info, key = build()
            prep = time.time() - t
            ref, seg, times = run_sheet(sheet, ref, ctx, key)
            row = {"id": sheet.id, **info, **score(sheet, ref, ctx.cfg, seg)}
            row["seconds"] = {"prepare": round(prep, 1), **times, "total": round(time.time() - t, 1)}
        except Skip as e:
            skipped.append({"id": sid, "reason": str(e)})
            print(f"skip {sid}: {e}", flush=True)
            continue
        except Exception as e:                           # keep going; failures are listed in the output
            record_failure(rows, sid, e, time.time() - t)
        else:
            rows.append(row)
            progress(len(rows), row)
        if len(rows) % SAVE_EVERY == 0:
            write(args, ctx, rows, skipped, time.time() - t0, partial=True)
    runtime = time.time() - t0
    result, out = write(args, ctx, rows, skipped, runtime)
    summary, groups, types = result["summary"], result["groups"], result["by_area_type"]
    seconds = [r["seconds"]["total"] for r in rows if "error" not in r]

    print(f"\n{args.mode}: {summary['sheets']} sheets, {summary['failed']} failed, {len(skipped)} skipped, "
          f"{runtime / 60:.1f} min ({metrics.med(seconds) or 0:.1f} s per sheet, median) -> {out}")
    print(table([("all", summary)] + [(f"{g}={v}", s) for g, d in groups.items() for v, s in d.items()]))
    if summary.get("pixel_iou"):
        print("segmenter pixel IoU: " + "  ".join(f"{k} {fmt(v)}" for k, v in summary["pixel_iou"].items()))
    co = summary.get("connectivity")
    if co:
        print(f"connections: open passages found {fmt(co.get('other_found'))}; predicted edges by opening kind: "
              + "; ".join(f"{k} " + ", ".join(f"{n} {v}" for n, v in d.items()) for k, d in co["counts"].get("by_opening", {}).items()))
    if summary.get("ocr"):
        print("OCR subset: " + "  ".join(f"{k} {fmt(v) if isinstance(v, float) else v}" for k, v in summary["ocr"].items()))
    if types:
        print("\n" + type_table(types))
    return result
