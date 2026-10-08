"""Benchmark the pilot v2 segmenter and the pipeline on CubiCasa5K test plans (real raster plans, three styles).

    python cubicasa_eval.py [split] [limit] [--out DIR] [--model PATH] [--config cfg.json] [--no-cache]
    # default: test, all 400 plans, data/cubicasa/, model v2

Zero-shot: the segmenter never saw CubiCasa5K. Every plan goes through stages 3-9 (fpeval.harness.run_sheet, the
segmenter output cached in data/harness/cache/cubicasa) and is scored by fpeval.score, the one protocol of every
benchmark: the segmenter's pixel confusion (as predicted), doors and windows by centre within 0.5 m one to one,
rooms one to one at IoU >= 0.5 (Hungarian), walls from the cleaned mask (IoU, boundary F, centre lines).

Changes with the shared scorer (fpeval, October 2026), kept in mind when reading older results_<split>.json:
- rooms are every region of room_min_area (0.25 m²) or more after stages 3b-9, rooms and flagged fragments alike
  (before: rooms of 1 m² or more after stage 6);
- outdoor spaces are ignored, so a prediction on a balcony no longer counts against precision (before: counted);
- openings are matched one to one (Hungarian) instead of greedily in reference order; doors and windows "found by
  any kind" come from the confusion matrix.
The published-protocol numbers stay as extra fields: the pooled confusion matrix (cm), per-class pixel IoU, the
wall ∪ door ∪ window IoU, door and window recall by any kind, and the earlier best-IoU room figures under legacy.
Per style group and overall, with the harness summary (summary, by_area_type) next to the old keys.
CubiCasa5K is CC BY-NC-SA 4.0: benchmark use only, outputs stay local.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from common import DATA
from fpeval import metrics
from fpeval.cli import cli, jsonable, record_failure, setup
from fpeval.datasets import cubicasa
from fpeval.harness import Context, run_sheet
from fpeval.raster import PALETTE, overlay
from fpeval.score import score
from fpx.model import CLASSES, DOOR, WALL, WINDOW

OUT = DATA / "cubicasa"
STRUCTURE = (WALL, DOOR, WINDOW)                # the classes whose union is the "structure" pixel IoU
OVERLAYS_PER_STYLE = 4


def run(plan, ctx):
    """One plan: the harness row plus the fields cubicasa_eval wrote before (same names, same meaning)."""
    sheet, ref, info, key = cubicasa.build(plan, ctx)
    ref, seg, times = run_sheet(sheet, ref, ctx, key)
    row = score(sheet, ref, ctx.cfg, seg)
    op, rooms = row["openings"], row["rooms"]
    conf = op["confusion"]
    legacy = {"recall50": int(sum(a[2] >= 0.5 for a in rooms["ref_assigned"])),
              "precision50": int(sum(b >= 0.5 for b in rooms["pred_best_iou"])),
              "mean_iou": float(np.mean([a[2] for a in rooms["ref_assigned"]])) if rooms["ref"] else None}
    out = {"plan": plan, "id": plan, "style": info["style"], "size": info["size"], "cm": row["classes"],
           "doors": [op["door"]["ref"] - conf["door"]["none"], op["door"]["ref"]],
           "windows": [op["window"]["ref"] - conf["window"]["none"], op["window"]["ref"]],
           "openings_pred": op["pred_any"], "openings_matched": op["found_any_kind"], "passages": row["passages"],
           "rooms_ref": rooms["ref"], "rooms_pred": rooms["pred"], "rooms_recall50": rooms["matched"],
           "rooms_precision50": rooms["matched"], "rooms_mean_iou": rooms["mean_iou_ref"],
           "area_err": [abs(e) for e in rooms["area_err_pct"]],
           "rooms_matching": "one-to-one", "rooms_pair_iou": rooms["pair_iou"], "area_err_signed": rooms["area_err_pct"],
           "area_err_m2": rooms["area_err_m2"], "rooms_legacy_best_iou": legacy, "openings_typed": op,
           "seconds": times, **row}
    return out, sheet, ref


def structure_iou(cm):
    """Pixel IoU of wall ∪ door ∪ window (one "structure" class) from a pooled confusion matrix."""
    cm = np.asarray(cm)
    s = list(STRUCTURE)
    inter = cm[np.ix_(s, s)].sum()
    union = cm[s].sum() + cm[:, s].sum() - inter
    return float(inter / union) if union else None


def summary(rows):
    rows = [r for r in rows if "error" not in r]
    k = len(CLASSES)
    cm = np.sum([np.array(r["cm"]) for r in rows], axis=0) if rows else np.zeros((k, k))
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    s = lambda key, i: sum(r[key][i] for r in rows)
    errs = [e for r in rows for e in r["area_err"]]
    rnd = lambda v, n=3: None if v is None else round(float(v), n)
    out = {"plans": len(rows), "pixel_iou": {c: round(float(inter[i] / max(union[i], 1)), 3) for i, c in enumerate(CLASSES)},
           "pixel_iou_wall_door_window": rnd(structure_iou(cm)),
           "door_recall": round(s("doors", 0) / max(s("doors", 1), 1), 3), "window_recall": round(s("windows", 0) / max(s("windows", 1), 1), 3),
           "opening_precision": round(sum(r["openings_matched"] for r in rows) / max(sum(r["openings_pred"] for r in rows), 1), 3),
           "room_recall50": round(sum(r["rooms_recall50"] for r in rows) / max(sum(r["rooms_ref"] for r in rows), 1), 3),
           "room_precision50": round(sum(r["rooms_precision50"] for r in rows) / max(sum(r["rooms_pred"] for r in rows), 1), 3),
           "room_mean_iou": rnd(np.mean([r["rooms_mean_iou"] for r in rows if r["rooms_mean_iou"] is not None])) if rows else None,
           "median_area_err_pct": round(float(np.median(errs)), 1) if errs else None}
    if rows:
        signed = [e for r in rows for e in r["area_err_signed"]]
        m2 = [abs(e) for r in rows for e in r["area_err_m2"]]
        pair = [e for r in rows for e in r["rooms_pair_iou"]]
        walls = metrics.mask_rates(metrics.add_counts([r["walls"] for r in rows]))
        ops = metrics.opening_rates(metrics.add_counts([r["openings_typed"] for r in rows]))
        legacy = lambda k: sum(r["rooms_legacy_best_iou"][k] for r in rows)
        out.update({"room_matching": "one-to-one (Hungarian on IoU), IoU >= 0.5; rooms and fragments of >= 0.25 m²",
                    "room_mean_iou_matched": rnd(np.mean(pair)) if pair else None,
                    "median_area_err_signed_pct": rnd(np.median(signed), 1) if signed else None,
                    "median_abs_area_err_m2": rnd(np.median(m2), 2) if m2 else None,
                    "wall_iou_cleaned": rnd(walls["iou"]), "wall_boundary_f": rnd(walls["boundary_f"]),
                    "wall_centreline_p": rnd(walls["centreline_p"]), "wall_centreline_r": rnd(walls["centreline_r"]),
                    "door_precision_typed": rnd(ops["door"]["precision"]), "door_recall_typed": rnd(ops["door"]["recall"]),
                    "window_precision_typed": rnd(ops["window"]["precision"]), "window_recall_typed": rnd(ops["window"]["recall"]),
                    "door_as_window": rnd(ops["door_as_window"]), "window_as_door": rnd(ops["window_as_door"]),
                    "legacy_best_iou": {"room_recall50": round(legacy("recall50") / max(sum(r["rooms_ref"] for r in rows), 1), 3),
                                        "room_precision50": round(legacy("precision50") / max(sum(r["rooms_pred"] for r in rows), 1), 3),
                                        "room_mean_iou": rnd(np.mean([r["rooms_legacy_best_iou"]["mean_iou"] for r in rows
                                                                      if r["rooms_legacy_best_iou"]["mean_iou"] is not None]))},
                    "harness": metrics.summarise(rows)})
    return out


def report(name, s):
    print(f"{name:28s} {s['plans']:3d} plans | IoU " + " ".join(f"{c} {v:.2f}" for c, v in s["pixel_iou"].items() if c != "background")
          + f" | doors {s['door_recall']:.2f} windows {s['window_recall']:.2f} prec {s['opening_precision']:.2f}"
          + f" | rooms R {s['room_recall50']:.2f} P {s['room_precision50']:.2f} mIoU {s['room_mean_iou']} area err {s['median_area_err_pct']}%")
    if "wall_boundary_f" in s:
        lg = s["legacy_best_iou"]
        print(f"{'':28s}     one-to-one rooms: mIoU of matches {s['room_mean_iou_matched']}, signed area err {s['median_area_err_signed_pct']}% "
              f"(|m²| {s['median_abs_area_err_m2']}) | best-IoU (old) R {lg['room_recall50']} P {lg['room_precision50']} mIoU {lg['room_mean_iou']}"
              f" | walls BF {s['wall_boundary_f']} CL P {s['wall_centreline_p']} R {s['wall_centreline_r']}"
              f" | doors P {s['door_precision_typed']} R {s['door_recall_typed']}, windows P {s['window_precision_typed']} "
              f"R {s['window_recall_typed']}, door as window {s['door_as_window']}, window as door {s['window_as_door']}")


def main(argv=None):
    ap = cli(__doc__.split("\n")[0], out=OUT, threads=12)
    ap.add_argument("split", nargs="?", default="test", help="CubiCasa5K split (default test)")
    ap.add_argument("limit", nargs="?", type=int, default=None, help="first N plans only")
    ap.add_argument("--no-cache", action="store_true", help="do not read or write cached segmenter outputs")
    a = ap.parse_args(argv)
    setup(a)
    a.n = a.limit
    out = Path(a.out)                                # write results and overlays elsewhere (e.g. a short test run)
    out.mkdir(parents=True, exist_ok=True)
    ctx = Context(a, mode="cubicasa")
    plans = cubicasa.ids(a.split, a.limit)
    rows, shown, t0 = [], {}, time.time()
    for i, (plan, build) in enumerate(cubicasa.items(ctx), 1):
        try:
            r, sheet, ref = run(plan, ctx)
        except Exception as e:                       # keep going; the row lists the error
            record_failure(rows, plan, e)
            continue
        rows.append(r)
        if shown.get(r["style"], 0) < OVERLAYS_PER_STYLE:    # a few overlays per style: plan | prediction | reference
            shown[r["style"]] = shown.get(r["style"], 0) + 1
            tile = np.hstack([sheet.img, overlay(sheet, names=False), PALETTE[ref["label"]]])
            cv2.imwrite(str(out / f"{r['style']}_{plan.split('/')[1]}.png"), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
        if i % 25 == 0:
            print(f"{i}/{len(plans)} plans, {(time.time() - t0) / i:.1f} s/plan", flush=True)
    ok = [r for r in rows if "error" not in r]
    result = {"split": a.split, "all": summary(ok),
              "by_style": {s: summary([r for r in ok if r["style"] == s]) for s in sorted({r["style"] for r in ok})},
              "summary": metrics.summarise(rows), "by_area_type": metrics.by_type(ok),
              "failed": [{"id": r["id"], "error": r["error"]} for r in rows if "error" in r],
              "model_sha1": ctx.model_sha, "per_plan": rows}
    (out / f"results_{a.split}.json").write_text(json.dumps(jsonable(result), indent=1), encoding="utf-8")
    for name, s in [("all", result["all"]), *result["by_style"].items()]:
        report(name, s)
    print(f"{len(ok)} plans, {len(rows) - len(ok)} failed -> {out / f'results_{a.split}.json'}")
    return result


if __name__ == "__main__":
    main()
