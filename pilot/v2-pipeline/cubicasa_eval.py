"""Benchmark the pilot v2 segmenter and pipeline stages 3-6 on CubiCasa5K test plans (real raster plans, three styles).

    python cubicasa_eval.py [split] [limit] [--out DIR] [--model PATH]   # default: test, all 400 plans, data/cubicasa/, v1 model

Zero-shot: the segmenter never saw CubiCasa5K. Per style group: pixel IoU per class, opening recall and precision
(centres within 0.5 m), and rooms matched one to one at IoU >= 0.5 (Hungarian, metrics.match_rooms; outdoor spaces
excluded). Added with the harness (review step 1c): signed area errors in % and m², wall boundary F-score and centre
lines, door and window precision and recall per kind with their confusion, and the earlier best-IoU room figures
under *legacy*. CubiCasa5K is CC BY-NC-SA 4.0: benchmark use only, outputs stay local.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import shapely
import torch
from shapely import affinity

import cubicasa
import metrics
from common import CLASSES, DATA, PX_PER_M, WALL
from fpx import Sheet, pipeline
from fpx.segment import load_model

OUT = DATA / "cubicasa"
SCALE = 100.0                                   # px per m of F1_scaled.png (all plans with metric labels: 99.9-100.1)
PAD = PX_PER_M                                  # 1 m white margin: CubiCasa crops end at the outer wall
PALETTE = np.array([[255, 255, 255], [40, 40, 40], [230, 40, 40], [40, 120, 230], [40, 170, 40], [200, 40, 200]], np.uint8)


def run(plan, model):
    d = cubicasa.load(plan)
    f = PX_PER_M / SCALE
    img = cv2.cvtColor(cv2.imread(str(d["image"])), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    gt = cubicasa.rasterise(d, img.shape[:2], f)
    img = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gt = cv2.copyMakeBorder(gt, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
    to_px = lambda g: affinity.translate(affinity.scale(g, f, f, origin=(0, 0)), PAD, PAD)

    sheet = Sheet(plan, plan, "F1_scaled.png", "raster", img, (0.0, 0.0), {"value": "1 cm/px", "method": "CubiCasa"})
    pipeline.run(sheet, model=model, stages=("segment", "walls", "openings", "stairs", "rooms"))

    k = len(CLASSES)
    cm = np.bincount(gt.ravel().astype(np.int64) * k + sheet.label.ravel(), minlength=k * k).reshape(k, k)
    # openings: reference doors and windows vs predicted doors, windows and interior openings
    ref_o = [(kind, to_px(p).centroid) for kind in ("doors", "windows") for p in d[kind]]
    pred_o = [o for o in sheet.openings if o["kind"] != "passage"]
    used, hit = set(), {"doors": 0, "windows": 0}
    for kind, c in ref_o:
        best = min(((np.hypot(*(o["centre"] - [c.x, c.y])), j) for j, o in enumerate(pred_o) if j not in used), default=(1e9, None))
        if best[0] < 0.5 * PX_PER_M:
            used.add(best[1])
            hit[kind] += 1
    # rooms: reference spaces (outdoor excluded) vs predicted rooms of at least 1 m², matched one to one
    ref_r = [to_px(r["poly"]) for r in d["rooms"] if not r["type"].startswith("Outdoor")]
    pred_r = [r["poly"] for r in sheet.rooms if r["poly"].area >= PX_PER_M ** 2]
    mr = metrics.match_rooms(pred_r, ref_r, 0.5, 1 / PX_PER_M ** 2)
    best_ref, best_pred = mr["best_iou"], mr["best_iou_pred"]            # earlier best-IoU matching, for comparison
    typed = metrics.opening_scores([(o["kind"], tuple(o["centre"])) for o in pred_o],
                                   [(k[:-1], (c.x, c.y)) for k, c in ref_o], 0.5 * PX_PER_M)
    return {"plan": plan, "style": plan.split("/")[0], "cm": cm.tolist(),
            "doors": [hit["doors"], len(d["doors"])], "windows": [hit["windows"], len(d["windows"])],
            "openings_pred": len(pred_o), "openings_matched": len(used), "passages": len(sheet.openings) - len(pred_o),
            "rooms_ref": len(ref_r), "rooms_pred": len(pred_r), "rooms_recall50": mr["matched"],
            "rooms_precision50": mr["matched"], "rooms_mean_iou": mr["mean_iou_ref"],
            "area_err": [abs(e) for e in mr["area_err_pct"]],
            "rooms_matching": "one-to-one", "rooms_pair_iou": mr["pair_iou"], "area_err_signed": mr["area_err_pct"],
            "area_err_m2": mr["area_err_m2"],
            "rooms_legacy_best_iou": {"recall50": int(sum(v >= 0.5 for v in best_ref)), "precision50": int(sum(v >= 0.5 for v in best_pred)),
                                      "mean_iou": float(np.mean(best_ref)) if ref_r else None},
            "walls": metrics.mask_scores(sheet.wall_mask, gt == WALL, 2, 0.05 * PX_PER_M), "openings_typed": typed}, sheet, gt


def summary(rows):
    k = len(CLASSES)
    cm = np.sum([np.array(r["cm"]) for r in rows], axis=0)
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    s = lambda key, i: sum(r[key][i] for r in rows)
    errs = [e for r in rows for e in r["area_err"]]
    rnd = lambda v, n=3: None if v is None else round(float(v), n)
    out = {"plans": len(rows), "pixel_iou": {c: round(float(inter[i] / max(union[i], 1)), 3) for i, c in enumerate(CLASSES)},
           "door_recall": round(s("doors", 0) / max(s("doors", 1), 1), 3), "window_recall": round(s("windows", 0) / max(s("windows", 1), 1), 3),
           "opening_precision": round(sum(r["openings_matched"] for r in rows) / max(sum(r["openings_pred"] for r in rows), 1), 3),
           "room_recall50": round(sum(r["rooms_recall50"] for r in rows) / max(sum(r["rooms_ref"] for r in rows), 1), 3),
           "room_precision50": round(sum(r["rooms_precision50"] for r in rows) / max(sum(r["rooms_pred"] for r in rows), 1), 3),
           "room_mean_iou": round(float(np.mean([r["rooms_mean_iou"] for r in rows if r["rooms_mean_iou"] is not None])), 3),
           "median_area_err_pct": round(float(np.median(errs)), 1) if errs else None}
    if rows and "walls" in rows[0]:                       # fields added with the harness (step 1c)
        signed = [e for r in rows for e in r["area_err_signed"]]
        m2 = [abs(e) for r in rows for e in r["area_err_m2"]]
        pair = [e for r in rows for e in r["rooms_pair_iou"]]
        walls = metrics.mask_rates(metrics.add_counts([r["walls"] for r in rows]))
        ops = metrics.opening_rates(metrics.add_counts([r["openings_typed"] for r in rows]))
        legacy = lambda k: sum(r["rooms_legacy_best_iou"][k] for r in rows)
        out.update({"room_matching": "one-to-one (Hungarian on IoU), IoU >= 0.5",
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
                                                                      if r["rooms_legacy_best_iou"]["mean_iou"] is not None]))}})
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--out" in args:                              # write results and overlays elsewhere (e.g. a short test run)
        i = args.index("--out")
        OUT = Path(args[i + 1])
        del args[i:i + 2]
    model_path = DATA / "model/segmenter.pt"
    if "--model" in args:                            # e.g. data/model-v2/segmenter.pt
        i = args.index("--model")
        model_path = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    split = args[0] if args else "test"
    limit = int(args[1]) if len(args) > 1 else None
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(12)
    model = load_model(model_path)
    plans = [l.strip().strip("/") for l in open(cubicasa.ROOT / f"{split}.txt") if l.strip()][:limit]
    rows, shown, t0 = [], {}, time.time()
    for i, plan in enumerate(plans, 1):
        try:
            r, sheet, gt = run(plan, model)
        except Exception as e:                       # keep going; report at the end
            print(f"FAIL {plan}: {type(e).__name__}: {e}", flush=True)
            continue
        rows.append(r)
        if shown.get(r["style"], 0) < 4:              # a few overlays per style: plan | prediction | reference
            shown[r["style"]] = shown.get(r["style"], 0) + 1
            pred = sheet.img.copy()
            for room in sheet.rooms:
                cv2.polylines(pred, [np.asarray(room["poly"].exterior.coords, np.int32)], True, (255, 140, 0), 2)
            m = sheet.label > 0
            pred[m] = (0.3 * pred[m] + 0.7 * PALETTE[sheet.label][m]).astype(np.uint8)
            tile = np.hstack([sheet.img, pred, PALETTE[gt]])
            cv2.imwrite(str(OUT / f"{r['style']}_{plan.split('/')[1]}.png"), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
        if i % 25 == 0:
            print(f"{i}/{len(plans)} plans, {(time.time() - t0) / i:.1f} s/plan", flush=True)
    result = {"split": split, "all": summary(rows),
              "by_style": {s: summary([r for r in rows if r["style"] == s]) for s in sorted({r["style"] for r in rows})},
              "per_plan": rows}
    (OUT / f"results_{split}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    for name, s in [("all", result["all"]), *result["by_style"].items()]:
        print(f"{name:28s} {s['plans']:3d} plans | IoU " + " ".join(f"{c} {v:.2f}" for c, v in s["pixel_iou"].items() if c != "background")
              + f" | doors {s['door_recall']:.2f} windows {s['window_recall']:.2f} prec {s['opening_precision']:.2f}"
              + f" | rooms R {s['room_recall50']:.2f} P {s['room_precision50']:.2f} mIoU {s['room_mean_iou']:.2f} area err {s['median_area_err_pct']}%")
        if "wall_boundary_f" in s:
            lg = s["legacy_best_iou"]
            print(f"{'':28s}     one-to-one rooms: mIoU of matches {s['room_mean_iou_matched']}, signed area err {s['median_area_err_signed_pct']}% "
                  f"(|m²| {s['median_abs_area_err_m2']}) | best-IoU (old) R {lg['room_recall50']} P {lg['room_precision50']} mIoU {lg['room_mean_iou']}"
                  f" | walls BF {s['wall_boundary_f']} CL P {s['wall_centreline_p']} R {s['wall_centreline_r']}"
                  f" | doors P {s['door_precision_typed']} R {s['door_recall_typed']}, windows P {s['window_precision_typed']} "
                  f"R {s['window_recall_typed']}, door as window {s['door_as_window']}, window as door {s['window_as_door']}")
