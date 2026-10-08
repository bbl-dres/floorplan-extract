"""FloorPlanCAD semantic symbol spotting with the pilot segmenter, after Fan et al. 2021 (ICCV), §6.1 and Tables 3 and 6:
the protocol (votes per primitive, F1 per category, wall PQ, pixel IoU, the report) and the run (main). Blocks,
label ids, scale check, rendering and masks: fpeval.datasets.fpcad; the command line: fpcad_eval.py.

FloorPlanCAD is CC BY-NC 4.0 (annotations; drawings stay with their owners): evaluation only, outputs stay in the
gitignored data folder (default data/fpcad/). Never train or select a deployed model on it.

Protocol, step by step (paper = Fan et al. 2021, research/papers-md/2021-fan-floorplancad.md; the PDF has the formulas)
1. Blocks. Each SVG of the split is one 10 m x 10 m block, viewBox 0 0 100 100 at 10 SVG units per metre
   (data/benchmark/floorplancad/SOURCE.md). Primitives are the <path> (one line or one arc each), <circle> and <ellipse>
   elements, labelled or not; <text> is drawn but is not a primitive (the paper's entities are segments, arcs and
   circles/curves, and text carries no label). Labels: attribute semantic-id (release order, SOURCE.md).
2. Scale check, per block (not per drawing: some drawings mix plans at different scales, e.g. 0407 has blocks at
   10 units/m and blocks drawn about 5x smaller, with full-size text and axis bubbles). Cue from the annotation: the
   median swing radius of the labelled single doors (id 3, at least 2 arcs) over REF_LEAF = 8.73 units, the median of
   blocks with 200 mm walls. Factor f below 0.45 or above 2.2 -> off scale, unless the wall thickness mode (gap
   between parallel labelled wall lines, REF_WALL = 2.0) is normal (veto: "conflict"). Off-scale blocks are rendered
   at 5 / f px per unit, i.e. at 50 px/m again (--scale-policy rescale; or skip, or none). Without door cue the block
   is assumed at 10 units/m ("ok" if the wall mode is plausible, "suspect" if it is under 7 cm, "unverified" if there
   is none); the wall cue alone never rescales, since finish and plaster lines 2-4 cm off the wall face fool it
   (checked by eye on 0039-0009, 0874-0001). On the test split (5,502 blocks): 38 blocks of 26 drawings off and
   rescaled (f 0.14-0.42), 4,613 ok, 801 unverified, 48 suspect, 2 conflicts, 8 door cues ignored (stray arcs).
   The paper's baselines needed no such step (they were trained on FloorPlanCAD); our segmenter expects a known
   scale, as in the pipeline. The cue reads the labels, i.e. it is an oracle scale.
3. Rendering. PyMuPDF renders the SVG (the same rasteriser the pipeline uses for vector PDFs) at PPU = 5 px per unit =
   50 px/m, the segmenter's working resolution. The paper does not give its raster size (the release's PNGs are
   1000 x 1000 px = 100 px/m; its label masks used 5 px lines, GAT-CADNet 2 px lines, at unstated resolutions). Black on
   white: every stroke and text fill set to black, the black page background dropped. Line width: all strokes in the
   release are 0.1 units (1 cm, no CAD line weights), so one uniform pen is used: LINE_PX = 1.25 px = 2.5 cm in the
   drawing = 0.25 mm on paper at 1:100, a medium pen of a plotted plan (fine 0.13-0.18 mm, walls 0.35-0.5 mm); the
   pen is kept at the same pixel width on rescaled blocks. Robustness check: --line-px 2.5 (0.5 mm at 1:100, bold).
4. Segmentation. fpx.segment.segment with the pipeline's config: 1024 px tiles, 128 px overlap, flip TTA on (four
   flips). Blocks of 500 x 500 px are one tile (padded white to 512).
5. Pixels -> primitives. Each primitive is sampled every <= 0.5 px along its path (lines, arcs, circles, ellipses;
   start, middle and end always included), and the predicted class map is read at the pixel under each sample (the
   paper gives no density). Votes:
   - "paper" (primary): Eq. 10 of the paper, PD(e_i) = argmax_l |{p_k in e_i : PD(p_k) = l}|, a plain majority over
     all labels including background (ties go to the lower class index, i.e. background first);
   - "paper3x3": Eq. 10 on a label map whose foreground is grown by one pixel (3 x 3; a background pixel takes the
     most probable neighbouring foreground class). Our walls are filled regions and the wall lines lie on their
     boundary, so a one-pixel offset can turn a wall line into background; this variant measures that effect;
   - "fg" (variant asked for in the task): majority over the non-background samples, background only if all samples
     are background. Long background lines crossing a predicted region become that class under this rule.
6. Classes. Our segmenter has background, wall, door, window, column, stairs. The paper's Table 3 has Door, Window,
   Stair, Appliance, Furniture, Equipment, Wall, Parking lot; we score the four we predict. The grouping follows the
   x axis of the paper's Fig. 7. Groupings (GROUPINGS):
   - "paper" (primary): Door = ids 3-8 (V1 had single, double, sliding door; the release adds folding, revolving,
     rolling door to the same super-class), Window = 9-12 (window, bay window, blind window, opening symbol),
     Stair = 30 (stairs only; escalator and elevator are Equipment), Wall = 1. The paper does NOT merge curtain wall
     into wall: its dataset (V1, 30 classes, never released) had no curtain wall class, and every 35-class paper keeps
     them apart. Curtain-wall primitives (2) are therefore ignored (left out of every count) in the primary score.
   - "curtain-as-other": curtain wall counted as a negative, like any other class.
   - "wall+curtain": Wall = 1 + 2 (printed as the extra column Wall+CW).
   - "curtain-as-window": Window = 2 + 9-12 (our model's semantics: a glazed facade is a window).
   Predicted column and background are "other". Everything else, labelled or not, is a negative (a wall vote on an
   unlabelled column outline is a wall FP, as for the published models). Column (no FloorPlanCAD label) is reported
   only as a note, against unlabelled primitives on CAD column layers (COLUMN, COLU, S-COLS, 砼柱 ...; not 柱网 axis
   grids), unverified.
7. Metrics, per category c on primitives: TP = gt c and pred c, FP = pred c and gt not c, FN = gt c and pred not c;
   P, R, F1 = 2TP / (2TP + FP + FN). Weighted (wF1): each primitive counts log(1 + L(e)) with L its length in SVG
   units (the paper: "entity length log(1 + L(e_i)) to weight the TP, FP and FN"; it does not state the unit; a code
   check of SymPoint's released evaluator found SVG units, not re-verified here). The paper's category columns in
   Table 3 are length-weighted F1:
   entity-count-weighted means of Table 6's per-class weighted F1 reproduce them (Door 0.847 vs 0.848 for the GCN,
   0.837 vs 0.837 for DeepLabv3+; Stair and Wall identical), so the table prints our per-category wF1, and F1 on
   counts in the detail. Door and Window are scored on merged labels (a single door voted "door" is a hit): our model
   predicts the categories directly. The paper's F1 and weighted F1 columns cover all 30 classes; they are left empty
   for our rows. Instead the table has the derived mean of the four category columns (Mean4) for every row, and for
   our rows the micro F1 / wF1 over the four categories (TP, FP, FN pooled, as CADTransformer's eval.py pools over
   all foreground classes), labelled F1-4c / wF1-4c: these are NOT the paper's F1 over all categories.
   Also reported: recall per category split by primitive kind (door arcs vs lines: our door class is trained on the
   Swiss Dwellings door polygon, the opening in the wall, while FloorPlanCAD doors include leaf and swing arc), and
   the wall PQ as a stuff class (one symbol per block, log-length IoU > 0.5, pooled over blocks; Eqs. 1-4).
8. Pixel IoU against filled label masks rasterised like the earlier sample (manifest.json label_variants.label), with
   the "paper" grouping: wall = closing of the labelled wall lines with a 0.45 m disk; window = closing with a
   0.30 m disk; door and stairs = convex hull per instance (doors include the swing); column = closed outlines on
   column layers, filled, 0.02-2.5 m2 (layer-derived, unverified); curtain wall closed like windows and ignored;
   painted in the order wall, column, curtain wall (ignored), window, stairs, door.

Outputs in --out: rows_<tag>.jsonl (one row per block: scale decision, sparse confusion of release id x predicted
class with counts and log-length weights for each vote, the same per primitive kind for the labelled ids of our
categories, pixel confusion, timings), summary_<tag>.json, and overlays/<block>_<tag>.png (render | prediction |
primitives by ground truth | primitives by prediction). --resume continues an interrupted run from its rows file;
with all rows present it only rewrites the summary. Runtime on the shared 16-thread CPU with 4 workers x 2 threads:
about 3 s (v1) to 3.5 s (v2) of worker time per 500 px block, 45-50 s per rescaled block, i.e. 0.8-0.9 s per block
wall clock: 75-95 min per model and pen width for the whole test split.
"""
import argparse
import json
import multiprocessing
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from common import DATA, MODEL
from fpeval.cli import record_failure, utf8_stdout
from fpeval.datasets.fpcad import (CATS, COLUMN_LAYER, FPCAD, GROUPINGS, ID_NAMES, IGNORE_PX, K, KIND_GT, KINDS, LINE_PX,
                                   N_GT, PPU, PRED_CAT, PX_PER_M, SAMPLE_PX, _poly_px, gt_mask, parse_svg, render,
                                   sample_pixels, scale_decision)
from fpeval.raster import PALETTE, tint
from fpx.model import BG, CLASSES, COLUMN

OUT = DATA / "fpcad"
VOTES = ("paper", "paper3x3", "fg")
# Fan et al. 2021, Table 3 (semantic symbol spotting, FloorPlanCAD test set, V1 annotations, 30 classes); the
# per-category values are length-weighted F1, F1 and weighted F1 are over all categories
PAPER_TABLE3 = [("HRNetsV2 W18", 0.821, 0.620, 0.845, 0.620, 0.656, 0.683),
                ("HRNetsV2 W48", 0.811, 0.640, 0.847, 0.624, 0.666, 0.693),
                ("DeepLabv3+ R50", 0.828, 0.659, 0.856, 0.630, 0.680, 0.705),
                ("DeepLabv3+ R101", 0.837, 0.666, 0.852, 0.634, 0.688, 0.714),
                ("GCN (paper's own)", 0.848, 0.709, 0.857, 0.814, 0.806, 0.798)]
CAT_COLOUR = {"Door": (230, 40, 40), "Window": (40, 120, 230), "Stair": (200, 40, 200), "Wall": (40, 40, 40)}


def tolerant_label(label, prob=None):
    """Label map with the foreground grown by one pixel: a background pixel takes the foreground class of its 3 x 3
    neighbourhood with the highest probability (among neighbours whose predicted class it is). Wall lines lie on the
    boundary of our filled wall regions; this keeps a one-pixel offset from turning them into background."""
    best = np.zeros(label.shape, np.float32)
    grown = np.zeros_like(label)
    k = np.ones((3, 3), np.uint8)
    for c in range(1, K):
        m = (label == c).astype(np.float32) * (prob[c].astype(np.float32) if prob is not None else 1.0)
        d = cv2.dilate(m, k)
        upd = d > best
        grown[upd], best[upd] = c, d[upd]
    return np.where(label > 0, label, grown).astype(label.dtype)


def vote(prims, label, ppu, label3=None):
    """Predicted class per primitive under each vote of VOTES, plus the sample counts (n, K) on `label`:
    paper = Eq. 10 on the label map; paper3x3 = Eq. 10 on tolerant_label (label3); fg = majority of the
    non-background samples, background only if all samples are background."""
    n = len(prims)
    label3 = tolerant_label(label) if label3 is None else label3
    counts = np.zeros((n, K), np.int64)
    counts3 = np.zeros((n, K), np.int64)
    if n:
        i, r, c = sample_pixels(prims, ppu, label.shape)
        np.add.at(counts, (i, label[r, c].astype(np.int64)), 1)
        np.add.at(counts3, (i, label3[r, c].astype(np.int64)), 1)
    fgc = counts.copy()
    fgc[:, BG] = 0
    return {"paper": counts.argmax(1),                        # argmax: ties go to the lower index, background first
            "paper3x3": counts3.argmax(1),
            "fg": np.where(fgc.sum(1) > 0, fgc.argmax(1), BG)}, counts


_MODEL = None


def _init_worker(model_path, threads):
    global _MODEL
    import torch
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    from fpx.segment import load_model
    _MODEL = load_model(model_path)


def evaluate_block(path, model=None, line_px=LINE_PX, scale_policy="rescale", overlay=None):
    """Row of one block. overlay: PNG path to write the four-panel overlay to."""
    from fpx import Sheet
    from fpx.segment import segment
    model = model if model is not None else _MODEL
    path = Path(path)
    t = {"start": time.perf_counter()}
    data = path.read_bytes()
    prims = parse_svg(data)
    ppu, sc = scale_decision(prims, scale_policy)
    row = {"id": path.stem, "drawing": path.stem.split("-")[0], "scale": sc, "n_prims": len(prims),
           "n_labelled": sum(p["sid"] > 0 for p in prims)}
    t["parse"] = time.perf_counter()
    if ppu is None:
        row.update({"skipped": True, "t": {"total": round(t["parse"] - t["start"], 3)}})
        return row
    img = render(data, ppu, line_px)
    t["render"] = time.perf_counter()
    sheet = Sheet(path.stem, path.stem, path.name, "vector", img, (0.0, 0.0),
                  {"value": f"{PX_PER_M} px/m", "method": f"FloorPlanCAD block, scale cue {sc['cue']}, factor {sc['factor']}"})
    label = segment(sheet, model)
    t["segment"] = time.perf_counter()
    votes, counts = vote(prims, label, ppu, tolerant_label(label, sheet.prob))
    gt = np.array([p["gt"] for p in prims], np.int64)
    kind = np.array([KINDS.index(p["kind"]) for p in prims], np.int64)
    w = np.log1p(np.array([p["length"] for p in prims]))
    sel = np.isin(gt, KIND_GT)
    sparse, sparse_kind = {}, {}
    for name, pred in votes.items():
        cell = gt * K + pred
        cnt = np.bincount(cell, minlength=N_GT * K)
        wsum = np.bincount(cell, weights=w, minlength=N_GT * K)
        sparse[name] = [[int(c // K), int(c % K), int(cnt[c]), round(float(wsum[c]), 4)] for c in np.nonzero(cnt)[0]]
        cell = (gt[sel] * len(KINDS) + kind[sel]) * K + pred[sel]          # labelled primitives by kind
        cnt = np.bincount(cell, minlength=N_GT * len(KINDS) * K)
        wsum = np.bincount(cell, weights=w[sel], minlength=N_GT * len(KINDS) * K)
        sparse_kind[name] = [[int(c // (len(KINDS) * K)), int(c // K % len(KINDS)), int(c % K), int(cnt[c]),
                              round(float(wsum[c]), 4)] for c in np.nonzero(cnt)[0]]
    mask = gt_mask(prims, label.shape, ppu)
    keep = mask != IGNORE_PX
    pix = np.bincount(mask[keep].astype(np.int64) * K + label[keep], minlength=K * K)
    t["map"] = time.perf_counter()
    row.update({"skipped": False, "ppu": round(ppu, 4), "size_px": list(label.shape), "cm": sparse,
                "cm_kind": sparse_kind, "pix_cm": pix.reshape(K, K).tolist(), "pix_ignored": int((~keep).sum()),
                "n_samples": int(counts.sum()),
                "t": {"parse": round(t["parse"] - t["start"], 3), "render": round(t["render"] - t["parse"], 3),
                      "segment": round(t["segment"] - t["render"], 3), "map": round(t["map"] - t["segment"], 3),
                      "total": round(t["map"] - t["start"], 3)}})
    if overlay:
        write_overlay(overlay, img, label, prims, votes["paper"], mask, ppu, row)
    return row


def _cat_of_gt(gt, grouping="paper"):
    for c in CATS:
        if gt in GROUPINGS[grouping][c]:
            return c
    return None


def write_overlay(path, img, label, prims, pred, mask, ppu, row):
    """2 x 2 panels: render | predicted pixels | primitives by ground truth | primitives by prediction (paper vote).
    Wall dark grey, door red, window blue, stairs magenta, column green; other primitives light grey, curtain wall
    (ignored) cyan; primitives whose predicted category differs from the ground truth are drawn 3 px wide in the
    prediction panel. The ground-truth panel also shows the outline of the filled pixel mask, faint."""
    H, W = label.shape
    tinted = tint(img, label)
    gt_p = np.full((H, W, 3), 255, np.uint8)
    pr_p = np.full((H, W, 3), 255, np.uint8)
    m6 = np.where(mask == IGNORE_PX, BG, mask)
    edge = (m6 > 0) & (cv2.erode((m6 > 0).astype(np.uint8), np.ones((3, 3))) == 0)
    gt_p[edge] = (0.5 * PALETTE[m6][edge] + 0.5 * 255).astype(np.uint8)          # faint outline of the pixel mask
    ignored = set(GROUPINGS["paper"]["ignore"])
    order = sorted(range(len(prims)), key=lambda k: _cat_of_gt(prims[k]["gt"]) is not None)
    for k in order:
        p = prims[k]
        q = _poly_px(p["pts"], ppu)
        gc = _cat_of_gt(p["gt"])
        pc = PRED_CAT.get(int(pred[k]))
        ign = p["gt"] in ignored                                             # curtain wall: cyan, never an error
        cv2.polylines(gt_p, [q], False, (0, 170, 170) if ign else CAT_COLOUR[gc] if gc else (190, 190, 190), 1,
                      cv2.LINE_AA, shift=4)
        colour = CAT_COLOUR[pc] if pc else ((40, 170, 40) if pred[k] == COLUMN else (190, 190, 190))
        cv2.polylines(pr_p, [q], False, colour, 3 if pc != gc and not ign else 1, cv2.LINE_AA, shift=4)
    s = min(1.0, 700 / max(H, W))
    panels = [cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else x
              for x in (img, tinted, gt_p, pr_p)]
    names = ["render", "segmenter", "primitives: ground truth", "primitives: prediction (Eq. 10)"]
    for x, name in zip(panels, names):
        cv2.putText(x, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 0), 1, cv2.LINE_AA)
        cv2.rectangle(x, (0, 0), (x.shape[1] - 1, x.shape[0] - 1), (160, 160, 160), 1)
    grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    cv2.putText(grid, f"{row['id']}  scale {row['scale']['cue']} f={row['scale']['factor']}  ppu {ppu:.2f}",
                (6, grid.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


# ----- metrics -----

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else None
    r = tp / (tp + fn) if tp + fn > 0 else None
    f = tp / (tp + 0.5 * fp + 0.5 * fn) if tp + fp + fn > 0 else None
    return p, r, f


CAT_COL = {cat: col for col, cat in PRED_CAT.items()}


def class_scores(cm, grouping="paper"):
    """Per category and pooled scores from a (N_GT, K) matrix (counts or weights). Rows of ignored ids are dropped.
    pooled = micro over the four categories (TP, FP, FN summed; as CADTransformer's eval.py does over all classes);
    mean4 = mean of the four category F1 (the derived column for the paper's rows)."""
    g = GROUPINGS[grouping]
    valid = np.ones(N_GT, bool)
    valid[g["ignore"]] = False
    out, tot = {}, np.zeros(3)
    for cat in CATS:
        col = CAT_COL[cat]
        rows = np.zeros(N_GT, bool)
        rows[g[cat]] = True
        rows &= valid
        tp = cm[rows, col].sum()
        fp = cm[valid & ~rows, col].sum()
        fn = cm[rows].sum() - tp
        p, r, f = prf(tp, fp, fn)
        out[cat] = {"P": p, "R": r, "F1": f, "TP": float(tp), "FP": float(fp), "FN": float(fn)}
        tot += [tp, fp, fn]
    p, r, f = prf(*tot)
    fs = [out[c]["F1"] for c in CATS]
    out["pooled"] = {"P": p, "R": r, "F1": f, "TP": float(tot[0]), "FP": float(tot[1]), "FN": float(tot[2])}
    out["mean4"] = float(np.mean(fs)) if all(v is not None for v in fs) else None
    return out


def kind_recall(rows, vote_name, grouping="paper"):
    """Recall per category split by primitive kind (line, arc, ...): {cat: {kind: {n, recall, as_background}}}."""
    g = GROUPINGS[grouping]
    cnt = np.zeros((N_GT, len(KINDS), K))
    for r in rows:
        for gt, kd, p, c, _ in r.get("cm_kind", {}).get(vote_name, []):
            cnt[gt, kd, p] += c
    out = {}
    for cat in CATS:
        ids = [i for i in g[cat] if i not in g["ignore"]]
        m = cnt[ids].sum(0)                                         # (kinds, K)
        out[cat] = {KINDS[k]: {"n": int(m[k].sum()), "recall": float(m[k, CAT_COL[cat]] / m[k].sum()),
                               "as_background": float(m[k, BG] / m[k].sum())}
                    for k in range(len(KINDS)) if m[k].sum() > 0}
    return out


def stuff_pq(rows, vote_name, cat="Wall", grouping="paper"):
    """Panoptic quality of a stuff class (Eqs. 1-4 of the paper): one symbol per block, the primitives voted `cat`
    against the ground-truth primitives of `cat`, matched at log-length IoU > 0.5; TP, FP, FN and IoU pooled over
    blocks (as SymPoint's evaluator pools over classes)."""
    g = GROUPINGS[grouping]
    ids, col = [i for i in g[cat] if i not in g["ignore"]], CAT_COL[cat]
    tp = fp = fn = iou_sum = 0.0
    for r in rows:
        W = np.zeros((N_GT, K))
        for gt, p, _, w in r["cm"][vote_name]:
            W[gt, p] += w
        W[g["ignore"]] = 0
        inter, gsum, psum = W[ids, col].sum(), W[ids].sum(), W[:, col].sum()
        union = gsum + psum - inter
        if gsum > 0 and psum > 0 and inter / union > 0.5:
            tp, iou_sum = tp + 1, iou_sum + inter / union
        else:
            fn += gsum > 0
            fp += psum > 0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if tp + fp + fn else None
    sq = iou_sum / tp if tp else None
    return {"PQ": rq * sq if rq is not None and sq is not None else (0.0 if rq is not None else None),
            "SQ": sq, "RQ": rq, "TP": tp, "FP": fp, "FN": fn}


def matrices(rows):
    """Summed (N_GT, K) count and weight matrices per vote."""
    C = {v: np.zeros((N_GT, K)) for v in VOTES}
    Wt = {v: np.zeros((N_GT, K)) for v in VOTES}
    for r in rows:
        for v in VOTES:
            for g, p, c, w in r["cm"][v]:
                C[v][g, p] += c
                Wt[v][g, p] += w
    return C, Wt


def rnd(x, n=4):
    if isinstance(x, dict):
        return {k: rnd(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), n)
    return x


def metrics(rows):
    C, Wt = matrices(rows)
    res = {}
    for v in VOTES:
        res[v] = {g: {"count": class_scores(C[v], g), "weighted": class_scores(Wt[v], g)} for g in GROUPINGS}
        col = C[v][:, COLUMN]
        colw = Wt[v][:, COLUMN]
        res[v]["column_note"] = {
            "predicted_column": float(col.sum()),
            "on_column_layers": float(col[COLUMN_LAYER]),
            "column_layer_primitives": float(C[v][COLUMN_LAYER].sum()),
            "precision_vs_layers": col[COLUMN_LAYER] / col.sum() if col.sum() else None,
            "recall_vs_layers": col[COLUMN_LAYER] / C[v][COLUMN_LAYER].sum() if C[v][COLUMN_LAYER].sum() else None,
            "weighted_precision_vs_layers": colw[COLUMN_LAYER] / colw.sum() if colw.sum() else None,
            "weighted_recall_vs_layers": colw[COLUMN_LAYER] / Wt[v][COLUMN_LAYER].sum() if Wt[v][COLUMN_LAYER].sum() else None,
            "predicted_column_by_gt": {ID_NAMES[g]: float(col[g]) for g in np.argsort(-col)[:6] if col[g] > 0}}
        res[v]["confusion_by_id"] = {ID_NAMES[g]: {CLASSES[k]: int(C[v][g, k]) for k in range(K) if C[v][g, k]}
                                     for g in range(N_GT) if C[v][g].sum()}
        res[v]["recall_by_kind"] = kind_recall(rows, v)
        res[v]["wall_pq"] = stuff_pq(rows, v)
    return res


def pixel_iou(rows):
    cm = np.sum([np.array(r["pix_cm"]) for r in rows], axis=0) if rows else np.zeros((K, K))
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    return {c: (float(inter[i] / union[i]) if union[i] else None) for i, c in enumerate(CLASSES)}, cm


def summarise(rows, args_dict, wall_clock=None):
    failed = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    done = [r for r in rows if not r.get("skipped")]
    native = [r for r in done if r["scale"]["status"] != "off"]
    actions = {}
    for r in rows:
        actions[r["scale"]["action"]] = actions.get(r["scale"]["action"], 0) + 1
    cues = {}
    for r in rows:
        key = f"{r['scale']['cue']}:{r['scale']['status']}"
        cues[key] = cues.get(key, 0) + 1
    tt = np.array([r["t"]["total"] for r in done]) if done else np.zeros(1)
    seg = np.array([r["t"]["segment"] for r in done]) if done else np.zeros(1)
    iou, pcm = pixel_iou(done)
    iou_native, _ = pixel_iou(native)
    out = {"settings": args_dict,
           "blocks": len(rows), "evaluated": len(done), "skipped": len(rows) - len(done), "failed": failed,
           "scale": {"actions": actions, "cues": cues,
                     "rescaled": [{"id": r["id"], "factor": r["scale"]["factor"], "cue": r["scale"]["cue"]}
                                  for r in rows if r["scale"]["action"] == "rescaled"],
                     "skipped": [r["id"] for r in rows if r.get("skipped")]},
           "primitives": int(sum(r["n_prims"] for r in done)), "labelled": int(sum(r["n_labelled"] for r in done)),
           "metrics": metrics(done),
           "metrics_native_scale_only": {v: m["paper"] for v, m in metrics(native).items() if v in VOTES},
           "pixel_iou": iou, "pixel_iou_native_scale_only": iou_native, "pixel_cm": pcm.tolist(),
           "runtime": {"per_block_mean_s": float(tt.mean()), "per_block_median_s": float(np.median(tt)),
                       "segment_mean_s": float(seg.mean()), "rescaled_mean_s":
                           float(np.mean([r["t"]["total"] for r in done if r["scale"]["action"] == "rescaled"]))
                           if any(r["scale"]["action"] == "rescaled" for r in done) else None,
                       "wall_clock_s": wall_clock,
                       "throughput_blocks_per_min": (len(rows) / wall_clock * 60) if wall_clock else None},
           "paper_table3": [dict(zip(["method"] + CATS + ["F1", "wF1"], r), mean4=float(np.mean(r[1:5])))
                            for r in PAPER_TABLE3]}
    return rnd(out)


def table(summary, label):
    """The paper's Table 3 layout with its published rows: Door, Window, Stair, Wall (length-weighted F1 per category),
    the derived mean of these four, F1 and weighted F1 (all 30 classes: published rows only), and for our rows the
    four-category micro F1 / wF1 and the Wall column with curtain wall merged (extra)."""
    f = lambda v: "" if v is None else f"{v:.3f}"
    h = (f"{'Method':30s} {'Door':>6s} {'Window':>6s} {'Stair':>6s} {'Wall':>6s} {'Mean4':>6s} | {'F1':>6s} {'wF1':>6s}"
         f" | {'F1-4c':>6s} {'wF1-4c':>6s} {'Wall+CW':>7s}")
    lines = ["Published rows: FloorPlanCAD paper, Table 3 (p.8), internal V1 dataset (30 classes, never released), "
             "in-domain training.",
             f"Our rows: November 2021 release, {summary['settings'].get('split', 'test')} split "
             f"({summary['evaluated']} of {summary['settings'].get('blocks_in_split', '?')} blocks), zero-shot. "
             "No published number exists on this split.",
             "Category columns: length-weighted F1 (log(1 + L), L in SVG units). F1, wF1: over all 30 classes, "
             "published rows only. F1-4c, wF1-4c: micro over our four categories, NOT comparable with F1, wF1.",
             h, "-" * len(h)]
    for r in PAPER_TABLE3:
        lines.append(f"{r[0]:30s} " + " ".join(f(v).rjust(6) for v in r[1:5]) + f" {f(float(np.mean(r[1:5]))).rjust(6)}"
                     f" | {f(r[5]).rjust(6)} {f(r[6]).rjust(6)} | {'':>6s} {'':>6s} {'':>7s}")
    for v in VOTES:
        m = summary["metrics"][v]["paper"]
        wc = summary["metrics"][v]["wall+curtain"]["weighted"]["Wall"]["F1"]
        lines.append(f"{label + ', ' + v:30s} " + " ".join(f(m["weighted"][c]["F1"]).rjust(6) for c in CATS)
                     + f" {f(m['weighted']['mean4']).rjust(6)} | {'':>6s} {'':>6s} | "
                     f"{f(m['count']['pooled']['F1']).rjust(6)} {f(m['weighted']['pooled']['F1']).rjust(6)} {f(wc).rjust(7)}")
    return "\n".join(lines)


def report(summary, label):
    f = lambda v: "  -  " if v is None else f"{v:.3f}"
    s = summary
    out = [f"\n{label}: {s['evaluated']} of {s['blocks']} blocks evaluated ({s['skipped']} skipped), "
           f"{s['primitives']} primitives ({s['labelled']} labelled)",
           f"scale: {s['scale']['actions']}  cues {s['scale']['cues']}",
           "\nSemantic symbol spotting in the layout of the paper's Table 3 (votes: paper = Eq. 10; paper3x3 = Eq. 10 "
           "with the foreground grown by one pixel; fg = foreground majority).", table(s, label),
           "\nDetail per vote and grouping (P, R, F1 on primitive counts; wF1 length-weighted; groupings: paper = "
           "curtain wall ignored, curtain-as-other = negative, wall+curtain, curtain-as-window, columns-ignored = "
           "diagnostic, unlabelled column-layer primitives also left out):"]
    for v in VOTES:
        for g in GROUPINGS:
            m = s["metrics"][v][g]
            out.append(f"  {v:8s} {g:17s} " + "  ".join(
                f"{c} P {f(m['count'][c]['P'])} R {f(m['count'][c]['R'])} F1 {f(m['count'][c]['F1'])} wF1 {f(m['weighted'][c]['F1'])}"
                for c in CATS) + f"  | 4c F1 {f(m['count']['pooled']['F1'])} wF1 {f(m['weighted']['pooled']['F1'])}"
                f" mean4 wF1 {f(m['weighted']['mean4'])}")
    for v in VOTES:
        m = s["metrics_native_scale_only"][v]
        out.append(f"  native-scale blocks only, {v:8s}: " + " ".join(f"{c} {f(m['weighted'][c]['F1'])}" for c in CATS)
                   + f" | mean4 {f(m['weighted']['mean4'])}")
    out.append("recall by primitive kind (paper grouping; n, recall, share voted background):")
    for v in VOTES:
        rk = s["metrics"][v]["recall_by_kind"]
        out.append(f"  {v:8s} " + "  ".join(f"{c}: " + ", ".join(f"{k} {d['n']} R {f(d['recall'])} bg {f(d['as_background'])}"
                                                                 for k, d in rk[c].items()) for c in CATS))
    out.append("wall PQ (stuff, one symbol per block, log-length IoU > 0.5): " + "  ".join(
        f"{v} PQ {f(s['metrics'][v]['wall_pq']['PQ'])} SQ {f(s['metrics'][v]['wall_pq']['SQ'])} RQ {f(s['metrics'][v]['wall_pq']['RQ'])}"
        for v in VOTES))
    cn = s["metrics"]["paper"]["column_note"]
    out.append(f"column (note, vote paper): {cn['predicted_column']:.0f} primitives predicted column, "
               f"{cn['on_column_layers']:.0f} of them on CAD column layers (precision vs layers {f(cn['precision_vs_layers'])}, "
               f"recall {f(cn['recall_vs_layers'])} of {cn['column_layer_primitives']:.0f} column-layer primitives; "
               f"unverified pseudo-labels); predicted column by ground truth: {cn['predicted_column_by_gt']}")
    out.append("pixel IoU (filled masks, paper grouping, curtain wall ignored): "
               + " ".join(f"{c} {f(v)}" for c, v in s["pixel_iou"].items()))
    rt = s["runtime"]
    out.append(f"runtime: {rt['per_block_mean_s']:.2f} s per block in a worker (median {rt['per_block_median_s']:.2f}, "
               f"segmenter {rt['segment_mean_s']:.2f}); wall clock {rt['wall_clock_s']} s, "
               f"{rt['throughput_blocks_per_min']} blocks/min")
    return "\n".join(out)


# ----- main -----

def pick_overlays(files, k):
    """The first k blocks of the (shuffled) selection with labelled walls and doors or windows."""
    out = []
    for p in files:
        if len(out) >= k:
            break
        ids = {pr["sid"] for pr in parse_svg(p)}
        if 1 in ids and ids & set(range(3, 13)):
            out.append(p.stem)
    return set(out)


def main(argv=None):
    import torch
    from fpx.segment import load_model
    utf8_stdout()
    ap = argparse.ArgumentParser(description="FloorPlanCAD semantic symbol spotting with the pilot segmenter (Fan et al. 2021, Table 3)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="seeded random sample of N blocks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=4, help="processes (the machine is shared)")
    ap.add_argument("--threads", type=int, default=2, help="torch threads per process")
    ap.add_argument("--line-px", type=float, default=LINE_PX, help="pen width in working pixels (robustness: 2.5)")
    ap.add_argument("--scale-policy", choices=["rescale", "skip", "none"], default="rescale")
    ap.add_argument("--overlays", type=int, default=5)
    ap.add_argument("--resume", action="store_true", help="keep rows already in the rows file")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)

    files = sorted((FPCAD / a.split).glob("*.svg"))
    if not files:
        sys.exit(f"no SVG in {FPCAD / a.split}")
    n_split = len(files)
    if a.limit:
        files = random.Random(a.seed).sample(files, min(a.limit, len(files)))
    model = load_model(a.model, device="cpu")     # the parent only reads the version; workers load their own (FPX_DEVICE)
    version = getattr(model, "fpx_version", 1)
    tag = a.tag or (f"{a.split}-v{version}-lw{a.line_px:g}" + (f"-n{a.limit}s{a.seed}" if a.limit else "")
                    + ("" if a.scale_policy == "rescale" else f"-{a.scale_policy}"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"rows_{tag}.jsonl"
    rows = []
    if a.resume and rows_path.exists():
        rows = [json.loads(l) for l in rows_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [r for r in rows if "error" not in r]          # failed blocks run again
    done = {r["id"] for r in rows}
    todo = [p for p in files if p.stem not in done]
    shown = pick_overlays(files, a.overlays) if a.overlays else set()
    ov = lambda p: str(out / "overlays" / f"{p.stem}_{tag}.png") if p.stem in shown else None
    print(f"{len(files)} of {n_split} blocks of {a.split}, model {a.model} (v{version}), lines {a.line_px} px, "
          f"{len(todo)} to run, {a.workers} workers x {a.threads} threads -> {out}", flush=True)
    t0 = time.time()
    with open(rows_path, "a" if a.resume else "w", encoding="utf-8") as fh:
        def keep(r):
            rows.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            n = len(rows)
            if n % 25 == 0 or n == len(files):
                el = time.time() - t0
                print(f"{n}/{len(files)} blocks, {el / max(n - len(done), 1):.2f} s/block wall clock", flush=True)

        if a.workers <= 1 or not todo:
            torch.set_num_threads(a.threads)
            for p in todo:
                try:
                    keep(evaluate_block(p, model, a.line_px, a.scale_policy, ov(p)))
                except Exception as e:                  # keep going; the row lists the error
                    keep(record_failure([], p.stem, e))
        else:
            del model
            # forked workers would inherit OpenCV pool locks and CUDA state; Windows always spawns
            ctx = multiprocessing.get_context("forkserver") if platform.system() != "Windows" else None
            with ProcessPoolExecutor(a.workers, mp_context=ctx, initializer=_init_worker,
                                     initargs=(a.model, a.threads)) as ex:
                futs = {ex.submit(evaluate_block, p, None, a.line_px, a.scale_policy, ov(p)): p for p in todo}
                for fu in as_completed(futs):
                    try:
                        keep(fu.result())
                    except Exception as e:                  # keep going; the row lists the error
                        keep(record_failure([], futs[fu].stem, e))
    wall = time.time() - t0
    old = out / f"summary_{tag}.json"
    if not todo and old.exists():                     # re-summarising finished rows: keep the measured wall clock
        wall = json.loads(old.read_text(encoding="utf-8")).get("runtime", {}).get("wall_clock_s") or wall
        done = set()
    settings = {"split": a.split, "limit": a.limit, "seed": a.seed, "model": a.model, "model_version": version,
                "line_px": a.line_px, "ppu": PPU, "px_per_m": PX_PER_M, "scale_policy": a.scale_policy,
                "workers": a.workers, "threads": a.threads, "tta": True, "sample_px": SAMPLE_PX,
                "blocks_in_split": n_split, "votes": list(VOTES), "groupings": GROUPINGS,
                "length_unit_for_weights": "SVG units (0.1 m at 10 units/m)"}
    summary = summarise(rows, settings, wall_clock=round(wall, 1) if len(done) == 0 else None)
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(report(summary, f"v{version} lines {a.line_px:g} px"))
    print(f"\nrows: {rows_path}\nsummary: {out / f'summary_{tag}.json'}")
