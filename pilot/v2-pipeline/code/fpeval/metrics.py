"""Evaluation metrics shared by every benchmark through fpeval.score (review step 1c, §4.6).

Geometry can be in any unit; the harness works in working pixels and passes the scale (m_per_unit) for metres.

    rooms         one-to-one matching at IoU >= 0.5 (Hungarian), recall, precision, mean IoU, area error in % and m²,
                  and why each unmatched reference room was missed (merged, oversized, split, missing)
    walls         pixel IoU, boundary F-score (2 px tolerance), centre-line precision and recall (skeletons)
    openings      centre precision and recall per kind (door, window) and the door/window confusion
    connectivity  room-to-room edges, mapped through the room matching
    regions       IoU and area error of an outline (GF)

Per-sheet functions return counts, so that totals over many sheets are pooled (micro averages) by summarise().
"""
import numpy as np
import shapely
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize

KINDS = ("door", "window")


# ---------- assignment ----------

def assign(score, threshold):
    """One-to-one assignment on a score matrix (rows: predictions, columns: references). It maximises first the number
    of pairs scoring at least threshold, then the total score. Returns all assigned (row, column) pairs; callers keep
    those at or above the threshold."""
    S = np.asarray(score, float)
    if S.ndim != 2 or not S.size:
        return []
    bonus = (S >= threshold) * (min(S.shape) + 1.0)   # one more pair at the threshold beats any gain in score
    rows, cols = linear_sum_assignment(-(S + bonus))
    return [(int(i), int(j)) for i, j in zip(rows, cols)]


def iou(a, b):
    u = a.union(b).area
    return a.intersection(b).area / u if u else 0.0


def iou_matrix(pred, ref):
    """IoU of every prediction with every reference polygon; only pairs whose extents overlap are computed."""
    M = np.zeros((len(pred), len(ref)))
    if not len(pred) or not len(ref):
        return M
    tree = shapely.STRtree(list(ref))
    i, j = tree.query(list(pred), predicate="intersects")
    for a, b in zip(i, j):
        M[a, b] = iou(pred[a], ref[b])
    return M


def ratio(a, b):
    return a / b if b else None


def f1(p, r):
    if p is None or r is None:
        return None
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


# ---------- rooms ----------

def match_rooms(pred, ref, threshold=0.5, m2_per_unit=1.0, ignore=()):
    """One-to-one room matching (Hungarian on IoU). pred, ref: lists of polygons in the same unit.

    ignore: polygons of reference areas that are not scored (outdoor spaces, voids, areas below the reference minimum):
    a prediction left unmatched that covers one of them at IoU >= threshold is not counted against the precision.
    Returns counts, recall, precision, mean IoU of the matches, signed area errors (% and m²) of each match, the
    matched pairs (pred index, ref index), and per reference room its assigned prediction and IoU (also below the
    threshold) and its best IoU without assignment (the old best-IoU matching, for comparison)."""
    M = iou_matrix(pred, ref)
    assigned = assign(M, threshold)
    pairs = [(i, j) for i, j in assigned if M[i, j] >= threshold]
    ref_pred, ref_iou = [None] * len(ref), [0.0] * len(ref)
    for i, j in assigned:
        if M[i, j] > 0:
            ref_pred[j], ref_iou[j] = i, float(M[i, j])
    used = {i for i, _ in pairs}
    ignored = []
    if len(ignore):
        I = iou_matrix(pred, list(ignore))
        ignored = [i for i in range(len(pred)) if i not in used and I.shape[1] and I[i].max() >= threshold]
    n_pred = len(pred) - len(ignored)
    pair_iou = [float(M[i, j]) for i, j in pairs]
    return {"ref": len(ref), "pred": n_pred, "matched": len(pairs), "ignored": len(ignored),
            "recall": ratio(len(pairs), len(ref)), "precision": ratio(len(pairs), n_pred),
            "mean_iou": float(np.mean(pair_iou)) if pairs else None,
            "mean_iou_ref": float(np.mean(ref_iou)) if ref else None,
            "area_err_pct": [(pred[i].area - ref[j].area) / ref[j].area * 100 for i, j in pairs],
            "area_err_m2": [(pred[i].area - ref[j].area) * m2_per_unit for i, j in pairs],
            "pairs": pairs, "pair_iou": pair_iou, "ref_pred": ref_pred, "ref_iou": ref_iou,
            "best_iou": M.max(0).tolist() if M.size else [0.0] * len(ref),
            "best_iou_pred": M.max(1).tolist() if M.size else [0.0] * len(pred), "ignored_pred": ignored}


def miss_reasons(pred, ref, matched_refs, cover=0.5):
    """Why each unmatched reference room was missed, from how much of it the predictions cover:
    merged     one prediction covers at least `cover` of it and of other reference rooms (returned as merged_with);
    oversized  one prediction covers at least `cover` of it but is much larger, without other reference rooms
               (it leaks into walls, outdoor space or unlabelled area);
    split      the predictions together cover at least `cover`, no single one does;
    missing    less than `cover` is covered.
    Returns {ref index: (reason, [indices of the other reference rooms merged with it], index of the prediction that
    covers most of it or None)}."""
    out = {}
    if not len(ref):
        return out
    tree = shapely.STRtree(list(pred)) if len(pred) else None
    ref_tree = shapely.STRtree(list(ref))
    for j, r in enumerate(ref):
        if j in matched_refs or r.area <= 0:
            continue
        cands = tree.query(r, predicate="intersects") if tree is not None else []
        cov = {int(i): pred[i].intersection(r).area / r.area for i in cands}
        best = max(cov, key=cov.get, default=None)
        if best is not None and cov[best] >= cover:
            p = pred[best]
            others = [k for k in ref_tree.query(p, predicate="intersects")
                      if k != j and ref[k].area > 0 and p.intersection(ref[k]).area / ref[k].area >= cover]
            out[j] = ("merged", [int(k) for k in others], best) if others else ("oversized", [], best)
        elif sum(cov.values()) >= cover:
            out[j] = ("split", [], best)
        else:
            out[j] = ("missing", [], best)
    return out


# ---------- masks: walls and classes ----------

def _crop(*masks, margin=4):
    """Bounding box of the union of masks plus a margin, so that distance transforms run on the drawing only."""
    any_ = np.zeros(masks[0].shape, bool)
    for m in masks:
        any_ |= m
    ys, xs = np.nonzero(any_)
    if not len(ys):
        return None
    H, W = any_.shape
    return slice(max(ys.min() - margin, 0), min(ys.max() + margin + 1, H)), slice(max(xs.min() - margin, 0), min(xs.max() + margin + 1, W))


def boundary(mask):
    """Inner boundary pixels of a binary mask (8-connected erosion)."""
    return mask & ~ndimage.binary_erosion(mask, np.ones((3, 3), bool), border_value=0)


def _near(a, b, tol):
    """Pixels of a lying within tol pixels of a pixel of b, counted: (|a|, hits)."""
    n = int(a.sum())
    if not n:
        return 0, 0
    if not b.any():
        return n, 0
    d = ndimage.distance_transform_edt(~b)
    return n, int((d[a] <= tol).sum())


def mask_scores(pred, ref, boundary_tol=2.0, centre_tol=2.5):
    """Counts for pixel IoU, boundary F-score (boundary pixels within boundary_tol of the other boundary) and
    centre-line precision and recall (skeleton pixels within centre_tol of the other skeleton), all in pixels.
    Turn them into rates with mask_rates(); counts of several sheets can be added first."""
    pred, ref = np.asarray(pred, bool), np.asarray(ref, bool)
    out = {"inter": int((pred & ref).sum()), "union": int((pred | ref).sum()), "pred_px": int(pred.sum()), "ref_px": int(ref.sum())}
    box = _crop(pred, ref, margin=int(max(boundary_tol, centre_tol)) + 2)
    if box is None:
        return {**out, **{k: 0 for k in ("bnd_pred", "bnd_pred_hit", "bnd_ref", "bnd_ref_hit", "cl_pred", "cl_pred_hit", "cl_ref", "cl_ref_hit")}}
    p, r = pred[box], ref[box]
    bp, br = boundary(p), boundary(r)
    out["bnd_pred"], out["bnd_pred_hit"] = _near(bp, br, boundary_tol)
    out["bnd_ref"], out["bnd_ref_hit"] = _near(br, bp, boundary_tol)
    sp, sr = skeletonize(p), skeletonize(r)
    out["cl_pred"], out["cl_pred_hit"] = _near(sp, sr, centre_tol)
    out["cl_ref"], out["cl_ref_hit"] = _near(sr, sp, centre_tol)
    return out


def mask_rates(c):
    """Rates from mask_scores() counts (or their sums over sheets)."""
    bp, br = ratio(c["bnd_pred_hit"], c["bnd_pred"]), ratio(c["bnd_ref_hit"], c["bnd_ref"])
    cp, cr = ratio(c["cl_pred_hit"], c["cl_pred"]), ratio(c["cl_ref_hit"], c["cl_ref"])
    return {"iou": ratio(c["inter"], c["union"]), "boundary_p": bp, "boundary_r": br, "boundary_f": f1(bp, br),
            "centreline_p": cp, "centreline_r": cr, "centreline_f": f1(cp, cr)}


def confusion(ref_label, pred_label, k):
    """k x k pixel confusion matrix (rows: reference class, columns: predicted class)."""
    return np.bincount(ref_label.ravel().astype(np.int64) * k + pred_label.ravel(), minlength=k * k).reshape(k, k)


def class_iou(cm, names):
    cm = np.asarray(cm)
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    return {n: (float(inter[i] / union[i]) if union[i] and cm[i].sum() else None) for i, n in enumerate(names)}


# ---------- openings ----------

def match_points(pred, ref, tol):
    """One-to-one matching of points within tol (Hungarian on distance): the most pairs, then the closest."""
    if not len(pred) or not len(ref):
        return []
    P, R = np.asarray(pred, float).reshape(-1, 2), np.asarray(ref, float).reshape(-1, 2)
    D = np.hypot(P[:, None, 0] - R[None, :, 0], P[:, None, 1] - R[None, :, 1])
    return [(i, j) for i, j in assign(tol - D, 0.0) if D[i, j] <= tol]


def opening_scores(pred, ref, tol):
    """pred, ref: [(kind, (x, y))], kind 'door' or 'window' (a reference of unknown kind, e.g. 'opening', takes part
    in the any-kind matching only). Per kind, one-to-one matching of centres within tol gives precision and recall;
    matching regardless of kind gives found_any_kind (with ref_any, pred_any) and the door/window confusion:
    confusion[ref kind][pred kind], with 'none' for unmatched references (missed) and unmatched predictions (extra)."""
    out = {}
    for k in KINDS:
        P = [xy for kk, xy in pred if kk == k]
        R = [xy for kk, xy in ref if kk == k]
        out[k] = {"ref": len(R), "pred": len(P), "matched": len(match_points(P, R, tol))}
    P = [(k, xy) for k, xy in pred if k in KINDS]
    R = list(ref)
    pairs = match_points([xy for _, xy in P], [xy for _, xy in R], tol)
    conf = {a: {b: 0 for b in (*KINDS, "none")} for a in (*KINDS, "none")}
    for i, j in pairs:
        if R[j][0] in KINDS:
            conf[R[j][0]][P[i][0]] += 1
    for j in set(range(len(R))) - {j for _, j in pairs}:
        if R[j][0] in KINDS:
            conf[R[j][0]]["none"] += 1
    for i in set(range(len(P))) - {i for i, _ in pairs}:
        conf["none"][P[i][0]] += 1
    out["confusion"] = conf
    out["found_any_kind"] = len(pairs)
    out["ref_any"], out["pred_any"] = len(R), len(P)
    out["pairs_any"] = [(int(i), int(j)) for i, j in pairs]          # (index into pred of kind door/window, index into ref)
    return out


def opening_rates(c):
    out = {}
    for k in KINDS:
        p, r = ratio(c[k]["matched"], c[k]["pred"]), ratio(c[k]["matched"], c[k]["ref"])
        out[k] = {"precision": p, "recall": r, "f1": f1(p, r)}
    conf = c["confusion"]
    out["door_as_window"] = ratio(conf["door"]["window"], sum(conf["door"].values()))
    out["window_as_door"] = ratio(conf["window"]["door"], sum(conf["window"].values()))
    return out


# ---------- connectivity ----------

def edge_scores(pred_edges, ref_edges, pairs, ignore_edges=()):
    """pred_edges: pairs of prediction indices; ref_edges: pairs of reference indices; pairs: the room matching.
    A predicted edge is correct when both rooms are matched and their references share an edge. Counts for all
    edges, and for the edges whose rooms are both matched (connection logic without the room errors).
    ignore_edges: reference connections that are neither required nor wrong (e.g. open passages when the reference
    lists doors); predicted edges onto them are left out of the precision and counted as other_found."""
    to_ref = dict(pairs)
    pe = {frozenset(e) for e in pred_edges if len(set(e)) == 2}
    re_ = {frozenset(e) for e in ref_edges if len(set(e)) == 2}
    other = {frozenset(e) for e in ignore_edges if len(set(e)) == 2} - re_
    mapped = {frozenset(to_ref[v] for v in e) for e in pe if all(v in to_ref for v in e)}
    unmapped = sum(1 for e in pe if not all(v in to_ref for v in e))
    found_other = mapped & other
    mapped -= found_other
    matched_refs = set(to_ref.values())
    return {"ref": len(re_), "pred": len(mapped) + unmapped, "correct": len(mapped & re_),
            "ref_matched": sum(1 for e in re_ if e <= matched_refs), "pred_mapped": len(mapped),
            "other_ref": len(other), "other_found": len(found_other)}


def edge_rates(c):
    p, r = ratio(c["correct"], c["pred"]), ratio(c["correct"], c["ref"])
    pm, rm = ratio(c["correct"], c["pred_mapped"]), ratio(c["correct"], c["ref_matched"])
    return {"precision": p, "recall": r, "f1": f1(p, r), "precision_matched": pm, "recall_matched": rm, "f1_matched": f1(pm, rm),
            "other_found": ratio(c.get("other_found", 0), c.get("other_ref", 0))}


# ---------- regions ----------

def region_scores(pred, ref, m2_per_unit=1.0):
    """IoU and area error of a predicted outline (e.g. the GF) against its reference."""
    if ref is None or ref.is_empty:
        return None
    a, b = pred.area if pred is not None else 0.0, ref.area
    return {"iou": iou(pred, ref) if pred is not None and not pred.is_empty else 0.0,
            "area_err_pct": (a - b) / b * 100, "area_err_m2": (a - b) * m2_per_unit,
            "pred_m2": a * m2_per_unit, "ref_m2": b * m2_per_unit}


# ---------- pooling over sheets ----------

def med(values):
    v = [x for x in values if x is not None]
    return float(np.median(v)) if v else None


def add_counts(dicts):
    """Sum nested count dicts; a key missing in some of them counts as 0 there."""
    dicts = [d for d in dicts if d]
    if not dicts:
        return None
    out = {}
    for k in dict.fromkeys(k for d in dicts for k in d):
        vs = [d[k] for d in dicts if k in d]
        if isinstance(vs[0], dict):
            out[k] = add_counts(vs)
        elif isinstance(vs[0], (int, float, np.integer, np.floating)) and not isinstance(vs[0], bool):
            out[k] = sum(vs)
    return out


def summarise(rows):
    """Pool per-sheet rows (harness.score) into one summary: counts are added before rates are taken (micro averages);
    area errors and GF scores are medians over all matches or sheets."""
    ok = [r for r in rows if "error" not in r]
    s = {"sheets": len(ok), "failed": len(rows) - len(ok)}
    rooms = [r["rooms"] for r in ok if r.get("rooms")]
    if rooms:
        ref, pred, matched = (sum(x[k] for x in rooms) for k in ("ref", "pred", "matched"))
        pct = [e for x in rooms for e in x["area_err_pct"]]
        m2 = [e for x in rooms for e in x["area_err_m2"]]
        ious = [e for x in rooms for e in x["pair_iou"]]
        rec, prec = ratio(matched, ref), ratio(matched, pred)
        s["rooms"] = {"ref": ref, "pred": pred, "matched": matched, "ignored": sum(x["ignored"] for x in rooms),
                      "recall": rec, "precision": prec, "f1": f1(rec, prec), "mean_iou": float(np.mean(ious)) if ious else None,
                      "recall_sheet_median": med([x["recall"] for x in rooms]),
                      "area_err_pct_median": med(pct), "area_err_pct_abs_median": med(np.abs(pct)),
                      "area_err_m2_median": med(m2), "area_err_m2_abs_median": med(np.abs(m2))}
    walls = add_counts([r.get("walls") for r in ok])
    if walls:
        s["walls"] = mask_rates(walls)
        s["walls"]["iou_sheet_median"] = med([mask_rates(r["walls"])["iou"] for r in ok if r.get("walls")])
    cms = [np.asarray(r["classes"]) for r in ok if r.get("classes") is not None]
    if cms:
        names = ok[0]["class_names"]
        ious = class_iou(sum(cms), names)
        for r in ok:                                  # classes without a reference in this source (e.g. CVC-FP doors)
            for n in r.get("classes_unscored", []):
                ious[n] = None
        s["pixel_iou"] = {k: v for k, v in ious.items() if k != "background"}
    op = add_counts([r.get("openings") for r in ok])
    if op:
        s["openings"] = {**opening_rates(op), "counts": op}
        s["openings"]["passages"] = sum(r.get("passages", 0) for r in ok)
    ed = add_counts([r.get("connectivity") for r in ok])
    if ed and ed["ref"]:
        s["connectivity"] = {**edge_rates(ed), "counts": ed}
    gf = [r["gf"] for r in ok if r.get("gf")]
    if gf:
        s["gf"] = {"iou_median": med([g["iou"] for g in gf]), "iou_mean": float(np.mean([g["iou"] for g in gf])),
                   "area_err_pct_median": med([g["area_err_pct"] for g in gf]),
                   "area_err_pct_abs_median": med([abs(g["area_err_pct"]) for g in gf]),
                   "area_err_m2_median": med([g["area_err_m2"] for g in gf]),
                   "iou_below_0.9": sum(g["iou"] < 0.9 for g in gf)}
        gfo = [r["gf_with_outdoor"] for r in ok if r.get("gf_with_outdoor")]
        if gfo:
            s["gf"]["iou_median_with_outdoor"] = med([g["iou"] for g in gfo])
            s["gf"]["area_err_pct_median_with_outdoor"] = med([g["area_err_pct"] for g in gfo])
    oc = [r["ocr"] for r in ok if r.get("ocr")]
    if oc:
        s["ocr"] = {"sheets": len(oc), **{k: ratio(sum(o[k] for o in oc), sum(o["matched"] for o in oc))
                                          for k in ("named", "stamp_area", "stamp_area_within_5pct")}}
    return s


MISS_REASONS = ("merged", "oversized", "split", "missing", "outside building")


def by_type(rows):
    """Reference rooms per area type: how many were matched, and why the others were missed."""
    out = {}
    for r in rows:
        for t, hit, reason, merged in (r.get("rooms") or {}).get("ref_rooms", []):
            if t is None:                                 # a reference without area types (polygons only)
                t = "room"
            d = out.setdefault(t, {"ref": 0, "matched": 0, **{k: 0 for k in MISS_REASONS}, "merged_with": {}})
            d["ref"] += 1
            if hit:
                d["matched"] += 1
            else:
                d[reason] = d.get(reason, 0) + 1
                for m in merged:
                    d["merged_with"][m] = d["merged_with"].get(m, 0) + 1
    for d in out.values():
        d["recall"] = ratio(d["matched"], d["ref"])
        d["merged_with"] = dict(sorted(d["merged_with"].items(), key=lambda kv: -kv[1])[:5])
    return dict(sorted(out.items(), key=lambda kv: -(kv[1]["ref"] - kv[1]["matched"])))
