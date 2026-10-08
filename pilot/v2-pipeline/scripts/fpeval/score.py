"""One scorer for every benchmark: score(sheet, ref, cfg) turns a sheet whose stages have run and a Reference (the
dict fpeval.oracle.reference() returns, see fpeval.datasets) into one row of metrics with one vocabulary:

    rooms         one-to-one at IoU >= ROOM_IOU over rooms and flagged fragments (every region of room_min_area or
                  more is a prediction), counts, rates, area errors, why each reference room was missed, ref_rooms
                  and ref_assigned per reference room (type, hit, reason, assigned prediction and IoU, best IoU)
    connectivity  room-to-room edges mapped through the room matching, by opening kind
    walls         pixel counts of the cleaned wall mask (wall_rates: IoU, boundary F at BOUNDARY_TOL_PX, centre lines
                  within CENTRE_TOL_M); classes: the segmenter's pixel confusion (seg_label, as predicted)
    openings      door and window centres within OPENING_TOL_M, with the door/window confusion; passages counted
    gf            IoU and area error of the gross outline (deducted voids added back)
    ocr           names and stamp areas of the matched rooms, when the text stage ran

Geometry is in working pixels; cfg.px_per_m turns the tolerances into pixels (a sheet already in metres scores with
px_per_m = 1). Rows are pooled over sheets by fpeval.metrics.summarise().
"""
import cv2
import numpy as np
import shapely
from shapely import affinity

from fpeval import metrics
from fpx.model import CLASSES, WALL

ROOM_IOU = 0.5          # a room is found at this IoU: the usual detection threshold; below it the prediction is another region
OPENING_TOL_M = 0.5     # opening centres match within half a door width: a door matches its own symbol, not the next one
CENTRE_TOL_M = 0.05     # wall centre lines match within a pixel on either side at 2 cm/px: a thickness error does not count
BOUNDARY_TOL_PX = 2     # wall boundaries match within the half-pixel offset of a traced contour plus a pixel of anti-aliasing
KIND = {"door": "door", "exterior door": "door", "window": "window", "interior opening": "window"}   # scored opening kinds


def score(sheet, ref, cfg, seg_label=None):
    """One row of metrics for a sheet whose stages have run. ref: a Reference (keys missing or None are not scored);
    seg_label: the segmenter's label as predicted (default sheet.seg_label, else sheet.label)."""
    m = cfg.px_per_m
    m2 = 1 / m ** 2
    row = {}
    if ref.get("rooms") is not None:
        regions = sheet.rooms + sheet.fragments
        pred = [r["poly"] for r in regions]
        refp = [p for _, p in ref["rooms"]]
        types = [t for t, _ in ref["rooms"]]
        mr = metrics.match_rooms(pred, refp, ROOM_IOU, m2, ignore=ref.get("ignore", []))
        hit = {j for _, j in mr["pairs"]}
        miss = metrics.miss_reasons(pred, refp, hit)
        b = getattr(sheet, "building_rough", None)
        for j, (reason, _, best) in list(miss.items()):    # a missing room outside the building mask: another drawing or wing
            if reason == "missing" and b is not None:
                x, y = refp[j].representative_point().coords[0]
                if not b[int(np.clip(y, 0, b.shape[0] - 1)), int(np.clip(x, 0, b.shape[1] - 1))]:
                    miss[j] = ("outside building", [], best)
        row["rooms"] = {k: mr[k] for k in ("ref", "pred", "matched", "ignored", "recall", "precision", "mean_iou", "mean_iou_ref")}
        row["rooms"].update(area_err_pct=mr["area_err_pct"], area_err_m2=mr["area_err_m2"], pair_iou=mr["pair_iou"],
                            area_err_pct_median=metrics.med(mr["area_err_pct"]),
                            fragments=len(sheet.fragments),
                            ref_rooms=[[types[j], j in hit, miss.get(j, (None,))[0], [types[k] for k in miss.get(j, (None, []))[1]]]
                                       for j in range(len(refp))],
                            ref_assigned=[[regions[i]["id"] if i is not None else None, mr["ref_iou"][j], mr["best_iou"][j]]
                                          for j, i in enumerate(mr["ref_pred"])],
                            pred_best_iou=mr["best_iou_pred"],
                            missed=[{"type": types[j], "m2": refp[j].area * m2, "reason": miss[j][0],
                                     "merged_with": [types[k] for k in miss[j][1]], "best_iou": mr["best_iou"][j]}
                                    for j in sorted(miss)])
        if ref.get("edges") is not None:
            idx = {r["id"]: i for i, r in enumerate(regions)}
            kind = {o["id"]: o["kind"] for o in sheet.openings}
            pe = [(idx[a], idx[b], kind.get(d.get("opening"), "?")) for a, b, d in sheet.connectivity.edges(data=True)
                  if a in idx and b in idx]
            other = ref.get("open_edges", set())
            row["connectivity"] = metrics.edge_scores([e[:2] for e in pe], ref["edges"], mr["pairs"], other)
            # which openings make correct and wrong connections; a wrong one at a prediction that swallowed another
            # reference room (merged, oversized) follows from that room error, not from the connection logic
            absorbed = {best for reason, _, best in miss.values() if reason in ("merged", "oversized")}
            to_ref, by = dict(mr["pairs"]), {}
            for a, b, k in pe:
                if a in to_ref and b in to_ref:
                    e = frozenset((to_ref[a], to_ref[b]))
                    v = ("correct" if e in ref["edges"] else "open" if e in other
                         else "wrong at merged room" if a in absorbed or b in absorbed else "wrong")
                    by.setdefault(k, {"correct": 0, "open": 0, "wrong": 0, "wrong at merged room": 0})[v] += 1
            row["connectivity"]["by_opening"] = by
        if getattr(sheet, "text", None):                    # OCR ran: names and stamp areas of the matched rooms
            named = stamp = close = 0
            for i, j in mr["pairs"]:
                r = regions[i]
                named += bool(r.get("names"))
                if r.get("area_stamp"):
                    stamp += 1
                    close += abs(r["area_stamp"] - refp[j].area * m2) <= 0.05 * refp[j].area * m2
            row["ocr"] = {"matched": len(mr["pairs"]), "named": named, "stamp_area": stamp, "stamp_area_within_5pct": close,
                          "text_items": len(sheet.text)}
    if ref.get("label") is not None:
        rl = ref["label"]
        seg = seg_label if seg_label is not None else (sheet.seg_label if getattr(sheet, "seg_label", None) is not None else sheet.label)
        row["walls"] = metrics.mask_scores(sheet.wall_mask, rl == WALL, BOUNDARY_TOL_PX, CENTRE_TOL_M * m)
        row["wall_rates"] = metrics.mask_rates(row["walls"])
        row["classes"] = metrics.confusion(rl, seg, len(CLASSES)).tolist()
        row["class_names"] = CLASSES
        row["classes_unscored"] = ref.get("unscored", [])
    if ref.get("openings") is not None:
        pred = [(KIND[o["kind"]], tuple(o["centre"])) for o in sheet.openings if o["kind"] in KIND]
        row["openings"] = metrics.opening_scores(pred, ref["openings"], OPENING_TOL_M * m)
        row["passages"] = sum(o["kind"] == "passage" for o in sheet.openings)
    if ref.get("gf") is not None:
        gross = shapely.union_all([sheet.gf] + [v["poly"] for v in sheet.voids if v.get("gf_deducted")])
        row["gf"] = metrics.region_scores(gross, ref["gf"], m2)
        if ref.get("gf_with_outdoor") is not None:
            row["gf_with_outdoor"] = metrics.region_scores(gross, ref["gf_with_outdoor"], m2)
    return row


def deskew_reference(ref, matrix, shape):
    """Move the reference into the deskewed frame when preprocessing rotated the sheet."""
    M = np.asarray(matrix, float)
    a, b, xoff, d, e, yoff = M.ravel()
    T = lambda g: affinity.affine_transform(g, [a, b, d, e, xoff, yoff])
    out = dict(ref)
    for key in ("ignore",):
        if ref.get(key) is not None:
            out[key] = [T(p) for p in ref[key]]
    if ref.get("rooms") is not None:
        out["rooms"] = [(k, T(p)) for k, p in ref["rooms"]]
    for key in ("gf", "gf_with_outdoor"):
        if ref.get(key) is not None:
            out[key] = T(ref[key])
    if ref.get("openings") is not None:
        out["openings"] = [(k, tuple(M @ [x, y, 1.0])) for k, (x, y) in ref["openings"]]
    if ref.get("label") is not None:
        out["label"] = cv2.warpAffine(ref["label"], M, (shape[1], shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
    return out
