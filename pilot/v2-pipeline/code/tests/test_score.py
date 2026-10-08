"""The shared scorer (fpeval.score) against the protocols it replaced, on the two-room fixture with a synthetic
reference: the harness row, cubicasa_eval's greedy opening matching and room rule, bench's best-IoU room figures and
pixel IoU, evaluate's any-kind opening matching. Equal counts on the fixture; the protocols differ only where the
review asked for one rule (C6: 0.5 m opening tolerance, Hungarian matching, rooms + fragments >= 0.25 m²)."""
import numpy as np
import pytest
from shapely.geometry import box

from conftest import two_rooms, px
from fpeval import metrics
from fpeval.score import OPENING_TOL_M, score
from fpx import DEFAULT, pipeline
from fpx.model import CLASSES, DOOR, WALL, WINDOW

M = DEFAULT.px_per_m


def reference():
    """The two-room fixture's own geometry as a Reference (working px): two rooms, the door and the window, the label
    map, the door edge and the GF."""
    sheet = two_rooms()
    rooms = [("room", box(px(1.3), px(1.3), px(5.85), px(6.7))), ("room", box(px(6.15), px(1.3), px(10.7), px(6.7)))]
    openings = [("door", (px(6.0), px(3.95))), ("window", (px(3.6), px(1.15)))]
    return sheet, {"rooms": rooms, "ignore": [], "edges": {frozenset((0, 1))}, "open_edges": set(), "openings": openings,
                   "label": sheet.label.copy(), "gf": box(px(1), px(1), px(11), px(7)), "unscored": []}


@pytest.fixture(scope="module")
def scored():
    sheet, ref = reference()
    pipeline.run(sheet, stages=pipeline.AFTER_SEGMENTATION)
    return sheet, ref, score(sheet, ref, DEFAULT)


def test_row_vocabulary(scored):
    sheet, ref, row = scored
    assert {"rooms", "connectivity", "walls", "wall_rates", "classes", "class_names", "openings", "passages", "gf", "ocr"} <= set(row)
    assert row["rooms"]["matched"] == 2 and row["rooms"]["pred"] == 2 and row["rooms"]["recall"] == 1.0
    assert row["rooms"]["ref_assigned"] == [[sheet.rooms[0]["id"], pytest.approx(row["rooms"]["pair_iou"][0]), pytest.approx(row["rooms"]["pair_iou"][0])],
                                            [sheet.rooms[1]["id"], pytest.approx(row["rooms"]["pair_iou"][1]), pytest.approx(row["rooms"]["pair_iou"][1])]]
    assert row["connectivity"]["correct"] == 1 and row["connectivity"]["by_opening"] == {"door": {"correct": 1, "open": 0, "wrong": 0, "wrong at merged room": 0}}
    assert row["wall_rates"]["iou"] == 1.0 and row["openings"]["door"]["matched"] == 1 and row["openings"]["window"]["matched"] == 1
    assert row["openings"]["found_any_kind"] == 2 and row["passages"] == 0 and row["gf"]["iou"] > 0.99


def legacy_cubicasa_openings(ref_o, pred_o, tol):
    """cubicasa_eval's old matching: each reference centre takes the nearest unused prediction of any kind within tol."""
    used, hit = set(), {"door": 0, "window": 0}
    for kind, c in ref_o:
        best = min(((np.hypot(*(np.asarray(o["centre"]) - c)), j) for j, o in enumerate(pred_o) if j not in used), default=(1e9, None))
        if best[0] < tol:
            used.add(best[1])
            hit[kind] += 1
    return hit, len(used)


def test_cubicasa_protocol_counts(scored):
    """Doors and windows found by any kind, openings matched, rooms of >= 1 m² one-to-one: the old figures."""
    sheet, ref, row = scored
    pred_o = [o for o in sheet.openings if o["kind"] != "passage"]
    hit, matched = legacy_cubicasa_openings(ref["openings"], pred_o, OPENING_TOL_M * M)
    conf = row["openings"]["confusion"]
    assert hit["door"] == row["openings"]["door"]["ref"] - conf["door"]["none"] == 1
    assert hit["window"] == row["openings"]["window"]["ref"] - conf["window"]["none"] == 1
    assert matched == row["openings"]["found_any_kind"] == 2
    old_rooms = [r["poly"] for r in sheet.rooms if r["poly"].area >= M ** 2]      # the old rule: rooms of 1 m² or more
    mr = metrics.match_rooms(old_rooms, [p for _, p in ref["rooms"]], 0.5, 1 / M ** 2)
    assert (mr["matched"], mr["ref"], mr["pred"]) == (row["rooms"]["matched"], row["rooms"]["ref"], row["rooms"]["pred"])
    assert mr["mean_iou_ref"] == pytest.approx(row["rooms"]["mean_iou_ref"])


def test_bench_protocol_counts(scored):
    """bench's old scores: per-class pixel IoU from the label, best-IoU room matches and their mean."""
    sheet, ref, row = scored
    k = len(CLASSES)
    cm = np.bincount(ref["label"].ravel().astype(np.int64) * k + sheet.label.ravel(), minlength=k * k).reshape(k, k)
    iou = lambda i: float(cm[i, i] / max(cm[i].sum() + cm[:, i].sum() - cm[i, i], 1))
    new = metrics.class_iou(row["classes"], CLASSES)
    assert (round(iou(WALL), 3), round(iou(DOOR), 3), round(iou(WINDOW), 3)) == (round(new["wall"], 3), round(new["door"], 3), round(new["window"], 3))
    refr = [p for _, p in ref["rooms"]]
    best = [max((g.intersection(r["poly"]).area / g.union(r["poly"]).area for r in sheet.rooms), default=0) for g in refr]
    assert int(sum(b >= 0.5 for b in best)) == sum(a[2] >= 0.5 for a in row["rooms"]["ref_assigned"]) == 2
    assert float(np.mean(best)) == pytest.approx(np.mean([a[2] for a in row["rooms"]["ref_assigned"]]))


def test_evaluate_protocol_counts(scored):
    """evaluate's old any-kind greedy matching of openings (reference without kinds) gives the same counts as the
    any-kind part of opening_scores on a reference typed 'opening'."""
    sheet, ref, row = scored
    untyped = [("opening", xy) for _, xy in ref["openings"]]
    pred = [(o["kind"], tuple(o["centre"])) for o in sheet.openings]
    c = metrics.opening_scores(pred, untyped, OPENING_TOL_M * M)
    used, hits = set(), []
    for _, xy in untyped:
        cands = [(np.hypot(*(np.asarray(p) - xy)), j) for j, (_, p) in enumerate(pred) if j not in used]
        dist, j = min(cands, default=(1e9, None))
        hits.append(j if dist < OPENING_TOL_M * M else None)
        if dist < OPENING_TOL_M * M:
            used.add(j)
    assert c["found_any_kind"] == sum(h is not None for h in hits) == 2 and c["ref_any"] == 2 and c["pred_any"] == 2
    assert c["door"]["ref"] == 0 and c["confusion"]["none"]["door"] == 0       # untyped references take no part in the confusion
