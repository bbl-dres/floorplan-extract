"""Tests of the evaluation metrics (fpeval.metrics) and the harness (fpeval.harness, fpeval.datasets): synthetic
geometry and masks for every metric, plus a few Swiss Dwellings test floors (oracle and render) when
data/floors-test.pkl is present."""
import json

import numpy as np
import pytest
import shapely
from shapely.geometry import box

from conftest import FLOORS, needs_floors, needs_fonts
from fpeval import metrics


# ---------- assignment and rooms ----------

def test_assign_maximises_matches_before_score():
    S = np.array([[0.55, 0.45], [0.45, 0.0]])           # plain max-sum would take the two 0.45 pairs
    pairs = metrics.assign(S, 0.5)
    assert (0, 0) in pairs and sum(S[i, j] >= 0.5 for i, j in pairs) == 1
    assert metrics.assign(np.zeros((0, 3)), 0.5) == []


def test_match_rooms_one_to_one():
    a, b = box(0, 0, 10, 10), box(10, 0, 20, 10)
    merged = box(0, 0, 19, 10)                           # IoU 0.53 with a, 0.45 with b
    r = metrics.match_rooms([merged], [a, b], 0.5, m2_per_unit=0.01)
    assert (r["matched"], r["ref"], r["pred"]) == (1, 2, 1)
    assert r["recall"] == 0.5 and r["precision"] == 1.0
    assert r["ref_pred"] == [0, None] and r["ref_iou"][1] == 0.0      # the merged room counts for one reference only
    assert r["best_iou"][1] == pytest.approx(0.45)                    # the old best-IoU matching gave it to both
    assert r["area_err_pct"] == [pytest.approx(90.0)] and r["area_err_m2"] == [pytest.approx(0.9)]
    reasons = metrics.miss_reasons([merged], [a, b], {0})
    assert reasons == {1: ("merged", [0], 0)}


def test_match_rooms_ignore_and_empty():
    room, balcony = box(0, 0, 10, 10), box(0, 12, 10, 15)
    r = metrics.match_rooms([room.buffer(-0.1), balcony], [room], ignore=[balcony])
    assert (r["matched"], r["pred"], r["ignored"]) == (1, 1, 1) and r["precision"] == 1.0
    r = metrics.match_rooms([], [room])
    assert r["matched"] == 0 and r["recall"] == 0.0 and r["precision"] is None


def test_miss_reasons():
    ref = [box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10)]
    pred = [box(0, 0, 4, 10), box(4, 0, 7, 10), box(7, 0, 10, 10),     # the first room in three pieces
            box(18, -2, 32, 12).union(box(32, 0, 38, 10))]  # the second inside a much larger region
    out = metrics.miss_reasons(pred, ref, set())
    assert out[0][:2] == ("split", []) and out[1] == ("oversized", [], 3) and out[2] == ("missing", [], None)


# ---------- masks ----------

def bar(y0, y1, x0=20, x1=180, shape=(100, 200)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_mask_scores_identical_and_shifted():
    ref = bar(40, 50)
    r = metrics.mask_rates(metrics.mask_scores(ref, ref))
    assert r["iou"] == 1 and r["boundary_f"] == 1 and r["centreline_f"] == 1
    near = metrics.mask_rates(metrics.mask_scores(bar(41, 51), ref, 2, 2.5))     # one pixel off
    assert near["iou"] == pytest.approx(9 / 11) and near["boundary_f"] > 0.95 and near["centreline_f"] > 0.95
    far = metrics.mask_rates(metrics.mask_scores(bar(46, 56), ref, 2, 2.5))      # six pixels off
    assert far["boundary_f"] < 0.6 and far["centreline_p"] == 0 and far["centreline_r"] == 0


def test_mask_scores_thick_wall_keeps_centre_line():
    """A wall drawn 2 px thicker on each side loses pixel IoU but not its centre line (review §4.6)."""
    r = metrics.mask_rates(metrics.mask_scores(bar(38, 52), bar(40, 50), 2, 2.5))
    assert r["iou"] == pytest.approx(10 / 14) and r["centreline_f"] > 0.95 and r["boundary_f"] > 0.95
    empty = metrics.mask_scores(np.zeros((10, 10), bool), np.zeros((10, 10), bool))
    assert empty["union"] == 0 and metrics.mask_rates(empty)["iou"] is None


def test_confusion_and_class_iou():
    ref = np.array([[0, 1, 1, 2]])
    pred = np.array([[0, 1, 2, 2]])
    cm = metrics.confusion(ref, pred, 3)
    assert cm.tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]
    assert metrics.class_iou(cm, ["bg", "wall", "door"]) == {"bg": 1.0, "wall": 0.5, "door": 0.5}


# ---------- openings and connectivity ----------

def test_opening_scores_and_confusion():
    ref = [("door", (0, 0)), ("window", (100, 0)), ("door", (200, 0))]
    pred = [("window", (5, 0)),                          # the first door read as a window
            ("door", (203, 0)), ("door", (400, 0))]      # the second door found; one extra door
    c = metrics.opening_scores(pred, ref, 25)
    assert c["door"] == {"ref": 2, "pred": 2, "matched": 1}
    assert c["window"] == {"ref": 1, "pred": 1, "matched": 0}
    conf = c["confusion"]
    assert conf["door"]["window"] == 1 and conf["door"]["door"] == 1
    assert conf["window"]["none"] == 1 and conf["none"]["door"] == 1
    assert c["found_any_kind"] == 2 and c["ref_any"] == 3 and c["pred_any"] == 3 and sorted(c["pairs_any"]) == [(0, 0), (1, 2)]
    rates = metrics.opening_rates(c)
    assert rates["door"]["recall"] == 0.5 and rates["door_as_window"] == 0.5


def test_match_points_one_to_one():
    pairs = metrics.match_points([(0, 0), (1, 0)], [(0.6, 0)], 1.0)
    assert len(pairs) == 1                               # one reference, at most one match
    assert metrics.match_points([], [(0, 0)], 1.0) == []


def test_edge_scores():
    pairs = [(0, 10), (1, 11), (2, 12)]                  # predicted room -> reference room
    ref_edges = [(10, 11), (11, 12), (12, 13)]
    pred_edges = [(0, 1), (0, 2), (2, 5)]                # correct, wrong, room 5 unmatched
    c = metrics.edge_scores(pred_edges, ref_edges, pairs)
    assert c == {"ref": 3, "pred": 3, "correct": 1, "ref_matched": 2, "pred_mapped": 2, "other_ref": 0, "other_found": 0}
    r = metrics.edge_rates(c)
    assert r["precision"] == pytest.approx(1 / 3) and r["recall_matched"] == 0.5
    c = metrics.edge_scores(pred_edges, ref_edges, pairs, ignore_edges=[(10, 12)])    # an open passage: neither right nor wrong
    assert (c["pred"], c["correct"], c["other_ref"], c["other_found"]) == (2, 1, 1, 1)


def test_region_scores():
    r = metrics.region_scores(box(0, 0, 11, 10), box(0, 0, 10, 10), 0.5)
    assert r["iou"] == pytest.approx(10 / 11) and r["area_err_pct"] == pytest.approx(10) and r["area_err_m2"] == pytest.approx(5)
    assert metrics.region_scores(box(0, 0, 1, 1), shapely.Polygon()) is None


def test_door_in_wall_takes_the_side_in_the_opening():
    from fpeval.datasets import cvcfp
    walls = shapely.union_all([box(0, 0, 100, 10), box(185, 0, 300, 10)])      # wall with an 85 px opening
    swing = box(100, 10, 185, 95)                         # the swing area below the opening
    c = cvcfp.door_in_wall(swing, walls, 5)
    assert np.allclose(c, (142.5, 10))


# ---------- Swiss Dwellings floors ----------

@needs_floors
def test_oracle_floors_score_near_perfect():
    from fpeval import oracle
    from fpeval.harness import POST
    from fpeval.score import score
    from fpx import DEFAULT, pipeline
    for rec in oracle.floors(FLOORS, 3):
        tf, shape, _ = oracle.frame(rec)
        sheet, _ = oracle.oracle_sheet(rec)
        ref = oracle.reference(rec, tf)
        assert ref["rooms"] and ref["edges"] and ref["openings"]
        assert not ref["edges"] & ref["open_edges"]
        assert all(0 <= x < shape[1] and 0 <= y < shape[0] for _, (x, y) in ref["openings"])
        assert ref["gf"].area >= sum(p.area for _, p in ref["rooms"])
        ref["label"] = sheet.label.copy()
        pipeline.run(sheet, stages=POST)
        row = score(sheet, ref, DEFAULT)
        assert row["wall_rates"]["iou"] > 0.98 and row["wall_rates"]["boundary_f"] > 0.95
        op = metrics.opening_rates(row["openings"])
        assert op["door"]["recall"] >= 0.9 and op["window"]["recall"] >= 0.9
        assert row["rooms"]["recall"] >= 0.6 and row["gf"]["iou"] > 0.8


@needs_floors
def test_harness_main_writes_summary(tmp_path):
    import harness
    out = tmp_path / "oracle.json"
    harness.main(["oracle", "--n", "2", "--out", str(out)])
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["summary"]["sheets"] == 2 and len(d["rows"]) == 2
    assert {"rooms", "walls", "openings", "connectivity", "gf"} <= set(d["summary"])
    assert d["by_area_type"]


@needs_floors
@needs_fonts
def test_render_floor_matches_reference_frame():
    from fpeval import oracle
    from fpeval.datasets import render
    from synth import Renderer
    rec = next(oracle.floors(FLOORS, 1))
    tf, shape, _ = oracle.frame(rec)
    r = Renderer(FLOORS)
    img, lab, st, info = render.render_floor(r, rec, tf, shape, np.random.default_rng([0, 0]), 50)
    assert img.shape == shape + (3,) and lab.shape == shape and img.dtype == np.uint8
    assert info["distortion_px"] <= render.MAX_SHIFT_PX
    ref = oracle.labels(rec, tf, shape)
    assert render.displacement(lab == 1, ref == 1) <= 2 * render.MAX_SHIFT_PX   # walls where the floor has them
    img2, lab2, _, _ = render.render_floor(r, rec, tf, shape, np.random.default_rng([0, 0]), 50)
    assert (img2 == img).all() and (lab2 == lab).all()   # seeded: the same render again


def test_displacement():
    from fpeval.datasets import render
    a = bar(40, 50)
    assert render.displacement(a, a) == 0
    assert render.displacement(bar(43, 53), a) == pytest.approx(3, abs=0.5)


def test_segmenter_cache_keeps_head_maps(tmp_path):
    """run_sheet stores the v2 head maps next to label and prob (float16) and restores them on a hit; a cache file
    without them (older) leaves the Sheet fields None."""
    from types import SimpleNamespace
    from conftest import two_rooms
    from fpeval import harness
    from fpx.model import CLASSES, HEADS
    ctx = SimpleNamespace(mode="cache-test", cfg=harness.DEFAULT, engine=None, cache=tmp_path, model=object(),
                          cache_path=lambda sid, key: tmp_path / f"{sid}.npz")
    sheet = two_rooms()
    sheet.interior_prob = np.random.default_rng(0).random(sheet.label.shape).astype(np.float32)

    def segment(sheet_, model, cfg):               # a stand-in for the segmenter: the fixture's labels and one head map
        sheet_.prob = (np.arange(len(CLASSES))[:, None, None] == sheet_.label[None]).astype(np.float32)
        sheet_.seg_label = sheet_.label.copy()
        sheet_.interior_prob = sheet.interior_prob
        return sheet_.label
    import fpx.segment
    fpx.segment.segment, orig = segment, fpx.segment.segment
    try:
        harness.run_sheet(sheet, {}, ctx, ["k"])
        z = np.load(tmp_path / "t.npz")
        assert "interior_prob" in z.files and z["interior_prob"].dtype == np.float16
        assert set(z.files) == {"label", "prob", "interior_prob"}
        again = two_rooms()
        ref, seg, times = harness.run_sheet(again, {}, ctx, ["k"])
        assert times["cached"] and np.allclose(again.interior_prob, sheet.interior_prob, atol=1e-3)
        assert again.interior_prob.dtype == np.float32 and again.boundary_prob is None
        np.savez_compressed(tmp_path / "t.npz", label=sheet.label, prob=sheet.prob.astype(np.uint8))    # an old file
        old = two_rooms()
        harness.run_sheet(old, {}, ctx, ["k"])
        assert old.interior_prob is None and all(getattr(old, f"{h}_prob") is None for h in HEADS)
    finally:
        fpx.segment.segment = orig
