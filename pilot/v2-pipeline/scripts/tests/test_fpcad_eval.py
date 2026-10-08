"""FloorPlanCAD evaluator (fpeval.datasets.fpcad, fpeval.fpcad): parsing, rendering alignment, votes, metrics, scale check,
one block end to end with a stub segmenter. Uses three test blocks of the local FloorPlanCAD copy; skipped when it is absent."""
import math

import numpy as np
import pytest

from fpeval import fpcad as fe
from fpeval.datasets import fpcad as fd
from conftest import FPCAD_BLOCKS, dark_is_wall, needs_fpcad as needs_data
from fpx.model import BG, DOOR, STAIRS, WALL, WINDOW

NORMAL, DENSE, REDUCED = FPCAD_BLOCKS


def test_arc_points_semicircle():
    pts, r = fd.arc_points((0.0, 0.0), 1, 1, 0, 0, 1, (2.0, 0.0))
    assert r == pytest.approx(1)
    assert np.allclose(np.hypot(pts[:, 0] - 1, pts[:, 1]), 1, atol=1e-9)
    assert pts[:, 1].min() == pytest.approx(-1, abs=0.03) and pts[:, 1].max() < 1e-9   # sweep 1: over y = -1
    assert np.hypot(*np.diff(pts, axis=0).T).sum() == pytest.approx(math.pi, rel=0.02)
    pts, _ = fd.arc_points((0.0, 0.0), 1, 1, 0, 0, 0, (2.0, 0.0))
    assert pts[:, 1].max() == pytest.approx(1, abs=0.03)
    pts, r = fd.arc_points((0.0, 0.0), 0.2, 0.2, 0, 0, 1, (2.0, 0.0))  # radius too small: scaled up to reach
    assert r == pytest.approx(1)


def test_path_and_ellipse_points():
    pts, kind, _ = fd.path_points("M 1,2 L 4,6")
    assert kind == "line" and np.allclose(pts, [[1, 2], [4, 6]])
    pts, kind, r = fd.path_points("M 75.234,37.985 A 2.5655,1.60164 3.97569e-16 1,0 75.234,35.843")
    assert kind == "arc" and r > 0 and len(pts) > 4
    e = fd.ellipse_points(0, 0, 2, 1, "rotate(90.0,0,0)")
    assert np.abs(e[:, 1]).max() == pytest.approx(2, abs=0.01) and np.abs(e[:, 0]).max() == pytest.approx(1, abs=0.01)


def test_column_layers():
    assert fd.is_column_layer("COLUMN") and fd.is_column_layer("S-COLS") and fd.is_column_layer("砼柱")
    assert fd.is_column_layer("xref$0$S_COLU_HATCH")
    assert not fd.is_column_layer("柱网") and not fd.is_column_layer("WALL") and not fd.is_column_layer("尺寸标注-柱")


@needs_data
def test_parse_block():
    prims = fd.parse_svg(NORMAL)
    assert len(prims) == 15
    walls = [p for p in prims if p["sid"] == 1]
    assert len(walls) == 4 and all(p["layer"] == "WALL" and p["kind"] == "line" for p in walls)
    assert walls[0]["length"] == pytest.approx(100 - 30.975255, abs=1e-6)


@needs_data
def test_render_aligns_with_samples():
    """Every primitive's samples land on ink of the render: rendering and the sampling convention agree."""
    for path in (NORMAL, DENSE):
        img = fd.render(path.read_bytes())
        assert img.shape == (500, 500, 3)
        assert np.median(img) == 255 and img.min() == 0                 # white page, black ink
        prims = fd.parse_svg(path)
        _, r, c = fd.sample_pixels(prims, fd.PPU, img.shape[:2])
        assert (img[r, c].mean(1) < 200).mean() > 0.97


def test_sampling_density():
    prims = [{"pts": np.array([[0.0, 0.5], [2.0, 0.5]])}, {"pts": np.array([[1.0, 1.0], [1.0, 1.0]])}]
    i, r, c = fd.sample_pixels(prims, 5.0, (20, 20))
    assert (i == 0).sum() == 21 and (i == 1).sum() == 1                # 10 px at <= 0.5 px: 20 steps, 21 points
    assert c[i == 0].min() == 0 and c[i == 0].max() == 10 and (r == 2).all() is not None


def test_votes():
    label = np.zeros((20, 20), np.uint8)
    label[:, 12:] = WALL                                                # right 40 % wall
    label[5, 0:2] = DOOR
    prims = [{"pts": np.array([[0.1, 1.1], [3.9, 1.1]])},              # 19 px along row 5 at ppu 5
             {"pts": np.array([[2.6, 3.0], [3.9, 3.0]])},              # entirely in the wall
             {"pts": np.array([[0.2, 0.2], [1.0, 0.2]])},              # entirely background
             {"pts": np.array([[2.2, 2.0], [2.38, 2.0]])}]             # one pixel left of the wall region
    v, counts = fe.vote(prims, label, 5.0)
    assert counts[0].sum() == 39 and counts[2].sum() == 9
    assert v["paper"].tolist() == [BG, WALL, BG, BG]                    # Eq. 10: background is a label like any other
    assert v["fg"].tolist() == [WALL, WALL, BG, BG]                     # foreground majority: 16 wall vs 4 door samples
    assert v["paper3x3"].tolist() == [BG, WALL, BG, WALL]               # one-pixel tolerance reaches the wall
    grown = fe.tolerant_label(label)
    assert (grown[:, 11] == WALL).all() and (grown[:, 10] == BG).all() and grown[4, 2] == DOOR


def test_class_scores_and_pooling():
    cm = np.zeros((fd.N_GT, fd.K))
    cm[1, WALL] = 8                    # wall hit
    cm[1, BG] = 2                      # wall missed
    cm[0, WALL] = 4                    # background called wall
    cm[3, DOOR] = 5                    # single door hit at category level
    cm[9, DOOR] = 1                    # window called door
    cm[2, WINDOW] = 3                  # curtain wall called window: ignored in "paper"
    cm[30, STAIRS] = 2
    s = fe.class_scores(cm, "paper")
    assert s["Wall"]["F1"] == pytest.approx(8 / (8 + 2 + 1))
    assert s["Door"]["P"] == pytest.approx(5 / 6) and s["Door"]["R"] == 1
    assert s["Window"]["TP"] == 0 and s["Window"]["FP"] == 0 and s["Window"]["FN"] == 1
    tp, fp, fn = 8 + 5 + 0 + 2, 4 + 1, 2 + 1
    assert s["pooled"]["F1"] == pytest.approx(tp / (tp + 0.5 * fp + 0.5 * fn))
    assert s["mean4"] == pytest.approx(np.mean([s[c]["F1"] for c in fd.CATS]))
    assert fe.class_scores(cm, "curtain-as-other")["Window"]["FP"] == 3
    assert fe.class_scores(cm, "curtain-as-window")["Window"]["TP"] == 3
    assert fe.class_scores(cm, "wall+curtain")["Wall"]["FN"] == 2 + 3


def test_wall_pq():
    def row(entries):
        return {"cm": {v: entries for v in fe.VOTES}}
    good = row([[1, WALL, 1, 9.0], [1, BG, 1, 1.0], [0, WALL, 1, 1.0]])    # IoU 9 / 11
    bad = row([[1, WALL, 1, 1.0], [1, BG, 1, 5.0]])                       # IoU 1 / 6: FP and FN
    pq = fe.stuff_pq([good, bad], "paper")
    assert pq["TP"] == 1 and pq["FP"] == 1 and pq["FN"] == 1
    assert pq["SQ"] == pytest.approx(9 / 11) and pq["RQ"] == pytest.approx(0.5)


@needs_data
def test_scale_check():
    assert fd.scale_check(fd.parse_svg(NORMAL))["status"] == "ok"
    sc = fd.scale_check(fd.parse_svg(REDUCED))                          # plan drawn about 5x too small
    assert sc["status"] == "off" and sc["cue"] == "doors" and 0.15 < sc["factor"] < 0.25
    ppu, sc = fd.scale_decision(fd.parse_svg(REDUCED), "rescale")
    assert sc["action"] == "rescaled" and ppu == pytest.approx(fd.PPU / sc["factor"])
    assert fd.scale_decision(fd.parse_svg(REDUCED), "skip")[0] is None
    stray = [{"sid": 3, "kind": "arc", "radius": 0.15, "pts": np.zeros((2, 2))}] * 3
    assert fd.scale_check(stray)["status"] == "unverified"              # stray arcs: cue ignored


@needs_data
def test_gt_mask_fills_wall():
    prims = fd.parse_svg(NORMAL)
    m = fd.gt_mask(prims, (500, 500), fd.PPU)
    assert (m[352:360, 400] == WALL).all()                              # between the wall lines at y 70.06 and 72.06
    assert (m[300, 400] == BG) and set(np.unique(m)) <= {BG, WALL}


@needs_data
def test_evaluate_block_with_stub(tmp_path):
    row = fe.evaluate_block(NORMAL, dark_is_wall(), overlay=tmp_path / "ov.png")
    assert not row["skipped"] and row["size_px"] == [500, 500] and row["scale"]["action"] == "none"
    for v in fe.VOTES:
        assert sum(c for _, _, c, _ in row["cm"][v]) == row["n_prims"]
    assert np.array(row["pix_cm"]).sum() == 500 * 500
    m = fe.metrics([row])
    assert m["paper"]["paper"]["count"]["Wall"]["R"] == 1.0           # the stub calls all ink wall
    assert (tmp_path / "ov.png").exists()
    s = fe.summarise([row], {"test": True}, wall_clock=1.0)
    text = fe.table(s, "stub")
    assert "DeepLabv3+ R101" in text and "stub, paper3x3" in text and "No published number" in text
    rep = fe.report(s, "stub")
    assert "pixel IoU" in rep and "wall PQ" in rep and "recall by primitive kind" in rep
