"""Renderer v3 and segmenter v2: seeded determinism, extra targets aligned with the labels (swing on door symbols
only, shafts and lifts as voids, the negative set as background), the floor-record normalisation of sd_prepare, the
sampling share of the sources, the v1 call signatures the evaluation harness uses, the v1 checkpoint giving the same
output as before model v2, and the training loss and validation cache of train.py."""
import os
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import segmentation_models_pytorch as smp
import shapely
import torch

from common import DATA
from conftest import needs_ifc, needs_render, needs_v1_model
from fpx import DEFAULT, Config, Sheet
from fpx.model import CLASSES, HEADS, WALL
from fpx.segment import build_model, checkpoint, load_model, normalise, segment

ROOT = Path(__file__).resolve().parents[1]
VAL, IFC, V1 = DATA / "floors-val.pkl", DATA / "floors-ifc.pkl", DATA / "model/segmenter.pt"
QUIET = dict(context=False, ornament=False, reveals=False, niches=False, pilasters=False, column_grid=False, zones=False,
             rubber_stamp=False, red_pen=False, annotations=False, title_block=False, legend=False, keyplan=False,
             multi=False, outdoor_hatch=False, room_hatch=0.0, room_fill=False, patterns=False, dims=False, axes=False,
             features=False, text=False, walls="solid", era="cad")
NEGATIVES = dict(dim_chains=True, axis_bubbles=True, section_marks=True, detail_bubbles=True, opening_tags=True,
                 level_marks=True, furniture=True)


@pytest.fixture(scope="module")
def renderer():
    from synth import Renderer
    return Renderer(VAL, size=256)


def crc(*arrays):
    return [zlib.crc32(np.ascontiguousarray(a).tobytes()) for a in arrays]


@needs_render
@pytest.mark.parametrize("negatives", [False, True])
def test_seed_reproduces_sample(negatives):
    from synth import Renderer
    renderer, other = Renderer(VAL, size=256, negatives=negatives), Renderer(VAL, size=256, negatives=negatives)
    for seed in (1, 2, 3):
        a = renderer.sample(np.random.default_rng(seed), targets=True)
        b = other.sample(np.random.default_rng(seed), targets=True)
        assert crc(a[0], a[1], *a[3].values()) == crc(b[0], b[1], *b[3].values()) and a[2] == b[2]
        c = renderer.sample(np.random.default_rng(seed))                    # without targets: same image and labels
        assert len(c) == 3 and crc(c[0], c[1]) == crc(a[0], a[1])


@needs_render
def test_no_process_dependence():
    """Python's hash() changes per process; the renderer must not depend on it (room colours did in v1)."""
    code = ("import sys, zlib, numpy as np; sys.path.insert(0, '.'); from synth import Renderer; from common import DATA; "
            "r = Renderer(DATA / 'floors-val.pkl', size=192, negatives=True); "
            "print([zlib.crc32(r.sample(np.random.default_rng(s), style={'room_fill': True})[0].tobytes()) for s in (5, 6)])")
    outs = [subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": h}, timeout=300).stdout.strip() for h in ("1", "2")]
    assert outs[0] and outs[0] == outs[1]


@needs_render
def test_targets_aligned_with_labels(renderer):
    texts = 0
    for seed in range(8):
        img, lab, tags, tg = renderer.sample(np.random.default_rng(100 + seed), targets=True, style={"text": True})
        assert img.shape == (256, 256, 3) and lab.shape == (256, 256) and set(tg) == set(HEADS)
        for k, t in tg.items():
            assert t.shape == lab.shape and t.dtype == np.uint8 and set(np.unique(t)) <= {0, 1, 255}, k
        known = tg["interior"] != 255
        walls = (lab == WALL) & known
        if walls.sum() > 50:                       # walls lie inside the building outline
            assert (tg["interior"][walls] == 1).mean() > 0.97, tags
        void = (tg["void"] == 1) & known
        if void.any():
            assert (tg["interior"][void] == 1).mean() > 0.97
        texts += tg["text"].sum() > 0
    assert texts >= 4                              # stamps and dimensions leave text boxes


def box_px(x0, y0, x1, y1, s=20.0):
    """Axis-aligned box in metres -> pixel polygon at s px/m, 20 px off the origin (the frame of floor_px)."""
    return shapely.box(x0 * s + 20, y0 * s + 20, x1 * s + 20, y1 * s + 20)


def floor_px(s=20.0):
    """A 10 x 6 m building drawn in pixel coordinates at s px/m: outer walls 0.3 m; an interior wall at x = 6 m; the
    west part split into two areas without a wall at y = 3.5 m (open plan); a void in the east room; a balcony south."""
    box = lambda x0, y0, x1, y1: box_px(x0, y0, x1, y1, s)
    outer, inner = box(0, 0, 10, 6), box(0.3, 0.3, 9.7, 5.7)
    walls = shapely.union_all([outer.difference(inner), box(5.85, 0.3, 6.15, 5.7)])
    g = {"walls": walls, "doors": box(5.85, 2.5, 6.15, 3.4), "windows": box(2, 0, 3.2, 0.3), "columns": shapely.Polygon(),
         "railings": shapely.Polygon()}
    areas = [("KITCHEN", box(0.3, 0.3, 5.85, 3.5), 18.0), ("LIVING_ROOM", box(0.3, 3.5, 5.85, 5.7), 12.0),
             ("ROOM", box(6.15, 0.3, 9.7, 5.7), 19.0), ("VOID", box(7, 1, 8.5, 2.5), 2.25), ("BALCONY", box(0, 6, 4, 7.5), 6.0)]
    return g, areas, (lambda x, y: (int(round(x * s + 20)), int(round(y * s + 20))))


@needs_render
def test_draw_targets_on_known_geometry(renderer):
    g, areas, px = floor_px()
    rng = np.random.default_rng(0)
    st = renderer.style(rng, **QUIET)
    tg = {}
    img, lab = renderer.draw(st, 20.0, g, [], [], areas, 0, rng, targets=tg)
    v = lambda k, x, y: tg[k][px(x, y)[1], px(x, y)[0]]
    assert lab[px(0.15, 3)[1], px(0.15, 3)[0]] == WALL
    for x, y in ((3, 2), (3, 4.5), (8, 4), (0.15, 3), (9.85, 3)):         # rooms, void, walls: inside
        assert v("interior", x, y) == 1
    for x, y in ((2, 6.8), (3, 7.6), (11, 3)):                                # balcony and outside
        assert v("interior", x, y) == 0
    assert v("boundary", 3, 3.5) == 1 and v("boundary", 3, 2) == 0           # open-plan boundary without a wall
    assert v("boundary", 2, 6) == 1                                           # balcony outline
    assert v("void", 7.75, 1.75) == 1 and v("void", 3, 2) == 0
    assert not tg["text"].any()                                               # text switched off


@needs_render
def test_void_target_includes_shafts_and_lifts(renderer):
    """Shafts and lifts are slab openings like a Luftraum (the drawings cross them alike): void 1, inside the outline."""
    g, areas, px = floor_px()
    areas = areas + [("SHAFT", box_px(6.5, 4, 7.5, 5), 1.0), ("ELEVATOR", box_px(8, 3.8, 9.5, 5.5), 2.55)]
    tg = {}
    renderer.draw(renderer.style(np.random.default_rng(0), **QUIET, void_cross=True), 20.0, g, [], [], areas, 0,
                  np.random.default_rng(0), targets=tg)
    v = lambda k, x, y: tg[k][px(x, y)[1], px(x, y)[0]]
    for x, y in ((7, 4.5), (8.75, 4.65), (7.75, 1.75)):                       # shaft, lift, Luftraum
        assert v("void", x, y) == 1 and v("interior", x, y) == 1, (x, y)
    assert v("void", 8.75, 2) == 0 and v("void", 3, 2) == 0                   # the room around them is not


@needs_render
def test_swing_target_marks_door_symbols(renderer):
    """The swing target is the leaf and the sector of a door symbol, never a casement sash and nothing for gaps."""
    g, areas, px = floor_px()
    rng = np.random.default_rng(0)
    tg = {}
    renderer.draw(renderer.style(rng, **QUIET, doors="arc", windows="casement"), 20.0, g, [], [], areas, 0, rng, targets=tg)
    ys, xs = np.nonzero(tg["swing"] == 1)
    assert len(xs) > 50
    door, window = px(6, 2.95), px(2.6, 0.15)
    assert np.hypot(xs - door[0], ys - door[1]).max() < 1.5 * 20              # all of it around the 0.9 m door
    assert np.hypot(xs - window[0], ys - window[1]).min() > 0.5 * 20          # none at the casement window
    for kind in ("gap", "sliding"):
        tg = {}
        renderer.draw(renderer.style(np.random.default_rng(1), **QUIET, doors=kind), 20.0, g, [], [], areas, 0,
                      np.random.default_rng(1), targets=tg)
        assert not tg["swing"].any(), kind


@needs_render
def test_negatives_are_background(renderer):
    """The v3 negative set draws symbols and furniture without touching the labels or the geometry targets; its
    figures and tags go into the text target."""
    g, areas, _ = floor_px()
    plain, neg = {}, {}
    img0, lab0 = renderer.draw(renderer.style(np.random.default_rng(0), **QUIET), 20.0, g, [], [], areas, 0,
                               np.random.default_rng(0), targets=plain)
    img1, lab1 = renderer.draw(renderer.style(np.random.default_rng(0), **QUIET, **NEGATIVES), 20.0, g, [], [], areas, 0,
                               np.random.default_rng(0), targets=neg)
    assert np.array_equal(lab0, lab1)
    for k in ("interior", "boundary", "void", "swing"):
        assert np.array_equal(plain[k], neg[k]), k
    assert (img1 != img0).any(axis=2).sum() > 300 and not plain["text"].any() and neg["text"].sum() > 20
    assert {"dim_chains", "furniture", "level_marks", "section_marks"} <= renderer._elements


@needs_render
def test_negatives_off_by_default(renderer):
    """Without negatives=True the style table of v2 applies: no v3 element is drawn, so a seed gives the v2 sample."""
    st = renderer.style(np.random.default_rng(0))
    assert not any(st[k] for k in NEGATIVES)


@needs_render
@needs_ifc
def test_pick_source_share():
    """The IFC share equals its weight also when column floors are wanted: v2 re-drew the source per attempt, and
    IFC storeys mostly have columns (18 % for a weight of 10 %, M8 of the review)."""
    from synth import Renderer
    r = Renderer([(VAL, 0.9), (IFC, 0.1)], size=256, column_floors=1.0)
    rng = np.random.default_rng(0)
    picks = [r._pick(rng)[0] for _ in range(300)]
    ifc = sum("licence" in f for f in picks) / len(picks)
    assert 0.04 <= ifc <= 0.17, ifc
    assert sum(not f["columns"].is_empty for f in picks) > 0.8 * len(picks)   # and the column floors were found


def test_floor_record_normalisation():
    """normalise_record (shared by sd_prepare and ifc_prepare) rotates every geometry alike to the dominant wall
    direction, shifts the walls to the origin and keeps the per-wall parts and the other keys."""
    from sd_prepare import FloorRecord, normalise_record
    parts = [shapely.box(0, 0, 10, 0.3), shapely.box(0, 0, 0.3, 6), shapely.box(4, 0, 4.2, 6)]
    rot = lambda g: shapely.affinity.translate(shapely.affinity.rotate(g, 30, origin=(0, 0)), 5, -7)
    rec = {"walls": rot(shapely.union_all(parts)), "walls_parts": [rot(p) for p in parts], "doors": rot(shapely.box(4, 2, 4.2, 3)),
           "windows": shapely.Polygon(), "columns": shapely.Polygon(), "railings": shapely.Polygon(),
           "stairs": [rot(shapely.box(6, 1, 7, 4))], "features": [("SINK", rot(shapely.box(1, 1, 1.5, 1.5)))],
           "areas": [("ROOM", rot(shapely.box(0.3, 0.3, 4, 6)))], "floor_id": "f1", "site": "s1", "public": 0.0}
    out = normalise_record(rec)
    assert set(FloorRecord.__annotations__) - {"licence"} <= set(out)
    assert out["rotation"] == pytest.approx(30, abs=0.5)
    x0, y0, _, _ = out["walls"].bounds
    assert abs(x0) < 1e-6 and abs(y0) < 1e-6
    assert shapely.union_all(out["walls_parts"]).symmetric_difference(out["walls"]).area < 1e-6
    assert len(out["walls_parts"]) == 3 and out["floor_id"] == "f1" and out["public"] == 0.0
    walls0 = shapely.union_all(parts)
    for a, b in ((out["doors"], shapely.box(4, 2, 4.2, 3)), (out["stairs"][0], shapely.box(6, 1, 7, 4)),
                 (out["features"][0][1], shapely.box(1, 1, 1.5, 1.5)), (out["areas"][0][1], shapely.box(0.3, 0.3, 4, 6))):
        assert a.area == pytest.approx(b.area)                                  # moved rigidly with the walls ...
        assert a.centroid.distance(out["walls"].centroid) == pytest.approx(b.centroid.distance(walls0.centroid), abs=1e-6)
        assert a.intersection(b).area > 0.85 * b.area                           # ... back to the drawn frame (0.5-degree bins)
    assert out["features"][0][0] == "SINK" and out["areas"][0][0] == "ROOM"


def test_floor_record_from_rows():
    """sd_prepare.floor_record keeps the walls one by one next to their union."""
    import pandas as pd
    from sd_prepare import floor_record
    rows = pd.DataFrame({"entity_type": ["separator", "separator", "opening", "area"],
                         "entity_subtype": ["WALL", "WALL", "DOOR", "ROOM"],
                         "geometry": [shapely.box(0, 0, 8, 0.3).wkt, shapely.box(0, 0, 0.3, 5).wkt,
                                      shapely.box(3, 0, 3.9, 0.3).wkt, shapely.box(0.3, 0.3, 8, 5).wkt]})
    rec = floor_record(rows)
    assert len(rec["walls_parts"]) == 2 and rec["walls"].area == pytest.approx(8 * 0.3 + 0.3 * 5 - 0.09)
    assert rec["doors"].area == pytest.approx(0.27) and rec["areas"][0][0] == "ROOM" and rec["rotation"] == pytest.approx(0, abs=0.5)
    assert floor_record(rows[rows.entity_subtype != "WALL"]) is None


@needs_render
def test_v1_call_signatures(renderer):
    """The evaluation harness calls style(), draw() and degrade() directly with the v1 arguments."""
    g, areas, _ = floor_px()
    rng = np.random.default_rng(3)
    st = renderer.style(rng)
    assert {"era", "walls", "doors", "windows", "ink", "pen", "thin_pen", "font"} <= set(st)
    out = renderer.draw(st, 20.0, g, [], [], areas, 0, rng)
    assert len(out) == 2 and out[0].shape == (256, 256, 3) and out[1].shape == (256, 256)
    out = renderer.degrade(*out, st, rng)
    assert len(out) == 2 and out[0].shape == (256, 256, 3) and out[1].dtype == np.uint8
    assert len(renderer.sample(rng)) == 3


@needs_ifc
def test_ifc_unknown_targets():
    """IFC storeys model no voids: the void target is unknown (255) wherever no panel covers it."""
    from synth import Renderer
    r = Renderer(IFC, size=192)
    for seed in range(3):
        img, lab, tags, tg = r.sample(np.random.default_rng(seed), targets=True, style={"multi": False})
        assert tags["source"] == "ifc" and set(np.unique(tg["void"])) <= {0, 255} and (tg["void"] == 255).any()


# ---------- segmenter ----------

def segment_v1(img, model, cfg=DEFAULT):
    """fpx.segment.segment before model v2 (7 October 2026), kept verbatim as the reference for v1 checkpoints."""
    tile, overlap = cfg.seg_tile, cfg.seg_overlap
    H, W = img.shape[:2]
    acc = np.zeros((len(CLASSES), H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    ramp = np.minimum(np.arange(tile) + 1, overlap) / overlap
    win = np.minimum.outer(np.minimum(ramp, ramp[::-1]), np.minimum(ramp, ramp[::-1])).astype(np.float32)
    step = tile - overlap
    with torch.no_grad():
        for y in range(0, max(H - overlap, 1), step):
            for x in range(0, max(W - overlap, 1), step):
                t = img[y:y + tile, x:x + tile]
                h, w = t.shape[:2]
                ph, pw = (-h) % 32, (-w) % 32
                t = np.pad(t, ((0, ph), (0, pw), (0, 0)), constant_values=255)
                x_ = normalise(torch.from_numpy(t).permute(2, 0, 1)[None])
                p = 0
                flips = ([], [3], [2], [2, 3]) if cfg.seg_tta else ([],)
                for dims in flips:
                    xf = torch.flip(x_, dims) if dims else x_
                    pf = torch.softmax(model(xf), 1)
                    p = p + (torch.flip(pf, dims) if dims else pf)
                p = (p / len(flips))[0, :, :h, :w].numpy()
                acc[:, y:y + h, x:x + w] += p * win[:h, :w]
                wsum[y:y + h, x:x + w] += win[:h, :w]
    prob = acc / np.maximum(wsum, 1e-6)
    return prob, prob.argmax(0).astype(np.uint8)


def plan_image():
    """A small black-on-white plan: two rooms, a door gap, a window, a stamp-like block of text strokes."""
    img = np.full((224, 320, 3), 255, np.uint8)
    img[20:35, 20:300] = img[190:205, 20:300] = 0
    img[20:205, 20:35] = img[20:205, 285:300] = 0
    img[35:110, 160:172] = img[150:190, 160:172] = 0
    img[20:35, 80:130] = 255
    img[26:29, 80:130] = 0
    for k in range(5):
        img[90:96, 60 + 8 * k:64 + 8 * k] = 0
    return img


@needs_v1_model
def test_v1_checkpoint_unchanged():
    model = load_model(V1)
    assert model.fpx_version == 1 and model.fpx_heads == []
    ref = smp.Unet("resnet34", encoder_weights=None, classes=len(CLASSES))   # the v1 loader
    ref.load_state_dict(torch.load(V1, map_location="cpu"))
    ref.eval()
    img = plan_image()
    sheet = Sheet("t", "t", "test", "synthetic", img, (0.0, 0.0), {})
    segment(sheet, model)
    prob, label = segment_v1(img, ref)
    assert np.array_equal(sheet.label, label) and np.array_equal(sheet.prob, prob)
    assert sheet.interior_prob is None and sheet.text_prob is None
    assert (label == WALL).mean() > 0.05                 # the model sees the walls


def test_v2_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = build_model(pretrained=False, heads=HEADS).eval()
    torch.save(checkpoint(model, HEADS, iter=7, args={"seed": 0}, torch=torch.__version__), tmp_path / "m.pt")
    loaded = load_model(tmp_path / "m.pt")
    assert loaded.fpx_version == 2 and loaded.fpx_heads == HEADS and loaded.fpx_meta["iter"] == 7
    img = plan_image()[:96, :128]
    sheet = Sheet("t", "t", "test", "synthetic", img, (0.0, 0.0), {})
    segment(sheet, loaded, Config(seg_tta=False))
    assert sheet.prob.shape == (len(CLASSES), 96, 128) and np.allclose(sheet.prob.sum(0), 1, atol=1e-4)
    assert sheet.label.shape == (96, 128) and sheet.label.dtype == np.uint8
    for h in HEADS:
        p = getattr(sheet, f"{h}_prob")
        assert p.shape == (96, 128) and 0 <= p.min() and p.max() <= 1
    with torch.no_grad():                                  # same numbers as the model that was saved
        x = normalise(torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None])
        assert torch.allclose(model(x), loaded(x))


# ---------- training ----------

def test_soft_dice_per_sample():
    """The head Dice is scored per sample over the samples that hold the target: a crop without it neither dilutes
    nor fires the term (v2 pooled the batch, so the void Dice fired only when some crop held a void)."""
    from train import soft_dice
    t = torch.zeros(3, 1, 8, 8)
    t[0, 0, 2:6, 2:6] = 1
    p = torch.full((3, 1, 8, 8), 0.1)
    p[0, 0, 2:6, 2:6] = 0.9                               # sample 0 fits; samples 1 and 2 have no target
    valid = torch.ones(3, 1, 8, 8, dtype=torch.bool)
    d = soft_dice(p, t, valid)
    alone = soft_dice(p[:1], t[:1], valid[:1])
    assert d is not None and torch.isclose(d, alone)
    assert soft_dice(p[1:], t[1:], valid[1:]) is None
    valid[0, 0, :, :4] = False                           # unknown pixels neither count as hits nor as misses
    assert soft_dice(p, t, valid) < d + 1e-6


@needs_render
def test_val_cache_roundtrip(tmp_path):
    """--val-cache writes the validation renders once, reads them back for the same renderer and refuses others."""
    import train
    cache = tmp_path / "val.pt"
    args = train.parser().parse_args(["--val", "2", "--size", "128", "--val-cache", str(cache), "--ifc-weight", "0"])
    X, Y, T, tags = train.val_set(args, 2, say=lambda *a: None)
    assert cache.exists() and X.shape == (2, 3, 128, 128) and T.shape == (2, len(HEADS), 128, 128)
    X2, Y2, T2, tags2 = train.val_set(args, 2, say=lambda *a: None)
    assert torch.equal(X, X2) and torch.equal(Y, Y2) and torch.equal(T, T2) and tags == tags2
    other = train.parser().parse_args(["--val", "2", "--size", "128", "--val-cache", str(cache), "--ifc-weight", "0", "--negatives"])
    with pytest.raises(SystemExit, match="other settings"):
        train.val_set(other, 2, say=lambda *a: None)
    assert "synth_sha256" in train.val_key(args, 2)
