"""Rendered Swiss Dwellings floors: each test floor drawn whole by synth.Renderer (style, draw, degrade) at exactly
px_per_m, unrotated, with a style seeded by (seed, floor index), in the oracle's frame, so that the floor's own
geometry is the reference. The renders are stored in data/harness/renders/seed<S>/ on first use and reused (a frozen
benchmark while synth.py changes; --rerender draws them again, the synth.py hash is recorded per sheet). Scan
defects that move the drawing by more than MAX_SHIFT_PX (perspective) are drawn again.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import shapely
from scipy import ndimage

from common import DATA, ROOT
from fpeval import metrics, oracle
from fpeval.cli import Skip, jsonable
from fpeval.datasets import file_sha1
from fpx.model import WALL, Sheet

RENDERS = DATA / "harness/renders"
MAX_SHIFT_PX = 3.5                              # scan defects may move the drawing this far (wobble, fold), not more: the label map stays with the floor's polygons


def displacement(a, b):
    """How far two masks are apart: 99th percentile of the distance (px) from a pixel of one to the other, both ways."""
    if not a.any() or not b.any():
        return 0.0 if a.any() == b.any() else float("inf")
    box = metrics._crop(a, b, margin=8)
    a, b = a[box], b[box]
    da = ndimage.distance_transform_edt(~b)[a]
    db = ndimage.distance_transform_edt(~a)[b]
    return float(max(np.percentile(da, 99), np.percentile(db, 99)))


def clear_render_caches():
    """The renderer caches textures and pixel grids per canvas size; whole floors vary in size, so drop them."""
    import synth
    for v in list(vars(synth).values()):
        if callable(getattr(v, "cache_clear", None)):
            v.cache_clear()


def render_floor(renderer, rec, tf, shape, rng, px_per_m, max_shift=None, attempts=5):
    """Draw a whole floor with the renderer's own steps (style, draw, degrade) at exactly px_per_m, unrotated, in the
    frame tf. The renderer draws on a square canvas of renderer.size pixels; the floor sits at its top left corner.
    Scan defects that move the drawing (wobble, fold, perspective) also move the label map but not the floor's
    polygons: a degradation that moves the walls by more than max_shift px is drawn again (the style is kept).
    Returns (img, label, style, info)."""
    max_shift = MAX_SHIFT_PX if max_shift is None else max_shift
    H, W = shape
    renderer.size = max(H, W)
    st = renderer.style(rng)
    st["theta"] = 0.0                                   # unrotated (the renderer orients hatches and context by theta)
    g = {k: tf(rec[k]) for k in ("walls", "doors", "windows", "columns", "railings")}
    stairs = [tf(p) for p in rec["stairs"]]
    feats = [(k, tf(p)) for k, p in rec["features"]]
    areas = [(k, tf(p), p.area) for k, p in rec["areas"]]
    try:
        try:
            img0, lab0 = renderer.draw(st, px_per_m, g, stairs, feats, areas, rec.get("public", 0), rng)
        except shapely.errors.GEOSException:            # invalid floor geometry (self-touching rings): repair, draw again
            fix = lambda x: shapely.make_valid(x).buffer(0)
            g = {k: fix(v) for k, v in g.items()}
            stairs, feats = [fix(p) for p in stairs], [(k, fix(p)) for k, p in feats]
            areas = [(k, fix(p), a) for k, p, a in areas]
            img0, lab0 = renderer.draw(st, px_per_m, g, stairs, feats, areas, rec.get("public", 0), rng)
        drawn = set(getattr(renderer, "_elements", ()) or ())     # elements the renderer drew (renderer v2), for the report
        walls0 = lab0 == WALL
        for attempt in range(1, attempts + 1):
            if hasattr(renderer, "_elements"):
                renderer._elements = set(drawn)
            img, lab = renderer.degrade(img0.copy(), lab0.copy(), st, rng)
            shift = displacement(lab == WALL, walls0)
            if shift <= max_shift:
                break
        elements = sorted(getattr(renderer, "_elements", ()) or ())
    finally:
        clear_render_caches()
    img = np.ascontiguousarray(img[:H, :W])
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img, np.ascontiguousarray(lab[:H, :W]), st, {"distortion_px": shift, "attempts": attempt, "elements": elements}


def stored_render(rec, index, ctx, tf, shape):
    """The rendered floor from data/harness/renders/seed<S>/ (frozen benchmark), rendered and stored on first use or
    with --rerender. Renders are kept because the renderer changes over time; the synth.py hash is recorded."""
    stem = RENDERS / f"seed{ctx.args.seed}" / f"sd{rec['floor_id']}"
    meta_path = stem.with_name(stem.name + ".json")
    if meta_path.exists() and not ctx.args.rerender:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        img = cv2.imread(str(stem) + ".png", cv2.IMREAD_COLOR)
        lab = cv2.imread(str(stem) + "_label.png", cv2.IMREAD_UNCHANGED)
        if img is not None and lab is not None and lab.shape == tuple(shape):
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), lab, meta, "stored"
    if ctx.renderer is None:
        from synth import Renderer
        ctx.renderer = Renderer(DATA / "floors-test.pkl")
        ctx.synth_sha = file_sha1(ROOT / "synth.py", 12)
    rng = np.random.default_rng([ctx.args.seed, index])
    img, lab, st, info = render_floor(ctx.renderer, rec, tf, shape, rng, ctx.cfg.px_per_m)
    meta = {"floor_id": str(rec["floor_id"]), "seed": ctx.args.seed, "index": index, "synth_sha1": ctx.synth_sha,
            "px_per_m": ctx.cfg.px_per_m, "era": str(st["era"]), "wall_style": str(st["walls"]), "door_style": str(st["doors"]),
            "window_style": str(st["windows"]), "text": bool(st.get("text")), **info}
    stem.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(stem) + ".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(stem) + "_label.png", lab)
    meta_path.write_text(json.dumps(jsonable(meta), indent=1), encoding="utf-8")
    return img, lab, meta, "new"


def build(rec, index, ctx):
    cfg, m = ctx.cfg, ctx.cfg.px_per_m
    tf, (H, W), origin = oracle.frame(rec, cfg)
    if max(H, W) > ctx.args.max_side:
        raise Skip(f"{W}x{H} px is larger than --max-side {ctx.args.max_side}")
    t = time.time()
    img, lab, meta, how = stored_render(rec, index, ctx, tf, (H, W))
    ref = oracle.reference(rec, tf, ctx.args.ref_min_room)
    ref["label"] = lab
    sid = f"sd{rec['floor_id']}"
    sheet = Sheet(sid, f"Swiss Dwellings floor {rec['floor_id']} (rendered)", "Swiss Dwellings v3.0.0", "rendered raster",
                  img, origin, {"value": f"{m} px/m", "method": "rendered at a known scale"}, px_per_m=m)
    if ctx.take_ocr():
        f = cfg.ocr_px_per_m / m
        sheet.ocr_img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    info = {"size": [W, H], **{k: meta.get(k) for k in ("era", "wall_style", "door_style", "window_style", "text",
                                                       "distortion_px", "attempts", "synth_sha1")},
            "elements": ",".join(meta.get("elements") or []), "render": how, "render_s": round(time.time() - t, 1)}
    return sheet, ref, info, [rec["floor_id"], ctx.args.seed, index, meta.get("synth_sha1")]


def items(ctx):
    for i, rec in oracle.test_floors(ctx.args.n):
        yield str(rec["floor_id"]), lambda rec=rec, i=i: build(rec, i, ctx)
