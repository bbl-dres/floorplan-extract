"""Evaluation harness on public data (review step 1c): scores the whole pipeline (segmenter and post-processing) and
the post-processing alone, so that thresholds are swept and frozen here instead of on BBL sheets.

    python harness.py oracle [--n N]                    # Swiss Dwellings test floors with perfect labels: post-processing alone
    python harness.py render [--n N] [--seed S] [--ocr [K]]   # the same floors rendered whole in random styles: full pipeline
    python harness.py cvcfp [--n N]                     # all CVC-FP scans, scale from the reference doors
    python harness.py waffle [--n N] [--scale proposed|reference]   # WAFFLE benchmark images (masks only)

Options: --config cfg.json (fpx.Config.from_file, for sweeps), --out file, --ids a,b (only these sheets), --model path,
--threads 8, --no-cache. Writes data/harness/<mode>.json (arguments, configuration, summary, groups, one row per
sheet) and prints a summary table.

Modes:
- oracle: data/floors-test.pkl rasterised into perfect labels (oracle.py); stages walls..qa. Floors are taken in a fixed
  shuffled order (ORDER_SEED), the same in render mode, so --n N gives the same N floors spread over sites in both.
- render: each test floor drawn whole by synth.Renderer (style, draw, degrade) at exactly px_per_m, unrotated, with a
  style seeded by (--seed, floor index); then triage, preprocessing, segmentation and stages walls..qa. The renders
  are stored in data/harness/renders/seed<S>/ on first use and reused (a frozen benchmark while synth.py changes;
  --rerender draws them again, the synth.py hash is recorded per sheet). Scan defects that move the drawing by more
  than MAX_SHIFT_PX (perspective) are drawn again. OCR is off (about 30 s per sheet); --ocr K turns the text stage on
  for the first K sheets (OCR on the render upscaled to the OCR resolution). Floors wider than --max-side pixels are
  skipped (memory). Reference: the floor's areas (outdoor and void excluded, as oracle.NOT_ROOMS), its doors,
  windows and door connections, and the label map of the renderer.
- cvcfp: data/benchmark/cvc-fp (CC BY-NC, evaluation only). Scale from the reference door symbols (0.85 m leaves,
  as bench.py). Door labels cover the swing, so door centres are taken on the side of the swing that lies in the
  wall opening, and door pixels are not scored. Rooms next to a "Separation" line (open plan) are typed
  "room (separation)"; "Parking" areas are not scored.
- waffle: data/benchmark/waffle/data/benchmark (per-image Commons licences; masks only: walls, doors, windows,
  interior). Scale unknown: proposed from detected door widths (bench.propose_scale, cached per image), or with
  --scale reference from the reference door masks (0.9 m, at least 3 doors) where possible. GF reference: interior
  plus walls and openings.

Metrics (metrics.py): rooms one-to-one at IoU >= 0.5 (rooms and flagged fragments are predictions), walls from the
cleaned wall mask (pixel IoU, boundary F at 2 px, centre lines within 5 cm), class pixel IoU of the segmenter,
openings by centre within 0.5 m, connectivity mapped through the room matching, GF gross (with deducted voids).
Segmenter outputs are cached in data/harness/cache, keyed by model, input, renderer and pre-segmentation settings,
so that a threshold sweep reruns only the post-processing.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    # the renderer picks room colours with hash(): fix the hash seed (before the heavy imports) for repeatable renders
    sys.exit(subprocess.call([sys.executable, *sys.argv], env={**os.environ, "PYTHONHASHSEED": "0"}))

import cv2
import numpy as np
import shapely
from PIL import Image
from scipy import ndimage
from shapely import affinity

import metrics
import oracle
from common import DATA, REPO
from fpx import DEFAULT, Config, Sheet, pipeline
from fpx.geometry import mask_polys
from fpx.model import CLASSES, DOOR, WALL, WINDOW

Image.MAX_IMAGE_PIXELS = None
OUT = DATA / "harness"
BENCH = REPO / "data/benchmark"
PRE = ("triage", "preprocess")
POST = ("walls", "openings", "stairs", "rooms", "attributes", "derived", "qa")
KIND = {"door": "door", "exterior door": "door", "window": "window", "interior opening": "window"}
CVC_DOOR_M = 0.85                               # CVC-FP door symbols: leaf length (bench.py)
WAFFLE_DOOR_M = 0.9                             # typical clear door width (bench.DOOR_M)
OPENING_TOL_M = 0.5                             # opening centres match within this distance
CENTRE_TOL_M = 0.05                             # wall centre lines match within this distance
BOUNDARY_TOL_PX = 2                             # wall boundaries match within this many pixels
MAX_SHIFT_PX = 3.5                              # render: scan defects may move the drawing this far (wobble, fold), not more
PRE_SEG_FIELDS = ("px_per_m", "seg_", "skew_", "deskew_", "solid_ink_", "flat_scan_")   # settings that change the segmenter input or output
ORDER_SEED = 0                                  # fixed order of the Swiss Dwellings test floors (--n takes the first n)
SAVE_EVERY = 10                                 # write partial results every this many sheets


class Skip(Exception):
    """A sheet that is not evaluated, with the reason (recorded, not counted as a failure)."""


def sha1(path, n=None):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def resized(img, f):
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)


def rgb(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("RGB"))


def fill(lab, polys_, value):
    for p in polys_:
        oracle._fill(lab, p, value)


# ---------- scoring ----------

def score(sheet, ref, cfg, seg_label=None):
    """One row of metrics for a sheet whose stages have run. ref: dict as oracle.reference(), plus 'label' (class map)
    and 'unscored' (class names without a reference)."""
    m = cfg.px_per_m
    m2 = 1 / m ** 2
    row = {}
    if ref.get("rooms") is not None:
        regions = sheet.rooms + sheet.fragments
        pred = [r["poly"] for r in regions]
        refp = [p for _, p in ref["rooms"]]
        types = [t for t, _ in ref["rooms"]]
        mr = metrics.match_rooms(pred, refp, 0.5, m2, ignore=ref.get("ignore", []))
        hit = {j for _, j in mr["pairs"]}
        miss = metrics.miss_reasons(pred, refp, hit)
        b = sheet.building_rough
        for j, (reason, _, best) in list(miss.items()):    # a missing room outside the building mask: another drawing or wing
            if reason == "missing" and b is not None:
                x, y = refp[j].representative_point().coords[0]
                if not b[int(np.clip(y, 0, b.shape[0] - 1)), int(np.clip(x, 0, b.shape[1] - 1))]:
                    miss[j] = ("outside building", [], best)
        row["rooms"] = {k: mr[k] for k in ("ref", "pred", "matched", "ignored", "recall", "precision", "mean_iou")}
        row["rooms"].update(area_err_pct=mr["area_err_pct"], area_err_m2=mr["area_err_m2"], pair_iou=mr["pair_iou"],
                            area_err_pct_median=metrics.med(mr["area_err_pct"]),
                            fragments=len(sheet.fragments),
                            ref_rooms=[[types[j], j in hit, miss.get(j, (None,))[0], [types[k] for k in miss.get(j, (None, []))[1]]]
                                       for j in range(len(refp))],
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
        if sheet.text:                                  # OCR ran: names and stamp areas of the matched rooms
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
        row["walls"] = metrics.mask_scores(sheet.wall_mask, rl == WALL, BOUNDARY_TOL_PX, CENTRE_TOL_M * m)
        row["wall_rates"] = metrics.mask_rates(row["walls"])
        row["classes"] = metrics.confusion(rl, seg_label if seg_label is not None else sheet.label, len(CLASSES)).tolist()
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


# ---------- running ----------

class Context:
    def __init__(self, args):
        self.args = args
        self.cfg = Config.from_file(args.config) if args.config else DEFAULT
        self.mode = args.mode
        self.model = self.engine = self.renderer = None
        self.model_sha = None
        self.cache = None if args.no_cache or args.mode == "oracle" else OUT / "cache" / args.mode
        if args.mode != "oracle":
            import torch
            from fpx.segment import load_model
            torch.set_num_threads(args.threads)
            path = Path(args.model)
            self.model = load_model(path)
            self.model_sha = sha1(path, 12)
        if args.ocr:
            self.engine = pipeline.ocr_engine()
        pre = {f: v for f, v in self.cfg.to_dict().items() if f.startswith(PRE_SEG_FIELDS)}
        self.pre_key = json.dumps(pre, sort_keys=True)

    def cache_path(self, sid, key):
        h = hashlib.sha1(json.dumps([self.model_sha, self.pre_key, key], sort_keys=True).encode()).hexdigest()[:12]
        return self.cache / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', sid)}-{h}.npz"


def run_sheet(sheet, ref, ctx, key):
    """Stages up to segmentation (segmenter output from the cache when present), then walls..qa. Returns the
    reference (moved if the sheet was deskewed), the segmenter label and the seconds per step."""
    cfg, times = ctx.cfg, {}
    if ctx.mode == "oracle":
        seg = sheet.label.copy()
        times.update(pipeline.run(sheet, cfg=cfg, stages=POST))
        return ref, seg, times
    stages = PRE + (("text",) if sheet.ocr_img is not None and ctx.engine is not None else ())
    times.update(pipeline.run(sheet, engine=ctx.engine, cfg=cfg, stages=stages))
    if "deskew" in sheet.meta:
        ref = deskew_reference(ref, sheet.meta["deskew"]["matrix"], sheet.img.shape[:2])
    path = ctx.cache_path(sheet.id, key) if ctx.cache is not None else None
    if path is not None and path.exists():
        z = np.load(path)
        sheet.label, sheet.prob = z["label"], z["prob"].astype(np.float32) / 255
        times["segment"] = 0.0
        times["cached"] = True
    else:
        times.update(pipeline.run(sheet, model=ctx.model, cfg=cfg, stages=("segment",)))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, label=sheet.label, prob=np.round(sheet.prob * 255).astype(np.uint8))
    seg = sheet.label.copy()                            # stage 4 edits the label (isolated opening blobs)
    times.update(pipeline.run(sheet, cfg=cfg, stages=POST))
    return ref, seg, times


# ---------- modes ----------

def test_floors(n=None):
    """Swiss Dwellings test floors as (index in the pickle, record), in a fixed shuffled order (seed ORDER_SEED), so
    that the first n are spread over sites and oracle and render modes take the same floors."""
    import pickle
    import zlib
    blobs = pickle.loads((DATA / "floors-test.pkl").read_bytes())
    for i in np.random.default_rng(ORDER_SEED).permutation(len(blobs))[:n]:
        yield int(i), pickle.loads(zlib.decompress(blobs[i]))


def oracle_items(ctx):
    for i, rec in test_floors(ctx.args.n):
        yield str(rec["floor_id"]), lambda rec=rec: build_oracle(rec, ctx)


def build_oracle(rec, ctx):
    cfg = ctx.cfg
    tf, shape, origin = oracle.frame(rec, cfg)
    sheet, _ = oracle.oracle_sheet(rec, cfg)
    ref = oracle.reference(rec, tf, ctx.args.ref_min_room)
    ref["label"] = sheet.label.copy()
    return sheet, ref, {"size": [shape[1], shape[0]]}, None


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


def render_items(ctx):
    for i, rec in test_floors(ctx.args.n):
        yield str(rec["floor_id"]), lambda rec=rec, i=i: build_render(rec, i, ctx)


def stored_render(rec, index, ctx, tf, shape):
    """The rendered floor from data/harness/renders/seed<S>/ (frozen benchmark), rendered and stored on first use or
    with --rerender. Renders are kept because the renderer changes over time; the synth.py hash is recorded."""
    stem = OUT / "renders" / f"seed{ctx.args.seed}" / f"sd{rec['floor_id']}"
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
        ctx.synth_sha = sha1(Path(__file__).resolve().parent / "synth.py", 12)
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


def build_render(rec, index, ctx):
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
    if ctx.engine is not None and ctx.ocr_left > 0:
        ctx.ocr_left -= 1
        f = cfg.ocr_px_per_m / m
        sheet.ocr_img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    info = {"size": [W, H], **{k: meta.get(k) for k in ("era", "wall_style", "door_style", "window_style", "text",
                                                       "distortion_px", "attempts", "synth_sha1")},
            "elements": ",".join(meta.get("elements") or []), "render": how, "render_s": round(time.time() - t, 1)}
    return sheet, ref, info, [rec["floor_id"], ctx.args.seed, index, meta.get("synth_sha1")]


def cvc_group(n):
    if n.isdigit():
        return "numbered"
    if n.startswith("image"):
        return "image"
    if n.startswith("II"):
        return "II"
    if n.startswith("I"):
        return "I (sommaire)"
    return "p"


def cvc_parse(svg):
    """CVC-FP ground truth: polygons per class (native pixels), as bench.load_cvcfp reads them."""
    text = svg.read_text(encoding="utf-8", errors="replace")
    out = {}
    for cls, pts in re.findall(r'<polygon class="([^"]+)"[^>]*points="([^"]+)"', text):
        xy = np.array([float(v) for v in re.split(r"[ ,]+", pts.strip()) if v]).reshape(-1, 2)
        if len(xy) >= 3:
            p = shapely.Polygon(xy).buffer(0)
            if p.area > 0:
                out.setdefault(cls, []).extend(oracle.polys(p))
    return out


def door_in_wall(door, walls, tol):
    """CVC-FP door symbols cover the swing: the door itself is the side of the swing's rectangle that spans the wall
    opening (both ends on a wall, its middle off the wall). Falls back to the centroid."""
    c = np.asarray(door.minimum_rotated_rectangle.exterior.coords)[:4]
    best = None
    for k in range(4):
        a, b = c[k], c[(k + 1) % 4]
        ends = max(walls.distance(shapely.Point(a)), walls.distance(shapely.Point(b)))
        mid = walls.distance(shapely.Point((a + b) / 2))
        if ends <= tol and (best is None or mid > best[0]):
            best = (mid, (a + b) / 2)
    return best[1] if best is not None else np.asarray(door.centroid.coords[0])


def cvcfp_items(ctx):
    folder = BENCH / "cvc-fp/ImagesGT"
    ids = sorted({p.name.split("_gt_")[0] for p in folder.glob("*_gt_*.svg")})
    for n in ids[:ctx.args.n]:
        yield n, lambda n=n: build_cvcfp(n, folder, ctx)


def build_cvcfp(n, folder, ctx):
    cfg, M = ctx.cfg, ctx.cfg.px_per_m
    svg = next(folder.glob(f"{n}_gt_*.svg"))
    img_path = next(p for p in (folder / f"{n}.png", folder / f"{n}.jpg") if p.exists())
    gt = cvc_parse(svg)
    doors = gt.get("Door", [])
    if not doors:
        raise Skip("no reference doors: scale unknown")
    edges = [np.diff(np.asarray(d.minimum_rotated_rectangle.exterior.coords)[:3], axis=0) for d in doors]
    ppm = float(np.median([np.hypot(e[:, 0], e[:, 1]).max() for e in edges])) / CVC_DOOR_M
    f, pad = M / ppm, int(M)
    img = rgb(img_path)
    work = cv2.copyMakeBorder(resized(img, f), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    H, W = work.shape[:2]
    tf = lambda g: affinity.translate(affinity.scale(g, f, f, origin=(0, 0)), pad, pad)
    lab = np.zeros((H, W), np.uint8)
    fill(lab, [tf(p) for p in gt.get("Wall", [])], WALL)
    fill(lab, [tf(p) for p in gt.get("Window", [])], WINDOW)
    walls = shapely.union_all(gt.get("Wall", [])) if gt.get("Wall") else shapely.Polygon()
    sep = shapely.union_all(gt["Separation"]).buffer(0.1 * ppm) if gt.get("Separation") else None
    rooms = [("room (separation)" if sep is not None and r.intersects(sep) else "room", tf(r)) for r in gt.get("Room", [])]
    openings = [("door", tuple(np.asarray(door_in_wall(d, walls, 0.15 * ppm)) * f + pad)) for d in doors]
    openings += [("window", tuple(np.asarray(w.centroid.coords[0]) * f + pad)) for w in gt.get("Window", [])]
    ref = {"rooms": rooms, "ignore": [tf(p) for p in gt.get("Parking", [])], "openings": openings, "label": lab,
           "unscored": ["door"]}
    sheet = Sheet(f"cvcfp_{n}", f"CVC-FP {n}", img_path.name, "raster (public benchmark)", work, (0.0, 0.0),
                  {"value": f"{ppm:.1f} px/m", "method": f"from {len(doors)} reference door symbols ({CVC_DOOR_M} m)"}, px_per_m=M)
    if ctx.engine is not None and ctx.ocr_left > 0:
        ctx.ocr_left -= 1
        fo = cfg.ocr_px_per_m / ppm
        sheet.ocr_img = cv2.copyMakeBorder(cv2.cvtColor(resized(img, fo), cv2.COLOR_RGB2GRAY), *(int(pad * cfg.ocr_px_per_m / M),) * 4,
                                           cv2.BORDER_CONSTANT, value=255)
    info = {"group": cvc_group(n), "size": [W, H], "px_per_m_native": ppm, "ref_doors": len(doors)}
    return sheet, ref, info, [n, img_path.stat().st_size, round(ppm, 4)]


def waffle_items(ctx):
    folder = BENCH / "waffle/data/benchmark"
    names = sorted(p.stem for p in (folder / "pngs").glob("*.png"))
    for name in names[:ctx.args.n]:
        yield name, lambda name=name: build_waffle(name, folder, ctx)


def reference_door_scale(door_mask):
    """px per m from reference door masks (median long side of the door blobs = WAFFLE_DOOR_M); None under 3 doors."""
    n, cc, stats, _ = cv2.connectedComponentsWithStats(door_mask.astype(np.uint8), connectivity=8)
    widths = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 20:
            continue
        ys, xs = np.nonzero(cc == i)
        (_, _), (w, h), _ = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
        widths.append(max(w, h))
    return (float(np.median(widths)) / WAFFLE_DOOR_M, len(widths)) if len(widths) >= 3 else (None, len(widths))


def proposed_scale(name, img, ctx):
    """bench.propose_scale (door widths of the segmenter at candidate scales), cached per image and model."""
    path = OUT / "cache" / "waffle_scale.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    key = f"{name}|{ctx.model_sha}"
    if key not in cache:
        import bench
        res = bench.propose_scale(img, ctx.model)
        cache[key] = {"px_per_m": float(res[0]), "method": str(res[1])}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return cache[key]["px_per_m"], cache[key]["method"]


def build_waffle(name, folder, ctx):
    cfg, M = ctx.cfg, ctx.cfg.px_per_m
    img = rgb(folder / "pngs" / f"{name}.png")
    seg = np.asarray(Image.open(folder / "segmented_descrete_pngs" / f"{name}_seg_colors.png").convert("RGB"))
    col = lambda c: np.all(seg == c, axis=-1)
    masks = {"walls": col((255, 0, 0)), "doors": col((0, 0, 255)), "windows": col((0, 255, 255)), "interior": col((255, 255, 255))}
    ref_ppm, ref_doors = reference_door_scale(masks["doors"])
    if ctx.args.scale == "reference" and ref_ppm:
        ppm, method = ref_ppm, f"reference: {ref_doors} reference doors measure {WAFFLE_DOOR_M} m"
    else:
        ppm, method = proposed_scale(name, img, ctx)
    f, pad = M / ppm, int(M)
    work = cv2.copyMakeBorder(resized(img, f), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    H, W = work.shape[:2]
    if max(H, W) > ctx.args.max_side:
        raise Skip(f"{W}x{H} px at {ppm:.1f} px/m is larger than --max-side {ctx.args.max_side}")
    small = {k: np.pad(cv2.resize(v.astype(np.uint8), (W - 2 * pad, H - 2 * pad), interpolation=cv2.INTER_NEAREST) > 0, pad)
             for k, v in masks.items()}
    lab = np.zeros((H, W), np.uint8)
    lab[small["walls"]] = WALL
    lab[small["windows"]] = WINDOW
    lab[small["doors"]] = DOOR
    openings = []
    for key, kind in (("doors", "door"), ("windows", "window")):
        n, cc, stats, cents = cv2.connectedComponentsWithStats(small[key].astype(np.uint8), connectivity=8)
        openings += [(kind, tuple(cents[i])) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 4]
    building = small["interior"] | small["walls"] | small["doors"] | small["windows"]
    gf = shapely.union_all(mask_polys(building, cfg.px2(1.0)))
    ref = {"openings": openings, "label": lab, "gf": gf}
    sheet = Sheet(f"waffle_{name}", f"WAFFLE {name}", f"{name}.png", "raster (public benchmark)", work, (0.0, 0.0),
                  {"value": f"{ppm:.1f} px/m", "method": method}, px_per_m=M)
    if ctx.engine is not None and ctx.ocr_left > 0:
        ctx.ocr_left -= 1
        fo = cfg.ocr_px_per_m / ppm
        sheet.ocr_img = cv2.copyMakeBorder(cv2.cvtColor(resized(img, fo), cv2.COLOR_RGB2GRAY), *(int(pad * cfg.ocr_px_per_m / M),) * 4,
                                           cv2.BORDER_CONSTANT, value=255)
    info = {"size": [W, H], "px_per_m_native": ppm, "scale_method": method, "ref_doors": ref_doors,
            "px_per_m_reference": ref_ppm, "scale_ratio_vs_reference": (ppm / ref_ppm) if ref_ppm else None}
    return sheet, ref, info, [name, round(ppm, 4)]


ITEMS = {"oracle": oracle_items, "render": render_items, "cvcfp": cvcfp_items, "waffle": waffle_items}
GROUPS = {"render": ("era", "wall_style", "door_style", "window_style"), "cvcfp": ("group",), "waffle": ()}


# ---------- report ----------

def fmt(v, spec=".2f"):
    return "-" if v is None else format(v, spec)


def table(named):
    """One line per summary: rooms, walls, openings, connections, GF."""
    head = (f"{'':24s} {'sheets':>6s} | {'rooms R':>7s} {'P':>5s} {'mIoU':>5s} {'err%':>6s} {'|err%|':>6s} {'|m2|':>5s} | "
            f"{'wall IoU':>8s} {'BF':>5s} {'CL-P':>5s} {'CL-R':>5s} | {'door P':>6s} {'R':>5s} {'win P':>5s} {'R':>5s} {'d>w':>5s} {'w>d':>5s} | "
            f"{'conn F1':>7s} {'F1m':>5s} | {'GF IoU':>6s} {'err%':>6s}")
    lines = [head, "-" * len(head)]
    for name, s in named:
        ro, wa, op, co, gf = (s.get(k) or {} for k in ("rooms", "walls", "openings", "connectivity", "gf"))
        lines.append(f"{name[:24]:24s} {s['sheets']:6d} | {fmt(ro.get('recall')):>7s} {fmt(ro.get('precision')):>5s} {fmt(ro.get('mean_iou')):>5s} "
                     f"{fmt(ro.get('area_err_pct_median'), '+.1f'):>6s} {fmt(ro.get('area_err_pct_abs_median'), '.1f'):>6s} "
                     f"{fmt(ro.get('area_err_m2_abs_median')):>5s} | {fmt(wa.get('iou')):>8s} {fmt(wa.get('boundary_f')):>5s} "
                     f"{fmt(wa.get('centreline_p')):>5s} {fmt(wa.get('centreline_r')):>5s} | "
                     f"{fmt((op.get('door') or {}).get('precision')):>6s} {fmt((op.get('door') or {}).get('recall')):>5s} "
                     f"{fmt((op.get('window') or {}).get('precision')):>5s} {fmt((op.get('window') or {}).get('recall')):>5s} "
                     f"{fmt(op.get('door_as_window')):>5s} {fmt(op.get('window_as_door')):>5s} | "
                     f"{fmt(co.get('f1')):>7s} {fmt(co.get('f1_matched')):>5s} | {fmt(gf.get('iou_median')):>6s} "
                     f"{fmt(gf.get('area_err_pct_median'), '+.1f'):>6s}")
    return "\n".join(lines)


def type_table(types, top=14):
    lines = [f"{'area type':26s} {'ref':>5s} {'found':>5s} {'recall':>6s} | missed: {'merged':>6s} {'oversz':>6s} {'split':>5s} "
             f"{'missing':>7s} {'outside':>7s} | merged with"]
    for t, d in list(types.items())[:top]:
        lines.append(f"{t[:26]:26s} {d['ref']:5d} {d['matched']:5d} {fmt(d['recall']):>6s} |         {d['merged']:6d} {d['oversized']:6d} "
                     f"{d['split']:5d} {d['missing']:7d} {d['outside building']:7d} | " + ", ".join(f"{k} {v}" for k, v in d["merged_with"].items()))
    return "\n".join(lines)


def write(args, ctx, rows, skipped, runtime, partial=False):
    """Summarise and write data/harness/<mode>.json (also during the run, marked partial)."""
    ok = [r for r in rows if "error" not in r]
    groups = {g: {v: metrics.summarise([r for r in ok if str(r.get(g)) == v]) for v in sorted({str(r.get(g)) for r in ok})}
              for g in GROUPS.get(args.mode, ())}
    seconds = [r["seconds"]["total"] for r in ok]
    result = {"mode": args.mode, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "partial": partial,
              "args": {k: v for k, v in vars(args).items()},
              "config_overrides": {k: v for k, v in ctx.cfg.to_dict().items() if DEFAULT.to_dict().get(k) != v},
              "model_sha1": ctx.model_sha, "synth_sha1": sorted({str(r["synth_sha1"]) for r in ok if r.get("synth_sha1")}),
              "runtime_s": round(runtime, 1), "seconds_per_sheet_median": metrics.med(seconds),
              "seconds_segment_median": metrics.med([r["seconds"].get("segment") for r in ok if "segment" in r["seconds"]]),
              "skipped": skipped, "failed": [{"id": r["id"], "error": r["error"]} for r in rows if "error" in r],
              "summary": metrics.summarise(rows), "groups": groups,
              "by_area_type": metrics.by_type(ok) if args.mode in ("oracle", "render", "cvcfp") else {}, "rows": rows}
    out = Path(args.out) if args.out else OUT / f"{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(result), ensure_ascii=False), encoding="utf-8")
    return result, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=sorted(ITEMS))
    ap.add_argument("--n", type=int, default=None, help="first n sheets (default: all); Swiss Dwellings floors in a fixed shuffled order, the same for oracle and render")
    ap.add_argument("--ids", default=None, help="comma-separated sheet ids (floor ids, CVC-FP or WAFFLE names)")
    ap.add_argument("--config", default=None, help="JSON thresholds (fpx.Config.from_file)")
    ap.add_argument("--seed", type=int, default=0, help="render: style seed (with the floor index)")
    ap.add_argument("--ocr", type=int, nargs="?", const=5, default=0, help="run the text stage (OCR) on the first K sheets (default 5)")
    ap.add_argument("--scale", choices=("proposed", "reference"), default="proposed", help="waffle: scale source")
    ap.add_argument("--ref-min-room", type=float, default=oracle.REF_MIN_ROOM, help="oracle/render: smallest scored reference room (m²)")
    ap.add_argument("--max-side", type=int, default=4500, help="render/waffle: skip sheets with a longer side (px)")
    ap.add_argument("--model", default=str(DATA / "model/segmenter.pt"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true", help="do not read or write cached segmenter outputs")
    ap.add_argument("--rerender", action="store_true", help="render: draw the floors again instead of using the stored renders")
    ap.add_argument("--out", default=None, help="output JSON (default data/harness/<mode>.json)")
    args = ap.parse_args(argv)
    if args.mode == "oracle":
        args.ocr = 0
    if hasattr(sys.stdout, "reconfigure"):              # sheet names such as "Baptisterium_ortodoxnych_pódorys" in a cp1252 console or log
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ctx = Context(args)
    ctx.ocr_left = args.ocr
    ctx.synth_sha = None
    only = set(args.ids.split(",")) if args.ids else None
    rows, skipped, t0 = [], [], time.time()
    for sid, build in ITEMS[args.mode](ctx):
        if only and sid not in only and f"sd{sid}" not in only:
            continue
        t = time.time()
        try:
            sheet, ref, info, key = build()
            prep = time.time() - t
            ref, seg, times = run_sheet(sheet, ref, ctx, key)
            row = {"id": sheet.id, **info, **score(sheet, ref, ctx.cfg, seg)}
            row["seconds"] = {"prepare": round(prep, 1), **times, "total": round(time.time() - t, 1)}
        except Skip as e:
            skipped.append({"id": sid, "reason": str(e)})
            print(f"skip {sid}: {e}", flush=True)
            continue
        except Exception as e:                           # keep going; failures are listed in the output
            traceback.print_exc()
            row = {"id": sid, "error": f"{type(e).__name__}: {e}", "seconds": {"total": round(time.time() - t, 1)}}
        rows.append(row)
        if "error" in row:
            print(f"FAIL {sid}: {row['error']}", flush=True)
        else:
            ro, wa, op = row.get("rooms") or {}, row.get("wall_rates") or {}, row.get("openings") or {}
            print(f"{len(rows):4d} {str(row['id']):>34s} {row['size'][0]:5d}x{row['size'][1]:<5d} "
                  f"rooms {ro.get('matched', '-')}/{ro.get('ref', '-')} (pred {ro.get('pred', '-')}) "
                  f"wall IoU {fmt(wa.get('iou'))} BF {fmt(wa.get('boundary_f'))} | "
                  f"doors {(op.get('door') or {}).get('matched', '-')}/{(op.get('door') or {}).get('ref', '-')} "
                  f"windows {(op.get('window') or {}).get('matched', '-')}/{(op.get('window') or {}).get('ref', '-')} "
                  f"| GF IoU {fmt((row.get('gf') or {}).get('iou'))} | {row['seconds']['total']:.1f} s"
                  + (" (cached)" if row["seconds"].get("cached") else ""), flush=True)
        if len(rows) % SAVE_EVERY == 0:
            write(args, ctx, rows, skipped, time.time() - t0, partial=True)
    runtime = time.time() - t0
    result, out = write(args, ctx, rows, skipped, runtime)
    summary, groups, types = result["summary"], result["groups"], result["by_area_type"]
    seconds = [r["seconds"]["total"] for r in rows if "error" not in r]

    print(f"\n{args.mode}: {summary['sheets']} sheets, {summary['failed']} failed, {len(skipped)} skipped, "
          f"{runtime / 60:.1f} min ({metrics.med(seconds) or 0:.1f} s per sheet, median) -> {out}")
    print(table([("all", summary)] + [(f"{g}={v}", s) for g, d in groups.items() for v, s in d.items()]))
    if summary.get("pixel_iou"):
        print("segmenter pixel IoU: " + "  ".join(f"{k} {fmt(v)}" for k, v in summary["pixel_iou"].items()))
    co = summary.get("connectivity")
    if co:
        print(f"connections: open passages found {fmt(co.get('other_found'))}; predicted edges by opening kind: "
              + "; ".join(f"{k} " + ", ".join(f"{n} {v}" for n, v in d.items()) for k, d in co["counts"].get("by_opening", {}).items()))
    if summary.get("ocr"):
        print("OCR subset: " + "  ".join(f"{k} {fmt(v) if isinstance(v, float) else v}" for k, v in summary["ocr"].items()))
    if types:
        print("\n" + type_table(types))
    return result


if __name__ == "__main__":
    main()
