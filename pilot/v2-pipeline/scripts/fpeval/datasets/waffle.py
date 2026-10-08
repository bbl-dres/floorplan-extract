"""WAFFLE benchmark images (per-image Wikimedia Commons licences): masks of walls, doors, windows and interior.

Scale unknown: proposed from detected door widths (fpx.scale.door_search, cached per image and model), or from the
reference door masks (DOOR_M, at least MIN_DOORS doors) where possible. GF reference: interior plus walls and
openings.
"""
import json

import cv2
import numpy as np
from PIL import Image

from fpeval.cli import Skip
from fpeval.datasets import BENCH, CACHE, Native, reference_from_masks
from fpeval.raster import rgb, to_working
from fpx.config import DEFAULT
from fpx.model import Sheet

FOLDER = BENCH / "waffle/data/benchmark"
DOOR_M = DEFAULT.scale_door_width               # typical clear door width (0.9 m): what a reference door mask measures
MIN_DOORS = 3                                   # the reference-door scale needs at least this many doors: one odd symbol must not set it
MIN_DOOR_PX = 20                                # smaller door blobs in the masks are noise
COLOURS = {"walls": (255, 0, 0), "doors": (0, 0, 255), "windows": (0, 255, 255), "interior": (255, 255, 255)}


def ids():
    return sorted(p.stem for p in (FOLDER / "pngs").glob("*.png"))


def image_path(name):
    return FOLDER / "pngs" / f"{name}.png"


def native(name):
    img = rgb(FOLDER / "pngs" / f"{name}.png")
    seg = np.asarray(Image.open(FOLDER / "segmented_descrete_pngs" / f"{name}_seg_colors.png").convert("RGB"))
    masks = {k: np.all(seg == c, axis=-1) for k, c in COLOURS.items()}
    return Native(img, None, None, None, masks)


def reference_door_scale(door_mask):
    """px per m from reference door masks (median long side of the door blobs = DOOR_M); None under MIN_DOORS."""
    n, cc, stats, _ = cv2.connectedComponentsWithStats(door_mask.astype(np.uint8), connectivity=8)
    widths = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_DOOR_PX:
            continue
        ys, xs = np.nonzero(cc == i)
        (_, _), (w, h), _ = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
        widths.append(max(w, h))
    return (float(np.median(widths)) / DOOR_M, len(widths)) if len(widths) >= MIN_DOORS else (None, len(widths))


def proposed_scale(name, img, ctx):
    """Door widths of the segmenter at candidate scales (fpx.scale.door_search), cached per image and model."""
    path = CACHE / "waffle_scale.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    key = f"{name}|{ctx.model_sha}"
    if key not in cache:
        from fpx import scale
        res = scale.door_search(img, ctx.model, DEFAULT)
        cache[key] = {"px_per_m": float(res["px_per_m"]), "method": str(res["method"])}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return cache[key]["px_per_m"], cache[key]["method"]


def reference(nat, work, cfg):
    return reference_from_masks(nat.masks, work.img.shape[:2], work.pad, cfg)[0]


def build(name, ctx):
    nat = native(name)
    ref_ppm, ref_doors = reference_door_scale(nat.masks["doors"])
    if getattr(ctx.args, "scale", "proposed") == "reference" and ref_ppm:
        ppm, method = ref_ppm, f"reference: {ref_doors} reference doors measure {DOOR_M} m"
    else:
        ppm, method = proposed_scale(name, nat.img, ctx)
    work = to_working(nat.img, ppm, ctx.cfg, ocr=ctx.take_ocr())
    H, W = work.img.shape[:2]
    if max(H, W) > ctx.args.max_side:
        raise Skip(f"{W}x{H} px at {ppm:.1f} px/m is larger than --max-side {ctx.args.max_side}")
    sheet = Sheet(f"waffle_{name}", f"WAFFLE {name}", f"{name}.png", "raster (public benchmark)", work.img, (0.0, 0.0),
                  {"value": f"{ppm:.1f} px/m", "method": method}, ocr_img=work.ocr_img, px_per_m=ctx.cfg.px_per_m)
    info = {"size": [W, H], "px_per_m_native": ppm, "scale_method": method, "ref_doors": ref_doors,
            "px_per_m_reference": ref_ppm, "scale_ratio_vs_reference": (ppm / ref_ppm) if ref_ppm else None}
    return sheet, reference(nat, work, ctx.cfg), info, [name, round(ppm, 4)]


def items(ctx):
    for name in ids()[:ctx.args.n]:
        yield name, lambda name=name: build(name, ctx)
