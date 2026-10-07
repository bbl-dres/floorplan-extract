"""Run stages 0-10 (docs/pipeline.md) on one sheet, or a subset of them.

    from fpx import pipeline
    times = pipeline.run(sheet, model=model, engine=ocr_engine, out_dir=out)          # all stages
    pipeline.run(sheet, stages=pipeline.AFTER_SEGMENTATION)                           # sheet.label already set

Stages fill in the sheet (fpx.model.Sheet) in place; run() returns the seconds spent per stage.
"""
import time

from .attributes import attributes
from .config import DEFAULT
from .derived import derived
from .export import export
from .openings import openings
from .qa import qa
from .rooms import rooms
from .segment import segment
from .stairs import stairs
from .text import text_layer
from .triage import preprocess, triage
from .walls import walls

STAGES = ("triage", "preprocess", "text", "segment", "walls", "openings", "stairs", "rooms", "attributes", "derived", "qa", "export")
AFTER_SEGMENTATION = STAGES[STAGES.index("walls"):STAGES.index("export")]
LABELS = {"triage": "0 triage", "preprocess": "1 preprocessing", "text": "2 text layer", "segment": "3 segmentation",
          "walls": "3 walls and wall graph", "openings": "4 openings", "stairs": "5 stairs and voids", "rooms": "6 rooms",
          "attributes": "7 room attributes", "derived": "8 derived outputs", "qa": "9 scale and QA", "export": "10 export"}


def run(sheet, model=None, engine=None, cfg=DEFAULT, out_dir=None, stages=STAGES):
    if sheet.px_per_m != cfg.px_per_m:
        raise ValueError(f"sheet is at {sheet.px_per_m} px/m, the configuration expects {cfg.px_per_m}")
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")
    if "segment" in stages and model is None:
        raise ValueError("stage 'segment' needs a model")
    if "export" in stages and out_dir is None:
        raise ValueError("stage 'export' needs out_dir")
    calls = {
        "triage": lambda: triage(sheet, cfg),
        "preprocess": lambda: preprocess(sheet, cfg),
        "text": lambda: text_layer(sheet, engine, cfg),
        "segment": lambda: segment(sheet, model, cfg),
        "walls": lambda: walls(sheet, cfg),
        "openings": lambda: openings(sheet, cfg),
        "stairs": lambda: stairs(sheet, cfg),
        "rooms": lambda: rooms(sheet, cfg),
        "attributes": lambda: attributes(sheet, cfg),
        "derived": lambda: derived(sheet, cfg),
        "qa": lambda: qa(sheet, cfg),
        "export": lambda: export(sheet, out_dir, cfg),
    }
    times = {}
    for name in STAGES:                                # always in pipeline order, whatever order was passed
        if name in stages:
            t = time.time()
            calls[name]()
            times[name] = round(time.time() - t, 1)
    return times


def ocr_engine():
    """Local OCR: RapidOCR with the PP-OCRv5 Latin recogniser (ONNX, CPU)."""
    from rapidocr import LangRec, OCRVersion, RapidOCR
    return RapidOCR(params={"Rec.lang_type": LangRec.LATIN, "Rec.ocr_version": OCRVersion.PPOCRV5})


# ---------------------------------------------------------------------------------------------------------------------
# A whole document: stage 0a normalisation, then per sheet 1 preprocessing and 1b layout, then per drawing 1c scale and
# stages 2-10 (docs/pipeline.md 3: stages 1-9 run once per drawing region of type floor plan)

DRAWING_STAGES = STAGES[STAGES.index("segment"):]       # after the per-drawing text layer


def run_document(path, model=None, engine=None, cfg=DEFAULT, out_dir=None, stages=STAGES, kinds=("floor plan",),
                 load_opts=None, packages=None, sheet_ocr=True):
    """Normalise an upload (fpx.inputs) and run every floor-plan drawing on it through the pipeline.

    Per sheet package: deskew (raster inputs), sheet text (native runs, plus OCR of the whole sheet with fpx.scale's
    pre-pass reader when engine is given and sheet_ocr), layout (fpx.layout: regions and drawings). Per drawing of a
    kind in `kinds`: crop with its mask (white outside), scale proposal (fpx.scale.drawing_scale: its own caption note,
    the title-block note as sheet default only, its dimension strings and scale bars, a viewport's exact scale),
    resample to cfg.px_per_m, then the stages in `stages` (triage, text with roles by region, segmentation ... export)
    on a Sheet whose regions and drawings carry the layout. Writes <drawing sheet id>.json/.dxf per drawing (export
    stage) and <package id>_sheet.json with the package summary, regions, drawings, scale records and outputs.
    Returns [{"package", "layout", "drawings": [{"drawing", "scale", "sheet" (fpx.model.Sheet or None), "times",
    "skipped"}]}]. pipeline.run(sheet, ...) is unchanged for sheets loaded whole."""
    import json
    from pathlib import Path

    from . import inputs, layout
    from . import scale as fscale
    if out_dir is None and "export" in stages:
        raise ValueError("stage 'export' needs out_dir")
    if "segment" in stages and model is None:
        raise ValueError("stage 'segment' needs a model")
    out_dir = None if out_dir is None else Path(out_dir)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    pkgs = packages if packages is not None else inputs.load(path, cfg, **(load_opts or {}))
    results = []
    for pkg in pkgs:
        t0 = time.time()
        times = {}
        skew = _deskew_package(pkg, cfg)
        times["preprocess"] = round(time.time() - t0, 1)
        t = time.time()
        ocr_items = []
        if engine is not None and sheet_ocr:
            dpi = pkg.dpi if pkg.dpi_trusted else None
            ocr_items, info = fscale.read_text(pkg.img, engine, cfg, dpi=dpi)
            pkg.provenance["sheet_ocr"] = info
        times["sheet text"] = round(time.time() - t, 1)
        t = time.time()
        lay = layout.analyse(pkg, texts=ocr_items, cfg=cfg)
        times["layout"] = round(time.time() - t, 1)
        texts = layout._merge_texts([x for x in pkg.text if not x.get("invisible") and x.get("text")], ocr_items)
        sheet_notes = lay["sheet"]["sheet_scale_notes"]
        bars = [r for r in lay["regions"] if r["class"] == "scale bar" and r.get("px_per_m")]
        entry = {"package": pkg, "layout": lay, "drawings": [], "times": times, "deskew": skew}
        for d in lay["drawings"]:
            rec = {"drawing": d, "scale": None, "sheet": None, "times": {}, "skipped": None}
            entry["drawings"].append(rec)
            if d["kind"] not in kinds:
                rec["skipped"] = f"kind {d['kind']}: only {', '.join(kinds)} go through stages 2-9"
                continue
            t = time.time()
            crop, (ox, oy), mask = _crop(pkg.img, d)
            inside = [dict(x, box=(x["box"][0] - ox, x["box"][1] - oy, x["box"][2] - ox, x["box"][3] - oy))
                      for x in texts if _in_mask(mask, x["box"], ox, oy)]
            own = d["scale_note"]["all"] if d["scale_note"]["source"] == "caption" else []
            vp_scale = d["scale_note"]["scale"] if d["scale_note"]["source"] == "viewport" else None
            linked = _linked_bars(d, bars, lay)
            s = fscale.drawing_scale(crop, inside, pkg.dpi, pkg.dpi_source, dpis=[tuple(a) for a in pkg.dpi_alternatives] or None,
                                     caption_notes=own, sheet_notes=sheet_notes, viewport_scale=vp_scale,
                                     model_px_per_m=(pkg.model_scale or {}).get("px_per_m"), extra=linked,
                                     nts=d.get("not_to_scale", False), model=model, cfg=cfg)
            rec["scale"] = s
            rec["times"]["scale"] = round(time.time() - t, 1)
            if not s["px_per_m"]:
                rec["skipped"] = "no scale cue for this drawing: to be set by hand (two-point calibration)"
                continue
            sheet = _drawing_sheet(pkg, d, lay, crop, (ox, oy), inside, s, cfg, mask=mask)
            rec["sheet"] = sheet
            run(sheet, model=model, engine=engine, cfg=cfg, out_dir=out_dir, stages=[x for x in ("triage",) if x in stages])
            if "triage" in sheet.meta:
                sheet.meta["triage"]["route"] = "raster pipeline per drawing (fpx.pipeline.run_document)"
            if skew:
                sheet.meta["deskew"] = skew                     # done once on the sheet, not again per drawing
            if "text" in stages:
                t = time.time()
                _drawing_text(sheet, engine, d, cfg)
                rec["times"]["text"] = round(time.time() - t, 1)
            rest = [x for x in DRAWING_STAGES if x in stages and x != "export"]
            if rest:
                rec["times"].update(run(sheet, model=model, engine=engine, cfg=cfg, out_dir=out_dir, stages=rest))
            sheet.qa = list(sheet.qa) + [dict(f) for f in s["flags"]]
            if "export" in stages:                              # the QA list now carries the scale flags too
                run(sheet, model=model, engine=engine, cfg=cfg, out_dir=out_dir, stages=("export",))
        if out_dir is not None:
            doc = {"sheet": pkg.summary(), "regions": lay["regions"], "layout_flags": lay["flags"],
                   "layout": lay["sheet"], "times": times, "deskew": skew,
                   "drawings": [{**{k: v for k, v in r["drawing"].items() if not k.startswith("_")},
                                 "scale": _scale_summary(r["scale"]), "skipped": r["skipped"], "times": r["times"],
                                 "outputs": None if r["sheet"] is None or "export" not in stages else
                                 {"json": f"{r['sheet'].id}.json", "dxf": f"{r['sheet'].id}.dxf"}}
                                for r in entry["drawings"]]}
            (out_dir / f"{pkg.id}_sheet.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=_jsonable),
                                                          encoding="utf-8")
        results.append(entry)
    return results


def _jsonable(v):
    import numpy as np
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def _deskew_package(pkg, cfg):
    """Stage 1 on the sheet raster: estimate the skew from long straight lines (as triage does, on a reduced copy) and
    rotate scans and photos; renders of vector files are never skewed. Text boxes follow the rotation (box centres)."""
    import cv2
    import numpy as np
    if not any(k in pkg.input_class for k in ("raster", "mixed")):
        return None
    g = cv2.cvtColor(pkg.img, cv2.COLOR_RGB2GRAY)
    f = min(1.0, 3000 / max(g.shape))
    small = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else g
    L = max(small.shape)
    lines = cv2.HoughLinesP(cv2.Canny(small, 50, 150), 1, np.pi / 1800, 150, minLineLength=int(0.05 * L), maxLineGap=4)
    if lines is None:
        return None
    ang = np.degrees(np.arctan2(lines[:, 0, 3] - lines[:, 0, 1], lines[:, 0, 2] - lines[:, 0, 0]))
    ang = (ang + 45) % 90 - 45
    ln = np.hypot(lines[:, 0, 3] - lines[:, 0, 1], lines[:, 0, 2] - lines[:, 0, 0])
    near = np.abs(ang) < 5
    if not near.any():
        return None
    skew = float(np.average(ang[near], weights=ln[near]))
    if abs(skew) <= cfg.deskew_min_deg:
        return {"deg": round(skew, 2), "applied": False, "note": f"below {cfg.deskew_min_deg} degrees: left alone"}
    h, w = g.shape
    R = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
    pkg.img = cv2.warpAffine(pkg.img, R, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    for t in pkg.text:
        x0, y0, x1, y1 = t["box"]
        cx, cy = R @ [(x0 + x1) / 2, (y0 + y1) / 2, 1]
        t["box"] = (cx - (x1 - x0) / 2, cy - (y1 - y0) / 2, cx + (x1 - x0) / 2, cy + (y1 - y0) / 2)
    rec = {"deg": round(skew, 2), "applied": True, "matrix": [[round(v, 9) for v in row] for row in R.tolist()],
           "note": "affine matrix from the original to the deskewed sheet raster; regions and drawings are in the deskewed frame"}
    pkg.transforms["deskew"] = rec["matrix"]
    if abs(skew) > cfg.flat_scan_max_deg:
        pkg.warn("skew", "medium", f"skew {skew:.1f} degrees: scanned or photographed at an angle")
    return rec


def _crop(img, d):
    """The drawing's mask bounding box cut from the sheet raster, everything outside the mask white."""
    import numpy as np
    from . import layout
    mask_full = layout.drawing_mask(d, img.shape)
    ys, xs = np.nonzero(mask_full)
    if not len(xs):
        x0, y0, x1, y1 = (int(v) for v in d["bbox_px"])
    else:
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    crop = img[y0:y1, x0:x1].copy()
    m = mask_full[y0:y1, x0:x1]
    crop[~m] = 255
    return crop, (int(x0), int(y0)), m


def _in_mask(m, box, ox, oy):
    cx, cy = int((box[0] + box[2]) / 2) - ox, int((box[1] + box[3]) / 2) - oy
    return 0 <= cy < m.shape[0] and 0 <= cx < m.shape[1] and bool(m[cy, cx])


def _linked_bars(d, bars, lay):
    """Scale bars outside the drawing polygon but within its caption distance: candidates for its scale."""
    import shapely
    from shapely.geometry import Polygon
    from .scale import candidate
    if not bars or len(d["polygon_px"]) < 3:
        return []
    poly = Polygon(d["polygon_px"])
    dist = lay["sheet"]["thresholds_px"]["caption"]
    out = []
    for b in bars:
        bp = shapely.box(*b["bbox_px"])
        if poly.contains(bp.centroid):
            continue                                     # inside: the crop's own scale-bar cue reads it
        if poly.distance(bp) <= dist:
            others = [Polygon(o["polygon_px"]) for o in lay["drawings"] if o is not d and len(o["polygon_px"]) >= 3]
            if all(o.distance(bp) > poly.distance(bp) for o in others):
                out.append(candidate("linked", b["px_per_m"], 0.5, 0.04, f"scale bar {b['id']} next to the drawing ({b['reason'][:80]})"))
    return out


def _drawing_sheet(pkg, d, lay, crop, offset, texts, s, cfg, mask=None):
    """A fpx.model.Sheet for one drawing at the working resolution: the crop resampled from its proposed scale, padded
    by a metre of white, with an OCR image, the layout records and the transform sheet px -> working px -> plan m.
    Where the OCR resolution (cfg.ocr_px_per_m) needs more pixels than the sheet render has and the sheet is a PDF
    page, the drawing is rendered again from the vector source (up to cfg.input_dpi_max) instead of upsampled: small
    lettering of reduced prints stays sharp."""
    import cv2
    import numpy as np
    from . import inputs
    from .model import Sheet
    p = s["px_per_m"]
    f = cfg.px_per_m / p
    pad = int(cfg.px_per_m)
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR
    work = cv2.copyMakeBorder(cv2.resize(crop, None, fx=f, fy=f, interpolation=interp), pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))
    fo = cfg.ocr_px_per_m / p
    po = int(round(pad * cfg.ocr_px_per_m / cfg.px_per_m))
    grey = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    size = (max(1, int(round(crop.shape[1] * fo))), max(1, int(round(crop.shape[0] * fo))))
    small = None
    if fo > 1.05 and pkg.dpi:
        ox_, oy_ = offset
        hi = inputs.rerender(pkg, (ox_, oy_, ox_ + crop.shape[1], oy_ + crop.shape[0]), min(cfg.input_dpi_max, pkg.dpi * fo))
        if hi is not None:
            small = cv2.resize(cv2.cvtColor(hi, cv2.COLOR_RGB2GRAY), size, interpolation=cv2.INTER_AREA)
            if mask is not None:
                small[cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) == 0] = 255
            pkg.provenance.setdefault("rerendered", []).append({"drawing": d["id"], "dpi": round(min(cfg.input_dpi_max, pkg.dpi * fo), 1)})
    if small is None:
        small = cv2.resize(grey, size, interpolation=cv2.INTER_AREA if fo < 1 else cv2.INTER_CUBIC)
    ocr_img = cv2.copyMakeBorder(small, po, po, po, po, cv2.BORDER_CONSTANT, value=255)
    H = work.shape[0]
    ox, oy = offset
    transform = {"sheet_px_to_work_px": [[f, 0.0, pad - ox * f], [0.0, f, pad - oy * f], [0.0, 0.0, 1.0]],
                 "work_px_to_plan_m": [[1 / cfg.px_per_m, 0.0, 0.0], [0.0, -1 / cfg.px_per_m, H / cfg.px_per_m], [0.0, 0.0, 1.0]],
                 "note": "plan metres of this drawing: x right, y up, origin at the lower-left corner of the padded crop"}
    method = (f"stage 1c proposal ({s['confidence']} confidence: "
              f"{', '.join(a['cue'] for a in s['consensus']['agreeing']) or 'none'} agree"
              + (f"; disagreeing: {', '.join(x['cue'] for x in s['consensus']['disagreeing'])}" if s["consensus"]["disagreeing"] else "")
              + "); to be confirmed in stage 9")
    scale_rec = {"value": f"{p:.2f} px/m on the sheet raster", "method": method, "native_px_per_m": round(p, 4),
                 "metres_per_px": s["metres_per_px"], "note": s["note"], "print_factor": s["print_factor"],
                 "cues": s["cues"], "consensus": s["consensus"], "flags": s["flags"]}
    title = d["title"] or f"{d['kind']} {d['id']}"
    sheet = Sheet(f"{pkg.id}-{d['id']}", title, str(pkg.source), pkg.input_class, work, (0.0, H / cfg.px_per_m),
                  scale_rec, ocr_img=ocr_img, meta={"package": pkg.id, "drawing": d["id"], "storey": d["storey"]})
    region_ids = {d["region"]} | ({d["caption"]["region"]} if d.get("caption") else set())
    sheet.regions = [r for r in lay["regions"] if r["id"] in region_ids or r["class"] in ("title block", "frame")]
    drec = {k: v for k, v in d.items() if not k.startswith("_")}
    drec["scale"] = _scale_summary(s)
    drec["transform"] = transform
    sheet.drawings = [drec]
    # native text of the drawing (PDF/DXF runs) in working pixels, for the text stage
    sheet.meta["native_runs"] = [dict(t, box=tuple(v * f + pad for v in t["box"]), height=t.get("height", 0) * f)
                                 for t in texts if t.get("source") in ("pdf", "dxf")]
    caption = d.get("caption")
    if caption:
        b = caption["box_px"]
        sheet.meta["caption_box"] = (b[0] * f + pad - ox * f, b[1] * f + pad - oy * f, b[2] * f + pad - ox * f, b[3] * f + pad - oy * f)
    return sheet


def _drawing_text(sheet, engine, d, cfg):
    """Stage 2 for one drawing: OCR of its crop (fpx.text.text_layer) merged with the native runs of the sheet package
    (a native run wins over OCR covering 30 % of it), then roles by region: the caption's text is "caption" (never a
    room stamp); the rest by content as before. Title block and legend lie outside the mask, so they never get here."""
    import shapely
    from .text import role, text_layer
    text_layer(sheet, engine, cfg)
    native = []
    for t in sheet.meta.pop("native_runs", []):
        native.append({"text": t["text"], "box": tuple(t["box"]), "conf": 1.0, "source": t["source"],
                       "angle": t.get("angle", 0), "height": t.get("height", 0.0)})
    keep = []
    for t in sheet.text:
        b = shapely.box(*t["box"])
        if any(b.intersection(shapely.box(*n["box"])).area > 0.3 * b.area for n in native):
            continue
        keep.append(t)
    items = native + keep
    cap = sheet.meta.get("caption_box")
    cb = shapely.box(*cap) if cap else None
    for t in items:
        b = shapely.box(*t["box"])
        if cb is not None and b.intersection(cb).area > 0.5 * b.area:
            t["role"] = "caption"
        else:
            t["role"] = role(t["text"].strip())
    sheet.text = items
    return items


def _scale_summary(s):
    """Compact JSON record of a drawing's stage 1c scale (the full cues are in the drawing's own JSON)."""
    if not s:
        return None
    c = s["consensus"]
    return {"px_per_m": s["px_per_m"], "metres_per_px": s["metres_per_px"], "confidence": s["confidence"],
            "note": s["note"], "print_factor": s["print_factor"],
            "agreeing": [{k: a[k] for k in ("cue", "px_per_m", "ratio", "evidence")} for a in c["agreeing"]],
            "disagreeing": [{k: a[k] for k in ("cue", "px_per_m", "ratio", "evidence")} for a in c["disagreeing"]],
            "snapped": c.get("snapped"), "scale": c.get("scale"), "scale_equivalent": c.get("scale_equivalent"),
            "flags": s["flags"], "confirmed": None}
