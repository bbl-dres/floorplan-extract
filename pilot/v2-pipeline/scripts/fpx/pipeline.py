"""Run stages 0-10 (docs/pipeline.md) on one sheet, or a subset of them.

    from fpx import pipeline
    times = pipeline.run(sheet, model=model, engine=ocr_engine, out_dir=out)          # all stages
    pipeline.run(sheet, stages=pipeline.AFTER_SEGMENTATION)                           # sheet.label already set

Stages fill in the sheet (fpx.model.Sheet) in place; run() returns the seconds spent per stage.

run_document(path, ...) is the whole-document route: normalisation, sheet text, layout, a scale proposal per drawing,
then the stages per floor plan drawing (see its docstring). The workflow app calls it twice: once with stages=() for
the proposals, then with the user's confirmations as overrides.
"""
import hashlib
import json
import math
import re
import time
import traceback
from pathlib import Path

from .attributes import attributes
from .config import DEFAULT
from .derived import derived
from .export import export
from .openings import openings
from .qa import qa
from .rooms import rooms
from .stairs import stairs
from .text import text_layer
from .triage import preprocess, triage
from .walls import walls

STAGES = ("triage", "preprocess", "layout", "text", "segment", "walls", "openings", "stairs", "rooms", "attributes", "derived", "qa", "export")
AFTER_SEGMENTATION = STAGES[STAGES.index("walls"):STAGES.index("export")]
LABELS = {"triage": "0 triage", "preprocess": "1 preprocessing", "layout": "1b layout and masking", "text": "2 text layer", "segment": "3 segmentation",
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
        "layout": lambda: __import__("fpx.layout", fromlist=["mask_sheet"]).mask_sheet(sheet, cfg),
        "text": lambda: text_layer(sheet, engine, cfg),
        "segment": lambda: __import__("fpx.segment", fromlist=["segment"]).segment(sheet, model, cfg),   # torch loads only here
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
    """Local OCR: RapidOCR with the PP-OCRv5 Latin recogniser (ONNX, CPU). Recognition runs in batches of 16 crops
    (fpx.text.recognise hands it every box of a sheet at once)."""
    from rapidocr import LangRec, OCRVersion, RapidOCR
    return RapidOCR(params={"Rec.lang_type": LangRec.LATIN, "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.rec_batch_num": 16})


# ---------------------------------------------------------------------------------------------------------------------
# A whole document: stage 0a normalisation, then per sheet 1 preprocessing, sheet text and 1b layout, then per drawing
# 1c scale and stages 2-10 (docs/pipeline.md 3: stages 1-9 run once per drawing region of type floor plan)

DRAWING_STAGES = STAGES[STAGES.index("segment"):]       # after the per-drawing text layer
OCR_RES_STEPS = 4                                       # the drawing OCR resolution is quantised to 2^(k/4): a scale change of a few per cent keeps the OCR cache key


def run_document(path, model=None, engine=None, cfg=DEFAULT, out_dir=None, stages=STAGES, kinds=("floor plan",),
                 load_opts=None, packages=None, sheet_ocr=True, overrides=None, cache_dir=None, progress=None,
                 scale=True, ocr_boxes=None, anchors=None, after=None):
    """Normalise an upload (fpx.inputs) and run every floor-plan drawing on it through the pipeline.

    Per sheet package: deskew (raster inputs), sheet text (native runs, plus one OCR pre-pass of the whole sheet with
    fpx.scale.read_text when engine is given and sheet_ocr), layout (fpx.layout: regions and drawings). Per drawing:
    crop with its mask (white outside), scale proposal (fpx.scale.drawing_scale: its own caption note, the title-block
    note as sheet default only, its dimension strings, scale bars and room sizes, a viewport's exact scale; without
    any cue the door-width search or the sheet-size prior, flagged). Then, for drawings of a kind in `kinds` (or
    selected by an override) and when `stages` is not empty: the text items inside the drawing are the sheet
    pre-pass's where its resolution is close to cfg.ocr_px_per_m (cfg.ocr_prepass_reuse), else one OCR pass inside
    the drawing's mask at cfg.ocr_px_per_m (re-rendered from a PDF where the sheet raster is too coarse); the items
    are transformed into the drawing's working frame, never OCR'd again. The crop is resampled to cfg.px_per_m and
    the stages in `stages` run on a Sheet whose regions and drawings carry the layout. After stage 9 the stamp-area
    cue joins the scale consensus; when it moves the scale by more than cfg.scale_cue_agreement the drawing runs a
    second time at the new scale (the OCR items are reused). Writes <drawing sheet id>.json/.dxf per drawing (export
    stage) and <package id>_sheet.json with the package summary, regions, drawings, scale records and outputs.

    overrides: {"<package id>/<drawing id>": {"polygon_px": [[x, y], ...], "px_per_m": float, "storey": str,
    "extract": bool, "note": str}} from a user's confirmation: a polygon replaces the drawing's region and mask
    (plus the layout margin, source "human"), px_per_m replaces the consensus (fpx.scale.confirm: recorded as
    confirmed, the measured cues listed as agreeing or disagreeing), extract False skips the drawing and True runs it
    whatever its kind, storey replaces the parsed one, note is recorded with the scale.

    cache_dir: OCR items are cached there as JSON per (package raster, region, resolution) so that a second call on
    the same sheet (the app's extraction after its analysis, a rerun after a scale change) costs no OCR; default
    out_dir/ocr_cache when out_dir is given. packages: already loaded (and deskewed) packages instead of `path`.
    progress: callback(key, stage label) before each step. scale=False stops after the layout (the app's building-area
    step: regions and drawings within seconds, no OCR, no model; the drawings' scale records stay None). ocr_boxes:
    {package id: [(x0, y0, x1, y1), ...]} in sheet pixels limits the sheet OCR pre-pass to those boxes (the app's
    scale step: title block, the caption band of a drawing, its confirmed area, scale bars), each cached on its own.
    anchors: {package id: {drawing id: drawing record}} from an earlier layout of the same sheet (the app's step 1,
    without text); the drawings of this run take the id of the anchored drawing they overlap most (bbox IoU >= 0.3),
    so overrides keyed by the earlier ids and the output names stay with the same drawing when the layout with text
    numbers the drawings differently, and an anchored drawing this layout no longer finds is kept with its earlier
    area (with_region); a drawing only this layout finds is skipped unless an override selects it: what the user
    confirmed is what runs. after: callback(key, stage, sheet) after each per-drawing stage (triage, text, then
    segment ... qa) with the fpx.model.Sheet as it stands, for live views of the work in progress.
    Returns [{"package", "layout", "texts", "drawings": [{"drawing", "scale", "sheet" (fpx.model.Sheet or None),
    "times", "skipped", "error", "texts", "crop_offset", "text_ocr", "second_pass"}], "times", "deskew"}].
    pipeline.run(sheet, ...) is unchanged for sheets loaded whole (it keeps its own OCR)."""
    from . import inputs, layout
    from . import scale as fscale
    if out_dir is None and "export" in stages:
        raise ValueError("stage 'export' needs out_dir")
    if "segment" in stages and model is None:
        raise ValueError("stage 'segment' needs a model")
    out_dir = None if out_dir is None else Path(out_dir)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is None and out_dir is not None:
        cache_dir = out_dir / "ocr_cache"
    cache_dir = None if cache_dir is None else Path(cache_dir)
    overrides = overrides or {}
    say = progress or (lambda key, stage: None)
    pkgs = packages if packages is not None else inputs.load(path, cfg, **(load_opts or {}))
    results = []
    for pkg in pkgs:
        t0 = time.time()
        times = {}
        skew = _deskew_package(pkg, cfg)
        times["preprocess"] = round(time.time() - t0, 1)
        t = time.time()
        say(pkg.id, "1 sheet text")
        ocr_items, info = [], None
        if engine is not None and sheet_ocr:
            ocr_items, info = sheet_text(pkg, engine, cfg, cache_dir, boxes=(ocr_boxes or {}).get(pkg.id))
            pkg.provenance["sheet_ocr"] = info
        times["sheet text"] = round(time.time() - t, 1)
        t = time.time()
        say(pkg.id, "1b layout")
        lay = layout.analyse(pkg, texts=ocr_items, cfg=cfg)
        if anchors and anchors.get(pkg.id):
            _anchor_drawings(lay, anchors[pkg.id], cfg)
        times["layout"] = round(time.time() - t, 1)
        texts = layout._merge_texts([x for x in pkg.text if not x.get("invisible") and x.get("text")], ocr_items)
        sheet_notes = lay["sheet"]["sheet_scale_notes"]
        bars = [r for r in lay["regions"] if r["class"] == "scale bar" and r.get("px_per_m")]
        entry = {"package": pkg, "layout": lay, "texts": texts, "drawings": [], "times": times, "deskew": skew}
        for d in lay["drawings"]:
            key = f"{pkg.id}/{d['id']}"
            ov = overrides.get(key) or {}
            rec = {"drawing": d, "scale": None, "sheet": None, "times": {}, "skipped": None, "error": None,
                   "override": ov or None, "second_pass": None}
            entry["drawings"].append(rec)
            try:
                if ov.get("storey"):
                    d["storey"], d["storey_source"] = ov["storey"], "human"
                if ov.get("polygon_px"):
                    d = with_region(d, ov["polygon_px"], lay, cfg)
                    rec["drawing"] = d
                selected = ov.get("extract")
                if selected is False:
                    rec["skipped"] = "deselected by the user" + (f": {ov['note']}" if ov.get("note") else "")
                elif selected is None and d.get("anchored") == "new":
                    rec["skipped"] = "found by the layout with text only, not among the confirmed drawings"
                elif selected is None and d["kind"] not in kinds:
                    rec["skipped"] = f"kind {d['kind']}: only {', '.join(kinds)} go through stages 2-9"
                run_it = bool(stages) and rec["skipped"] is None
                want_model = model if rec["skipped"] is None and not ov.get("px_per_m") else None   # the door search only for drawings to extract
                if not scale:                              # layout only (the app's building-area step)
                    continue
                t = time.time()
                say(key, "1c scale")
                crop, (ox, oy), mask = _crop(pkg.img, d)
                inside = [dict(x, box=(x["box"][0] - ox, x["box"][1] - oy, x["box"][2] - ox, x["box"][3] - oy))
                          for x in texts if _in_mask(mask, x["box"], ox, oy)]
                own = d["scale_note"]["all"] if d["scale_note"]["source"] == "caption" else []
                vp_scale = d["scale_note"]["scale"] if d["scale_note"]["source"] == "viewport" else None
                linked = _linked_bars(d, bars, lay)
                confirmed = ov.get("px_per_m")
                s = fscale.drawing_scale(crop, inside, pkg.dpi, pkg.dpi_source, dpis=[tuple(a) for a in pkg.dpi_alternatives] or None,
                                         caption_notes=own, sheet_notes=sheet_notes, viewport_scale=vp_scale,
                                         model_px_per_m=(pkg.model_scale or {}).get("px_per_m"), extra=linked,
                                         nts=d.get("not_to_scale", False), model=want_model,
                                         extent_px=(crop.shape[1], crop.shape[0]), cfg=cfg)
                if confirmed:
                    fscale.confirm(s, float(confirmed), ov.get("note"), cfg)
                rec["scale"], rec["texts"], rec["crop_offset"] = s, inside, (ox, oy)
                rec["times"]["scale"] = round(time.time() - t, 1)
                if not run_it:
                    continue
                if not s["px_per_m"]:
                    rec["skipped"] = "no scale: no cue, no prior and nothing confirmed (two-point calibration)"
                    continue
                _drawing_stages(pkg, d, lay, crop, (ox, oy), mask, inside, s, rec, model, engine, cfg, out_dir, stages,
                                ocr_items, info, cache_dir, skew, say, key, after)
            except Exception as ex:                        # one drawing failing must not take the others down
                rec["error"] = f"{type(ex).__name__}: {ex}"
                rec["traceback"] = traceback.format_exc()
        if out_dir is not None:
            doc = {"sheet": pkg.summary(), "regions": lay["regions"], "layout_flags": lay["flags"],
                   "layout": lay["sheet"], "times": times, "deskew": skew,
                   "drawings": [{**{k: v for k, v in r["drawing"].items() if not k.startswith("_")},
                                 "scale": _scale_summary(r["scale"]), "skipped": r["skipped"], "error": r["error"],
                                 "override": r["override"], "times": r["times"], "text_ocr": r.get("text_ocr"),
                                 "second_pass": r["second_pass"],
                                 "outputs": None if r["sheet"] is None or "export" not in stages else
                                 {"json": f"{r['sheet'].id}.json", "dxf": f"{r['sheet'].id}.dxf"}}
                                for r in entry["drawings"]]}
            (out_dir / f"{pkg.id}_sheet.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=_jsonable),
                                                          encoding="utf-8")
        results.append(entry)
    return results


def _drawing_stages(pkg, d, lay, crop, offset, mask, inside, s, rec, model, engine, cfg, out_dir, stages,
                    ocr_items, info, cache_dir, skew, say, key, after=None):
    """Stages 2-10 of one drawing at its proposed (or confirmed) scale, with the stamp-area cue after stage 9 and one
    rerun when it moves an unconfirmed scale by more than cfg.scale_cue_agreement. after(key, stage, sheet) is
    called after each stage that ran."""
    from . import scale as fscale
    done = after or (lambda key, stage, sheet: None)
    t = time.time()
    say(key, "2 text (OCR inside the drawing)")
    items, tinfo = drawing_text_items(pkg, d, crop, offset, mask, s["px_per_m"], ocr_items, info, engine, cfg, cache_dir)
    rec["text_ocr"] = tinfo
    t_ocr = round(time.time() - t, 1)
    sheet = None
    for n_pass in (1, 2):
        times = {"text ocr": t_ocr if n_pass == 1 else 0.0}
        sheet = drawing_sheet(pkg, d, lay, crop, offset, inside, s, cfg, mask=mask, ocr_items=items)
        rec["sheet"] = sheet
        if n_pass == 2:
            sheet.meta["second_pass"] = rec["second_pass"]
        say(key, LABELS["triage"])
        times.update(run(sheet, model=model, engine=engine, cfg=cfg, out_dir=out_dir, stages=[x for x in ("triage",) if x in stages]))
        if "triage" in sheet.meta:
            sheet.meta["triage"]["route"] = "raster pipeline per drawing (fpx.pipeline.run_document)"
        if skew:
            sheet.meta["deskew"] = skew                     # done once on the sheet, not again per drawing
        if "triage" in stages:
            done(key, "triage", sheet)
        if "text" in stages:
            say(key, LABELS["text"])
            t = time.time()
            _drawing_text(sheet, engine, d, cfg)
            times["text"] = round(time.time() - t, 1)
            done(key, "text", sheet)
        for stage in DRAWING_STAGES:
            if stage in stages and stage != "export":
                say(key, LABELS[stage])
                times.update(run(sheet, model=model, engine=engine, cfg=cfg, out_dir=out_dir, stages=(stage,)))
                done(key, stage, sheet)
        if n_pass == 1:
            rec["times"].update(times)
        else:
            rec["second_pass"]["times_first_pass"] = {k: v for k, v in rec["times"].items() if k != "scale"}
            rec["times"] = {"scale": rec["times"].get("scale"), **times}
        # stage 9: the stamp-area cue joins the scale consensus; an unconfirmed scale that moves is rerun once
        stamp = fscale.stamp_area_cue(sheet.scale.get("stamp_area_cue"), s["px_per_m"], cfg) if "qa" in stages else {"candidates": []}
        if stamp["candidates"]:
            before = s["px_per_m"]
            fscale.add_drawing_cue(s, "stamp_areas", stamp, cfg)
            moved = abs(math.log(s["px_per_m"] / before))
            if n_pass == 1 and not s.get("confirmed") and moved > cfg.scale_cue_agreement:
                rec["second_pass"] = {"from": round(before, 3), "to": s["px_per_m"], "ratio": round(s["px_per_m"] / before, 4),
                                      "reason": f"the stamp-area cue moved the scale by {moved * 100:.1f} % (more than "
                                                f"{cfg.scale_cue_agreement * 100:g} %): the drawing is run again at the new scale"}
                continue
            if moved > cfg.scale_cue_agreement:            # confirmed or a second pass: the value stays, the cue is recorded
                s.update(px_per_m=before, metres_per_px=round(1.0 / before, 8))
                s["consensus"]["px_per_m"] = round(before, 3)
            sheet.scale.update(cues=s["cues"], consensus=s["consensus"], flags=s["flags"], confirmed=s.get("confirmed"))
            sheet.drawings[0]["scale"] = _scale_summary(s)
        break
    if rec["second_pass"]:
        sp = rec["second_pass"]
        s["flags"].append({"check": "scale", "severity": "low", "element": "drawing",
                           "message": f"second pass at {sp['to']:.1f} px/m: the stamp areas moved the scale "
                                      f"{sp['ratio']:.3f}x from the first pass at {sp['from']:.1f} px/m"})
    sheet.qa = list(sheet.qa) + [dict(f) for f in s["flags"]]
    if "export" in stages:                                  # the QA list now carries the scale flags too
        say(key, LABELS["export"])
        rec["times"].update(run(sheet, model=model, engine=engine, cfg=cfg, out_dir=out_dir, stages=("export",)))
    return sheet


def _jsonable(v):
    import numpy as np
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


# ---------------------------------------------------------------------------------------------------------------------
# Sheet text: the OCR pre-pass of the whole sheet and the pass inside a drawing, cached per raster, region and
# resolution (as bench.ocr_cached does per sheet) so that a second call on the same sheet costs no OCR

def _slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))[:80]


def _raster_digest(img):
    """Identity of a sheet raster for the OCR cache: every 8th pixel and the shape (deskewed rasters differ)."""
    h = hashlib.sha1(img[::8, ::8].tobytes())
    h.update(str(img.shape).encode())
    return h.hexdigest()


def _mask_digest(d):
    """Identity of a drawing's mask polygon(s) on the sheet."""
    return hashlib.sha1(json.dumps([d["mask"]["polygon_px"], d["mask"].get("parts")], default=_jsonable).encode()).hexdigest()


def _cache_path(cache_dir, pkg, region, tag):
    if cache_dir is None:
        return None
    return Path(cache_dir) / f"{_slug(pkg.id)}__{_raster_digest(pkg.img)[:10]}__{region[:10]}__{tag}.json"


def _cache_read(path):
    if path is None or not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    for t in d["items"]:
        t["box"] = tuple(t["box"])
    return d


def _cache_write(path, items, info):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"items": items, "info": info}, ensure_ascii=False, default=_jsonable), encoding="utf-8")


def sheet_text(pkg, engine, cfg=DEFAULT, cache_dir=None, boxes=None):
    """The OCR pre-pass of the sheet (fpx.scale.read_text: characters resampled to ~16 px, boxes in sheet pixels),
    cached in cache_dir per raster and parameter set. With boxes [(x0, y0, x1, y1)] only those parts of the sheet are
    read, each cached on its own (the app reads the title block, the caption band and the confirmed area of a
    drawing instead of the whole sheet); an item whose centre lies in an earlier box is not read twice. Returns
    (items, info); info["factor"] is the smallest resampling factor of the boxes."""
    from . import scale as fscale
    dpi = pkg.dpi if pkg.dpi_trusted else None
    tag = (f"pre-c{cfg.scale_ocr_char_px:g}-{cfg.scale_ocr_small_px:g}-{cfg.scale_ocr_enlarge_px:g}_t{cfg.scale_ocr_tile}"
           f"_o{cfg.scale_ocr_overlap}_m{cfg.scale_ocr_max_side}_r{cfg.ocr_rotated_scale:g}_c{cfg.ocr_min_conf:g}_d{dpi or 0:g}")
    if boxes:
        H, W = pkg.img.shape[:2]
        items, factor, secs, hits, done = [], None, 0.0, 0, []
        for b in boxes:
            x0, y0 = max(0, int(math.floor(b[0]))), max(0, int(math.floor(b[1])))
            x1, y1 = min(W, int(math.ceil(b[2]))), min(H, int(math.ceil(b[3])))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            path = _cache_path(cache_dir, pkg, f"box{x0}-{y0}-{x1}-{y1}", tag)
            cached = _cache_read(path)
            if cached is not None:
                part, pinfo = cached["items"], cached["info"]
                hits += 1
            else:
                part, pinfo = fscale.read_text(pkg.img[y0:y1, x0:x1], engine, cfg, dpi=dpi)
                part = [dict(t, box=(t["box"][0] + x0, t["box"][1] + y0, t["box"][2] + x0, t["box"][3] + y0)) for t in part]
                _cache_write(path, part, pinfo)
            for t in part:
                cx, cy = (t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2
                if not any(bx0 <= cx < bx1 and by0 <= cy < by1 for bx0, by0, bx1, by1 in done):
                    items.append(t)
            done.append((x0, y0, x1, y1))
            f = pinfo.get("factor")
            factor = f if factor is None or f is None else min(factor, f)
            secs += pinfo.get("seconds") or 0.0
        return items, {"source": "sheet pre-pass (bounded)", "boxes": len(done), "factor": factor, "items": len(items),
                       "seconds": round(secs, 1), "cached": bool(done) and hits == len(done)}
    path = _cache_path(cache_dir, pkg, "sheet", tag)
    cached = _cache_read(path)
    if cached is not None:
        return cached["items"], {**cached["info"], "cached": True}
    items, info = fscale.read_text(pkg.img, engine, cfg, dpi=dpi)
    _cache_write(path, items, info)
    return items, info


def drawing_text_items(pkg, d, crop, offset, mask, px_per_m, sheet_items, sheet_info, engine, cfg=DEFAULT, cache_dir=None):
    """OCR items inside one drawing, in sheet pixels: the sheet pre-pass's items when its resolution is at least
    cfg.ocr_prepass_reuse of cfg.ocr_px_per_m, else one OCR pass of the masked crop at cfg.ocr_px_per_m (quantised to
    quarter-octave steps so that a scale change of a few per cent keeps the cache key; re-rendered from the vector
    source where the sheet raster is too coarse, up to cfg.input_dpi_max). Returns (items, info)."""
    import cv2
    import numpy as np
    from . import inputs
    from .text import ocr
    ox, oy = offset
    if engine is None:
        return [], {"source": "no OCR engine", "items": 0}
    fo = cfg.ocr_px_per_m / px_per_m
    pre = (sheet_info or {}).get("factor")
    if pre is not None and pre >= cfg.ocr_prepass_reuse * fo:
        items = [x for x in sheet_items if _in_mask(mask, x["box"], ox, oy)]
        return items, {"source": "sheet pre-pass", "factor": pre, "px_per_m": round(pre * px_per_m, 1), "items": len(items),
                       "seconds": 0.0}
    fq = 2.0 ** (round(math.log2(fo) * OCR_RES_STEPS) / OCR_RES_STEPS)
    tag = f"ocr-f{fq:.4f}_t{cfg.ocr_tile}_o{cfg.ocr_overlap}_r{cfg.ocr_rotated_scale:g}_c{cfg.ocr_min_conf:g}"
    path = _cache_path(cache_dir, pkg, _mask_digest(d), tag)
    cached = _cache_read(path)
    if cached is not None:
        return cached["items"], {**cached["info"], "cached": True}
    t0 = time.time()
    grey = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    size = (max(1, int(round(crop.shape[1] * fq))), max(1, int(round(crop.shape[0] * fq))))
    small, rerendered = None, None
    if fq > 1.05 and pkg.dpi:
        dpi = min(cfg.input_dpi_max, pkg.dpi * fq)
        hi = inputs.rerender(pkg, (ox, oy, ox + crop.shape[1], oy + crop.shape[0]), dpi)
        if hi is not None:
            small = cv2.resize(cv2.cvtColor(hi, cv2.COLOR_RGB2GRAY), size, interpolation=cv2.INTER_AREA)
            if mask is not None:
                small[cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) == 0] = 255
            rerendered = round(dpi, 1)
            pkg.provenance.setdefault("rerendered", []).append({"drawing": d["id"], "dpi": rerendered})
    if small is None:
        small = cv2.resize(grey, size, interpolation=cv2.INTER_AREA if fq < 1 else cv2.INTER_CUBIC)
    items = []
    for t in ocr(small, engine, cfg.ocr_tile, cfg.ocr_overlap, cfg.ocr_min_conf, cfg.ocr_rotated_scale):
        x0, y0, x1, y1 = t["box"]
        items.append(dict(t, box=(x0 / fq + ox, y0 / fq + oy, x1 / fq + ox, y1 / fq + oy), height=t["height"] / fq))
    info = {"source": "drawing pass", "factor": round(fq, 4), "px_per_m": round(fq * px_per_m, 1), "ocr_shape": list(small.shape),
            "rerendered_dpi": rerendered, "items": len(items), "seconds": round(time.time() - t0, 1)}
    _cache_write(path, items, info)
    return items, info


# ---------------------------------------------------------------------------------------------------------------------
# Per-sheet and per-drawing helpers

def _deskew_package(pkg, cfg):
    """Stage 1 on the sheet raster: estimate the skew from long straight lines (as triage does, on a reduced copy) and
    rotate scans and photos; renders of vector files are never skewed. Text boxes follow the rotation (box centres).
    A package deskewed by an earlier call (pkg.provenance["deskew"]) is left as it is."""
    import cv2
    import numpy as np
    if "deskew" in pkg.provenance:
        return pkg.provenance["deskew"]
    if not any(k in pkg.input_class for k in ("raster", "mixed")):
        return None
    g = cv2.cvtColor(pkg.img, cv2.COLOR_RGB2GRAY)
    f = min(1.0, 3000 / max(g.shape))
    small = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else g
    L = max(small.shape)
    lines = cv2.HoughLinesP(cv2.Canny(small, 50, 150), 1, np.pi / 1800, 150, minLineLength=int(0.05 * L), maxLineGap=4)
    if lines is None:
        pkg.provenance["deskew"] = None
        return None
    ang = np.degrees(np.arctan2(lines[:, 0, 3] - lines[:, 0, 1], lines[:, 0, 2] - lines[:, 0, 0]))
    ang = (ang + 45) % 90 - 45
    ln = np.hypot(lines[:, 0, 3] - lines[:, 0, 1], lines[:, 0, 2] - lines[:, 0, 0])
    near = np.abs(ang) < 5
    if not near.any():
        pkg.provenance["deskew"] = None
        return None
    skew = float(np.average(ang[near], weights=ln[near]))
    if abs(skew) <= cfg.deskew_min_deg:
        rec = {"deg": round(skew, 2), "applied": False, "note": f"below {cfg.deskew_min_deg} degrees: left alone"}
        pkg.provenance["deskew"] = rec
        return rec
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
    pkg.provenance["deskew"] = rec
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
            else:                                        # a bar between two plans of one sheet serves both, the nearer one first
                out.append(candidate("linked", b["px_per_m"], 0.3, 0.04,
                                     f"scale bar {b['id']} near the drawing, nearer to another drawing ({b['reason'][:80]})"))
    return out


def with_region(d, polygon, lay, cfg=DEFAULT):
    """The drawing with a polygon confirmed by the user as its region: polygon, bbox and mask (the polygon plus the
    layout margin, clipped to the sheet) replaced, source "human". Raises ValueError for an invalid polygon."""
    import shapely
    from shapely.geometry import Polygon
    try:
        poly = Polygon(polygon)
    except (ValueError, TypeError) as ex:
        raise ValueError(f"the region polygon is not valid: {ex}") from ex
    if not poly.is_valid or poly.area <= 0:
        raise ValueError("the region polygon is not valid (it must enclose an area)")
    margin = float(d["mask"].get("margin_px") or 0.0)
    W, H = lay["sheet"]["raster_px"]
    m = poly.buffer(margin, join_style="mitre").intersection(shapely.box(0, 0, W, H)) if margin else poly
    ring = lambda p: [[round(float(x), 1), round(float(y), 1)] for x, y in list(p.exterior.coords)[:-1]]
    if m.geom_type != "Polygon":
        m = max(shapely.get_parts(m), key=lambda p: p.area)
    out = dict(d)
    out["polygon_px"] = ring(poly)
    out["bbox_px"] = [round(float(v), 1) for v in poly.bounds]
    out["source"] = "human"
    out["confidence"] = "high"
    out["mask"] = {"polygon_px": ring(m), "parts": None, "margin_px": margin, "margin_mm": d["mask"].get("margin_mm"),
                   "reason": "region confirmed by the user, plus the layout margin"}
    pxmm = lay["sheet"].get("dpi")
    out["polygon_mm"] = None if not pxmm else [[round(x / (pxmm / 25.4), 2), round(y / (pxmm / 25.4), 2)] for x, y in out["polygon_px"]]
    out.pop("_mask_full", None)
    return out


def drawing_sheet(pkg, d, lay, crop, offset, texts, s, cfg=DEFAULT, mask=None, ocr_items=None):
    """A fpx.model.Sheet for one drawing at the working resolution: the crop resampled from its proposed (or
    confirmed) scale s, padded by a metre of white, the layout records and the transform sheet px -> working px ->
    plan m. ocr_items: the drawing's OCR items in sheet pixels (drawing_text_items), transformed into the working
    frame for the text stage (sheet.meta["ocr_items"]); without them the sheet gets an OCR image at cfg.ocr_px_per_m
    for fpx.text.text_layer to read (re-rendered from a PDF page where the sheet render is too coarse)."""
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
    ox, oy = offset
    ocr_img = None
    if ocr_items is None:
        fo = cfg.ocr_px_per_m / p
        po = int(round(pad * cfg.ocr_px_per_m / cfg.px_per_m))
        grey = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        size = (max(1, int(round(crop.shape[1] * fo))), max(1, int(round(crop.shape[0] * fo))))
        small = None
        if fo > 1.05 and pkg.dpi:
            hi = inputs.rerender(pkg, (ox, oy, ox + crop.shape[1], oy + crop.shape[0]), min(cfg.input_dpi_max, pkg.dpi * fo))
            if hi is not None:
                small = cv2.resize(cv2.cvtColor(hi, cv2.COLOR_RGB2GRAY), size, interpolation=cv2.INTER_AREA)
                if mask is not None:
                    small[cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) == 0] = 255
                pkg.provenance.setdefault("rerendered", []).append({"drawing": d["id"], "dpi": round(min(cfg.input_dpi_max, pkg.dpi * fo), 1)})
        if small is None:
            small = cv2.resize(grey, size, interpolation=cv2.INTER_AREA if fo < 1 else cv2.INTER_CUBIC)
        ocr_img = cv2.copyMakeBorder(small, po, po, po, po, cv2.BORDER_CONSTANT, value=255)
    H = work.shape[0]
    transform = {"sheet_px_to_work_px": [[f, 0.0, pad - ox * f], [0.0, f, pad - oy * f], [0.0, 0.0, 1.0]],
                 "work_px_to_plan_m": [[1 / cfg.px_per_m, 0.0, 0.0], [0.0, -1 / cfg.px_per_m, H / cfg.px_per_m], [0.0, 0.0, 1.0]],
                 "note": "plan metres of this drawing: x right, y up, origin at the lower-left corner of the padded crop"}
    c = s["consensus"]
    if s.get("confirmed"):
        r = s["confirmed"]
        method = (f"confirmed by the user ({r['note']})"
                  + (f"; the pipeline proposed {r['proposal_px_per_m']:.2f} px/m" if r.get("proposal_px_per_m") else "; the pipeline had no proposal")
                  + (f"; agrees with {', '.join(r['agrees_with'])}" if r["agrees_with"] else "")
                  + (f"; disagrees with {', '.join(r['disagrees_with'])}" if r["disagrees_with"] else ""))
    else:
        method = (f"stage 1c proposal ({s['confidence']} confidence: "
                  f"{', '.join(a['cue'] for a in c['agreeing']) or 'none'} agree"
                  + (f"; disagreeing: {', '.join(x['cue'] for x in c['disagreeing'])}" if c["disagreeing"] else "")
                  + "); to be confirmed in stage 9")
    scale_rec = {"value": f"{p:.2f} px/m on the sheet raster", "method": method, "native_px_per_m": round(p, 4),
                 "metres_per_px": s["metres_per_px"], "note": s["note"], "print_factor": s["print_factor"],
                 "cues": s["cues"], "consensus": c, "flags": s["flags"], "confirmed": s.get("confirmed")}
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
    if ocr_items is not None:
        sheet.meta["ocr_items"] = [dict(t, box=((t["box"][0] - ox) * f + pad, (t["box"][1] - oy) * f + pad,
                                                (t["box"][2] - ox) * f + pad, (t["box"][3] - oy) * f + pad),
                                        height=t.get("height", 0) * f) for t in ocr_items]
    caption = d.get("caption")
    if caption:
        b = caption["box_px"]
        sheet.meta["caption_box"] = (b[0] * f + pad - ox * f, b[1] * f + pad - oy * f, b[2] * f + pad - ox * f, b[3] * f + pad - oy * f)
    return sheet


def _drawing_text(sheet, engine, d, cfg):
    """Stage 2 for one drawing: its OCR items (sheet.meta["ocr_items"], or OCR of its image when there are none;
    fpx.text.text_layer) merged with the native runs of the sheet package (a native run wins over OCR covering 30 % of
    it), then roles by region: the caption's text is "caption" (never a room stamp); the rest by content as before.
    Title block and legend lie outside the mask, so they never get here."""
    import shapely
    from .text import role
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


def _bbox_iou(a, b):
    """Intersection over union of two boxes (x0, y0, x1, y1)."""
    def area(bb):
        return max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1])
    inter = area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def _anchor_drawings(lay, anchors, cfg=DEFAULT, min_iou=0.3):
    """Give each drawing of the layout the id of the anchored drawing ({id: record of an earlier layout}) it overlaps
    most (bbox IoU >= min_iou, each anchor used once); drawings without an anchor get fresh ids that collide with no
    anchor id; anchors without a drawing are revived from their record with their area as the region (with_region)
    and appended, flagged in the record and in lay["flags"]. In place."""
    drawings = lay["drawings"]
    taken = {}
    for aid, rec in anchors.items():
        best, best_iou = None, min_iou
        for d in drawings:
            if id(d) in taken:
                continue
            iou = _bbox_iou([float(v) for v in d["bbox_px"]], [float(v) for v in rec["bbox_px"]])
            if iou > best_iou:
                best, best_iou = d, iou
        if best is not None:
            taken[id(best)] = aid
    used, n = set(anchors) | set(taken.values()), 0
    for d in drawings:
        if id(d) in taken:
            d["id"] = taken[id(d)]
            d["anchored"] = "matched"
        else:
            n += 1
            while f"d{n}" in used:
                n += 1
            d["id"] = f"d{n}"
            d["anchored"] = "new"
            used.add(d["id"])
    for aid, rec in anchors.items():
        if aid in taken.values():
            continue
        try:
            d = with_region({k: v for k, v in rec.items() if not k.startswith("_")}, rec["polygon_px"], lay, cfg)
        except (ValueError, KeyError) as ex:
            lay["flags"].append({"severity": "medium", "message": f"drawing {aid} of the earlier layout could not be kept: {ex}"})
            continue
        d["id"] = aid
        d["mask"]["reason"] = "area of the earlier layout (no text), kept: the layout with text has no drawing here"
        d["kind_reason"] = (d.get("kind_reason") or "") + "; kept from the earlier layout"
        d["anchored"] = "revived"
        lay["flags"].append({"severity": "low", "message": f"drawing {aid}: not found by the layout with text, kept with its earlier area"})
        drawings.append(d)


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
            "flags": s["flags"], "confirmed": True if s.get("confirmed") else None, "confirmation": s.get("confirmed")}
