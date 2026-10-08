"""Access to the pilot v2 pipeline (package fpx) for the workflow app.

Everything the app needs from the pipeline goes through this module: the one path constant (PIPELINE_DIR), the lazy
model and OCR engine with clear errors when something is missing, and the three phases of a sheet, one per step of
the workflow (docs/pipeline.md §3, "What runs at each step"):

    analyse(job_dir, upload, log)                        stage 0a normalisation, deskew, 1b layout: regions and drawings,
                                                         no OCR and no model (seconds) -> step 1, building area
    scale_step(job_dir, session, analysis, confirm,      stage 1c for the confirmed drawings: OCR of the title block, the
               keys, log, doors=False)                   caption band and the confirmed area, scale bars; the door-width
                                                         search (the segmenter) only with doors=True -> step 2, scale
    extract(job_dir, session, confirm, log, progress,    stages 2-10 per confirmed drawing inside its region at its scale,
            analysis=None)                               the OCR of step 2 reused from the cache -> step 3, run

All three are fpx.pipeline.run_document on the same packages (kept in memory in the session, pickled next to the
job): analyse() with stages=() and scale=False, scale_step() with stages=() and the OCR bounded to boxes, extract()
with the user's confirmations as `overrides` (confirmed region, scale, storey, selection) and the same boxes.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

APP = Path(__file__).resolve().parent
REPO = APP.parent

# The pipeline: pilot/v2-pipeline today, its code/ subfolder once the refactor moves the sources there.
PIPELINE_DIR = REPO / "pilot" / "v2-pipeline"
if (PIPELINE_DIR / "scripts" / "fpx").is_dir():
    PIPELINE_DIR = PIPELINE_DIR / "scripts"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

THREADS = int(os.environ.get("FPX_APP_THREADS", "4"))     # the machine is shared: torch and OpenCV threads
# the OCR cache is keyed by sheet id, raster digest, region and OCR settings (fpx.pipeline._cache_path), so one folder
# serves every job: the same sheet uploaded again (the demo, a corrected area) costs no OCR
CACHE_DIR = Path(os.environ.get("FPX_APP_OCR_CACHE") or (REPO / "app" / "data" / "ocr_cache"))
PREVIEW_MAX = 2200                                         # longer side of the sheet preview (px)
FLOOR_PLAN = "floor plan"


class AppError(RuntimeError):
    """An error to show in the UI as it is (a missing file, library or converter)."""


_lock = threading.Lock()
_model = None
_engine = None


# ---------------------------------------------------------------------------------------------------------------------
# Environment

def model_path():
    env = os.environ.get("V2_MODEL")
    if env:
        return Path(env)
    try:
        import common
        return Path(common.MODEL)
    except Exception:                                      # common.py moved or its imports failed: the known default
        return PIPELINE_DIR / "data" / "model-v2" / "segmenter.pt"


def check():
    """What the app needs, each with ok/message: shown on the upload page before anything runs."""
    items = {}

    def probe(name, fn):
        try:
            items[name] = {"ok": True, "detail": fn()}
        except Exception as ex:                            # noqa: BLE001 - every failure is reported, none hides
            items[name] = {"ok": False, "detail": f"{type(ex).__name__}: {ex}"}

    def pipeline_ok():
        __import__("fpx.pipeline", fromlist=["run_document"])
        return str(PIPELINE_DIR)

    def model_ok():
        p = model_path()
        if not p.exists():
            raise FileNotFoundError(f"segmenter checkpoint not found at {p} (train it with train.py or set V2_MODEL)")
        return str(p)

    def version(name):
        m = __import__(name)
        return getattr(m, "__version__", None) or getattr(m, "version", None) or "installed"

    probe("pipeline", pipeline_ok)
    probe("model file", model_ok)
    probe("torch", lambda: version("torch"))
    probe("opencv", lambda: version("cv2"))
    probe("shapely", lambda: version("shapely"))
    probe("ocr engine (rapidocr)", lambda: version("rapidocr"))
    probe("pdf (pymupdf)", lambda: version("fitz"))
    probe("dxf (ezdxf)", lambda: version("ezdxf"))
    probe("dwg converter", _dwg_converter)
    return {"pipeline_dir": str(PIPELINE_DIR), "python": sys.version.split()[0], "threads": THREADS, "items": items,
            "ok": all(v["ok"] for k, v in items.items() if k != "dwg converter")}


def _dwg_converter():
    from fpx import inputs
    try:
        conv = inputs.dwg_converter()
        return f"{type(conv).__name__}"
    except Exception as ex:                                # DwgConverterMissing: DWG uploads will fail with this text
        raise RuntimeError(f"none installed: DWG uploads fail, DXF works ({ex})")


def model():
    """The segmenter, loaded once (torch loads only here)."""
    global _model
    with _lock:
        if _model is None:
            path = model_path()
            if not path.exists():
                raise AppError(f"Segmenter model not found: {path}. Train it with pilot/v2-pipeline/train.py or point "
                               f"V2_MODEL at a checkpoint.")
            try:
                import torch
                import cv2
                torch.set_num_threads(THREADS)
                cv2.setNumThreads(min(THREADS, 4))
                from fpx.segment import load_model
                _model = load_model(path)
            except AppError:
                raise
            except Exception as ex:
                raise AppError(f"The segmenter could not be loaded from {path}: {type(ex).__name__}: {ex}") from ex
        return _model


def engine():
    """The OCR engine (RapidOCR, ONNX on CPU), loaded once."""
    global _engine
    with _lock:
        if _engine is None:
            try:
                from fpx.pipeline import ocr_engine
                _engine = ocr_engine()
            except ImportError as ex:
                raise AppError(f"OCR engine not available: {ex}. Install it with 'pip install rapidocr onnxruntime'.") from ex
            except Exception as ex:
                raise AppError(f"The OCR engine could not be started: {type(ex).__name__}: {ex}") from ex
        return _engine


def _jsonable(v):
    import numpy as np
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    return str(v)


def dump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_jsonable), encoding="utf-8")


def _public(d):
    """A drawing or region record without the private shapely objects."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------------------------------------------------
# Phase A: normalisation, layout and scale proposals (run_document without stages)

def analyse(job_dir, upload, log, want_model=False):
    """Step 1: load the upload (fpx.inputs) and run fpx.pipeline.run_document with no stages and scale=False: deskew,
    regions and drawings (fpx.layout, from the ink alone: no OCR, no model). Writes preview images and analysis.json
    (drawings without a scale record yet) into job_dir; returns (analysis, session) with the session holding the
    packages in memory for scale_step() and extract(). want_model is accepted for older callers and ignored."""
    from fpx import inputs, pipeline
    from fpx.config import DEFAULT as cfg
    job_dir = Path(job_dir)
    t_all = time.time()
    log(f"loading {Path(upload).name} (fpx.inputs.load)")
    try:
        pkgs = inputs.load(upload, cfg)
    except inputs.DwgConverterMissing as ex:
        raise AppError(f"DWG files need a converter, none is installed: {ex}") from ex
    except ValueError as ex:
        raise AppError(str(ex)) from ex
    if not pkgs:
        raise AppError("The file contains no page or layout to process.")
    for pkg in pkgs:
        log(f"sheet {pkg.id}: {pkg.input_class}, {pkg.img.shape[1]} x {pkg.img.shape[0]} px"
            + (f", {pkg.dpi:g} dpi ({pkg.dpi_source})" if pkg.dpi else ", resolution unknown"))
    log("  layout from the ink: drawings, title block, legend, notes (no OCR, no model)")
    entries = pipeline.run_document(None, model=None, engine=None, cfg=cfg, stages=(), packages=pkgs, sheet_ocr=False,
                                    scale=False, cache_dir=CACHE_DIR, progress=lambda key, stage: log(f"  [{key}] {stage}"))
    session = {"packages": []}
    analysis = {"packages": [], "seconds": None}
    for k, (pkg, entry) in enumerate(zip(pkgs, entries)):
        lay, times, skew = entry["layout"], entry["times"], entry["deskew"]
        if skew and skew.get("applied"):
            log(f"  deskewed by {skew['deg']} degrees")
        kinds = {}
        for d in lay["drawings"]:
            kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
        log(f"  layout: {len(lay['regions'])} regions, {len(lay['drawings'])} drawings "
            f"({', '.join(f'{n} {k}' for k, n in kinds.items()) or 'none'})")
        bars = [r for r in lay["regions"] if r["class"] == "scale bar" and r.get("px_per_m")]
        drawings = []
        for r in entry["drawings"]:
            d, s = r["drawing"], r["scale"]
            rec = _public(d)
            rec["scale"] = None                            # step 2 fills it (scale_step)
            rec["error"] = r["error"]
            drawings.append(rec)
            log(f"  {d['id']} {d['kind']}" + (f" '{d['title']}'" if d["title"] else "") + f": {d['bbox_px']}"
                + (f"; error: {r['error']}" if r["error"] else ""))
        preview, pf = _preview(pkg.img, job_dir / f"{pkg.id}_preview.jpg")
        analysis["packages"].append({
            "id": pkg.id, "index": k, "summary": pkg.summary(), "deskew": skew, "times": times,
            "preview": {"file": preview.name, "scale": pf, "width": int(pkg.img.shape[1] * pf), "height": int(pkg.img.shape[0] * pf)},
            "raster_px": [int(pkg.img.shape[1]), int(pkg.img.shape[0])],
            "layout": lay["sheet"], "layout_flags": lay["flags"],
            "regions": [_public(r) for r in lay["regions"]], "drawings": drawings})
        session["packages"].append({"pkg": pkg, "skew": skew})
    analysis["seconds"] = round(time.time() - t_all, 1)
    dump(analysis, job_dir / "analysis.json")
    log(f"layout finished in {analysis['seconds']} s")
    return analysis, session


def _ocr_boxes(analysis, confirm, keys):
    """Where the scale cues of the drawings in `keys` sit, as OCR boxes per package (sheet pixels): the confirmed (or
    detected) area with the caption band around it, the title block, the scale bars. The same boxes serve step 2 and
    step 3, so the extraction finds the OCR in the cache."""
    boxes = {}
    for a_pkg in analysis["packages"]:
        lay = a_pkg.get("layout") or {}
        reach = float((lay.get("thresholds_px") or {}).get("caption") or 0.0)
        regions = {r["id"]: r for r in a_pkg.get("regions", [])}
        wanted = [d for d in a_pkg["drawings"] if f"{a_pkg['id']}/{d['id']}" in keys]
        if not wanted:
            continue
        out = []
        for d in wanted:
            poly = (confirm.get(f"{a_pkg['id']}/{d['id']}") or {}).get("polygon_px") or d["polygon_px"]
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            out.append((min(xs) - reach, min(ys) - reach, max(xs) + reach, max(ys) + reach))
        tb = regions.get(lay.get("title_block"))
        if tb:
            out.append(tuple(tb["bbox_px"]))
        for r in a_pkg.get("regions", []):
            if r["class"] in ("scale bar", "scale note", "caption"):
                out.append(tuple(r["bbox_px"]))
        boxes[a_pkg["id"]] = out
    return boxes


def scale_step(job_dir, session, analysis, confirm, keys, log, doors=False):
    """Step 2: the scale cues of the drawings in `keys` (fpx.scale.drawing_scale through run_document with stages=()),
    with the sheet OCR bounded to the boxes of _ocr_boxes and the confirmed area as the drawing's region. The
    door-width search runs only with doors=True (it needs the segmenter). The analysis is updated in place (each
    drawing's scale record with its cue boxes, the caption and title found by the layout now that text is read) and
    written again; returns the keys that got a record."""
    from fpx import pipeline
    from fpx.config import DEFAULT as cfg
    job_dir = Path(job_dir)
    t0 = time.time()
    eng = engine()
    mdl = model() if doors else None
    pkgs = [P["pkg"] for P in session["packages"]]
    boxes = _ocr_boxes(analysis, confirm, keys)
    anchors = _anchors(analysis)
    overrides = {}
    for key in keys:
        c = confirm.get(key) or {}
        overrides[key] = {"polygon_px": c.get("polygon_px") or None, "extract": True, "storey": c.get("storey") or None}
    log("  reading the scale cues where they sit: title block, caption band, the confirmed area, scale bars"
        + (" · searching doors with the segmenter" if doors else ""))
    entries = pipeline.run_document(None, model=mdl, engine=eng, cfg=cfg, stages=(), packages=pkgs, sheet_ocr=True,
                                    ocr_boxes=boxes, overrides=overrides, anchors=anchors, cache_dir=CACHE_DIR,
                                    progress=lambda key, stage: log(f"  [{key}] {stage}"))
    done = []
    for pkg, entry in zip(pkgs, entries):
        a_pkg = next((a for a in analysis["packages"] if a["id"] == pkg.id), None)
        if a_pkg is None:
            continue
        lay = entry["layout"]
        info = pkg.provenance.get("sheet_ocr") or {}
        log(f"  {pkg.id}: {info.get('items', 0)} text items in {info.get('boxes', 0)} boxes, {info.get('seconds', 0)} s"
            + (" (cached)" if info.get("cached") else ""))
        bars = [r for r in lay["regions"] if r["class"] == "scale bar" and r.get("px_per_m")]
        for r in entry["drawings"]:
            d, s = r["drawing"], r["scale"]
            target = next((x for x in a_pkg["drawings"] if x["id"] == d["id"]), None)   # ids anchored to step 1
            if target is None:
                continue
            key = f"{pkg.id}/{target['id']}"
            if key not in keys:
                continue
            target["scale"] = pipeline._scale_summary(s)
            if target["scale"] is not None:
                target["scale"]["cue_boxes"] = _cue_boxes(d, s, r.get("texts") or [], r.get("crop_offset") or (0, 0), lay, bars)
                target["scale"]["seconds"] = r["times"].get("scale")
            for k in ("title", "caption", "storey", "storey_source", "scale_note", "kind", "kind_reason", "anchored"):
                if d.get(k) and k in d:
                    target[k] = d[k] if not isinstance(d[k], dict) or k != "caption" else _public(d[k])
            target["error"] = r["error"]
            v = s["px_per_m"] if s else None
            log(f"  {key}" + (f" '{d['title']}'" if d.get("title") else "") + ": "
                + (f"{v:.1f} px/m ({s['confidence']}: {', '.join(a['cue'] for a in s['consensus']['agreeing'])})"
                   if v else "no scale cue, to be measured") + (f"; error: {r['error']}" if r["error"] else ""))
            done.append(key)
    analysis["scale_seconds"] = round(time.time() - t0, 1)
    dump(analysis, job_dir / "analysis.json")
    log(f"scale cues read in {analysis['scale_seconds']} s")
    return done


def _anchors(analysis):
    """The drawing records of the analysis per package, for run_document(anchors=...): the later layouts (with text)
    keep the ids and the drawings the user confirmed in step 1 (a drawing the layout with text no longer finds is
    kept with its step-1 area)."""
    return {a_pkg["id"]: {d["id"]: {k: v for k, v in d.items() if k not in ("scale", "error")} for d in a_pkg["drawings"]}
            for a_pkg in analysis["packages"]}


def _preview(img, path):
    import cv2
    h, w = img.shape[:2]
    f = min(1.0, PREVIEW_MAX / max(h, w))
    small = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA) if f < 1 else img
    cv2.imwrite(str(path), cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Path(path), f


def _cue_boxes(d, s, inside, offset, lay, bars):
    """Where the scale cues sit on the sheet (raster px), for the highlights of the scale step: the caption with a
    scale note, the title block note, scale bars and the dimension strings that agree with the proposal."""
    ox, oy = offset
    out = []
    note = s.get("note") or {}
    if note.get("source") == "caption" and d.get("caption"):
        cap = d["caption"]["text"]
        out.append({"cue": "scale_note", "box": d["caption"]["box_px"], "label": f"Caption · “{cap[:36]}{'…' if len(cap) > 36 else ''}”"})
    elif note.get("source") and lay["sheet"].get("title_block"):
        tb = next((r for r in lay["regions"] if r["id"] == lay["sheet"]["title_block"]), None)
        if tb:
            out.append({"cue": "sheet_scale_note", "box": tb["bbox_px"], "label": f"Title block · 1:{note['scale']}"})
    for r in lay["regions"]:
        if r["class"] == "scale note":
            out.append({"cue": "scale_note", "box": r["bbox_px"], "label": f"Scale note · {r.get('text', '')[:40]}"})
    cues = s.get("cues") or {}
    if cues.get("scale_bar", {}).get("candidates") or cues.get("linked", {}).get("candidates"):
        for b in bars:
            out.append({"cue": "scale_bar", "box": b["bbox_px"], "label": "Graphical scale bar"})
    dims = cues.get("dimension_strings") or {}
    seen = set()
    for ex in dims.get("examples", []):
        if not ex.get("inlier"):
            continue
        for t in inside:
            if t.get("text") == ex["text"] and id(t) not in seen:
                seen.add(id(t))
                x0, y0, x1, y1 = t["box"]
                out.append({"cue": "dimension_strings", "box": [x0 + ox, y0 + oy, x1 + ox, y1 + oy], "label": ex["text"]})
                break
    return out[:60]


# ---------------------------------------------------------------------------------------------------------------------
# Phase B: stages 2-10 per confirmed drawing (run_document with the confirmations as overrides)

def _override(c):
    """A confirmation record of the app -> run_document override for one drawing (extract None: by kind)."""
    measured = c.get("measured")
    source = c.get("scale_source")
    if measured:
        note = f"measured by hand: {measured.get('px', 0):.0f} px = {measured.get('distance')} {measured.get('unit', 'm')}"
    elif source == "proposal":
        note = "proposal confirmed as shown"
    elif source:
        note = f"confirmed by hand ({source})"
    else:
        note = None
    return {"extract": None if c.get("extract") is None else bool(c["extract"]), "polygon_px": c.get("polygon_px") or None,
            "px_per_m": float(c["px_per_m"]) if c.get("px_per_m") else None, "storey": c.get("storey") or None,
            "note": note}


def extract(job_dir, session, confirm, log, progress, analysis=None):
    """Step 3: run the confirmed drawings through stages 2-10. confirm: {"<package>/<drawing>": {"extract": bool,
    "polygon_px": [[x, y], ...] or None, "px_per_m": float or None, "scale_source": str, "measured": {...}, "storey": str}}.
    With the analysis given, the sheet OCR is bounded to the same boxes as in step 2 (so it comes from the cache) and
    the drawing ids stay those of step 1 (anchors).
    Writes <drawing>.json/.dxf, <drawing>_work.jpg, <drawing>_overlay.png and <package>_sheet.json into job_dir/out,
    and while a drawing runs <key>_live.json with the geometry found so far (the app's live view).
    Returns {"<package>/<drawing>": result record}."""
    import cv2
    from fpx import pipeline
    from fpx.config import DEFAULT as cfg
    job_dir = Path(job_dir)
    out = job_dir / "out"
    out.mkdir(parents=True, exist_ok=True)
    mdl, eng = model(), engine()
    pkgs = [P["pkg"] for P in session["packages"]]
    overrides = {}
    for key, c in confirm.items():
        pkg_id, _, did = key.partition("/")
        if not any(p.id == pkg_id for p in pkgs):
            continue
        overrides[key] = _override(c)
    for key, ov in overrides.items():
        if ov["polygon_px"]:
            log(f"  {key}: building area set by hand ({len(ov['polygon_px'])} corners)")
        if ov["px_per_m"]:
            log(f"  {key}: scale {ov['px_per_m']:.2f} px/m ({ov['note']})")
    keys = [k for k, c in confirm.items() if c.get("extract") is not False]
    boxes = _ocr_boxes(analysis, confirm, keys) if analysis else None
    for stale in out.glob("*_live.json"):                  # live geometry of an earlier run must not show for the new one
        stale.unlink(missing_ok=True)

    def after(key, stage, sheet):                          # the live view: never let it break the extraction
        try:
            _live(out, key, stage, sheet)
        except Exception as ex:                            # noqa: BLE001
            log(f"  {key}: live view not written after {stage}: {type(ex).__name__}: {ex}")

    entries = pipeline.run_document(None, model=mdl, engine=eng, cfg=cfg, out_dir=out, packages=pkgs, overrides=overrides,
                                    ocr_boxes=boxes, anchors=_anchors(analysis) if analysis else None,
                                    cache_dir=CACHE_DIR, progress=progress, after=after)
    results = {}
    for pkg, entry in zip(pkgs, entries):
        for r in entry["drawings"]:
            d, sheet, s = r["drawing"], r["sheet"], r["scale"]
            key = f"{pkg.id}/{d['id']}"
            c = confirm.get(key) or {}
            rec = {"key": key, "package": pkg.id, "drawing": d["id"], "title": d["title"], "kind": d["kind"],
                   "storey": c.get("storey") or d["storey"], "skipped": r["skipped"], "times": r["times"], "outputs": None,
                   "error": r["error"], "second_pass": r.get("second_pass")}
            results[key] = rec
            if r["error"]:
                log(f"  {key}: failed: {r['error']}\n{r.get('traceback', '')}")
            if rec["skipped"] and not c.get("extract", d["kind"] == FLOOR_PLAN):
                rec["skipped"] = "not selected for extraction"
            if sheet is None:
                if r["skipped"] and "no scale" in r["skipped"]:
                    rec["error"] = ("No scale: the drawing has no scale cue and none was measured. Go back to the scale step "
                                    "and measure a known distance.")
                continue
            if s and s.get("confirmed"):
                sheet.meta["app"] = {"region_by_hand": bool(overrides.get(key, {}).get("polygon_px")), "scale_by_hand": True}
            work = out / f"{sheet.id}_work.jpg"
            cv2.imwrite(str(work), cv2.cvtColor(sheet.img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
            ov = out / f"{sheet.id}_overlay.png"
            cv2.imwrite(str(ov), cv2.cvtColor(_overlay(sheet), cv2.COLOR_RGB2BGR))
            rec["outputs"] = {"json": f"{sheet.id}.json", "dxf": f"{sheet.id}.dxf", "work": work.name, "overlay": ov.name}
            rec["sheet_id"] = sheet.id
            rec["px_per_m"] = cfg.px_per_m
            rec["work_px"] = [int(sheet.img.shape[1]), int(sheet.img.shape[0])]
            rec["times"]["total"] = round(sum(v for v in r["times"].values() if isinstance(v, (int, float))), 1)
            rec["summary"] = {"rooms": len(sheet.rooms), "gf_area": round(sheet.gf_area, 1),
                              "openings": {k: sum(o["kind"] == k for o in sheet.openings) for k in ("door", "exterior door", "window", "passage", "interior opening")},
                              "walls": len(sheet.wall_segments), "stairs": len(sheet.stairs), "qa": len(sheet.qa),
                              "style": (sheet.meta.get("triage") or {}).get("graphical_style")}
            if r.get("second_pass"):
                log(f"  {key}: second pass at {r['second_pass']['to']:.1f} px/m ({r['second_pass']['reason']})")
            log(f"  {key}: {len(sheet.rooms)} rooms, GF {sheet.gf_area:.1f} m2, {len(sheet.qa)} QA issues, {rec['times']['total']} s")
    return results


# ---------------------------------------------------------------------------------------------------------------------
# The analysed session on disk (pickle of the packages), so a server restart does not force the analysis (sheet OCR,
# door search) to run again before the extraction

def save_session(job_dir, session):
    import pickle
    try:
        with open(Path(job_dir) / "session.pkl", "wb") as f:
            pickle.dump({k: v for k, v in session.items() if k != "upload"}, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception as ex:                                # noqa: BLE001 - a session that cannot be saved is recomputed
        return f"{type(ex).__name__}: {ex}"


def load_session(job_dir):
    """The saved session, or None (the app's own file, written by save_session)."""
    import pickle
    p = Path(job_dir) / "session.pkl"
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:                                      # noqa: BLE001
        return None


LIVE_STAGES = ("triage", "walls", "openings", "stairs", "rooms")


def live_name(key):
    """The live file of a drawing key '<package>/<drawing>' in the job's out folder."""
    return f"{key.replace('/', '-')}_live.json"


def _rings(g):
    """Exterior rings of a shapely (multi)polygon in working px, rounded to 0.1 px; [] for anything else."""
    import shapely
    if g is None or getattr(g, "is_empty", True):
        return []
    polys = list(getattr(g, "geoms", [g]))
    out = []
    for p in polys:
        if p.geom_type != "Polygon":
            continue
        out.append([[round(float(x), 1), round(float(y), 1)] for x, y in p.exterior.coords])
    return out


def _live(out, key, stage, sheet):
    """After a per-drawing stage: the working image (once, after triage) and the geometry found so far, in working
    px of that image, as <key>_live.json for GET /api/sheets/<id>/live. Only the stages that add geometry write."""
    import cv2
    if stage not in LIVE_STAGES:
        return
    work = out / f"{sheet.id}_work.jpg"
    if not work.exists():
        cv2.imwrite(str(work), cv2.cvtColor(sheet.img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    rec = {"stage": stage, "work": work.name, "size": [int(sheet.img.shape[1]), int(sheet.img.shape[0])], "updated": time.time(),
           "walls": [], "openings": [], "stairs": [], "rooms": [], "gf": []}
    if stage != "triage":
        for p in (getattr(sheet, "wall_polys", None) or []):
            rec["walls"].extend(_rings(p))
    if stage in ("openings", "stairs", "rooms"):
        rec["openings"] = [{"kind": o.get("kind"), "rings": _rings(o.get("poly"))} for o in (getattr(sheet, "openings", None) or [])]
    if stage in ("stairs", "rooms"):
        for p in (getattr(sheet, "stairs", None) or []):
            rec["stairs"].extend(_rings(p))
    if stage == "rooms":
        rec["rooms"] = [{"rings": _rings(r.get("poly")), "name": r.get("name")} for r in (getattr(sheet, "rooms", None) or [])]
        rec["gf"] = _rings(getattr(sheet, "gf", None))
    dump(rec, out / live_name(key))


def _overlay(sheet):
    """Segmenter classes tinted over the working image plus room outlines, as run_pipeline.py draws it."""
    import cv2
    import numpy as np
    colours = np.array([[255, 255, 255], [40, 40, 40], [230, 40, 40], [40, 120, 230], [40, 170, 40], [200, 40, 200]], np.uint8)
    img = sheet.img.copy()
    if sheet.label is not None:
        seg = colours[np.clip(sheet.label, 0, len(colours) - 1)]
        m = sheet.label > 0
        img[m] = (0.35 * img[m] + 0.65 * seg[m]).astype(np.uint8)
    rng = np.random.default_rng(1)
    for r in sheet.rooms:
        c = tuple(int(v) for v in rng.integers(60, 220, 3))
        cv2.polylines(img, [np.asarray(r["poly"].exterior.coords, np.int32)], True, c, 2)
    for o in sheet.openings:
        if o["kind"] == "passage" and o.get("line"):
            a, b = o["line"]
            cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (255, 140, 0), 3)
    return img
