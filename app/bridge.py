"""Access to the pilot v2 pipeline (package fpx) for the workflow app.

Everything the app needs from the pipeline goes through this module: the one path constant (PIPELINE_DIR), the lazy
model and OCR engine with clear errors when something is missing, and the two phases of a sheet:

    analyse(job_dir, upload, log)            stage 0a normalisation, deskew, sheet OCR, 1b layout, 1c scale proposals
    extract(job_dir, session, confirm, log)  stages 2-10 per confirmed drawing, with the confirmed region and scale

Both phases are fpx.pipeline.run_document: analyse() calls it with stages=() and keeps the packages in memory;
extract() calls it again on those packages with the user's confirmations as `overrides` (confirmed region, scale,
storey, selection). The sheet OCR is cached by run_document in the job folder, so the second call costs no OCR.
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

def analyse(job_dir, upload, log, want_model=True):
    """Load the upload (fpx.inputs) and run fpx.pipeline.run_document with no stages: deskew, sheet OCR (cached in
    job_dir/ocr_cache), regions and drawings (fpx.layout) and a scale proposal per drawing (fpx.scale.drawing_scale,
    with the door-width search for floor plans when the model is available). Writes preview images and
    analysis.json into job_dir; returns (analysis, session) with the session holding the packages in memory for
    extract()."""
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
    eng = engine()                                         # OCR is needed for every text cue (scale notes, captions)
    mdl = None
    if want_model:
        try:
            mdl = model()                                  # door-width cue when no precise cue exists
        except AppError as ex:
            log(f"warning: {ex} (the door-width scale cue is skipped; extraction will fail until the model exists)")
    for pkg in pkgs:
        log(f"sheet {pkg.id}: {pkg.input_class}, {pkg.img.shape[1]} x {pkg.img.shape[0]} px"
            + (f", {pkg.dpi:g} dpi ({pkg.dpi_source})" if pkg.dpi else ", resolution unknown"))
    log("  reading the sheet text (OCR, this takes a while on large sheets), layout and scale proposals")
    entries = pipeline.run_document(None, model=mdl, engine=eng, cfg=cfg, stages=(), packages=pkgs,
                                    cache_dir=job_dir / "ocr_cache", progress=lambda key, stage: log(f"  [{key}] {stage}"))
    session = {"packages": []}
    analysis = {"packages": [], "seconds": None}
    for k, (pkg, entry) in enumerate(zip(pkgs, entries)):
        lay, times, skew = entry["layout"], entry["times"], entry["deskew"]
        if skew and skew.get("applied"):
            log(f"  deskewed by {skew['deg']} degrees")
        info = pkg.provenance.get("sheet_ocr") or {}
        log(f"  {info.get('items', 0)} text items in {info.get('seconds', 0)} s" + (" (cached)" if info.get("cached") else ""))
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
            rec["scale"] = pipeline._scale_summary(s)
            if rec["scale"] is not None:
                rec["scale"]["cue_boxes"] = _cue_boxes(d, s, r.get("texts") or [], r.get("crop_offset") or (0, 0), lay, bars)
                rec["scale"]["seconds"] = r["times"].get("scale")
            rec["error"] = r["error"]
            drawings.append(rec)
            v = s["px_per_m"] if s else None
            log(f"  {d['id']} {d['kind']}" + (f" '{d['title']}'" if d["title"] else "") + ": "
                + (f"{v:.1f} px/m ({s['confidence']}: {', '.join(a['cue'] for a in s['consensus']['agreeing'])})"
                   if v else "no scale cue, to be measured") + (f"; error: {r['error']}" if r["error"] else ""))
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
    log(f"analysis finished in {analysis['seconds']} s")
    return analysis, session


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


def extract(job_dir, session, confirm, log, progress):
    """Run the confirmed drawings through stages 2-10. confirm: {"<package>/<drawing>": {"extract": bool,
    "polygon_px": [[x, y], ...] or None, "px_per_m": float or None, "scale_source": str, "measured": {...}, "storey": str}}.
    Writes <drawing>.json/.dxf, <drawing>_work.jpg, <drawing>_overlay.png and <package>_sheet.json into job_dir/out.
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
    entries = pipeline.run_document(None, model=mdl, engine=eng, cfg=cfg, out_dir=out, packages=pkgs, overrides=overrides,
                                    cache_dir=job_dir / "ocr_cache", progress=progress)
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
