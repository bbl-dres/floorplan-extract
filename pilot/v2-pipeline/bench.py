"""Run pipeline stages 0-10 on sample plans from public datasets, for the viewer and a first look at unseen styles.

    python bench.py                 # all samples below
    python bench.py waffle commons  # only these sources
    python bench.py scale-eval cubicasa:60 cvcfp:40 commons landgut [--doors]   # scale cues against known scales

Sources and scale (the segmenter needs PX_PER_M):
- CubiCasa5K (CC BY-NC-SA): 100 px/m, known from the room dimension labels; reference walls, openings, rooms.
- CVC-FP (CC BY-NC): scale from the reference door symbols (0.85 m leaves); reference walls, openings, rooms.
- WAFFLE benchmark (per-image Commons licences): reference walls, doors, windows and interior as masks.
- Commons Swiss plans (public domain, CC0): no reference.
Sheets without a known scale take the consensus of the scale cues (fpx.scale: scale note, dimension strings, scale
bar, door widths; stamp areas after a first pass). Sheets with a known scale keep it, and the cues are checked
against it. All cues end up in the sheet's scale record (exported JSON).
All of these are benchmark data: outputs stay in data/bench (gitignored).
"""
import json
import math
import re
import sys
import time
from dataclasses import replace

import cv2
import numpy as np
import shapely
import torch
from PIL import Image
from shapely import affinity
from shapely.geometry import Polygon

import cubicasa
from common import CLASSES, DATA, DOOR, PX_PER_M, REPO, WALL, WINDOW
from fpx import DEFAULT, Sheet, pipeline, scale
from fpx.geometry import mask_polys
from fpx.segment import load_model

Image.MAX_IMAGE_PIXELS = None
OUT = DATA / "bench"
SCALE_OUT = OUT / "scale_eval"
BENCH = REPO / "data/benchmark"
SAMPLES = {
    "cubicasa": ["high_quality/7523", "high_quality_architectural/2085", "colorful/11260", "high_quality_architectural/2530"],
    "cvcfp": ["1", "image014", "Ib_CT0601_sommaire", "IIa_GC0801", "p2"],
    "waffle": ["HouseFlrPlan", "Grundriss_Hall_house_105b", "Rzut_budynku", "Tearoom_layout", "Typical_Dogtrot_Floorplan",
               "Categories_of_Aile_Richelieu", "Complesso_del_Pio_Monte_della_Misericordia", "Hofburg_Vienna_plan"],
    "commons": [f"c{k:02d}" for k in range(1, 19)],
}
DOOR_M = DEFAULT.scale_door_width               # typical clear door width for the scale search
OCR_PX_PER_M = DEFAULT.ocr_px_per_m
SECOND_PASS = 0.05                              # rerun once if the stamp-area cue moves an estimated scale this much


# ---------- sources ----------

def rgb(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("RGB"))


def cvcfp_paths(n):
    folder = BENCH / "cvc-fp/ImagesGT"
    return next(folder.glob(f"{n}_gt_*.svg")), next(p for p in (folder / f"{n}.png", folder / f"{n}.jpg") if p.exists())


def image_path(source, sid):
    if source == "cubicasa":
        return cubicasa.ROOT / sid / "F1_scaled.png"
    if source == "cvcfp":
        return cvcfp_paths(sid)[1]
    if source == "waffle":
        return BENCH / "waffle/data/benchmark/pngs" / f"{sid}.png"
    return next((BENCH / "commons-plans/images").glob(f"{sid}.*"))


def meta_dpi(path):
    """dpi in the image file's metadata (unverified: renditions often carry a default)."""
    d = Image.open(path).info.get("dpi")
    return float(d[0]) if d and d[0] and d[0] > 1 else None


def load_cubicasa(pid):
    d = cubicasa.load(pid)
    ref = {"walls": d["walls"], "doors": d["doors"], "windows": d["windows"],
           "rooms": [r["poly"] for r in d["rooms"] if not r["type"].startswith("Outdoor")]}
    return rgb(d["image"]), 100.0, "known: room dimension labels (1 cm/px)", ref, None


def load_cvcfp(n):
    svg, img_path = cvcfp_paths(n)
    text = svg.read_text(encoding="utf-8", errors="replace")
    polys = {}
    for cls, pts in re.findall(r'<polygon class="([^"]+)"[^>]*points="([^"]+)"', text):
        xy = np.array([float(v) for v in re.split(r"[ ,]+", pts.strip()) if v]).reshape(-1, 2)
        if len(xy) >= 3:
            p = Polygon(xy).buffer(0)
            if p.area > 0:
                polys.setdefault(cls, []).append(p)
    doors = polys.get("Door", [])
    sizes = [max(np.hypot(*(np.asarray(d.minimum_rotated_rectangle.exterior.coords)[k + 1] - np.asarray(d.minimum_rotated_rectangle.exterior.coords)[k])) for k in range(2)) for d in doors]
    px_per_m = float(np.median(sizes)) / 0.85
    ref = {"walls": polys.get("Wall", []), "doors": doors, "windows": polys.get("Window", []), "rooms": polys.get("Room", [])}
    return rgb(img_path), px_per_m, f"from {len(doors)} reference door symbols (0.85 m)", ref, None


def load_waffle(name):
    folder = BENCH / "waffle/data/benchmark"
    img = rgb(folder / "pngs" / f"{name}.png")
    seg = np.asarray(Image.open(folder / "segmented_descrete_pngs" / f"{name}_seg_colors.png").convert("RGB"))
    col = lambda c: np.all(seg == c, axis=-1)
    masks = {"walls": col((255, 0, 0)), "doors": col((0, 0, 255)), "windows": col((0, 255, 255)), "interior": col((255, 255, 255))}
    return img, None, None, None, masks


def load_commons(cid):
    path = next((BENCH / "commons-plans/images").glob(f"{cid}.*"))
    return rgb(path), None, None, None, None


LOADERS = {"cubicasa": load_cubicasa, "cvcfp": load_cvcfp, "waffle": load_waffle, "commons": load_commons}


# ---------- stage 9: scale ----------

def resized(img, f):
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)


def propose_scale(img, model):
    """Segment at candidate scales; choose the one where detected doors are closest to DOOR_M wide.
    The search now lives in fpx.scale.door_search (one cue of the consensus); kept here for existing callers."""
    r = scale.door_search(img, model, DEFAULT)
    return r["px_per_m"], r["method"], r["rows"]


def ocr_cached(key, img, engine, cache_dir=None, dpi=None):
    """Pre-pass OCR (fpx.scale.read_text), cached as JSON per sheet and parameter set."""
    cfg = DEFAULT
    cache_dir = cache_dir or SCALE_OUT / "ocr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"c{cfg.scale_ocr_char_px:g}-{cfg.scale_ocr_small_px:g}-{cfg.scale_ocr_enlarge_px:g}_t{cfg.scale_ocr_tile}"
           f"_o{cfg.scale_ocr_overlap}_m{cfg.scale_ocr_max_side}")
    path = cache_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', key)}__{tag}.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        for t in d["items"]:
            t["box"] = tuple(t["box"])
        return d["items"], {**d["info"], "cached": True}
    items, info = scale.read_text(img, engine, cfg, dpi=dpi)
    path.write_text(json.dumps({"items": items, "info": info}, ensure_ascii=False), encoding="utf-8")
    return items, info


def scale_prepass(key, img, engine, model=None, dpi=None, dpi_source=None, meta=None, native_text=(), cache_dir=None):
    """All scale cues on the native image (OCR cached); door search only when a model is given."""
    trusted = dpi if dpi_source in ("render", "scan") else None
    texts, info = ([], None) if engine is None else ocr_cached(key, img, engine, cache_dir, trusted)
    r = scale.prepass(img, model=model, dpi=dpi, dpi_source=dpi_source, native_text=native_text, meta_dpi=meta,
                      texts=texts)
    r["ocr"] = info
    if info:
        r["seconds"]["ocr"] = info["seconds"]
    return r


def scale_record(pre, px_per_m, method):
    """Sheet.scale: the value used, how it was obtained, and every cue with the consensus (and its flags)."""
    return {"value": f"{px_per_m:.1f} px/m", "method": method, "native_px_per_m": round(float(px_per_m), 3),
            **scale.record(pre)}


# ---------- run ----------

def one_pass(source, sid, img, px_per_m, scale_rec, model, engine):
    f = PX_PER_M / px_per_m
    pad = PX_PER_M
    work = cv2.copyMakeBorder(resized(img, f), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    fo = OCR_PX_PER_M / px_per_m
    ocr_img = cv2.copyMakeBorder(cv2.cvtColor(resized(img, fo), cv2.COLOR_RGB2GRAY), int(pad * 3), int(pad * 3), int(pad * 3), int(pad * 3),
                                 cv2.BORDER_CONSTANT, value=255)
    sheet_id = f"{source}_{sid.replace('/', '_')}"
    sheet = Sheet(sheet_id, f"{source}: {sid}", sid, "raster (public benchmark)", work, (0.0, 0.0), scale_rec, ocr_img=ocr_img)
    pipeline.run(sheet, model=model, engine=engine, stages=pipeline.STAGES[:-1])      # all but export
    return sheet, f, pad


def run(source, sid, model, engine):
    img, px_per_m, scale_method, ref, masks = LOADERS[source](sid)
    t0 = time.time()
    known = px_per_m is not None
    pre = scale_prepass(f"{source}_{sid}", img, engine, None if known else model, meta=meta_dpi(image_path(source, sid)))
    con = pre["consensus"]
    search = pre["cues"].get("door_widths", {}).get("rows")
    if known:
        scale.compare_known(con, px_per_m, scale_method)
    else:
        if not con["px_per_m"]:
            raise ValueError("no scale cue: set the scale by hand")
        px_per_m = con["px_per_m"]
        scale_method = (f"consensus of scale cues ({con['confidence']} confidence: "
                        f"{', '.join(a['cue'] for a in con['agreeing']) or 'none'} agree"
                        f"{'; disagreeing: ' + ', '.join(d['cue'] for d in con['disagreeing']) if con['disagreeing'] else ''}); to be confirmed")
    sheet, f, pad = one_pass(source, sid, img, px_per_m, scale_record(pre, px_per_m, scale_method), model, engine)
    # stamp areas after the first pass: a further cue; an estimated scale that it moves by more than 5 % is rerun once
    stamp = scale.stamp_area_cue(sheet.scale.get("stamp_area_cue"), px_per_m)
    if stamp["candidates"]:
        con = scale.add_cue(pre, "stamp_areas", stamp)
        if known:
            scale.compare_known(con, px_per_m, scale_method)
        elif con["px_per_m"] and abs(math.log(con["px_per_m"] / px_per_m)) > SECOND_PASS:
            first = px_per_m
            px_per_m = con["px_per_m"]
            scale_method += f"; second pass after stamp areas (first pass at {first:.1f} px/m)"
            sheet, f, pad = one_pass(source, sid, img, px_per_m, scale_record(pre, px_per_m, scale_method), model, engine)
            con["second_pass_from"] = round(first, 3)
        sheet.scale.update(scale_record(pre, px_per_m, scale_method))
    sheet.qa.extend(con["flags"])                      # scale disagreements are QA issues, never resolved silently
    pipeline.run(sheet, stages=("export",), out_dir=OUT)
    cv2.imwrite(str(OUT / f"{sheet.id}_sheet.jpg"), cv2.cvtColor(sheet.img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])

    # reference in plan metres (native px -> working px -> plan) and scores
    to_plan = lambda g: shapely.transform(affinity.translate(affinity.scale(g, f, f, origin=(0, 0)), pad, pad),
                                          lambda xy: np.c_[xy[:, 0] / PX_PER_M, -xy[:, 1] / PX_PER_M])
    H, W = sheet.label.shape
    gt = np.zeros((H, W), np.uint8)
    reference = {}
    if ref is not None:
        reference = {k: [to_plan(p) for p in v] for k, v in ref.items()}
        for key, val in (("walls", WALL), ("windows", WINDOW), ("doors", DOOR)):
            for p in ref[key]:
                for q in getattr(p, "geoms", [p]):
                    cv2.fillPoly(gt, [np.round((np.asarray(q.exterior.coords) * f + pad) * 16).astype(np.int32)], val, shift=4)
    elif masks is not None:
        for key, val in (("walls", WALL), ("windows", WINDOW), ("doors", DOOR)):
            m = cv2.resize(masks[key].astype(np.uint8), (W - 2 * pad, H - 2 * pad), interpolation=cv2.INTER_NEAREST)
            gt[pad:H - pad, pad:W - pad][m > 0] = val
            reference[key] = [shapely.transform(p, lambda xy: np.c_[xy[:, 0] / PX_PER_M, -xy[:, 1] / PX_PER_M])
                              for p in mask_polys(np.pad(m, pad) > 0, 4)]
    scores = {}
    if ref is not None or masks is not None:
        k = len(CLASSES)
        cm = np.bincount(gt.ravel().astype(np.int64) * k + sheet.label.ravel(), minlength=k * k).reshape(k, k)
        iou = lambda i: float(cm[i, i] / max(cm[i].sum() + cm[:, i].sum() - cm[i, i], 1))
        scores = {"wall_iou": round(iou(WALL), 3), "door_iou": round(iou(DOOR), 3), "window_iou": round(iou(WINDOW), 3)}
        if ref is not None and ref.get("rooms"):
            pred = [shapely.transform(r["poly"], lambda xy: np.c_[xy[:, 0] / PX_PER_M, -xy[:, 1] / PX_PER_M]) for r in sheet.rooms]
            refr = reference["rooms"]
            best = [max((g.intersection(p).area / g.union(p).area for p in pred), default=0) for g in refr]
            scores.update(rooms_ref=len(refr), rooms_pred=len(pred), rooms_matched=int(sum(b >= 0.5 for b in best)),
                          rooms_mean_iou=round(float(np.mean(best)), 3) if best else None)
    entry = {"id": sheet.id, "source": source, "sid": sid, "px_per_m": px_per_m, "scale_method": scale_method,
             "scale_search": search, "scale_consensus": con, "scores": scores, "seconds": round(time.time() - t0),
             "reference": {k: [shapely.geometry.mapping(shapely.set_precision(p, 0.001)) for p in v] for k, v in reference.items()}}
    (OUT / f"{sheet.id}_bench.json").write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
    return entry, sheet


# ---------- scale evaluation (review step 6b) ----------

def eval_sheets(spec):
    """(source, sid) pairs for 'cubicasa:60', 'cvcfp:40', 'commons', 'landgut'."""
    out = []
    for s in spec:
        name, _, n = s.partition(":")
        n = int(n) if n else None
        if name == "cubicasa":                           # test split in its order, two thirds architectural (as in the split)
            ids = [l.strip().strip("/") for l in (cubicasa.ROOT / "test.txt").read_text().splitlines() if l.strip()]
            n = n or 60
            arch = 2 * n // 3
            take = {"high_quality_architectural": arch, "high_quality": (n - arch) // 2, "colorful": n - arch - (n - arch) // 2}
            for style, k in take.items():
                out += [("cubicasa", i) for i in ids if i.startswith(style + "/")][:k]
        elif name == "cvcfp":
            ids = sorted({p.name.split("_gt_")[0] for p in (BENCH / "cvc-fp/ImagesGT").glob("*_gt_*.svg")})
            step = max(1, len(ids) // (n or len(ids)))
            out += [("cvcfp", i) for i in ids[::step][:n or len(ids)]]
        elif name == "commons":
            out += [("commons", i) for i in SAMPLES["commons"][:n]]
        elif name == "landgut":
            out += [("landgut", i) for i in ("s1", "s2", "s3")]
    return out


AREA_RE = re.compile(r"(\d{1,3})\s*[.,]\s*(\d{1,2})\s*m\s*[²2*]?", re.I)


def stamp_reference(rooms, texts):
    """CVC-FP reference independent of the cues: printed room areas ("12,56 m²", read by the pre-pass OCR) against the
    ground-truth room polygons (px²) they fall into; median over rooms with exactly one area stamp (at least 3)."""
    from shapely.geometry import Point
    ratios = []
    for poly in rooms:
        areas = [float(f"{m.group(1)}.{m.group(2)}") for t in texts if (m := AREA_RE.fullmatch(t["text"].strip())
                 or AREA_RE.search(t["text"]) if re.search(r"m\s*[²2*]", t["text"]) else None)
                 and poly.contains(Point((t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2))]
        if len(areas) == 1 and areas[0] >= 2:
            ratios.append(math.sqrt(poly.area / areas[0]))
    if len(ratios) < 3:
        return None, len(ratios), None
    r = np.array(ratios)
    return float(np.median(r)), len(r), float(np.median(np.abs(r / np.median(r) - 1)))


def eval_one(source, sid, engine, model, door_cue=None):
    """Cues and consensus for one sheet against its known scale (native px per m), without running the pipeline."""
    native_text, dpi, dpi_source, meta, truth, truth_note, cache = (), None, None, None, None, None, None
    if source == "landgut":                             # BBL plans: local only; outputs stay in the gitignored data folder
        import sheets
        d = sheets.native(sid)
        img, native_text, dpi, dpi_source = d["img"], d["text"], d["dpi"], d["dpi_source"]
        truth, truth_note = d["px_per_m"], d["method"]
        cache = DATA / "scale_eval_landgut" / "ocr"
    elif source == "cubicasa":
        img = rgb(image_path(source, sid))
        truth, truth_note = cubicasa.load(sid)["px_per_m"], "CubiCasa SVG dimension marks"
        if truth is None:                                # no metric labels in the SVG: the dataset's scaled images are 100 px/m
            truth, truth_note = 100.0, "CubiCasa F1_scaled.png convention (100 px/m)"
    elif source == "cvcfp":
        img, truth, truth_note, cvc_ref, _ = load_cvcfp(sid)
    else:
        img = rgb(image_path(source, sid))
    if source != "landgut":
        meta = meta_dpi(image_path(source, sid))
    t = time.time()
    pre = scale_prepass(f"{source}_{sid}", img, engine, model, dpi, dpi_source, meta, native_text, cache)
    if door_cue is not None:                            # a door search saved by an earlier --doors run
        scale.add_cue(pre, "door_widths", door_cue)
        pre["seconds"]["door_widths"] = door_cue.get("seconds", 0)
    con = pre["consensus"]
    door_ref = None
    if source == "cvcfp":                               # door symbols (0.85 m) are a rough reference: prefer the room areas
        door_ref = truth
        ref, n, spread = stamp_reference(cvc_ref["rooms"], pre["texts"])
        if ref:
            truth, truth_note = ref, f"printed room areas vs ground-truth room polygons ({n} rooms, MAD {spread * 100:.1f} %)"
    row = {"source": source, "sid": sid, "shape": list(img.shape[:2]), "truth": truth, "truth_note": truth_note,
           "door_reference": door_ref,
           "consensus": con, "seconds": {**pre["seconds"], "wall": round(time.time() - t, 1)}, "ocr": pre["ocr"],
           "cues": {}}
    for k, cue in pre["cues"].items():
        best = max(cue["candidates"], key=lambda c: c["weight"]) if cue["candidates"] else None
        row["cues"][k] = {"best": best, "candidates": cue["candidates"],
                          **{kk: vv for kk, vv in cue.items() if kk != "candidates"}}
    return row


def rel_err(v, truth):
    return None if not v or not truth else v / truth - 1


def eval_report(rows):
    """Coverage and accuracy per cue and for the consensus, per source."""
    lines = []
    cues = ["scale_note", "dimension_strings", "dimension_strings (precise)", "scale_bar", "door_widths", "consensus",
            "consensus (high/medium)"]

    def value(r, cue):
        if cue.startswith("consensus"):
            c = r["consensus"]
            return c["px_per_m"] if cue == "consensus" or c["confidence"] in ("high", "medium") else None
        best = r["cues"].get(cue.split(" ")[0], {}).get("best") or {}
        if cue.endswith("(precise)") and not best.get("precise"):
            return None
        return best.get("px_per_m")

    fails = []
    for source in sorted({r["source"] for r in rows}):
        rs = [r for r in rows if r["source"] == source]
        lines.append(f"\n{source}: {len(rs)} sheets (truth known for {sum(r['truth'] is not None for r in rs)})")
        lines.append(f"  {'cue':28s} {'coverage':>9s} {'med |err|':>9s} {'<=2%':>6s} {'<=5%':>6s} {'<=10%':>6s} {'>20%':>6s}")
        for cue in cues:
            vals = [(value(r, cue), r["truth"]) for r in rs]
            got = [v for v, _ in vals if v]
            errs = [abs(rel_err(v, t)) for v, t in vals if v and t]
            if not any(cue.split(" ")[0] in r["cues"] for r in rs) and not cue.startswith("consensus"):
                continue
            for r in rs:
                v = value(r, cue)
                if v and r["truth"] and abs(rel_err(v, r["truth"])) > 0.05 and "(" not in cue:
                    ev = r["consensus"] if cue == "consensus" else (r["cues"][cue]["best"] or {})
                    fails.append(f"  {source} {r['sid']} {cue}: {v:.1f} vs {r['truth']:.1f} ({rel_err(v, r['truth']) * 100:+.0f}%) "
                                 + (ev.get("evidence", "") if cue != "consensus" else
                                    f"{ev['confidence']}; agree {[a['cue'] for a in ev['agreeing']]}")[:150])
            cue = f"{cue:28s}"
            cov = f"{len(got)}/{len(vals)}"
            if errs:
                e = np.array(errs)
                lines.append(f"  {cue} {cov:>9s} {np.median(e) * 100:8.1f}% {np.mean(e <= .02) * 100:5.0f}% "
                             f"{np.mean(e <= .05) * 100:5.0f}% {np.mean(e <= .1) * 100:5.0f}% {np.mean(e > .2) * 100:5.0f}%")
            else:
                lines.append(f"  {cue} {cov:>9s}")
        confs = {}
        for r in rs:
            c = r["consensus"]["confidence"]
            e = rel_err(r["consensus"]["px_per_m"], r["truth"])
            confs.setdefault(c, []).append(e)
        for c in ("high", "medium", "low", "none"):
            if c in confs:
                es = [abs(e) for e in confs[c] if e is not None]
                lines.append(f"    consensus {c:6s}: {len(confs[c]):3d} sheets"
                             + (f", within 5%: {sum(e <= .05 for e in es)}/{len(es)}, median |err| {np.median(es) * 100:.1f}%" if es else ""))
        if any(r.get("door_reference") for r in rs):     # CVC-FP: the second, rougher reference
            for name, key in (("room-area reference", "truth"), ("door-symbol reference", "door_reference")):
                es = [abs(rel_err(r["consensus"]["px_per_m"], r[key])) for r in rs
                      if r["consensus"]["px_per_m"] and r.get(key) and r["consensus"]["confidence"] in ("high", "medium")
                      and (key == "door_reference" or r["truth"] != r["door_reference"])]
                if es:
                    lines.append(f"    consensus (high/medium) vs {name}: {len(es)} sheets, median |err| "
                                 f"{np.median(es) * 100:.1f}%, within 5%: {sum(e <= .05 for e in es)}")
            both = [(r["truth"], r["door_reference"]) for r in rs if r.get("door_reference") and r["truth"] != r["door_reference"]]
            if both:
                d = [t / dr - 1 for t, dr in both]
                lines.append(f"    room-area / door-symbol reference: median {np.median(d) * 100:+.1f}%, "
                             f"5-95 % {np.percentile(d, 5) * 100:+.1f}..{np.percentile(d, 95) * 100:+.1f}% ({len(d)} sheets)")
        flagged = sum(bool(r["consensus"]["flags"]) for r in rs)
        wrong = [r for r in rs if r["truth"] and r["consensus"]["px_per_m"] and abs(rel_err(r["consensus"]["px_per_m"], r["truth"])) > .05]
        lines.append(f"    flagged: {flagged}/{len(rs)}; wrong by >5%: {len(wrong)}, of which flagged or low: "
                     f"{sum(bool(r['consensus']['flags']) or r['consensus']['confidence'] == 'low' for r in wrong)}")
        secs = [r["seconds"].get("ocr", 0) or 0 for r in rs]
        lines.append(f"    seconds per sheet: OCR median {np.median(secs):.1f}, cues median "
                     f"{np.median([r['seconds'].get('text_cues', 0) for r in rs]):.2f}"
                     + (f", door search median {np.median([r['seconds'].get('door_widths', 0) for r in rs]):.1f}"
                        if any('door_widths' in r['seconds'] for r in rs) else ""))
    if fails:
        lines += ["\nerrors above 5 %:"] + fails
    return "\n".join(lines)


def scale_eval(args):
    doors = "--doors" in args
    reuse = "--reuse-doors" in args                     # recompute the text cues, take the door cue from a --doors run
    spec = [a for a in args if not a.startswith("--")]
    SCALE_OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    model = load_model(DATA / "model/segmenter.pt") if doors else None
    engine = pipeline.ocr_engine()
    rows = []
    for source, sid in eval_sheets(spec):
        out_dir = DATA / "scale_eval_landgut" if source == "landgut" else SCALE_OUT
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{source}_{sid.replace('/', '_')}{'_doors' if doors or reuse else ''}.json"
        door_cue = None
        if reuse:
            if not path.exists():
                continue
            saved = json.loads(path.read_text(encoding="utf-8"))
            rows_ = saved["cues"]["door_widths"].get("rows")
            cands = ([scale.door_choice(rows_)[2]] if rows_ else saved["cues"]["door_widths"]["candidates"])
            door_cue = {"candidates": cands, "rows": rows_, "seconds": saved["seconds"].get("door_widths", 0)}
        try:
            row = eval_one(source, sid, engine, model, door_cue)
        except Exception as ex:                          # keep going; report at the end
            print(f"FAIL {source} {sid}: {type(ex).__name__}: {ex}", flush=True)
            continue
        path.write_text(json.dumps(row, ensure_ascii=False, default=float), encoding="utf-8")
        rows.append(row)
        c = row["consensus"]
        e = rel_err(c["px_per_m"], row["truth"])
        print(f"{source:9s} {sid:36s} truth {row['truth'] or 0:8.2f}  consensus {c['px_per_m'] or 0:8.2f} "
              f"({c['confidence']:6s}) err {'' if e is None else f'{e * 100:+6.1f}%':>7s}  "
              + "  ".join(f"{k[:4]}={(v['best'] or {}).get('px_per_m', '-')}" for k, v in row["cues"].items())
              + f"  {row['seconds'].get('wall')} s", flush=True)
    report = eval_report(rows)
    print(report)
    tag = "_".join(a.replace(":", "") for a in spec) + ("_doors" if doors or reuse else "")
    pub = [r for r in rows if r["source"] != "landgut"]
    if pub:
        (SCALE_OUT / f"report_{tag}.txt").write_text(eval_report(pub), encoding="utf-8")
    return rows


if __name__ == "__main__":
    if sys.argv[1:2] == ["scale-eval"]:
        scale_eval(sys.argv[2:])
        sys.exit()
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    torch.set_num_threads(12)
    model = load_model(DATA / "model/segmenter.pt")
    engine = pipeline.ocr_engine()
    for source, ids in SAMPLES.items():
        if only and source not in only:
            continue
        for sid in ids:
            try:
                e, sheet = run(source, sid, model, engine)
            except Exception as ex:                          # keep going; report at the end
                print(f"FAIL {source} {sid}: {type(ex).__name__}: {ex}", flush=True)
                continue
            print(f"{e['id']:55s} {e['px_per_m']:7.1f} px/m  {len(sheet.rooms):3d} rooms  {e['scores']}  {e['seconds']} s | {e['scale_method'][:70]}", flush=True)
