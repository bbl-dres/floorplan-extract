"""Run pipeline stages 0-10 on the Landgut Lohn test sheets, locally (BBL plans never leave this machine).

    python run_pipeline.py [s1 s2 s3]

Needs data/model/segmenter.pt (from train.py). Writes data/out/<sheet>.json, .dxf and overlay images.
Thresholds: fpx.config.Config; pass --config file.json to override them. --model PATH and --out DIR run another
segmenter (e.g. data/model-v2/segmenter.pt) into its own folder.
"""
import sys
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from common import DATA
from fpx import DEFAULT, Config, pipeline
from fpx.segment import load_model
from sheets import load_all

OUT = DATA / "out"
COLOURS = np.array([[255, 255, 255], [40, 40, 40], [230, 40, 40], [40, 120, 230], [40, 170, 40], [200, 40, 200]], np.uint8)


def overlay(sheet):
    """Segmenter classes over the sheet, plus room outlines, openings and passages."""
    img = sheet.img.copy()
    seg = COLOURS[sheet.label]
    m = sheet.label > 0
    img[m] = (0.35 * img[m] + 0.65 * seg[m]).astype(np.uint8)
    rng = np.random.default_rng(1)
    for r in sheet.rooms:
        c = tuple(int(v) for v in rng.integers(60, 220, 3))
        cv2.polylines(img, [np.asarray(r["poly"].exterior.coords, np.int32)], True, c, 2)
        x, y = r["poly"].representative_point().coords[0]
        cv2.putText(img, f"{r['id']} {r['name'] or ''}"[:28], (int(x) - 30, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    for o in sheet.openings:
        if o["kind"] == "passage":
            a, b = o["line"]
            cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (255, 140, 0), 3)
    return img


if __name__ == "__main__":
    args = sys.argv[1:]
    cfg = DEFAULT
    model_path = DATA / "model/segmenter.pt"
    if "--config" in args:
        i = args.index("--config")
        cfg = Config.from_file(args[i + 1])
        del args[i:i + 2]
    if "--model" in args:                               # e.g. data/model-v2/segmenter.pt
        i = args.index("--model")
        model_path = Path(args[i + 1])
        del args[i:i + 2]
    if "--out" in args:                                 # keep outputs of another model apart; local data folder only
        i = args.index("--out")
        OUT = Path(args[i + 1])
        del args[i:i + 2]
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(args)
    torch.set_num_threads(8)
    model = load_model(model_path)
    engine = pipeline.ocr_engine()
    for sheet in load_all():
        if only and sheet.id not in only:
            continue
        t0 = time.time()
        times = pipeline.run(sheet, model=model, engine=engine, cfg=cfg, out_dir=OUT)
        times = {pipeline.LABELS[k]: v for k, v in times.items()}
        cv2.imwrite(str(OUT / f"{sheet.id}_overlay.png"), cv2.cvtColor(overlay(sheet), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT / f"{sheet.id}_sheet.jpg"), cv2.cvtColor(sheet.img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        kinds = {k: sum(o["kind"] == k for o in sheet.openings) for k in ("door", "exterior door", "window", "passage")}
        print(f"{sheet.id}: {sheet.img.shape[1]}x{sheet.img.shape[0]} px, style '{sheet.meta['triage']['graphical_style']}', "
              f"skew {sheet.meta['triage']['skew_deg']} deg | {len(sheet.wall_segments)} wall segments, openings {kinds}, "
              f"{len(sheet.stairs)} stairs, {len(sheet.rooms)} rooms, GF {sheet.gf_area:.1f} m², "
              f"{len(sheet.connectivity.edges)} connections, {len(sheet.qa)} QA issues | {time.time() - t0:.0f} s {times}")
        for r in sheet.rooms:
            print(f"   {r['id']} {cfg.m2(r['poly'].area):6.1f} m²  {r['confidence']:6s} name={r['name']!r} stamp={r['area_stamp']} usage={r['usage']}")
