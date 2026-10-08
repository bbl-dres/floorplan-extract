"""Run pipeline stages 0-10 on the Landgut Lohn test sheets, locally (BBL plans never leave this machine).

    python run_pipeline.py [s1 s2 s3] [--model PATH] [--out DIR] [--config file.json]

Needs the segmenter (default model v2, common.MODEL). Writes data/out/<sheet>.json, .dxf and overlay images.
Thresholds: fpx.config.Config; pass --config file.json to override them. --model PATH and --out DIR run another
segmenter (e.g. data/model/segmenter.pt, v1) into its own folder.
"""
import time
from pathlib import Path

import cv2

from common import DATA
from fpeval.cli import cli, setup
from fpeval.datasets.landgut import load_all
from fpeval.raster import overlay
from fpx import pipeline
from fpx.segment import load_model

OUT = DATA / "out"


def main(argv=None):
    ap = cli(__doc__.split("\n")[0], out=OUT)
    ap.add_argument("sheets", nargs="*", help="sheet ids to run (default: all)")
    a = ap.parse_args(argv)
    cfg = setup(a)
    out = Path(a.out)                                   # keep outputs of another model apart; local data folder only
    out.mkdir(parents=True, exist_ok=True)
    only = set(a.sheets)
    model = load_model(a.model)
    engine = pipeline.ocr_engine()
    for sheet in load_all():
        if only and sheet.id not in only:
            continue
        t0 = time.time()
        times = pipeline.run(sheet, model=model, engine=engine, cfg=cfg, out_dir=out)
        times = {pipeline.LABELS[k]: v for k, v in times.items()}
        cv2.imwrite(str(out / f"{sheet.id}_overlay.png"), cv2.cvtColor(overlay(sheet), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{sheet.id}_sheet.jpg"), cv2.cvtColor(sheet.img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        kinds = {k: sum(o["kind"] == k for o in sheet.openings) for k in ("door", "exterior door", "window", "passage")}
        print(f"{sheet.id}: {sheet.img.shape[1]}x{sheet.img.shape[0]} px, style '{sheet.meta['triage']['graphical_style']}', "
              f"skew {sheet.meta['triage']['skew_deg']} deg | {len(sheet.wall_segments)} wall segments, openings {kinds}, "
              f"{len(sheet.stairs)} stairs, {len(sheet.rooms)} rooms, GF {sheet.gf_area:.1f} m², "
              f"{len(sheet.connectivity.edges)} connections, {len(sheet.qa)} QA issues | {time.time() - t0:.0f} s {times}")
        for r in sheet.rooms:
            print(f"   {r['id']} {cfg.m2(r['poly'].area):6.1f} m²  {r['confidence']:6s} name={r['name']!r} stamp={r['area_stamp']} usage={r['usage']}")


if __name__ == "__main__":
    main()
