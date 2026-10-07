"""Landgut Lohn 1. OG test sheets: load each one at the working resolution (pipeline stage 1, per input class).

The plan files are non-public and stay in the v1 pilot's gitignored data folder. No parameter is tuned on them.
"""
import json

import cv2
import numpy as np
import pymupdf
from PIL import Image

from common import PX_PER_M, V1
from fpx.config import DEFAULT
from fpx.model import Sheet

Image.MAX_IMAGE_PIXELS = None
INPUTS = V1 / "data/inputs"
CAL = json.loads((V1 / "data/reference/plan_calibration.json").read_text(encoding="utf-8"))["sheets"]["og1"]
OCR_PX_PER_M = DEFAULT.ocr_px_per_m                     # resolution for OCR on scans (lettering ~0.25 m -> ~40 px)


def load_s1():
    """Vector PDF (2005 CAD print): render the hand-drawn region of interest at exactly PX_PER_M, keep native text."""
    path = INPUTS / "2051-AA-1. OG 2005-136979-ARCH-1-OG-100.pdf"
    page = pymupdf.open(path)[0]
    roi = pymupdf.Rect(84.06, 199.26, 317.16, 373.14)            # main building, drawn on the preview (as in v1)
    zoom = PX_PER_M * CAL["metresPerPt"]                          # px per pt
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=roi, colorspace=pymupdf.csRGB)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
    ox, oy = CAL["originPt"]
    origin = ((roi.x0 - ox) * CAL["metresPerPt"], (oy - roi.y0) * CAL["metresPerPt"])
    text = []
    for b in page.get_text("rawdict")["blocks"]:
        for ln in b.get("lines", []):
            for sp in ln["spans"]:
                for ch in sp["chars"]:
                    x0, y0, x1, y1 = ch["bbox"]
                    if ch["c"].strip() and roi.contains(pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)):
                        text.append((ch["c"], (x0 - roi.x0) * zoom, (y0 - roi.y0) * zoom, (x1 - roi.x0) * zoom,
                                     (y1 - roi.y0) * zoom, sp["size"] * zoom, tuple(ln["dir"])))
    oz = OCR_PX_PER_M * CAL["metresPerPt"]
    opix = page.get_pixmap(matrix=pymupdf.Matrix(oz, oz), clip=roi, colorspace=pymupdf.csGRAY)
    ocr = np.frombuffer(opix.samples, np.uint8).reshape(opix.height, opix.width).copy()
    return Sheet("s1", "2005 CAD print (vector PDF)", path.name, "vector PDF with text layer", img, origin,
                 {"value": f"{CAL['equivalentScale']} (1:100 plan printed reduced)", "metresPerPt": CAL["metresPerPt"],
                  "method": "dimension strings on the sheet (calibration from the reconstruction project)"},
                 native_text=text, ocr_img=ocr)


def load_scan(sid, file, title):
    """Raster scan at 1:50: resample to PX_PER_M (working) and OCR_PX_PER_M (text)."""
    path = INPUTS / file
    im = Image.open(path)
    dpi = float(im.info.get("dpi", (400, 400))[0])
    native = np.asarray(im.convert("L"))
    px_per_m = dpi / 0.0254 / 50                                   # 1:50 from the title block
    work = cv2.resize(native, None, fx=PX_PER_M / px_per_m, fy=PX_PER_M / px_per_m, interpolation=cv2.INTER_AREA)
    ocr = cv2.resize(native, None, fx=OCR_PX_PER_M / px_per_m, fy=OCR_PX_PER_M / px_per_m, interpolation=cv2.INTER_AREA)
    return Sheet(sid, title, file, "raster scan (1-bit)" if im.mode == "1" else "raster scan",
                 cv2.cvtColor(work, cv2.COLOR_GRAY2RGB), (0.0, 0.0),
                 {"value": "1:50", "dpi": dpi, "method": "title block (1:50) and scan resolution; to be confirmed"},
                 ocr_img=ocr)


SCANS = {"s2": "2051-AA-1. OG-54467-0056.tif", "s3": "2051-AA-1. OG-54477-0066.tif"}


def native(sid, dpi=1200):
    """A sheet at its native resolution for the scale pre-pass (fpx.scale), next to the loaders above (which are
    unchanged). S1: the whole PDF page rendered at dpi (high, for the small lettering of a reduced print) with the
    native text in image pixels; S2, S3: the scan with its dpi. px_per_m is what the loader assumes, in native pixels per metre
    (S1: calibration from the reconstruction project; S2, S3: 1:50 from the title block)."""
    from fpx.text import merge_chars
    if sid == "s1":
        page = pymupdf.open(INPUTS / "2051-AA-1. OG 2005-136979-ARCH-1-OG-100.pdf")[0]
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
        chars = []
        for b in page.get_text("rawdict")["blocks"]:
            for ln in b.get("lines", []):
                for sp in ln["spans"]:
                    for ch in sp["chars"]:
                        if ch["c"].strip():
                            x0, y0, x1, y1 = (v * zoom for v in ch["bbox"])
                            chars.append((ch["c"], x0, y0, x1, y1, sp["size"] * zoom, tuple(ln["dir"])))
        return {"img": img, "dpi": float(dpi), "dpi_source": "render", "text": merge_chars(chars),
                "px_per_m": zoom / CAL["metresPerPt"], "method": "calibration from dimension strings (reconstruction project)"}
    im = Image.open(INPUTS / SCANS[sid])
    dpi = float(im.info.get("dpi", (400, 400))[0])
    return {"img": np.asarray(im.convert("L")), "dpi": dpi, "dpi_source": "scan", "text": [],
            "px_per_m": dpi / 0.0254 / 50, "method": "title block (1:50) and scan resolution"}


def load_all():
    return [load_s1(),
            load_scan("s2", "2051-AA-1. OG-54467-0056.tif", "Historical plan, poché walls (scan)"),
            load_scan("s3", "2051-AA-1. OG-54477-0066.tif", "Survey plan, outlined walls (scan)")]
