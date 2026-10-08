"""Stage 0, triage and input quality, and stage 1, preprocessing."""
import cv2
import numpy as np

from .config import DEFAULT


def triage(sheet, cfg=DEFAULT):
    g = cv2.cvtColor(sheet.img, cv2.COLOR_RGB2GRAY)
    ink = g < 128
    hsv = cv2.cvtColor(sheet.img, cv2.COLOR_RGB2HSV)
    colour = float(hsv[..., 1][ink].mean()) > 40 if ink.any() else False
    k = int(cfg.px(cfg.solid_ink_width)) | 1
    solid = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    solid_share = float(solid.sum() / max(ink.sum(), 1))
    lines = cv2.HoughLinesP(cv2.Canny(g, 50, 150), 1, np.pi / 1800, 200, minLineLength=int(cfg.px(cfg.skew_min_line)), maxLineGap=5)
    skew = 0.0
    if lines is not None:
        ang = np.degrees(np.arctan2(lines[:, 0, 3] - lines[:, 0, 1], lines[:, 0, 2] - lines[:, 0, 0]))
        ang = (ang + 45) % 90 - 45
        ln = np.hypot(lines[:, 0, 3] - lines[:, 0, 1], lines[:, 0, 2] - lines[:, 0, 0])
        near = np.abs(ang) < 5
        skew = float(np.average(ang[near], weights=ln[near])) if near.any() else 0.0
    style = "colour" if colour else "solid (poché) walls" if solid_share > cfg.solid_ink_share else "outlined walls"
    sheet.meta["triage"] = {
        "input_class": sheet.input_class,
        "graphical_style": style,
        "solid_ink_share": round(solid_share, 3),
        "skew_deg": round(skew, 2),
        "text_layer": ("partial: dimensions and fixture words only; room stamps are outlined, so OCR is needed"
                       if sheet.native_text else "none (raster)"),
        "checks": {
            "one floor per sheet": True,
            "scale reference": sheet.scale["method"],
            "resolution": f"{cfg.px_per_m} px/m working, OCR at {cfg.ocr_px_per_m} px/m",
            "flat scan": abs(skew) < cfg.flat_scan_max_deg,
            "real 2D plan": True,
            "overlays hide walls": colour,
        },
        "route": "raster pipeline (stages 1-10)",
    }
    return sheet.meta["triage"]


def preprocess(sheet, cfg=DEFAULT):
    skew = sheet.meta["triage"]["skew_deg"]
    if abs(skew) > cfg.deskew_min_deg:                   # deskew working and OCR images about their centres
        for attr in ("img", "ocr_img"):
            im = getattr(sheet, attr)
            if im is None:
                continue
            h, w = im.shape[:2]
            R = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
            setattr(sheet, attr, cv2.warpAffine(im, R, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255,) * 3))
            if attr == "img":
                sheet.meta["deskew"] = {"deg": skew, "matrix": [[round(v, 9) for v in row] for row in R.tolist()],
                                        "note": "affine matrix from original to deskewed working pixels; "
                                                "all coordinates are in the deskewed frame"}
        sheet.meta["deskewed_deg"] = skew
