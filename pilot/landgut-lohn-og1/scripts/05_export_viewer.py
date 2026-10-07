"""Package sheet images and results for viewer.html into data/out/viewer-data.js (gitignored).

Images are embedded as data URIs so the viewer also works when opened directly from disk.
Confidence uses only signals available in production (no reference); accuracy uses the reference.
"""
import base64
import importlib
import io
import json

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from shapely.ops import unary_union

from common import BUILDING, CAL, OUT, ROOMS, WALLS, iou, main_component, mask_to_geometry

hybrid = importlib.import_module("03_score_hybrid_2005")
scan = importlib.import_module("04_segment_scan_0056")
REF = {name: (la, g) for name, la, g in ROOMS}
REF_WALLS = unary_union(WALLS)


def polygons_with_holes(geom, tol=0.02, min_area=0.02):
    """[[exterior, hole, ...], ...] with rings as [[x, y], ...]."""
    geom = geom.simplify(tol)
    out = []
    for p in getattr(geom, "geoms", [geom]):
        if p.area < min_area:
            continue
        ring = lambda r: [[round(x, 3), round(y, 3)] for x, y in r.coords[:-1]]
        out.append([ring(p.exterior)] + [ring(h) for h in p.interiors if abs(Polygon(h).area) >= min_area])
    return out


def walls_entry(geom):
    return {"polygons": polygons_with_holes(geom), "area": round(geom.area, 2),
            "refArea": round(REF_WALLS.area, 2), "iou": round(iou(geom, REF_WALLS), 2)}


def rings(geom, tol=0.01):
    """Exterior rings of a (Multi)Polygon as [[x, y], ...] lists, simplified."""
    geom = geom.simplify(tol)
    parts = getattr(geom, "geoms", [geom])
    return [[[round(x, 3), round(y, 3)] for x, y in p.exterior.coords[:-1]] for p in parts if p.area > 0.05]


def label_point(geom):
    p = geom.representative_point()
    return [round(p.x, 3), round(p.y, 3)]


def data_uri(image, fmt="PNG", **kw):
    buf = io.BytesIO()
    image.save(buf, fmt, **kw)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def stamp_confidence(area, read):
    if read is None:
        return "low", "Stamp area not readable: no cross-check possible"
    err = abs(area - read) / read * 100
    level = "high" if err <= 5 else "medium" if err <= 15 else "low"
    return level, f"Polygon area deviates {err:.1f}% from the stamp ({read:.2f} m²)"


def sheet_2005():
    rooms, rejected = hybrid.labelled_rooms()
    out = []
    for name, (p, read) in rooms.items():
        la, g = REF[name]
        q = p.buffer(hybrid.SNAP_M, join_style=2)
        conf, reason = stamp_confidence(p.area, read)
        conf_s, reason_s = stamp_confidence(q.area, read)
        out.append({
            "name": name, "sub": None, "label": label_point(p),
            "polygons": rings(p), "polygonsSnapped": rings(q),
            "area": round(p.area, 2), "areaSnapped": round(q.area, 2), "stamp": read, "stampRef": la,
            "iou": round(iou(p, g), 2), "iouSnapped": round(iou(q, g), 2),
            "conf": conf, "reason": reason, "confSnapped": conf_s, "reasonSnapped": reason_s,
        })
    # image extent in plan metres: crop of the 400 dpi render
    X0, Y0, X1, Y1 = 467, 1107, 1762, 2073
    s, (ox, oy), k = CAL["metresPerPt"], CAL["originPt"], 72.0 / 400
    extent = [((X0 * k) - ox) * s, (oy - Y1 * k) * s, ((X1 * k) - ox) * s, (oy - Y0 * k) * s]
    image = Image.open(OUT / "s1_main_crop_400dpi.png").convert("L")
    cands, wall_px = hybrid.seg.run(hybrid.SEAL_PX, hybrid.KEEP_PX, masks=True)
    walls = mask_to_geometry(wall_px, hybrid.seg.px_to_plan, min_area=0.05)
    # stair void ("Luftraum", candidate 17 in the VLM labelling): shown separately, not floor area
    void = {i: p for i, (_, p, _) in enumerate(cands, 1)}[17]
    voids = [{"name": "Luftraum (void)", "polygons": rings(void), "area": round(void.area, 2),
              "note": "Open void over the stair: not floor area; deducted from GF if over 5 m² (CAD-Richtlinie)"}]
    return {
        "id": "s1", "title": "2005 CAD print (S1)",
        "image": data_uri(image), "extent": [round(v, 3) for v in extent],
        "rooms": out, "rejected": [r for p in rejected for r in rings(p)], "walls": walls_entry(walls), "voids": voids,
        "confidenceNote": "Stamp-area check, available in production: high ≤ 5 %, medium ≤ 15 %, low > 15 % or stamp unreadable.",
        "accuracyNote": "IoU with the reference room of the same name (evaluation only).",
    }


def sheet_scan():
    cands, bx, outline = scan.run(15, 0.8)
    poly = {i: p for i, (_, p, _) in enumerate(cands, 1)}
    out = []
    for i, (hist, now) in scan.LABELS.items():
        p = poly[i]
        rect = p.area / p.minimum_rotated_rectangle.area
        if "merged" in hist:
            conf, reason = "low", "VLM review: region merges several rooms"
        elif hist.startswith("("):
            conf, reason = "medium", "No room label found on the plan"
        elif rect >= 0.9:
            conf, reason = "high", f"Labelled room with a regular outline (rectangularity {rect:.2f}); no area on the plan to cross-check"
        else:
            conf, reason = "medium", f"Labelled room, irregular outline (rectangularity {rect:.2f})"
        target = REF[now][1] if now else max((g for _, (_, g) in REF.items()), key=lambda g: iou(p, g))
        out.append({
            "name": hist, "sub": f"2005: {now}" if now else "changed or merged since; best 2005 overlap shown",
            "label": label_point(p), "polygons": rings(p), "polygonsSnapped": None,
            "area": round(p.area, 2), "areaSnapped": None, "stamp": None, "stampRef": None,
            "iou": round(iou(p, target), 2), "iouSnapped": None, "conf": conf, "reason": reason,
        })
    # image: crop around the building at half resolution, downscaled, extent in plan metres
    m = scan.M_PER_PX
    dx, dy = BUILDING.bounds[0] - bx.bounds[0], BUILDING.bounds[1] - bx.bounds[1]
    x, y, w, h = cv2.boundingRect(outline)
    pad = 60
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(scan.IMG.shape[1], x + w + pad), min(scan.IMG.shape[0], y + h + pad)
    crop = Image.fromarray(scan.IMG[y0:y1, x0:x1])
    f = 2400 / max(crop.size)
    crop = crop.resize((int(crop.size[0] * f), int(crop.size[1] * f)), Image.LANCZOS)
    extent = [x0 * m + dx, -y1 * m + dy, x1 * m + dx, -y0 * m + dy]
    walls = mask_to_geometry(scan.wall_mask(15), lambda x, y: (x * m + dx, -y * m + dy), min_area=0.05)
    return {
        "id": "s2", "title": "Historical scan 1:50 (S2)",
        "image": data_uri(crop, "JPEG", quality=85), "extent": [round(v, 3) for v in extent],
        "rooms": out, "rejected": [], "walls": walls_entry(walls), "voids": [],
        "confidenceNote": "No areas on this plan. Low = region merges several rooms (VLM review); medium = no label or irregular outline; high = labelled room with a regular outline.",
        "accuracyNote": "IoU with the 2005 counterpart. Rooms changed since the scan score low by design: the plan shows an older state.",
    }


if __name__ == "__main__":
    data = {
        "reference": {"rooms": [{"name": n, "polygons": rings(g)} for n, (_, g) in REF.items()],
                      "walls": [r for g in WALLS for r in rings(g)]},
        "sheets": [sheet_2005(), sheet_scan()],
    }
    target = OUT / "viewer-data.js"
    target.write_text("window.VIEWER_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")
    for sh in data["sheets"]:
        print(sh["id"], sh["title"], "rooms", len(sh["rooms"]), "confidence", {c: sum(r["conf"] == c for r in sh["rooms"]) for c in ("high", "medium", "low")})
        w = sh["walls"]
        print(f"   walls: {len(w['polygons'])} polygon(s), {w['area']} m2 vs reference {w['refArea']} m2, IoU {w['iou']}")
