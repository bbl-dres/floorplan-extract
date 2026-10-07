"""Mode C geometry on the rendered 2005 sheet: classical room segmentation, scored against the reference.

Walls are drawn with the thickest pens (0.705/0.987 pt = 3.9/5.5 px at 400 dpi); a morphological opening
keeps them and drops door swings, hatching, lettering and dimension lines. Door gaps are closed by eroding
the free space (seeds), then the regions grow back with a watershed.
"""
import json

import cv2
import numpy as np
from shapely.geometry import Polygon

from common import CAL, OUT, ROOMS, best_match

DPI = 400
BIG = 81                                       # closing kernel for the building mask (px)
WALL_HALF_PX = 22                              # enclosed strips with half-width < 22 px (~0.4 m) count as wall body
X0, Y0, X1, Y1 = 467, 1107, 1762, 2073         # main-building crop in the 400 dpi render
IMG = cv2.imread(str(OUT / "s1_pdf2005_400dpi.png"), cv2.IMREAD_GRAYSCALE)[Y0:Y1, X0:X1]
INK = (IMG < 140).astype(np.uint8)


def px_to_plan(x, y):
    """Crop pixel -> plan metres (calibration of the 2005 sheet)."""
    k, s = 72.0 / DPI, CAL["metresPerPt"]
    ox, oy = CAL["originPt"]
    return ((X0 + x) * k - ox) * s, (oy - (Y0 + y) * k) * s


def run(seal_px, keep_px=5, masks=False):
    """Return candidate rooms [(label, polygon in plan metres, contour in px)].
    With masks=True also return the wall mask: building pixels not covered by any candidate room."""
    thick = cv2.morphologyEx(INK, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (keep_px, keep_px)))
    big = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (BIG, BIG)))
    cnts, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bmask = np.zeros_like(INK)
    cv2.drawContours(bmask, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
    free = ((thick == 0) & (bmask == 1)).astype(np.uint8)
    seeds = cv2.erode(free, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * seal_px + 1, 2 * seal_px + 1)))
    n, markers = cv2.connectedComponents(seeds)
    markers = markers.astype(np.int32)
    markers[(free == 1) & (seeds == 0)] = 0
    markers[free == 0] = n + 1
    ws = cv2.watershed(cv2.cvtColor(IMG, cv2.COLOR_GRAY2BGR), markers)
    cands, covered = [], np.zeros_like(INK)
    for lab in range(1, n):
        reg = ((ws == lab) & (free == 1)).astype(np.uint8)
        if reg.sum() < 2500:
            continue
        cs, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        if len(c) < 3:
            continue
        p = Polygon([px_to_plan(x, y) for x, y in c[:, 0, :]]).buffer(0)
        if p.area >= 0.8:
            cands.append((lab, p, c))
            covered |= reg
    if not masks:
        return cands
    # wall bodies = wall-pen lines + enclosed free-space strips narrower than ~0.8 m (between double outlines)
    n2, comp, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 3)
    peak = np.zeros(n2, np.float32)
    np.maximum.at(peak, comp.ravel(), dist.ravel())
    narrow = (peak[comp] < WALL_HALF_PX) & (free == 1)
    walls = (thick == 1) | narrow
    # restrict to the footprint: filled outer contour of the largest wall-pen network (exterior wall outline);
    # bridge gaps up to window width (~1.3 m) first: window openings split the exterior wall into piers
    net = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, np.ones((71, 71), np.uint8))
    nt, tl, ts, _ = cv2.connectedComponentsWithStats(net, connectivity=8)
    big_net = (tl == 1 + int(np.argmax(ts[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cs, _ = cv2.findContours(big_net, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    footprint = np.zeros_like(INK)
    cv2.drawContours(footprint, cs, -1, 1, -1)
    walls = (walls & (footprint == 1)).astype(np.uint8)
    return cands, walls


def score(cands):
    pairs = [(lab, p) for lab, p, _ in cands]
    return [(name, label_area, ref.area, *best_match(pairs, ref)) for name, label_area, ref in ROOMS]


if __name__ == "__main__":
    print("parameter sweep (tuned on this sheet - optimistic)")
    for keep in (3, 4, 5):
        for seal in (14, 20, 26, 32):
            rows = score(run(seal, keep))
            ious = [r[5] for r in rows]
            print(f"keep={keep}px seal={seal:2d}px  rooms IoU>=0.5: {sum(i >= 0.5 for i in ious):2d}/15  mean IoU {np.mean(ious):.2f}")
    cands = run(32, 5)
    ids = {lab: i for i, (lab, _, _) in enumerate(cands, 1)}
    print(f"\nbest setting keep=5px seal=32px: {len(cands)} candidates")
    for name, la, ra, lab, p, i in score(cands):
        print(f"{name:<22} stamp {la:6.2f}  CV {p.area:6.2f}  IoU {i:.2f}  err {(p.area - la) / la * 100:+6.1f}%  #{ids[lab]}")
    # numbered overlay for the VLM labelling step (mode C, semantics)
    vis = cv2.cvtColor(IMG, cv2.COLOR_GRAY2BGR)
    rng = np.random.default_rng(7)
    for _, _, c in cands:
        ov = vis.copy()
        cv2.drawContours(ov, [c], -1, [int(v) for v in rng.integers(60, 230, 3)], -1)
        vis = cv2.addWeighted(ov, 0.40, vis, 0.60, 0)
    for i, (_, _, c) in enumerate(cands, 1):
        m = cv2.moments(c)
        cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
        cv2.putText(vis, str(i), (cx - 12, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 5)
        cv2.putText(vis, str(i), (cx - 12, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)
    cv2.imwrite(str(OUT / "s1_modeC_candidates.png"), vis)
    json.dump({i: round(p.area, 2) for i, (_, p, _) in enumerate(cands, 1)}, open(OUT / "s1_modeC_candidates.json", "w"))
