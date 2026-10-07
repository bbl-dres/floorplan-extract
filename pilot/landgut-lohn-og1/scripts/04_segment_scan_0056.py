"""Mode C on the historical scan 0056 ("Grundriss vom ersten Stock, Masstab 1:50"): solid black walls,
no door swings, calligraphic labels. Scale from the title block (1:50 at 400 dpi); registered to the
reference by translation only. LABELS is the VLM reading of out/s2_modeC_candidates.png.
"""
import cv2
import numpy as np
from PIL import Image
from shapely import affinity
from shapely.geometry import Polygon

from common import BUILDING, OUT, ROOMS, SCAN_0056, best_match

Image.MAX_IMAGE_PIXELS = None
F = 2                                                   # work at half resolution
FULL = np.array(Image.open(SCAN_0056).convert("L"))
IMG = cv2.resize(FULL, (FULL.shape[1] // F, FULL.shape[0] // F), interpolation=cv2.INTER_AREA)
M_PER_PX = 0.0254 / 400 * 50 * F                        # 1:50 at 400 dpi, half resolution
INK = (IMG < 128).astype(np.uint8)

# VLM reading of the numbered overlay: historical name, and the 2005 room it corresponds to (if unchanged)
LABELS = {1: ("Herren-Zimmer", None), 2: ("Treppe + Vestibule + corridors (merged)", None),
          3: ("Antichambre", None), 4: ("Diensten-Zimmer", "Kreidolf-Zimmer"), 5: ("Toilette", "Bad/WC NW"),
          6: ("Halle", "Balkon"), 7: ("Bad", None), 8: ("Zimmer", "Biedermeier-Zimmer"),
          9: ("Schlafzimmer", "Damenzimmer"), 10: ("Zimmer + Boudoir (merged)", None), 11: ("(no label)", "Bad/WC SW")}


def run(wall_k=15, seal_m=0.8):
    walls = cv2.morphologyEx(INK, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (wall_k, wall_k)))
    big = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (301, 301)))
    cnts, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = max(cnts, key=cv2.contourArea)
    bm = np.zeros_like(INK)
    cv2.drawContours(bm, [outline], -1, 1, -1)
    free = ((walls == 0) & (bm == 1)).astype(np.uint8)
    r = int(seal_m / M_PER_PX)
    seeds = cv2.erode(free, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n, mk = cv2.connectedComponents(seeds)
    mk = mk.astype(np.int32)
    mk[(free == 1) & (seeds == 0)] = 0
    mk[free == 0] = n + 1
    ws = cv2.watershed(cv2.cvtColor(IMG, cv2.COLOR_GRAY2BGR), mk)
    bx = Polygon([(x * M_PER_PX, -y * M_PER_PX) for x, y in outline[:, 0, :]])
    dx, dy = BUILDING.bounds[0] - bx.bounds[0], BUILDING.bounds[1] - bx.bounds[1]
    cands = []
    for lab in range(1, n):
        reg = ((ws == lab) & (free == 1)).astype(np.uint8)
        cs, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        p = Polygon([(x * M_PER_PX, -y * M_PER_PX) for x, y in c[:, 0, :]]).buffer(0)
        if p.area >= 1.0:
            cands.append((lab, affinity.translate(p, dx, dy), c))
    return cands, bx, outline


def wall_mask(wall_k=15, fill_k=7):
    """Solid (poché) wall pixels inside the building outline. Each wall segment between openings is its own
    blob, so all blobs are kept; a small closing fills the white speckles of the scan."""
    walls = cv2.morphologyEx(INK, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (wall_k, wall_k)))
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, np.ones((fill_k, fill_k), np.uint8))
    _, _, outline = run(wall_k)
    bm = np.zeros_like(INK)
    cv2.drawContours(bm, [outline], -1, 1, -1)
    return (walls & bm).astype(np.uint8)


if __name__ == "__main__":
    for wk in (7, 11, 15):
        for seal in (0.4, 0.6, 0.8):
            cands, bx, _ = run(wk, seal)
            ious = [best_match([(l, p) for l, p, _ in cands], g)[2] for _, _, g in ROOMS]
            print(f"wall_k={wk:2d} seal={seal:.1f} m  building {bx.bounds[2] - bx.bounds[0]:.2f} x {bx.bounds[3] - bx.bounds[1]:.2f} m"
                  f"  IoU>=0.5: {sum(i >= 0.5 for i in ious):2d}/15  mean {np.mean(ious):.2f}")
    cands, bx, outline = run(15, 0.8)
    poly = {i: p for i, (_, p, _) in enumerate(cands, 1)}
    print(f"\nscale check: building {bx.bounds[2] - bx.bounds[0]:.2f} x {bx.bounds[3] - bx.bounds[1]:.2f} m (scan, 1:50 from title)"
          f" vs reference {BUILDING.bounds[2] - BUILDING.bounds[0]:.2f} x {BUILDING.bounds[3] - BUILDING.bounds[1]:.2f} m")
    ref = {name: g for name, _, g in ROOMS}
    print("\nrooms judged unchanged since the scan (historical name -> 2005 name):")
    for i, (hist, now) in LABELS.items():
        if now:
            g = ref[now]
            print(f"  #{i:<2} {hist:<16} -> {now:<20} scan {poly[i].area:6.2f} m2  2005 {g.area:6.2f} m2"
                  f"  IoU {poly[i].intersection(g).area / poly[i].union(g).area:.2f}")
    print("other candidates:", {i: (h, round(poly[i].area, 1)) for i, (h, now) in LABELS.items() if not now})
    # numbered overlay, cropped to the building and downscaled for viewing
    vis = cv2.cvtColor(IMG, cv2.COLOR_GRAY2BGR)
    rng = np.random.default_rng(5)
    for _, _, c in cands:
        ov = vis.copy()
        cv2.drawContours(ov, [c], -1, [int(v) for v in rng.integers(60, 230, 3)], -1)
        vis = cv2.addWeighted(ov, 0.45, vis, 0.55, 0)
    for i, (_, _, c) in enumerate(cands, 1):
        m = cv2.moments(c)
        cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
        cv2.putText(vis, str(i), (cx - 30, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 14)
        cv2.putText(vis, str(i), (cx - 30, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 200), 6)
    x, y, w, h = cv2.boundingRect(outline)
    crop = vis[max(0, y - 60):y + h + 60, max(0, x - 60):x + w + 60]
    s = 1400 / max(crop.shape[:2])
    cv2.imwrite(str(OUT / "s2_modeC_candidates.png"), cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
