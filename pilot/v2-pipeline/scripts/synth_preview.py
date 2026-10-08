"""Preview renderer v3: rendered samples with their labels and extra targets, written to data/.

    python synth_preview.py [seed] [--no-negatives] [--size N]
        data/synth_preview_v3_<era>_<seed>.png   three random samples per style era, sampled as train.py samples
                                                 (its defaults: source mix, scale, column floors, focus; negatives on)
        data/synth_gallery_v3_<seed>.png         one row per style element, forced on (v2 elements, then v3 ones)

Each row: image | class labels (wall black, door blue, window orange, column green, stairs magenta) | interior (blue) |
room boundary (red) | void (purple) | text boxes (green) | door swing (orange). The caption gives era/walls/doors/
windows, the scale factor and the elements present. Grey stripes: target unknown for this floor (not trained on).
"""
import argparse
import time

import cv2
import numpy as np

from common import DATA, HEADS
from synth import IGNORE

COLOURS = np.array([[255, 255, 255], [0, 0, 0], [0, 0, 230], [230, 120, 0], [0, 170, 0], [200, 0, 200]], np.uint8)
TARGET_COLOURS = {"interior": (60, 120, 255), "boundary": (230, 0, 0), "void": (150, 0, 200), "text": (0, 170, 0),
                  "swing": (230, 120, 0)}
ERAS = ["cad", "historical", "colour", "mixed"]
GALLERY = [   # (title, style overrides, element tag that must appear)
    ("hatched terraces + floor finishes", {"outdoor_hatch": True, "room_hatch": 0.6, "era": "cad", "walls": "outline"}, "outdoor_hatch"),
    ("context: roofs, terrain, neighbours", {"context": True, "era": "cad", "walls": "solid"}, "context"),
    ("ceiling ornament", {"ornament": True, "era": "historical", "walls": "outline"}, "ornament"),
    ("window reveals + niches", {"reveals": True, "niches": True, "era": "historical", "walls": "solid", "windows": "two"}, "reveals"),
    ("pilasters + plinths", {"pilasters": True, "era": "historical", "walls": "outline"}, "pilasters"),
    ("casement windows: 2 sashes from 0.9 m", {"windows": "casement", "era": "cad", "walls": "outline", "doors": "arc"}, "casement"),
    ("doors: arcs, 2 leaves from 1.5 m", {"doors": "arc", "windows": "three", "era": "cad", "walls": "outline", "text": False}, ""),
    ("Swiss material hatching", {"walls": "material", "era": "cad", "insulation": True}, "insulation"),
    ("column grid + axes", {"column_grid": True, "era": "cad", "walls": "outline"}, "column_grid"),
    ("colour zones + red pen", {"zones": True, "red_pen": True, "era": "cad", "walls": "grey", "defects": ("colour",)}, "zones"),
    ("stamps + hand notes", {"rubber_stamp": True, "annotations": True, "era": "historical", "walls": "solid"}, "rubber_stamp"),
    ("title block + legend", {"title_block": True, "legend": True, "era": "cad", "walls": "hatch"}, "legend"),
    ("key plan + second drawing", {"keyplan": True, "multi": True, "era": "mixed", "walls": "split"}, "keyplan"),
    ("scan: fold, shadow, bleed", {"defects": ("fold", "shadow", "bleed"), "era": "historical", "walls": "solid"}, "fold"),
    ("scan: fading, perspective, dither", {"defects": ("fade", "perspective", "dither", "lowres"), "era": "historical", "walls": "hatch"}, "perspective"),
    # v3: negatives (background) and the void relabel
    ("v3 dimension chains on the facades", {"dim_chains": True, "dims": False, "era": "cad", "walls": "outline"}, "dim_chains"),
    ("v3 axis bubbles on plain axes", {"axis_bubbles": True, "column_grid": False, "era": "cad", "walls": "hatch"}, "axis_bubbles"),
    ("v3 section markers", {"section_marks": True, "era": "cad", "walls": "grey"}, "section_marks"),
    ("v3 detail bubbles", {"detail_bubbles": True, "era": "cad", "walls": "outline"}, "detail_bubbles"),
    ("v3 door + window tags", {"opening_tags": True, "era": "cad", "walls": "outline"}, "opening_tags"),
    ("v3 level markers", {"level_marks": True, "era": "cad", "walls": "solid"}, "level_marks"),
    ("v3 furniture", {"furniture": True, "era": "cad", "walls": "outline", "features": False}, "furniture"),
    ("v3 shafts + lifts as voids", {"void_cross": True, "era": "cad", "walls": "solid", "text": True}, ""),
]


def row(img, lab, tags, tg, title=None):
    n = img.shape[0]
    over = (0.45 * img + 0.55 * COLOURS[lab]).astype(np.uint8)
    over[lab == 0] = img[lab == 0]
    tiles = [img.copy(), over]
    faint = (0.35 * img + 0.65 * 255).astype(np.uint8)
    for k in HEADS:
        t = faint.copy()
        m = tg[k] == 1
        t[m] = (0.4 * t[m] + 0.6 * np.array(TARGET_COLOURS[k])).astype(np.uint8)
        unknown = tg[k] == IGNORE                       # not trained on (e.g. voids of IFC storeys)
        t[unknown & ((np.add.outer(np.arange(n), np.arange(n)) // 6) % 2 == 0)] = (170, 170, 170)
        cv2.putText(t, k + (" (unknown)" if unknown.all() else ""), (4, n - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        tiles.append(t)
    cap = f"{title + ' | ' if title else ''}{tags['era']}/{tags['walls']}/{tags['doors']}/{tags['windows']} x{tags['scale']} {tags['source']}"
    for i, txt in enumerate([cap, tags["elements"][:70], tags["elements"][70:140]]):
        if txt:
            cv2.putText(tiles[0], txt, (4, 12 + 12 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    sep = np.full((n, 4, 3), 128, np.uint8)
    return np.hstack([x for t in tiles for x in (t, sep)])


def main():
    import train                                        # the renderer and its sampling exactly as train.py builds them
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seed", type=int, nargs="?", default=0)
    ap.add_argument("--no-negatives", action="store_true", help="renderer without the v3 negative set")
    ap.add_argument("--size", type=int, default=320, help="crop size of the preview renders (training: 512)")
    ap.add_argument("--rows", type=int, default=3, help="random samples per era")
    a = ap.parse_args()
    args = train.parser().parse_args(["--size", str(a.size)] + ([] if a.no_negatives else ["--negatives"]))
    r = train.renderer(args, "val")
    suffix = "" if args.negatives else "_noneg"
    t0, count = time.time(), 0
    for era in ERAS:
        rng = np.random.default_rng([a.seed, ERAS.index(era)])
        rows = []
        for i in range(a.rows):
            img, lab, tags, tg = r.sample(rng, targets=True, style={"era": era})
            rows.append(row(img, lab, tags, tg))
            count += 1
        out = DATA / f"synth_preview_v3_{era}{suffix}_{a.seed}.png"
        cv2.imwrite(str(out), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
        print("wrote", out)
    print(f"{(time.time() - t0) / count * 1000:.0f} ms per sample at {a.size} px")
    rows = []
    for j, (title, force, tag) in enumerate(GALLERY):
        for k in range(12):                             # first seed in which the element actually shows
            img, lab, tags, tg = r.sample(np.random.default_rng([a.seed, j, k]), targets=True, style=force)
            if not tag or tag in tags["elements"].split(","):
                break
        rows.append(row(img, lab, tags, tg, title))
    out = DATA / f"synth_gallery_v3{suffix}_{a.seed}.png"
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    print("wrote", out)


if __name__ == "__main__":
    main()
