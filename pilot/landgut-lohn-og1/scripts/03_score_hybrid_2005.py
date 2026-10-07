"""Mode C hybrid on the 2005 sheet: CV candidates (02) + VLM labelling of the numbered overlay.

LABELS is the VLM reading of out/s1_modeC_candidates.png: which candidates form which room, and the
stamp area read in mode B (None = not readable). REJECTED are candidates the VLM marked as not a room.
"""
import importlib

import numpy as np
from shapely.ops import unary_union

from common import ROOMS, iou

seg = importlib.import_module("02_segment_2005")

LABELS = {
    "Bad/WC NW": ([15], 6.89), "Kreidolf-Zimmer": ([13], 27.37), "Treppe / Luftraum": ([11, 14, 18, 17], 17.98),
    "Churchill-Zimmer": ([12], 40.73), "Balkon": ([19], 17.87), "Flur West OG": ([24], 9.79),
    "Obere Halle": ([20], 16.96), "Vorplatz OG": ([25], 5.94), "Bad/WC NE1": ([22], None),
    "Bad/WC NE2": ([23], 5.46), "Biedermeier-Zimmer": ([26], 18.29), "Bad/WC SW": ([31], 7.13),
    "Mittelzimmer": ([28], 28.01), "Bad/WC S": ([30, 33], 10.87), "Damenzimmer": ([27], 31.94),
}
REJECTED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 21, 29, 32]   # strips outside the facade
SEAL_PX, KEEP_PX, SNAP_M = 32, 5, 0.10


def labelled_rooms():
    """Return (rooms, rejected): rooms = {name: (merged polygon, stamp area read by the VLM or None)}."""
    cands = seg.run(SEAL_PX, KEEP_PX)
    poly = {i: p for i, (_, p, _) in enumerate(cands, 1)}
    assert sorted(sum((ids for ids, _ in LABELS.values()), []) + REJECTED) == list(range(1, len(cands) + 1))
    rooms = {name: (unary_union([poly[i] for i in ids]), read) for name, (ids, read) in LABELS.items()}
    return rooms, [poly[i] for i in REJECTED]


if __name__ == "__main__":
    rooms, _ = labelled_rooms()
    ref = {name: (la, g) for name, la, g in ROOMS}
    print(f"{'room':<20}{'read':>7}{'stamp':>7}{'CV m2':>8}{'IoU':>6}{'error':>9}  flag (>10% vs read stamp)")
    for name, (p, read) in rooms.items():
        la, g = ref[name]
        flag = "!" if read is None or abs((p.area - read) / read) > 0.10 else ""
        print(f"{name:<20}{(read or '?')!s:>7}{la:>7.2f}{p.area:>8.2f}{iou(p, g):>6.2f}{(p.area - la) / la * 100:>+8.1f}%  {flag}")

    print("\noutward offset (snap to wall face) vs area error - offset tuned on this sheet, optimistic")
    for d in (0.0, 0.05, 0.10, 0.15, 0.20):
        errs, ious = [], []
        for name, (p, _) in rooms.items():
            la, g = ref[name]
            q = p.buffer(d, join_style=2)
            errs.append(abs(q.area - la) / la * 100)
            ious.append(iou(q, g))
        e = np.array(errs)
        print(f"  {d:.2f} m: median |error| {np.median(e):5.1f}%  within 3%: {int((e <= 3).sum()):2d}/15  within 5%: {int((e <= 5).sum()):2d}/15  mean IoU {np.mean(ious):.2f}")
