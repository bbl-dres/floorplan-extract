import numpy as np
from shapely.geometry import LineString

from fpx.geometry import clean, dominant_angle, mask_polys, touches_border


def test_mask_polys_follow_pixel_edges():
    m = np.zeros((300, 350), bool)
    m[20:220, 30:280] = True                           # 200 x 250 px
    (p,) = mask_polys(m)
    assert p.area == m.sum()
    assert p.bounds == (30, 20, 280, 220)
    (c,) = mask_polys(m, edges=False)                  # through pixel centres: half a pixel short on each side
    assert c.area == 199 * 249


def test_mask_polys_holes_and_minimum():
    m = np.zeros((100, 100), bool)
    m[10:90, 10:90] = True
    m[40:60, 40:60] = False
    m[95:97, 95:97] = True                             # 4 px speck
    polys = mask_polys(m, min_px=10)
    assert len(polys) == 1
    assert abs(polys[0].area - m[:94, :94].sum()) <= 1     # the traced hole has chamfered corners


def test_clean():
    m = np.zeros((50, 50), bool)
    m[5:15, 5:15] = True                               # 100 px
    m[30:33, 30:33] = True                             # 9 px
    out = clean(m, 10)
    assert out[10, 10] and not out[31, 31]


def test_dominant_angle():
    a = np.radians(30)
    segs = [{"line": LineString([(0, 0), (100 * np.cos(a), 100 * np.sin(a))])},
            {"line": LineString([(0, 0), (-50 * np.sin(a), 50 * np.cos(a))])}]
    assert abs(dominant_angle(segs) - 30) <= 2      # 1-degree bins, smoothed over three
    assert dominant_angle([]) == 0.0


def test_touches_border():
    m = np.zeros((20, 20), bool)
    m[5:10, 5:10] = True
    assert not touches_border(m)
    m[0, 7] = True
    assert touches_border(m)
