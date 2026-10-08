"""Shared fixtures: synthetic sheets with known label maps (two_rooms, plan_image, paint), a stub segmenter
(DarkIsWall), the data-dependent skip markers (needs_*), and --update-golden for the golden-output test."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))         # pilot/v2-pipeline/scripts: fpx, fpeval, the scripts

from common import DATA, FONTS, REPO  # noqa: E402
from fpx import DEFAULT, Sheet  # noqa: E402
from fpx.model import CLASSES, DOOR, WALL, WINDOW  # noqa: E402

M = DEFAULT.px_per_m
FLOORS = DATA / "floors-test.pkl"
FPCAD_TEST = REPO / "data/benchmark/floorplancad/test"
FPCAD_BLOCKS = [FPCAD_TEST / f"{b}.svg" for b in ("0000-0003", "0060-0042", "0407-0064")]
_fonts = any(FONTS.glob("*.ttf"))

needs_floors = pytest.mark.skipif(not FLOORS.exists(), reason="needs data/floors-test.pkl (sd_prepare.py)")
needs_fonts = pytest.mark.skipif(not _fonts, reason="needs the renderer fonts (data/public/fonts)")
needs_render = pytest.mark.skipif(not (DATA / "floors-val.pkl").exists() or not _fonts,
                                  reason="needs data/floors-val.pkl (sd_prepare.py) and the OFL fonts")
needs_ifc = pytest.mark.skipif(not (DATA / "floors-ifc.pkl").exists() or not _fonts, reason="needs data/floors-ifc.pkl")
needs_v1_model = pytest.mark.skipif(not (DATA / "model/segmenter.pt").exists(), reason="needs the v1 checkpoint data/model/segmenter.pt")
needs_fpcad = pytest.mark.skipif(not all(p.exists() for p in FPCAD_BLOCKS), reason="FloorPlanCAD test split not present")
needs_curated_images = pytest.mark.skipif(not (REPO / "data/curated/images").exists(), reason="data/curated/images not on this machine")


def px(v):
    return int(round(v * M))


def plan_image(draw=None, ink=None, text=()):
    """A 12 x 8 m sheet with a 10 x 6 m building of 0.3 m walls (outer faces 1 m from the sheet edge). draw(rect)
    adds to the label map; ink(rect) draws extra black ink on the image (default: the wall label); text: (string, x, y,
    role) stamps of 1.2 x 0.25 m."""
    lab = np.zeros((px(8), px(12)), np.uint8)
    rect = lambda x0, y0, x1, y1, v: lab.__setitem__((slice(px(y0), px(y1)), slice(px(x0), px(x1))), v)
    rect(1, 1, 11, 7, WALL)
    rect(1.3, 1.3, 10.7, 6.7, 0)
    if draw:
        draw(rect)
    img = np.full(lab.shape + (3,), 255, np.uint8)
    img[lab == WALL] = 0
    if ink:
        ink(lambda x0, y0, x1, y1: img.__setitem__((slice(px(y0), px(y1)), slice(px(x0), px(x1))), 0))
    sheet = Sheet("t", "test", "test", "synthetic", img, (0.0, 8.0), {"value": "1:50", "method": "test"})
    sheet.label, sheet.seg_label = lab, lab.copy()
    sheet.prob = (np.arange(len(CLASSES))[:, None, None] == lab[None]).astype(np.float32)
    sheet.text = [{"text": s, "box": (px(x), px(y), px(x + 1.2), px(y + 0.25)), "conf": 1.0, "source": "pdf", "angle": 0,
                   "height": px(0.25), "role": r} for s, x, y, r in text]
    return sheet


def two_rooms():
    """The 10 x 6 m building with an interior wall at x = 6 m holding a 0.9 m door, a 1.2 m window in the north
    facade, and a stamp ("Büro", 24.57 m2) in the west room."""
    sheet = plan_image(lambda r: (r(5.85, 1.3, 6.15, 6.7, WALL), r(5.85, 3.5, 6.15, 4.4, DOOR), r(3, 1, 4.2, 1.3, WINDOW)),
                       text=[("Büro", 3, 3.8, "room stamp"), ("24.57 m2", 3, 4.1, "number")])
    sheet.id, sheet.title = "t", "two rooms"
    return sheet


def paint(sheet, x0, y0, x1, y1, cls, ink=None):
    """Paint a rectangle (sheet metres, y down) into the label map and the image."""
    sl = (slice(px(y0), px(y1)), slice(px(x0), px(x1)))
    sheet.label[sl] = cls
    sheet.img[sl] = (0 if cls == WALL else 255) if ink is None else ink
    sheet.seg_label = sheet.label.copy()
    sheet.prob = (np.arange(len(CLASSES))[:, None, None] == sheet.label[None]).astype(np.float32)


def dark_is_wall():
    """Stub segmenter: wall where the input is dark, background elsewhere (v1 interface: logits only)."""
    import torch

    class DarkIsWall(torch.nn.Module):
        def forward(self, x):
            dark = -x.mean(1, keepdim=True)                   # normalised input: ink is negative
            out = torch.zeros(x.shape[0], len(CLASSES), *x.shape[2:])
            out[:, WALL:WALL + 1] = 10 * dark
            return out
    return DarkIsWall()


def pytest_addoption(parser):
    parser.addoption("--update-golden", action="store_true", default=False,
                     help="rewrite tests/golden/*.json from the current output instead of comparing")


@pytest.fixture
def update_golden(request):
    return request.config.getoption("--update-golden")
