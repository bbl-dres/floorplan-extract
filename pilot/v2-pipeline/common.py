"""Shared paths and constants for pilot v2.

Paths can be overridden with environment variables, so the same code runs locally and on a RunPod pod.
"""
import os
from pathlib import Path

from fpx.config import DEFAULT
from fpx.model import BG, CLASSES, COLUMN, DOOR, HEADS, STAIRS, WALL, WINDOW  # noqa: F401 (re-exported)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DATA = Path(os.environ.get("V2_DATA", ROOT / "data"))                  # gitignored: floors, models, outputs
FONTS = Path(os.environ.get("V2_FONTS", REPO / "data/public/fonts"))
SD_CSV = REPO / "data/public/swiss-dwellings-v3.0.0/geometries.csv"
V1 = REPO / "pilot/landgut-lohn-og1"                                   # Landgut Lohn inputs and reference

PX_PER_M = DEFAULT.px_per_m                     # working resolution of the segmenter: 2 cm per pixel
