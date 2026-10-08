"""Shared paths and the current model for the pilot scripts (the CLI helpers are in fpeval.cli).

Paths can be overridden with environment variables, so the same code runs locally and on a RunPod pod.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent                                 # pilot/v2-pipeline/scripts: the sources
PILOT = ROOT.parent                                                     # pilot/v2-pipeline: README, data, viewer
REPO = ROOT.parents[2]
DATA = Path(os.environ.get("V2_DATA", PILOT / "data"))                 # gitignored: floors, models, outputs
FONTS = Path(os.environ.get("V2_FONTS", REPO / "data/public/fonts"))
SD_CSV = REPO / "data/public/swiss-dwellings-v3.0.0/geometries.csv"
V1 = REPO / "pilot/landgut-lohn-og1"                                   # Landgut Lohn inputs and reference

MODEL = Path(os.environ.get("V2_MODEL", DATA / "model-v3/segmenter.pt"))   # the current segmenter (v3 = run E3 of the second review; v2: model-v2, v1: model)

# Transitional re-exports for synth.py, train.py and synth_preview.py, which still import the classes and the
# working resolution from here; they belong to fpx.model and fpx.config and go when those scripts import them there.
from fpx.config import DEFAULT  # noqa: E402
from fpx.model import BG, CLASSES, COLUMN, DOOR, HEADS, STAIRS, WALL, WINDOW  # noqa: E402, F401

PX_PER_M = DEFAULT.px_per_m
