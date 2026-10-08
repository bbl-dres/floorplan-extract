"""CLI helpers shared by the scripts: the common options, console and thread setup, and the bookkeeping of a
per-sheet loop (a sheet that fails becomes a row with its error, a sheet that is skipped keeps its reason)."""
import argparse
import sys
import traceback

from common import MODEL
from fpx.config import DEFAULT


class Skip(Exception):
    """A sheet that is not evaluated, with the reason (recorded, not counted as a failure)."""


def cli(description, out=None, threads=8):
    """Argument parser with the options every script shares: --model, --out, --config, --threads."""
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(MODEL), help=f"segmenter checkpoint (default {MODEL.name} in {MODEL.parent})")
    ap.add_argument("--out", default=None if out is None else str(out), help="output folder" + (f" (default {out})" if out else ""))
    ap.add_argument("--config", default=None, help="JSON thresholds (fpx.Config.from_file)")
    ap.add_argument("--threads", type=int, default=threads, help=f"torch and OpenCV threads (default {threads})")
    return ap


def utf8_stdout():
    """m², sheet names such as "Baptisterium_ortodoxnych_pódorys" and umlauts in a cp1252 console or a redirected log."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def setup(args):
    """Apply the shared options: UTF-8 console, thread caps; returns the Config."""
    import cv2
    import torch
    from fpx.config import Config
    utf8_stdout()
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(min(args.threads, 4))
    return Config.from_file(args.config) if args.config else DEFAULT


def jsonable(o):
    """Result rows -> plain JSON: numpy scalars and arrays unwrapped, floats rounded to 4 places, NaN/inf to null."""
    import numpy as np
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def record_failure(rows, sid, exc, seconds=None):
    """A sheet whose run raised: the traceback goes to the console, and {"id", "error"} joins the rows, so the output
    lists what failed instead of silently holding fewer sheets. Returns the row."""
    traceback.print_exc()
    row = {"id": sid, "error": f"{type(exc).__name__}: {exc}"}
    if seconds is not None:
        row["seconds"] = {"total": round(seconds, 1)}
    rows.append(row)
    print(f"FAIL {sid}: {row['error']}", flush=True)
    return row
