"""The server under test without the pipeline: a stub `bridge` module is installed before server.py is imported, so
the routes, the job model and the worker can be exercised in a second with no model, OCR or torch."""
import json
import sys
import threading
import types
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


class AppError(RuntimeError):
    pass


def _jsonable(v):
    return str(v)


def dump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=_jsonable), encoding="utf-8")


ANALYSIS = {"packages": [{"id": "p1", "index": 0, "summary": {"input_class": "raster", "dpi": 300},
                          "preview": {"file": "p1_preview.jpg", "scale": 1.0}, "regions": [], "layout": {}, "layout_flags": [],
                          "drawings": [{"id": "d1", "kind": "floor plan", "title": "Plan", "polygon_px": [[0, 0], [10, 0], [10, 10], [0, 10]],
                                        "bbox_px": [0, 0, 10, 10], "scale": None, "storey": None}]}], "seconds": 0.1}


def make_bridge(gate=None):
    """A bridge whose analyse() returns a fixed analysis; with a gate (threading.Event) it waits until set, so a job
    stays 'analysing' as long as a test wants."""
    b = types.ModuleType("bridge")
    b.AppError = AppError
    b._jsonable = _jsonable
    b.dump = dump
    b.PIPELINE_DIR = APP
    b.FLOOR_PLAN = "floor plan"
    b.check = lambda: {"pipeline_dir": str(APP), "python": "x", "threads": 1, "items": {}, "ok": True}

    def analyse(job_dir, upload, log, want_model=True):
        if gate is not None:
            gate.wait(10)
        log("stub analysis")
        return json.loads(json.dumps(ANALYSIS)), {"packages": []}

    def extract(job_dir, session, confirm, log, progress, analysis=None):
        progress("p1/d1", "stub stage")
        return {"p1/d1": {"key": "p1/d1", "outputs": None, "error": "stub: no pipeline", "skipped": None, "times": {}}}

    def scale_step(job_dir, session, analysis, confirm, keys, log, doors=False):
        if gate is not None:
            gate.wait(10)
        for p in analysis["packages"]:
            for d in p["drawings"]:
                if f"{p['id']}/{d['id']}" in keys:
                    d["scale"] = {"px_per_m": 50.0 if not doors else 48.0, "confidence": "high", "agreeing": [{"cue": "door_widths" if doors else "dimension_strings"}]}
        log("stub scale cues")
        return list(keys)

    b.scale_step = scale_step
    b.live_name = lambda key: f"{key.replace('/', '-')}_live.json"

    b.analyse = analyse
    b.extract = extract
    b.save_session = lambda job_dir, session: True
    b.load_session = lambda job_dir: None
    return b


@pytest.fixture
def served(tmp_path, monkeypatch):
    """server module with its data folder in tmp_path, a worker thread, and the HTTP server on a free port.
    Yields (base url, server module, gate)."""
    gate = threading.Event()
    gate.set()
    sys.modules["bridge"] = make_bridge(gate)
    sys.modules.pop("server", None)
    import server
    monkeypatch.setattr(server, "DATA", tmp_path)
    monkeypatch.setattr(server, "SHEETS", tmp_path / "sheets")
    server.JOBS.clear()
    server.SESSIONS.clear()
    server.load_jobs()
    threading.Thread(target=server.worker, daemon=True).start()
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", server, gate
    finally:
        srv.shutdown()
        srv.server_close()
