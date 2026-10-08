"""Local server of the floor plan workflow app (upload, building area, scale, storeys and run, results, export).

    python app/server.py [--port 8765]

Standard library only (http.server with a threading server, JSON endpoints, uploads as the raw request body). Binds to
127.0.0.1: plans never leave this machine. Pipeline work (analysis after an upload, extraction after confirmation)
runs on one worker thread, one job at a time, so a shared machine is not flooded; the page polls the job status.

Data: app/data/sheets/<job id>/ holds the upload, the previews, analysis.json, confirm.json, job.json, log.txt,
review.json and out/ with the pipeline's outputs. app/data is gitignored.

Endpoints (all JSON unless a file is served):
    GET    /                                 the page (index.html)
    GET    /static/<file>                    app.js, app.css
    GET    /api/health                       what the pipeline needs (model file, OCR engine, libraries, DWG converter)
    POST   /api/upload?name=<file name>      body = the file; creates a sheet job and queues its analysis -> {id}
    GET    /api/sheets                       all sheet jobs on disk, newest first
    GET    /api/sheets/<id>                  the job: status, progress, log tail, analysis, confirmations, results
    DELETE /api/sheets/<id>                  remove the job and its files
    POST   /api/sheets/<id>/confirm          {"confirm": {"<package>/<drawing>": {extract, polygon_px, px_per_m, ...}}}
    POST   /api/sheets/<id>/run              queue the extraction of the confirmed drawings
    GET    /api/sheets/<id>/results?key=...  one drawing's results for the viewer (rooms, openings, walls, QA ...)
    GET    /api/sheets/<id>/file/<name>      a file of the job (preview, working image, overlay, JSON, DXF)
    GET    /api/sheets/<id>/rooms.csv?key=.. the room list as CSV
    POST   /api/sheets/<id>/review           {"key": ..., "rooms": {"r01": {"checked": true}}} -> review.json
    GET    /api/sheets/<id>/log              the full log as text
"""
import argparse
import csv
import io
import json
import queue
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import bridge

APP = Path(__file__).resolve().parent
DATA = APP / "data"
SHEETS = DATA / "sheets"
STATIC = APP / "static"
MAX_UPLOAD = 300 * 1024 * 1024
LOG_KEEP = 400
EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp", ".dxf", ".dwg"}
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".dxf": "application/dxf", ".txt": "text/plain; charset=utf-8", ".csv": "text/csv; charset=utf-8",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".ifc": "application/x-step",
        ".svg": "image/svg+xml"}

JOBS = {}                 # id -> job record (also job.json on disk)
SESSIONS = {}             # id -> in-memory packages and layouts from the analysis (lost on restart, recomputed)
LOCK = threading.RLock()
WORK = queue.Queue()
HEALTH = {}


# ---------------------------------------------------------------------------------------------------------------------
# Jobs

def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def job_dir(jid):
    return SHEETS / jid


def save(job):
    with LOCK:
        bridge.dump(job, job_dir(job["id"]) / "job.json")


def log(job, line):
    stamp = time.strftime("%H:%M:%S")
    text = f"{stamp} {line}"
    with LOCK:
        job["log"].append(text)
        del job["log"][:-LOG_KEEP]
    try:
        with open(job_dir(job["id"]) / "log.txt", "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except OSError:
        pass
    print(f"[{job['id']}] {text}", flush=True)


def load_jobs():
    SHEETS.mkdir(parents=True, exist_ok=True)
    for p in sorted(SHEETS.glob("*/job.json")):
        try:
            job = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if job.get("status") in ("analysing", "queued", "running"):
            job["status"] = "error"
            job["error"] = "The server was stopped while this job ran. Upload the sheet again or run the extraction again."
        JOBS[job["id"]] = job
    if JOBS:
        print(f"{len(JOBS)} sheet jobs found in {SHEETS}")


def new_job(name, body):
    safe = re.sub(r"[^\w.\-]+", "_", Path(name).name, flags=re.U).strip("._") or "sheet"
    ext = Path(safe).suffix.lower()
    if ext not in EXTENSIONS:
        raise bridge.AppError(f"Unsupported file type {ext or '(none)'}: upload PDF, JPG, PNG, TIFF, DXF or DWG.")
    jid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    d = job_dir(jid)
    d.mkdir(parents=True, exist_ok=True)
    stem = Path(safe).stem[:60]
    upload = d / f"{stem}{ext}"
    upload.write_bytes(body)
    job = {"id": jid, "name": name, "file": upload.name, "size": len(body), "created": now(), "status": "analysing",
           "progress": None, "error": None, "log": [], "analysis": None, "confirm": {}, "results": None, "times": {}}
    with LOCK:
        JOBS[jid] = job
    save(job)
    log(job, f"uploaded {name} ({len(body) / 1e6:.1f} MB)")
    WORK.put(("analyse", jid))
    return job


def worker():
    while True:
        kind, jid = WORK.get()
        job = JOBS.get(jid)
        if job is None:
            continue
        try:
            if kind == "analyse":
                run_analysis(job)
            elif kind == "extract":
                run_extraction(job)
        except Exception as ex:                            # noqa: BLE001 - the worker must survive every job
            with LOCK:
                job["status"] = "error"
                job["error"] = f"{type(ex).__name__}: {ex}"
            log(job, f"internal error\n{traceback.format_exc()}")
            save(job)
        finally:
            WORK.task_done()


def run_analysis(job):
    d = job_dir(job["id"])
    upload = d / job["file"]
    t0 = time.time()
    with LOCK:
        job["status"] = "analysing"
        job["error"] = None
        job["progress"] = {"key": None, "stage": "0a normalisation, 1b layout, 1c scale"}
    save(job)
    try:
        analysis, session = bridge.analyse(d, upload, lambda s: log(job, s))
    except bridge.AppError as ex:
        with LOCK:
            job["status"], job["error"] = "error", str(ex)
        log(job, f"error: {ex}")
        save(job)
        return
    except Exception as ex:
        with LOCK:
            job["status"], job["error"] = "error", f"{type(ex).__name__}: {ex}"
        log(job, f"error: {type(ex).__name__}: {ex}\n{traceback.format_exc()}")
        save(job)
        return
    session["upload"] = upload
    SESSIONS[job["id"]] = session
    saved = bridge.save_session(d, session)
    if saved is not True:
        log(job, f"note: the analysed session could not be saved ({saved}); after a server restart the analysis runs again")
    with LOCK:
        job["analysis"] = analysis
        job["status"] = "ready"
        job["progress"] = None
        job["times"]["analysis"] = round(time.time() - t0, 1)
        if not job["confirm"]:
            job["confirm"] = default_confirm(analysis)
    save(job)


def default_confirm(analysis):
    """What the pipeline proposes, as the starting point of the confirmations."""
    out = {}
    for p in analysis["packages"]:
        for dr in p["drawings"]:
            s = dr.get("scale") or {}
            out[f"{p['id']}/{dr['id']}"] = {"extract": dr["kind"] == bridge.FLOOR_PLAN, "polygon_px": None,
                                            "px_per_m": None, "scale_source": None, "measured": None,
                                            "storey": dr.get("storey"), "proposal_px_per_m": s.get("px_per_m"),
                                            "area_confirmed": False, "scale_confirmed": False}
    return out


def run_extraction(job):
    d = job_dir(job["id"])
    t0 = time.time()
    with LOCK:
        job["status"] = "running"
        job["error"] = None
        job["results"] = None
        job["progress"] = {"key": None, "stage": "starting"}
    save(job)

    def progress(key, stage):
        with LOCK:
            job["progress"] = {"key": key, "stage": stage}
        log(job, f"  [{key}] {stage}")

    try:
        session = SESSIONS.get(job["id"])
        if session is None:
            session = bridge.load_session(d)
            if session is not None:
                log(job, "analysed session loaded from disk (the server was restarted)")
                session["upload"] = d / job["file"]
                SESSIONS[job["id"]] = session
        if session is None:
            log(job, "the analysis is not in memory (server restarted): running it again first")
            analysis, session = bridge.analyse(d, d / job["file"], lambda s: log(job, s))
            session["upload"] = d / job["file"]
            SESSIONS[job["id"]] = session
            bridge.save_session(d, session)
            with LOCK:
                job["analysis"] = analysis
        results = bridge.extract(d, session, dict(job["confirm"]), lambda s: log(job, s), progress)
    except bridge.AppError as ex:
        with LOCK:
            job["status"], job["error"] = "error", str(ex)
        log(job, f"error: {ex}")
        save(job)
        return
    except Exception as ex:
        with LOCK:
            job["status"], job["error"] = "error", f"{type(ex).__name__}: {ex}"
        log(job, f"error: {type(ex).__name__}: {ex}\n{traceback.format_exc()}")
        save(job)
        return
    with LOCK:
        job["results"] = results
        job["progress"] = None
        job["times"]["extraction"] = round(time.time() - t0, 1)
        failed = [k for k, r in results.items() if r.get("error")]
        done = [k for k, r in results.items() if r.get("outputs")]
        job["status"] = "done" if done or not failed else "error"
        if failed:
            job["error"] = "; ".join(f"{k}: {results[k]['error']}" for k in failed)
    log(job, f"extraction finished in {job['times']['extraction']} s")
    save(job)


# ---------------------------------------------------------------------------------------------------------------------
# Results for the viewer

def results_payload(job, key):
    r = (job.get("results") or {}).get(key)
    if not r:
        raise KeyError(f"no results for {key}")
    if not r.get("outputs"):
        return {"key": key, "error": r.get("error"), "skipped": r.get("skipped"), "record": r}
    out = job_dir(job["id"]) / "out"
    data = json.loads((out / r["outputs"]["json"]).read_text(encoding="utf-8"))
    review = {}
    rp = job_dir(job["id"]) / "review.json"
    if rp.exists():
        try:
            review = json.loads(rp.read_text(encoding="utf-8")).get(key, {})
        except ValueError:
            review = {}
    rooms = []
    for room in data["rooms"]:
        rooms.append({k: room.get(k) for k in ("id", "name", "usage", "number", "aoid", "area", "area_net", "area_gross",
                                                "area_stamp", "area_deviation_pct", "area_basis", "confidence", "reasons",
                                                "small_region", "review", "neighbours", "geometry", "name_raw")})
    voids = [{k: v.get(k) for k in ("id", "kind", "label", "area", "gf_deducted", "geometry")} for v in data["voids"]]
    return {"key": key, "sheet_id": r["sheet_id"], "title": data["sheet"]["title"], "storey": r.get("storey"),
            "px_per_m": r["px_per_m"], "work_px": r["work_px"], "extent": data["sheet"]["extent"],
            "files": r["outputs"], "scale": data["sheet"]["scale"], "triage": data["sheet"].get("triage"),
            "times": r["times"], "summary": r.get("summary"),
            "rooms": rooms, "voids": voids, "stairs": data["stairs"], "walls": data["wall_polygons"],
            "columns": [c["geometry"] for c in data.get("structure", [])],
            "openings": [{k: o.get(k) for k in ("id", "kind", "width", "exterior", "connects", "geometry")} for o in data["openings"]],
            "floor": {k: data["floor"].get(k) for k in ("gf", "gf_area", "agf_area", "sum_net", "sum_gross")},
            "connectivity": data["connectivity"], "qa": data["qa"],
            "stamps": [{"name": s.get("name"), "area": s.get("area"), "number": s.get("number"), "box": s.get("box")} for s in data.get("stamps", [])],
            "drawing": (data.get("drawings") or [None])[0], "review": review}


def rooms_csv(job, key):
    p = results_payload(job, key)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["id", "name", "usage", "number", "area_net_m2", "area_gross_m2", "area_stamp_m2", "deviation_pct",
                "confidence", "checked", "reasons"])
    for r in p["rooms"]:
        rv = (p["review"].get("rooms") or {}).get(r["id"], {})
        w.writerow([r["id"], r["name"] or "", r["usage"] or "", r["number"] or "", r["area_net"], r["area_gross"],
                    r["area_stamp"] if r["area_stamp"] is not None else "", r["area_deviation_pct"] if r["area_deviation_pct"] is not None else "",
                    r["confidence"], "yes" if rv.get("checked") else "", "; ".join(r.get("reasons") or [])])
    return buf.getvalue()


# ---------------------------------------------------------------------------------------------------------------------
# HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "fpx-workflow/0.1"

    def log_message(self, fmt, *args):                     # quiet: the job log is what matters
        if "/api/" in (args[0] if args else "") and str(args[1] if len(args) > 1 else "") not in ("200", "204"):
            sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    # --- helpers
    def send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False, default=bridge._jsonable).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status, message):
        self.send_json({"error": message}, status)

    def send_file(self, path, download=None):
        path = Path(path)
        if not path.is_file():
            return self.send_error_json(404, f"{path.name} not found")
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def send_text(self, text, mime="text/plain; charset=utf-8", download=None):
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise bridge.AppError(f"The upload is larger than {MAX_UPLOAD // 1024 // 1024} MB.")
        return self.rfile.read(n) if n else b""

    def read_json(self):
        body = self.read_body()
        return json.loads(body.decode("utf-8")) if body else {}

    def job(self, jid):
        job = JOBS.get(jid)
        if job is None:
            raise KeyError(f"sheet {jid} not found")
        return job

    def public_job(self, job):
        with LOCK:
            j = dict(job)
            j["log"] = job["log"][-40:]
        return j

    # --- routing
    def do_GET(self):
        try:
            self.route("GET")
        except KeyError as ex:
            self.send_error_json(404, str(ex))
        except bridge.AppError as ex:
            self.send_error_json(400, str(ex))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as ex:                            # noqa: BLE001
            traceback.print_exc()
            self.send_error_json(500, f"{type(ex).__name__}: {ex}")

    do_POST = do_DELETE = do_GET

    def route(self, method):
        method = self.command
        u = urlparse(self.path)
        path, q = u.path, parse_qs(u.query)
        if method == "GET" and path in ("/", "/index.html"):
            return self.send_file(APP / "index.html")
        if method == "GET" and path.startswith("/static/"):
            name = Path(path[len("/static/"):]).name
            return self.send_file(STATIC / name)
        if method == "GET" and path == "/api/health":
            if not HEALTH:
                HEALTH.update(bridge.check())
            return self.send_json(HEALTH)
        if method == "POST" and path == "/api/upload":
            name = (q.get("name") or [self.headers.get("X-File-Name") or "sheet"])[0]
            body = self.read_body()
            if not body:
                raise bridge.AppError("The upload is empty.")
            job = new_job(name, body)
            return self.send_json({"id": job["id"], "name": job["name"], "status": job["status"]})
        if method == "GET" and path == "/api/sheets":
            with LOCK:
                rows = [{"id": j["id"], "name": j["name"], "status": j["status"], "created": j["created"],
                         "drawings": sum(len(p["drawings"]) for p in (j.get("analysis") or {}).get("packages", []))}
                        for j in sorted(JOBS.values(), key=lambda j: j["id"], reverse=True)]
            return self.send_json({"sheets": rows})
        m = re.match(r"^/api/sheets/([\w\-]+)(?:/(.*))?$", path)
        if not m:
            raise KeyError(f"no route for {method} {path}")
        job = self.job(m.group(1))
        sub = m.group(2) or ""
        if method == "GET" and sub == "":
            return self.send_json(self.public_job(job))
        if method == "DELETE" and sub == "":
            with LOCK:
                JOBS.pop(job["id"], None)
                SESSIONS.pop(job["id"], None)
            shutil.rmtree(job_dir(job["id"]), ignore_errors=True)
            return self.send_json({"deleted": job["id"]})
        if method == "POST" and sub == "confirm":
            data = self.read_json()
            with LOCK:
                for key, val in (data.get("confirm") or {}).items():
                    cur = job["confirm"].setdefault(key, {})
                    cur.update(val)
            save(job)
            return self.send_json({"confirm": job["confirm"]})
        if method == "POST" and sub == "run":
            if job["status"] in ("analysing", "queued", "running"):
                raise bridge.AppError(f"The sheet is busy ({job['status']}).")
            if not any(c.get("extract") for c in job["confirm"].values()):
                raise bridge.AppError("No drawing is selected for extraction.")
            with LOCK:
                job["status"] = "queued"
                job["error"] = None
                job["progress"] = {"key": None, "stage": "waiting for the worker"}
            save(job)
            WORK.put(("extract", job["id"]))
            return self.send_json({"status": "queued"})
        if method == "GET" and sub == "results":
            key = (q.get("key") or [""])[0]
            return self.send_json(results_payload(job, key))
        if method == "GET" and sub == "rooms.csv":
            key = (q.get("key") or [""])[0]
            return self.send_text(rooms_csv(job, key), "text/csv; charset=utf-8", f"{key.replace('/', '-')}_rooms.csv")
        if method == "GET" and sub.startswith("file/"):
            name = Path(sub[len("file/"):]).name
            d = job_dir(job["id"])
            path = d / "out" / name if (d / "out" / name).is_file() else d / name
            download = name if name.endswith((".json", ".dxf")) and "download" in q else None
            return self.send_file(path, download)
        if method == "GET" and sub == "log":
            p = job_dir(job["id"]) / "log.txt"
            return self.send_text(p.read_text(encoding="utf-8") if p.exists() else "\n".join(job["log"]))
        if method == "POST" and sub == "review":
            data = self.read_json()
            key = data.get("key")
            if not key:
                raise bridge.AppError("review needs a drawing key")
            p = job_dir(job["id"]) / "review.json"
            try:
                review = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            except ValueError:
                review = {}
            entry = review.setdefault(key, {"rooms": {}})
            for rid, val in (data.get("rooms") or {}).items():
                entry["rooms"].setdefault(rid, {}).update(val)
            entry["updated"] = now()
            bridge.dump(review, p)
            return self.send_json({"review": entry})
        raise KeyError(f"no route for {method} {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    DATA.mkdir(parents=True, exist_ok=True)
    load_jobs()
    threading.Thread(target=worker, daemon=True, name="pipeline-worker").start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    print(f"Floor plan workflow app: http://127.0.0.1:{a.port}/   (pipeline: {bridge.PIPELINE_DIR}; data: {DATA})", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
