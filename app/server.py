"""Local server of the floor plan workflow app (upload, building area, scale, storeys and run, results, export).

    python app/server.py [--port 8765]

Standard library only (http.server with a threading server, JSON endpoints, uploads as the raw request body). Binds to
127.0.0.1: plans never leave this machine. Pipeline work (analysis after an upload, extraction after confirmation)
runs on one worker thread, one job at a time, so a shared machine is not flooded; the page polls the job status.

Data: app/data/sheets/<job id>/ holds the upload, the previews, analysis.json, confirm.json, job.json, log.txt,
review.json and out/ with the pipeline's outputs. app/data is gitignored.

Endpoints (all JSON unless a file is served):
    GET    /                                 the page (index.html)
    GET    /js/<path>, /css/<path>            the app's modules (ES modules, vanilla JS) and stylesheets (tokens, styles)
    GET    /api/health                       what the pipeline needs (model file, OCR engine, libraries, DWG converter)
    POST   /api/upload?name=<file name>      body = the file; creates a sheet job and queues its analysis -> {id}
    GET    /api/demo                         the bundled demo sheet (app/demo/demo.json: file, title, source, licence) or null
    POST   /api/demo                         a sheet job from the demo sheet, as an upload -> {id}
    GET    /api/sheets                       all sheet jobs on disk, newest first
    GET    /api/sheets/<id>                  the job: status, progress, log tail, analysis, confirmations, results, rev;
                                             with ?rev=<n> of the record the client holds, {"unchanged": true} when nothing changed
    DELETE /api/sheets/<id>                  removes the job and its files; a busy job is discarded once the worker is done with it                  remove the job and its files
    POST   /api/sheets/<id>/confirm          {"confirm": {"<package>/<drawing>": {extract, polygon_px, px_per_m, ...}}}
    POST   /api/sheets/<id>/scale            {"keys": ["<package>/<drawing>"], "doors": false}: queue step 2, the scale
                                             cues of those drawings (doors: the door-width search with the segmenter)
    POST   /api/sheets/<id>/run              queue the extraction of the confirmed drawings
    GET    /api/sheets/<id>/results?key=...  one drawing's results for the viewer
    GET    /api/sheets/<id>/live?key=...     the geometry found so far while the drawing is extracted (walls, openings, stairs,
                                             rooms in working px of the working image) or {"stage": null} (rooms, openings, walls, QA ...)
    GET    /api/sheets/<id>/file/<name>      a file of the job (preview, working image, overlay, JSON, DXF)
    GET    /api/sheets/<id>/rooms.csv?key=.. the room list as CSV
    POST   /api/sheets/<id>/review           {"key": ..., "rooms": {"r01": {"checked": true}}} -> review.json
    GET    /api/sheets/<id>/log              the full log as text
"""
import argparse
import copy
import csv
import io
import json
import os
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
ASSETS = ("js", "css")                                    # app/js and app/css are served as /js/... and /css/...
DEMO = APP / "demo"                                        # the demo sheet: demo.json (file, title, source, licence) and the file
BUSY = ("analysing", "scaling", "queued", "running")
SESSION_KEEP = int(os.environ.get("FPX_APP_SESSIONS", "3"))   # analysed sessions kept in memory (each holds the sheet rasters); the rest reload from disk
PAYLOAD_KEEP = 8                                           # results payloads kept in memory, keyed by file
MAX_UPLOAD = 300 * 1024 * 1024
LOG_KEEP = 400
EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp", ".dxf", ".dwg"}
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".dxf": "application/dxf", ".txt": "text/plain; charset=utf-8", ".csv": "text/csv; charset=utf-8",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".ifc": "application/x-step",
        ".svg": "image/svg+xml"}

class NotFound(Exception):
    """A missing sheet, route or result: answered with 404 (a KeyError in the code stays a 500, as it should)."""


JOBS = {}                 # id -> job record (also job.json on disk)
SESSIONS = {}             # id -> in-memory packages and layouts from the analysis (at most SESSION_KEEP; the pickle on disk has the rest)
PAYLOADS = {}             # (id, key) -> (mtime, payload) of results_payload
LOCK = threading.RLock()
WORK = queue.Queue()
HEALTH = {}


# ---------------------------------------------------------------------------------------------------------------------
# Jobs

def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def job_dir(jid):
    return SHEETS / jid


def touch(job):
    """Count a change of the record: GET /api/sheets/<id>?rev= answers 'unchanged' for the revision the client holds."""
    job["rev"] = job.get("rev", 0) + 1


def save(job):
    with LOCK:
        touch(job)
        bridge.dump(job, job_dir(job["id"]) / "job.json")


def log(job, line):
    stamp = time.strftime("%H:%M:%S")
    text = f"{stamp} {line}"
    with LOCK:
        job["log"].append(text)
        del job["log"][:-LOG_KEEP]
        touch(job)
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
        if job.get("cancel"):                              # removed while busy, the server stopped before the worker got to it
            shutil.rmtree(p.parent, ignore_errors=True)
            continue
        if job.get("status") in BUSY:
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
    WORK.put(("analyse", jid, None))
    return job


def demo_info():
    """The bundled demo sheet: app/demo/demo.json with file, title, source and licence, plus the file's size; None when
    there is none."""
    p = DEMO / "demo.json"
    try:
        info = json.loads(p.read_text(encoding="utf-8"))
        f = DEMO / str(info.get("file") or "")
        if not f.is_file() or f.parent.resolve() != DEMO.resolve():
            return None
        info["size"] = f.stat().st_size
        return info
    except (OSError, ValueError):
        return None


def forget(jid):
    """Drop a job from memory and disk."""
    with LOCK:
        JOBS.pop(jid, None)
        SESSIONS.pop(jid, None)
        for k in [k for k in PAYLOADS if k[0] == jid]:
            PAYLOADS.pop(k, None)
    shutil.rmtree(job_dir(jid), ignore_errors=True)


def discard(job):
    """Forget a job and remove its folder (a removal requested while the worker was busy with it)."""
    forget(job["id"])
    print(f"[{job['id']}] discarded (removed while busy)")


def remember_session(jid, session):
    """Keep the analysed session in memory, the most recent SESSION_KEEP of them: a session holds the sheet rasters
    (tens of MB each); older ones come back from their pickle through _session()."""
    with LOCK:
        SESSIONS.pop(jid, None)
        SESSIONS[jid] = session
        while len(SESSIONS) > max(1, SESSION_KEEP):
            SESSIONS.pop(next(iter(SESSIONS)), None)


def run_job(job, busy_status, fail_status, work, finish):
    """The scaffold of the three runners: mark the job busy, run `work()` (its result goes to `finish(result)` under
    the lock), turn an AppError or any other exception into the job's error and `fail_status`."""
    with LOCK:
        job["status"] = busy_status
        job["error"] = None
    save(job)
    try:
        result = work()
    except bridge.AppError as ex:
        with LOCK:
            job["status"], job["error"], job["progress"] = fail_status, str(ex), None
        log(job, f"error: {ex}")
        save(job)
        return False
    except Exception as ex:                                # noqa: BLE001 - reported on the job, the worker goes on
        with LOCK:
            job["status"], job["error"], job["progress"] = fail_status, f"{type(ex).__name__}: {ex}", None
        log(job, f"error: {type(ex).__name__}: {ex}\n{traceback.format_exc()}")
        save(job)
        return False
    with LOCK:
        finish(result)
        job["progress"] = None
    save(job)
    return True


def worker():
    while True:
        kind, jid, payload = WORK.get()
        job = JOBS.get(jid)
        if job is None:
            WORK.task_done()
            continue
        if job.get("cancel"):                              # removed while it waited in the queue
            discard(job)
            WORK.task_done()
            continue
        try:
            if kind == "analyse":
                run_analysis(job)
            elif kind == "scale":
                run_scale(job, payload or {})
            elif kind == "extract":
                run_extraction(job)
        except Exception as ex:                            # noqa: BLE001 - the worker must survive every job
            try:
                with LOCK:
                    job["status"] = "error"
                    job["error"] = f"{type(ex).__name__}: {ex}"
                log(job, f"internal error\n{traceback.format_exc()}")
                save(job)
            except Exception:                              # noqa: BLE001 - e.g. the job folder is gone: the loop must go on
                traceback.print_exc()
        finally:
            if job.get("cancel"):
                discard(job)
            WORK.task_done()


def run_analysis(job):
    """Step 1: layout from the ink (bridge.analyse); the session is kept for the later steps."""
    d = job_dir(job["id"])
    upload = d / job["file"]
    t0 = time.time()
    with LOCK:
        job["progress"] = {"key": None, "stage": "0a normalisation, 1b layout"}

    def work():
        analysis, session = bridge.analyse(d, upload, lambda s: log(job, s))
        session["upload"] = upload
        remember_session(job["id"], session)
        saved = bridge.save_session(d, session)
        if saved is not True:
            log(job, f"note: the analysed session could not be saved ({saved}); after a server restart the analysis runs again")
        return analysis

    def finish(analysis):
        job["analysis"] = analysis
        job["status"] = "ready"
        job["times"]["analysis"] = round(time.time() - t0, 1)
        if not job["confirm"]:
            job["confirm"] = default_confirm(analysis)

    run_job(job, "analysing", "error", work, finish)


def _session(job, d):
    """The analysed session of the job: in memory, from its pickle, or analysed again (no pickle: the server was
    restarted before it was written). The job's analysis record is kept as it is in that last case: it carries the
    scale records and confirmations of the later steps, which a fresh layout pass does not."""
    session = SESSIONS.get(job["id"])
    if session is None:
        session = bridge.load_session(d)
        if session is not None:
            log(job, "analysed session loaded from disk")
    if session is None:
        log(job, "the analysed session is not on disk: the layout runs again first")
        analysis, session = bridge.analyse(d, d / job["file"], lambda s: log(job, s))
        bridge.save_session(d, session)
        with LOCK:
            if not job.get("analysis"):
                job["analysis"] = analysis
    session["upload"] = d / job["file"]
    remember_session(job["id"], session)
    return session


def run_scale(job, payload):
    """Step 2 for the drawings in payload["keys"]: their scale cues, the door-width search when payload["doors"]."""
    d = job_dir(job["id"])
    keys = list(payload.get("keys") or [])
    doors = bool(payload.get("doors"))
    t0 = time.time()
    with LOCK:
        job["progress"] = {"key": keys[0] if keys else None, "stage": "1c door search (segmenter)" if doors else "1c scale cues", "doors": doors}

    def work():
        session = _session(job, d)
        with LOCK:                                         # the worker edits copies; the live record is swapped at the end
            analysis, confirm = copy.deepcopy(job["analysis"]), copy.deepcopy(job["confirm"])
        done = bridge.scale_step(d, session, analysis, confirm, keys, lambda s: log(job, s), doors=doors)
        return analysis, done

    def finish(result):
        analysis, done = result
        job["analysis"] = analysis
        for a_pkg in analysis["packages"]:
            for dr in a_pkg["drawings"]:
                key = f"{a_pkg['id']}/{dr['id']}"
                if key in done:
                    c = job["confirm"].setdefault(key, {})
                    c["proposal_px_per_m"] = (dr.get("scale") or {}).get("px_per_m")
                    c["scale_pending"] = False
                    if dr.get("storey") and not c.get("storey"):
                        c["storey"] = dr["storey"]
        job["status"] = "ready"
        job["times"]["scale"] = round(job["times"].get("scale", 0) + time.time() - t0, 1)

    run_job(job, "scaling", "ready", work, finish)


def default_confirm(analysis):
    """What the pipeline proposes, as the starting point of the confirmations."""
    out = {}
    for p in analysis["packages"]:
        for dr in p["drawings"]:
            s = dr.get("scale") or {}
            out[f"{p['id']}/{dr['id']}"] = {"extract": dr["kind"] == bridge.FLOOR_PLAN, "polygon_px": None,
                                            "px_per_m": None, "scale_source": None, "measured": None,
                                            "storey": dr.get("storey"), "proposal_px_per_m": s.get("px_per_m"),
                                            "area_confirmed": False, "scale_confirmed": False, "scale_pending": True}
    return out


def run_extraction(job):
    """Step 3: stages 2-10 for the confirmed drawings (bridge.extract), the confirmations as they are at this moment."""
    d = job_dir(job["id"])
    t0 = time.time()
    with LOCK:
        job["results"] = None
        job["progress"] = {"key": None, "stage": "starting"}
        for k in [k for k in PAYLOADS if k[0] == job["id"]]:
            PAYLOADS.pop(k, None)

    def progress(key, stage):
        with LOCK:
            job["progress"] = {"key": key, "stage": stage}
        log(job, f"  [{key}] {stage}")

    def work():
        session = _session(job, d)
        with LOCK:                                         # a confirm posted meanwhile must not reach into the run
            confirm, analysis = copy.deepcopy(job["confirm"]), copy.deepcopy(job["analysis"])
        return bridge.extract(d, session, confirm, lambda s: log(job, s), progress, analysis=analysis)

    def finish(results):
        job["results"] = results
        job["times"]["extraction"] = round(time.time() - t0, 1)
        failed = [k for k, r in results.items() if r.get("error")]
        done = [k for k, r in results.items() if r.get("outputs")]
        job["status"] = "done" if done or not failed else "error"
        if failed:
            job["error"] = "; ".join(f"{k}: {results[k]['error']}" for k in failed)

    if run_job(job, "running", "error", work, finish):
        log(job, f"extraction finished in {job['times']['extraction']} s")


# ---------------------------------------------------------------------------------------------------------------------
# Results for the viewer

def results_payload(job, key):
    """One drawing's results for the viewer: the output JSON reduced to what the screen draws. Cached per file time,
    since the JSON of a large sheet runs to megabytes and the screen asks for it on every render."""
    r = (job.get("results") or {}).get(key)
    if not r:
        raise NotFound(f"no results for {key}")
    if not r.get("outputs"):
        return {"key": key, "error": r.get("error"), "skipped": r.get("skipped"), "record": r}
    out = job_dir(job["id"]) / "out"
    src = out / r["outputs"]["json"]
    stamp = (src.stat().st_mtime_ns, (job_dir(job["id"]) / "review.json").stat().st_mtime_ns if (job_dir(job["id"]) / "review.json").exists() else 0)
    with LOCK:
        hit = PAYLOADS.get((job["id"], key))
    if hit and hit[0] == stamp:
        return hit[1]
    payload = _results_payload(job, key, r, out, src)
    with LOCK:
        PAYLOADS[(job["id"], key)] = (stamp, payload)
        while len(PAYLOADS) > PAYLOAD_KEEP:
            PAYLOADS.pop(next(iter(PAYLOADS)), None)
    return payload


def _results_payload(job, key, r, out, src):
    data = json.loads(src.read_text(encoding="utf-8"))
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
        """A file with an ETag from its size and time: the browser revalidates and gets a 304 while it is unchanged
        (the vendored 3D library and the sheet images are the large ones)."""
        path = Path(path)
        if not path.is_file():
            return self.send_error_json(404, f"{path.name} not found")
        st = path.stat()
        etag = f'"{st.st_size:x}-{st.st_mtime_ns:x}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(st.st_size))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("ETag", etag)
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
        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except (ValueError, UnicodeDecodeError) as ex:
            raise bridge.AppError(f"Malformed JSON body: {ex}") from ex
        if not isinstance(data, dict):
            raise bridge.AppError("The JSON body must be an object.")
        return data

    def job(self, jid):
        job = JOBS.get(jid)
        if job is None or job.get("cancel"):                 # removed, or removed while busy and not yet discarded
            raise NotFound(f"sheet {jid} not found")
        return job

    def public_job(self, job):
        with LOCK:
            j = dict(job)
            j["log"] = job["log"][-40:]
        return j

    # --- routing
    def do_GET(self):
        try:
            self.route()
        except NotFound as ex:
            self.send_error_json(404, str(ex))
        except bridge.AppError as ex:
            self.send_error_json(400, str(ex))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as ex:                            # noqa: BLE001
            traceback.print_exc()
            self.send_error_json(500, f"{type(ex).__name__}: {ex}")

    do_POST = do_DELETE = do_GET

    def route(self):
        method = self.command
        u = urlparse(self.path)
        path, q = u.path, parse_qs(u.query)
        if method == "GET" and path in ("/", "/index.html"):
            return self.send_file(APP / "index.html")
        if method == "GET" and path.split("/")[1] in ASSETS:
            folder = APP / path.split("/")[1]
            target = (folder / path[len(path.split("/")[1]) + 2:]).resolve()   # js/..., css/...; nothing outside those folders
            if not target.is_relative_to(folder.resolve()):
                raise NotFound("not found")
            return self.send_file(target)
        if method == "GET" and path == "/api/health":
            if not HEALTH or "refresh" in q:                # the check imports the libraries once; ?refresh=1 looks again
                HEALTH.clear()
                HEALTH.update(bridge.check())
            return self.send_json(HEALTH)
        if path == "/api/demo":
            info = demo_info()
            if method == "GET":
                return self.send_json({"demo": info})
            if method == "POST":
                if info is None:
                    raise bridge.AppError("No demo sheet is bundled with this server (app/demo/).")
                job = new_job(info["file"], (DEMO / info["file"]).read_bytes())
                return self.send_json({"id": job["id"], "name": job["name"], "status": job["status"]})
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
                        for j in sorted(JOBS.values(), key=lambda j: j["id"], reverse=True) if not j.get("cancel")]
            return self.send_json({"sheets": rows})
        m = re.match(r"^/api/sheets/([\w\-]+)(?:/(.*))?$", path)
        if not m:
            raise NotFound(f"no route for {method} {path}")
        job = self.job(m.group(1))
        sub = m.group(2) or ""
        if method == "GET" and sub == "":
            rev = (q.get("rev") or [None])[0]
            if rev is not None and rev.isdigit() and int(rev) == job.get("rev", 0):
                return self.send_json({"id": job["id"], "rev": job["rev"], "unchanged": True})
            return self.send_json(self.public_job(job))
        if method == "DELETE" and sub == "":
            if job["status"] in BUSY:                      # the worker cannot be interrupted: it discards the job when done
                with LOCK:
                    job["cancel"] = True
                log(job, "removed while busy: discarded as soon as the worker is done with it")
                return self.send_json({"deleted": job["id"], "pending": True})
            forget(job["id"])
            return self.send_json({"deleted": job["id"]})
        if method == "POST" and sub == "confirm":
            data = self.read_json()
            confirm = data.get("confirm") or {}
            if not isinstance(confirm, dict) or not all(isinstance(v, dict) for v in confirm.values()):
                raise bridge.AppError("confirm must map '<package>/<drawing>' keys to objects.")
            with LOCK:
                for key, val in confirm.items():
                    cur = job["confirm"].setdefault(key, {})
                    cur.update(val)
            save(job)
            return self.send_json({"confirm": job["confirm"]})
        if method == "POST" and sub == "scale":
            data = self.read_json()
            keys = [k for k in (data.get("keys") or []) if isinstance(k, str)]
            if job["status"] in BUSY:
                raise bridge.AppError(f"The sheet is busy ({job['status']}).")
            if not job.get("analysis"):
                raise bridge.AppError("The sheet is not analysed yet.")
            known = {f"{p['id']}/{d['id']}" for p in job["analysis"]["packages"] for d in p["drawings"]}
            keys = [k for k in keys if k in known]
            if not keys:
                raise bridge.AppError("scale needs the keys of drawings of this sheet.")
            with LOCK:
                job["status"] = "scaling"
                job["error"] = None
                job["progress"] = {"key": keys[0], "stage": "waiting for the worker", "doors": bool(data.get("doors"))}
            save(job)
            WORK.put(("scale", job["id"], {"keys": keys, "doors": bool(data.get("doors"))}))
            return self.send_json({"status": "scaling", "keys": keys})
        if method == "POST" and sub == "run":
            if job["status"] in BUSY:
                raise bridge.AppError(f"The sheet is busy ({job['status']}).")
            if not any(c.get("extract") for c in job["confirm"].values()):
                raise bridge.AppError("No drawing is selected for extraction.")
            with LOCK:
                job["status"] = "queued"
                job["error"] = None
                job["progress"] = {"key": None, "stage": "waiting for the worker"}
            save(job)
            WORK.put(("extract", job["id"], None))
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
            download = name if name.endswith((".json", ".dxf", ".xlsx", ".ifc", ".csv", ".png")) and "download" in q else None
            return self.send_file(path, download)
        if method == "GET" and sub == "live":
            key = (q.get("key") or [""])[0]
            if not re.match(r"^[\w\-][\w\-.]*/[\w\-]+$", key):
                raise bridge.AppError("live needs key=<package>/<drawing>.")
            p = job_dir(job["id"]) / "out" / bridge.live_name(key)
            if not p.exists():
                return self.send_json({"stage": None})
            try:
                return self.send_json(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):                   # being written right now: ask again on the next poll
                return self.send_json({"stage": None})
        if method == "GET" and sub == "log":
            p = job_dir(job["id"]) / "log.txt"
            return self.send_text(p.read_text(encoding="utf-8") if p.exists() else "\n".join(job["log"]))
        if method == "POST" and sub == "review":
            data = self.read_json()
            key = data.get("key")
            if not key:
                raise bridge.AppError("review needs a drawing key")
            rooms = data.get("rooms") or {}
            if not isinstance(key, str) or not isinstance(rooms, dict) or not all(isinstance(v, dict) for v in rooms.values()):
                raise bridge.AppError("review needs {key, rooms: {<room id>: {checked: bool, ...}}}.")
            p = job_dir(job["id"]) / "review.json"
            with LOCK:                                     # two ticks in a row must not lose each other
                try:
                    review = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
                except ValueError:
                    review = {}
                entry = review.setdefault(key, {"rooms": {}})
                for rid, val in rooms.items():
                    entry["rooms"].setdefault(rid, {}).update(val)
                entry["updated"] = now()
                bridge.dump(review, p)
            return self.send_json({"review": entry})
        raise NotFound(f"no route for {method} {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--warm", action="store_true", help="load the OCR engine and the segmenter at start-up, in the background, so the first sheet does not wait for them")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    DATA.mkdir(parents=True, exist_ok=True)
    load_jobs()
    threading.Thread(target=worker, daemon=True, name="pipeline-worker").start()
    if a.warm:
        def warm():
            for name, fn in (("OCR engine", bridge.engine), ("segmenter", bridge.model)):
                try:
                    t0 = time.time()
                    fn()
                    print(f"warm-up: {name} ready in {time.time() - t0:.1f} s", flush=True)
                except Exception as ex:                    # noqa: BLE001 - the health check will say the same
                    print(f"warm-up: {name} not available: {ex}", flush=True)
        threading.Thread(target=warm, daemon=True, name="warm-up").start()
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
