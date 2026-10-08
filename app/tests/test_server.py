"""Routes and job model of app/server.py with the stub bridge of conftest.py: static files and the traversal guard,
upload validation, the analysis through the worker, confirm validation, the deferred delete of a busy job, the demo
sheet, 404s and malformed JSON.

    python -m pytest app/tests -q      (from the repository root)
"""
import json
import time
import urllib.error
import urllib.request


def call(base, path, method="GET", body=None, headers=None):
    data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers or ({"Content-Type": "application/json"} if body is not None and not isinstance(body, bytes) else {}))
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if r.headers.get("content-type", "").startswith("application/json") else raw)
    except urllib.error.HTTPError as e:
        raw = e.read()
        return e.code, (json.loads(raw) if e.headers.get("content-type", "").startswith("application/json") else raw)


def wait_status(base, jid, wanted, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        status, job = call(base, f"/api/sheets/{jid}")
        if job["status"] in wanted:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {jid} did not reach {wanted}")


def test_static_files_and_traversal_guard(served):
    base, server, _ = served
    status, body = call(base, "/css/tokens.css")
    assert status == 200 and b"--accent" in body
    status, body = call(base, "/js/app.js")
    assert status == 200 and b"import" in body
    assert call(base, "/js/../server.py")[0] == 404
    assert call(base, "/css/../tests/conftest.py")[0] == 404
    assert call(base, "/css/nothing.css")[0] == 404


def test_upload_rejects_unknown_type_and_empty_body(served):
    base, server, _ = served
    status, body = call(base, "/api/upload?name=notes.txt", "POST", b"hello", {"Content-Type": "application/octet-stream"})
    assert status == 400 and "Unsupported file type" in body["error"]
    status, body = call(base, "/api/upload?name=plan.png", "POST", b"", {"Content-Type": "application/octet-stream"})
    assert status == 400 and "empty" in body["error"]


def test_upload_analysis_confirm_run_and_delete(served):
    base, server, _ = served
    status, body = call(base, "/api/upload?name=plan.png", "POST", b"\x89PNG fake", {"Content-Type": "application/octet-stream"})
    assert status == 200 and body["status"] == "analysing"
    jid = body["id"]
    job = wait_status(base, jid, ("ready", "error"))
    assert job["status"] == "ready", job.get("error")
    assert job["analysis"]["packages"][0]["drawings"][0]["id"] == "d1"
    assert job["confirm"]["p1/d1"]["extract"] is True and job["confirm"]["p1/d1"]["proposal_px_per_m"] is None   # step 2 fills it
    assert job["confirm"]["p1/d1"]["scale_pending"] is True
    # confirm: merged per key; a non-object entry is refused
    status, body = call(base, f"/api/sheets/{jid}/confirm", "POST", {"confirm": {"p1/d1": {"area_confirmed": True, "px_per_m": 48.5}}})
    assert status == 200 and body["confirm"]["p1/d1"]["px_per_m"] == 48.5 and body["confirm"]["p1/d1"]["extract"] is True
    # step 2: the scale cues of the confirmed drawing, then once more with the door search
    assert call(base, f"/api/sheets/{jid}/scale", "POST", {"keys": ["p9/d9"]})[0] == 400
    status, body = call(base, f"/api/sheets/{jid}/scale", "POST", {"keys": ["p1/d1"]})
    assert status == 200 and body["status"] == "scaling"
    job = wait_status(base, jid, ("ready", "error"))
    assert job["analysis"]["packages"][0]["drawings"][0]["scale"]["px_per_m"] == 50.0
    assert job["confirm"]["p1/d1"]["proposal_px_per_m"] == 50.0 and job["confirm"]["p1/d1"]["scale_pending"] is False
    status, body = call(base, f"/api/sheets/{jid}/scale", "POST", {"keys": ["p1/d1"], "doors": True})
    assert status == 200
    job = wait_status(base, jid, ("ready", "error"))
    assert job["analysis"]["packages"][0]["drawings"][0]["scale"]["px_per_m"] == 48.0
    assert call(base, f"/api/sheets/{jid}/confirm", "POST", {"confirm": {"p1/d1": 5}})[0] == 400
    assert call(base, f"/api/sheets/{jid}/confirm", "POST", b"{not json", {"Content-Type": "application/json"})[0] == 400
    # run: the stub extraction fails per drawing, which the job reports as an error with the message
    status, body = call(base, f"/api/sheets/{jid}/run", "POST", {})
    assert status == 200 and body["status"] == "queued"
    job = wait_status(base, jid, ("done", "error"))
    assert job["status"] == "error" and "stub: no pipeline" in job["error"]
    assert call(base, f"/api/sheets/{jid}/results?key=p1/d1")[1]["error"] == "stub: no pipeline"
    assert call(base, f"/api/sheets/{jid}/results?key=p1/d9")[0] == 404
    # the live view: nothing until the bridge writes it, then the file as is
    assert call(base, f"/api/sheets/{jid}/live?key=p1/d1")[1] == {"stage": None}
    assert call(base, f"/api/sheets/{jid}/live?key=../x")[0] == 400
    (server.SHEETS / jid / "out").mkdir(exist_ok=True)
    (server.SHEETS / jid / "out" / "p1-d1_live.json").write_text(json.dumps({"stage": "walls", "walls": [[[0, 0], [1, 0], [1, 1]]]}), encoding="utf-8")
    assert call(base, f"/api/sheets/{jid}/live?key=p1/d1")[1]["stage"] == "walls"
    assert call(base, f"/api/sheets/{jid}/log")[0] == 200
    # the sheet list, then delete
    status, body = call(base, "/api/sheets")
    assert status == 200 and [s["id"] for s in body["sheets"]] == [jid]
    assert call(base, f"/api/sheets/{jid}", "DELETE")[0] == 200
    assert call(base, f"/api/sheets/{jid}")[0] == 404
    assert not (server.SHEETS / jid).exists()


def test_busy_job_delete_is_deferred_and_worker_survives(served):
    base, server, gate = served
    gate.clear()                                           # the stub analysis waits: the job stays 'analysing'
    status, body = call(base, "/api/upload?name=plan.png", "POST", b"\x89PNG fake", {"Content-Type": "application/octet-stream"})
    jid = body["id"]
    assert call(base, f"/api/sheets/{jid}")[1]["status"] == "analysing"
    assert call(base, f"/api/sheets/{jid}/run", "POST", {})[0] == 400
    # the abort: the job is gone for the client at once, its folder once the worker is done with it
    status, body = call(base, f"/api/sheets/{jid}", "DELETE")
    assert status == 200 and body["pending"] is True
    assert call(base, f"/api/sheets/{jid}")[0] == 404
    assert jid not in [s["id"] for s in call(base, "/api/sheets")[1]["sheets"]]
    gate.set()
    t0 = time.time()
    while (server.SHEETS / jid).exists() and time.time() - t0 < 10:
        time.sleep(0.05)
    assert not (server.SHEETS / jid).exists() and jid not in server.JOBS
    # a second job after the first: the single worker is still alive
    status, body = call(base, "/api/upload?name=plan2.png", "POST", b"\x89PNG fake", {"Content-Type": "application/octet-stream"})
    assert wait_status(base, body["id"], ("ready",))["status"] == "ready"


def test_demo_sheet(served, tmp_path, monkeypatch):
    base, server, _ = served
    demo = tmp_path / "demo"
    demo.mkdir()
    monkeypatch.setattr(server, "DEMO", demo)              # an empty demo folder: no demo
    assert call(base, "/api/demo")[1]["demo"] is None
    assert call(base, "/api/demo", "POST", {})[0] == 400
    (demo / "house.png").write_bytes(b"\x89PNG fake")
    (demo / "demo.json").write_text(json.dumps({"file": "house.png", "title": "A house", "source": "test", "licence": "CC0"}), encoding="utf-8")
    monkeypatch.setattr(server, "DEMO", demo)
    status, body = call(base, "/api/demo")
    assert status == 200 and body["demo"]["title"] == "A house" and body["demo"]["size"] == 9
    status, body = call(base, "/api/demo", "POST", {})
    assert status == 200 and body["name"] == "house.png"
    assert wait_status(base, body["id"], ("ready",))["status"] == "ready"


def test_unknown_routes(served):
    base, server, _ = served
    assert call(base, "/api/nothing")[0] == 404
    assert call(base, "/api/sheets/no-such-id")[0] == 404
    assert call(base, "/api/sheets/no-such-id/results?key=x")[0] == 404
