"""Code review 2: the revision counter of the job record, ETags on files, the review body validation, the analysis
kept when the session has to be rebuilt, the bounded session cache."""
import json
import time
import urllib.error
import urllib.request

from test_server import call, wait_status


def head(base, path, headers=None):
    req = urllib.request.Request(base + path, method="GET", headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_rev_makes_polls_cheap(served):
    base, server, _ = served
    status, body = call(base, "/api/upload?name=plan.png", "POST", b"\x89PNG fake", {"Content-Type": "application/octet-stream"})
    jid = body["id"]
    job = wait_status(base, jid, ("ready",))
    assert isinstance(job["rev"], int)
    status, again = call(base, f"/api/sheets/{jid}?rev={job['rev']}")
    assert status == 200 and again == {"id": jid, "rev": job["rev"], "unchanged": True}
    # a confirm changes the record: the next poll with the old revision gets the full job again
    call(base, f"/api/sheets/{jid}/confirm", "POST", {"confirm": {"p1/d1": {"area_confirmed": True}}})
    status, fresh = call(base, f"/api/sheets/{jid}?rev={job['rev']}")
    assert status == 200 and "unchanged" not in fresh and fresh["rev"] > job["rev"] and fresh["confirm"]["p1/d1"]["area_confirmed"] is True


def test_files_carry_etags(served):
    base, server, _ = served
    status, headers, body = head(base, "/js/app.js")
    assert status == 200 and headers.get("ETag") and headers.get("Cache-Control") == "no-cache"
    status, headers2, body2 = head(base, "/js/app.js", {"If-None-Match": headers["ETag"]})
    assert status == 304 and not body2


def test_review_body_is_validated(served):
    base, server, _ = served
    status, body = call(base, "/api/upload?name=plan.png", "POST", b"\x89PNG fake", {"Content-Type": "application/octet-stream"})
    jid = body["id"]
    wait_status(base, jid, ("ready",))
    assert call(base, f"/api/sheets/{jid}/review", "POST", {"key": "p1/d1", "rooms": {"r01": 5}})[0] == 400
    assert call(base, f"/api/sheets/{jid}/review", "POST", {"key": "p1/d1", "rooms": "x"})[0] == 400
    status, body = call(base, f"/api/sheets/{jid}/review", "POST", {"key": "p1/d1", "rooms": {"r01": {"checked": True}}})
    assert status == 200 and body["review"]["rooms"]["r01"]["checked"] is True


def test_analysis_survives_a_rebuilt_session(served, monkeypatch):
    base, server, _ = served
    status, body = call(base, "/api/upload?name=plan.png", "POST", b"\x89PNG fake", {"Content-Type": "application/octet-stream"})
    jid = body["id"]
    wait_status(base, jid, ("ready",))
    call(base, f"/api/sheets/{jid}/scale", "POST", {"keys": ["p1/d1"]})
    job = wait_status(base, jid, ("ready",))
    assert job["analysis"]["packages"][0]["drawings"][0]["scale"]["px_per_m"] == 50.0
    # the session is gone (restart without a pickle): the layout runs again, but the scale record of step 2 stays
    server.SESSIONS.clear()
    import bridge
    original = bridge.analyse
    monkeypatch.setattr(bridge, "analyse", lambda job_dir, upload, log, want_model=False: (json.loads(json.dumps(original(job_dir, upload, log)[0])), {"packages": []}))
    call(base, f"/api/sheets/{jid}/run", "POST", {})
    job = wait_status(base, jid, ("done", "error"))
    assert job["analysis"]["packages"][0]["drawings"][0]["scale"]["px_per_m"] == 50.0


def test_session_cache_is_bounded(served, monkeypatch):
    base, server, _ = served
    monkeypatch.setattr(server, "SESSION_KEEP", 2)
    ids = []
    for k in range(3):
        status, body = call(base, f"/api/upload?name=plan{k}.png", "POST", b"\x89PNG fake", {"Content-Type": "application/octet-stream"})
        ids.append(body["id"])
        wait_status(base, body["id"], ("ready",))
    assert len(server.SESSIONS) == 2 and ids[0] not in server.SESSIONS and ids[2] in server.SESSIONS
