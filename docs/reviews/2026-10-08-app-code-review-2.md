# Second Code Review of the Workflow App

*8 October 2026 · [app/](../../app/README.md) after the three-step split, the live results screen and the 3D view. Reviewed as a senior developer and as the person who has to keep the ML side running: bugs, performance, robustness, and where code could be merged, with the pipeline under `pilot/v2-pipeline` left as it is (it improves on its own schedule; the app only calls it). Every finding carries its decision; all are implemented unless the row says otherwise. Verified by the app tests (12 against a stub bridge, one of them in a browser), the pilot suite (236) and a browser walk of the demo sheet.*

## 1. Findings

### 1.1 Bugs

| # | Where | Finding | Decision |
|---|---|---|---|
| B1 | `server.py` `_session` | When the pickled session was missing (a server stopped before it was written) the extraction ran the layout again and **replaced the job's analysis** with the layout-only record: the scale records of step 2 and the titles were gone, the scale step showed "pending" again. | **Done.** The session is rebuilt, the analysis record is kept. Tested. |
| B2 | `server.py` runners | The worker read `dict(job["confirm"])`, a shallow copy: a confirmation posted while a run was in progress reached into the records the pipeline was reading (an open item of the first review). | **Done.** Deep copies of the confirmations and the analysis at the start of every run. |
| B3 | `server.py` review | `rooms: {"r01": 5}` or `rooms: "x"` in a review body ended in a `TypeError` and a 500; two ticks posted at once could lose each other (read, modify, write without the lock). | **Done.** The body is validated (400 with the expected shape); the file is updated under the lock. Tested. |
| B4 | `server.py` health | The health check was cached for the life of the process: a model installed afterwards stayed "missing" until a restart. | **Done.** `GET /api/health?refresh=1` looks again. |
| B5 | `results.js` | The progress panel read the job record captured at render time (fixed during the UX review, recorded here for completeness). | Done earlier. |

### 1.2 Performance

| # | Where | Finding | Decision |
|---|---|---|---|
| P1 | polling | `GET /api/sheets/<id>` serialised the whole job on every poll (1.5 s), analysis included: the regions and drawings with their polygons, hundreds of kilobytes per poll per sheet, parsed again by the browser each time. | **Done.** The job record carries a revision (`rev`, bumped on every save and log line). The client sends the revision it holds and gets `{"unchanged": true}` (90 bytes) while nothing changed. Tested. |
| P2 | files | Every file went out with `Cache-Control: no-store`: the 844 KB of three.js on every switch to 3D, the stylesheets and modules on every load, the sheet images on every render. | **Done.** Files carry an ETag (size and time); `If-None-Match` gets a 304; `Cache-Control: no-cache` so the browser always revalidates and never shows a stale file. Tested. |
| P3 | memory | `SESSIONS` kept every analysed session of the process in memory, and a session holds the sheet rasters (a 6800 x 4400 sheet is 90 MB). Ten uploads, a gigabyte. | **Done.** The most recent three stay in memory (`FPX_APP_SESSIONS`); the others come back from their pickle when needed. Tested. |
| P4 | results | `results_payload` parsed the drawing's output JSON (megabytes on a large sheet) on every request of the results screen. | **Done.** Cached per job and drawing, keyed by the file times of the output and the review, eight entries. |
| P5 | OCR cache | The OCR cache lived in the job folder, so the same sheet uploaded twice (the demo, a corrected upload) paid its OCR twice; the cache key already carries the raster digest and the OCR settings. | **Done.** One cache for all jobs, `app/data/ocr_cache` (`FPX_APP_OCR_CACHE`). The demo's second run skips 25-45 s of OCR. |
| P6 | first run | The OCR engine and the segmenter load on first use, inside the first sheet's steps (a few seconds each, longer on a cold disk). | **Done.** `python app/server.py --warm` loads both in the background at start-up; off by default, since the model takes memory the upload page does not need. |
| P7 | 3D view | The render loop ran at the display rate while the view was open, for the damping of the controls, also with nothing moving. | **Done.** The loop runs from a pointer or wheel event until the controls settle (a second), then sleeps. |
| P8 | ML, noted | The drawing OCR (25-45 s on the demo sheet) and the segmentation (25 s) dominate a run; both are pipeline matters. ONNX Runtime's thread pool is not capped while torch is capped at `FPX_APP_THREADS`; the one 34-minute OCR seen during the UX review is the reason to keep an eye on it. | **Open**, pipeline: cap the ONNX intra-op threads in `fpx.pipeline.ocr_engine` when the shared-machine setting is on. |

### 1.3 Robustness

| # | Where | Finding | Decision |
|---|---|---|---|
| R1 | worker | One worker thread, so a long extraction holds back the scale step of a second sheet. By design for one user on one machine (the pipeline is CPU-bound and the model is loaded once); the queue order is visible in the status. | No change; recorded. |
| R2 | restart | Jobs found busy at start-up are marked as interrupted with a message; a job removed while busy is discarded at start-up if the server stopped first. | Already so. |
| R3 | uploads | The body is read into memory up to 300 MB; the file name is reduced to a safe stem; the file endpoint takes the base name only. | Already so. |
| R4 | import map | Browsers without import maps (Safari before 16.4) run the 2D app as before and fail only when 3D is chosen, with a notice. | Acceptable for a local tool; recorded. |
| R5 | tests | No test ran the page in a browser; the Playwright walks of the UX review were scratch scripts. | **Done.** `app/tests/test_browser.py`: the page boots against the stub bridge without a script error, the landing page and its demo link render, no control lacks a name, the demo moves on to step 1. Skipped where Playwright is not installed. |
| R6 | `index.html` | The page had no icon, so every browser asked the server for `/favicon.ico` and got a 404 (the smoke test caught it). | **Done.** An inline SVG icon. |

### 1.4 Complexity

| # | Where | Finding | Decision |
|---|---|---|---|
| C1 | `server.py` | The three runners repeated the same scaffold: mark busy, save, try, two except blocks, save, finish. | **Done.** `run_job(job, busy, fail, work, finish)` holds the scaffold; the runners keep only their work and their finish. |
| C2 | client | `try { await postJSON(...) } catch (e) { notify(e.message) }` six times across the screens. | **Done.** `post(path, body)` in `ui.js` reports and returns null; the callers carry on from the result. |
| C3 | client | Two waiting screens with the same card: the upload page while sheets are analysed and a step opened before its sheet is analysed. | **Done.** One `busyScreen(jobs, {abort, back})` in `ui.js`; the abort moved next to it. |
| C4 | `server.py` | Deleting a job and discarding a cancelled one removed the same three things in two places. | **Done.** `forget(jid)`. |
| C5 | `results.js` | At 390 lines the module is the largest of the app, but it is one screen with one state object and its functions are short; splitting the live poll or the 3D bridge out would add imports without removing complexity. | No change; recorded. |

## 2. What changed, by file

- `app/server.py`: `touch`/`rev`, `?rev=` on the job endpoint, ETags and 304 in `send_file`, `remember_session` with `SESSION_KEEP`, `forget`, `run_job` and the three runners on it, deep copies for the worker, the payload cache, the review validation under the lock, `?refresh=1` on the health check, `--warm`.
- `app/bridge.py`: `CACHE_DIR` for the OCR cache shared across jobs.
- `app/js/state.js`: `refreshJob` sends the revision and keeps the record on "unchanged".
- `app/js/ui.js`: `post`, `busyScreen`, `abortProject`.
- `app/js/screens/{upload,area,scale,results}.js`: on the helpers above; `area.js` keeps `renderWaiting` as a thin wrapper for the scale screen.
- `app/js/view3d.js`: the render loop sleeps when nothing moves.
- `app/tests/test_review2.py` (5 tests), `app/tests/test_browser.py` (1, optional).

## 3. Open items

- P8: cap the ONNX Runtime threads next to the torch threads (pipeline side).
- The polling could become one server-sent event stream; with the revision counter a poll costs almost nothing, so this is no longer pressing.
- `results_payload` reduces the output JSON on the server; the client could ask for the layers it has switched on only, which matters on sheets with hundreds of rooms.
- The OCR cache folder grows without bound; a size cap with least-recently-used removal belongs to the day it becomes a problem.
