# Code Review of the Workflow App

*8 October 2026 · [app/](../../app/README.md): `server.py`, `bridge.py`, the single-page script and its structure. Reviewed for bugs, robustness and maintainability after the [design review](2026-10-08-app-design-review.md) of the same day. Every finding carries its decision; all of them are implemented and verified (headless screenshots of the five screens are pixel-identical before and after the split; the new server tests pass).*

## 1. Bugs

| # | Where | Finding | Fix |
|---|---|---|---|
| B1 | `server.py` worker | The one worker thread caught every job exception, but the handler itself could raise: deleting a sheet while its job ran removed the folder, `save()` then failed inside the `except`, the thread died, and every later job stayed "queued" with no message. | The handler is guarded (a second failure is printed, the loop goes on), and `DELETE` refuses a busy job with a 400 that names the state. A test deletes a busy job, lets it finish, and checks that a second job still runs. |
| B2 | `server.py` errors | `KeyError` was the "not found" signal, so a `KeyError` from a programming error (an output JSON without `stairs`, say) was answered as `404 'stairs'`. | A `NotFound` exception for sheets, routes and results; any other error is a 500 with its message, as it should be. |
| B3 | `server.py` input | A malformed JSON body or a `confirm` entry of the wrong shape ended in a 500 (`ValueError`, `TypeError`). | 400 with a message; `confirm` must map keys to objects. Tested. |
| B4 | `server.py` static files | `/static/<path>` kept only the file name, so no subfolder could be served, and the safety of the route rested on that flattening. | Any path below `static/` is served after `resolve()`, with a 404 for anything outside. Tested with `/static/../server.py`. |
| B5 | `server.py` downloads | Only `.json` and `.dxf` got the attachment header; the Excel, IFC, CSV and PNG links relied on the anchor's `download` attribute. | All export types. |
| B6 | client polling | When any job changed, the step on show was re-rendered, also when the change came from another sheet's job: the user's handle positions or measurement points were lost. | `onJobsChanged(changed)` receives the ids that changed; a step re-renders only when its own sheet's job did. |
| B7 | client state | `storeyRows()` sorted by and rewrote `P.order` on every call, from render paths and from `projectName()`: a getter with side effects. | Gone with the storey feature: `drawingRows()` follows sheet, page and drawing order; `P.order` is dropped from the project record. |
| B8 | client, scale step | A page without drawings left `dr` undefined and the step crashed. | The step returns to the building-area step for that page. |
| B9 | client, run step | The earlier `Cannot access 'listEl'` error: the poll refresh called a closure of a different render pass that read `const`s declared later. | The list, button and note are found by id when the poll refreshes them. |
| B10 | `server.py` | `route(self, method)` ignored its parameter and read `self.command`. | `route(self)`. |

## 2. Structure

The 1,040-line `static/app.js` is now ten ES modules under `static/js/` (since moved up to `app/js/`, with the stylesheets in `app/css/`), loaded with `<script type="module">`:

| Module | Lines | Holds |
|---|---|---|
| `core.js` | 66 | constants, the design-token colours for the canvas (`T('--walls')`), `h`, `$`, `put`, `esc`, `api`, `fmt` |
| `state.js` | 100 | the project record `P` (localStorage), the job records `J`, polling with a listener, access to the analysis (`keyOf`, `confOf`, `selectedDrawings`, `drawingRows`, `firstUnconfirmed`) |
| `viewer.js` | 205 | the canvas viewer (pan, zoom, overlays), `RectTool`, `MeasureTool`, polygon helpers |
| `ui.js` | 65 | `mount`, the screen registry and `go`, the current viewer (`useViewer`), `badge`, `stepHead`, `loadPreview`, `errorBox`, `projectName` |
| `screens/upload.js` | 98 | board 1c |
| `screens/area.js` | 95 | board 1d, the waiting screen, the walk to the next package |
| `screens/scale.js` | 182 | boards 2a–2c |
| `screens/run.js` | 93 | board 1f |
| `screens/results.js` | 172 | boards 1g, 1h |
| `app.js` | 36 | registers the screens, wires the top bar, boots |

*Later the same day:* the run step was folded into the results screen (`run.js` is gone): confirming the scale starts the extraction, and `results.js` shows the progress per drawing until the results are there. The keys `storeys` and `run` of saved projects open the results.

Decisions behind it:
- **No module cycle.** Screens need the router and the router needs the screens; the screens register themselves (`register(name, fn)` in `app.js`) and `ui.js` never imports a screen.
- **No assignment to imported bindings** (ES modules forbid it, and the first run of the split caught two): the screen name is set through `setScreen()`, a new project through `resetProject()`, and each screen gets its viewer from `useViewer(stage)`, which destroys the previous one.
- **Shared pieces instead of copies:** the status badge (built four times in three screens), the step head (progress bars, eyebrow, title; three times), the preview loading with its error box (three times), polygon scaling (five times).
- The imports were generated from the declarations and then checked (every imported name is exported by its module, every exported name used elsewhere is imported); `node --check` parses each module.

## 3. Robustness and Maintainability

- **Tests:** `app/tests/` runs `server.py` against a stub `bridge` (no model, OCR or torch; 3 s): static files and the traversal guard, upload validation, analysis through the worker, confirm merging and validation, run, results, log, list, delete, the busy-delete guard with the worker surviving, 404s and malformed JSON. `python -m pytest app/tests -q` from the repository root.
- **Fatal errors stay visible:** the inline handler in `index.html` shows any uncaught error or rejected promise with file and line in a box; this is how the two module-binding errors of the split were found.
- **Server:** the job dictionary is edited under one lock; the worker is a daemon thread fed by a queue; jobs found "running" at start-up are marked as interrupted.
- **Bridge:** unchanged in this pass; it is the only module that touches the pipeline and follows the renamed `scripts/` folder.

## 4. Open Items

- Polling every 1.5 s could become a server-sent event stream; fine for one user on one machine.
- A `confirm` posted while an extraction runs edits records the worker is reading (the worker takes a shallow copy). A single user cannot do both at once through the UI; a deep copy at run start would close it.
- `results_payload` re-reads and filters the output JSON on every request; a cache keyed by file time would help on very large sheets.
- The analysed session is a pickle in the job folder (local trust boundary only; never load one from elsewhere).
- `alert()` for upload and confirm errors; an inline message component would fit the design.
- No client-side tests yet; the five screens are verified by headless screenshots against deep links of processed jobs. A Playwright run over the same links would make that repeatable.
- The language switch and the settings button in the top bar are study decoration (see the design review).
