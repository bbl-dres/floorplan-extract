# Floor plan workflow app (local prototype)

A working prototype of the workflow designed in `docs/wireframes/261007_Viewer and Workflow UX study.html`:
upload, building area, scale, run, results and export (the storey naming and ordering of the study is left out for now). Processing uses the pilot v2 pipeline
(`pilot/v2-pipeline`, package `fpx`). Everything runs on this machine; nothing is uploaded anywhere.

## Start

From the repository root:

    python app/server.py            # or: python app/server.py --port 8765
    Floor plan workflow app: http://127.0.0.1:8765/

Open the printed address in a browser. `app/index.html` opened from disk only shows a notice that the server must
be running (a page opened as a file cannot reach the local server). The same page served by another local server (for
example `python -m http.server` in the repository) loads but cannot upload: that server answers the upload with 501 and
closes the connection, which the browser reports as "Failed to fetch".

Needs the pipeline's dependencies (torch, opencv, shapely, rapidocr, pymupdf, ezdxf, ...) and the segmenter
checkpoint (`pilot/v2-pipeline/data/model-v2/segmenter.pt`, or `V2_MODEL`). The upload page shows a pipeline check;
anything missing (model file, OCR engine, a library, a DWG converter) is named there and in the job log.
`FPX_APP_THREADS` caps the torch threads (default 4: the machine is shared).

## Steps

| Step | Board | What happens |
|---|---|---|
| Upload | 1c | Drop PDF, JPG, PNG, TIFF or DXF files. Each file becomes a sheet job: normalisation (`fpx.inputs`), deskew, OCR of the sheet, layout (`fpx.layout`: regions and drawings), a scale proposal per drawing (`fpx.scale.drawing_scale`). |
| 1 Building area | 1d | The detected drawings of the sheet with the floor plan's bounding box and handles. Select which drawings to extract, move the handles, or skip the sheet. |
| 2 Scale | 2a, 2b, 2c | The proposal with its cues highlighted on the sheet (caption or title-block note, scale bar, dimension strings), scale, resolution and pixel size side by side; change the dpi; or measure a known distance with two clicks. |
| 3 Run | 1f | The confirmed drawings (leave one out with ×, add a sheet), then run the extraction (stages 2-10, one sheet at a time, with progress per stage). Storey names and order are not asked for yet; the screen keeps the key `storeys` in saved projects and deep links. |
| Results | 1g, 1h | Rooms, walls, openings, stairs, voids and GF on the drawing (colour by room or QA confidence, layers, zoom), the extraction summary, the "needs review" list, all QA issues, a room list with a checked state per room, downloads: JSON, DXF, overlay PNG, room list CSV. |

## Layout

    app/
      index.html        the page (served at /)
      static/app.js     the single-page app (vanilla JS, no external resources)
      static/app.css    the study's design tokens
      server.py         http.server (standard library), JSON endpoints, one worker thread for the pipeline
      bridge.py         the only module that touches the pipeline: PIPELINE_DIR, model/OCR loading, analyse(), extract()
      data/             uploads and outputs, one folder per sheet job (gitignored)

`bridge.PIPELINE_DIR` points at `pilot/v2-pipeline` and switches to `pilot/v2-pipeline/scripts` when that folder holds
`fpx` (the refactor). Nothing else in the app knows where the pipeline lives.

Per sheet job (`app/data/sheets/<id>/`): the upload, `<package>_preview.jpg`, `analysis.json` (regions, drawings,
scale proposals and cue boxes), `session.pkl` (the analysed packages and layouts, so a restarted server need not
run the sheet OCR and the scale search again), `job.json` (status, confirmations, results), `log.txt`, `review.json`
(checked rooms) and `out/` with the pipeline's `<drawing>.json`, `<drawing>.dxf`, `<drawing>_work.jpg`,
`<drawing>_overlay.png` and `<package>_sheet.json`.

The page keeps the current project (which sheet jobs, their order, the current step) in the browser's localStorage.
A deep link sets it: `/?sheets=<id>[,<id>]&screen=area|scale|storeys|results[&pkg=0&drawing=0&sheet=0]`.

## Endpoints

    GET    /api/health                        pipeline check
    POST   /api/upload?name=<file>            body = the file -> {id}; queues the analysis
    GET    /api/sheets                        all sheet jobs
    GET    /api/sheets/<id>                   job: status, progress, log tail, analysis, confirm, results
    DELETE /api/sheets/<id>
    POST   /api/sheets/<id>/confirm           {"confirm": {"<package>/<drawing>": {extract, polygon_px, px_per_m, scale_source, measured, dpi, storey}}}
    POST   /api/sheets/<id>/run               queues stages 2-10 for the confirmed drawings
    GET    /api/sheets/<id>/results?key=...   one drawing's results for the viewer
    GET    /api/sheets/<id>/file/<name>       preview, working image, overlay, JSON, DXF
    GET    /api/sheets/<id>/rooms.csv?key=... room list
    POST   /api/sheets/<id>/review            {"key": ..., "rooms": {"r01": {"checked": true}}}
    GET    /api/sheets/<id>/log

## How the confirmed values reach the pipeline

`bridge.analyse()` calls `fpx.pipeline.run_document(stages=())`: normalisation, deskew, the sheet OCR, the layout and
the scale proposal per drawing, with the packages kept for the second call and the OCR items cached per job
(`ocr_cache/`). `bridge.extract()` calls `run_document(packages=..., overrides=...)` with one entry per confirmed
drawing: `polygon_px` replaces the drawing's polygon and mask (source "human"), `px_per_m` replaces the consensus
scale (recorded as `sheet.scale.confirmed` and `drawings[0].scale.confirmed` in the drawing's JSON, the pipeline's
own cues listed as agreeing or disagreeing with it), `storey` overrides the parsed one and `extract: false` skips the
drawing. Every drawing then goes through stages 2–10 and writes JSON, DXF, Excel and IFC.

## Known gaps

- One drawing per storey; several pages of a PDF are walked page by page. The building area is a rectangle (the
  study's handles), not the detected polygon, when the user changes it.
- Review is a checked state per room; rooms cannot be edited or merged. DWG and PDF exports are placeholders; XLSX and IFC come from the pipeline (`fpx/xlsx.py`, `fpx/ifc.py`).
- The server keeps the analysed packages in memory; after a restart the extraction re-runs the analysis first.
- Single user, single worker, no authentication: a local tool.
