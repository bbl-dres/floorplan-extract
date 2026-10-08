# Second Design and UX Review of the Workflow App

*8 October 2026 · [app/](../../app/README.md) after the three-step split and the removal of the run step. Reviewed as a product designer would: the flow, what each screen asks for and shows, feedback during waits, honesty of the controls, accessibility, and the results screen against board 1i of the [UX study](../wireframes/261007_Viewer%20and%20Workflow%20UX%20study.html) (the reduced viewer: layers left, figures and rooms right, both panels collapsible). Every finding carries its decision; all are implemented unless the row says otherwise. Verified in a real browser (Playwright, Chromium) by walking the demo sheet from the landing page to the exports; screenshots and timings are quoted where they matter.*

## 1. Findings

### 1.1 Flow and steps

| # | Where | Finding | Decision |
|---|---|---|---|
| U1 | Steps | Three "steps" were shown (area, scale, results), but the user takes two decisions; the third bar only said that something happens. The landing cards numbered the results as a step too. | **Done.** Two steps: "Step 1 of 2 · Building area", "Step 2 of 2 · Scale". The results are the outcome, with both bars filled and no number; the landing page shows "Step 1", "Step 2", "Then · Results". `docs/pipeline.md` §3 and the app README follow. |
| U2 | Results while running | The panel said the same thing twice: a spinner line with the stage and, in the drawing row below, the same stage as a badge (the user's screenshot). The stage names were the pipeline's own ("2 text (OCR inside the drawing)"). | **Done.** One place: the stage in plain words ("Reading the text in the drawing", "Tracing the walls", …) with a determinate bar (stage k of 12). The drawing rows appear only when the project has more than one drawing. |
| U3 | Results while running | The user watched a static sheet preview for a minute. | **Done.** The canvas shows the work in progress: as soon as the drawing is cut out at its working resolution it replaces the sheet, and walls, openings, stairs and rooms are drawn on it as the pipeline finds them, each layer fading in (see §2.3). |
| U4 | Results | The screen followed boards 1g/1h of the study (a long right column with figures, a review list, an inspector, a room table and an export block). The study's later board 1i is simpler: layers left, figures and rooms right, both collapsible, a room card pinned on the canvas. | **Done.** Rebuilt on board 1i (§2.2). |
| U5 | Exports | Seven export rows in the right column, two of them disabled "Soon" buttons (DWG, PDF). | **Done.** A **Download** menu in the header, shown on finished results: JSON, Excel, DXF, IFC, CSV, PNG, each with a one-line note. DWG and PDF are not offered until they exist. |

### 1.2 Canvas

| # | Where | Finding | Decision |
|---|---|---|---|
| U6 | Results canvas | "Does not always reset to the drawing" (the user). Cause: the fit to the GF outline ran in the image's `onload`, before the stage's `ResizeObserver` had given the canvas a size, so the view was computed with width 0 and left broken until "Reset view", which then fitted the whole working image (with its white padding), not the drawing. The scale step had the same race for its cue framing. | **Done.** The viewer has a *home* box: `load(url, home)` frames it once the stage has a size and again on resize; `fit()` returns to it ("Fit" in the toolbar). Results frame the GF outline (or the walls), the scale step frames the drawing with its cues at most 3x closer than the sheet. |
| U7 | Results canvas | A "Top | 3D" switch with 3D permanently disabled; "Reset view" fitted the padded image. | **Done.** The toolbar is "2D | 3D · Plan [opacity] · Fit". 3D is real now: three.js (vendored, loaded on first use) draws the drawing as the ground with walls, columns, doors, windows and stairs as prisms at nominal heights, rooms as coloured slabs and the GF outline at wall height; orbit, pan and zoom, the layer switches, the colouring, the opacity and the room card work in both views. |
| U8 | Results canvas | Hovering a room showed a tooltip with figures and clicking filled an inspector block in the right column, far from the room. | **Done.** As in board 1i: hover outlines the room; a click pins a card next to it on the canvas (name, net area, stamp with deviation, confidence, the first reasons, the "checked" box) with × and Escape to close; the room's row in the list highlights and scrolls into view. |

### 1.3 Honest chrome and feedback

| # | Where | Finding | Decision |
|---|---|---|---|
| U9 | Top bar | A language switch that switched nothing, a disabled settings cog and an avatar "DR": decoration from the study that invites clicks and disappoints. | **Done.** Removed. The bar holds the project name, Start over, Download (on results) and Help. |
| U10 | Errors | Nine `alert()` dialogs (upload, confirm, run, review) and a `Log` button that dumped the log into an alert. | **Done.** Inline notices at the bottom of the window (`notify`, announced to assistive technology, dismissable, self-closing); the log opens as a page. |
| U11 | Help | The help listed "Results" twice and still described three steps and the whole-sheet OCR at upload. | **Done.** Rewritten for the two steps, the demo sheet and the live results. |

### 1.4 Accessibility

| # | Where | Finding | Decision |
|---|---|---|---|
| A1 | Screens | Moving between screens left the focus wherever it was; a screen reader heard nothing. | **Done.** Each screen's heading takes focus on mount (no visible ring on headings); the help dialog is a `role=dialog` with `aria-modal`, opens on the close button, closes on Escape and returns the focus to Help. |
| A2 | Controls | Icon-only buttons (×, ↻, ?, the collapse chevrons, the removal of a sheet) had no accessible name; the canvases had no name. | **Done.** `aria-label`s throughout; every canvas is `role=img` with a sentence naming what it shows; the step bars are `role=img` "Step 1 of 2". The walk-through counts zero unnamed controls on every screen. |
| A3 | Progress | Progress text changed silently. | **Done.** The busy card and the progress block are `role=status` / `aria-live=polite`; the progress bar is a `role=progressbar` with value and max. |
| A4 | Motion | The fade-in of layers and the spinner animate regardless of the user's preference. | **Done.** `prefers-reduced-motion` turns the fade-in off and slows the spinner. |
| A5 | Keyboard | Room rows and the Download menu were mouse-only. | **Done.** Rows are focusable buttons (Enter/Space), the menu is `aria-haspopup`/`aria-expanded`, focuses its first item, closes on Escape and outside clicks. |
| A6 | Keyboard | The canvas tools (area handles, the two-point measurement) are pointer-only. | **Open.** Arrow-key nudging of the handles and a typed measurement are the next step; the numbers next to the canvas can already be changed by keyboard (dpi, distance). |
| A7 | Contrast | Muted text (#646b74 on white, 5.5:1) and the accent on white (4.6:1) pass AA; disabled text (#9aa1ab, 2.6:1) is used only on disabled controls. | No change. |

### 1.5 Robustness seen on the way

| # | Where | Finding | Decision |
|---|---|---|---|
| R1 | Results while running | The progress block read the job record captured at render time; the poll replaces that record, so the stage stayed at "starting" for the whole run. | **Done.** The block reads the fresh record. |
| R2 | Extraction | One run of the demo sheet in the browser took 34 minutes for the OCR inside the drawing (normally 25-45 s); the same OCR took 24 s in isolation from both shells, a second browser run took 45 s, and a measured poll of the two endpoints the screen uses showed 20 ms latencies. The slow run coincided with a 115 MB browser download being unpacked and scanned on the machine. | **Open.** Not reproducible; recorded here. ONNX Runtime spins its threads when the machine is oversubscribed, so capping its intra-op threads like the torch threads (`FPX_APP_THREADS`) is the first thing to try if it returns. |
| R3 | Results layout | Collapsing the left panel squeezed the canvas to zero width and pushed the right panel to the left: a hidden grid child leaves the grid, and the others slid into its column. Found by the browser walk ("element is outside of the viewport"). | **Done.** The three children are placed in their columns explicitly. |

### 1.6 Seen in the demo results (pipeline, not the app)

| # | Where | Finding | Decision |
|---|---|---|---|
| P1 | Room stamps | The demo sheet stamps every room in square feet ("TEAM RM 150 SF"). The stamp reader takes no SF figure, so no room has a stamp area, none can be checked, and the confidence share reads 0 % although the rooms are clean. | **Open**, pipeline: read `SF` / `SQ FT` stamps and convert; the sheet's dimension strings are already read in feet and inches. |
| P2 | Room names | Revision tags of the sheet ("<REV>", "</REV>") and OCR slips ("CENTER" read as "ENTRÉE") land in room names. | **Open**, pipeline: drop tag-like tokens from stamps; the Latin recogniser's bias towards accented words is a model matter. |

## 2. What changed

### 2.1 Steps

Upload (landing with the demo link and the busy card), Step 1 of 2 building area, Step 2 of 2 scale, then the results. The confirmations are the only steps; the extraction runs by itself.

### 2.2 Results, after board 1i

- **Left, "Layers"** (collapsible with the chevron at the stage's edge): Show all / Hide all, Colour rooms by Room | Confidence with the legend, and three groups that fold: Rooms (fill, labels, connections), Structure (walls, doors, windows, passages, columns, stairs and voids), Analysis (GF outline, segmentation overlay). Counts fill in as the layers arrive.
- **Right** (collapsible): while running, the stage in words with the progress bar; when done, the drawing's name with "Finished", two text links ("‹ Area and scale", "Run again"), the figures (rooms, net and gross area, openings, needs review, confidence share, scale, time), and two tabs in one block: **Rooms** (sortable by confidence or area, with the "checked" boxes) and **Issues** (the rooms, voids and checks that ask for a look, then every QA finding). No button row at the bottom.
- **Canvas**: hover outline, pinned room card, "2D | 3D · Plan [opacity] · Fit"; the 3D view is `js/view3d.js` on three.js 0.186.1 (`js/vendor/three`, MIT), loaded on demand so the 2D app stays light.
- **Header**: Download menu with the six export files.
- The panel states, the colouring, the layers, the opacity and the sort survive a reload (localStorage); the selection does not.

### 2.3 The live canvas

`fpx.pipeline.run_document` takes an `after(key, stage, sheet)` hook, called after each per-drawing stage with the `Sheet` as it stands. The app's bridge writes `<key>_live.json` into the job's output folder after triage (the working image), walls, openings, stairs and rooms, with the exterior rings in working pixels; `GET /api/sheets/<id>/live?key=…` serves it, `{"stage": null}` before it exists. The results screen polls it every 1.5 s while the job is busy, swaps the sheet preview for the working image when it appears, and draws each new layer with a 0.7 s fade (none under reduced motion). The cost is a few milliseconds per stage; the file is removed before a new run.

Until the working image exists (the text recognition inside the drawing comes first, 25-45 s on the demo sheet) the sheet preview is framed on the confirmed drawing. Measured on a rerun of the demo sheet in the browser: the working image at 2 s, "Finding walls and openings" 6-30 s, walls (29), doors (63) and windows (22) on the canvas at 34 s, the 70 rooms and the GF outline at 36 s, finished at 38 s. On a first run the text recognition precedes this (95 s in all).

### 2.4 Verification

- `app/tests`: 6 passed (routes, worker, deferred delete, demo, live endpoint). `pilot/v2-pipeline/scripts/tests`: 236 passed, including the `after` hook.
- Browser walk-through of the demo sheet with Playwright: landing → demo → busy card → area (auto-advance) → confirm → scale (12 s of cues) → confirm → results live → finished → Download menu → room card → both panels collapsed and reopened → sort by area → help opened and closed by Escape. No console errors; zero unnamed controls per screen.

## 3. Open items

- A6: keyboard access to the two canvas tools.
- R2: the one slow OCR run; cap the ONNX threads if it returns.
- The room card shows the first three reasons; a "more" link to the issues list would complete it.
- The Download menu lists every file; a project-level export (all sheets as one Excel) belongs to the storey work that is still deferred.
- Dark mode remains uncovered by the tokens (the pilot viewer has one).
