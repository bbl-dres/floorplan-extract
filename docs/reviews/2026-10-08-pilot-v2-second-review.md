# Second Review of Pilot v2: Heads, Retraining, Merged Evaluation Code, Exports

*8 October 2026. Follows the [first code review](2026-10-07-pilot-v2-code-review.md) of the previous evening. Scope: [motivation-goals.md](../motivation-goals.md) and [pipeline.md](../pipeline.md) against [pilot/v2-pipeline](../../pilot/v2-pipeline/README.md), with the items the first review left open because they needed training or a larger refactor: the unused v2 heads, the door regression and construction-drawing clutter of the model, the duplicated evaluation code, the OCR cost, the exports the goals promise. Methods were drawn from the papers and the cloned code in `research/` (Zhang's reconciliation and MSD's separation rule for open plan, SALI-FP-style gating of geometric steps, the CubiCasa and FloorPlanCAD evaluation conventions). Everything marked **done** was implemented and measured; the training runs used two RunPod RTX 4090 pods (public data only, deleted afterwards).*

## 1. Verdict

The largest accuracy gain of the day came from reading what the model already predicts: the v2 boundary and interior heads, trained since yesterday morning and consumed by nothing. Used as separation evidence and as the building mask they lift rooms on real scans (CVC-FP) from recall 0.639 / precision 0.707 to **0.733 / 0.769**, and on renders to 0.888 / 0.942, with walls untouched. The stamps arbitrate the cuts on stamped sheets, so BBL plans do not lose rooms to false ridges (Landgut S1 stays at 15/15).

Retraining with renderer 3.0 answered the three model questions of the first review: the heads cost the classes nothing (E1a without heads and E2 with five heads score the same mean element IoU on the frozen validation set), the void head rises from 0.26 to 0.88 once shafts and lifts are labelled as the slab openings they are drawn as, and the door/casement confusion and the construction-drawing clutter are measured in §4 against model v2 on CubiCasa, FloorPlanCAD and the harness.

The code is smaller and more honest than it was: one `fpeval` package scores every benchmark through one function, failures are recorded rows instead of dropped sheets, a golden-output test locks the whole pipeline's JSON, the text stage costs a third of before, the workflow app drives the pipeline through a proper hook, and every drawing now leaves the pipeline as JSON, DXF, Excel and IFC 4.3.

## 2. Findings and Decisions

Status: **done**, open, or rejected with the reason. Paths are relative to `pilot/v2-pipeline/code/`.

### 2.1 Accuracy: the v2 heads in the room stage (P5 of the first review)

| # | Finding | Evidence | Decision |
|---|---|---|---|
| H1 | The boundary head outlines every area, also open-plan splits without a wall (synthetic IoU 0.75), and was read by nothing | render sd8986: the head marks the kitchen/corridor divisions; oracle: every remaining miss on perfect labels is an open-plan merge | **done**: `rooms.separations()`: inside a free region, pixels at `separation_prob` 0.5 or more cut it; a cut stays when two parts reach `separation_min_area` 2 m², the cut pixels and small parts join the nearest part. Renders (117 floors, v2): merged 297 → 199, rooms R 0.840 → 0.875 at P 0.920 → 0.928; CVC-FP (122 scans): R 0.639 → 0.722, P 0.707 → 0.737, "room (separation)" recall 0.29 → 0.42 |
| H2 | On a stamped sheet the head also draws ridges inside rooms that have one stamp (Landgut S1: the Vorplatz and the stair) | S1 mean IoU 0.924 → 0.911, names 15 → 14 with the cut on | **done**: stamps arbitrate: on a stamped sheet a cut is kept only when two of its parts carry a stamp (a room without a stamp is no room to BBL). S1 back at 15/15, 0.924; renders and CVC-FP (no OCR) unchanged |
| H3 | The interior head gives the footprint; the enclosed-barrier building mask loses wings and rooms behind unsegmented facades | renders: 157 "outside building" misses; CVC-FP 27 | **done**: `building_from_interior`: the head's footprint (closed, filled) joins the enclosed-barrier mask; a region counts as a room only when `building_enclosure` 0.5 of its boundary is wall, door or window, so a blob on empty paper is no room. Renders: outside 157 → 76, GF IoU 0.913 → 0.940; CVC-FP: outside 27 → 2, P 0.707 → 0.753 |
| H4 | Both together | | **done**, defaults: renders R 0.888 P 0.942 mIoU 0.939; CVC-FP R 0.733 P 0.769; Landgut 15/12/13 unchanged, S1 mean IoU 0.924 |

### 2.2 Model: renderer 3.0 and the training experiments (M1–M9 of the first review)

| # | Finding | Decision |
|---|---|---|
| M1 | Casement windows were drawn with the door symbol; `door()` compared pixels, not metres, so doors from 0.94 m got two leaves at scale 1.6 | **done** (`synth.py`): doors take the scale, casements as two sashes only from 0.9 m and never a full-width single sash, sash arcs 1 px or dashed, glass lines always drawn, casement share 0.20 → 0.08; a fifth head `swing` (leaf + swept sector) so that arcs are explained by a head rather than by the window class |
| M2 | The four heads took half the loss and nothing consumed them | **done**: E1a (no heads) vs E2 (five heads) on the frozen validation set: mean element IoU 0.756 vs 0.755: the heads cost the classes nothing; consumed since this morning (§2.1) |
| M3 | The void head learned a label that contradicts the drawing convention (shafts and lifts carry the same cross) | **done**: `VOIDS = {VOID, AIR, SHAFT, ELEVATOR, LIGHTWELL}`, per-sample Dice. Void head IoU 0.264 (v2) → 0.878 (E2) |
| M4 | No negative set for callouts, dimension chains, section markers, tags, furniture | **done** (`--negatives`, E3): dimension chains on the facades and interior strings, section markers, detail bubbles, axis bubbles on plain axes, door/window tags, level markers, furniture; all background, their lettering in the text head |
| M7 | Checkpoints were selected on a validation set re-rendered per run | **done**: `--val-cache` writes the 400 validation renders once with the renderer hash; all four runs share it (E3 its own, because the negatives change the renders) |
| M8 | The IFC share was 18 % instead of the nominal 10 % | **done**: the source is drawn once, the column floor searched inside it |
| M9 | `synth.py` structure, duplicated preparation code | partly: dead code removed, `FloorRecord` + `normalise_record()` shared by `sd_prepare.py` and `ifc_prepare.py`, `walls_parts` kept for centre-line labels; the `Canvas`/`Style` split stays open |
| M5, M6 | Inference batching, the 12× re-segmentation of the door-width scale search | open (S each) |

Training (RunPod RTX 4090, 16 vCPU, 14 render workers, 30,000 iterations × 16 crops): E1a 44 min, E2 51 min, E1b and E3 about 55 min; about USD 0.65 per run.

### 2.3 Text stage, scale, pipeline hooks (T5–T9 of the first review)

| # | Finding | Decision |
|---|---|---|
| T6 | The same ink was OCR'd three to four times (sheet pass, per drawing, each at two rotations) | **done**: one detection per sheet (the pre-pass reused inside drawings when its resolution is within 0.8 of `ocr_px_per_m`, else one pass inside the drawing masks), items transformed into each drawing's frame, one batched recognition, tall boxes recognised rotated instead of a second full pass (a reduced-scale rotated *detection* pass stays: upright detection alone lost 30 % of vertical numbers), OCR cached per package and mask. S1 266 → 94 s, S2 96 → 32 s, El Paso 84 → 30 s for the text stages; names 15/15 and stamp areas 12/15 unchanged |
| T7 | A reduced print stayed "low" confidence although the √2-step model explained the note | **done**: the note re-enters as `scale_note (r=0.354)`; S1 is now "high" with one medium "reduced print" flag |
| T8 | A drawing without any scale cue was skipped; stage 9's stamp-area cue never reran the scale | **done**: the sheet-size prior (1:50/1:100/1:200 by the drawing's extent) runs the drawing with a high flag; after stage 9 the stamp-area cue recomputes the consensus and reruns the drawing once when the scale moves by more than 3 % |
| T9 | Drawing kind and storey parsing: first match wins, several storeys collapsed | **done**: all kinds scored with penalties for parentheses and dashes, ceiling/roof/foundation/escape-route kinds, storey lists with a flag, Niveau/Level/Piano/Zwischengeschoss |
| S1 | Scale cues the curated review missed: metre bars with a unit word, feet-and-inch notes and dimensions, room sizes in feet | **done**: `3/16" = 1'-0"` → 1:64, `20'`, `6'-6"`, bar labels "1 … 20 Meter", broken bar lines; Wien 1888 gets 46 px/m from its bar, El Paso 123 px/m (note + room sizes) instead of being skipped |
| A1 | The workflow app mirrored the pipeline's private per-drawing loop because `run_document` had no hook | **done**: `run_document(overrides={"<package>/<drawing>": {polygon_px, px_per_m, storey, extract, note}}, cache_dir=, progress=)`, `drawing_sheet()` public, per-drawing errors recorded; `app/bridge.py` uses it. App flow verified on the Wien plan (two drawings, one hand-set region, one measured scale) |

### 2.4 Exports (X8–X9 of the first review)

| # | Finding | Decision |
|---|---|---|
| X9a | Excel export | **done**: `fpx/xlsx.py`: Räume (id, AOID, number, name, usage, net / gross / polygon / stamp area, deviation and basis, confidence, reasons, stair flights, neighbours), Geschoss (GF, AGF, sums, EBF proposal, deducted voids, bridged gaps), Öffnungen, QA, Meta |
| X9b | IFC 4.3 export | **done**: `fpx/ifc.py` with IfcOpenShell 0.9: project/site/building/storey, IfcSpace per room (net polygon extruded by a nominal height; `Qto_SpaceBaseQuantities` NetFloorArea / GrossFloorArea / Height in IFC's own meaning; `Pset_BBL_Raum` with the stamp area, the wall-share gross area, the basis, usage, confidence), IfcWall per segment with `Pset_WallCommon.IsExternal`, IfcDoor (empty openings as USERDEFINED "empty opening") and IfcWindow, IfcStairFlight, IfcColumn, IfcSlab with the deducted voids as IfcOpeningElement, first-level IfcRelSpaceBoundary between every door and the two rooms it connects; reproducible GUIDs from the element ids. The two-room fixture round-trips (quantities read back equal the JSON) and validates; Landgut S1 writes 21 spaces, 101 walls, 15 doors, 27 windows, 4 flights, 30 boundaries and validates with 0 errors. Hosting openings in their walls follows with the wall graph (P3) |
| X8 | Doors and windows as separate lists | open: the IFC and Excel exports already separate them; the JSON keeps `openings[]` with `kind` for one more schema version |
| X9c | The app's export step | **done**: Excel and IFC download rows; DWG and PDF stay placeholders |

### 2.5 Maintainability (C6–C9 of the first review)

| # | Finding | Decision |
|---|---|---|
| C6 | Four scorers, four protocols; loaders, rasterisation, palettes, confusion matrices, failure handling duplicated three to five times | **done**: `fpeval/` (`cli`, `raster`, `metrics`, `oracle`, `score`, `harness`, `fpcad`, `datasets/{cubicasa, cvcfp, waffle, commons, landgut, render, fpcad, curated}`); `harness.py`, `fpcad_eval.py`, `curated.py` are thin CLIs; `metrics.py`, `oracle.py`, `cubicasa.py`, `sheets.py` deleted; 4,124 script lines → 1,303 + 2,614 in the package. Every benchmark calls `fpeval.score.score()`; CubiCasa's rooms are now scored like the harness (rooms + fragments ≥ 0.25 m² after stage 9, outdoor spaces ignored) with the published-protocol numbers kept as extra fields; old result keys kept, new ones added |
| C7 | No golden test; ~21 tests skipping silently | **done**: `tests/fixtures/synthetic_floor.py` (four rooms and a corridor, a stair with a stamp inside its outline, a stair eye, a column island, an interior window, an exterior door, a shaft, a hyphenated stamp with area and AOID, a passage, a leaking opening) with `tests/golden/synthetic_floor.json` compared exactly (`--update-golden` regenerates) plus DXF counts per layer; shared fixtures in `conftest.py`; `pytest.ini` with `-rs`. 233 tests, 0 skipped on this machine |
| C9 | Documentation drift | **done** for README, MODEL_CARD, pipeline.md §5 and §10, app README; the model card's result tables are updated in §4 below |
| C8 | Magic numbers outside `config.py`, silent `except` in `inputs.py` | open |
| – | `common.py` keeps a transitional re-export block for `synth.py`/`train.py` | open, S |

## 3. Implementation Log (8 October 2026)

- Room stage: `separations()`, `enclosed()`, the interior-head building mask (`fpx/rooms.py`, `fpx/config.py`: `separation_prob`, `separation_min_area`, `building_from_interior`, `building_enclosure`).
- Exports: `fpx/xlsx.py`, `fpx/ifc.py`, `export()` writes both; `tests/test_exports.py`; config `ifc_*` heights.
- Renderer 3.0 and training: `synth.py`, `train.py` (`--val-cache`, `--negatives`, `--w-swing`, per-sample Dice), `sd_prepare.py` (`FloorRecord`, `normalise_record`, `walls_parts`), `ifc_prepare.py`, `synth_preview.py`, `fpx/model.py` (`swing` head), `experiments.md`, ten tests in `tests/test_synth.py`.
- Text stage and hooks: `fpx/pipeline.py` (OCR once per sheet, `overrides`, `cache_dir`, `progress`, second pass after stage 9), `fpx/text.py` (`detect`, `dedupe_quads`, `recognise`, batched recognition), `fpx/scale.py` (imperial and unit-word cues, room sizes, reduced-print note, sheet-size prior, `confirm`, `add_drawing_cue`), `fpx/layout.py` (kind scoring, storey lists), `app/bridge.py`, 51 tests in `tests/test_document.py`.
- Evaluation package `fpeval/` and thin CLIs; `tests/test_golden.py`, `tests/test_score.py`, `conftest.py`, `pytest.ini`.
- Measurements of this review: `data/harness/ab/render-v2-{base,sep,interior,both}.json`, `cvcfp-v2-*.json` (one segmentation per sheet, post-processing on copies), `data/harness/{render,cvcfp,waffle}-{e1a,e1b,e2,e3}.json`, `data/cubicasa-e*/`, `data/fpcad-e*/`.

## 4. Measurements

### 4.1 Heads in the Room Stage (model v2, same sheets, one segmentation per sheet)

| Benchmark | Variant | Rooms R | Rooms P | Mean IoU | GF IoU | Misses: merged / outside | "room (separation)" R |
|---|---|---|---|---|---|---|---|
| Render, 117 floors | base (after the first review) | 0.840 | 0.920 | 0.936 | 0.913 | 297 / 157 | – |
| | separations | 0.875 | 0.928 | 0.939 | 0.913 | 199 / 157 | – |
| | interior mask | 0.849 | 0.935 | 0.935 | 0.940 | 354 / 76 | – |
| | **both (default now)** | **0.888** | **0.942** | 0.939 | **0.940** | 238 / 76 | – |
| CVC-FP, 122 scans | base | 0.639 | 0.707 | 0.896 | – | 401 / 27 | 0.29 |
| | separations | 0.722 | 0.737 | 0.897 | – | 295 / 27 | 0.42 |
| | interior mask | 0.640 | 0.753 | 0.894 | – | 419 / 2 | 0.30 |
| | **both** | **0.733** | **0.769** | 0.895 | – | 306 / 2 | **0.43** |

Walls, doors and windows are identical across variants (the segmenter's). Landgut S1–S3 with both: rooms 15/12/13, S1 mean IoU 0.924, names 15/15 (the stamp rule of H2).

### 4.2 Retrained Models

PENDING_TRAINING

## 5. Next Steps

1. Choose the production model from §4.2 and make it `common.MODEL`; re-export the viewer and the curated set with it.
2. P3/P4 of the first review (the continuous wall graph with openings as intervals, faces reconciled with regions): the gross partition and the IFC hosting both wait for it.
3. Doors and windows as separate JSON lists (X8), then the DWG converter decision (§9 of pipeline.md).
4. M5/M6 (inference batching; the scale search on a crop), C8, the `common.py` re-exports, the `synth.py` split.
5. A real validation set of BBL sheets for checkpoint selection (the frozen synthetic set now exists; the real one does not).
