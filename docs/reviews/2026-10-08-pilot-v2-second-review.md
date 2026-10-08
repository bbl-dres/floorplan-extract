# Second Review of Pilot v2: Heads, Retraining, Merged Evaluation Code, Exports

*8 October 2026. Follows the [first code review](2026-10-07-pilot-v2-code-review.md) of the previous evening. Scope: [motivation-goals.md](../motivation-goals.md) and [pipeline.md](../pipeline.md) against [pilot/v2-pipeline](../../pilot/v2-pipeline/README.md), with the items the first review left open because they needed training or a larger refactor: the unused v2 heads, the door regression and construction-drawing clutter of the model, the duplicated evaluation code, the OCR cost, the exports the goals promise. Methods were drawn from the papers and the cloned code in `research/` (Zhang's reconciliation and MSD's separation rule for open plan, SALI-FP-style gating of geometric steps, the CubiCasa and FloorPlanCAD evaluation conventions). Everything marked **done** was implemented and measured; the training runs used two RunPod RTX 4090 pods (public data only, deleted afterwards).*

## 1. Verdict

The largest accuracy gain of the day came from reading what the model already predicts: the v2 boundary and interior heads, trained since yesterday morning and consumed by nothing. Used as separation evidence and as the building mask they lift rooms on real scans (CVC-FP) from recall 0.639 / precision 0.707 to **0.733 / 0.769**, and on renders to 0.888 / 0.942, with walls untouched. The stamps arbitrate the cuts on stamped sheets, so BBL plans do not lose rooms to false ridges (Landgut S1 stays at 15/15).

Retraining with renderer 3.0 answered the three model questions of the first review: the heads cost the classes nothing (E1a without heads and E2 with five heads score the same mean element IoU on the frozen validation set), the void head rises from 0.26 to 0.88 once shafts and lifts are labelled as the slab openings they are drawn as, and the door/casement confusion and the construction-drawing clutter are measured in §4 against model v2 on CubiCasa, FloorPlanCAD and the harness.

The code is smaller and more honest than it was: one `fpeval` package scores every benchmark through one function, failures are recorded rows instead of dropped sheets, a golden-output test locks the whole pipeline's JSON, the text stage costs a third of before, the workflow app drives the pipeline through a proper hook, and every drawing now leaves the pipeline as JSON, DXF, Excel and IFC 4.3.

## 2. Findings and Decisions

Status: **done**, open, or rejected with the reason. Paths are relative to `pilot/v2-pipeline/scripts/` (`code/` until 8 October).

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

All four runs use renderer 3.0, the same frozen validation set (400 renders) and the room stage of §4.1 ("both"). Heads: interior, boundary, void, text and swing. Negatives: the construction-drawing clutter set (dimension chains, axes, hatches, furniture, annotation symbols as background). CubiCasa wall IoU is without openings (own metric; the published-protocol rows are in the model card); WAFFLE is on the 40 plans all runs share (the E3 run was cut short).

| Run | Heads | Negatives | Frozen validation IoU wall / door / window / column / stairs | Render, 117 floors: rooms R / P, wall IoU, door / window F1, connections of matched rooms | CVC-FP, 122 scans: rooms R / P, wall IoU, door / window F1 | CubiCasa test, 400 plans: wall / door / window IoU, rooms R / P | FloorPlanCAD mean of 4 F1 | WAFFLE walls P / R / IoU |
|---|---|---|---|---|---|---|---|---|
| E1a | none | no | 0.866 / 0.664 / 0.757 / 0.593 / 0.899 | 0.834 / 0.923, 0.892, 0.926 / 0.933, 0.928 | 0.588 / 0.697, 0.777, 0.795 / 0.882 | 0.662 / 0.485 / 0.636, 0.627 / 0.738 | 0.323 | 0.851 / 0.708 / 0.630 |
| E1b | low weight | no | 0.869 / 0.669 / 0.762 / 0.599 / 0.912 | 0.871 / 0.942, 0.895, 0.927 / 0.929, 0.916 | 0.740 / 0.755, 0.787, 0.813 / 0.888 | 0.665 / 0.533 / 0.644, 0.716 / 0.791 | 0.304 | 0.789 / 0.721 / 0.604 |
| E2 | full weight | no | 0.878 / 0.671 / 0.760 / 0.576 / 0.891 | 0.889 / 0.945, 0.900, 0.930 / 0.929, 0.915 | 0.772 / 0.749, 0.779, 0.817 / 0.881 | 0.666 / 0.548 / 0.641, 0.744 / 0.792 | 0.304 | 0.793 / 0.736 / 0.617 |
| **E3 = model v3** | full weight | yes | 0.873 / 0.687 / 0.743 / 0.556 / 0.896 | 0.886 / 0.944, 0.901, 0.927 / 0.927, 0.920 | 0.775 / 0.758, 0.781, 0.819 / 0.897 | 0.669 / 0.551 / 0.649, 0.739 / 0.783 | 0.321 | 0.823 / 0.761 / 0.654 |
| v2 (renderer 2.0, four heads) | full weight | no | – | 0.888 / 0.942, 0.90, 0.92 / 0.92, 0.90 (§4.1) | 0.733 / 0.769, 0.788, 0.780 / 0.878 | 0.665 / 0.451 / 0.608, – | 0.282 | 0.771 / 0.637 / 0.535 |

Reading: the heads are what lifts rooms on real scans (CVC-FP recall 0.59 → 0.74 → 0.77 from E1a to E2: the boundary head's separations and the interior head's building mask need the heads trained at full weight), and they lift CubiCasa doors (0.485 → 0.548). The negatives lift construction drawings (FloorPlanCAD 0.304 → 0.321, wall F1 0.367 → 0.408) and WAFFLE walls (0.617 → 0.654) at no cost on renders. E1a's FloorPlanCAD figure (0.323) comes with the lowest room recall of all runs. Column IoU on the frozen set falls from E1a to E3 (0.593 → 0.556) while nothing on a real benchmark follows it: synthetic validation does not select the checkpoint. **Decision:** E3 is model v3 (`common.MODEL`); the viewer and the curated set are re-exported with it in the wall focus below.

### 4.3 Wall Focus (8 October, afternoon)

The wall focus of the afternoon (layout masking as stage 1b, rejection rules, regularised segments with thickness, the axis through openings with IFC hosting, drawn-line separations; pipeline.md §5 stages 1b, 3 and 6) was measured with model v3:

| Benchmark | Before (model v3, morning) | After |
|---|---|---|
| Render harness, 117 held-out floors: rooms R / P, wall IoU, door / window F1 | 0.886 / 0.944, 0.901, 0.927 / 0.927 | 0.885 / 0.955, 0.901, 0.926 / 0.928 (same floors; the rejection rules remove false wall pieces, so fewer spurious rooms) |
| CVC-FP, 122 scans: rooms R / P, wall IoU, door / window F1 | 0.775 / 0.758, 0.781, 0.819 / 0.897 | CVCFP_AFTER |
| Landgut S1 (CAD print): wall IoU mask / regularised, rooms, mean IoU, names, openings ext / int, connections | 0.86 / –, 15/15, 0.921, 15/15, 24/24 / 10/13, 10/12 | 0.86 / 0.79, 15/15, 0.921, 15/15, 24/24 / 10/13, 10/12 |
| Landgut S2 (poché scan) | 0.56 / –, 13/15, 0.734 | 0.61 / 0.59, 12/15, 0.742, connections 7/12 |
| Landgut S3 (survey scan, whole sheet) | 0.34 / – (wall area 91.6 m² against 52.3: scan border and stucco as walls), 13/15, 0.777 | 0.59 / 0.57 (45.1 m²), 13/15, 0.781, openings 24/24 / 12/13 |
| CVC-FP CA0501 (open plan, four stamps in one area) | one room "DEGT / Cuisine / Séjour / Repas", 67 m² | Séjour cut off along its dashed line (21.8 m², stamp 23.07); kitchen, dining and DEGT stay one room (no line between them) with the QA warning |
| USACE first floor (construction drawing, whole sheet) | notes strip and title block inside the drawing: a 1.5 m title bar as a wall, 162 m² of walls | notes strip and title block masked, 143 m² |

Side findings: `dominant_angle` returned a histogram bin centre, so a plan drawn at exactly 0° was snapped to 0.5° (fixed); the regularised centre lines were in pixel-index coordinates while every polygon of the pipeline follows pixel edges, a half-pixel shift on every wall (fixed); 2 × EDT − 1 is a pixel thin on walls of even width (replaced by cross-section widths); the OCR reads dashed lines as text ("= = ="), which hid them from the separation search (text without an alphanumeric character is ignored there); the IfcRelSpaceBoundary GUIDs were renormalised by entity id and could change between writes (fixed); the thin-stroke rule first rejected CVC-FP's single-line partition walls (rooms recall 0.775 → 0.756, wall IoU 0.781 → 0.774 over the 122 scans, up to 0.057 on one sheet) until a length cap of 1.5 m (`stroke_max_length`) kept them: on the eight worst sheets rooms recall went 0.660 → 0.745 and wall IoU 0.769 → 0.810, against 0.755 / 0.813 with the rule off. Model v3 finds four fewer doors on S1 than v2 (11 instead of 15), which costs two openings and two connections there; the render and CVC-FP door F1 do not show it, so it is a casement-window style question for the next renderer.

### 4.4 Published Protocols

The benchmark tables of the former model card (merged into the pilot README on 8 October); the README keeps a one-line summary per benchmark.

#### Benchmarking method

The models are trained only on synthetic renders and evaluated **zero-shot** on public benchmarks, each with its published protocol where one exists; every benchmark is scored by `fpeval.score.score()` (one scorer since the second review):

- **FloorPlanCAD** (Fan et al., ICCV 2021): semantic symbol spotting. Each block of the November 2021 test split (5,502 blocks) is rendered black on white at 50 px/m; each vector primitive takes the majority class of points sampled along it (Eq. 10); F1 per category over primitives weighted by log(1 + length) (§10.1), with the categories of Fig. 7. Only door, window, stair and wall are scored (curtain wall ignored). `fpcad_eval.py`.
- **CubiCasa5K** (Kalervo et al. 2019, Table 4): pixel IoU per class on the 400 official test plans, one confusion matrix pooled over the split, on the segmenter's own output. For the comparison, wall includes the door and window openings, as in CubiCasa's labels. `cubicasa_eval.py`.
- **WAFFLE** (Ganon et al. 2025, Table 4): pixel precision, recall and IoU per class, pooled over the 110 annotated plans; wall includes predicted columns. `harness.py waffle`.
- **CVC-FP** (Dodge et al. 2017): wall IoU (Jaccard index) and mean IoU and mean accuracy over wall vs. non-wall, pooled over all 122 plans. `harness.py cvcfp`.
- **Own metrics:** rooms matched one to one (Hungarian on IoU ≥ 0.5), area errors, openings and connectivity, on held-out Swiss Dwellings floors and on BBL sheets (`fpeval/metrics.py`).

Every published row below was trained **in-domain** (on the benchmark's own training data), except on WAFFLE and the cross-dataset CVC-FP row, which are zero-shot like ours. The gap therefore measures the domain shift as much as the method. Deviations from the published protocols: plans without a scale (WAFFLE, CVC-FP) are scaled from detected door widths, so a wrong scale is part of our error; scoring happens at our working resolution (50 px/m), since the papers do not state theirs; test-time augmentation is four flips (CubiCasa: rotations). Sources and protocol details: [published baselines](../reports/2026-10-07-published-baselines.md).

#### FloorPlanCAD

Length-weighted F1 per category, in the layout of Fan et al. Table 3:

| Method | Training | Door | Window | Stair | Wall | Mean of 4 | F1 (30 classes) | wF1 (30 classes) |
|---|---|---|---|---|---|---|---|---|
| HRNetV2 W18 | In-domain | 0.821 | 0.620 | 0.845 | 0.620 | 0.727 | 0.656 | 0.683 |
| HRNetV2 W48 | In-domain | 0.811 | 0.640 | 0.847 | 0.624 | 0.731 | 0.666 | 0.693 |
| DeepLabv3+ R50 | In-domain | 0.828 | 0.659 | 0.856 | 0.630 | 0.743 | 0.680 | 0.705 |
| DeepLabv3+ R101 | In-domain | 0.837 | 0.666 | 0.852 | 0.634 | 0.747 | 0.688 | 0.714 |
| GCN (PanCADNet) | In-domain | 0.848 | 0.709 | 0.857 | 0.814 | 0.807 | 0.806 | 0.798 |
| **Pilot v1** | Zero-shot | 0.147 | 0.293 | 0.298 | 0.362 | 0.275 | n/a | n/a |
| **Pilot v2** | Zero-shot | 0.108 | 0.306 | 0.325 | 0.387 | 0.282 | n/a | n/a |
| **Pilot v3** | Zero-shot | 0.137 | 0.339 | 0.399 | **0.408** | **0.321** | n/a | n/a |

- Published rows are on FloorPlanCAD V1 (30 classes, never released); ours are on the November 2021 release. No published number exists on this split. Later vector models reach wF1 0.80–0.93 on the August 2021 release.
- **Doors** in FloorPlanCAD are the leaf and swing arc; ours are the opening in the wall, so most door primitives vote background. Counting any foreground prediction on the primitive raises door F1 to about 0.5 for every pilot model.
- v3's gain comes from the negative set: wall precision 0.322 → 0.329 at recall 0.487 → 0.538, wall PQ 0.050 → 0.066, wall pixel IoU 0.364 → 0.394. The ablation without negatives (E2) stays at 0.304 on the mean of four.

#### CubiCasa5K

Pixel IoU on the 400 test plans:

| Model | Training | Wall (incl. openings) | Door | Window |
|---|---|---|---|---|
| CubiCasa5K model (Kalervo et al. 2019) | In-domain | 0.730 | 0.536 | 0.668 |
| **Pilot v1** | Zero-shot | 0.619 | 0.495 | 0.570 |
| **Pilot v2** | Zero-shot | 0.699 | 0.451 | 0.608 |
| **Pilot v3** | Zero-shot | **0.706** | **0.551** | **0.649** |

Own metrics, v3: wall IoU without openings 0.669, columns 0.101, stairs 0.377; the share of door pixels predicted as window, v2's regression, is back from 18.6 % to 9.7 % (v1: 8.6 %); doors found 88 %, windows 74 %, opening precision 88 %; rooms matched 74 % with precision 78 % and a median area error of 2.4 % (scored with the harness rules since the second review: rooms and fragments ≥ 0.25 m² after stage 9, so not comparable with the v1/v2 room rows of the earlier card).

Ablations on the same renderer (frozen validation set in brackets): no heads E1a door 0.485, window 0.636 (0.756); heads at low weight E1b 0.533 / 0.644 (0.762); heads E2 0.548 / 0.641 (0.755); E2 + negatives = v3 0.551 / 0.649 (0.751). The heads help real doors; the negatives cost nothing here and gain on construction drawings.

#### WAFFLE

Pixel precision / recall / IoU on the 110 annotated plans; all rows zero-shot:

| Model | Training | Walls | Doors | Windows |
|---|---|---|---|---|
| CubiCasa5K model (Ganon et al. 2025) | CubiCasa5K | 0.737 / 0.590 / 0.488 | 0.201 / 0.163 / 0.099 | 0.339 / 0.334 / 0.202 |
| Diffusion wall segmenter (Ganon et al. 2025) | CubiCasa5K walls | 0.746 / 0.805 / 0.632 | n/a | n/a |
| **Pilot v1** | Synthetic | 0.741 / 0.704 / 0.565 | 0.176 / 0.126 / 0.079 | 0.141 / 0.288 / 0.105 |
| **Pilot v2** | Synthetic | 0.797 / 0.730 / 0.615 | 0.308 / 0.128 / 0.100 | 0.188 / 0.508 / 0.159 |
| **Pilot v3** | Synthetic | 0.823 / 0.761 / 0.654 † | 0.324 / 0.206 / 0.144 † | 0.196 / 0.441 / 0.157 † |

† v3 on the first 40 of the 110 plans (the run was cut short; the full run is queued). v2 on the same 40 plans: walls 0.771 / 0.637 / 0.535, doors 0.339 / 0.183 / 0.135, windows 0.240 / 0.445 / 0.185, so v3 gains 0.12 wall IoU on the common subset.

#### CVC-FP

Wall vs. non-wall on all 122 plans, in %:

| Model | Training | Mean accuracy | Mean IoU | Wall IoU (JI) |
|---|---|---|---|---|
| FCN-2s (Dodge et al. 2017) | CVC-FP, 5-fold | 97.3 | 94.4 | 89.2 |
| FCN-2s (Dodge et al. 2017) | R-FP only (cross-dataset) | 84.2 | 81.7 | 64.7 |
| **Pilot v1** | Synthetic | 91.7 | 82.5 | 66.2 |
| **Pilot v2** | Synthetic | 92.5 | 89.1 | 78.7 |
| **Pilot v3** | Synthetic | 92.9 | 88.7 | 78.0 |

Rooms on CVC-FP (own metric, one to one, current post-processing): v2 recall 0.733, precision 0.769; v3 recall 0.775, precision 0.758. v3 trades 0.7 points of wall IoU for wall recall (mean class accuracy 92.5 → 92.9) and finds more rooms.

#### Own benchmarks

| Benchmark | v2 | v3 |
|---|---|---|
| Frozen synthetic validation, renderer 3.0 (400 renders, IoU wall / door / window / column / stairs; v2 scored on its own renderer-2.0 set) | 0.86 / 0.64 / 0.72 / 0.58 / 0.92; heads interior 0.95, boundary 0.75, void 0.26, text 0.71 | 0.87 / 0.69 / 0.74 / 0.56 / 0.90; heads interior 0.97, boundary 0.76, void 0.83, text 0.68, swing 0.76 |
| Held-out Swiss Dwellings renders, full pipeline (117 test floors, same frozen renders; rooms recall / precision, wall IoU, door / window F1, connectivity recall) | 0.888 / 0.942, 0.90, 0.92 / 0.92, 0.90 | 0.886 / 0.944, 0.90, 0.93 / 0.93, 0.92 |
| Landgut Lohn S1, 2005 CAD print (BBL, local) | 15/15 rooms, mean IoU 0.924, names 15/15, connections 12/12 | LANDGUT_V3 |

- The post-processing rules were partly written while looking at the Landgut Lohn sheets, so those results are optimistic; details in the [pilot README](../../pilot/v2-pipeline/README.md#results).
- Post-processing alone (oracle labels on 120 held-out Swiss Dwellings floors, `fpeval/oracle.py`, after the first review): rooms recall 0.865, precision 0.959, median area error −1.6 %; every remaining miss with perfect labels is an open-plan merge.

Reproduce (any model via `--model`; `FPX_DEVICE=cuda` on a GPU), from `pilot/v2-pipeline/scripts/`: `fpcad_eval.py --workers 8`, `cubicasa_eval.py test --out ../data/cubicasa-v3`, `harness.py {render,cvcfp,waffle} --out ../data/harness/<mode>-v3.json`, `run_pipeline.py --out ../data/out-v3` then `evaluate.py --out ../data/out-v3`.

## 5. Next Steps

1. ~~Choose the production model from §4.2 and make it `common.MODEL`~~ (E3 is v3); the viewer and the curated set are re-exported in §4.3.
2. P4 of the first review (faces of the wall graph reconciled with the free-space regions). P3 is done in §4.3: the wall axis runs on through doors and windows, openings are intervals on their wall, and the IFC cuts them from the host wall.
3. Doors and windows as separate JSON lists (X8), then the DWG converter decision (§9 of pipeline.md).
4. M5/M6 (inference batching; the scale search on a crop), C8, the `common.py` re-exports, the `synth.py` split.
5. A real validation set of BBL sheets for checkpoint selection (the frozen synthetic set now exists; the real one does not).
6. Training experiment E4 with ArchCAD (`data/benchmark/archcad/`, CC BY-NC 4.0, so research only, never in deployed weights): convert its primitive-level JSON into our label classes (wall bodies filled between paired wall outlines, door openings from the door instances clipped to the wall band, windows from the glass lines, stairs and columns from instance boxes), mix 20–30 % real CAD line-work into the render stream like the IFC storeys, and compare walls on FloorPlanCAD, CubiCasa5K, CVC-FP, WAFFLE and Landgut S1 against v3. A gain would say which BBL DWG layers to export as licence-clean training data of the same kind; first a cheaper check: score v3 on ArchCAD rasters with the FloorPlanCAD protocol to see how far the style is from ours.
