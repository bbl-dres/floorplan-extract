# Code Review of Pilot v2: Accuracy, Robustness, Performance, Maintainability

*7 October 2026, evening. Review of [pilot/v2-pipeline](../../pilot/v2-pipeline/README.md) against [motivation-goals.md](../motivation-goals.md) and [pipeline.md](../pipeline.md), with the decision on the [centre-line report](../reports/2026-10-07-centre-line-bim-reconstruction.md). Five review passes (geometry stages, segmentation and training, text/scale/layout/inputs, export and domain rules, code quality and tests) were consolidated here; the recommendations marked **done** were implemented the same evening and measured on the public harness. Numbers are from the first 120 held-out Swiss Dwellings floors unless stated.*

## 1. Verdict

Pilot v2 does what the design asks of it, and its weakest parts were post-processing rules written while looking at one building, not the model. The review found four rules that hurt on public data with *perfect* labels: the convex-hull stair split (cuts 80 % of staircases in two), the largest-blob building mask (loses every second wing), end-to-face passages (split corridors at corners) and the one-hypothesis stamp check. Removing them and adding what the goals document asks for (net, gross and tagged area side by side; stairs as objects; voids by kind; doors separate from windows in the connectivity) moved rooms on perfect labels from recall 0.853 / precision 0.836 to **0.866 / 0.965** at a third less compute, with no retraining.

The centre-line proposal is adopted in the form the report recommends (graph as canonical model, rooms reconciled with free-space regions), and the review measured its ceiling: faces of the centre-line graph of walls ∪ doors ∪ windows reach **recall 0.89 at precision 1.00** on oracle floors. The implementation order is fixed below (§5); this iteration took the parts that need no model: gap bridging from the sheet's ink, the gross partition along the wall centre lines, hosts along extended centre lines, stairs as objects.

What the review could not fix without training is also clear: open plan (every remaining miss on perfect labels is a merge of areas without a wall between them), the door regression of model v2 (the renderer draws casement windows with the door symbol), and construction drawings (callouts, dimension chains and hatches read as walls). These are the training experiments in §6, in order of expected gain per hour of GPU.

## 2. Findings and Decisions

Severity: **bug** (wrong output), **accuracy**, **robustness**, **performance**, **maintainability**. Status: done, open, or rejected with the reason. File references are to `pilot/v2-pipeline/`.

### 2.1 Geometry (stages 3b–6, 8)

| # | Finding | Severity | Evidence | Decision |
|---|---|---|---|---|
| G1 | The convex-hull stair split cuts enclosed staircases and the corridors next to them in two (`rooms.py`) | accuracy | 276 of 345 Swiss Dwellings staircases satisfy the split rule, 88 % of those are enclosed by walls; oracle STAIRCASE split 88, CORRIDOR split 68; S1 "Treppe" −15.8 % with an unnamed 3 m² stair piece | **done**: stairs are objects of the room they lie in (`stair_flights`); a stair room is cut only when the stamps demand it (a stair stamp inside the outline, or stamps on both sides), or, without a stamp on the stair, when the stair is open to a hall at least as large as its outline and walled on 60 % of the outline (the Landgut scans, whose calligraphy OCR cannot read). STAIRCASE recall 0.645 → 0.891, splits 75 → 3 |
| G2 | Wall-gap passages accepted a wall end facing the flank of another wall (`openings.passages`) | accuracy | oracle: 858 passages, 4 correct edges, 36 wrong; 65 of 68 corridor splits on sheets with passages | **done**: a passage needs two facing wall ends. Passages 367 → 60 on 120 floors, wrong passage edges 24 → 8. Cost: 94 "open" hits that happened to split open-plan corridors correctly are gone (§4) |
| G3 | Three of four openings had no host wall: centre lines stop at the jambs, so an opening's centre lies half its width from every segment (`openings.py`) | bug | S1: 32 of 42 unhosted, all 23 windows; 31 QA issues per sheet | **done in part**: host along the segment's centre line extended by 0.5 m + half the opening width, within the wall thickness and 15° of the opening's direction; nearest-segment fallback. The two-room fixture hosts both openings, S1 only 13 of 42: its thin outlined walls fragment into 101 segments whose ends lie far from the jambs. Open until openings become intervals on a continuous wall graph (P3) |
| G4 | The building mask kept the largest enclosed blob only; detached wings and second buildings lost every room and broke the GF (`openings.rough_building`) | accuracy | oracle 310 "outside building" misses, 277 on 24 sheets with GF IoU < 0.8; render-v2 510 | **done**: every enclosed region holding a door or window label is kept (a legend box has none). Outside-building misses 139 → 27 on 120 floors |
| G5 | GF included balconies; the exterior test used the GF with its holes; interior openings created connectivity edges (`rooms.py`, `derived.py`) | accuracy | S1 GF +9.5 % (Balkon 17.8 m²); oracle interior-opening edges 9 correct vs 22 wrong; S1 extra edge Halle–Treppe | **done**: outdoor rooms leave the GF in stage 8 and are reported as AGF; the exterior test uses the outline without holes; only doors connect rooms, edges carry type and confidence |
| G6 | Voids only inside stair hulls, only with a label, and shrunk by the slit closing; no kind | accuracy | S1 void 4.57 m² vs ~5.2 m² reference, so the 5 m² deduction never fired | **done**: voids have a kind (stair eye, air space), are measured on the unclosed barrier; an air-space label turns its region into a void; GF deduction for stair eyes > 5 m² and every air space (plan-check wording; to confirm with BBL); the DXF writes every void on `R_RAUMPOLYGON-ABZUG` |
| G7 | `rooms()`, `openings()` and `seal()` were O(pixels × regions): four full-image `cc == i` passes per region | performance | fit on oracle rows: 22 ms per Mpx × region; an A0 at 50 px/m would take ~11 min in the room stage alone | **done** for rooms, openings, the gross partition and voids (`ndimage.find_objects` slices, `bincount`, `ndimage.mean`); 4.98 → 4.42 s per floor in the A/B under the same load, 1.4 s on a quiet machine. Open: `seal()` and `rough_building` still run on the full image; the skeleton graph is pure Python (0.15–0.23 s/Mpx) |
| G8 | Spur pruning left degree-2 nodes, so straight walls fragmented (S1: 114 segments for 29 wall polygons); thickness carried the distance transform's +1 px | robustness | 0.30 m walls read 0.32 m | **done**: degree-2 nodes contracted; thickness = 2·median − 1 px |
| G9 | Room polygons kept every pixel step (25–133 vertices on S1) | output quality | `poly_simplify` 0.25 px removes collinear points only | **done**: Douglas–Peucker at 0.6 px on the pixel-centre contour before the half-pixel offset. 1 px was tried first and cost 0.8 % of a room (it flattens one-pixel plateaus along whole edges, which are real offsets); 0.6 px keeps those. Two-room fixture 25 → 19 vertices, area unchanged |
| G10 | Connection probing took the first label of a 7×7 window; the along-axis fallback reached rooms a door does not connect | robustness | oracle door edges wrong 62 | **done** (centre pixel first, then the majority); the fallback stays, marked by the opening's confidence. Connectivity recall of matched rooms 0.855 → 0.958, precision 0.549 → 0.672 |
| G11 | `seal()` absorbs corridors narrower than 1.3 m and rooms under ~5 m²; the watershed runs on the sheet image | robustness | design-level | open: keep as last resort with a flag; with model v2 seed "outside" from `interior_prob` |
| G12 | Stages mutate other stages' outputs (`openings` edits `sheet.label`, `qa` reorders rooms, `derived` retypes openings); the harness and `cubicasa_eval` scored different label maps | maintainability, comparability | `harness.py` copies the label before stage 4, `cubicasa_eval.py` did not | **done** in part: `segment()` keeps `sheet.seg_label` as predicted, both benchmarks score it. Open: a `masks()` helper computing barrier/building/free once, kinds decided in one place |

### 2.2 Segmentation, Training, Renderer

| # | Finding | Severity | Evidence | Decision |
|---|---|---|---|---|
| M1 | Model v2's door regression is a door → window flip caused by the renderer's casement windows: a casement under 0.8 m is drawn exactly like a single door, wider ones like a double door; `door()` never receives the scale, so at scale 1.6 doors ≥ 0.94 m get two leaves 60 % of the time (`synth.py`) | accuracy | CubiCasa: GT-door pixels predicted window 8.6 % → 18.6 %; door recall 0.643 → 0.524; plans where doors flip to windows 19 → 74 of 400; CVC-FP door-as-window 0.016 → 0.13 | open, experiment E2: pass the scale into `door()`, casements only as two sashes for L ≥ 0.9 m, dashed 1 px sash arcs, casement share 0.20 → 0.08, a swing head. The pipeline guard proposed (interior "window" at door width → door) is implemented but **off** by default (`interior_window_door_width`): with perfect labels it turned windows to unlabelled balconies into doors (window recall 1.0 → 0.78) |
| M2 | The four v2 heads take ~52 % of the loss and no stage consumes them | accuracy | loss at it 30000: classes 0.362 vs weighted heads 0.394 | open, experiment E1 (ablation). Consumption is planned with P5 of §5 (interior head as the building mask, boundary head as gap and separation evidence) |
| M3 | The void head is trained on a label that contradicts the drawing convention (elevators and shafts carry the same cross but are labelled interior; 23 voids vs 893 shafts on 150 val floors); the batch-level Dice fires only when a crop holds a void | accuracy | val void IoU wanders 0.14–0.36, historical 0.02 | open, experiment E2: `VOIDS = {VOID, AIR, SHAFT, ELEVATOR, LIGHTWELL}` as slab openings, per-sample Dice |
| M4 | No negative set for callouts, section markers, aligned dimension chains, door/window tags, furniture | accuracy | FloorPlanCAD wall precision 0.32; WAFFLE false-wall share 11 % on the Aile Richelieu sheet | open, experiment E3: renderer v3 negatives (listed in §6) |
| M5 | Inference runs the four flips sequentially at batch 1, without `inference_mode`; float32 accumulators twice; the last tile can be 129 px and context-starved | performance | CPU 512 px tile: 1.90 s → 1.02 s with batched flips + `inference_mode`; a 14 k² A0 sheet needs ~12 GB | open, S: batch the flips, align the last tile to the sheet edge, reflect-pad, `np.divide(out=)`, skip blank tiles |
| M6 | The door-width scale search re-segments the whole sheet up to 12 times (`scale.py`) | performance | FloorPlanCAD: segment 0.05 s vs rescaled 2.65 s per block | open, S: search on a centre crop, coarse then fine |
| M7 | Checkpoint selection on a validation set re-rendered per run; "best" is always the last iteration | robustness | val mIoU monotone 0.627 → 0.744 | open: freeze a validation render set with the renderer hash; add a small real set |
| M8 | IFC share is 18 %, not 10 %: `_pick` re-draws the source when a column floor is wanted | accuracy (minor) | train.log sources ifc 71 / sd 329 | open, S |
| M9 | `synth.py` is one 1,800-line class with hidden state; `sd_prepare`/`ifc_prepare` duplicate the rotate-to-dominant-angle block; the floor-record schema is implicit | maintainability | | open: `Canvas`, `Style`, drawers as functions, `labels.py`, `FloorRecord` TypedDict |

### 2.3 Text, Scale, Layout, Inputs

| # | Finding | Severity | Evidence | Decision |
|---|---|---|---|---|
| T1 | The stamp-area regex required a decimal part and knew only `m2`/`m²`: "12 m²", "18 qm", `m^2`, OCR's `m?` were lost; prefixes over 3 letters and suffixes failed | bug | harness render-ocr: 27 % of stamp areas read | **done**: one regex with optional label, integer areas with a unit, unit variants; bare decimals under 1.5 are room numbers |
| T2 | `RAUM` was a void word, so the most generic room word became a void label; any string with two letters is a room stamp | bug | `role("Raum") == "void label"` | **done** for the void words: voids are decided on the clustered stamp name (also "Luft-"/"raum" over two lines). Open: roles for levels, axis labels, tags on dense construction sheets |
| T3 | Vocabulary snapping rewrote native PDF/DXF text and over-snapped at 0.65 ("Büro Chef" → "Büro Küche", "Zimmer 12" → "Zimmer") | accuracy | probes | **done**: correction only for OCR'd stamps; a word snaps only without digits and with a length ratio ≥ 0.7 |
| T4 | Without a unit the last number won ("24.50 / 1.12" → area 1.12); room numbers were not modelled; the hyphen rule was case-dependent | accuracy | | **done**: `number` on stamps and rooms; the hyphen join also checks the vocabulary |
| T5 | OCR de-duplication was O(n²) in shapely (66 s for 2,000 boxes); the tile-skip rule dropped a lone stamp in a large hall | performance, bug | 0.0009 ink share < 0.002 | **done**: `STRtree`; absolute ink threshold |
| T6 | The same ink is OCR'd three to four times (sheet pass, per drawing, each at two rotations); OCR is 33–55 s of 38–74 s per sheet | performance | timings in `review-text/` | open, plan in §6 (one detection per sheet reused per drawing, rotate only tall boxes, batched recognition, cache): 38–74 s → ~15–25 s |
| T7 | A reduced print stays "low" confidence with a high flag although the √2-step model explains the note (S1) | accuracy | `scale_eval_landgut_final.log` | open, S |
| T8 | A drawing without any scale cue is skipped; stage 9's stamp-area cue never reruns the scale (`run_document`) | design mismatch | S2 consensus "none" | open, M: run at the best prior with a flag, rerun once when the cue moves the scale by > 3 % |
| T9 | Drawing-kind and storey parsing: first match wins ("Grundriss … (Schnitt A-A)" → section); several storeys silently collapse to one; interleave merge swallows a drawing in an L-plan's notch | robustness | probes | open, S–M |
| T10 | Raster uploads are unbounded (`MAX_IMAGE_PIXELS = None`); the PDF path channel and annotation diff have no consumer; duplicated helpers (`scale_bar_cue` vs `find_scale_bars`, four copies of the 30 % overlap merge) | robustness, maintainability | | open: cap and downsample scans; delete ~400 lines; split the three large modules |

### 2.4 Export, Conformance, QA, Schema

| # | Finding | Severity | Evidence | Decision |
|---|---|---|---|---|
| X1 | `area_gross` was the polygon area including voids, not a gross area | schema | | **done**: `area_net`, `area_gross` (wall share to the centre lines), `area_polygon`, `area_stamp`, `area_deviation_pct`, `area_basis`; `area` stays the net alias |
| X2 | QA severity equalled the room's confidence (a low-confidence room got severity "low") | bug | | **done**: severities error / warning / info per check; confidence stays on the room |
| X3 | The stamp was checked against one hypothesis; "Treppe" 17.98 m² vs net 14.94 (−17 %) although the polygon with the stair eye is 19.5 | domain | S1 r017 | **done**: net and polygon are both checked, the closer one counts (`area_basis`); tolerance at least 0.3 m² |
| X4 | The AOID reader was stricter than plan-check (`[A-Z0-9]{2}` vs `[A-Za-z0-9]{1,4}`) | bug | | **done**: one pattern, as plan-check |
| X5 | plan-check's score hid a release blocker: the one failing rule (AOID_001, error) means "not released" | domain | 39/40 on S1–S3 | **done**: `errors`, `warnings`, `releasable` next to the score |
| X6 | Provenance lost in the JSON: stamps, columns, void kinds, wall kind, `aoid` written even when not exported | schema | | **done**: `schema_version 1.0`, `stamps[]`, `structure[]` (columns), voids with id and kind, walls with `kind` and `load_bearing: unknown`, `aoid` only when written plus `aoids_read`, `wall_bridges[]`, `floor.agf_area`, `sum_net`, `sum_gross`, connectivity edges with type and confidence |
| X7 | `wall_construction` (massive/lightweight) lived in the exporter and mutated `wall_segments` | maintainability | | **done**: moved to stage 8 (`derived.wall_construction`), next to the new wall `kind` |
| X8 | Doors and windows are one `openings[]` list with a `kind` | schema | goals: separate classes | open, S: `doors[]` {type incl. empty opening} and `windows[]` in the next schema version; consumers listed in the export review (viewer, evaluate, harness, cubicasa_eval) |
| X9 | IFC 4.3 writer, Excel export | feature | goals | open: IfcOpenShell 0.9 API verified; ~400–500 lines plus a round-trip test (`ifcopenshell.validate`, footprint IoU > 0.99, quantities equal to the JSON); Excel with `openpyxl` (sheets Räume, Geschoss, Öffnungen, QA, Meta). Order: schema with doors/windows → gross area (done) → Excel → IFC |
| X10 | Guideline constants duplicated (layers and colours in `export.py` and `conformance.py`, 0.25 / 5 m² in four places, two AOID regexes); `conformance.check` is one 190-line function | maintainability | | open: `fpx/guideline.py`, rules as small functions |

### 2.5 Code Quality, Tests, Scripts

| # | Finding | Severity | Evidence | Decision |
|---|---|---|---|---|
| C1 | `import fpx.pipeline` loaded torch, smp, scikit-image and ezdxf eagerly (63 s under load; test collection 40 s) | performance | | **done**: lazy imports; `import fpx.pipeline` 0.9 s without torch |
| C2 | Five scripts scanned `sys.argv` by hand (a typo became a sheet id); the default model was v1 everywhere; `bench.py` could not take another model | robustness | | **done**: `common.cli()` / `setup()` (argparse with `--model`, `--out`, `--config`, `--threads`, UTF-8 console, thread caps); `common.MODEL` defaults to model v2 (env `V2_MODEL`) |
| C3 | Obsolete `PYTHONHASHSEED` re-exec in `harness.py` and `curated.py` (the renderer is deterministic since the index fix) | maintainability | `test_no_process_dependence` | **done**: removed |
| C4 | The harness cache stored probabilities as uint8, so cached runs scored differently from fresh ones | robustness | | **done**: float16, old uint8 files still read |
| C5 | `evaluate.py` read the BBL reference at import; `export_viewer.py` imported it, so the viewer export failed on machines without BBL plans | robustness | | **done**: the reference is optional in `export_viewer.py` |
| C6 | Four scorers with four protocols (`cubicasa_eval` stops at stage 6 and keeps rooms ≥ 1 m²; opening tolerance 0.5 vs 0.6 m; greedy and Hungarian matching in one row); loaders, rasterisation, palettes and confusion matrices duplicated three to five times | maintainability, comparability | | open, M: the `fpeval/` package proposed by the review (metrics, oracle, raster, score, datasets/, harness, cli); every benchmark calls one `score()`; ordered refactor steps with the test that guards each |
| C7 | No direct tests for stages 4, 5, 6, 8, 9; scripts untested; ~21 tests skip silently without local data | tests | | **done** in part: `tests/test_stages.py` (10 tests: bridging, hosts, passages, stairs, building mask, voids, areas, wall kinds, severities, stamp numbers, graph contraction). Open: the golden-output test of the full pipeline on a synthetic floor (bit-identity on `to_json()`), `pytest -rs` on the data runner |
| C8 | Non-pixel magic numbers outside `config.py` (`back < 10`, saturation 40, clustering ratios); silent `except Exception` in `inputs.py` and the per-sheet loops of the scripts | maintainability, robustness | | open, S |
| C9 | Documentation drift: README test time, missing modules, v1 numbers next to v2; `run_pipeline.py` names the v1 weights; MODEL_CARD said the scripts default to v1 | docs | | **done** where the code changed (README, MODEL_CARD, pipeline.md §5); the rest is listed in §6 |

## 3. Implementation Log (7–8 October 2026)

All tests pass (162: 152 before plus 10 new). The sources moved afterwards into `pilot/v2-pipeline/code/` (the pilot folder keeps README, MODEL_CARD, `data/` and `viewer.html`); the paths below are relative to `code/`.

- `fpx/walls.py`: `bridge_gaps` (ink band test, de-duplicated per gap), `wall_graph` with `contract_degree2`, `end_direction` shared with the passage detector, thickness without the +1 px; `sheet.wall_bridges`.
- `fpx/openings.py`: `rough_building(anchors=)`, `host_wall` along extended centre lines, `passages` between facing ends only; blob statistics by `bincount` and `ndimage.mean`.
- `fpx/stairs.py` unchanged; `fpx/rooms.py`: `regions()` slices, `at_border`, `stair_rooms` by stamps, `voids` by kind on the unclosed barrier, `stair_flights` per room, contour simplification.
- `fpx/derived.py`: `gross_partition` (nearest-room propagation into the wall band, `sheet.gross_label`), `wall_kind`, `wall_construction` (from the exporter), GF without outdoor rooms and `agf_area`, three areas per room, typed connectivity edges, majority probing, the opt-in interior-window rule.
- `fpx/qa.py`: severities per check, `stamp_check` against net and polygon, tolerance floor, bridged-gap issues.
- `fpx/attributes.py` and `fpx/text.py`: area regex with units and labels, room numbers, void names on clustered stamps, OCR-only correction with length ratio, `STRtree` de-duplication, absolute ink threshold, shared AOID pattern.
- `fpx/export.py`: schema 1.0 fields, stamps, structure, bridges, every void on `R_RAUMPOLYGON-ABZUG`; `fpx/conformance.py`: errors, warnings, `releasable`.
- `fpx/config.py`: `gap_bridge_min/max/ink`, `contour_simplify`, `opening_host_extension/angle`, `stamp_tolerance_m2`, `interior_window_door_width` (off), each with its reason; `fpx/model.py`: the new fields.
- `fpx/segment.py`: `sheet.seg_label`; `fpx/pipeline.py`, `walls.py`, `export.py`: lazy heavy imports.
- `common.py`: `MODEL`, `cli()`, `setup()`; `run_pipeline.py`, `cubicasa_eval.py`, `bench.py`, `curated.py`, `harness.py`, `fpcad_eval.py`, `export_viewer.py`: argparse, model default v2, no re-exec, float16 cache, optional BBL reference.
- `evaluate.py --out`, `tests/test_stages.py`, `tests/test_attributes.py`, `tests/test_export.py` (polygon vs gross area).
- Docs: `pipeline.md` §5 stage notes, this review, `MODEL_CARD.md` and the pilot README (code layout, defaults).

## 4. Measurements

Baselines are the stored runs of the same day (`data/harness/oracle.json`, `render-v1.json`, `cvcfp-v1.json`, `waffle-v1.json`, `data/out-v2/evaluation.json`); the new runs are in `data/harness/ab/` and `data/out-v2b/`. Oracle and render use the first 120 of the 295 test floors; CVC-FP and WAFFLE all sheets; the benchmark rows of render/CVC-FP/WAFFLE use model v1 from the segmenter cache, so they measure the post-processing changes alone.

**Oracle (perfect labels, 120 floors):**

| | Rooms R | Rooms P | Mean IoU | Area error (median) | Misses: merged / outside / split | STAIRCASE R | CORRIDOR R | Connections: recall of matched rooms / precision | Passages (wrong) | s per floor |
|---|---|---|---|---|---|---|---|---|---|---|
| Before | 0.853 | 0.836 | 0.930 | −1.81 % | 267 / 139 / 75 | 0.645 | 0.701 | 0.855 / 0.549 | 367 (24) | 4.98 |
| Stairs as objects + wings only | 0.890 | 0.904 | – | – | 297 / 27 / 31 | 0.891 | 0.724 | – | – | 4.42 |
| All changes | **0.866** | **0.965** | **0.949** | **−1.63 %** | 414 / 27 / 1 | **0.899** | 0.628 | **0.958 / 0.672** | 60 (8) | 3.44 |
| + stair fallback (final) | 0.865 | 0.959 | 0.947 | −1.63 % | 413 / 27 / 3 | 0.891 | 0.628 | 0.958 / 0.672 | 60 (8) | 1.37 (quiet) |

The drop from 0.890 to 0.866 recall between the second and the third row is the passage change (G2): the old end-to-face passages split 94 open-plan corridors and kitchens correctly by luck and 24 wrongly; without them those areas merge (CORRIDOR merged 120 → 202, KITCHEN 115 → 151). Precision gains 13 points and the wrong passages are gone, so the change stays; the merges are the open-plan problem that needs a separation signal (§5, P5 and E5), not a geometric rule.

The third row has the stamp-free stair fallback (G1, ratio 1.0 and enclosure 0.6, decided on these floors: it recut one staircase of 138 and gave S2 and S3 their stair rooms back). Per floor 1.4 s on a quiet machine; the baseline's 5.0 s was measured under load, so only the oracle A/B row (same conditions, 4.98 → 4.42 s) is a fair timing.

### 4.1 Full Pipeline: Renders, Scans, BBL Sheets

The segmenter's own outputs (wall, door and window pixels) do not change with any of these edits, so wall IoU and door/window recall are the same before and after; the gains are in rooms, connections and precision. Each row compares the same model on the same sheets.

| Benchmark | Model | Rooms R | Rooms P | Mean IoU | Connections: recall / precision | Passages | Misses: merged / outside / split |
|---|---|---|---|---|---|---|---|
| Render, 117 floors | v1 before | 0.800 | 0.748 | 0.914 | 0.801 / 0.490 | 375 | 255 / 226 / 88 |
| | v1 after | **0.827** | **0.853** | **0.932** | **0.891 / 0.635** | 24 | 313 / 158 / 17 |
| | v2 before | 0.822 | 0.813 | 0.923 | 0.835 / 0.542 | 262 | 238 / 202 / 76 |
| | v2 after | **0.840** | **0.934** | **0.939** | **0.912 / 0.673** | 14 | 301 / 157 / 4 |
| CVC-FP, 122 scans | v1 before | 0.472 | 0.556 | 0.870 | – | 131 | – |
| | v1 after | **0.595** | **0.652** | **0.884** | – | 7 | – |
| | v2 before | 0.609 | 0.665 | 0.881 | – | 96 | – |
| | v2 after | **0.639** | **0.708** | **0.894** | – | 6 | – |

- Render, model v2: STAIRCASE recall 0.504 → 0.785; wall IoU 0.900 before and after; doors 0.899 / 0.941 and windows 0.902 before and after. GF IoU 0.913 both.
- CVC-FP: "room" recall 0.576 → 0.736 (v1) and 0.757 → 0.803 (v2); "room (separation)" (open plan next to a separation line) 0.25 → 0.29 either way, the open-plan limit again. Walls 0.665 → 0.667 (v1), doors and windows unchanged.
- WAFFLE (110 plans, masks only): walls, doors and windows unchanged; passages 68 → 1.
- Landgut Lohn, model v2 (`data/out-v2` → `data/out-v2b`): S1 15/15 rooms, mean IoU 0.926 → 0.924, connections F1 0.889 → 0.923, GF IoU 0.913 (the balcony stays in the GF, as in the 2005 reference), hosts 10 → 13 of 42 openings, 6 gaps bridged (0.06–0.58 m), QA 57 issues (16 "low", 41 "medium" by confidence) → 64 (55 info, 9 warnings), every room with net, gross and polygon areas, walls 55 exterior / 44 interior / 2 unknown; S2 12/15, mean IoU 0.753 → 0.752, connections 0.759; S3 13/15, 0.777, connections 0.733 → 0.759. Without the stair fallback S2 and S3 lost their stair room (11/15 and 12/15): the scans' calligraphy gives OCR no stamp to decide with.

What did not move: wall IoU (bridging closes gaps that are small in area; its effect is that rooms stop leaking), door and window recall (the segmenter's), and the open-plan merges (up everywhere, see §4).

## 5. Decision on the Centre-Line Report

The report's recommendation (§10.1: the graph as canonical model and export, rooms reconciled room by room with the free-space regions, stairs as non-bounding objects) is adopted. The review added the measurement the report asked for first (§10.8 experiment 1): on 15 oracle floors, faces of the skeleton of walls ∪ doors ∪ windows ∪ columns, minus the wall bodies, reach recall 0.892 at precision 1.000, against 0.899 / 0.792 for the free-space regions on the same floors; all 44 misses are open-plan merges; the skeleton must include the opening bodies (1,750 dangling ends for walls alone vs 230 with openings), and extending dangling ends by 0.9 m adds nothing on perfect labels (the gaps are railings and open plan). Order, with what this iteration did:

| Step | Content | Status |
|---|---|---|
| P1 | Oracle BIM test as a harness mode (`--rooms faces\|regions\|reconciled`; variant with separation lines from the area polygons as the ceiling of a boundary head) | open, S; the regression test for P3 |
| P2 | Stairs as objects, no hull split | **done** (G1) |
| P3 | Continuous centre-line graph of wall ∪ door ∪ window ∪ column bodies with opening intervals per edge; `contract_degree2`; `extend_dangling` with ink or probability support; faces by `polygonize` | partly done: gap bridging from ink (walls.py), degree-2 contraction, hosts along the graph; the graph still comes from the wall mask alone and rooms from regions |
| P4 | Room-by-room reconciliation of faces and regions (IoU ≥ 0.8 agree; several regions in one face → add edges; several faces in one region → keep the split only with evidence; region without face → region, flagged) | open, M |
| P5 | Use the v2 heads: `interior_prob` as the building mask, `boundary_prob` as support for gap extension and as separation evidence, `void_prob` to confirm voids | open, S–M; gated by `render-v2`, `cvcfp-v2` ("room (separation)" recall 0.29) and S1–S3 |
| Gross area | The gross partition along the wall centre lines (G/X1) is the face-based gross area without the graph's fragility: every wall pixel goes to the nearest room | **done** |

## 6. Next Steps, in Order

1. **Training experiments** (RunPod RTX 4090, ~49 min and USD 0.60 per run): E1 head ablation (`--heads none`; `--w-boundary 0.2 --w-interior 0.2`) scored on CubiCasa door/window IoU and the render harness; E2 casement/door fix, `door(s)` scale, swing head, void relabel; E3 renderer v3 negatives (dimension chains aligned to facades and jambs, section and detail markers, grid bubbles on plain axes, door/window tags, level markers, furniture outlines) scored on FloorPlanCAD wall F1/precision and the WAFFLE false-wall share; E4 inference-only (batched flips, `inference_mode`, blank-tile skip, TTA on/off); E5 centre-line and junction heads with per-wall axes from `sd_prepare` (keep `walls_parts`), read out in `walls.py`; E6 `--no-pretrained` (licence question).
2. **OCR speed-up** (T6): one detection per sheet reused per drawing, tall boxes rotated instead of a second pass, batched recognition, cache per package and mask; expected 38–74 s → 15–25 s per sheet.
3. **Schema 1.0, second half** (X8): `doors[]` and `windows[]`, then Excel, then the IFC 4.3 writer with its round-trip test (X9).
4. **P1, P4, P5** of §5; `seal()` and `rough_building` on crops; a vectorised skeleton graph (G7, G11).
5. **`fpeval/` package and one scorer** (C6), the golden-output test (C7), `guideline.py` (X10), the `synth.py` split (M9), the text-stage clean-up (T10) and the remaining magic numbers (C8).
6. **Scale**: the reduced-print confidence (T7) and the no-cue route with a rerun after stage 9 (T8); drawing-kind and storey parsing (T9).

## 7. Open Questions for BBL

- The void deduction rule: plan-check's wording "Treppenaugen > 5 m², Lufträume" is implemented as a threshold for stair eyes and always for air spaces; confirm.
- The per-room gross area is an allocation rule (half of each interior wall, the inner half of exterior walls); SIA 416 knows gross areas per storey only. Confirm the convention before it reaches SAP, and whether AGF polygons (balconies) belong on `R_RAUMPOLYGON`.
- Interior windows: historical plans draw open archways like windows (S2), CubiCasa draws almost no interior windows. The opt-in rule that turns an interior window of door width into a door needs a BBL sample to decide.
