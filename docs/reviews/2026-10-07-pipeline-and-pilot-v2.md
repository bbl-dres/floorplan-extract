# Review: Pipeline Design and Pilot v2

*Technical review · 7 October 2026 · Scope: [pipeline design](../pipeline.md) draft 0.2 and [pilot v2](../../pilot/v2-pipeline/README.md) (code and results). Perspective: software engineering, computer vision and machine learning, Python, and BBL area management (SIA 416, CAD-Richtlinie BBL V1.0). Evidence: the 34 papers in `research/papers-md` (through the [literature review](../literature-review.md)), AxcelerateAI's published pipeline (`research/web`), plan-check's rule set, a new zero-shot benchmark on CubiCasa5K run for this review (§2.2), and the solution and data catalogues: [open](../solutions-open.md) and [closed solutions](../solutions-closed.md), [open datasets](../datasets-open.md), [open building documentation](../datasets-open-docu.md) and [closed datasets](../datasets-closed.md). Revised the same day once those catalogues were complete.*

## 1. Verdict

- **The design is right, and it now has evidence.** The literature has converged on the same structure (literature review, key finding 1). AxcelerateAI's production pipeline chains the same steps:
  - read the scale and the text layer;
  - segment walls and detect openings;
  - close wall loops into rooms named by their labels;
  - turn doors into graph edges;
  - compute quantities at drawing scale and flag low confidence.

  Pilot v2 adds the missing proof: a segmenter trained only on public data, rendered in random styles, transfers to real drawing styles it has never seen.
- **The evidence is still thin and optimistic.** It rests on:
  - one building;
  - post-processing rules shaped while looking at the test sheets;
  - model selection on synthetic data;
  - no regression tests.

  The CubiCasa5K benchmark (§2.2) is the first check on hundreds of unseen real plans. It confirms the transfer: zero-shot walls, doors and windows on 400 plans reach IoU 0.57 / 0.49 / 0.57, against 0.73 / 0.54 / 0.67 for CubiCasa's own in-domain model. It also shows the weak spots: hatching that is not wall, thin lines, columns, and rooms without walls between them.
- **The design document and the code have drifted apart** (§3.2). The pilot found better answers than the document in some places, for example rooms as regions between walls and closed openings, which need no snapping. In others it cut corners, for example voids found only from a label, and passages found by ray casting.
- **Five changes matter most:**
  1. **Evaluation harness on public data, before any further tuning.** Swiss Dwellings test floors rendered end to end, with exact rooms, doors and connections; CubiCasa5K and CVC-FP as real-image benchmarks; one-to-one matching; areas in m². Freeze the post-processing parameters on this harness, then rerun Landgut Lohn untouched.
  2. **The model should predict what the post-processing now guesses:** interior (building footprint), room boundaries, voids, text, and door instances with their swing.
  3. **The renderer should cover what failed:** hatching that is not wall, ceiling ornament, window reveals, casement windows with swings, Swiss hatching conventions, column grids, overlays, scan defects (Augraphy) and a wider scale range.
  4. **The export must pass plan-check:** AOIDs only on `R_AOID`, solid hatches only for massive walls, Arial text on allowed layers, voids cut out of the GF outline, and DWG written through ODA so plan-check can actually run.
  5. **The engineering needs basics:** modules, typed stage outputs, one config for the roughly 30 metric thresholds, tests, a pinned environment and a model card.
- **The catalogues settle three questions:**
  - **Training:** none of the roughly 40 pretrained floor plan weight releases is licence-clean, so BBL has to train its own models. Permissive code exists for most components (§4.3).
  - **Evaluation:** BBL's own 148 Bautendokumentationen are worth investigating as input in publication style, paired with BBL's CAD geometry as the reference; the rights stay in-house. Their published area figures are often entered by hand and not always derived from the plan geometry, so they are a noisy reference to check, not ground truth (§6.2).
  - **Market:** no product covers the whole chain and no AI converter offers Swiss hosting. The SAP landing step may exist off the shelf, and Kanton Zürich has tendered the manual route before (§3.3, §5).
- **Data:** public data is enough for steps 1–3. BBL samples become essential for model selection, room stamps and calibration (§6.2). Partner routes and five legal questions follow in §6.3 and §6.4.

## 2. Evidence

### 2.1 Landgut Lohn (pilot v2)

| | S1 CAD print | S2 historical poché scan | S3 survey scan |
|---|---|---|---|
| Walls IoU vs 2005 | 0.89 | 0.58 | 0.29 (ornament read as walls) |
| Rooms matched (IoU ≥ 0.5) | 15/15, mean IoU 0.88 | 12/15 (building changed since) | 12/15 |
| Median area error vs stamp | 2.2% | 3.5% | 8.1% |
| Exterior / interior openings found | 24/24, 12/13 | 24/24, 10/13 | 24/24, 9/13 |
| Room connections found | 12/12 | 11/12 | 8/12 |

Two corrections to how these figures are read:
- **The GF overestimate on S1 (+8.5%) comes mostly from the reference, not the extraction.** The reference "building" is rooms ∪ walls. It leaves out the small spiral stair (3.4 m²), which is not modelled as a room, and the façade pilasters and plinths. The extents of the extracted GF and the reference agree to 2 cm. GF needs a proper reference: SIA 416 outer dimensions, decided case by case for projections.
- **The S1 rules were added after looking at S1** (slit closing, sealing, stair split, interior windows). The numbers measure what is achievable, not what to expect on new sheets.

### 2.2 CubiCasa5K Zero-Shot Benchmark (new)

**Setup.** 400 plans from the CubiCasa5K test split, in three styles, all Finnish residential (`pilot/v2-pipeline/cubicasa_eval.py`):
- **The model never saw CubiCasa.** The v2 segmenter ran unchanged, with flip averaging, followed by pipeline stages 3–6.
- **Exact scale:** every room in the SVG carries its size in metres, which places all scaled images at 100 px/m (5th–95th percentile 99.9–100.1). They were resampled to the model's 50 px/m.
- **Outdoor spaces** are excluded from the room scores.
- **Licence:** CC BY-NC-SA 4.0, so this is a benchmark only.

| Style group | Plans | Pixel IoU walls / doors / windows / stairs | Doors found | Windows found | Opening precision | Rooms found / correct (IoU ≥ 0.5) | Median room area error |
|---|---|---|---|---|---|---|---|
| colorful | 67 | 0.69 / 0.59 / 0.70 / 0.51 | 0.91 | 0.82 | 0.89 | 0.66 / 0.76 | 4.5% |
| high_quality (scans, some hand annotation) | 63 | 0.74 / 0.63 / 0.74 / 0.48 | 0.92 | 0.85 | 0.92 | 0.70 / 0.84 | 4.0% |
| high_quality_architectural | 270 | 0.53 / 0.46 / 0.52 / 0.35 | 0.86 | 0.67 | 0.77 | 0.63 / 0.66 | 5.2% |
| **All** | 400 | 0.57 / 0.49 / 0.57 / 0.38 | 0.88 | 0.70 | 0.81 | 0.65 / 0.69 | 4.9% |
| *Reference: CubiCasa's own model, trained on CubiCasa (paper, test split)* | 400 | 0.73 / 0.54 / 0.67 / – raw; 0.48 / 0.41 / 0.41 after polygonisation | – | – | – | – | – |

Doors and windows count as found when a predicted opening's centre lies within 0.5 m. Column IoU is 0.09.

- **The transfer holds on real plans.** Without seeing CubiCasa, the segmenter reaches the raw IoU of CubiCasa's own model on two of three style groups, and exceeds its polygonised scores in all three. This is indicative, not a like-for-like comparison: resolution, label conventions and evaluation details differ.
- **Openings are found:** 88% of doors and 70% of windows, with 81% precision.
- **Rooms are the weak link:** 65% of reference rooms are matched; matched rooms are within 4.9% of their area. Misses fall into three groups:
  - open-plan rooms separated only by virtual lines, which no wall-based method can find;
  - rooms merged through a missed door;
  - multi-floor sheets.
- **Architectural drawings are hardest:**
  - hatched roof slopes and terraces are read as walls, because the renderer hatches only walls (§4.2);
  - thin knee walls drawn as single lines are missed;
  - windows drawn as two thin lines are often missed.
- **Columns and stairs are weak** (IoU 0.09 and 0.38). Columns are mostly labelled as wall: Swiss Dwellings has few, and the renderer draws them like walls.
- **Caveats:**
  - The references are simplified redrawings: wall thickness differs and some partitions are missing, so pixel IoU partly measures convention.
  - A parser bug read only the first floor of multi-floor sheets. All floors share one image there, so every other floor counted as false positives. It was found and fixed during this review; the fix raised wall IoU from 0.48 to 0.57.

### 2.3 Synthetic Validation

On 400 renders from held-out Swiss Dwellings sites, the final checkpoint reaches IoU 0.91 for walls, 0.96 stairs, 0.85 windows, 0.77 doors and 0.41 columns. Historical styles are hardest for windows (0.74 vs 0.88 on CAD styles). The best synthetic checkpoint was not the best on the CAD print. Synthetic validation shows that training works, but it cannot be used to select a model.

### 2.4 Benchmark Samples in the Viewer

Thirty-five sample sheets (4 CubiCasa5K, 5 CVC-FP, 8 WAFFLE, 18 Swiss Commons plans) run through stages 0–10 and appear in the [viewer](../../pilot/v2-pipeline/viewer.html) next to Landgut Lohn, with ground truth where it exists. First observations:
- **The door-width scale proposal works on plans with doors.** On a WAFFLE apartment plan it settles where 23 detected doors measure 0.92 m, and walls then score 0.81 IoU. It fails on plans with few or unusual doors: churches, palaces, very wide historical portals.
- **Historical federal plans are within reach.** On the Bundeshaus 1902 main floor the pipeline finds 62 rooms with names read by OCR ("PRÄSIDENT", "BUNDESRAT", "VESTIBULE", "DRUCKSACHEN"). Seating in the chambers is wrongly split into small rooms.
- **Stamp parsing picks up dimension numbers.** On CVC-FP, "CHAMBRE 3" took the dimension "3.50" as its area instead of the printed "11.75 m²". Area tokens with an m² suffix must win over bare numbers.
- **Room merges dominate on CVC-FP** (3 of 10 rooms found on one plan), where doors are drawn as thin swing arcs only.
- **Monumental plans are out of domain** (Hofburg, Louvre wing: wall IoU 0.05–0.11): solid poché masses, columns and vaults drawn at a much smaller scale.

## 3. Pipeline Design Review

### 3.1 What Holds

- **Stage order and representation.** Read the text, segment walls, attach openings to walls, form rooms from closed walls, attach stamps, derive GF and connections. This matches the literature and AxcelerateAI.
- **Principles 4, 6 and 7:** exact data first, nothing filled in silently, data stays in Switzerland. The pilot followed all three, including training on public data only on RunPod.
- **Triage and routing (stage 0)** with input-quality checks and per-style tagging.
- **Evaluation (§8):** strata, per-style reporting and correction minutes as the decisive metric are right. The tooling is missing (§3.3, §7).

### 3.2 Where Document and Pilot Disagree

| Topic | Document says | Pilot v2 does | Recommendation |
|---|---|---|---|
| Rooms | Faces of the planar wall graph, polygonised along inner wall faces | Connected free-space regions between walls and closed openings, vectorised afterwards. The wall graph serves openings and centre lines only | Keep the region approach: it gave 2.2% median area error and is robust to graph errors. Document it, then regularise the outlines (angle-based vertex selection plus Douglas–Peucker, after PolyRoom). Use the wall graph for centre lines, thickness, junction QA and export |
| Text | Masked before wall and room segmentation | Not masked. The segmenter is trained with text as a distractor | Mask only for classical baselines: masking removes wall pixels where stamps overlap walls. Better, add a text output to the segmenter |
| Snapping | A stage of its own (10 cm to the wall face) | Not needed: the model segments wall bodies, not ink lines | Replace with the steps that were needed: closing slits under 0.3 m, sealing rooms that leak through an unclosed opening, splitting stairs from halls |
| Openings | Symbol detector plus gaps between wall ends, fused | Segmentation classes plus ray-cast wall gaps. The wall-gap rule produced most false passages on S3 | Add an instance head or detector for doors and windows (hinge, swing, width). Accept a wall-gap passage only between substantial walls, and confirm it with a stamp on each side or a learned cue |
| Voids | Detected as part of stairs | Found only where a void label such as LUFTRAUM is printed | Add a void class: Swiss Dwellings has VOID, AIR and LIGHTWELL areas |
| Graphical style | Tagged at triage | A pixel heuristic read the outlined S3 as "solid (poché) walls" | Derive the style after segmentation from how much ink fills the wall mask (solid, hatched, outlined) |
| Scale | Several cues, robust consensus, reviewer confirms | Calibration or title block, plus the stamp-area cue | Add dimension strings: the S1 text layer and OCR both contain them. Add scale-bar detection (S3 has one) and the scale note, as AxcelerateAI reads it |

### 3.3 Gaps in the Design

- **Reconciliation with SAP.** BBL already holds room lists (AOID, usage, area) for many buildings, so they should serve as prior and as check. Match extracted rooms to SAP rooms by floor, area, position and adjacency (an assignment problem). Then attach AOIDs and flag differences: changed walls, merged rooms, stale stamps. SAP areas can be manual entries as well, so a mismatch is a question for review, not automatically a pipeline error. Stage 7 mentions this for historical plans; it deserves its own step with metrics, because it is how CV output reaches the CAFM without a second system. The [closed solutions](../solutions-closed.md#36-cafmiwms-and-sap-re-fx-integration) list an off-the-shelf candidate for the last step: Korasoft links CAD room polylines to RE-FX objects inside SAP. This is seen in search results only and must be verified with BBL's SAP team. SAP is steering real estate towards Planon, while RE-FX stays supported until at least 2040.
- **Open-plan areas.** CubiCasa marks rooms separated only by virtual lines, and office floors have open-plan zones. The design says a face with several stamps "points to a missing wall". It should also split such a face, by a watershed seeded at the stamps, and flag it for review.
- **Several drawings on one sheet.** v2 keeps only the largest building. About 20% of CubiCasa images hold several floors, and archive sheets often show several buildings, sections or details. Triage must split a sheet into drawings first, using building-mask components plus captions.
- **Inner courtyards and light wells.** Filling holes closes them into the GF. Detect them (no stamp, enclosed by exterior walls with windows facing in, labels such as "Lichthof") and cut them out.
- **Oblique and curved walls.** Swiss Dwellings contains both. Manhattan snapping must stay optional, per dominant direction.
- **Provenance.** Every value should link back to its place on the original sheet, as AxcelerateAI also claims. v2 loses the deskew angle and transforms only to plan metres. Store a sheet-pixel ↔ plan transform per sheet.
- **Door semantics.** Escape-route and accessibility checks need hinge side, swing direction and clear width, not only position. Add MSD's edge types (door, passage, front door) and window adjacency to the connectivity graph.
- **Wall types.** `A_SCHRAFFUR` holds massive walls only, so each wall segment needs a material cue: fill or hatch style, the legend, or a wall-type tag (AxcelerateAI groups walls by the type label).
- **Review and correction tooling.** Correction minutes per sheet is the decisive metric, yet the viewer is read-only. An editor is needed, for example QGIS on the JSON exported as GeoJSON layers, or a web editor. Every correction should be logged, both for the metric and for the learning loop.
- **Confidence.** v2's confidence is rule-based. Combine it with the model's own signals (class probability margin per element, agreement between flipped predictions) and calibrate on the benchmarks, so that "high" means a measured error rate.

### 3.4 Smaller Corrections

- Principle 3 lost words: "Geometry comes from pixel-precise methods, meaning from OCR and VLMs" should read "geometry from pixel-precise methods, text and meaning from OCR and VLMs, consistency from rules".
- The header still cites only the v1 pilot.
- Stage 1: document the two resolutions the pilot uses: 50 px/m for geometry, 150 px/m for OCR. At 50 px/m an 8 cm partition is 4 px wide, so modern drywall offices may need 100 px/m.
- §6 Components should list what v2 uses: segmentation_models_pytorch U-Net/ResNet-34, RapidOCR with PP-OCRv5 Latin (ONNX), scikit-image, flip averaging.
- §7 Training data: add the renderer, IFC-Bench and the corrected CubiCasa5K licence (CC BY-NC-SA 4.0 on Zenodo).
- §8 Evaluation: add the public benchmarks, the synthetic end-to-end benchmark, one-to-one matching, boundary-sensitive wall metrics and area error in m².

## 4. Pilot v2 Code Review

### 4.1 Engineering

| Finding | Impact | Fix |
|---|---|---|
| `stages.py` (847 lines) mutates one `Sheet` object with attributes added along the way | Hard to test or reuse; stage contracts are implicit | One module per stage with typed inputs and outputs (dataclasses or pydantic) mirroring the §4 data model; a versioned JSON schema |
| About 30 metric thresholds hard-coded (0.3 m slit closing, 0.65 m seal, 1.6 m building closing, 2.2 m maximum passage, area minimums) | Silent coupling; tuned on the test sheets | One config file with a rationale per value; sweep them on the public harness, not on BBL test sheets |
| No tests | Every fix risks a regression elsewhere (it happened twice in the pilot) | Unit tests for the geometry helpers (stamp clustering, name correction, sealing, passages); an end-to-end regression on rendered Swiss Dwellings floors in CI |
| Global Python environment, no requirements file | Not reproducible on a Swiss server | Pinned `pyproject` or `requirements.txt`; documented CPU and GPU paths |
| Model file without a model card | Licence and provenance of the weights are not traceable | Model card: training data and licences, renderer version, synthetic and benchmark scores, commit hash |
| Non-deterministic renderer: worker seeds from the clock; Python `hash()` picks room colours and differs per process | Training runs cannot be repeated | Seeded generators throughout |
| OCR takes 31–52 s of 35–70 s per sheet; the skeleton graph is pure Python | Fine for a pilot, slow for the archive | Batch OCR on GPU; a vectorised skeleton graph (e.g. the `skan` library) |

### 4.2 Renderer (`synth.py`)

It already covers wall styles (solid, outlined, hatched, grey, colour, split by thickness), door and window conventions, stairs, fixtures, multilingual stamps in several lettering styles, dimension chains and axes, and basic scan defects. It is missing what failed or what the archive will contain:
- **hatched regions that are not walls** (roof slopes, terraces, ground, section fills). Hatching appears only on walls in today's renders, so the model reads any dense hatch as wall. This was the largest false-positive source on CubiCasa's architectural sheets (§2.2);
- ceiling ornament and cornice lines inside rooms, window reveals and niches, pilasters (S3);
- casement windows drawn with swing arcs, which the final model read as doors (S1);
- Swiss material hatching per wall type (concrete, masonry, insulation), so that massive walls can be told apart;
- column grids with axis labels. Columns are rare in Swiss Dwellings, so sample floors with columns more often (column IoU is 0.41);
- office content (desks, partitions, open-plan zones): render IFC-Bench models, since Swiss Dwellings is residential;
- overlays: colour zones such as fire compartments, stamps, red-pen revisions, hand annotations (CubiCasa's high_quality group has blue-pen notes);
- several drawings, title blocks and legends on one sheet;
- scan defects through Augraphy (folds, shadows, bleed-through, ink fading), plus perspective and stronger resolution loss;
- a wider scale range: training covers ±20% around 50 px/m, while a wrong scale guess can be off by a factor of 2;
- extra targets that cost nothing to render: interior footprint, room boundaries, text boxes, junctions, voids.

### 4.3 Model and Training

- **Architecture.** A U-Net with a ResNet-34 encoder on 512 px crops at 50 px/m, 30k iterations, is a sound baseline. Next steps, in order of value:
  - multi-task heads for interior footprint, room boundary, void, text and wall junctions (CubiCasa5K's multi-task heatmaps, Raster-to-Vector's junctions, DeepFloorplan's room-boundary attention);
  - a boundary-aware loss for thin walls;
  - a 100 px/m variant for drywall offices.
- **Pretrained encoder licence.** The ResNet-34 encoder starts from ImageNet weights. ImageNet's terms allow "only non-commercial research and educational" use of the images. Whether that extends to weights trained on them is unsettled and belongs in the legal review. Alternatives are a self-supervised encoder pretrained on our own renders, or training from scratch. Avoid smp's SegFormer (`mit_b*`) encoders, which are under NVIDIA's non-commercial licence ([open-source solutions](../solutions-open.md)).
- **Reuse code, not weights.** None of the roughly 40 pretrained floor plan weight releases is licence-clean: they are trained on non-commercial, research-only or undisclosed data ([open solutions](../solutions-open.md)). Permissive components to try, all trained on our own renders:
  - RF-DETR, sizes N–L and its segmentation models (Apache-2.0; the XL and larger sizes are under a different licence);
  - CubiCasa5k-Next (Apache-2.0) for typed junctions;
  - HAWP and DeepLSD (MIT) for wall-line and junction candidates;
  - nnU-Net (Apache-2.0) as a segmentation baseline;
  - TRDG or SynthTIGER (MIT) with PARSeq (Apache-2.0) to generate DE/FR/IT room stamps and fine-tune a recogniser for hand and calligraphic lettering.
- **Doors and windows** as instances, from an Apache-2.0 detector (RF-DETR) or an instance head with hinge and swing keypoints. Opening type and connections depend on swing and position, which a segmentation class cannot express.
- **Model selection.** Keep several checkpoints and an exponential moving average of the weights, and select on real images. That means the BBL gold set; until it exists, a permissive real set: BBL's own Bautendokumentationen once labelled from the plan geometry, or CC0 and public-domain historical plans (§6.1). Selecting with non-commercial data (CubiCasa5K, CVC-FP) influences production weights, a grey zone for BBL's legal service (public datasets catalogue, §1).
- **Domain adaptation with BBL data, in Switzerland:**
  - fine-tune on a few labelled sheets and measure the learning curve (10, 50, 200 sheets);
  - self-train on the unlabelled archive, keeping only pseudo-labels on which the flipped predictions agree.

### 4.4 Post-Processing (`stages.py`)

| Step | Weakness | Fix |
|---|---|---|
| Passages from wall ends (ray casting) | False passages across rooms (S3), despite the size and direction filters | Learned cue (opening class or interior/room-boundary head); keep the rule only as a QA hint |
| Building mask (closing 1.6 m, fill holes, largest component) | Fills courtyards; drops further buildings; reaches the sheet border along exterior elements | Predicted interior footprint; split drawings at triage |
| Sealing leaking rooms (erosion 0.65 m) | Loses leaking rooms narrower than 1.3 m | Becomes rare once openings and the interior footprint are predicted; keep as a fallback with a flag |
| Stair outline (convex hull of flights) | Swallows landings and corridor next to L-shaped or spiral stairs | Stair class covering flights and landings; void class for the stairwell |
| Deskew | Rotates the images without recording it in the transform | Store the rotation in the sheet transform |
| Door versus window typing | Model output decides; casement windows came out as doors | Type from model, host wall (exterior or interior) and swing; flag disagreements |
| Name correction | Vocabulary written after seeing S2 | A fixed vocabulary per language from BBL's usage taxonomy; report raw OCR alongside |
| Confidence | Rules only | Add model signals and calibrate (§3.3) |

### 4.5 Export and plan-check Conformance

Checked against [plan-check's rules](https://github.com/bbl-dres/plan-check/blob/main/docs/pruefregeln-de.md):

| Rule | v2 export | Fix |
|---|---|---|
| `R_AOID`: one unique AOID text per room polygon, base point inside (AOID_001–005) | Writes the room name, or the room id, when no AOID is known | AOID only when known or matched from SAP; names on `V_TEXT`; rooms without an AOID flagged, not filled in |
| `A_SCHRAFFUR`: solid hatches of massive walls only | Hatches every wall | Hatch only walls classified as massive |
| Text only on `V_PLANLAYOUT`, `V_ACHSEN`, `V_TEXT`, `R_AOID`, in Arial (TEXT_001) | Default text style | Arial text style; allowed layers only |
| GF: voids over 5 m² cut out with one continuous polyline | GF exterior on `R_GESCHOSSPOLYGON`, void separately on `R_RAUMPOLYGON-ABZUG` | Write the GF with the void cut out as plan-check expects, and keep the deduction polygon |
| Every room of 0.25 m² or more needs a polygon | Unlabelled fragments under 1.5 m² are dropped | Keep them as rooms flagged for review (shafts, niches) |
| plan-check reads DWG | Only DXF written | Convert with ODA File Converter, then run plan-check in the pipeline (stage 10) |

### 4.6 Evaluation Code

- Rooms are matched by best IoU without a one-to-one assignment, so one merged room can count for two references. Use Hungarian matching.
- The reference is incomplete: some openings are missing, and so is the spiral stair. Pilot extras were checked by eye. Audit the reference, and mark elements it does not model.
- Pixel IoU punishes thin walls: one pixel per face costs about 30% on a 6 px partition. Add centre-line precision and recall within 5 cm, boundary F-score and area error in m², as §8 of the design already lists.

## 5. Comparison with AxcelerateAI

AxcelerateAI publishes its pipeline as six stages, plus a component guide and a general diagram (`research/web`). Its accuracy claims ("98%+") are unverified.

| Stage | AxcelerateAI | Pilot v2 | Take-away |
|---|---|---|---|
| Sheet and scale | Clean, tile, read the scale note (1:100, 1/4" = 1'-0") | Calibration or title block, stamp-area cue | Read scale notes and dimension strings (§3.2) |
| Text | Angle-aware OCR (DBNet + CRNN), text kept with position | PP-OCRv5 (DBNet + SVTR) with an extra 90° pass | Equivalent; add text-role classification for dimensions and door tags |
| Walls | U-Net/HRNet; DXF read as exact vectors (ezdxf) | U-Net; DXF and vector PDF rendered to raster | Add a vector path for DXF and vector PDF, using exact geometry where available |
| Symbols | YOLO with oriented boxes, Mask R-CNN | Segmentation classes | Instance detector (Apache-2.0, not Ultralytics' AGPL) |
| Rooms and graph | Wall loops named from labels; doors as edges with the two spaces they connect | Same | Confirmed |
| Office features | Walls grouped by type label; door tags matched to door schedules; template search for a symbol; hatch regions recognised by finish | – | Relevant for modern office DWGs and PDFs; later |
| Checking | Annotated overlay; low confidence flagged; every value linked to the sheet | Viewer, QA list | Provenance and an editor (§3.3) |

The general diagram adds a language step for "labels, dimensions", the same conclusion as above: dimension strings serve both scale and QA.

**Market view** ([closed solutions](../solutions-closed.md)):
- No product covers the whole chain, and no AI converter offers Swiss hosting.
- On-premise routes are few: AxcelerateAI (custom build), RasterScan (Docker image), forks of the Archilyse code.
- All accuracy figures are vendor claims. BBL's own AmpliFY test is still the only independent result.
- For the worst scans, Kanton Zürich tendered a manual service in 2018: ICFM, about 1,350 floors into SAP RE-FX, CHF 1.76M including building-services data and five years of maintenance. A similar tender could require paired scan and DWG output, which would also give training data.

## 6. Data

### 6.1 Public Data to Use Now

| Data | Use | Status | Action |
|---|---|---|---|
| Swiss Dwellings v3.0.0 (CC BY 4.0) | Training | In `data/public/` | Oversample floors with columns, stairs and public use; add void, interior and boundary targets |
| Swiss Dwellings test split, rendered | End-to-end benchmark with exact rooms, doors, connections and GF | Available, no new data | Build the harness (§7, step 1) |
| MSD on 4TU (CC BY 4.0) | Door-level access graphs for the connectivity metric | Not downloaded | Download; align with the Swiss Dwellings floors |
| IFC-Bench models under CC BY 3.0/4.0 or MIT; buildingSMART samples | Non-residential render source: offices, hospital, clinic, hotels; columns, stairs, named spaces | 17 models in `data/public/ifc-bench/`; converted to 72 storeys (`ifc_prepare.py`) | Mix into training (renderer supports weighted sources) |
| CubiCasa5K (CC BY-NC-SA 4.0) | Real-image benchmark | In `data/benchmark/` | Keep as benchmark only |
| CVC-FP (122 scans, non-commercial) | Real scans in four styles, with walls, doors, windows and rooms | In `data/benchmark/cvc-fp/`; 5 sample sheets in the viewer (`bench.py`) | Full benchmark run; door labels cover the swing area, so score doors as instances |
| WAFFLE (110 densely labelled images; per-image licences) | Unseen styles, historical and public buildings | Partial in `data/benchmark/waffle/`: metadata, 1,208 SVGs, the 110-image benchmark; the raster images were refused by SharePoint (size limit). 8 samples in the viewer | Fetch images from Commons at a low rate; check licences per image |
| Versailles-FP (academic) | Historical wall masks | Needs an academic partner | Later |
| Swiss plans on Wikimedia Commons (public domain, CC0) | Unseen historical styles of Swiss and federal buildings (Bundeshaus 1902, Federal Supreme Court, Landesmuseum) | 18 sheets in `data/benchmark/commons-plans/`; in the viewer | No ground truth; scale proposed from door widths |
| TU Berlin Architekturmuseum via Europeana and DDB (CC0 per record) | Licence-clean historical drawing styles: about 20,685 open floor plans | Not downloaded ([closed datasets](../datasets-closed.md)) | Harvest by API, check each record's rights statement; style coverage and a candidate selection set after labelling |
| HABS/HAER/HALS (US public domain) | 72,828 measured-drawing sheets, historical buildings | Not downloaded | As above; no area figures |
| Public building documentation ([survey](../datasets-open-docu.md)) | About 1,400 Swiss documentations; vector plans since 2011; SIA 416 areas per building, often entered by hand (a noisy reference) | Not downloaded; rights reserved by the publishers | Evaluation only with the publisher's consent; complete floor sets only in Kanton Zürich, armasuisse Kriens, Vaud and Ticino samples |

### 6.2 BBL Sample Data to Request

Ordered by value per sheet. Process all of it locally or on Swiss infrastructure only, and never send it to a cloud VLM.

1. **BBL's own Bautendokumentationen, investigated with scepticism:** the 148 published documentations (vector plans, GF per building) with the matching CAD files. The brochure plan is the input in publication style; the CAD geometry is the reference.
   - The published area figures are often entered by hand and not always derived from the plan geometry. Treat them as a noisy reference, not ground truth.
   - Measure how far they deviate from the CAD geometry. That deviation is itself a data-quality finding for BBL.
   - The rights stay in-house; check the planner contracts for usage rights. armasuisse's 53 documentations can follow through federal cooperation.
2. **Paired floors:** for 5–10 floors, the compliant DWG and an older scan or PDF of the same floor. Registration then turns the DWG into ground truth for the scan, as with Landgut Lohn's 2005 DWG and its historical scans. Mark known building changes.
3. **10–20 DWGs that pass plan-check,** with their PDFs, from different authors and eras. They give exact rooms, AOIDs, GF and massive walls for label projection and end-to-end evaluation, and stamp text for an OCR benchmark.
4. **SAP room lists** (AOID, usage, area, floor) for the same floors, to test reconciliation and AOID assignment.
5. **20–30 unlabelled raster or PDF sheets across styles** (poché, hatched, colour, hand-lettered, CAD print), including office floors with open-plan areas and column grids. They serve style coverage, self-training and the first gold-set annotations.
6. **Typical legends and title blocks** from BBL's main authors, to make the renderer realistic.

### 6.3 Partner and Licence Routes

From the [closed datasets](../datasets-closed.md) catalogue, in order of value:
1. **Plan data pool with peer owners.** Members of the CADexchange association include BBL, armasuisse, Kanton Zürich, Basel-Stadt, Stadt Zürich, Graubünden and Zug. Their CAD guidelines require room polygons, floor polygons and room stamps, as BBL's does, so a data-sharing agreement would give labelled office, school and administrative plans in BBL's format family.
2. **Federal holdings.** The Federal Archives hold a real-estate plans fonds (K3*, 1848–2002), searchable in the reading room, with free digitisation on order. The National Library's monument-preservation archive holds about 110,000 historic plans, barely digitised.
3. **e-periodica** (Schweizerische Bauzeitung, Werk, Bauen + Wohnen and others; an estimated 10,000–30,000 plans). It has IIIF access, but systematic use needs written consent from ETH Library and the rights holders.
4. **De Gruyter's Building Types Online** (about 8,000 mostly vector drawings, many office and educational buildings). It needs a negotiated mining and training licence; the standard terms reserve text and data mining.
5. **Contract clauses.** Add ML training and evaluation rights to BBL's own planning contracts and competition programmes (KBOB documents), so future deliveries become usable data. Competition entries carry no such rights today.

### 6.4 Questions for BBL's Legal Service

1. Does Art. 24d URG (copies for scientific research) cover internal evaluation and model training on lawfully accessed plans, and can contracts or opt-outs override it?
2. Can weights pretrained on ImageNet (the pilot's ResNet-34 encoder) be used in production?
3. May non-commercial benchmarks (CubiCasa5K, CVC-FP) be used for model selection, given that this influences production weights?
4. Which usage rights do BBL's planner contracts grant for published documentation plans and delivered CAD files?
5. What wording should an ML clause in future planning contracts and competition programmes have?

## 7. Recommended Next Steps

Status: Done, Open or Declined (as of 7 October 2026).

| # | Step | Data | Effort | Status |
|---|---|---|---|---|
| 1a | CubiCasa5K zero-shot benchmark (`pilot/v2-pipeline/cubicasa_eval.py`) | Public | Small | Done |
| 1b | Benchmark sample sheets in the viewer: CubiCasa5K, CVC-FP, WAFFLE, 18 Swiss Commons plans (`pilot/v2-pipeline/bench.py`) | Public | Small | Done |
| 1c | Evaluation harness: Swiss Dwellings test renders end to end, full CVC-FP and WAFFLE runs; one-to-one matching; area in m²; boundary metrics | Public | Medium | Open |
| 2 | Freeze the post-processing parameters on the harness, move them into one config, rerun Landgut Lohn untouched (one config done, §8.1) | Public | Small | Open |
| 3 | Export conformance (§4.5), DWG through ODA, plan-check in the pipeline | – | Small | Open |
| 4a | IFC-Bench models converted to floor records for the renderer (`ifc_prepare.py`, 72 storeys); weighted source mixing in the renderer | Public | Small | Done |
| 4b | Renderer v2 (§4.2) and retraining with Swiss Dwellings and IFC storeys (about $1 per run on RunPod, public data only) | Public | Medium | Open |
| 5 | Model v2: interior, room-boundary, void, text and junction heads; door and window instances with swing | Public | Medium–large | Open |
| 6a | Scale proposal from detected door widths (segmenter run at candidate scales; `bench.py`) | Public | Small | Done |
| 6b | Scale cues: dimension strings, scale bar, scale note; consensus with the door-width cue | Public + BBL | Small–medium | Open |
| 7 | Vector path for DXF and vector PDF (exact geometry and text) | BBL DWGs | Medium | Open |
| 8 | SAP reconciliation prototype | BBL room lists | Medium | Open |
| 9 | Review editor and correction-time logging | BBL sheets | Large | Open |
| 10 | Engineering: modules, typed outputs, tests, pinned environment, model card (in parallel); see §8.1 | – | Medium | Done |
| 11 | Solution and data catalogues: open and closed solutions, open datasets, open building documentation, closed datasets | – | Medium | Done |
| 12 | Investigate BBL's Bautendokumentationen as an evaluation source: brochure plans as input, CAD geometry as reference, and the deviation of the published area figures from the geometry (§6.2) | BBL | Medium | Open |
| 13 | Harvest licence-clean historical plans (TU Berlin via Europeana and DDB, HABS) for style coverage and a permissive selection set | Public | Medium | Open |
| 14 | Partner routes (§6.3): data pool with peer owners, Federal Archives, ML clause in planning contracts | Partners | Medium | Open |
| 15 | Legal review (§6.4) | – | Small–medium | Open |
| 16 | Vendor tests on non-sensitive plans (Mappedin, Kreo or Togal, AxcelerateAI, RasterScan); verify the SAP landing step (Korasoft) with BBL's SAP team | Non-sensitive plans | Small–medium | Open |

## 8. Implementation Log

Work on the steps above, with the evidence for each change. The findings in §1–§6 stay as written on the review date.

### 8.1 Engineering Foundation (Step 10), 7 October 2026

- **Package.** `pilot/v2-pipeline/fpx/` replaces `stages.py`:
  - one module per stage, a typed sheet and element model (`fpx/model.py`), and `fpx.pipeline.run()`;
  - every threshold in one `Config` with its reason (`fpx/config.py`; a test fails if a reason is missing), overridable per run from JSON;
  - inference no longer imports the renderer or the BBL calibration file.
- **The refactor reproduces the old pipeline exactly.** The S1–S3 JSON outputs are byte-identical, the S1 DXF has the same entities, and the nine CubiCasa5K and CVC-FP sample sheets give identical results.
- **Tests:** 29 tests in about 10 s (`python -m pytest tests`):
  - unit tests per stage;
  - an end-to-end test on a synthetic two-room plan, through to JSON and DXF;
  - an oracle test that rasterises held-out Swiss Dwellings floors into perfect labels (`oracle.py`).
- **Environment and model:** pinned `requirements.txt` and `MODEL_CARD.md`.
- **Oracle finding.** On 60 held-out floors with perfect labels, post-processing alone finds rooms with recall 0.77 and precision 0.80 (one-to-one matching), median area error −1.7%. Most misses are kitchens, living areas and corridors that Swiss Dwellings separates without a wall. Free space between walls cannot split them. This measures the limit of the region approach (§3.2) and supports the room-boundary head of step 5.
- **Fixes found by the tests and the oracle:**
  - polygons traced from masks were half a pixel too small on every side;
  - simplification with a 1 px tolerance tilted whole room sides, losing about 1% of the area (now 0.25 px);
  - stamp areas with an explicit m² now win over bare numbers such as room numbers;
  - hyphenated line breaks in stamps are rejoined;
  - the deskew rotation is recorded in the output (§4.4).
- **Effect.** On the oracle test, the median area error went from −3.2% to −1.7%. On Landgut Lohn:
  - S1: median area error from 2.2% to 2.7%, mean room IoU from 0.88 to 0.89. The segmenter draws S1's outlined walls slightly thin, which the old bias happened to offset.
  - S2: from 3.5% to 3.3%.
  - S3: from 8.1% to 7.3%.
  - CubiCasa5K, 400 test plans: median room area error from 4.9% to 2.9%, rooms matched 65% (unchanged), mean room IoU from 0.64 to 0.65. Segmentation and openings are unchanged, as expected.
- **Still open from §4.1:**
  - a versioned JSON schema for the export;
  - continuous integration;
  - seeded renderer generators (with step 4b);
  - faster OCR.
  - Hungarian matching is in `oracle.py` but not yet in `evaluate.py` and `cubicasa_eval.py` (step 1c).
