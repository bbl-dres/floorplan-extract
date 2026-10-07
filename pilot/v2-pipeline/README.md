# Pilot v2: Graph Pipeline with a Style-Randomised Segmenter

*October 2026. Pipeline stages 0–10 from the [pipeline design](../../docs/pipeline.md), on the same Landgut Lohn floor as [pilot v1](../landgut-lohn-og1/README.md), plus a second historical scan.*

## Question

Can a segmenter trained only on public data, rendered in random graphical styles, drive the structural pipeline on real BBL sheets it has never seen? The sheets come in three styles. Training data and BBL plans stay strictly apart: training ran on RunPod with public data only, and the BBL sheets were processed locally only.

## Setup

| Step | What | Where |
|---|---|---|
| Training data | [Swiss Dwellings v3.0.0](../../data/README.md) (CC BY 4.0): one floor per plan, 8,939 floors split by site (8,277 train / 367 validation / 295 test) | `sd_prepare.py`, local |
| Renderer | Every sample rendered on the fly in a random style: walls solid, outlined, hatched, grey or coloured; doors with swing, leaf only or as gaps; windows as glass lines or gaps; stairs with treads; fixtures; room stamps (DE/FR/IT names, areas, numbers, AOIDs) in print, hand, calligraphic or Fraktur lettering ([OFL fonts](../../data/README.md)); dimension chains, axes; scan defects. Labels: wall, door, window, column, stairs | `synth.py` (`synth_preview.py` writes a sample grid) |
| Segmenter | U-Net, ResNet-34 encoder (ImageNet), 512 px crops at 50 px/m, 30,000 iterations × 16 samples | `train.py`; RunPod RTX 4090 (EU), 37.5 min, about $0.60 |
| Pipeline | Stages 0–10 on three sheets: triage, render or resample to 50 px/m, text layer (native PDF text + RapidOCR with PP-OCRv5 Latin), segmentation with flip averaging, wall graph, openings and wall-gap passages, stairs and voids, rooms, stamps, GF and connections, QA, export to JSON and DXF in the CAD-Richtlinie layers | [`fpx/`](fpx/__init__.py) (one module per stage; every threshold with its reason in [`fpx/config.py`](fpx/config.py)), `sheets.py`, `run_pipeline.py`; local CPU, 40–75 s per sheet (mostly OCR) |
| Evaluation | Against the curated 2005 reference (15 rooms, 25 walls, 37 openings); the scans registered by translation only | `evaluate.py` |
| Tests | Unit tests per stage; an end-to-end test on a synthetic two-room plan; an oracle test that rasterises held-out Swiss Dwellings floors into perfect labels, to check the post-processing without the segmenter | `tests/`, `oracle.py`; `python -m pytest tests` (about 10 s) |
| Viewer | Original sheets with rooms, walls, openings, stairs, voids, GF and connections in 2D/3D, plus public benchmark sheets with their ground truth | `export_viewer.py`, `viewer.html` |
| Public benchmarks | CubiCasa5K zero-shot test (400 plans); sample sheets from CubiCasa5K, CVC-FP, WAFFLE and Swiss plans on Wikimedia Commons, with a scale proposed from door widths where none is known | `cubicasa.py`, `cubicasa_eval.py`, `bench.py`, local CPU |
| IFC render source | 17 permissively licensed IFC-Bench models cut into 72 storeys in the Swiss Dwellings floor format, for the next training run | `ifc_prepare.py` |

Run order: `sd_prepare.py` → `train.py` (GPU) → `run_pipeline.py` → `evaluate.py` → `bench.py` (optional) → `export_viewer.py`; then open `viewer.html` (e.g. `viewer.html#sheet=s2&colour=accuracy&graph=1&ref=1`). Model, outputs and plan-derived files stay in `data/` (gitignored). Environment: [requirements.txt](requirements.txt). Model details: [MODEL_CARD.md](MODEL_CARD.md). Thresholds can be overridden per run with `run_pipeline.py --config file.json`.

**Segmenter on held-out synthetic floors** (IoU, 400 renders from other sites): walls 0.91, stairs 0.96, windows 0.85, doors 0.77, columns 0.41. By style era: walls 0.90–0.93 everywhere; windows 0.74 on historical styles vs 0.88 on CAD styles; columns weak except in mixed styles.

## Test Sheets

| Sheet | Input | Style |
|---|---|---|
| S1 | 2005 vector PDF, 1:100 printed reduced | Outlined walls, door and casement-window swings, stamps with names and areas. The text layer holds only dimensions and fixture words; the stamps are outlined, so they need OCR |
| S2 | Historical scan 0056, 1:50, 400 dpi, 1-bit | Solid black (poché) walls, windows as gaps, calligraphic labels, stippled background |
| S3 | Later survey scan 0066, 1:50, 400 dpi, 1-bit | Outlined walls with reveals and pilasters, ceiling stucco drawn inside rooms, almost no labels |

S2 and S3 show earlier states of the building. Against the 2005 reference, only unchanged elements are meaningful: the outline, exterior openings and some rooms.

This README and `evaluate.py` contain room names and areas derived from the non-public plans; check them before publishing the repository.

## Results

| | S1 v1 | **S1 v2** | S2 v1 | **S2 v2** | **S3 v2** |
|---|---|---|---|---|---|
| Walls IoU vs 2005 | 0.61 | **0.87** | 0.36 | **0.58** | 0.29 |
| Rooms matched (IoU ≥ 0.5) | 14/15 | **15/15** (17 found: 2 small unlabelled strips) | 6 unchanged rooms 0.82–0.90 | **12/15** | 12/15 |
| Mean room IoU | 0.76 | **0.89** | – | 0.76 | 0.68 |
| Median area error vs 2005 stamp | 14.7% (3.8% after a tuned 10 cm snap) | **2.7%**, no snap | – | 3.3% | 7.3% |
| Room names | 15/15 (cloud VLM) | **15/15 (local OCR)** | 11/11 (cloud VLM) | 0/11 raw OCR, **8/11 after dictionary correction** | no names on sheet |
| Stamp areas read | 14/15 (cloud VLM) | 12/15 (local OCR) | – | – | – |
| Exterior / interior openings found | ~21–23/24, 12/13 (VLM count) | **24/24, 12/13** | – | 24/24, 10/13 | 24/24, 9/13 |
| Room connections found | – | **12/12** (3 extra) | – | 11/12 | 8/12 |
| GF outline IoU | – | 0.92 (+8.9%) | – | 0.91 | 0.91 |

The three extra connections on S1 are openings or open transitions visible on the plan but not modelled in the reference: the open stair landing, hall to Vorplatz, and a bathroom door.

*Updated 7 October 2026, after the refactor into `fpx`.* The refactor itself reproduced every output byte for byte. Two geometry fixes followed:
- **Pixel-edge polygons.** Polygons traced from masks ran through pixel centres, so every room, wall and floor outline was half a pixel (1 cm) too small on each side.
- **Finer simplification.** Room sides are now simplified with a 0.25 px tolerance instead of 1 px. The coarser tolerance tilted whole sides across a one-pixel step and cost about 1% of a room's area.

On the oracle test, the median room area error went from −3.2% to −1.7%. On S1 it rose from 2.2% to 2.7%: the segmenter draws S1's outlined walls slightly thin, which the old bias happened to offset. Wall IoU on S1 fell from 0.89 to 0.87 for the same reason, while S2 and S3 improved.

### Public Benchmarks

On the 400 CubiCasa5K test plans, never seen in training, the segmenter scores pixel IoU 0.57 for walls, 0.49 for doors and 0.57 for windows. It finds 88% of doors and 70% of windows, and matches 65% of rooms, with a median area error of 2.9% (4.9% before the polygon fix of 7 October). CubiCasa's own in-domain model scores 0.73 / 0.54 / 0.67. Details, per-style results and the weak spots are in the [review](../../docs/reviews/2026-10-07-pipeline-and-pilot-v2.md#22-cubicasa5k-zero-shot-benchmark-new).

## Findings

1. **Style randomisation transfers.** A model trained only on rendered Swiss Dwellings floors segments three unseen drawing styles: outlined CAD, poché and survey. It reaches wall IoU 0.87 on the CAD print. On the poché scan, wall IoU against 2005 roughly doubles v1 classical CV (0.58 vs 0.36). No BBL plan was used for training.
2. **The graph pipeline fixes v1's main failures.** Doors and windows close rooms, so rooms no longer merge through doorways. Stairs become their own room via the stair outline. The stairwell void is found from its LUFTRAUM label and deducted from the GF (7.2 m², above the CAD-Richtlinie 5 m² rule). Areas are within 2.7% (median) of the stamps with no snapping step, because rooms are bounded by the wall faces themselves.
3. **Room connections are now a by-product.** Every door, passage and interior opening links two rooms. On S1 all 12 reference connections are found.
4. **Local OCR is enough for printed stamps, not for calligraphy.** PP-OCRv5 Latin read all 15 printed names and 12 of 15 areas on S1, entirely locally. On the calligraphic scan it gets close (mean similarity 0.72: "Herron-Timmer", "Doudoir", "festibule"), but no name is exact. A room-name vocabulary repairs 8 of 11. A self-hosted VLM remains the candidate for hand lettering.
5. **Historical plans draw open archways like windows.** Treating "windows" in interior walls as openings raised the connections found on S2 from 8 to 11 of 12.
6. **Synthetic validation is not a reliable model selector.** The final checkpoint was best on synthetic validation. On S1, though, it labelled many casement windows as doors and left thin slits next to doors, unlike an early checkpoint. Generic post-processing absorbs this: closing slits under 0.3 m and sealing rooms that leak through an unclosed opening. A small real validation set of BBL sheets is needed to choose checkpoints.
7. **Ornament is the next style gap.** On S3 the ceiling stucco drawn inside the Saal is partly read as walls, and the facades are drawn thicker. Walls come out at 105 m² vs 52 m², and the Saal splits into four pieces. Training renders never contained ceiling ornament.
8. **Columns are weak** (synthetic IoU 0.41). They are rare in Swiss Dwellings and none occur on these sheets.

## Caveats

- One building, one floor, three sheets. The **model** never saw them. The **post-processing rules** do not use the reference, but several were added after inspecting failures on these sheets: stair split and relabelling, slit closing, sealing leaking rooms, interior windows as openings, and conservative passages. The results are therefore optimistic until they are repeated on held-out BBL sheets.
- The name vocabulary was written after seeing the historical sheet, so the corrected-name score (8/11) is optimistic. The raw OCR score (0/11 exact) is not.
- S1's scale comes from the sheet's own dimension strings, calibrated in the reconstruction project. The scans use the title-block scale (1:50) and the scan resolution. The stamp-area cue in stage 9 agrees with S1's scale.
- The S1 region of interest (main building) was drawn by hand, as in v1. On the scans the pipeline finds the building itself.
- The DXF export follows the CAD-Richtlinie layer names. plan-check reads DWG only, so the DXF output has not been run through it yet.

## Next Steps

1. **Real validation set:** label 10–20 BBL sheets from other buildings and styles for checkpoint selection and threshold choice. Then rerun this pilot unchanged as a held-out test.
2. **Renderer:** add ceiling ornament and cornice lines, window reveals and pilasters, casement windows with swings, and more columns. Then retrain (about $0.60 per run).
3. **Stamps on historical plans:** compare a self-hosted VLM (e.g. Qwen3-VL) with OCR plus vocabulary on the calligraphic labels, entirely locally.
4. **Voids without labels:** add a void class, or detect stairwell voids geometrically.
5. **DWG export and plan-check:** write DWG (e.g. through ODA) and validate the output with plan-check.
