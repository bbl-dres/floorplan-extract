# Pilot v2: Graph Pipeline with a Style-Randomised Segmenter

*October 2026. Pipeline stages 0–10 from the [pipeline design](../../docs/pipeline.md), on the same Landgut Lohn floor as [pilot v1](../landgut-lohn-og1/README.md), plus two historical scans. The segmenter (models v1–v3) is described in [Model](#model) below; the detailed benchmark tables are in the [second review](../../docs/reviews/2026-10-08-pilot-v2-second-review.md#4-measurements).*

## Question

Can a segmenter trained only on public data, rendered in random graphical styles, drive the structural pipeline on real BBL sheets it has never seen? The sheets come in three styles. Training data and BBL plans stay strictly apart: training ran on RunPod with public data only, and the BBL sheets were processed locally only.

## Setup

| Step | What | Where |
|---|---|---|
| Training data | [Swiss Dwellings v3.0.0](../../data/README.md) (CC BY 4.0), one floor per plan, split by site (8,277 train / 367 validation / 295 test), plus 10 % storeys of public IFC models | `sd_prepare.py`, `ifc_prepare.py` |
| Renderer | Every sample drawn on the fly in a random style: walls solid, outlined, hatched, grey or coloured; doors with swing, leaf only or as gaps; windows as glass lines or gaps; stairs, fixtures, room stamps in DE/FR/IT print, hand, calligraphic or Fraktur lettering; dimension chains, axes, scan defects; since renderer 3.0 a negative set of construction-drawing clutter | `synth.py` (`synth_preview.py` draws a sample grid) |
| Segmenter | U-Net, ResNet-34 encoder, 30,000 iterations × 16 crops; v3 is the default (`common.MODEL`) | `train.py`, RunPod RTX 4090, about USD 0.65 per run |
| Pipeline | Stages 0–10: triage, resampling to 50 px/m, sheet layout and masking (title block, notes, scan borders), text layer (native PDF text + RapidOCR), segmentation, wall graph with straight segments and thickness, openings and passages, stairs and voids, rooms with separation lines, stamps, net/gross areas and connections, QA, export to JSON, DXF (CAD-Richtlinie layers), Excel and IFC 4.3 | [`fpx/`](scripts/fpx/__init__.py), one module per stage; every threshold with its reason in [`fpx/config.py`](scripts/fpx/config.py); `run_pipeline.py`; local CPU, 40–75 s per sheet (mostly OCR) |
| Evaluation | Against the curated 2005 reference (15 rooms, 25 walls, 37 openings); the scans registered by translation only | `evaluate.py`, one scorer for every benchmark in `fpeval/` |
| Tests | Unit tests per stage, an end-to-end test on a synthetic two-room plan, an oracle test on perfect labels of held-out floors, a golden-output test on a synthetic floor | `python -m pytest -q tests` from `scripts/` (233 tests, about a minute) |
| Viewer | A fixed set of plans (the curated core set and the Landgut Lohn sheets), each with its pipeline versions behind a history control, in 2D/3D with rooms, walls, openings, stairs, voids, GF and connections; the data is tracked in `viewer-data/`, so the page runs on GitHub Pages | `export_viewer.py`, `viewer.html` |
| Public benchmarks | CubiCasa5K, FloorPlanCAD, WAFFLE, CVC-FP, held-out Swiss Dwellings renders, and a curated set of public sheets | `cubicasa_eval.py`, `fpcad_eval.py`, `harness.py`, `curated.py`, `fpeval/datasets/` |

**Layout.** This folder keeps the README, the [viewer](viewer.html) and `data/` (gitignored: models, outputs, plan-derived files). Every script, the `fpx` package, `fpeval` and the tests live in [`scripts/`](scripts/) and run from there (`cd pilot/v2-pipeline/scripts`). Scripts share `--model`, `--out`, `--config` and `--threads`. Run order: `sd_prepare.py` → `train.py` (GPU) → `run_pipeline.py` → `evaluate.py` → `curated.py run core` → `export_viewer.py`, then open `viewer.html` over HTTP (`python -m http.server` in this folder, or GitHub Pages: `https://<owner>.github.io/<repo>/pilot/v2-pipeline/viewer.html`). Environment: [requirements.txt](scripts/requirements.txt). The training ablations are in [`scripts/experiments.md`](scripts/experiments.md).

**Viewer.** One entry per plan: the 24 core plans of the [curated set](../../data/curated/README.md) (one per challenge, all publicly showable) and the three Landgut Lohn sheets. Every run folder of a plan is a version (v1, v2, v3; a folder `data/curated-model-<name>` holds a benchmark model's output and shows as a version of that kind); the latest pipeline version is the default, the `‹ v3 ›` control and the keys H / Shift+H step through them, and the History block under Details lists the versions with rooms, matches and wall IoU side by side. `export_viewer.py` writes `viewer-data/` (index, one JSON per version fetched on demand, one image per plan; about 30 MB); `export_viewer.py --local` adds the benchmark-only plans to a gitignored export under `data/out/` that `viewer.html#data=local` reads.

## Test Sheets

| Sheet | Input | Style |
|---|---|---|
| S1 | 2005 vector PDF, 1:100 printed reduced | Outlined walls, door and casement-window swings, stamps with names and areas; the stamps are outlined, so they need OCR |
| S2 | Historical scan 0056, 1:50, 400 dpi, 1-bit | Solid black (poché) walls, windows as gaps, calligraphic labels, stippled background |
| S3 | Later survey scan 0066, 1:50, 400 dpi, 1-bit | Outlined walls with reveals and pilasters, ceiling stucco drawn inside rooms, almost no labels |

S2 and S3 show earlier states of the building. Against the 2005 reference, only unchanged elements are meaningful: the outline, exterior openings and some rooms.

This README and `evaluate.py` contain room names and areas derived from the non-public plans; check them before publishing the repository.

## Results

Pilot v1 → v2 → v3 (v3: model v3 with the wall focus of 8 October; v2 figures as published in October before the reviews):

| | S1 (CAD print) | S2 (poché scan) | S3 (survey scan) |
|---|---|---|---|
| Walls IoU vs 2005 (segmenter mask) | 0.61 → 0.87 → **0.86** | 0.36 → 0.58 → **0.61** | – → 0.29 → **0.59** |
| Walls IoU, regularised straight segments (v3 only) | 0.79 | 0.59 | 0.57 |
| Rooms matched (IoU ≥ 0.5) | 14/15 → 15/15 → **15/15** | 6 unchanged → 12/15 → 12/15 | 12/15 → **13/15** |
| Mean room IoU | 0.76 → 0.89 → **0.92** | – → 0.76 → 0.74 | 0.68 → **0.78** |
| Median area error vs stamp | 14.7 % → 2.7 % → **2.8 %** | – → 3.3 % → 6.7 % | 7.3 % → **4.4 %** |
| Room names | 15/15 (cloud VLM) → 15/15 (local OCR) → **15/15** | 11/11 (VLM) → 8/11 → **9/11** after dictionary correction (raw OCR 0/11) | no names on the sheet |
| Exterior / interior openings found | ~22/24, 12/13 → 24/24, 12/13 → 24/24, 10/13 | – → 24/24, 10/13 → same | 24/24, 9/13 → **24/24, 12/13** |
| Room connections found | – → 12/12 → 10/12 | – → 11/12 → 7/12 (10/12 with the earlier mapping) | 8/12 → 8/12 (9/12) |
| GF outline IoU | – → 0.92 → 0.91 | – → 0.91 → 0.91 | 0.91 → 0.91 |

Model v3 finds four fewer doors than v2 on S1 (11 instead of 15: two reference openings and two connections lost), the one regression of the retraining on these sheets. Connections are now mapped through the one-to-one room matching (IoU ≥ 0.5); the v2 figures used the earlier mapping at IoU ≥ 0.3, shown in brackets for v3. S3's wall area is 45 m² against the reference's 52 m² (v2 before the wall focus: 99 m², with the scan border and the ceiling stucco as walls). The rooms are bounded by the wall faces themselves, so areas need no snapping step; since 7 October polygons follow pixel edges and are simplified at a quarter pixel (half a pixel per side and about 1 % of a room's area, which the earlier output lost).

### Public Benchmarks

All pilot rows are zero-shot (trained on renders only); the published rows are in-domain unless marked. Protocols, sources and the full tables with v1 and v2: [second review §4.4](../../docs/reviews/2026-10-08-pilot-v2-second-review.md#44-published-protocols) and the [published baselines](../../docs/reports/2026-10-07-published-baselines.md).

| Benchmark | Metric | Published | Pilot v3 |
|---|---|---|---|
| CubiCasa5K, 400 test plans | pixel IoU wall (incl. openings) / door / window | 0.730 / 0.536 / 0.668 (CubiCasa model) | 0.706 / 0.551 / 0.649 |
| FloorPlanCAD, Nov 2021 test split | length-weighted F1, mean of door, window, stair, wall | 0.73–0.81 (raster and graph models, on the unreleased V1 labels) | 0.321 |
| WAFFLE, 110 annotated plans | wall precision / recall / IoU | 0.746 / 0.805 / 0.632 (diffusion wall segmenter, zero-shot) | v2 0.797 / 0.730 / 0.615; v3 0.823 / 0.761 / 0.654 on the first 40 plans (full run queued) |
| CVC-FP, 122 scans | wall IoU (Jaccard), % | 89.2 (5-fold), 64.7 cross-dataset (FCN-2s) | 78.1; rooms recall 0.77, precision 0.76 |
| Swiss Dwellings renders, 117 held-out floors | rooms recall / precision, wall IoU, door / window F1 | – | 0.884 / 0.952, 0.90, 0.93 / 0.93 |

## Model

| | v3 (default) | v2 | v1 |
|---|---|---|---|
| Weights (gitignored) | `data/model-v3/segmenter.pt`, SHA-256 `11fbeb97de291ffc…` (run E3 of the second review) | `data/model-v2/segmenter.pt`, `ac962255c8597731…` | `data/model/segmenter.pt`, `0d14e5a5e8d710a6…` |
| Renderer | 3.0 with the negative set | 2.0 | 1 |
| Heads besides the six classes | interior, boundary, void, text, swing | interior, boundary, void, text | none |
| Frozen validation, mean element IoU | 0.751 (iteration 28,000 of 30,000) | 0.744 (own renderer-2.0 set) | – |

- **Task and architecture:** semantic segmentation of a sheet into background, wall, door, window, column and stairs; U-Net with a ResNet-34 encoder (`segmentation-models-pytorch`, ImageNet initialisation from `timm`), 24.4 M parameters, plus sigmoid heads for the room interior, room boundaries (open-plan separations), floor voids, text and (v3) door swings. Input: RGB at 50 px/m, any size, inferred on 1,024 px tiles with blended overlap and four flips; CPU by default, `FPX_DEVICE=cuda` on a GPU. `fpx/segment.py` reads every version.
- **Training:** 30,000 iterations × 16 crops of 512 px, AdamW (5e-4, one-cycle), EMA 0.999, bf16; class-weighted cross-entropy plus Dice, BCE plus Dice per head; the checkpoint with the best mean element IoU on a frozen set of 400 renders. 52 minutes on an RTX 4090. Reproduce from `scripts/`: `python train.py --iters 30000 --batch 16 --workers 14 --require-cuda --negatives --val-cache ../data/val-v3neg.pt --out ../data/model-v3`.
- **Data:** no BBL plan and no image of CubiCasa5K, CVC-FP, WAFFLE or FloorPlanCAD was used for training or model selection. Benchmarks are evaluation only and their outputs stay local (CubiCasa5K CC BY-NC-SA 4.0, CVC-FP CC BY-NC, FloorPlanCAD CC BY-NC 4.0, WAFFLE per image).
- **Intended use:** research on BBL archive plans as stage 3 of the pipeline, every result checked by QA and a person. Not for production data in the CAFM without review, not for scale estimation on its own, not for sections, elevations or site plans.
- **Limitations:** construction drawings (FloorPlanCAD 0.32 against 0.73–0.81 in-domain: dimension chains, hatches and furniture still partly read as walls, and door symbols drawn as leaf and arc are not the opening our class means); columns weak on real plans (CubiCasa IoU 0.10); open-plan areas merge unless a drawn line or the boundary head separates them; the scale must be known or proposed; synthetic validation does not select the checkpoint for real sheets (a small real validation set is needed); on S1 v3 misses four doors v2 found.
- **Licences:** Swiss Dwellings CC BY 4.0, IFC-Bench models CC BY 3.0/4.0 and MIT (per model in `ifc_prepare.py`), fonts SIL OFL; `segmentation-models-pytorch` MIT, `timm` Apache-2.0, PyTorch BSD-3. Open question for BBL's legal service: ImageNet-pretrained encoder weights in production (a licence-clean encoder or training from scratch is the fallback).

## Findings

1. **Style randomisation transfers.** A model trained only on rendered Swiss Dwellings floors segments three unseen drawing styles: outlined CAD, poché and survey. Wall IoU reaches 0.86 on the CAD print and 0.58–0.60 on the scans (v1 classical CV: 0.61 and 0.36). No BBL plan was used for training.
2. **The graph pipeline fixes v1's main failures.** Doors and windows close rooms, so rooms no longer merge through doorways; areas are within 3 % (median) of the stamps on the CAD print with no snapping step, because rooms are bounded by the wall faces themselves; the stairwell void is found from its label and deducted from the GF.
3. **Room connections are a by-product.** Every door, passage and interior opening links two rooms; v2 found all 12 reference connections on S1, v3 10 of 12 (four doors missed by the model).
4. **Local OCR is enough for printed stamps, not for calligraphy.** PP-OCRv5 Latin read all 15 printed names and 12 of 15 areas on S1 locally; on the calligraphic scan no name is exact (mean similarity 0.75), a room-name vocabulary repairs 9 of 11. A self-hosted VLM remains the candidate for hand lettering.
5. **Historical plans draw open archways like windows.** Treating "windows" in interior walls as openings raised the connections found on S2 from 8 to 11 of 12.
6. **Synthetic validation is not a reliable model selector.** The best synthetic checkpoint is not the best on real sheets, for v1 and for v3 alike; a small real validation set of BBL sheets is the missing piece.
7. **Layout masking before segmentation pays off.** Title blocks, notes strips and scan borders became walls and text on whole sheets; masking them (stage 1b, without OCR) halved S3's false wall area and removed the title-block bars on construction drawings.
8. **Walls as straight segments with a thickness work on real sheets.** Wall pieces are first rejected with a reason (fixture outlines, stair railings, thin strokes, text), then the skeleton is simplified, snapped to the plan's directions and refitted to the mask; the axis runs on through doors and windows, so a facade is one wall with opening intervals, and the IFC hosts every opening in its wall. On S1 the regularised walls reach IoU 0.87 against the mask (0.79 against the reference; mask 0.86).
9. **Open plan needs a drawn line.** A thin or dashed line from wall to wall between two stamped areas now divides them (CVC-FP's living-room example: the Séjour is cut off along its dashed line); areas with several stamps and no line stay one room with a QA warning.
10. **Columns are weak** (CubiCasa IoU 0.10); they are rare in Swiss Dwellings and none occur on these sheets.

## Caveats

- One building, one floor, three sheets. The **model** never saw them. Several **post-processing rules** were written after inspecting failures on these sheets (stair handling, slit closing, sealing leaking rooms, interior windows as openings, passages, the rejection rules of the wall focus), so the results are optimistic until repeated on held-out BBL sheets; the public benchmarks and the curated set are the check.
- The name vocabulary was written after seeing the historical sheet; the raw OCR score (0/11 exact) is the honest one.
- S1's scale comes from the sheet's own dimension strings; the scans use the title-block scale and the scan resolution. The stamp-area cue agrees with S1's scale.
- The DXF export follows the CAD-Richtlinie layer names; plan-check reads DWG only, so the DXF output has not been run through it yet.

## Next Steps

1. **Real validation set:** label 10–20 BBL sheets from other buildings and styles for checkpoint selection and threshold choice, then rerun this pilot unchanged as a held-out test.
2. **Doors on CAD prints:** v3 misses four of S1's doors that v2 found; check the casement/door renderer change against real casement windows before the next training run.
3. **Open plan without a line:** areas with several stamps and no drawn divider (kitchen counter only) need a stamp-driven split or the boundary head trained on more open-plan renders.
4. **Faces of the wall graph** reconciled with the free-space regions (P4 of the first review), now that walls are straight segments with hosted openings.
5. **Stamps on historical plans:** a self-hosted VLM against OCR plus vocabulary on calligraphic labels, entirely locally.
6. **DWG export and plan-check:** write DWG (e.g. through ODA) and validate the output with plan-check.
