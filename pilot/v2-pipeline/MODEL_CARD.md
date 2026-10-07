# Model Card: Pilot v2 Segmenter

*Pilot models, October 2026. Not for production use: see the [licence questions](#licences) and the [review](../../docs/reviews/2026-10-07-pipeline-and-pilot-v2.md). Weights are gitignored:*

| | v2 (current) | v1 |
|---|---|---|
| Weights | `data/model-v2/segmenter.pt` (SHA-256 starts `ac962255c8597731`) | `data/model/segmenter.pt` (`0d14e5a5e8d710a6`) |
| Use | `--model data/model-v2/segmenter.pt` (the scripts still default to v1) | Default of all scripts |

## Model

| | |
|---|---|
| Task | Semantic segmentation of floor plan sheets into background, wall, door, window, column, stairs |
| Architecture | U-Net with a ResNet-34 encoder (`segmentation-models-pytorch` 0.5.0), 24.4 M parameters. v2 adds four sigmoid heads: room interior, room boundary, floor void and text (`fpx/segment.py` reads both formats) |
| Input | RGB sheet at 50 px/m (2 cm per pixel), any size: inference runs on 1,024 px tiles with 128 px blended overlap and averages four flips. CPU by default, GPU with `FPX_DEVICE=cuda` |
| Output | Class probabilities per pixel (plus the head maps in v2); the pipeline (`fpx`) turns them into walls, openings, stairs, rooms and floor outline |
| Encoder initialisation | ImageNet weights from `timm` (licence question, see below) |

## Intended Use

- **In scope:** research on BBL's archive plans (raster scans and rendered vector PDFs), as stage 3 of the [pipeline](../../docs/pipeline.md), with every result checked by QA and a person.
- **Out of scope:** production data in the CAFM without review, scale estimation on its own, sections, elevations and site plans, and any processing of BBL plans outside Switzerland.

## Training Data

- **Swiss Dwellings** [v3.0.0](../../data/README.md) (CC BY 4.0), one floor per plan: 8,277 training floors, split by site from 367 validation and 295 test floors (`sd_prepare.py`).
- **v2 only:** 10 % of samples from storeys of public IFC models (IFC-Bench projects under CC BY 3.0/4.0 or MIT, `ifc_prepare.py`); two projects (`wbdg_office`, `digital_hub`) are held out for validation.
- **Rendering** on the fly in a random graphical style (`synth.py`): walls solid, outlined, hatched, grey or coloured; doors as swings, leaves or gaps; windows as glass lines or gaps; stairs with treads; fixtures; room stamps in DE/FR/IT in print, hand, calligraphic or Fraktur lettering (OFL fonts); dimension chains and axes; scan defects. Renderer 2.0 (v2) adds hatch and ornament distractors, casement windows, Swiss material hatches, column grids, colour overlays, several drawings per sheet and scale jitter 0.6–1.6 (v1: −20 % to +25 %).
- **No BBL plan** and no image of CubiCasa5K, CVC-FP, WAFFLE or FloorPlanCAD was used for training or model selection.

## Training

| | v2 | v1 |
|---|---|---|
| Schedule | 30,000 iterations × 16 crops of 512 px, AdamW (learning rate 5e-4, one-cycle, weight decay 1e-4), EMA of the weights (0.999), bf16 | Same, without EMA |
| Loss | Class-weighted cross-entropy (background 0.5, wall 1, door 3, window 3, column 5, stairs 2) plus Dice over the element classes; per head BCE plus soft Dice (weights interior 0.5, boundary 1, void 0.5, text 0.3) | Class weights 0.5 / 1 / 3 / 3 / 3 / 1.5, no heads |
| Checkpoint | Last iteration, also best on 400 validation renders (mean element IoU 0.744) | Iteration 28,000 of 30,000 |
| Compute | RunPod RTX 4090 (EU), 49 minutes, about USD 0.60. Public data only on the pod | 37.5 minutes, about USD 0.60 |

Reproduce: `python sd_prepare.py`, `python ifc_prepare.py`, then `python train.py --iters 30000 --batch 16 --workers 12 --require-cuda` on a GPU ([requirements.txt](requirements.txt), CUDA build of torch).

## Evaluation

### Benchmarking Method

The models are trained only on synthetic renders and evaluated **zero-shot** on public benchmarks, each with its published protocol where one exists:

- **FloorPlanCAD** (Fan et al., ICCV 2021): semantic symbol spotting. Each block of the November 2021 test split (5,502 blocks) is rendered black on white at 50 px/m; each vector primitive takes the majority class of points sampled along it (Eq. 10); F1 per category over primitives weighted by log(1 + length) (§10.1), with the categories of Fig. 7. Only door, window, stair and wall are scored (curtain wall ignored); the overall F1 over 30 classes cannot be computed for a four-class model. `fpcad_eval.py`.
- **CubiCasa5K** (Kalervo et al. 2019, Table 4): pixel IoU per class on the 400 official test plans, one confusion matrix pooled over the split. For the comparison, wall includes the door and window openings, as in CubiCasa's labels. `cubicasa_eval.py`.
- **WAFFLE** (Ganon et al. 2025, Table 4): pixel precision, recall and IoU per class, pooled over the 110 annotated plans; wall includes predicted columns. `harness.py waffle`.
- **CVC-FP** (Dodge et al. 2017): wall IoU (Jaccard index) and mean IoU and mean accuracy over wall vs. non-wall, pooled over all 122 plans. `harness.py cvcfp`.
- **Own metrics:** rooms matched one to one (Hungarian on IoU ≥ 0.5), area errors, openings and connectivity, on held-out Swiss Dwellings floors and on BBL sheets (`metrics.py`).

Every published row below was trained **in-domain** (on the benchmark's own training data), except on WAFFLE and the cross-dataset CVC-FP row, which are zero-shot like ours. The gap therefore measures the domain shift as much as the method. Deviations from the published protocols: plans without a scale (WAFFLE, CVC-FP) are scaled from detected door widths, so a wrong scale is part of our error; scoring happens at our working resolution (50 px/m), since the papers do not state theirs; test-time augmentation is four flips (CubiCasa: rotations). Sources and protocol details: [published baselines](../../docs/reports/2026-10-07-published-baselines.md).

### FloorPlanCAD

Length-weighted F1 per category, in the layout of Fan et al. Table 3:

| Method | Training | Door | Window | Stair | Wall | Mean of 4 | F1 (30 classes) | wF1 (30 classes) |
|---|---|---|---|---|---|---|---|---|
| HRNetV2 W18 | In-domain | 0.821 | 0.620 | 0.845 | 0.620 | 0.727 | 0.656 | 0.683 |
| HRNetV2 W48 | In-domain | 0.811 | 0.640 | 0.847 | 0.624 | 0.731 | 0.666 | 0.693 |
| DeepLabv3+ R50 | In-domain | 0.828 | 0.659 | 0.856 | 0.630 | 0.743 | 0.680 | 0.705 |
| DeepLabv3+ R101 | In-domain | 0.837 | 0.666 | 0.852 | 0.634 | 0.747 | 0.688 | 0.714 |
| GCN (PanCADNet) | In-domain | 0.848 | 0.709 | 0.857 | 0.814 | 0.807 | 0.806 | 0.798 |
| **Pilot v1** | Zero-shot | 0.147 | 0.293 | 0.298 | 0.362 | 0.275 | n/a | n/a |
| **Pilot v2** | Zero-shot | 0.108 | 0.306 | 0.325 | 0.387 | 0.281 | n/a | n/a |

- Published rows are on FloorPlanCAD V1 (30 classes, never released); ours are on the November 2021 release. No published number exists on this split. Later vector models reach wF1 0.80–0.93 on the August 2021 release (SymPoint family, CADSpotting, VecFormer).
- **Doors** in FloorPlanCAD are the leaf and swing arc; ours are the opening in the wall, so most door primitives vote background. Counting any foreground prediction on the primitive raises door F1 to 0.48 (v1) and 0.46 (v2).
- These are construction drawings (dimension chains in 90 % of blocks, axes, hatches, furniture), the style furthest from our training data and the main gap to close.

### CubiCasa5K

Pixel IoU on the 400 test plans:

| Model | Training | Wall (incl. openings) | Door | Window |
|---|---|---|---|---|
| CubiCasa5K model (Kalervo et al. 2019) | In-domain | 0.730 | 0.536 | 0.668 |
| **Pilot v1** | Zero-shot | 0.619 | 0.495 | 0.570 |
| **Pilot v2** | Zero-shot | 0.699 | 0.451 | 0.608 |

Our own metrics (v1 → v2): wall IoU without openings 0.575 → 0.665, doors found 88 % → 83 %, windows found 70 % → 72 %, opening precision 81 % → 87 %, rooms matched 65 % → 61 % with precision 69 % → 76 %, median room area error 2.9 % → 2.4 %.

### WAFFLE

Pixel precision / recall / IoU on the 110 annotated plans; all rows zero-shot:

| Model | Training | Walls | Doors | Windows |
|---|---|---|---|---|
| CubiCasa5K model (Ganon et al. 2025) | CubiCasa5K | 0.737 / 0.590 / 0.488 | 0.201 / 0.163 / 0.099 | 0.339 / 0.334 / 0.202 |
| Diffusion wall segmenter (Ganon et al. 2025) | CubiCasa5K walls | 0.746 / 0.805 / 0.632 | n/a | n/a |
| **Pilot v1** | Synthetic | 0.741 / 0.704 / 0.565 | 0.176 / 0.126 / 0.079 | 0.141 / 0.288 / 0.105 |
| **Pilot v2** | Synthetic | 0.797 / 0.730 / 0.615 | 0.308 / 0.128 / 0.100 | 0.188 / 0.508 / 0.159 |

WAFFLE's interior and background classes are not scored yet. The diffusion model needs 16 × 50 denoising steps per plan.

### CVC-FP

Wall vs. non-wall on all 122 plans, in %:

| Model | Training | Mean accuracy | Mean IoU | Wall IoU (JI) |
|---|---|---|---|---|
| FCN-2s (Dodge et al. 2017) | CVC-FP, 5-fold | 97.3 | 94.4 | 89.2 |
| FCN-2s (Dodge et al. 2017) | R-FP only (cross-dataset) | 84.2 | 81.7 | 64.7 |
| **Pilot v1** | Synthetic | 91.7 | 82.5 | 66.2 |
| **Pilot v2** | Synthetic | 92.5 | 89.1 | 78.7 |

Rooms (own metric, one to one): recall 0.47 → 0.61, precision 0.56 → 0.67.

### Own Benchmarks

| Benchmark | v1 | v2 |
|---|---|---|
| Synthetic validation, own renderer (400 renders, IoU wall / door / window / column / stairs) | 0.91 / 0.77 / 0.85 / 0.41 / 0.96 (renderer 1) | 0.86 / 0.64 / 0.72 / 0.58 / 0.92 (renderer 2.0, harder; heads: interior 0.95, boundary 0.75, void 0.26, text 0.71) |
| Held-out Swiss Dwellings renders, full pipeline (287 test floors, the same frozen renders for both; rooms recall / precision, wall IoU, door / window F1) | 0.81 / 0.74, 0.81, 0.86 / 0.82 | 0.82 / 0.81, 0.91, 0.92 / 0.92 |
| Held-out IFC renders (validation, IoU wall / door / window) | n/a | 0.78 / 0.15 / 0.21 |
| Landgut Lohn S1, 2005 CAD print (BBL, local) | Walls 0.87, rooms 15/15 (mean IoU 0.89), area vs. stamp −0.6 %, vs. reference polygon +2.3 % | Walls 0.90, rooms 15/15 (mean IoU 0.93), area vs. stamp −6.3 %, vs. reference polygon −1.2 % |
| Landgut Lohn S3, survey scan with stucco (BBL, local) | Walls 0.29, room precision 0.43, opening precision 0.55, connections 8/12 | Walls 0.34, room precision 0.77, opening precision 0.82, connections 11/12 |

- The post-processing rules were partly written while looking at the Landgut Lohn sheets, so those results are optimistic; details in the [pilot README](README.md#results).
- On S1, v2 rooms fit the reference polygons better, but are 6 % smaller than the stamp areas: v1's walls were too thin, which made rooms larger and closer to the stamps by accident.
- Post-processing alone (oracle labels on 60 held-out Swiss Dwellings floors, `oracle.py`): rooms recall 0.77, precision 0.80, median area error −1.7 %. Free space that Swiss Dwellings splits without a wall cannot be split.

Reproduce (any model via `--model`; `FPX_DEVICE=cuda` on a GPU): `fpcad_eval.py --workers 8`, `cubicasa_eval.py test --out data/cubicasa-v2`, `harness.py {render,cvcfp,waffle} --out data/harness/<mode>-v2.json`, `run_pipeline.py --out data/out-v2` then `evaluate.py --out data/out-v2`.

## Limitations

- **Construction drawings** (FloorPlanCAD): mean F1 0.28 against 0.73–0.81 in-domain. Dimension chains, axes, hatches and furniture are read as walls, and door symbols are not recognised as such. BIM-generated drawings show the same gap (doors 0.15, windows 0.21 on held-out IFC renders).
- **Doors** got worse from v1 to v2 on real data (CubiCasa 0.50 → 0.45, FloorPlanCAD), while windows and walls improved. Door instances are not separated: adjacent doors merge.
- **Columns** are weak on real plans (CubiCasa 0.10) despite 0.58 on synthetic renders.
- **Annotation symbols:** grid callouts and section markers are partly read as walls (on one WAFFLE sheet the false wall share fell from 44 % in v1 to 11 % in v2), and the interior head can mark empty paper as a room.
- **Void head** is weak (0.26 on synthetic renders); stair and shaft voids still come mainly from labels and stair outlines.
- **Scale** must be known or proposed: the model assumes 50 px/m.
- **Synthetic validation does not select checkpoints reliably** for real sheets; a small real validation set is needed.

## Licences

- **Training data:** Swiss Dwellings, CC BY 4.0 (attribution in `data/README.md`); IFC-Bench models, CC BY 3.0/4.0 and MIT (per model in `ifc_prepare.py`). Fonts: SIL OFL.
- **Code:** `segmentation-models-pytorch` (MIT), `timm` (Apache-2.0), PyTorch (BSD-3).
- **Open question:** whether ImageNet-pretrained encoder weights may be used in production. The review lists it for BBL's legal service (§6.4); a training run from scratch or from a licence-clean encoder is the fallback.
- **Benchmarks** are used for evaluation only and their outputs stay local: CubiCasa5K (CC BY-NC-SA 4.0), CVC-FP (CC BY-NC), FloorPlanCAD (CC BY-NC 4.0), WAFFLE (licences per image).
