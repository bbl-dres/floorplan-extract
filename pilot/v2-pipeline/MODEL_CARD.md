# Model Card: Pilot v2 Segmenter

*Pilot models, October 2026. Not for production use: see the [licence questions](#licences) and the [reviews](../../docs/reviews/README.md). Weights are gitignored:*

| | v3 (current) | v2 | v1 |
|---|---|---|---|
| Weights | `data/model-v3/segmenter.pt` (SHA-256 starts `11fbeb97de291ffc`; run E3 of the [second review](../../docs/reviews/2026-10-08-pilot-v2-second-review.md)) | `data/model-v2/segmenter.pt` (`ac962255c8597731`) | `data/model/segmenter.pt` (`0d14e5a5e8d710a6`) |
| Use | Default of all scripts (`common.MODEL`, env `V2_MODEL`) | `--model data/model-v2/segmenter.pt` | `--model data/model/segmenter.pt` |
| Renderer | 3.0 with the negative set | 2.0 | 1 |
| Heads | interior, boundary, void, text, swing | interior, boundary, void, text | none |

## Model

| | |
|---|---|
| Task | Semantic segmentation of floor plan sheets into background, wall, door, window, column, stairs |
| Architecture | U-Net with a ResNet-34 encoder (`segmentation-models-pytorch` 0.5.0), 24.4 M parameters, six softmax classes. v2 adds four sigmoid heads (room interior, room boundary, floor void, text), v3 a fifth (door swing: leaf and swept sector). `fpx/segment.py` reads every format |
| Input | RGB sheet at 50 px/m (2 cm per pixel), any size: inference runs on 1,024 px tiles with 128 px blended overlap and averages four flips. CPU by default, GPU with `FPX_DEVICE=cuda` |
| Output | Class probabilities per pixel plus the head maps; the pipeline (`fpx`) turns them into walls, openings, stairs, rooms and floor outline. Since the second review the room stage reads the boundary head (open-plan separations, arbitrated by the stamps) and the interior head (building mask) |
| Encoder initialisation | ImageNet weights from `timm` (licence question, see below) |

## Intended Use

- **In scope:** research on BBL's archive plans (raster scans and rendered vector PDFs), as stage 3 of the [pipeline](../../docs/pipeline.md), with every result checked by QA and a person.
- **Out of scope:** production data in the CAFM without review, scale estimation on its own, sections, elevations and site plans, and any processing of BBL plans outside Switzerland.

## Training Data

- **Swiss Dwellings** [v3.0.0](../../data/README.md) (CC BY 4.0), one floor per plan: 8,277 training floors, split by site from 367 validation and 295 test floors (`sd_prepare.py`).
- **v2 and v3:** 10 % of samples from storeys of public IFC models (IFC-Bench projects under CC BY 3.0/4.0 or MIT, `ifc_prepare.py`; two projects held out for validation). In v2 the share was 18 % through a sampling slip, fixed in v3.
- **Rendering** on the fly in a random graphical style (`synth.py`): walls solid, outlined, hatched, grey or coloured; doors as swings, leaves or gaps; windows as glass lines or gaps; stairs with treads; fixtures; room stamps in DE/FR/IT in print, hand, calligraphic or Fraktur lettering (OFL fonts); dimension chains and axes; scan defects. Renderer 2.0 (v2) added hatch and ornament distractors, casement windows, Swiss material hatches, column grids, colour overlays, several drawings per sheet and scale jitter 0.6–1.6. Renderer 3.0 (v3) fixes the casement/door confusion (casements as two sashes only from 0.9 m, never a full-width single sash identical to a door; doors take the scale), labels shafts and lifts as voids, and draws a negative set of construction-drawing clutter as background: dimension chains on the facades and interior strings, section markers, detail bubbles, axis bubbles on plain axes, door and window tags, level markers, furniture.
- **No BBL plan** and no image of CubiCasa5K, CVC-FP, WAFFLE or FloorPlanCAD was used for training or model selection.

## Training

| | v3 | v2 | v1 |
|---|---|---|---|
| Schedule | 30,000 iterations × 16 crops of 512 px, AdamW (learning rate 5e-4, one-cycle, weight decay 1e-4), EMA of the weights (0.999), bf16 | Same | Same, without EMA |
| Loss | Class-weighted cross-entropy (background 0.5, wall 1, door 3, window 3, column 5, stairs 2) plus Dice over the element classes; per head BCE plus soft Dice per sample (weights interior 0.5, boundary 1, void 0.5, text 0.3, swing 0.5) | Same with four heads and batch-level Dice | Class weights 0.5 / 1 / 3 / 3 / 3 / 1.5, no heads |
| Checkpoint | Iteration 28,000, best mean element IoU 0.751 on a frozen validation set of 400 renders (`--val-cache`, shared by every run of the second review) | Last iteration (0.744 on a set re-rendered per run) | Iteration 28,000 of 30,000 |
| Compute | RunPod RTX 4090 (EU), 52 minutes, about USD 0.65. Public data only on the pod | 49 minutes | 37.5 minutes |

Reproduce: `python sd_prepare.py`, `python ifc_prepare.py`, then from [`code/`](code/) `python train.py --iters 30000 --batch 16 --workers 14 --require-cuda --negatives --val-cache ../data/val-v3neg.pt --out ../data/model-v3` on a GPU ([requirements.txt](code/requirements.txt), CUDA build of torch); the ablations are in [`code/experiments.md`](code/experiments.md).

## Evaluation

### Benchmarking Method

The models are trained only on synthetic renders and evaluated **zero-shot** on public benchmarks, each with its published protocol where one exists; every benchmark is scored by `fpeval.score.score()` (one scorer since the second review):

- **FloorPlanCAD** (Fan et al., ICCV 2021): semantic symbol spotting. Each block of the November 2021 test split (5,502 blocks) is rendered black on white at 50 px/m; each vector primitive takes the majority class of points sampled along it (Eq. 10); F1 per category over primitives weighted by log(1 + length) (§10.1), with the categories of Fig. 7. Only door, window, stair and wall are scored (curtain wall ignored). `fpcad_eval.py`.
- **CubiCasa5K** (Kalervo et al. 2019, Table 4): pixel IoU per class on the 400 official test plans, one confusion matrix pooled over the split, on the segmenter's own output. For the comparison, wall includes the door and window openings, as in CubiCasa's labels. `cubicasa_eval.py`.
- **WAFFLE** (Ganon et al. 2025, Table 4): pixel precision, recall and IoU per class, pooled over the 110 annotated plans; wall includes predicted columns. `harness.py waffle`.
- **CVC-FP** (Dodge et al. 2017): wall IoU (Jaccard index) and mean IoU and mean accuracy over wall vs. non-wall, pooled over all 122 plans. `harness.py cvcfp`.
- **Own metrics:** rooms matched one to one (Hungarian on IoU ≥ 0.5), area errors, openings and connectivity, on held-out Swiss Dwellings floors and on BBL sheets (`fpeval/metrics.py`).

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
| **Pilot v2** | Zero-shot | 0.108 | 0.306 | 0.325 | 0.387 | 0.282 | n/a | n/a |
| **Pilot v3** | Zero-shot | 0.137 | 0.339 | 0.399 | **0.408** | **0.321** | n/a | n/a |

- Published rows are on FloorPlanCAD V1 (30 classes, never released); ours are on the November 2021 release. No published number exists on this split. Later vector models reach wF1 0.80–0.93 on the August 2021 release.
- **Doors** in FloorPlanCAD are the leaf and swing arc; ours are the opening in the wall, so most door primitives vote background. Counting any foreground prediction on the primitive raises door F1 to about 0.5 for every pilot model.
- v3's gain comes from the negative set: wall precision 0.322 → 0.329 at recall 0.487 → 0.538, wall PQ 0.050 → 0.066, wall pixel IoU 0.364 → 0.394. The ablation without negatives (E2) stays at 0.304 on the mean of four.

### CubiCasa5K

Pixel IoU on the 400 test plans:

| Model | Training | Wall (incl. openings) | Door | Window |
|---|---|---|---|---|
| CubiCasa5K model (Kalervo et al. 2019) | In-domain | 0.730 | 0.536 | 0.668 |
| **Pilot v1** | Zero-shot | 0.619 | 0.495 | 0.570 |
| **Pilot v2** | Zero-shot | 0.699 | 0.451 | 0.608 |
| **Pilot v3** | Zero-shot | **0.706** | **0.551** | **0.649** |

Own metrics, v3: wall IoU without openings 0.669, columns 0.101, stairs 0.377; the share of door pixels predicted as window, v2's regression, is back from 18.6 % to 9.7 % (v1: 8.6 %); doors found 88 %, windows 74 %, opening precision 88 %; rooms matched 74 % with precision 78 % and a median area error of 2.4 % (scored with the harness rules since the second review: rooms and fragments ≥ 0.25 m² after stage 9, so not comparable with the v1/v2 room rows of the earlier card).

Ablations on the same renderer (frozen validation set in brackets): no heads E1a door 0.485, window 0.636 (0.756); heads at low weight E1b 0.533 / 0.644 (0.762); heads E2 0.548 / 0.641 (0.755); E2 + negatives = v3 0.551 / 0.649 (0.751). The heads help real doors; the negatives cost nothing here and gain on construction drawings.

### WAFFLE

Pixel precision / recall / IoU on the 110 annotated plans; all rows zero-shot:

| Model | Training | Walls | Doors | Windows |
|---|---|---|---|---|
| CubiCasa5K model (Ganon et al. 2025) | CubiCasa5K | 0.737 / 0.590 / 0.488 | 0.201 / 0.163 / 0.099 | 0.339 / 0.334 / 0.202 |
| Diffusion wall segmenter (Ganon et al. 2025) | CubiCasa5K walls | 0.746 / 0.805 / 0.632 | n/a | n/a |
| **Pilot v1** | Synthetic | 0.741 / 0.704 / 0.565 | 0.176 / 0.126 / 0.079 | 0.141 / 0.288 / 0.105 |
| **Pilot v2** | Synthetic | 0.797 / 0.730 / 0.615 | 0.308 / 0.128 / 0.100 | 0.188 / 0.508 / 0.159 |
| **Pilot v3** | Synthetic | WAFFLE_V3 |

### CVC-FP

Wall vs. non-wall on all 122 plans, in %:

| Model | Training | Mean accuracy | Mean IoU | Wall IoU (JI) |
|---|---|---|---|---|
| FCN-2s (Dodge et al. 2017) | CVC-FP, 5-fold | 97.3 | 94.4 | 89.2 |
| FCN-2s (Dodge et al. 2017) | R-FP only (cross-dataset) | 84.2 | 81.7 | 64.7 |
| **Pilot v1** | Synthetic | 91.7 | 82.5 | 66.2 |
| **Pilot v2** | Synthetic | 92.5 | 89.1 | 78.7 |
| **Pilot v3** | Synthetic | CVCFP_V3 |

Rooms on CVC-FP (own metric, one to one, current post-processing): v2 recall 0.733, precision 0.769; v3 CVCFP_ROOMS_V3.

### Own Benchmarks

| Benchmark | v2 | v3 |
|---|---|---|
| Frozen synthetic validation, renderer 3.0 (400 renders, IoU wall / door / window / column / stairs; v2 scored on its own renderer-2.0 set) | 0.86 / 0.64 / 0.72 / 0.58 / 0.92; heads interior 0.95, boundary 0.75, void 0.26, text 0.71 | 0.87 / 0.69 / 0.74 / 0.56 / 0.90; heads interior 0.97, boundary 0.76, void 0.83, text 0.68, swing 0.76 |
| Held-out Swiss Dwellings renders, full pipeline (117 test floors, same frozen renders; rooms recall / precision, wall IoU, door / window F1, connectivity recall) | 0.888 / 0.942, 0.90, 0.92 / 0.92, 0.90 | 0.886 / 0.944, 0.90, 0.93 / 0.93, 0.92 |
| Landgut Lohn S1, 2005 CAD print (BBL, local) | 15/15 rooms, mean IoU 0.924, names 15/15, connections 12/12 | LANDGUT_V3 |

- The post-processing rules were partly written while looking at the Landgut Lohn sheets, so those results are optimistic; details in the [pilot README](README.md#results).
- Post-processing alone (oracle labels on 120 held-out Swiss Dwellings floors, `fpeval/oracle.py`, after the first review): rooms recall 0.865, precision 0.959, median area error −1.6 %; every remaining miss with perfect labels is an open-plan merge.

Reproduce (any model via `--model`; `FPX_DEVICE=cuda` on a GPU), from `code/`: `fpcad_eval.py --workers 8`, `cubicasa_eval.py test --out ../data/cubicasa-v3`, `harness.py {render,cvcfp,waffle} --out ../data/harness/<mode>-v3.json`, `run_pipeline.py --out ../data/out-v3` then `evaluate.py --out ../data/out-v3`.

## Limitations

- **Construction drawings** (FloorPlanCAD): mean F1 0.32 against 0.73–0.81 in-domain. Dimension chains, axes, hatches and furniture are still partly read as walls (wall precision 0.33), and door symbols drawn as leaf and arc are not the opening our class means.
- **Columns** are weak on real plans (CubiCasa 0.10) despite 0.56 on synthetic renders.
- **Open plan:** the boundary head finds separations without a wall (CVC-FP "room (separation)" recall 0.29 → 0.43 in the room stage), but more than half of such areas still merge.
- **Annotation symbols:** grid callouts and section markers are read as walls less often than before (the negative set), not never; the interior head can mark empty paper, which the room stage guards with an enclosure test.
- **Scale** must be known or proposed: the model assumes 50 px/m.
- **Synthetic validation does not select checkpoints for real sheets** (v3's best iteration on the frozen set is not its best on CubiCasa); a small real validation set is needed.

## Licences

- **Training data:** Swiss Dwellings, CC BY 4.0 (attribution in `data/README.md`); IFC-Bench models, CC BY 3.0/4.0 and MIT (per model in `ifc_prepare.py`). Fonts: SIL OFL.
- **Code:** `segmentation-models-pytorch` (MIT), `timm` (Apache-2.0), PyTorch (BSD-3).
- **Open question:** whether ImageNet-pretrained encoder weights may be used in production. The review lists it for BBL's legal service (§6.4); a training run from scratch or from a licence-clean encoder is the fallback.
- **Benchmarks** are used for evaluation only and their outputs stay local: CubiCasa5K (CC BY-NC-SA 4.0), CVC-FP (CC BY-NC), FloorPlanCAD (CC BY-NC 4.0), WAFFLE (licences per image).
