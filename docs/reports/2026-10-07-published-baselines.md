# Published Baselines for the Model Card

*Status: notes as delivered on 7 October 2026; the "to fill" rows are filled in the [second review §4.4](../reviews/2026-10-08-pilot-v2-second-review.md#44-published-protocols) (formerly the model card, merged into the [pilot README](../../pilot/v2-pipeline/README.md#model) on 8 October). Correction: the code-level points marked "code check" (SymPoint's log(1 + length) weights, CADTransformer's micro F1 over the foreground classes, CubiCasa's `original_size=True`, rotation test-time augmentation, pooled `runningScore` and openings kept as Wall) were confirmed later the same day in local read-only clones of the official repositories (`research/code/`).*

Published results of established floor plan models on FloorPlanCAD, CubiCasa5K, WAFFLE, CVC-FP and a few other benchmarks, with each benchmark's protocol and what we must do so that our zero-shot numbers can be compared.

Conventions:

- **Figures** are quoted as printed. Where the papers print fractions, the fraction is kept.
- **Sources** are given as paper id (file stem in `research/papers-original/` and `research/papers-md/`), table number and PDF page of the local copy (p. = page in the PDF viewer).
- **Derived values** are marked *derived* and are not in any paper.
- **Code checks** were made on 7 October 2026 by reading the official repositories on GitHub (only public URLs were fetched). They are marked *code check*. The fetch tool summarises pages, so verify these points in the code before relying on them.
- **Markdown copies:** the formulas are missing from all `papers-md` copies ("formula-not-decoded"). They were read from the PDFs with PyMuPDF.

## Overview

| Benchmark | Local data | Main published metric | Training regime of published rows | Can we evaluate? |
|---|---|---|---|---|
| FloorPlanCAD | Nov-2021 release, test 5,502 blocks (`data/benchmark/floorplancad/`) | Semantic symbol spotting: F1 and length-weighted F1 per primitive; panoptic PQ/SQ/RQ | In-domain (trained on the FloorPlanCAD train split) | Yes, after rendering and primitive voting (§1.8). The test split differs from the one most papers use (§1.1) |
| CubiCasa5K | Full dataset, official 4,200/400/400 split (`data/benchmark/cubicasa5k/`) | Pixel IoU per class (12 room and 11 icon classes), mean IoU, accuracy | In-domain | Yes (already done in `cubicasa_eval.py`). Fix two label conventions first (§2.4) |
| WAFFLE segmentation benchmark | 110 annotated plans (`data/benchmark/waffle/data/benchmark/`) | Per-class precision, recall and IoU (wall, door, window, interior, background) | **Zero-shot** (all models trained on CubiCasa5K) | Yes; the closest match to our regime |
| CVC-FP | 122 scans with SVG ground truth (`data/benchmark/cvc-fp/`) | Wall Jaccard index, mean IoU and mean accuracy over wall vs. non-wall | In-domain (5-fold cross-validation), plus one cross-dataset row | Yes; compare mainly with the cross-dataset row |
| R2V, R3D, R-FP, Raster2Graph, Structured3D, LS-CAD, ArchCAD-400K, ResPlan-FP | Not local | Various (§5) | In-domain | No |

---

## 1. FloorPlanCAD

### 1.1 Versions, classes and splits

Three versions appear in the literature. Numbers from different versions are **not** on the same test set.

| Version | Classes | Drawings and split | Used by |
|---|---|---|---|
| **V1** (internal, used for the ICCV paper's results) | 30: 28 thing classes + 2 stuff classes (wall, parking). No curtain wall, railing or row chairs; no folding, revolving or rolling door | Not stated separately. The paper text describes 15,663 drawings (10,161 train / 5,502 test, 800 of the train drawings held out for validation), 10 m × 10 m blocks, split by project ("60 projects are randomly chosen for training", 2021-fan-floorplancad §4.3 p.5). Table 6 caption: "the results are reported based on V1 dataset which contains only 30 classes" (p.9) | 2021-fan-floorplancad (all tables); its numbers are re-quoted by later papers as "PanCADNet" and for the CNN baselines |
| **Aug-2021 public release** ("first version … released in August 13, 2021", 2022-fan-cadtransformer §4 p.6) | 35: 30 thing + 5 stuff (row chairs, parking spot, wall, curtain wall, railing) | 11,602 drawings: 6,965 train / 810 val / 3,827 test (2022-fan-cadtransformer §4.1 p.6; 2025-wei-vecformer §4.1 p.7, "official data split"). Block size: "14m × 14m" in CADTransformer, "10 m × 10 m" in GAT-CADNet (§4 p.5) | CADTransformer, GAT-CADNet, SymPoint, SymPoint-V2, CADSpotting, DPSS (ArchCAD-400K paper), VecFormer, VectorGraphNET; PolarSym compares against these numbers |
| **Nov-2021 release** ("FloorPlanCAD-V2" in TextCAD) | 35 (same names; the id order in the SVGs differs from the papers' tables, see `data/benchmark/floorplancad/SOURCE.md`) | 15,663 drawings: 10,161 train / 5,502 test, no validation split (local `SOURCE.md`). TextCAD uses its own 9,533 / 4,597 / 1,533 split (2026-gong-textcad App. C.1 p.26) | 2026-gong-textcad. **This is our local copy** |

Consequences for us:

- No published number is on the 5,502-block test split we have.
- **Table 3 numbers** (the table used as the model card template) are on V1, a 30-class version that was never released. They can be matched in protocol, not in data.
- **SymPoint family:** SymPoint's `download_data.py` (GitHub) fetches separate train, validation and test archives, different files from the Nov-2021 archives in our `SOURCE.md`. If those archives are still downloadable, the 3,827-drawing test split would make our numbers comparable with the SymPoint family. This was not checked.

### 1.2 Protocol: semantic symbol spotting

- **Unit.** A drawing is a set of graphical entities (primitives: line segments, arcs, circles, curves). Each has a ground-truth semantic label, or background, and an instance index (2021-fan-floorplancad §3 p.4). The task maps every entity to a label: F_s : e_k ↦ l_k.
- **Raster models.**
  - Training masks: FloorPlanCAD trains its CNN baselines on masks made "by projecting graphical entities onto the background canvas with line width of 5 pixels" (§6.1 p.7).
  - Pixel to entity: pixel predictions are converted to entity labels "by sampling and voting": PD(e_i) = argmax_l |{p_k | PD(p_k) = l, p_k ∈ e_i}| (Eq. 10, p.7). The class that most sample points on the entity receive wins.
  - GAT-CADNet renders "with line width of 2 pixels" and retrieves labels "by sampling on the predicted mask with a majority voting strategy" (§4.1 p.6).
  - Resolution and sampling density: no paper states them. DPSS's image branch uses 700 × 700 images (2025-luo-archcad-400k App. p.17). The released test PNGs are 1000 × 1000 px (local `SOURCE.md`).
- **Per-class score.** For class c, count over entities: TP_c (ground truth c, predicted c), FP_c (predicted c, ground truth not c) and FN_c (ground truth c, predicted not c). Then F1_c = 2TP/(2TP+FP+FN).
- **Weighted F1 (wF1).** The same, but each entity is weighted by log(1 + L(e)), with L the arc length: "we use weighted F1 score as the metric which use the entity length log(1 + L(e_i)) to weight the TP, FP and FN" (§10.1 p.9; "weights are similarly defined as Equation 1", §6.1 p.7).
  - The unit of L is not stated.
  - *Code check:* SymPoint's released evaluator applies `np.log(1 + lengths)` to the per-primitive lengths from its SVG parser, i.e. in SVG units.
- **Overall F1 and wF1.** The papers do not define how classes are combined.
  - *Code check:* CADTransformer's `eval.py` counts primitives (unweighted) per class, with background as class 0, and then computes `tp = sum(cnt_tp[1:])`, `gt = sum(cnt_gt[1:])`, `pred = sum(cnt_prd[1:])`, precision = tp/pred, recall = tp/gt and F1 from these.
  - This is a micro F1 over the foreground classes. Background entities predicted as foreground count as false positives; foreground entities predicted as background count as false negatives.
  - So the overall F1 is dominated by walls, which have the most entities: 1,880 × 10⁴ of 7,600 × 10⁴ in V1 (Table 6).
  - SymPoint's released code computes no F1, only mIoU over primitives with ground-truth background ignored (`pos_inds = gt_sem != ignore_label`).
- **Category columns of Table 3** (Door, Window, Stair, Appliance, Furniture, Equipment, Wall, Parking lot). The grouping is not given in the text. It follows from the x-axis of Figure 7 (p.5), which groups the 28 V1 thing classes:

  | Category | V1 classes |
  |---|---|
  | Door | single door, double door, sliding door |
  | Window | window, bay window, blind window, opening symbol |
  | Stair | stair(s) |
  | Appliance ("home appliance") | gas stove, refrigerator, washing machine |
  | Furniture | sofa, bed, chair, table, bedside cupboard, TV cabinet, half-height cabinet, high cabinet, wardrobe, sink, bath, bath tub, squat toilet, urinal, toilet |
  | Equipment | elevator, escalator |
  | Wall, Parking lot | wall, parking (stuff classes) |

  - **Check (derived).** Entity-count-weighted means of the per-class length-weighted F1 in Table 6 reproduce Table 3 within 0.02, for example:
    - GCN Door 0.847 vs 0.848 and Equipment 0.925 vs 0.926;
    - DeepLabv3+ Door 0.837 vs 0.837;
    - Wall and Stair identical for the GCN.

    So the category columns are length-weighted F1 under this grouping.
  - Whether a category score merges the labels (confusions inside a category forgiven) or averages the class scores is not stated. Our model predicts the categories directly, so we compute F1 on merged labels.
  - The 35-class releases keep the same super-classes ("doors, windows, stairs, home appliances, furniture, and equipment", 2022-fan-cadtransformer §4.1 p.6). They add folding, revolving and rolling doors (Door) and cabinet and air conditioner, and they turn row chairs, parking spot, wall, curtain wall and railing into stuff.
- **Wall and curtain wall** are separate classes in every 35-class paper (per-class tables of SymPoint, SymPoint-V2 and CADSpotting). No paper merges them. V1 has no curtain wall class.

### 1.3 FloorPlanCAD Table 3 (verified)

2021-fan-floorplancad **Table 3** (p.8), V1. Caption: "Statistical results on the proposed dataset of different semantic segmentation models and our GCN-based method. Both HRNetsV2 [42] and DeepLabv3+ [5] show that deeper networks produce better results."

The values match the image. Raster rows were scored by entity voting (Eq. 10). "Ours" is the GCN-based semantic spotter (HRNetV2 backbone + GCN head). Later papers call it PanCADNet.

| Categories | Door | Window | Stair | Appliance | Furniture | Equipment | Wall | Parking lot | F1 | Weighted F1 | *Mean of Door, Window, Stair, Wall (derived)* |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HRNetsV2 W18 | 0.821 | 0.620 | 0.845 | 0.597 | 0.726 | 0.880 | 0.620 | 0.610 | 0.656 | 0.683 | *0.7265* |
| HRNetsV2 W48 | 0.811 | 0.640 | 0.847 | 0.651 | 0.754 | 0.889 | 0.624 | 0.577 | 0.666 | 0.693 | *0.7305* |
| DeepLabv3+ R50 | 0.828 | 0.659 | 0.856 | 0.684 | 0.763 | 0.895 | 0.630 | 0.664 | 0.680 | 0.705 | *0.7433* |
| DeepLabv3+ R101 | 0.837 | 0.666 | 0.852 | 0.725 | 0.780 | 0.895 | 0.634 | 0.669 | 0.688 | 0.714 | *0.7473* |
| Ours (GCN, "PanCADNet") | 0.848 | 0.709 | 0.857 | 0.769 | 0.764 | 0.926 | 0.814 | 0.539 | 0.806 | 0.798 | *0.8070* |
| **Our segmenter (zero-shot)** | to fill | to fill | to fill | n/a | n/a | n/a | to fill | n/a | n/a | n/a | to fill |

Notes:

- The text confirms the definitions of the last two columns: "improvement of 11.8% on F1 score and 8.4% weighted F1 score, whose weights are similarly defined as Equation 1" (p.7). 0.806 − 0.688 = 0.118 and 0.798 − 0.714 = 0.084.
- The F1 and Weighted F1 columns cover all 30 classes. They are not comparable with a four-category model, so leave them empty for our row and use the derived mean of the four categories we predict.
- Small inconsistencies between Table 3 and Table 6, where a category is a single class:
  - GCN parking 0.539 vs 0.529;
  - DeepLabv3+ stairs 0.852 vs 0.853 and parking 0.669 vs 0.667.

Per-class V1 values for our classes (2021-fan-floorplancad **Table 6**, p.9; length-weighted F1; the "DeepLabv3+" column equals R101 in Table 3):

| Class | GCN-based | DeepLabv3+ | PanCADNet PQ / SQ / RQ |
|---|---|---|---|
| single door | 0.885 | 0.827 | 0.763 / 0.878 / 0.869 |
| double door | 0.796 | 0.831 | 0.748 / 0.845 / 0.885 |
| sliding door | 0.874 | 0.876 | 0.763 / 0.895 / 0.852 |
| window | 0.691 | 0.603 | 0.459 / 0.795 / 0.577 |
| bay window | 0.050 | 0.163 | 0.154 / 0.595 / 0.260 |
| blind window | 0.833 | 0.856 | 0.706 / 0.869 / 0.813 |
| opening symbol | 0.451 | 0.721 | 0.455 / 0.945 / 0.481 |
| stairs | 0.857 | 0.853 | 0.608 / 0.784 / 0.775 |
| elevator | 0.948 | 0.900 | 0.838 / 0.897 / 0.935 |
| escalator | 0.744 | 0.864 | 0.439 / 0.718 / 0.612 |
| wall | 0.814 | 0.634 | 0.451 / 0.661 / 0.682 |
| total | 0.798 | 0.714 | 0.561 / 0.838 / 0.660 |

The V1 panoptic total is in Table 5 (p.8): PQ 0.561, SQ 0.838, RQ 0.660. This PQ is not SQ × RQ (0.838 × 0.660 = 0.553). GAT-CADNet (Table 3, p.7) and SymPoint (Table 3, p.8) quote PanCADNet as 0.553 / 55.3.

### 1.4 Semantic symbol spotting on the 35-class releases (F1 / wF1, %)

The prior column tells whether layer or colour metadata of the CAD file was used:

- **w/** = layer or colour metadata of the CAD file used as input;
- **w/o** = geometry only;
- **n/s** = not stated.

Unless marked otherwise, all rows are on the Aug-2021 release (test 3,827).

| Method | Prior | F1 | wF1 | Source (table, page) | Note |
|---|---|---|---|---|---|
| HRNetV2-W18 / W48 (pixel + vote) | w/o | 65.6 / 66.6 | 68.3 / 69.3 | 2022-zheng-gat-cadnet T1 p.6 (printed as fractions, 0.656 etc.) | Identical to FloorPlanCAD V1 Table 3, so probably copied from it, not recomputed |
| DeepLabv3+ R50 / R101 (pixel + vote) | w/o | 68.0 / 68.8 | 70.5 / 71.4 | GAT-CADNet T1 p.6; R101 also 2025-luo-archcad-400k T3 p.8 | As above (V1 values) |
| PanCADNet (GCN) | w/o | 80.6 | 79.8 | 2024-liu-sympoint T1 p.7; GAT-CADNet T1; SymPoint-V2 T1 p.10; CADSpotting T2 p.5 | V1 value carried forward |
| CADTransformer | w/o | 82.2 | 80.1 | SymPoint T1 p.7 | Not in the CADTransformer paper. 2024-yang-cadspotting T2 prints wF1 "90.1", probably a typo |
| GAT-CADNet | w/o | 85.0 | 82.3 | GAT-CADNet T1 p.6 | |
| PointT‡ (Point Transformer, double channels) | w/o | 83.2 | 80.7 | SymPoint T1 p.7 | |
| SymPoint | w/o | 86.8 | 85.5 | SymPoint T1 p.7 | |
| SymPoint-V2 | w/ (layers) | 89.5 | 88.3 | 2024-liu-sympoint-v2 T1 p.10 | |
| SymPoint-V2 | w/o | 87.0 | 86.3 | ArchCAD-400K T3 p.8 | Re-run by the DPSS authors |
| CADSpotting | w/o (coordinates only) | 92.8 | 93.2 | CADSpotting T2 p.5 | |
| CADSpotting | w/ (colour) | 93.5 | 93.9 | ArchCAD-400K T3 p.8 | |
| DPSS | w/ | 93.1 | 93.2 | ArchCAD-400K T3 p.8 | |
| DPSS | w/o | 92.0 | 93.2 | ArchCAD-400K T3 p.8 | |
| VecFormer | n/s | 93.8 | 92.2 | 2025-wei-vecformer T1b p.8 | |
| VectorGraphNET | w/o | 79.4 | 89.0 | 2024-carrara-vectorgraphnet T3 p.19 | Different metric: macro-average F1 and support-weighted F1 incl. an "other" class (classification report, T5 p.22). The caption wrongly says "TUM Dataset" |

TextCAD re-implemented ten baselines on its own split of the **Nov-2021** release (test 1,533, mean of three runs; 2026-gong-textcad **Table 1**, p.15). Values are PQ / PQ-Thing / PQ-Stuff / F1 / wF1.

| Method | w/o prior | w/ prior |
|---|---|---|
| PanCADNet | 57.60 / 65.78 / 53.34 / 78.7 / 78.0 | n/a |
| GAT-CADNet | 71.31 / 73.29 / 58.08 / 84.4 / 81.4 | n/a |
| CADTransformer | 71.75 / 73.65 / 58.39 / 80.4 / 78.5 | n/a |
| SymPoint | 83.27 / 86.68 / 57.26 / 85.7 / 84.7 | n/a |
| SymPointV2 | 82.86 / 86.24 / 57.07 / 86.1 / 85.7 | 89.34 / 90.55 / 81.01 / 89.3 / 88.7 |
| DPSS | 84.46 / 87.56 / 61.45 / 91.5 / 91.0 | 89.39 / 90.54 / 81.31 / 92.2 / 91.5 |
| VecFormer | 88.14 / 87.26 / 89.17 / 87.8 / 89.8 | 90.76 / 90.64 / 90.90 / 90.0 / 91.3 |
| PFL-Net | 73.39 / 75.36 / 59.72 / 80.6 / 78.9 | n/a |
| TNet | 73.85 / 75.93 / 59.44 / 81.0 / 78.9 | n/a |
| TriNet | 83.98 / 87.25 / 59.23 / 91.2 / 90.2 | n/a |
| TextCAD | 91.09 / 91.22 / 90.97 / 91.7 / 91.4 | 92.67 / 93.05 / 92.32 / 93.4 / 91.9 |

### 1.5 Panoptic symbol spotting (PQ, SQ, RQ in %)

- **Matching.** A predicted symbol matches a ground-truth symbol if the labels agree and IoU > 0.5. Symbol IoU is Σ log(1 + L(e)) over the shared entities divided by the same sum over the union (Eq. 1, p.4).
- **Formulas.** RQ = |TP| / (|TP| + ½|FP| + ½|FN|); SQ = mean IoU of the matched pairs; PQ = RQ × SQ (Eqs. 2–4, p.4). A stuff symbol is all entities of that stuff class in a drawing ("we ignore the instance index of entities belonging to stuff", §3 p.4).
- *Code check:* SymPoint's `InstanceEval` pools TP, FP, FN and the IoU sums over all classes before dividing (`sRQ = sum(tp)/(sum(tp)+0.5*sum(fp)+0.5*sum(fn))`). It does not average per-class PQ. Thing classes are 0–29 and stuff classes 30–34; predictions scoring under 0.1 are dropped.

Rows are on the Aug-2021 release.

| Method | Prior | PQ | SQ | RQ | PQ_th | SQ_th | RQ_th | PQ_st | SQ_st | RQ_st | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PanCADNet (V1, recomputed) | w/o | 55.3 | 83.8 | 66.0 | | | | | | | SymPoint T3 p.8 (= FloorPlanCAD T5 with PQ = SQ × RQ) |
| PanCADNet (re-implemented by CADTransformer) | w/o | 0.5953 | 0.8258 | 0.6693 | 0.6557 | 0.8614 | 0.7611 | 0.5872 | 0.813 | 0.7222 | 2022-fan-cadtransformer T1 p.7 (mAP 0.5630) |
| CADTransformer | w/o | 0.6732 | 0.8754 | 0.7226 | 0.7713 | 0.9355 | 0.8245 | 0.5793 | 0.8102 | 0.7151 | CADTransformer T1 p.7 |
| CADTransformer + Random Layer | w/o at test time (layers used only for training augmentation) | 0.6894 | 0.8832 | 0.7333 | 0.7849 | 0.9404 | 0.8346 | 0.5855 | 0.8188 | 0.7151 | CADTransformer T1 p.7; quoted as 68.9 / 88.3 / 73.3 later |
| GAT-CADNet | w/o | 0.737 | 0.914 | 0.807 | | | | | | | 2022-zheng-gat-cadnet T3 p.7 |
| PointT‡ Cluster | w/o | 49.8 | 85.6 | 58.2 | | | | | | | SymPoint T3 p.8 |
| SymPoint (300 / 500 epochs) | w/o | 79.6 / 81.9 | 89.4 / 90.6 | 89.0 / 90.4 | | | | | | | SymPoint T3 p.8 |
| SymPoint (1000 epochs) | w/o | 83.3 | 91.4 | 91.1 | 84.1 | 94.7 | 88.8 | 48.2 | 69.5 | 69.4 | SymPoint T3 p.8; th/st from SymPoint-V2 T3 p.11 |
| SymPoint-V2 | w/ (layers) | 90.1 | 96.3 | 93.6 | 90.8 | 96.6 | 94.0 | 80.8 | 90.9 | 88.9 | SymPoint-V2 T3 p.11 (mIoU 74.0) |
| SymPoint-V2 | w/o | 83.2 | 91.3 | 91.1 | 85.8 | 92.5 | 92.7 | 49.3 | 70.3 | 70.1 | ArchCAD-400K T2 p.8 |
| CADSpotting | w/o (coordinates only) | 87.4 | 93.5 | 93.4 | 88.3 | 94.2 | 93.7 | 71.5 | 82.9 | 86.3 | 2024-yang-cadspotting T4 p.6 |
| CADSpotting | w/ (colour) | 88.9 | 95.6 | 93.0 | 89.7 | 96.2 | 93.2 | 80.6 | 89.7 | 89.8 | ArchCAD-400K T2 p.8. CADSpotting's own T10 (p.11) prints these under "w/o color" but its text says colour adds 1.5 PQ, so the labels are swapped |
| DPSS | w/ | 89.5 | 96.2 | 93.1 | 90.4 | 96.6 | 93.5 | 79.7 | 91.1 | 87.5 | ArchCAD-400K T2 p.8 |
| DPSS | w/o | 86.2 | 93.0 | 92.6 | 88.0 | 94.1 | 93.5 | 64.7 | 83.0 | 77.9 | ArchCAD-400K T2 p.8 |
| VecFormer | w/o | 88.4 | | | 90.9 | | | 85.9 | | | 2025-wei-vecformer T1a p.8 |
| VecFormer | w/ | 91.1 | | | 91.8 | | | 90.4 | | | VecFormer T1a p.8 |
| PolarSym (50 epochs, 2 GPUs) | w/ (layer IDs via LFEM) | 87.15 | 95.28 | 91.48 | | | | | | | 2026-chen-polarsym T1 p.12 (mIoU 70.87) |
| SymPoint-V2 re-run on the same budget | w/ | 85.42 | 94.98 | 89.92 | | | | | | | PolarSym T2 p.12 (mIoU 66.56) |

Notes:

- **CADTransformer's totals are not SQ × RQ:**
  - 0.8754 × 0.7226 = 0.633 ≠ 0.6732;
  - 0.8258 × 0.6693 = 0.553 ≠ 0.5953;
  - its thing and stuff values are SQ × RQ.

  Later papers copy the totals unchanged. All other rows satisfy PQ ≈ SQ × RQ.
- **PolarSym's Table 1** swaps the SQ and RQ columns for its baseline rows (e.g. PanCADNet "PQ 59.5, RQ 82.6, SQ 66.9"). The values above are in the correct columns.
- **The stuff gap** between w/ and w/o prior (SymPoint-V2 PQ_st 80.8 vs 49.3) shows how much stuff classes such as wall depend on CAD layer names. Our raster model has no layers, so the **w/o** rows are the relevant comparison.

Per-class PQ / RQ / SQ (%) for the classes we predict. CADSpotting prints its columns as PQ, SQ, RQ; they are reordered here.

| Class | SymPoint final (T5 col. E, p.14; w/o) | SymPoint-V2 (App. T1, p.18; w/ layers) | CADSpotting (T11 col. A, p.11; w/o) |
|---|---|---|---|
| single door | 91.7 / 96.0 / 95.5 | 94.4 / 97.1 / 97.3 | 91.90 / 95.40 / 96.33 |
| double door | 91.5 / 96.6 / 94.7 | 94.5 / 97.3 / 97.1 | 94.10 / 97.55 / 96.46 |
| sliding door | 94.8 / 97.7 / 97.0 | 97.2 / 97.9 / 99.3 | 96.50 / 98.46 / 98.02 |
| folding door | 73.8 / 87.0 / 84.8 | 82.7 / 90.0 / 91.9 | 77.06 / 86.36 / 89.23 |
| revolving, rolling door | 0.0 (no test instances) | 0.0 | 0.00 |
| window | 78.9 / 90.4 / 87.3 | 90.1 / 93.1 / 96.8 | 85.41 / 93.80 / 91.06 |
| bay window | 35.4 / 42.3 / 83.6 | 54.1 / 55.1 / 98.3 | 51.18 / 56.74 / 90.19 |
| blind window | 80.6 / 92.1 / 87.5 | 91.0 / 92.3 / 98.5 | 86.00 / 93.22 / 92.25 |
| opening symbol | 33.1 / 40.9 / 80.7 | 40.7 / 51.6 / 78.9 | 52.67 / 68.35 / 77.06 |
| stairs | 72.5 / 85.3 / 85.0 | 84.8 / 89.8 / 94.4 | 84.41 / 91.38 / 92.37 |
| escalator | 60.6 / 75.6 / 80.2 | 68.7 / 80.7 / 85.2 | 74.96 / 85.55 / 87.62 |
| wall (stuff) | 53.5 / 77.5 / 69.0 | 83.7 / 92.5 / 90.6 | 73.21 / 89.91 / 81.42 |
| curtain wall (stuff) | 44.2 / 60.2 / 73.5 | 60.0 / 70.1 / 85.6 | 58.66 / 71.01 / 82.61 |
| railing (stuff) | 53.0 / 66.3 / 80.0 | 70.7 / 77.0 / 91.8 | 67.25 / 76.10 / 88.38 |
| total | 83.3 / 91.1 / 91.4 | 90.1 / 93.6 / 96.3 | 87.36 / 93.54 / 93.40 |

### 1.6 Instance symbol spotting (box AP, for completeness)

Values are AP50 / AP75 / mAP.

- **V1** (FloorPlanCAD T4, p.8): Faster R-CNN R101 0.602 / 0.510 / 0.452; FCOS R101 0.624 / 0.491 / 0.453; YOLOv3 0.639 / 0.452 / 0.413; SCIP 0.231 / 0.151 / 0.135; Graph Matching 0.137 / 0.118 / 0.102.
- **GAT-CADNet** (T2, p.7): Faster R-CNN 0.693 / 0.631 / 0.568; YOLOv3 0.656 / 0.431 / 0.395; FCOS 0.648 / 0.572 / 0.525; GAT-CADNet 0.735 / 0.680 / 0.690.
- **SymPoint** (T2, p.7): repeats the V1 detector rows and adds DINO R50 64.0 / 54.9 / 47.5 and SymPoint 66.3 / 55.7 / 52.8.
- **Later methods:**
  - SymPoint-V2 71.3 / 60.7 / 60.1 (T2, p.10);
  - CADSpotting 70.6 / 66.6 / 66.3 (T3, p.5);
  - DPSS w/ 74.8 / 71.0 / 70.8 and w/o 67.0 / 61.8 / 61.5 (ArchCAD-400K T3, p.8);
  - SymPoint-V2 w/o 66.4 / 57.7 / 57.5 (ArchCAD-400K T3, p.8).

Our segmenter has no instance output, so this is not applicable.

### 1.7 Comparability

- **Data:** see §1.1.
- **Representation:** all post-2021 models read the vector primitives directly. Only the V1 CNN rows (HRNetV2, DeepLabv3+) are raster models scored by voting, which is what we would do. Those are the like-for-like rows, and they were trained in-domain.
- **Classes:** our model predicts 4 of the 8 Table 3 categories (plus column, which FloorPlanCAD does not label: "Columns are not labelled", local `SOURCE.md`). Overall F1 and wF1 therefore cannot be compared; use the per-category columns and the derived four-category mean.
- **Prior:** we use no layers or colours. Compare with the **w/o** rows.
- **Regime:** every published row is in-domain. Ours is zero-shot, so the gap measures domain shift plus model capacity, not a ranking of methods.

### 1.8 How to make our FloorPlanCAD numbers comparable

1. **Data.** Use the Nov-2021 test split (5,502 blocks) and label it so. If the Aug-2021 test archive (3,827) can still be obtained, report on it as well.
2. **Primitives.** One entity per `<path>`, `<circle>` and `<ellipse>` element. Label = `semantic-id` mapped by class name (the local id order differs from the papers'). Unlabelled = background. Text is background.
3. **Category map** (Table 3 / Figure 7 grouping):
   - Door ← single, double, sliding, folding, revolving and rolling door.
   - Window ← window, bay window, blind window, opening symbol.
   - Stair ← stairs only (escalator and elevator belong to Equipment).
   - Wall ← wall only. Curtain wall did not exist in V1, so **ignore** curtain-wall primitives in the primary score and report a variant with curtain wall → window. This differs from the local `sample/` mapping, which puts curtain wall into window and escalator into stairs.
   - Everything else, including our column class, is "other". It counts as a negative: a wall prediction on an unlabelled column primitive is a wall FP, as for the published models.
4. **Render.** Each block black on white at our input scale of 50 px/m (10 m block = 500 px; 10 SVG units per metre per `SOURCE.md`). Use original line widths with a 1 px minimum and keep all layers including text. Run the segmenter as in the pipeline (tiles and flips). The papers do not fix resolution or line width; state ours.
5. **Vote** (FloorPlanCAD Eq. 10):
   - sample each primitive every ≤ 0.5 px (at least start, mid and end point);
   - read the predicted class at each sample;
   - assign the majority class and break ties towards background.
   - Our walls are filled regions; wall lines lie on their boundary. Also run a variant that reads the most probable class in a 3 × 3 neighbourhood, so that a one-pixel offset does not turn wall lines into background.
6. **Doors.** Our door class is trained on the Swiss Dwellings door polygon (`synth.py` fills `g["doors"]`). FloorPlanCAD door primitives include the leaf and the swing arc outside the wall. Those will mostly vote background, so expect low door recall. Check whether the SD door polygon covers the swing, and report a variant that votes inside the convex hull of each predicted door component expanded by the door width.
7. **Score.** For each of Door, Window, Stair and Wall, compute F1 with entity weights log(1 + L), L in SVG units (as in SymPoint's code); this is the number for the table. Also report unweighted F1. Fill the four-category mean for our row. Leave the overall F1 and wF1 cells empty, or give a separate four-class micro F1 computed as in CADTransformer's `eval.py` and label it clearly.
8. **Optional panoptic numbers.**
   - **Wall PQ (stuff):** one symbol per block = all primitives voted wall. Match against all ground-truth wall primitives with the log-length IoU > 0.5, pooled over blocks as in SymPoint's evaluator. This is directly computable from semantic output and comparable with the per-class wall PQ above (SymPoint 53.5 w/o, CADSpotting 73.21 w/o).
   - **Thing PQ** needs instances. Use connected components of the predicted class mask as instances, with the caveat that adjacent doors merge (model card limitation).

---

## 2. CubiCasa5K

### 2.1 Protocol (2019-kalervo-cubicasa5k)

- **Data.** 5,000 plans (3,732 high quality architectural, 992 high quality, 276 colourful), "randomly splitted into training, validation and test sets so that there are 4200, 400, and 400" (§3, p.4). Annotations are SVG polygons.
- **Classes.** "Some original room types and icon types are coupled so that our targets cover altogether 12 room and 11 icon classes" (§5, p.10):
  - Rooms: Background, Outdoor, Wall, Kitchen, Living Room, Bedroom, Bath, Hallway, Railing, Storage, Garage, Other rooms.
  - Icons: Empty, Window, Door, Closet, Electr. Appl., Toilet, Sink, Sauna bench, Fire Place, Bathtub, Chimney (Table 4 header, p.12).
- **Prediction.** Two segmentation maps (rooms with walls, and icons with openings) plus heat maps; a post-processor polygonises them.
- **Metrics.** "Overall accuracy …, mean accuracy … averaged over all the classes … mean intersection over union (IoU) … averaged over all the classes", for raw segmentation and for polygonised output ("P"), on validation and test (§5, p.11). Per-class IoU and accuracy are in Table 4.
- **Evaluation resolution.** Scaled or original images? **The paper does not say.**
  - *Code check (official repo):* `create_lmdb.py` builds the dataset with `FloorplanSVG(..., original_size=True)`. With this flag the loader reads `F1_original.png` and nearest-interpolates the label (drawn in the SVG / `F1_scaled.png` frame) to the original image size. `eval.py` reads that LMDB without resizing, accumulates one confusion matrix over the whole split (`runningScore.update`, pooled, not averaged per image) and calls `get_evaluation_tensors(..., rotate=True)` (rotation test-time augmentation).
  - So if the published tables come from the released code, they were computed at the **original** image resolution with pooled confusion matrices.
- **Label conventions** (*code check*, `house.py`):
  - windows and doors are drawn into the icon map only; the wall map is not changed, so an opening pixel is still **Wall** in the room map;
  - the door polygon is the opening rectangle in the wall, not the swing.

### 2.2 Published numbers (in-domain: trained on the CubiCasa5K train split)

2019-kalervo-cubicasa5k **Table 3** (p.11), values for val / test:

| | Overall Acc | Mean Acc | Mean IoU |
|---|---|---|---|
| Rooms | 84.5 / 82.7 | 72.3 / 69.8 | 61.0 / 57.5 |
| Rooms P | 79.0 / 77.3 | 64.2 / 61.6 | 52.4 / 49.3 |
| Icons | 97.8 / 97.6 | 62.8 / 61.5 | 56.5 / 55.7 |
| Icons P | 97.0 / 96.7 | 94.8 / 45.3 | 43.7 / 41.6 |

The Icons P mean accuracy (val 94.8, test 45.3) is printed so and is presumably a typo.

2019-kalervo-cubicasa5k **Table 4** (p.12), test split, per class; IoU / IoU after polygonisation / Acc / Acc after polygonisation (val IoU in brackets for the main classes):

| Class | IoU | IoU P | Acc | Acc P |
|---|---|---|---|---|
| Background | 87.3 (88.3) | 79.2 | 93.6 | 92.9 |
| Outdoor | 64.4 | 48.5 | 77.7 | 60.6 |
| **Wall** | **73.0** (74.0) | 47.9 | 85.8 | 52.8 |
| Kitchen | 65.0 | 58.3 | 79.9 | 74.2 |
| Living Room | 66.6 | 62.2 | 82.6 | 81.3 |
| Bedroom | 74.2 | 68.7 | 86.2 | 83.6 |
| Bath | 60.6 | 56.6 | 73.4 | 69.9 |
| Hallway | 55.6 | 53.5 | 71.2 | 68.3 |
| Railing | 23.6 (29.0) | 5.8 | 28.7 | 6.1 |
| Storage | 44.8 | 40.8 | 53.9 | 49.4 |
| Garage | 33.7 | 30.0 | 47.2 | 43.8 |
| Other rooms | 41.4 | 39.8 | 57.1 | 55.8 |
| Empty (icon background) | 97.6 | 96.7 | 99.3 | 99.4 |
| **Window** | **66.8** (67.3) | 40.9 | 73.7 | 44.8 |
| **Door** | **53.6** (56.7) | 41.2 | 59.8 | 47.4 |
| Closet | 69.2 | 63.3 | 77.6 | 69.1 |
| Electr. Appl. | 66.0 | 59.8 | 75.7 | 67.6 |
| Toilet | 62.8 | 56.2 | 68.4 | 60.8 |
| Sink | 55.7 | 45.5 | 66.1 | 52.9 |
| Sauna bench | 67.3 | 48.0 | 74.2 | 49.7 |
| Fire Place | 36.2 | 3.1 | 40.4 | 3.1 |
| Bathtub | 26.7 | 0.0 | 30.1 | 0.0 |
| Chimney | 11.2 | 2.9 | 11.7 | 2.9 |

The rows were read from the PDF layout (the markdown copy of Table 4 is garbled). The 0.73 / 0.54 / 0.67 in our model card are these test IoUs for wall, door and window.

Other papers on CubiCasa5K:

| Paper | Split, input | Metric | Numbers | Comparable? |
|---|---|---|---|---|
| 2023-huang-muranet T1 p.11, T2 p.12 | Own split 4,000 / 500 / 500; images resized to 1536 × 1536 (aspect ratio changed); trained from scratch | Wall IoU (single class); door and window box AP | Wall IoU: U-Net base 65.5, U-Net 5-stage 74.4, U-Net 6-stage 75.8, MuraNet base 78.4, MuraNet 5-stage 75.7, MuraNet 6-stage 76.4. AP50 doors / windows: YOLOv3 base 89.2 / 90.1, MuraNet base 91.2 / 92.2; AP@[.5:.95] 43.6 / 55.4 and 47.9 / 59.7 | Partly: other split; whether walls include openings is not stated; figures are labelled "validation" while the tables say "tested" |
| 2026-phung-raster2seq T1 p.5, T10 p.16 | "4,199/399/399" plans, split into single floors: 5,267 / 503 / 511; 256 × 256 input | Room / Corner / Angle F1 (room matched at IoU > 0.5; corner within 10 px and 5°), Room semantic F1, Window & Door F1 | HEAT 78.2 / 53.7 / 32.3; PolyRoom 54.1 / 37.1 / 23.0; FRI-Net 77.1 / 50.8 / 38.0; RoomFormer 83.5 / 55.5 / 34.1, room semantic 63.0, W&D 78.5; Raster2Seq 88.7 / 59.4 / 37.4, 63.8, 77.8 | Room F1 only, with the caveats that it is computed at 256 px on per-floor crops and that rooms are labelled polygons |
| 2026-zhang-readout-vectorization T2 p.6, T4 p.9 | Corrected "skv4" annotations, official ids 3,328 / 318 / 296 (plans that convert); 256 px input; own "fpeval" | Wall F1 at tolerance 0.05 of a 1,024-unit frame, room F1, opening F1, edit cost | Raster2Seq + reconciliation (cc5k) wall F1 0.769, rooms 0.777, openings 0.840; mix 0.781 / 0.784 / 0.848; wall-first decoder 0.747–0.751; graph readout 0.818 | Single-paper protocol; would need their converter |
| 2023-yang-vectorfloorseg T2 p.6 | Walls-only vector input, room-type labels; 4,192 / 399 / 400; 256 px | Room mIoU / mAcc / RI (test) | DFPR 47.73 / 58.68 / 38.57; DeepLabV3+ R50 58.18 / 71.75 / 35.16; OCRNet R101 57.13 / 70.62 / 41.89; VectorFloorSeg (R101) 62.49 / 75.48 / 67.51 | No (different input and task) |
| 2026-gong-textcad T1 p.15 | SVG primitives as panoptic symbol spotting (10 thing + 2 stuff classes); 4,200 / 400 / 400 | PQ / PQ-Thing / PQ-Stuff / F1 / wF1 | e.g. SymPoint 89.16 / 90.58 / 50.47 / 93.6 / 88.7; VecFormer 94.58 / 95.59 / 86.95 / 96.2 / 96.4; TextCAD 96.53 / 97.23 / 91.48 / 98.1 / 98.3 (all ten baselines in the paper) | No (vector input; the CubiCasa SVG is the annotation, not a drawing) |
| Pilot v2 (`pilot/v2-pipeline/README.md`, not a paper) | Official test, 400 plans, **zero-shot**; `F1_scaled.png` resampled to 50 px/m with 1 m padding | Pixel IoU, pooled confusion matrix | Wall 0.57, door 0.49, window 0.57, column 0.09, stairs 0.38 | Our current numbers; see §2.4 |

### 2.3 In-domain vs. zero-shot

Every published CubiCasa5K number above is in-domain: trained on CubiCasa5K, including MuraNet's own split. None is zero-shot. The only zero-shot evaluations of CubiCasa-trained models are on WAFFLE (§3) and Raster2Seq's cross-dataset heat maps (Fig. 6, values only in the figure).

### 2.4 How to make our CubiCasa5K numbers comparable to Table 4

1. **Split:** official `test.txt` (400 plans). Already done.
2. **Pooled confusion matrix** over the split: already done (`cubicasa_eval.summary`). Also the official convention.
3. **Wall definition.**
   - **Problem:** CubiCasa's Wall keeps opening pixels (doors and windows are only in the icon map). The pilot's `cubicasa.rasterise` paints door and window over wall ("stairs < wall < column < window < door"), so our wall ground truth excludes openings. Our wall IoU is then computed against a smaller target than CubiCasa's 73.0.
   - **Fix:** for the comparison, compute wall IoU on wall ∪ door ∪ window in both ground truth and prediction. In the ground truth, the wall polygon already spans the openings. Keep the current single-label numbers as our own metric.
4. **Door and window definitions.** CubiCasa doors and windows are the opening rectangles in the wall. Our classes are trained the same way if the Swiss Dwellings polygons are openings; check `synth.py`. No change otherwise.
5. **Resolution.** The official code evaluates at original image size. We evaluate on `F1_scaled.png` (about 100 px/m per `cubicasa_eval.py`) resampled to 50 px/m.
   - For the table, upsample our class map (nearest) to the `F1_scaled.png` frame and score there, without the 1 m padding.
   - Original-size scoring would also need the scaled → original mapping. The effect is expected to be small, but state the frame used.
6. **Test-time augmentation.** The official evaluation uses rotations (`rotate=True`); ours averages four flips. Report ours with its test-time augmentation and say so.
7. **Classes.** Report wall, door and window only next to Table 4.
   - CubiCasa has no column evaluation class. Its Railing is a room-map class we do not predict; leave it out, or map railing pixels to "ignore".
   - Stairs are not an evaluated CubiCasa class (stairs are drawn but not in the 12 / 11 classes), so our stairs IoU has no published counterpart.

---

## 3. WAFFLE segmentation benchmark

### 3.1 Protocol (2024-ganon-waffle §4.3 p.6, App. C.4 p.15)

- **Data.** "We created a benchmark of 110 SVG images, containing wall, windows and door annotations", taken from the WAFFLE test set plus SVGs removed during dataset filtering. They were annotated in Inkscape by whole SVG components.
- **Classes.** Pixel-level maps "for more than a hundred images over categories applicable to most building types: wall, door, window, interior and background" (p.6). Local README colours: background black, interior white, walls red, doors blue, windows cyan. The local copy has `pngs/`, `svgs/`, `segmented_pngs/`, `segmented_descrete_pngs/` and `segmented_svgs/`, 110 each.
- **Metric.** Per-class precision, recall and IoU (Table 4). Not stated:
  - whether pixels are pooled over images or averaged per image;
  - the evaluation resolution;
  - how CubiCasa's classes were mapped to "interior" and "background".
- **Raster2Seq** (2026-phung-raster2seq App. E.2, Table 6 p.14) evaluates "interior segmentation" IoU / precision / recall on "the WAFFLE test set" ("approximately 100 annotated samples", §4.1 p.5). All models are trained on CubiCasa5K only.

### 3.2 Published numbers (all zero-shot from CubiCasa5K)

| Model | Walls P / R / IoU | Doors P / R / IoU | Windows P / R / IoU | Interior P / R / IoU | Background P / R / IoU | Source |
|---|---|---|---|---|---|---|
| CubiCasa5K model (ResNet-152 hourglass) | 0.737 / 0.590 / 0.488 | 0.201 / 0.163 / 0.099 | 0.339 / 0.334 / 0.202 | 0.799 / 0.521 / 0.461 | 0.697 / 0.912 / 0.653 | 2024-ganon-waffle T4 p.6 |
| Diffusion wall segmenter (ControlNet on SD 1.4, trained on CubiCasa5K walls) | 0.746 / 0.805 / 0.632 | n/a | n/a | n/a | n/a | WAFFLE T7 p.16 |
| FRI-Net | n/a | n/a | n/a | 63.4 / 84.2 / 56.7 | n/a | Raster2Seq T6 p.14 |
| RoomFormer | n/a | n/a | n/a | 65.7 / 88.3 / 60.5 | n/a | Raster2Seq T6 p.14 |
| Raster2Seq | n/a | n/a | n/a | 81.6 / 88.6 / 73.9 | n/a | Raster2Seq T6 p.14 |
| **Our segmenter v2 (zero-shot)** | to fill | to fill | to fill | to fill | to fill | |

Raster2Seq lists "CubiCasa5K pretrained†" with IoU 46.1, precision 79.9 and recall 52.1, "Reported in [Ganon et al. 2025]". That is the Interior column of WAFFLE Table 4.

### 3.3 Comparability and how to evaluate

- **Regime:** the same as ours (trained elsewhere, zero-shot), which makes WAFFLE the fairest public comparison we have.
- **Class map:**
  - wall ← wall ∪ column (WAFFLE has no column class);
  - door ← door;
  - window ← window;
  - interior ← v2 interior head ∧ not (wall, door, window, column), so stairs and rooms count as interior, because white excludes the red, blue and cyan elements;
  - background ← the rest.
- **Ground truth:** use `segmented_descrete_pngs` (discrete colours). Score at the resolution of the benchmark PNG: run the model at its scale, then resize the class map back with nearest neighbour.
- **Aggregation:** report pixel totals pooled over the 110 images (as CubiCasa does) and the per-image mean, since the paper does not fix this.
- **Scale:** WAFFLE plans have no scale. The pilot proposes one from door widths, so a wrong scale is part of our error. Report the scale method. Optionally add an oracle run with a hand-set scale on a subset, to separate scale errors from segmentation errors.

---

## 4. CVC-FP

### 4.1 Protocol

- **Data.** "122 high-resolution images in four different drawing styles" (2017-dodge-parsing-floor-plan-images, p.2). Local files by name group (my check of image sizes):
  - 90 `I*`/`II*` files at 2,479–2,480 × 3,508 px;
  - 19 `image0xx` at 5,671 × 7,383;
  - 10 files `1`–`10` of varying size;
  - 4 files `p1`–`p4` at 2,550 × 3,300.

  No local paper names or reports the subsets.
- **Wall segmentation (Dodge et al. 2017).**
  - Metrics: "the mean pixel accuracy and mean Intersection-over-Union (IoU), as previously introduced in [14] (FCN) as well as the Jaccard Index (JI) for wall pixels as proposed in [3] (de las Heras et al. 2013)" (p.2).
  - Mean accuracy and mean IoU are averaged over the two classes (wall, non-wall); JI is the wall IoU.
  - In-domain: "We perform 5-fold cross validation, whereas [3] perform leave-one-out cross validation for certain subsets" (p.3).
  - Evaluation resolution: not stated.
- **Swaileh et al. 2021:** 5-fold cross-validation (3/5 train, 1/5 val, 1/5 test), "mean over all test folds"; mean accuracy, mean IoU and mean Dice (§5, p.13).
- **Room detection.** The per-subset detection rates of de las Heras et al. (2014/2015) are **not in the local corpus**. The only local room-detection figures are Ahmed et al. on "a data set containing original floor plan images … introduced in [Macé et al. 2010] … The size of each floor plan image … is 2479 × 3508", 80 images from CVC, binarised. They are scored with the Phillips and Chhabra (1999) protocol: one-to-one detection rate, recognition accuracy, and one-to-many / many-to-one counts (2011-ahmed-improved-analysis §IV pp.4–5).
  - The image size matches the 90 local `I*`/`II*` images, so this is probably a subset of them. No paper says so.

### 4.2 Published numbers

Wall segmentation (Mean acc. / Mean IoU / JI or Dice, %):

| Method | Training | Mean acc. | Mean IoU | JI (wall IoU) | Source |
|---|---|---|---|---|---|
| Global threshold (GT) | CVC 5-fold | 78.7 | 69.6 | 41.5 | 2017-dodge T2 p.3 |
| GT + morphological closing | CVC 5-fold | 95.1 | 80.1 | 61.9 | Dodge T2 |
| GT + MC + text removal | CVC 5-fold | 95.3 | 85.0 | 71.2 | Dodge T2 |
| Patches – RF | CVC 5-fold | 92.8 | 89.3 | 79.2 | Dodge T2 |
| Patches – SVM | CVC 5-fold | 92.6 | 89.0 | 78.5 | Dodge T2 |
| Patches – BoW | CVC 5-fold | 93.0 | 87.9 | 76.6 | Dodge T2 |
| FCN-2s | CVC 5-fold | 97.3 | 94.4 | 89.2 | Dodge T2 |
| **FCN-2s, cross-dataset** | **R-FP only (zero-shot on CVC)** | **84.2** | **81.7** | **64.7** | Dodge T3 p.3 |
| FCN-2s | R-FP + CVC | 96.0 | 92.9 | 86.3 | Dodge T3 |
| U-net (original CVC-FP) | CVC 5-fold | 99.73 | 95.01 | Dice 95.65 | 2021-swaileh-versailles-fp T2 p.14 |
| U-net (CVC-FP with Versailles background) | 5-fold | 99.71 | 93.45 | Dice 95.01 | Swaileh T2 |
| **Our segmenter (zero-shot)** | Swiss Dwellings synthetic | to fill | to fill | to fill | |

Swaileh T1 (p.13) also reports its automatic wall-mask generator against CVC-FP ground truth: Dice 90.75 % and IoU 83.05 % on filled walls (70 % of the set), 34.45 % and 21.78 % on hollow walls. This is not a learned model.

Room detection, 80-image CVC set (2011-ahmed-improved-analysis T I p.5; 2012-ahmed-room-detection-labeling T I p.5):

| System | Detection rate % | Recognition accuracy % | One-to-many | Many-to-one |
|---|---|---|---|---|
| Macé et al. 2010 [ref. 18 / 5] | 85 | 69 | 2 | 0.76 |
| Ahmed 2011 | 89 | 79 | 1.50 | 1.65 |
| Ahmed 2011 without boundary detection | 70.63 | 82.32 | 0.96 | 1.11 |
| Ahmed 2012 with semantic division | 85 | 82 | 1.25 | 1.79 |

### 4.3 Comparability and how to evaluate

- **Regime:** the published deep-learning rows are in-domain cross-validation. The fair comparison for us is Dodge's **R-FP → CVC** row (mean IoU 81.7, JI 64.7), a model trained on another style. Everything else is an upper bound.
- **Evaluate on all 122 images** (no folds needed, since we do not train): wall vs. non-wall mean accuracy, mean IoU and JI. Report pooled pixels, and per name group as our own breakdown.
- **Classes:** column → wall, if CVC-FP walls include pillars; check the SVG ground truth. Door and window are not wall. Check whether CVC-FP wall polygons span the openings and match our convention (as for CubiCasa).
- **Rooms:** Ahmed's room figures use an old matching protocol on 80 images. Without de las Heras's per-subset rates, room detection on CVC-FP is only an internal metric for us. Use one-to-one matching at IoU ≥ 0.5, as in `metrics.match_rooms`.

---

## 5. Other benchmarks with several comparable published numbers

| Benchmark | Data | Metric | Published numbers (examples) | Local? | Evaluable by us? |
|---|---|---|---|---|---|
| R2V (LIFULL subset, Liu et al. 2017) | 815 images (715 / 100, 2019-zeng-deepfloorplan p.4); 870 (770 / 100) per 2023-yang-vectorfloorseg | Junction, opening, icon and room accuracy and recall (2017-liu-raster-to-vector T1 p.8); pixel accuracy per class and mIoU (DeepFloorplan T2 p.7) | Liu (full IP): opening acc / recall 91.9 / 90.2, room 84.5 / 88.4. CubiCasa model + IP on R2V: opening 93.2 / 92.6, room 91.7 / 90.8 (2019-kalervo T2 p.10). DeepFloorplan: wall acc 0.89, door-and-window acc 0.89, mIoU 0.74 (0.76 post-processed); DeepLabV3+ 0.80 / 0.72 / 0.69; PSPNet 0.84 / 0.76 / 0.70 | No | No |
| R3D (Rent3D 214 + 18 round plans) | 179 / 53 | As DeepFloorplan T2 | DeepFloorplan wall acc 0.98, door-and-window 0.83, mIoU 0.63 (0.66); DeepLabV3+ 0.93 / 0.60 / 0.50; PSPNet 0.91 / 0.54 / 0.50 | No | No |
| R-FP (Dodge et al.) | 500 Japanese real-estate plans | Wall mean acc. / mean IoU / JI | FCN-2s 94.0 / 89.7 / 80.9 (Dodge T1 p.3) | No | No |
| Raster2Graph dataset | 9,803 / 500 / 499 | Junction, wall-segment, region and room P / R / F1; wall pixel accuracy | Wall accuracy: R2V 77.2, DFPR 74.4, R2G 81.6 (2024-hu-raster-to-graph T3 p.11); room F1: Raster2Seq 85.1, R2G 83.4 (Raster2Seq T1) | No | No |
| Structured3D (density maps) and Structured3D-B (binary renders) | 3,000 / 250 / 250 | Room / Corner / Angle F1 | RoomFormer, PolyRoom, FRI-Net tables; Raster2Seq on Structured3D-B Room F1 99.6 | No | No (point-cloud density maps, not drawings) |
| LS-CAD (2024-yang-cadspotting) | 45 large CAD floor plans | PQ / SQ / RQ, F1, mAP | SPv2 + block partitioning PQ 26.7; CADSpotting + block partitioning 58.8; CADSpotting + sliding windows 75.5 (T6 p.6) | No | Possibly, if released |
| ArchCAD-400K (2025-luo-archcad-400k) | 413,062 chunks of 14 m × 14 m; test 82,612 | PQ, F1 / wF1, AP (no priors) | PQ: CADTransformer 60.0, SymPoint 47.6, SymPoint-V2 60.5, DPSS 70.6 (T4 p.9) | No (gated; `data/README.md` defers it to legal review) | Later |
| ResPlan-FP (2026-zhang-readout-vectorization) | 1,000 test plans from ResPlan | fpeval wall F1 | Zero-shot fusion 0.688; wall-first arm 0.640 / 0.621 (§6.6) | No | No; single-paper protocol |

The FloorPlanCAD-trained SymPoint drops from PQ 83.3 to 33.2 on CADSpotting's cross-dataset test column (2024-yang-cadspotting T8 p.11). It is the only published zero-shot-like number for a vector symbol spotter.

---

## 6. Suggested wording for the model card

**Benchmarking method.** The segmenter is trained only on synthetic renders of Swiss Dwellings and evaluated **zero-shot** on public benchmarks, following each benchmark's published protocol:

- **FloorPlanCAD** semantic symbol spotting (Fan et al., ICCV 2021, §6.1):
  - each test block is rendered at 50 px/m;
  - each vector primitive takes the majority class of points sampled along it (Fan et al. Eq. 10);
  - per-category F1 is computed over primitives weighted by log(1 + length) (Fan et al. §10.1), with the categories of Fan et al. Fig. 7;
  - only door, window, stair and wall are reported, on the November 2021 test split (5,502 blocks).
- **CubiCasa5K:** per-class pixel IoU on the 400 official test plans, one confusion matrix pooled over the split (Kalervo et al. 2019, Table 4); wall includes the door and window openings as in the CubiCasa labels.
- **WAFFLE:** per-class precision, recall and IoU on the 110 annotated plans (Ganon et al. 2025, Table 4; interior IoU as in Phung and Averbuch-Elor 2026, Table 6).
- **CVC-FP:** wall Jaccard index and mean IoU / accuracy over wall vs. non-wall on all 122 plans (Dodge et al. 2017).

Published results are in-domain (trained on the benchmark's training split), except on WAFFLE and the cross-dataset CVC-FP row. The comparison therefore shows the cost of the domain shift, not a ranking of methods.

---

## 7. Open points

- **CubiCasa5K evaluation resolution:** whether the published tables were produced with the released code, at original size and with rotation test-time augmentation. The paper is silent.
- **Code-level points:**
  - the unit of L in log(1 + L);
  - CADTransformer's micro F1;
  - SymPoint's pooled PQ;
  - CubiCasa's `original_size=True` and the label conventions.

  These were read through a summarising web fetch. Confirm in the repositories before citing them in the model card.
- **FloorPlanCAD Table 3:** whether the categories merge labels or average classes. The arithmetic check fits within 0.02 but not exactly.
- **FloorPlanCAD Aug-2021 test archive (3,827 drawings):** availability not checked.
- **WAFFLE:** aggregation (pooled or per image), resolution and the CubiCasa class mapping are unstated in both WAFFLE and Raster2Seq.
- **CVC-FP:** the subset names and per-subset room detection rates of de las Heras et al. are not in the local corpus. The papers (IJDAR 2013 and 2015) would be needed.
