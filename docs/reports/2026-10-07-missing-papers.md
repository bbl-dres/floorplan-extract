# Missing Papers: Literature Gap Report for the Floor Plan Extraction Pilot

*Status: report as delivered on 7 October 2026; the papers it recommends are now in the [research index](../../research/README.md) and the [literature review](../literature-review.md). Three claims were corrected against the papers afterwards: VectorGraphNET's 89.0 weighted F1 is on FloorPlanCAD (plain F1 79.4 against SymPoint's 86.8); PP-OCRv6's dictionaries do include French, German and Italian; HRDA's gains are on fine-detail classes. `research/state-of-the-art.md` has since been merged into the literature review.*

*Research report · 7 October 2026 · Scope: papers the project does not yet have (`research/papers.json`, 42 entries) or cites only without a local copy (`research/sources.md`, 678 entries). Ranked by expected impact on pilot v2 and its known weak spots. Nothing in the repository was changed.*

## 1. Method and Caveats

- **Baseline.** I read `papers.json`, `research/README.md`, `sources.md`, `state-of-the-art.md`, `docs/literature-review.md` and the 7 October pilot review, to avoid duplicates and to rank against the failures that were actually measured:
  - wall IoU 0.57 zero-shot on CubiCasa5K, against 0.91 on synthetic data;
  - S3 ornament read as walls (IoU 0.29);
  - column IoU 0.09 and stairs 0.38;
  - open-plan and merged rooms (65% of rooms matched);
  - door swings, scale and multi-floor sheets.
- **Search.** Five parallel searches, one per group of the nine requested areas. Sources: arXiv, CVF, ACL Anthology, PMLR, publisher DOI pages, Crossref, OpenAlex, Zenodo, GitHub and Hugging Face. Every entry below was checked on its own page.
  - I re-checked the 2025–2026 must-read items myself on arXiv, Crossref, Zenodo and GitHub.
  - CVF, ASCE, Springer, Wiley and MDPI pages often refused automated fetching. For those papers, title, venue and abstract were checked through Crossref, OpenAlex or Semantic Scholar by DOI. The entries say so.
- **Limits.** The shared web-search quota ran out near the end of the search. These topics were searched less thoroughly:
  - door-width priors for scale (no peer-reviewed paper found);
  - classic hatched-area detection;
  - Versailles-FP follow-ups;
  - 2025–26 raster wall papers on Mask2Former, SAM or DINOv2;
  - column detection in architectural (not structural) plans.
- **Licence language.**
  - "Permissive" means MIT, Apache-2.0 or BSD.
  - NC means a non-commercial or research-only licence.
  - "No licence" means no licence file, so all rights are reserved by default.
  - Weights trained on CubiCasa5K, Structured3D, LIFULL or Cityscapes are treated as NC regardless of the tag on the weights.
- **Housekeeping found on the way:**
  - `papers-original/2026-guo-llm-floor-plan-analysis.pdf` (and its Markdown version) is not in `papers.json`. It is the Preprints.org version (doi:10.20944/preprints202605.1893.v1). The published version is Guo et al., *Applied Sciences* 2026, doi:10.3390/app16136290 (CC BY). It uses the same design as our pilot (ResNet-34 U-Net, YOLOv8, manual scale) on 101 plans. Index it, or drop it.
  - None of the must-read papers below appears in `sources.md`.

## 2. Summary

The project's reading list covers floor-plan-specific methods well. It has almost nothing on the levers that would move the measured numbers most:

1. **Using the unlabelled archive scans.** Unsupervised domain adaptation (MIC, DAFormer/HRDA) and, once a few sheets are labelled, semi-supervised training (UniMatch).
2. **Better synthetic data.** Procedural hatching and textures cut from real scans and composited into wall *and* non-wall regions (Petitpierre 2026). This addresses the pilot's finding that "the renderer hatches only walls".
3. **Topology-aware losses** against thin-wall gaps that make rooms leak or merge (TopoMortar, clDice, Skeleton Recall), plus closed-shape extraction by watershed (Chen et al. 2024).
4. **A primitive classifier for vector PDFs** (VectorGraphNET) and deterministic scale and column rules (Talebi-Kalaleh & Mei 2026, Lv et al. 2021).
5. **Decoding room stamps against a dictionary inside the CTC decoder** (Word Beam Search), instead of fuzzy matching afterwards.
6. **An annotation protocol** for the first real labelled sheets (Petitpierre & Guhennec 2023).

## 3. Must-Read Tier (11 papers)

Ranked by expected impact. Suggested filenames follow `year-firstauthorsurname-topic.pdf`; groups are existing `papers.json` groups.

| # | Paper | Venue / ID | Open access | Weak spot | Filename | Group |
|---|---|---|---|---|---|---|
| 1 | MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation (Hoyer et al.) | CVPR 2023, arXiv 2212.01322 | Yes | Synthetic-to-real gap, hatching | `2023-hoyer-mic.pdf` | Raster deep learning and vectorisation |
| 2 | Generalizable Multiscale Segmentation of Heterogeneous Map Collections (Petitpierre) | arXiv 2603.05037 (2026) | Yes | Hatching, ornament, style gap | `2026-petitpierre-semap.pdf` | Raster deep learning and vectorisation |
| 3 | Domain Adaptive and Generalizable Network Architectures and Training Strategies for Semantic Image Segmentation (Hoyer et al.; DAFormer + HRDA) | TPAMI 2023, arXiv 2304.13615 | Yes | Thin lines, rare classes (columns, stairs), large sheets | `2023-hoyer-daformer-hrda.pdf` | Raster deep learning and vectorisation |
| 4 | TopoMortar: A dataset to evaluate image segmentation methods focused on topology accuracy (Valverde et al.) | BMVC 2025 (oral), arXiv 2503.03365 | Yes | Thin lines, broken walls causing leaking or merged rooms | `2025-valverde-topomortar.pdf` | Raster deep learning and vectorisation |
| 5 | Raster2Seq: Polygon Sequence Generation for Floorplan Reconstruction (Phung & Averbuch-Elor) | SIGGRAPH 2026, arXiv 2602.09016 | Yes | Room polygons, open-plan and merged rooms, openings | `2026-phung-raster2seq.pdf` | Raster deep learning and vectorisation |
| 6 | VectorGraphNET: Graph Attention Networks for Accurate Segmentation of Complex Technical Drawings (Carrara et al.) | arXiv 2410.01336 (2024) | Yes | Vector-PDF input, thin lines, poché via fill | `2024-carrara-vectorgraphnet.pdf` | Vector symbol spotting |
| 7 | Word Beam Search: A Connectionist Temporal Classification Decoding Algorithm (Scheidl et al.) | ICFHR 2018, doi:10.1109/ICFHR-2018.2018.00052 | Yes (author copy) | Stamp OCR, vocabulary correction, hand lettering | `2018-scheidl-word-beam-search.pdf` | VLMs, benchmarks and OCR |
| 8 | Residential Floor Plan Recognition and Reconstruction (Lv et al.) | CVPR 2021, doi:10.1109/CVPR46437.2021.01644 | Yes | Scale from dimension strings, inclined walls | `2021-lv-residential-floor-plan-recognition.pdf` | Raster deep learning and vectorisation |
| 9 | Training-Free Agentic Computer Vision for Structural Component Detection in 2D Structural Framing Plans (Talebi-Kalaleh & Mei) | arXiv 2608.17237 (2026) | Yes | Scale consensus, columns, PDF operator parsing | `2026-talebi-kalaleh-framing-plan-parsing.pdf` | Structural and graph-based pipeline |
| 10 | Automatic vectorization of historical maps: A benchmark (Chen et al.) | PLOS ONE 19(2) e0298217, 2024, doi:10.1371/journal.pone.0298217 | Yes (CC BY 4.0) | Rooms as closed shapes, room PQ evaluation | `2024-chen-historical-map-vectorization.pdf` | Raster deep learning and vectorisation |
| 11 | Effective annotation for the automatic vectorization of cadastral maps (Petitpierre & Guhennec) | Digital Scholarship in the Humanities 38(3):1227–1237, 2023, doi:10.1093/llc/fqad006 | Yes (CC BY) | Small labelled sets, gold-set labelling | `2023-petitpierre-cadastral-annotation.pdf` | Datasets |

Most of the must-read papers land in "Raster deep learning and vectorisation". A new group such as "Domain adaptation, training data and annotation" would fit #1–#4 and #11 better, if adding groups is acceptable.

### 3.1 Details

**1. MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation**
- **Paper:** Lukas Hoyer et al., 2023, CVPR 2023, arXiv 2212.01322.
- **PDF:** https://arxiv.org/pdf/2212.01322
- **Code:** https://github.com/lhoyer/MIC. `seg/LICENSES.md` (verified):
  - MIC, HRDA, DAFormer and MMSegmentation 0.16: Apache-2.0
  - DACS parts: MIT
  - SegFormer/MiT files: NVIDIA Source Code License (non-commercial)
  - AdaptSegNet files: "non-commercial research purposes only"
- **Weights:** checkpoints are GTA/Synthia→Cityscapes only. Do not use them.
- **Mixing baseline:** DACS (Tranheden et al., WACV 2021, arXiv 2007.08702; code MIT).
- **Stage:** segmenter training (feeds stages 3–6).
- **Weak spots:** synthetic-to-real gap, hatching and ornament read as walls.
- **Why it matters for us:**
  - Keep the ResNet-34 U-Net as student and add an EMA teacher that pseudo-labels unlabelled archive, WAFFLE and Commons scans.
  - Use DACS class-mixing to paste synthetic wall, door and window pixels into real scans.
  - Use MIC's masked-patch consistency to force context reasoning ("hatching inside a room is not wall"). The method is reported to work on CNNs (DeepLabV2-R101).
  - Reimplement the few hundred lines in our loop. Do not import the NVIDIA- or AdaptSegNet-licensed files.
  - Judge it on the CubiCasa5K, CVC-FP and WAFFLE harness, not on synthetic validation.

**2. Generalizable Multiscale Segmentation of Heterogeneous Map Collections**
- **Paper:** Remi Petitpierre (EPFL), March 2026, arXiv 2603.05037 (preprint, 30 pp.).
- **PDF:** https://arxiv.org/pdf/2603.05037
- **Data:** Semap on Zenodo, doi:10.5281/zenodo.16164781. The record licence is CC BY 4.0, but the images carry their own rights in `license_images.md`, so check them per use.
  - 1,439 annotated and 12,122 synthetic samples.
  - `model.zip` holds Mask2Former Swin-L weights built on MMSegmentation (Apache-2.0).
  - I found no separate repository for the synthesis code.
- **Predecessor:** "Generic Semantic Segmentation of Historical Maps" (CHR 2021, CEUR Vol-2989). Its code has no licence.
- **Stage:** renderer (training data) and inference.
- **Weak spots:** hatching or ornament read as walls, thin versus thick strokes, style gap.
- **Why it matters for us:**
  - The procedural synthesis randomises fills, dot patterns, hatchings, texture masks cut from real scans, GMM-sampled colours, stroke width and JPEG artefacts. Synthetic pretraining added +5.1 mIoU. Inference averages logits at full and half resolution.
  - We would cut poché, hatching and ornament textures from unlabelled archive scans and composite them into both wall and non-wall regions of the Swiss Dwellings renders. This fixes the "renderer hatches only walls" failure seen on CubiCasa.
  - We would also add two-scale test-time averaging.
  - It comes from a Swiss group with cadastral and map expertise.

**3. Domain Adaptive and Generalizable Network Architectures and Training Strategies for Semantic Image Segmentation**
- **Paper:** Lukas Hoyer, Dengxin Dai, Luc Van Gool, TPAMI 2023, arXiv 2304.13615. It consolidates DAFormer (CVPR 2022, arXiv 2111.14887) and HRDA (ECCV 2022, arXiv 2204.13132) and adds domain generalisation.
- **PDF:** https://arxiv.org/pdf/2304.13615
- **Code:** https://github.com/lhoyer/HRDA (domain generalisation on the `dg` branch). The LICENSE text is Apache-2.0, with component licences listed separately. Same NVIDIA/SegFormer caveat as #1.
- **Stage:** segmenter training and inference on large sheets.
- **Weak spots:** thin walls and window lines, columns (IoU 0.09), stairs (0.38), A0 sheets.
- **Why it matters for us:**
  - HRDA trains on a high-resolution detail crop plus a 2× downscaled context crop, fused by learned scale attention. Its largest gains were on thin classes.
  - DAFormer's Rare Class Sampling and ImageNet feature-distance regularisation are backbone-agnostic.
  - Both can be added to the U-Net loop. Rare Class Sampling alone is a cheap first test for columns and stairs.
  - The domain-generalisation results apply where no target scans exist for an archive.

**4. TopoMortar: A dataset to evaluate image segmentation methods focused on topology accuracy**
- **Paper:** Juan Miguel Valverde et al., 2025, BMVC 2025 (oral), arXiv 2503.03365.
- **PDF:** https://arxiv.org/pdf/2503.03365
- **Code:** https://github.com/jmlipman/TopoMortar (MIT). One PyTorch framework with these losses: CE+Dice, clDice, Skeleton Recall, cbDice, RegionWise, Warping, TopoLoss and others.
- **Stage:** segmenter loss.
- **Weak spots:** thin walls, gaps that make rooms leak and merge, spurious strokes.
- **Why it matters for us:**
  - Mortar lines that partition bricks are a close analogue of walls that partition rooms.
  - Results: clDice gave the most topologically accurate masks; Skeleton Recall was the most robust to noisy labels; augmentation plus self-distillation lifted plain CE+Dice to rival most topology losses.
  - We would A/B three variants on the wall channel, scored by room-match rate and Betti-1 error rather than pixel IoU:
    - CE+Dice;
    - CE+Dice plus soft-clDice (our labels are clean, vector-derived);
    - CE+Dice plus Skeleton Recall (once MIC pseudo-labels enter training).
  - Read it together with clDice and Skeleton Recall (§4.2).
  - Counter-evidence: #10 found plain BCE beat MOSIN, TopoLoss and BALoss on historical maps, so the harness has to decide.

**5. Raster2Seq: Polygon Sequence Generation for Floorplan Reconstruction**
- **Paper:** Hao Phung, Hadar Averbuch-Elor, 2026, SIGGRAPH 2026, arXiv 2602.09016 (v3, 7 Aug 2026).
- **PDF:** https://arxiv.org/pdf/2602.09016
- **Code:** https://github.com/Cornell-VAILab/Raster2Seq (MIT).
- **Weights:** https://huggingface.co/haopt/Raster2Seq, tagged MIT but trained on Structured3D, CubiCasa5K (CC BY-NC) and Raster2Graph/LIFULL (research only). Do not ship them.
- **Input:** 256 px, with 512 px variants.
- **Stage:** stage 6 rooms; openings.
- **Weak spots:** open-plan rooms that no wall-based method finds; rooms merged through a missed door.
- **Why it matters for us:**
  - It is the 2026 state of the art for raster room, door and window polygons. The decoder is autoregressive, with anchor-guided corner sequences.
  - Reported results: CubiCasa5K room F1 88.7 against 83.5 for RoomFormer; zero-shot on WAFFLE 73.9 against 60.5 IoU.
  - We would retrain it on our style-randomised renders, cropped per unit, as a second learned room hypothesis. QA would flag where it disagrees with the free-space rooms.
  - It does not replace the region method: the resolution is low and non-Manhattan walls are not handled.

**6. VectorGraphNET: Graph Attention Networks for Accurate Segmentation of Complex Technical Drawings**
- **Paper:** Andrea Carrara, Stavros Nousias, André Borrmann (TUM), 2024, arXiv 2410.01336 (preprint).
- **PDF:** https://arxiv.org/pdf/2410.01336 (paper licence CC BY-NC-SA 4.0).
- **Code:** none released. The TUM evaluation set is proprietary.
- **Stage:** the vector-PDF branch (CAD prints, like S1).
- **Weak spots:** thin lines, poché and hatching (fill and stroke attributes are features).
- **Why it matters for us:**
  - Method: PDF→SVG, each path becomes a node with geometric and style features, joined in a kNN graph and labelled hierarchically by graph attention (wall/door/window, then e.g. load-bearing wall).
  - Reported results: weighted F1 0.89 on FloorPlanCAD (SymPoint 0.868) and 0.97 on 76 real TUM university-building plans, with 1.3 M parameters.
  - We would classify PDF paths directly instead of rasterising CAD prints. Training data would be vector PDFs generated from Swiss Dwellings and compliant DWGs, where labels come free.

**7. Word Beam Search: A Connectionist Temporal Classification Decoding Algorithm**
- **Paper:** Harald Scheidl, Stefan Fiel, Robert Sablatnig, 2018, ICFHR 2018, pp. 253–258, doi:10.1109/ICFHR-2018.2018.00052.
- **PDF:** author copy at https://repositum.tuwien.at/retrieve/1835
- **Code:** https://github.com/githubharald/CTCWordBeamSearch (MIT). No weights needed.
- **Stages:** 2 and 7 (stamp OCR and vocabulary correction).
- **Weak spots:** hand-lettered and calligraphic stamps, stamps whose numbers get "corrected" into words.
- **Why it matters for us:**
  - The PP-OCR recogniser is CTC-based. We would decode its per-frame softmax with a dictionary of DE/FR/IT room usages and abbreviations in "Words" mode, leaving digits, "m²" and AOIDs unconstrained.
  - This constrains the vocabulary at decode time and never rewrites numbers, unlike Levenshtein matching afterwards.
  - It needs the recogniser logits (e.g. from ONNX) and a reordering of the blank column, which Paddle puts at index 0.

**8. Residential Floor Plan Recognition and Reconstruction**
- **Paper:** Xiaolei Lv, Shengchu Zhao, Xinyang Yu, Binqiang Zhao, CVPR 2021, pp. 16717–16726, doi:10.1109/CVPR46437.2021.01644.
- **PDF:** https://openaccess.thecvf.com/content/CVPR2021/papers/Lv_Residential_Floor_Plan_Recognition_and_Reconstruction_CVPR_2021_paper.pdf
- **Code:** none found. Whether the RFP dataset was released is unverified.
- **Stages:** 9 (scale); 3 (vectorisation).
- **Weak spots:** scale for plans without a known scale; inclined walls.
- **Why it matters for us:**
  - Dimension numbers are detected and read, dimension-line endpoints come from a keypoint network, and cluster analysis of the ratios rejects mismatched pairs.
  - This is the peer-reviewed template for the "dimension string" scale cue the review asks for. It sits next to the stamp-area and door-width cues in a robust consensus.
  - Its iterative vectorisation that allows inclined walls is a model for refining our skeleton graph.

**9. Training-Free Agentic Computer Vision for Structural Component Detection in 2D Structural Framing Plans**
- **Paper:** Mohammad Talebi-Kalaleh, Qipei Mei, 2026, arXiv 2608.17237 (preprint, v2 26 Aug 2026).
- **PDF:** https://arxiv.org/pdf/2608.17237
- **Code:** not released. A 100-drawing vector benchmark is promised.
- **Stages:** vector-PDF parsing, 9 (scale), columns.
- **Weak spots:** scale, columns.
- **Why it matters for us:** it gives deterministic, auditable rules we can port almost directly to vector PDFs like S1, and to scans once OCR provides the dimension text.
  - The parser walks the PDF graphics operators with the transformation stack, line width, dash pattern and fill.
  - Scale comes from dimension text paired with thin dimension lines:
    - the text is within 12 mm of the line and projects into its middle 90%;
    - ticks or arrows are present;
    - outliers are rejected by median absolute deviation (MAD), with the inlier fraction as confidence;
    - the result snaps to standard scales within 1.5%.
    - Reported scale error: within 0.1%.
  - Columns are closed glyphs of 60–2,000 mm with aspect ratio ≤ 6.
  - Caveats: it was tested on structural framing plans; skip its agentic layer, which calls an external VLM.

**10. Automatic vectorization of historical maps: A benchmark**
- **Paper:** Yizi Chen, Joseph Chazalon, Edwin Carlinet, Minh Ôn Vũ Ngoc, Clément Mallet, Julien Perret, 2024, PLOS ONE 19(2) e0298217, doi:10.1371/journal.pone.0298217.
- **PDF:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0298217 (CC BY 4.0).
- **Code:** https://github.com/soduco/Benchmark_historical_map_vectorization has no licence, so reimplement only. It has 18 weight files.
- **Data:** Zenodo 10567765, CC BY 4.0.
- **Lineage:**
  - ICDAR 2021 paper "Vectorization of Historical Maps Using Deep Edge Filtering and Closed Shape Extraction", doi:10.1007/978-3-030-86337-1_34.
  - MapSeg competition, arXiv 2105.13265. Its evaluation tool is MIT.
- **Stages:** 6 (rooms), evaluation.
- **Weak spots:** leaking and merged rooms.
- **Why it matters for us:**
  - Building blocks are closed shapes bounded by lines, like rooms. The best method was a U-Net edge probability map plus Meyer watershed (PQ 51.1%).
  - We would run watershed on the wall probability map, seeded by the room stamps. Rooms would then no longer depend on a perfectly closed binary wall mask. Room scoring would use COCO PQ.
  - Its negative result on topology losses is a necessary check on #4.

**11. Effective annotation for the automatic vectorization of cadastral maps**
- **Paper:** Remi Petitpierre, Paul Guhennec, 2023, Digital Scholarship in the Humanities 38(3):1227–1237, doi:10.1093/llc/fqad006.
- **PDF:** via https://academic.oup.com/dsh/article/38/3/1227/7070514 (CC BY).
- **Code:** none.
- **Stage:** labelling the BBL gold set and the first real fine-tuning sheets.
- **Weak spot:** small labelled sets.
- **Why it matters for us:** it distils five large cadastral vectorisation projects in Switzerland, Venice and Paris:
  - an optimal ontology has only 2–3 visually homogeneous classes;
  - annotate only what is visible;
  - enforce consistency and work iteratively;
  - "a dozen of cadastral sheets is already sufficient" to bootstrap a model.

  We would label visual classes (e.g. wall fill, hatching that is not wall, line work), map them to semantics afterwards, and budget about 12 sheets per archive style.

## 4. Useful Tier

Grouped by pipeline stage. Each entry: title, first author, year, venue and ID, open access and PDF, code and licence, stage or weak spot, and why it matters.

### 4.1 Domain Adaptation, Synthetic Data and Backbones

- **Exploring the Benefits of Vision Foundation Models for Unsupervised Domain Adaptation** (VFM-UDA)
  - Englert et al., 2024, CVPR 2024 Workshops, arXiv 2406.09896. Follow-up: VFM-UDA++, ICCV 2025 Workshops, arXiv 2503.10685.
  - OA: https://arxiv.org/pdf/2406.09896
  - Code: https://github.com/tue-mps/vfm-uda (MIT); standalone PyTorch Lightning, DINOv2 backbones (Apache-2.0). HF checkpoint licence and VFM-UDA++ code: unverified.
  - Stage: segmenter.
  - Why: the cleanest permissive route if we accept a ViT. Train DINOv2 + UDA on renders plus unlabelled scans, then use it as a teacher to pseudo-label scans for the ResNet-34 U-Net.
- **Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation** (UniMatch)
  - Yang et al., 2023, CVPR 2023, arXiv 2208.09910. UniMatch V2: arXiv 2410.10777 (accepted to TPAMI).
  - OA: yes.
  - Code: https://github.com/LiheYoung/UniMatch and https://github.com/LiheYoung/UniMatch-V2, both MIT.
  - Stage: fine-tuning once 10–30 real sheets are labelled.
  - Why: the standard MIT recipe for synthetic labels plus a few real labels plus many unlabelled scans. V1 works with CNNs; V2 argues that an encoder upgrade (DINOv2) beats method tweaks.
- **Stronger, Fewer, & Superior: Harnessing Vision Foundation Models for Domain Generalized Semantic Segmentation** (Rein)
  - Wei et al., 2024, CVPR 2024, arXiv 2312.04265. Permissive alternative: **SoMA**, Yun et al., CVPR 2025, arXiv 2412.04077.
  - OA: yes.
  - Code: Rein is **GPL-3.0** (https://github.com/w1oves/Rein). SoMA is MIT (https://github.com/ysj9909/SoMA) but acknowledges the Rein code base, so check file provenance.
  - Stage: domain generalisation with frozen DINOv2.
  - Why: best evidence that a frozen VFM with tiny adapters generalises from synthetic data alone. Use SoMA if we test this route.
- **Deep Floor Plan Analysis for Complicated Drawings Based on Style Transfer**
  - Kim, Park, Kim, Yu, 2021, J. Computing in Civil Engineering 35(2) 04020066, doi:10.1061/(ASCE)CP.1943-5487.0000942.
  - **Not open access.** An author-hosted copy was reported but not checked.
  - Code: none.
  - Weak spot: hatching and ornament.
  - Why: a conditional GAN maps any plan style to one clean style before vectorisation. Our renderer already produces the aligned pairs (randomised style ↔ canonical render) to train such a normaliser, or an auxiliary "canonical" decoder head. Metadata and abstract checked via Crossref and OpenAlex.
- **Automatic Uncertainty-Aware Synthetic Data Bootstrapping for Historical Map Segmentation**
  - Arzoumanidis et al., 2025 (v2 April 2026), arXiv 2511.15875.
  - OA: yes.
  - Code: https://github.com/hcu-cml/bootstrap-HistoricalMap, no licence.
  - Weak spot: style gap.
  - Why: unpaired translation (CycleGAN, UNSB) of rendered vector data into a historical style beat hand styling (about 87% vs 84% accuracy) and degradation alone (81%). Train a renders→archive-scans translator; watch for erased thin lines.
- **SynthPID: P&ID digitization from Topology-Preserving Synthetic Data**
  - Prasad & Mahapatra, 2026, arXiv 2604.16513. CVPR 2026 Workshops inferred from the CVF URL, not confirmed.
  - OA: yes.
  - Code: the named repo returned 404 (unverified).
  - Stage: generator design.
  - Why: perturbed real topologies beat template-based synthesis (63.8% vs about 33% edge mAP with zero real data), and seed diversity, not volume, limits gains. Run the same ablation: number of Swiss Dwellings buildings against styles per building.
- **A semi-supervised approach for building wall layout segmentation based on transformers and limited data**
  - Xie et al., 2025, Computer-Aided Civil and Infrastructure Engineering 40(10):1295–1313, doi:10.1111/mice.13397.
  - OA: hybrid, CC BY-NC-ND per Crossref.
  - Code: none.
  - Why: floor-plan-specific evidence that FixMatch-style pseudo-label consistency on unlabelled plans adds more than 4 IoU points for walls.
- **Wall-Layer Segmentation in Floor Plans for Energy Retrofit Planning**
  - Schönfelder, Brauksiepe, König, 2026 (ASCE Computing in Civil Engineering 2025, pp. 410–418), doi:10.1061/9780784486436.044.
  - Not OA. Code: none.
  - Weak spot: poché and hatch inside walls.
  - Why: U-Net on real plus synthetic German plans, segmenting wall material layers encoded by hatch patterns. Supports adding an explicit "wall fill / hatch" class instead of treating hatching as noise. Checked via Crossref and OpenAlex.
- **Deep extraction of manga structural lines**
  - Li, Liu, Wong, 2017, ACM TOG 36(4) art. 117, doi:10.1145/3072959.3073675.
  - OA: author PDF via https://ttwong12.github.io/papers/linelearn/linelearn.html
  - Code: https://github.com/ljsabc/MangaLineExtraction_PyTorch (MIT). Weights have no separate licence.
  - Weak spot: hatching and screen texture.
  - Why: a CNN that removes arbitrary screen patterns but keeps structural lines, trained on pattern-over-line overlays, the same construction we can do with hatch patterns. A quick pre-filter experiment.

### 4.2 Losses, Topology and Polygonisation

- **clDice – A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation**
  - Shit et al., 2021, CVPR 2021, arXiv 2003.07311.
  - OA: yes.
  - Code: https://github.com/jocpae/clDice (MIT).
  - Weak spots: thin walls, gaps; its precision term also penalises spurious strokes.
  - Why: add 0.3–0.5 × soft-clDice on the wall channel, about 20 lines. The soft-skeleton iterations must exceed half the thickest wall width, so downsample the loss input for thick poché.
- **Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures**
  - Kirchhoff et al., 2024, ECCV 2024, arXiv 2404.03010.
  - OA: yes.
  - Code: https://github.com/MIC-DKFZ/Skeleton-Recall (Apache-2.0).
  - Why: near-zero cost, since we can render wall centrelines from the vectors as the skeleton target. Most robust to noisy labels (TopoMortar), so pair it with MIC pseudo-labels.
- **Promoting Connectivity of Network-Like Structures by Enforcing Region Separation**
  - Oner et al., 2022, IEEE TPAMI, doi:10.1109/TPAMI.2021.3074366, arXiv 2009.07011.
  - OA: yes.
  - Code: no official repo verified.
  - Weak spot: leaking rooms.
  - Why: penalises exactly the case where a wall gap lets the two background regions on either side merge, and also false splits. It is the formulation closest to "rooms are free space between walls". Prototype after clDice.
- **Centerline Boundary Dice Loss for Vascular Segmentation** (cbDice)
  - Shi et al., 2024, MICCAI 2024, arXiv 2407.01517.
  - OA: yes.
  - Code: https://github.com/PengchengShi1220/cbDice (Apache-2.0).
  - Why: re-balances Dice by structure diameter, so thin partitions are not drowned out by thick outer walls. Use it if thin walls lag while thick ones look fine.
- **Boundary loss for highly unbalanced segmentation**
  - Kervadec et al., MIDL 2019 / Medical Image Analysis 2021, arXiv 1812.07032.
  - Code: https://github.com/LIVIAETS/boundary-loss (MIT).
  - Why: weights false positives by distance to the true boundary, a cheap test against hatching far from walls. It does not close gaps.
- **Improved Road Connectivity by Joint Learning of Orientation and Segmentation**
  - Batra et al., 2019, CVPR 2019, doi:10.1109/CVPR.2019.01063.
  - OA: CVF.
  - Code: https://github.com/anilbatra2185/road_connectivity (MIT, PyTorch 0.3).
  - Why: an auxiliary wall-orientation head (labels free from the vectors) plus a refinement net trained on deliberately cut walls in our renders.
- **Polygonal Building Segmentation by Frame Field Learning**
  - Girard et al., 2021, CVPR 2021, arXiv 2004.14875.
  - OA: yes.
  - Code: https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning (BSD-3). The aerial weights are useless to us.
  - Stage: mask→polygon for rooms and outline.
  - Why: a 4-channel frame-field head, supervised free from vector tangents, plus active-skeleton polygonisation gives sharp right-angle and oblique corners for DXF output and plan-check. It also defines the max-tangent-angle (MTA) metric.
- **HiSup: Accurate polygonal mapping of buildings in satellite imagery with hierarchical supervision**
  - Xu et al., 2023, ISPRS J. Photogramm. Remote Sens. 198:284–296, doi:10.1016/j.isprsjprs.2023.03.006, arXiv 2208.00609.
  - Code: https://github.com/SarahwXU/HiSup (MIT).
  - Why: vertex heatmap and attraction-field heads that make masks convert cleanly to polygons. An alternative to frame fields.
- **ScaleLSD: Scalable Deep Line Segment Detection Streamlined**
  - Ke et al., 2025, CVPR 2025, arXiv 2506.09369.
  - Code: https://github.com/ant-research/scalelsd (MIT).
  - Weights: https://huggingface.co/cherubicxn/scalelsd, Apache-2.0, **but trained on SA-1B (research only)**.
  - Why: zero-shot line segments to bridge mask gaps and to flag hatching (dense, short, parallel segments). Supersedes HAWP and DeepLSD (sources [171], [172]) for this use. Needs a licence review or retraining.
- **Cartographic reconstruction of building footprints from historical maps: A study on the Swiss Siegfried map**
  - Heitzler & Hurni, 2020, Transactions in GIS 24(2):442–461, doi:10.1111/tgis.12610.
  - Not OA. Code: none.
  - Why: a U-Net ensemble plus orientation-based clustering for polygon regularisation. It reports correction at about 45 minutes per sheet, the KPI our design calls decisive. Swiss (ETH).

### 4.3 Rooms, Open Plans, Openings, and Vector or CAD Input

- **VectorFloorSeg: Two-Stream Graph Attention Network for Vectorized Roughcast Floorplan Segmentation** (**known, [176], repository only**)
  - Yang et al., 2023, CVPR 2023.
  - OA: https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_VectorFloorSeg_Two-Stream_Graph_Attention_Network_for_Vectorized_Roughcast_Floorplan_Segmentation_CVPR_2023_paper.pdf
  - Code: no licence; NC training data.
  - **Download the paper.**
  - Why: once primitives are labelled, rooms become "classify the faces of the line arrangement". This is a learned, exact-geometry version of our free-space rooms for vector PDFs and DWGs.
- **Partitioning Open Plan Areas in Floor Plans**
  - Madugalla, Marriott, Marinai, 2017, ICDAR 2017, pp. 47–52, doi:10.1109/ICDAR.2017.17.
  - Not OA. Code: none.
  - Weak spot: open-plan rooms.
  - **Method content is unverified.** A search summary says it uses an obstacle-avoiding Voronoi; whether room labels seed it is unconfirmed. Metadata checked via Crossref.
  - Why: if labels are the seeds, it is the recipe for "two stamps in one free-space region → split".
- **Generation of Indoor Open Street Maps for Robot Navigation from CAD Files**
  - Zhang et al., 2025 (v3 March 2026), arXiv 2507.00552.
  - OA: yes.
  - Code: https://github.com/jiajiezhang7/osmAG-from-cad, no licence.
  - Why: DWG → rooms plus passages via Voronoi AreaGraph, with segmentation thresholds tied to door and corridor widths. Above 90% room and passage recall on 24 plans. A rule for splitting large halls; reimplement it.
- **Room segmentation: Survey, implementation, and analysis**
  - Bormann et al., 2016, ICRA 2016, doi:10.1109/ICRA.2016.7487234.
  - Paywalled.
  - Code: ipa_room_segmentation, "LGPL for academic and non-commercial use". **NC.**
  - Why: the standard comparison of morphological, distance-transform and Voronoi splitting of free space; baselines for open plans.
- **Semiautomatic detection of floor topology from CAD architectural drawings**
  - Domínguez, García, Feito, 2012, Computer-Aided Design 44(5):367–378, doi:10.1016/j.cad.2011.12.009.
  - Green OA: https://hdl.handle.net/10953/6402
  - Why: the classic Wall Adjacency Graph from parallel line pairs in CAD. A no-ML wall graph for clean DWGs and an auto-labeller for VectorGraphNET training.
- **Geometry-based symbol spotting in born-digital architectural floor plans**
  - Rezvanifar, Cote, Branzan Albu, 2021, J. Electronic Imaging 30(4) 043015, doi:10.1117/1.JEI.30.4.043015.
  - OA: not confirmed. Code: none.
  - Weak spot: door swings.
  - Why: voting on partial ellipses recovers door arcs with hinge and swing from vector or clean raster plans.
- **Evidence-gated multimodal parsing and vectorization of architectural floor plans** (SALI-FP)
  - Chen et al., 2026, arXiv 2609.25615.
  - OA: yes. Code: none; data governed.
  - Stage: QA.
  - Why: an automatic fix is accepted only if source ink supports it, under explicit thresholds, with logs. Reusable for our stage 9 or 10. Its use of external APIs is not.
- **Designing a Human-in-the-Loop System for Object Detection in Floor Plans**
  - Jakubik et al., 2022, AAAI 36(11):12524–12530 (IAAI), doi:10.1609/aaai.v36i11.21522.
  - OA: https://ojs.aaai.org/index.php/AAAI/article/view/21522
  - Why: a deployed floor-plan review UI that sorts detections by uncertainty, with up to +12.9% accuracy from selective expert input. Template for reviewing door swings, columns and stairs.
- **SAM 2: Segment Anything in Images and Videos**
  - Ravi et al., 2024, arXiv 2408.00714 (ICLR 2025 per search, not confirmed on arXiv).
  - Code and checkpoints: https://github.com/facebookresearch/sam2, Apache-2.0.
  - Why: one click per mis-segmented or open-plan room in the correction UI. It is weak on thin walls, so use it for regions only. The docs list SAM 3 (custom licence); SAM 2 is the licence-clean option.
  - Related: **Piciarelli et al.**, "Segmenting Anything in Architecture: A Tailored Approach to Segmenting Floor Plan Documents", ICIAP 2025, LNCS pp. 416–428, doi:10.1007/978-3-032-10185-3_33. Closed access; abstract seen only via search. It reports SAM beating semantic-segmentation baselines for rooms.
- **Semantics-based connectivity graph for indoor pathfinding powered by IFC-Graph**
  - Zhu et al., 2025, Automation in Construction 171:106019, doi:10.1016/j.autcon.2025.106019.
  - OA: Cambridge repository says CC BY 4.0; Crossref says CC BY-NC 4.0 (conflict).
  - Why: an IFC-aligned schema for typed space-to-space edges, including vertical links, for our connectivity JSON and checks against IFC.

### 4.4 Scale, Sheets and Multi-Drawing Layout

- **Towards a Robust Deep Learning-Based Scale Inference Approach in Construction Drawings**
  - Faltin, Schönfelder, König, 2024 (ASCE Computing in Civil Engineering 2023, pp. 721–728), doi:10.1061/9780784485248.087.
  - Not OA. Code: promised in the abstract, not found.
  - Why: the only paper found that is purely about inferring scale. Each detected dimension line plus its OCR value votes for px/mm; the vote picks the global scale. Abstract checked via OpenAlex.
- **A Hybrid Deep Learning and Rule-Based Method for Architectural Drawing Vectorization and CAD Reconstruction**
  - Lin & Wang, 2026, Buildings 16(5) 1043, doi:10.3390/buildings16051043.
  - OA: CC BY 4.0, https://www.mdpi.com/2075-5309/16/5/1043/pdf
  - Code: none.
  - Why: scanned historical architectural drawings → CAD, with axis-grid and dimension detection, scale recovery and constraint-based rectification. Axis grids give a second scale cue independent of dimension strings.
- **eDOCr** (Villena Toro et al., Frontiers in Manufacturing Technology 2023, doi:10.3389/fmtec.2023.1154132) and **eDOCr2** (Villena Toro & Tarkian, Machines 13(3):254, 2025, doi:10.3390/machines13030254)
  - **Known: repository [212]; papers not downloaded.**
  - OA: both CC BY 4.0.
  - Code: https://github.com/javvi51/edocr2 (MIT).
  - Why: dimension-text grouping plus a script for training synthetic custom recognisers. Train one for Swiss dimension notation (e.g. 3.62⁵). Do not use its optional GPT-4o step.
- **Benchmarking Deep Learning Approaches for AEC Engineering Drawing Layout Detection and Information Extraction**
  - Huang et al., 2026, EC3 2026, arXiv 2607.18997.
  - OA: yes. Code: none.
  - Weak spot: multi-drawing sheets.
  - Why: RF-DETR mAP50 0.949 for sheet layout (views, title block, legend), and generic document-layout models degrade on drawings. Fine-tune RF-DETR (Apache-2.0, already on our list) on our own sheets.
- **AI-based extraction and management of text and view information from 2D bridge engineering drawings**
  - Peng et al., 2025, EG-ICE 2025, doi:10.17868/strath.00093312.
  - OA: CC BY, https://strathprints.strath.ac.uk/93312/
  - Why: detects views, reads view titles and matches title to view. The template for splitting "Erdgeschoss / Schnitt A-A" sheets and routing each view.
  - Related, lower priority:
    - Khan et al., ICIEA 2026, arXiv 2510.21862: multi-view drawings; YOLOv11 is AGPL, so swap the detector.
    - Lombardi et al., EC3 2025, arXiv 2504.08645: title-block detection; replace its GPT-4o step with a local VLM.
- **ICDAR 2021 Competition on Historical Map Segmentation** (MapSeg)
  - Chazalon et al., 2021, arXiv 2105.13265.
  - Data: Zenodo 10.5281/zenodo.4817662, CC BY 4.0.
  - Evaluation code: https://github.com/icdar21-mapseg/icdar21-mapseg-eval (MIT).
  - Why: reuse the MIT panoptic-quality evaluator for room polygons. Task 2 (separating map content from frame, legend and title area) is the same problem as masking title blocks and legends.

### 4.5 Text: Room Stamps and Labels

- **PP-OCRv6: From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR Tasks** (**known as a tool, [99]; report not downloaded**)
  - Zhang et al., 2026, arXiv 2606.13108.
  - Weights: Apache-2.0 on Hugging Face.
  - Why: the planned upgrade from v5. Reported +5.1 points recognition and +4.6 points detection Hmean over PP-OCRv5_server. Still CTC, so compatible with Word Beam Search. **Unverified:** DE/FR/IT accents and "²" in its dictionary.
- **LIGHT: Multi-Modal Text Linking on Historical Maps**
  - Lin et al., 2025, ICDAR 2025 (LNCS pp. 60–77), arXiv 2506.22589.
  - OA: yes.
  - Code: https://github.com/knowledge-computing/multimodal-text-linking, no licence; built on LayoutLMv3 (CC BY-NC-SA weights).
  - Why: predicts which text fragments belong to one label from geometry, vision and language, i.e. "which words form one stamp". Reimplement with LiLT (MIT, arXiv 2202.13669) trained on synthetic stamps.
- **ICDAR 2024 and 2025 Competitions on Historical Map Text Detection, Recognition, and Linking**
  - Li et al. 2024, doi:10.1007/978-3-031-70552-6_22; Lin et al. 2025, doi:10.1007/978-3-032-04630-7_33.
  - Data:
    - IGN French land-register sheets (19th-century, hand-lettered): **CC BY-SA 4.0**.
    - Synthetic sets: **CC BY 4.0**.
    - Rumsey: CC BY-NC-SA.
  - Why: detection, recognition and linking metrics for stamp QA, plus permissive near-domain pretraining data for rotated, hand-lettered labels.
- **Hyper-Local Deformable Transformers for Text Spotting on Historical Maps** (PALETTE, SynthMap+)
  - Lin & Chiang, 2024, KDD 2024, arXiv 2506.15010.
  - OA: yes. Weights are not open (commercial licence on request).
  - Why: the SynthMap+ recipe (render vectors with styled, rotated, spaced labels) applied to our renders also gives ground truth for stamp-to-room assignment.
- **The Adaptability of a Transformer-Based OCR Model for Historical Documents**
  - Ströbel et al., 2023, ICDAR 2023 Workshops (LNCS 14193 pp. 34–48), doi:10.1007/978-3-031-41498-5_3. Companion abstract: arXiv 2203.11008.
  - Model: https://huggingface.co/dh-unibe/trocr-kurrent (MIT, about 2.7% CER on 19th-century Kurrent; base model fine-tuned on IAM, whose terms are NC and were not re-checked).
  - Why: a second-stage reader for low-confidence, Kurrent-era stamps; from a Swiss team.
- **How to Choose Pretrained Handwriting Recognition Models for Single Writer Fine-Tuning**
  - Pippi et al., 2023, ICDAR 2023, arXiv 2305.02593.
  - OA: yes. Code: none listed.
  - Why: usable single-writer HTR from about 5 real lines. One draftsman letters a plan set, so fine-tune the stamp reader per archive batch.
- **Evaluation of HTR models without Ground Truth Material**
  - Ströbel et al., 2022, LREC 2022, ACL Anthology 2022.lrec-1.467, arXiv 2201.06170.
  - OA: yes.
  - Why: lexicon-based and masked-language-model scores without ground truth. Lexicon hit rate per stamp and per batch is an unsupervised QA and model-selection signal for us.
- **Text Recognition and Classification in Floor Plan Images**
  - Ravagli, Ziran, Marinai, 2019, ICDAR Workshops (GREC 2019), doi:10.1109/ICDARW.2019.00006.
  - Institutional copy (fetch refused).
  - Why: the closest direct prior art for classifying text roles in floor plans (room name, area, dimension and so on). A baseline for our role classifier, which picks area tokens and must not pick dimensions. Metadata checked via Crossref; the class list was not read.
- **General Models for Handwritten Text Recognition: Feasibility and State-of-the Art. German Kurrent as an Example**
  - Hodel et al., 2021, J. Open Humanities Data 7:13, doi:10.5334/johd.46.
  - OA: CC BY.
  - Data: test set "Minutes of the Swiss Federal Council (1848–1903)", Zenodo 10.5281/zenodo.4746341, CC BY 4.0.
  - Why: a permissive Swiss federal Kurrent benchmark for checking any hand-lettering reader.

### 4.6 Active Learning and Few-Shot Adaptation

- **Best Practices in Active Learning for Semantic Segmentation**
  - Mittal et al., 2023, GCPR 2023, arXiv 2302.04075.
  - OA: yes.
  - Why: at low budgets, random selection plus semi-supervised training often matches active learning. Diversity methods win on redundant data, such as many near-identical sheets from one series. Benchmark against random selection before building active-learning tooling.
- **Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets** (TypiClust)
  - Hacohen et al., 2022, ICML 2022, arXiv 2202.02794.
  - Code: https://github.com/avihu111/TypiClust (MIT).
  - Why: embed all archive sheets with DINOv2, cluster them, and label the most typical sheet per cluster first.
- **Active Label Correction for Semantic Segmentation with Foundation Models**
  - Kim et al., 2024, ICML 2024 (PMLR 235), arXiv 2403.10820. Successor: A²LC, Jeon et al., AAAI 2026, arXiv 2506.11599.
  - OA: yes. Code: none found.
  - Why: reviewers correct superpixels of pseudo-labels chosen by an acquisition function, so QA time becomes training labels.
- **MapSAM2: Adapting SAM2 for Automatic Segmentation of Historical Map Images and Time Series**
  - Xia et al., 2025, arXiv 2510.27547. Predecessor MapSAM: arXiv 2411.06971.
  - Code: https://github.com/Xue-Xia/MapSAM2, Apache-2.0 per the repo page. It acknowledges YOLO code, so check the time-series part for AGPL files.
  - Why: a 10-shot SAM2 + LoRA baseline on Swiss Siegfried line drawings (ETH). Neighbouring tiles are treated as "video" to share context across large sheets.
- **Few-Shot Segmentation of Historical Maps via Linear Probing of Vision Foundation Models**
  - Sterzinger et al., 2025, ICDAR 2025, arXiv 2506.21826.
  - Paper CC BY-NC-SA. Repo has no licence. The RADIO backbone licence is unverified.
  - Why: a per-archive linear-probe or DoRA fine-tune with about 0.7 M parameters. Reimplement it on DINOv2.

### 4.7 Evaluation Metrics

- **Boundary IoU: Improving Object-Centric Image Segmentation Evaluation**
  - Cheng et al., 2021, CVPR 2021, arXiv 2103.16562.
  - Code: https://github.com/bowenc0221/boundary-iou-api (BSD-2; bundles panopticapi).
  - Why: report boundary IoU and boundary PQ for walls and rooms. Pixel IoU 0.57 hides boundary quality, and the review asks for an evaluation harness first.
- **Panoptic Segmentation** (Panoptic Quality, PQ)
  - Kirillov et al., 2019, CVPR 2019, arXiv 1801.00868.
  - Why: rooms as instances. PQ = SQ × RQ captures merged and split rooms in one number. Wall can be "stuff".
- **Polygon metrics:**
  - PoLiS: Avbelj et al., IEEE GRSL 12(1) 2015, doi:10.1109/LGRS.2014.2330695.
  - C-IoU and N-ratio: from PolyWorld, Zorzi et al., CVPR 2022, arXiv 2111.15491. PolyWorld code is research-only; use the definitions only.
  - Max tangent angle (MTA): Girard 2021.
  - Why: catch over- or under-vertexed DXF polygons that IoU does not see.
- **Connectivity metrics:**
  - APLS: SpaceNet, Van Etten et al., 2018, arXiv 1807.01232; code https://github.com/CosmiQ/apls (Apache-2.0).
  - Graph edit distance as used in House-GAN: Nauata et al., ECCV 2020, arXiv 2003.06988.
  - Why: compare circulation graphs (room centroids plus door midpoints) by path length. A missed door or a false passage shows up directly.
- **Strict topology metrics:**
  - Betti matching: Stucki et al., ICML 2023, PMLR 202, arXiv 2211.15272; code https://github.com/nstucki/Betti-matching (MIT).
  - Topograph: Lux et al., ICLR 2025, arXiv 2411.03228; only an anonymous review repository.
  - Why: on a wall mask, Betti-1 counts enclosed regions. A leak lowers it and hatching loops raise it, which makes it a cheap regression metric. Use these as metrics, not losses.

## 5. Known Papers Worth Downloading

These are already cited in `sources.md` (or present as tools in the docs) but have no local copy:

| Source | Paper | Download? | Reason |
|---|---|---|---|
| [176] | VectorFloorSeg (CVPR 2023) | Yes | Room faces from line primitives (§4.3); open access at CVF |
| [137] | Trivi, "Archival analog drawings for semantic segmentation of Roman Architectural Heritage using Deep Learning", ISPRS Archives XLVIII-2-W12-2026, 487–494, doi:10.5194/isprs-archives-XLVIII-2-W12-2026-487-2026 (CC BY 4.0) | Yes | Closest domain match: same U-Net/ResNet-34 setup, archival hatch and texture codes, only 19 annotated drawings. Note: the datasets doc calls them "Sapienza archive drawings"; the paper title is now confirmed |
| [113] | Versailles-FP (arXiv 2103.08064) | Yes | Wall detection in historical plans; reference for the historical style test |
| [212] | eDOCr2 paper (Machines 2025) | Yes | Dimension-text reading and synthetic recogniser training (§4.4) |
| [99] | PP-OCRv6 report (arXiv 2606.13108) | Yes | Planned OCR upgrade (§4.5) |
| [209] | PARSeq (ECCV 2022, arXiv 2207.06966) | Optional | Includes a 90°/180°/270° rotation-robustness benchmark relevant to rotated stamps |
| [134] | Text detection on technical drawings (arXiv 2205.02659) | Optional | Synthetic generator pattern; superseded for us by SynthMap+ ideas |
| [171], [172] | HAWP, DeepLSD | No | ScaleLSD supersedes them for this use |
| [144], [175] | HEAT, Floor-SP | No | Point-cloud density-map methods |
| Docs (tool) | Augraphy paper (Groleau et al., arXiv 2208.14558; ICDAR 2023 per search, venue unconfirmed) | Optional | Citation for a tool already planned |

## 6. Skip or Known Tier

- **Pizarro et al. 2022**, "Automatic floor plan analysis and recognition", Autom. Constr. 140:104348: a 1995–2021 review; background only.
- **Rezvanifar et al.**, IPSJ TCVA 2019 survey and CVPRW 2020 (arXiv 2006.00684): background and a tiling recipe only.
- **Xu et al. 2024, ArchNetv2** (Autom. Constr. 165:105486): multiscale detection with no code; covered by the RF-DETR plan.
- **De Nardin et al. 2025** (CACAIE, doi:10.1111/mice.70030): residential room types, no code.
- **Chen & Tu 2025** (Results in Engineering 26:105575): closed-face extraction from a CAD layer, equivalent to existing rules.
- **Xing et al. 2025**, "Comprehensive floor plan vectorization with sparse point set representation" (Autom. Constr. 173:106023): closed access; content not verifiable.
- **Kratochvila et al. 2024** (arXiv 2408.01526; Apache-2.0 code): multi-unit reconstruction, no sheet handling; its scale step cites Lv 2021.
- **David & Leitão 2023** (door classification, APJ 28(3)): swing and hinge classes known only from a secondary source.
- **LayoutGKN** (van Engelenburg et al., BMVC 2025, arXiv 2509.03737): graph similarity on RPLAN; no licence; possible later for SAP reconciliation.
- **Blueprint-Bench** (arXiv 2509.25229): rooms matched by size rank, which is crude.
- **MMFE** (arXiv 2609.12723) and **FGSSNet** (arXiv 2507.10343): low relevance or no code.
- **PolyDiffuse, SLIBO-Net, CAGE, FloorSAM:** point-cloud density maps, not drawings.
- **DoorDet** (arXiv 2508.07714) and **MURF:** derived from CubiCasa5K (NC), or unreleased.
- **Pix2Poly** (WACV 2025, MIT): aerial sequence model; frame fields or HiSup fit better.
- **TopoLoss** (NeurIPS 2019) and **DMT loss** (ICLR 2021): expensive; superseded by clDice and Skeleton Recall.
- **PolyWorld code:** research-only licence. Keep its metrics only.
- **Curve-GCN and Polygon-RNN++:** GPL-3.0 code, plus an active University of Toronto patent (US10643130B2) on interactive polygon annotation.
- **dhSegment:** GPL-3.0 and superseded. **Moreno-García et al. 2019:** a P&ID-focused review.
- **DocCreator** (LGPL), **ShabbyPages**, **Synthesis in Style** (GPL), **FDA** (no licence, fixes tone only): overlap with Augraphy, or licence issues.
- **DUDA** (arXiv 2504.09814): the right idea (distil a UDA teacher into a small student), but no code.
- **CHURRO** (EMNLP 2025): weights under the Qwen Research licence (NC). Benchmark use only.
- **GOT-OCR2.0:** conflicting code and data licences.
- **mapKurator:** CC BY-NC code.
- **DTrOCR:** no official code.
- **MinerU2.5:** AGPL weights.
- **dots.ocr:** no paper and a custom licence.
- **DeepSeek-OCR and olmOCR 2:** permissive, but page parsers rather than stamp readers.
- **LayoutLMv3:** NC weights.
- **Zhang & Zhang 2025** (few-shot SAM plus GPT-4): external API; weak novelty.
- **"What Makes Synthetic Data Effective in Image Segmentation"** (ICML 2026), **MFuser**, **DepthForge:** natural-scene priors.
- **SlugTrails dataset** (arXiv 2609.19876): CC BY 4.0 campus plans announced but not released. Watch it as a small non-residential test set.
- **New permissive datasets for 2025–2026:** none found that are released, permissive, and non-residential, historical or vector-PDF. Everything else found was NC, unreleased or residential.

## 7. Leads Not Verified (Excluded from the Ranking)

- Wu, Schindler, Heitzler, Hurni 2023, ISPRS J., domain adaptation for segmenting historical maps: known from a search snippet only; the ETH page failed.
- EC3 2026, "Open-Source VLMs for Engineering Drawing Metadata: Qwen-VL title-block detection": page behind a bot check.
- de las Heras et al., "Unsupervised Wall Detector" (ICDAR 2013) and "Notation-Invariant Patch-Based Wall Detector": classic hatched-wall handling, not checked.
- Al-Wesabi et al. 2023 (old scanned drawings of existing buildings) and Schönfelder et al. 2025 (historical development plans): metadata only.
- Weinman et al., ICDAR 2019: lexicon in beam search for map text; claim from a search snippet only.
- Meng et al. 2025 (Knowledge-Based Systems): scale-bar detection in microscopy; metadata only.
- Cai et al., CVPR 2021, superpixel active learning: seen via search only.
- No peer-reviewed paper on door-width priors for scale was found. ResPlan's area anchor and the pilot's own door-width cue remain the references.

## 8. Weak Spot → Paper Map

| Weak spot (pilot review) | First reads | Then |
|---|---|---|
| Hatching or ornament read as walls | Petitpierre 2026 (#2), MIC (#1) | Kim 2021 normaliser, Schönfelder 2026 wall-fill class, manga line extraction, boundary loss |
| Thin lines, gaps, leaking or merged rooms | TopoMortar (#4), HRDA (#3), Chen 2024 watershed (#10) | clDice, Skeleton Recall, region separation, orientation head |
| Columns | HRDA rare-class sampling (#3), Talebi-Kalaleh glyph rules (#9) | (raster column detection not covered; gap) |
| Open-plan rooms | Raster2Seq (#5), stamp-seeded watershed (#10) | Madugalla 2017, osmAG from CAD, Bormann 2016, SAM 2 click correction |
| Hand and calligraphic lettering | Word Beam Search (#7) | PP-OCRv6, trocr-kurrent, Pippi single-writer fine-tuning, MapText data |
| Scale without a known scale | Lv 2021 (#8), Talebi-Kalaleh (#9) | Faltin 2024, Lin & Wang 2026 axis grid, eDOCr2 |
| Door instances and swings | Raster2Seq door polygons (#5) | Rezvanifar 2021 arc voting, Jakubik 2022 review UI |
| Multi-drawing sheets | – | Huang 2026 layout benchmark, Peng 2025 view titles, MapSeg task 2 |
| Vector PDFs | VectorGraphNET (#6), Talebi-Kalaleh (#9) | VectorFloorSeg [176], Domínguez 2012 |
| Small labelled sets and HITL | Petitpierre & Guhennec (#11) | UniMatch, Mittal 2023, TypiClust, active label correction, MapSAM2 |
| Evaluation harness | Chen 2024 PQ (#10) | Boundary IoU, PoLiS / C-IoU / MTA, APLS, Betti matching |
