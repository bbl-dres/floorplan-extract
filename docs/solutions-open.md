# Open-Source Solutions for Floor Plan Extraction

*Research catalogue · 7 October 2026 · Open-source implementations relevant to the pilot pipeline: code released with papers, pretrained weights, Hugging Face models and Spaces, and general building blocks, with verified code licences, the provenance of the weights and recommended use for BBL. Commercial products and ready-to-use tools are in [closed solutions](solutions-closed.md), methods and paper findings in the [literature review and state of the art](literature-review.md), training data in [open datasets](datasets-open.md). Bracketed numbers refer to the shared [source list](../research/sources.md).*

## 1. Scope and Method

**Corpus.** About 120 implementations in seven groups: raster segmentation, vectorisation and graph reconstruction, vector CAD symbol spotting, multimodal parsers and benchmarks, Hugging Face models and Spaces, OCR for text in drawings, and general frameworks. They come from:
- the code links in the 34 papers of the [research index](../research/README.md) and on their Hugging Face paper pages;
- GitHub repository search for each paper's name and for floor plan recognition, segmentation, vectorisation, parsing, symbol spotting, VLM and OCR topics;
- the Hugging Face Hub API: models, datasets and Spaces searched for floorplan, floor plan, floor-plan, floor_plan, blueprint, architectural, cubicasa, floorplancad, deepfloorplan, wall and room segmentation, symbol spotting, CAD drawing, house and building plan, and the method names (WAFFLE, ArchCAD, SymPoint, VecFormer, RoomFormer, PolyRoom, FloorplanVLM, ResPlan, MLSTRUCT, MSD)\[141\];
- Zenodo software records, GitLab project search and Roboflow Universe. Papers with Code was shut down on 24 July 2025 and redirects to Hugging Face Trending Papers\[142\]; its archive mirrors did not load.

**Method.** For each implementation we recorded task, input and output, code licence, weights and their training data, last activity, limitations and relevance. Code licences come from the LICENSE file, read through the GitHub REST API (`license.spdx_id`) and, where GitHub reports "NOASSERTION" or no licence, from the raw LICENSE file and README. Model licences come from the Hugging Face API (`cardData.license`) and the model card; base-model licences were checked the same way. Training data is as stated in the README or model card. Activity is the last push to the repository (GitHub `pushed_at`) or the last modification on Hugging Face, as of 7 October 2026. Nothing was downloaded, cloned or run. "Not verified" marks what could not be checked.

**Conventions.**
- Tables use Implementation (linked) | Task | Input → Output | Code licence | Weights and training data | Activity | Limitations | Relevance.
- Licence verdicts as in [open datasets](datasets-open.md#1-scope-and-method): P = permissive (MIT, Apache-2.0, BSD); SA = copyleft (GPL-3.0, AGPL-3.0; MPL-2.0 at file level); NC = non-commercial; R = research or academic use only; U = unclear, conflicting or no licence file. Without a licence file all rights are reserved: the code can be read, not reused.
- Activity: active = push in the last 6 months; low = 6–24 months; inactive = more than 24 months; archived as flagged by GitHub.
- Weights trained on NC or R data are research-only for BBL, whatever licence tag the weights carry (§6).
- Relevance: High = worth trying in the pilot now as a baseline or component: permissive code, self-hostable, and either licence-clean weights or trainable on BBL's own renders. Medium = useful reference, comparator or research baseline that needs a licence review, retraining or major adaptation. Low = wrong domain, no code, superseded, cloud-only, or the licence rules out use.

## 2. Key Findings

1. **No licence-clean pretrained floor plan model exists.** None of the roughly 40 weight releases found for walls, rooms, openings or symbols is trained on permissively licensed data. They use CubiCasa5K, FloorPlanCAD, ArchCAD or Structured3D (all NC), R2V/LIFULL or R3D (research terms), or undisclosed data. The reusable value is in code, not weights: training on Swiss Dwellings renders, as pilot v2 does, remains necessary.
2. **Several code licences are stricter than catalogued so far.**
   - CubiCasa5K: the CC BY-NC 4.0 LICENSE covers the whole repository, model code included\[5\].
   - SymPoint: an IDEA "License for Non-commercial Scientific Research Purposes" that also prohibits military and surveillance use\[143\].
   - DPSS (ArchCAD): "ACADEMIC USE LICENSE … Commercial use is strictly prohibited"\[26\]; HEAT: not for commercial use\[144\]; FP4S: CC BY-NC-SA 4.0\[145\].
   - DeepFloorplan and Raster-to-Graph are GPL-3.0\[36\]\[40\].
   - CADSpotting's code (2026) has no licence and builds on OneFormer3D, which is CC BY-NC 4.0\[146\]\[147\].
3. **Permissive code exists for most building blocks, but without usable weights.** Raster-to-Vector (MIT; its PyTorch port solves the integer programme with the open PuLP library), RoomFormer and Floor-SP (MIT), CADTransformer (MIT), VecFormer (Apache-2.0, no weights released), CubiCasa5k-Next (Apache-2.0 clean-room reimplementation, September 2026), HAWP and DeepLSD (MIT; DeepLSD weights also MIT), the ResPlan utilities (MIT) and Deep Vectorization (MPL-2.0).
4. **At least ten methods in the index have no public code**: MuraNet, Chen et al. 2023 (GLSP), Dodge et al., GAT-CADNet (an unofficial reimplementation exists), Pang et al. (dataset only), TextCAD, PolarSym, Sketch2BIM, Ayanzadeh & Oates, and FloorplanVLM. FloorplanVLM's FPBench-2K is announced as open source but was not found. CADSpotting's code appeared in 2026; LS-CAD is still unreleased.
5. **Hugging Face offers demos, not foundations.** Fewer than 50 of several hundred hits address floor plan recognition, mostly personal fine-tunes. The most downloaded, a CubiCasa5K wall segmenter on a non-commercial base model, has 1,856 downloads a month. Licence tags are unreliable:
   - MIT, Apache-2.0 or CC BY tags on weights trained on CubiCasa5K or FloorPlanCAD;
   - fine-tunes of non-commercial base models: Qwen2.5-VL-3B (Qwen Research licence), Bria RMBG-1.4, NVIDIA SegFormer;
   - Apache-2.0 tags on Ultralytics YOLO checkpoints (framework AGPL-3.0).
   A few cards label the restriction correctly (BDivyesh, v1nz, mjlading, OsamaMo).
6. **The most-liked Spaces are thin clients to cloud services.** RasterScan's Space (57 likes) posts images to the vendor's backend, whose Docker image needs licence activation\[148\]\[149\]. Viraj2307's Space calls Roboflow's hosted API\[150\]. Neither is a self-hostable open model.
7. **Every public FloorplanVLM reproduction is non-commercial.** Two training codebases without a licence and several LoRA adapters fine-tune Qwen2.5-VL-3B, itself under the Qwen Research licence\[151\], on CubiCasa5K\[43\]\[152\]\[153\]. A licence-clean variant means writing the paper's SFT + GRPO recipe anew, with an Apache-2.0 base (Qwen2.5-VL-7B or Qwen3-VL) and Swiss Dwellings renders.
8. **Maintenance is weak.** Most research repositories had no push in the last six months, and many pin old stacks: Python 2.7 with TensorFlow 1.10 (DeepFloorplan), Lua Torch7 (Raster-to-Vector), PyTorch 1.0 (CubiCasa5K), PyTorch 1.9–1.10 with CUDA 11.1 (FRI-Net, SymPoint-V2). The MLSTRUCT repositories and Mask2Former are archived.
9. **The pilot's own stack is licence-clean, with two cautions.** segmentation_models_pytorch (MIT), RapidOCR (Apache-2.0; models converted from PaddleOCR under Apache-2.0\[154\]), Shapely and networkx are fine. However:
   - smp's Mix Vision Transformer (SegFormer) encoders are "for non-commercial use only"\[155\]; avoid `mit_b*`.
   - ImageNet-pretrained encoders, such as the pilot's ResNet-34, rest on images whose terms of access allow use "only for non-commercial research and educational purposes"\[156\]. Whether this extends to the weights is unsettled; include it in the legal review.
10. **Room connectivity and room stamps have almost no open code.** One Zenodo release (August 2026) recovers door, cased-opening and shared-wall relations between rooms from raster plans. Its code is CC BY 4.0, but its weights are trained on CubiCasa5K\[157\]. No open implementation parses multi-field room stamps. The reusable parts are generic OCR, synthetic text generators (TRDG, SynthTIGER; MIT) and recognisers to fine-tune (PARSeq; Apache-2.0).

## 3. Recommended Use

| Use | Implementations | Licence | Notes |
|---|---|---|---|
| Baseline in the pilot | [nnU-Net](https://github.com/MIC-DKFZ/nnUNet); [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) (in use) | Apache-2.0; MIT | nnU-Net as a self-configuring comparator for the U-Net on the same renders; in smp use CNN encoders, not `mit_b*` |
| Component: openings, columns, stairs, stamps, sheet layout | [RF-DETR](https://github.com/roboflow/rf-detr) N/S/M/L and Seg; [D-FINE](https://github.com/Peterande/D-FINE); [RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Apache-2.0 | Train on own renders; not RF-DETR XL, 2XL or A/F/P (PML 1.0) |
| Component: typed wall junctions | [CubiCasa5k-Next](https://github.com/Lqm1/CubiCasa5k-Next) | Apache-2.0 | Train on Swiss Dwellings renders; new single-author code, validate first |
| Component: wall-line and junction candidates | [HAWP](https://github.com/cherubicXN/hawp); [DeepLSD](https://github.com/cvg/DeepLSD) | MIT | HAWP retrained on renders as an open stand-in for GLSP; DeepLSD for line candidates |
| Component: constraint layer | [Raster-to-Vector](https://github.com/art-programmer/FloorplanTransformation) integer programme (PyTorch port) | MIT | Reuse the formulation with an open solver (PuLP/CBC); not its weights |
| Component: room polygon fallback | [RoomFormer](https://github.com/ywyue/RoomFormer) | MIT | Retrain on wall and room masks; capped at 20 rooms |
| Component: room-stamp OCR | [RapidOCR](https://github.com/RapidAI/RapidOCR) (in use); [OnnxTR](https://github.com/felixdittrich92/OnnxTR) or [EasyOCR](https://github.com/JaidedAI/EasyOCR) as second engine; [PARSeq](https://github.com/baudm/parseq) with [TRDG](https://github.com/Belval/TextRecognitionDataGenerator) or [SynthTIGER](https://github.com/clovaai/synthtiger) for fine-tuning | Apache-2.0; MIT | Synthetic DE/FR/IT stamps with the OFL fonts in `data/public/fonts/` |
| Component: graph QA and connectivity | [ResPlan utilities](https://github.com/m-agour/ResPlan) | MIT (code) | Edge schema and QA rules for [pipeline stages 8–9](pipeline.md) |
| Optional DWG route | [VecFormer](https://github.com/WesKwong/VecFormer) | Apache-2.0 | Train on compliant BBL DWGs; no public weights |
| Gold set and review | [CVAT](https://github.com/cvat-ai/cvat); [Label Studio](https://github.com/HumanSignal/label-studio) | MIT; Apache-2.0 | Self-hosted annotation of BBL sheets |
| VLM evaluation | [FloorplanQA](https://github.com/OldDeLorean/FloorplanQA); [AEC-Bench](https://github.com/nomic-ai/aec-bench); AECV-Bench protocol | MIT; Apache-2.0; none | Run against self-hosted VLMs; reimplement AECV-Bench prompts, its code has no licence |
| Research comparators only | CubiCasa5K model, TF2DeepFloorplan, Raster-to-Graph, SymPoint, DPSS, WAFFLE layout detector, Zenodo connectivity models, Hugging Face fine-tunes on CubiCasa5K or FloorPlanCAD | NC, R, GPL-3.0 or none | Internal benchmarks, never production weights; legal check first ([open datasets §7](datasets-open.md#7-limitations-of-this-catalogue)) |
| Read, then reimplement | FloorplanVLM reimplementations, MSD graph code, VecFloorSeg, SymPoint-V2, PolyRoom, FRI-Net | No licence | Method reference only |
| Avoid | Ultralytics-based checkpoints (also when tagged Apache-2.0), RasterScan, Roboflow-hosted models, RMBG-1.4- and SegFormer-based weights, Qwen2.5-VL-3B adapters, ArchCAD-gpu | AGPL-3.0, closed, cloud, NC, academic | – |

## 4. Implementations by Category

### 4.1 Raster Wall, Room and Opening Segmentation

Permissive code in this group is useful only for training on BBL's own renders; every released weight is research-only. Pilot v2 already uses the architecture most of these repositories use (U-Net with a ResNet-34 encoder).

| Implementation | Task | Input → Output | Code licence | Weights and training data | Activity | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [DeepFloorplan](https://github.com/zlzeng/DeepFloorplan) (ICCV 2019) | Multi-task room-boundary and room-type segmentation | Raster → wall, opening and room-type masks | GPL-3.0\[36\] – SA | Yes (SharePoint); R2V (LIFULL) and R3D, research terms – R | Last push 2024-02; inactive | Python 2.7, TensorFlow 1.10; pixel masks only | Low (superseded, old stack) |
| [TF2DeepFloorplan](https://github.com/zcemycl/TF2DeepFloorplan) | TensorFlow 2 rewrite with TFLite deployment | Raster → masks | GPL-3.0\[37\] – SA | Yes (Google Drive); R3D only – R | 2023-07; inactive | Copyleft on distribution; residential | Medium (internal comparator only) |
| [PyTorch-DeepFloorplan](https://github.com/zcemycl/PyTorch-DeepFloorplan) | Minimal PyTorch replication | Raster → masks | MIT\[158\] – P | None found; R3D training script | 2024-10; low | No results or documentation | Low |
| [CubiCasa5K model](https://github.com/CubiCasa/CubiCasa5k) | Multi-task hourglass: junction heatmaps, room and icon segmentation, polygonisation | Raster → heatmaps, masks, polygons | CC BY-NC 4.0 for the whole repository\[5\] – NC | Yes (Google Drive); CubiCasa5K – NC\[38\] | 2026-02; low | Python 3.6, PyTorch 1.0; Finnish residential | Medium (benchmark comparator) |
| [CubiCasa5k-Next](https://github.com/Lqm1/CubiCasa5k-Next) | Clean-room reimplementation of the CubiCasa5K model "derived solely from the papers": training, junction NMS, rooms, icons, openings, ONNX export | Raster → heatmaps, masks, polygons | Apache-2.0\[159\] – P | None released | Created 2026-09-15; active; single author | Unproven; clean-room claim is the author's | High (only permissive typed-junction model; train on Swiss Dwellings renders) |
| [MLStructFP benchmarks](https://github.com/MLSTRUCT/MLStructFP_benchmarks) | Wall segmentation and wall vectorisation baselines | Raster crops → wall masks, vectors | MIT\[160\] – P | Yes (Google Drive); MLSTRUCT-FP, dataset terms not stated\[109\] – U | Archived (last push 2026-03) | Walls only | Medium (high-resolution wall baseline; data terms open) |
| [MitUNet](https://github.com/aliasstudio/mitunet) (2025) | Wall segmentation with a Mix-Transformer encoder and U-Net decoder | Raster → wall mask | MIT\[161\] – P | Not released (local path in README); pre-trained on CubiCasa5K (NC), fine-tuned on Floor Plan CIS (500 Russian and CIS listing images, CC BY 4.0)\[162\]; MiT encoder under NVIDIA's non-commercial licence\[155\] | 2026-04; low | Three licence restrictions in the chain | Low |
| [floorplan-to-3d](https://github.com/Yytsi/floorplan-to-3d) | U-Net/ResNet-34 floor, wall, door, window segmentation; polygon extraction; API; 3D viewer | Raster → masks, polygons | MIT\[163\] – P | Yes (Hugging Face, MIT tag); CubiCasa5K – NC\[45\] | 2026-05; active | Residential | Low (same architecture as pilot v2, trained on NC data) |
| [OpenBIM-FloorPlan-AI](https://github.com/Chunling1/OpenBIM-FloorPlan-AI) | Wall, window, door segmentation (U-Net, DeepLabV3+); annotation tool; IFC export | Raster → masks, IFC | MIT\[164\] – P | Yes (GitHub release v1.0.0, three checkpoints); CubiCasa5K – NC | 2026-05; active | Self-reported 0.787 mIoU on CubiCasa5K | Low (code reference for IFC export) |
| [floor-plan-room-segmentation](https://github.com/ozturkoktay/floor-plan-room-segmentation) | U-Net with ResNet encoder for rooms, walls, doors, windows | Raster → masks | MIT\[165\] – P | None; dataset not named | 2025-02; low | Tutorial level | Low |
| [deep-floor-plan-recognition](https://github.com/whchien/deep-floor-plan-recognition) | DeepFloorplan derivative | Raster → masks | GPL-3.0\[166\] – SA | Yes (Google Drive); lineage R2V/R3D, not verified | 2024-07; inactive | – | Low |
| [FP4S](https://github.com/JanineCHEN/FP4S) | Scribble-based semi-weakly-supervised segmentation | Raster + scribbles → masks | CC BY-NC-SA 4.0, "for NonCommercial use only"\[145\] – NC | Not verified | 2024-09; inactive | NC code | Low (idea: scribbles for cheap gold-set labels) |
| [FullScenarioFloorplanParsing](https://github.com/dididi0924/FullScenarioFloorplanParsing) | Pre- and post-processing around a wall-mask model | Raster → wall mask | MIT\[167\] – P | ONNX model on a cloud bucket; training data not stated – U | 2025-10; low | Placeholder citation ("Author, A."), no evaluation | Low |
| [RasterScan Floor-Plan-Recognition](https://github.com/RasterScan/Floor-Plan-Recognition) | Raster-to-vector service citing Raster-to-Vector | Raster → vectors (HTTP API) | No licence file; Docker image activated per machine for a "lifetime license"\[148\] – U | Closed, inside the image | 2025-06; low | Not open source | Low |
| [FloorplanToBlender3d](https://github.com/grebtsew/FloorplanToBlender3d) | Classical OpenCV wall and room detection to Blender | Raster → 3D mesh | GPL-3.0\[168\] – SA | No learning | 2024-10; low | Simple drawings only | Low |
| MuraNet\[41\], Dodge et al. 2017 | Joint wall segmentation and opening detection; FCN walls with OCR-based scale | Raster → masks, boxes | No public code found (paper text, GitHub search, Hugging Face paper page) | – | – | – | Low |

### 4.2 Vectorisation, Line and Graph Reconstruction

| Implementation | Task | Input → Output | Code licence | Weights and training data | Activity | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [Raster-to-Vector](https://github.com/art-programmer/FloorplanTransformation) (FloorplanTransformation, ICCV 2017) | Junction heatmaps + integer programming into a wall, door and room graph | Raster → vector plan | MIT\[169\] – P | Yes (Google Drive, Torch7 and PyTorch); LIFULL images – R\[107\]; vector annotations shared | 2022-06; inactive | Torch7 original; PyTorch port "not well-tested", IP via PuLP or Gurobi; Manhattan, 256 px | Medium (reuse the IP formulation with an open solver) |
| [Raster-to-Graph](https://github.com/SizheHu/Raster-to-Graph) (EG 2024) | Autoregressive structural-graph prediction | Raster → wall graph with room semantics | GPL-3.0\[40\] – SA | Yes (Google Drive); >10,000 LIFULL plans, data by request – R | 2026-05; active | Residential, Manhattan, ≤50 nodes | Medium (internal comparator) |
| Chen et al. 2023 (GLSP) | Junction heatmap + GNN classification of line segments | Raster → typed line segments | No public code found | – | – | – | Low (method only; see HAWP) |
| FloorplanVLM\[44\] (official) | Fine-tuned VLM emitting walls, openings and rooms as JSON | Raster → JSON vectors | No official code, weights or FPBench-2K found | Proprietary Floorplan-2M / HQ-300K | – | – | Low (until released) |
| FloorplanVLM reimplementations: [miladmirzazadeh/FloorPlanVLM](https://github.com/miladmirzazadeh/FloorPlanVLM), [manitocross/floorplan-vlm-training](https://huggingface.co/manitocross/floorplan-vlm-training) | SFT (LoRA) + GRPO after the paper, resumable on one A100 | Raster → JSON (walls, doors, windows, rooms) | No licence file\[152\]\[43\] – U | Checkpoints pushed to the user's own Hub repositories; CubiCasa5K – NC; base Qwen2.5-VL-3B, Qwen Research licence – NC\[151\] | 2026-06 and 2026-05; active | README states it does not reach the paper's numbers | Medium (recipe reference; reimplement with an Apache-2.0 base) |
| [Deep Vectorization of Technical Drawings](https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings) (ECCV 2020) | Scan cleaning, primitive estimation, optimisation | Raster → lines and curves | MPL-2.0\[170\] – SA (file level) | Yes (Yandex Disk); synthetic data and PFP floor plans, terms not verified | 2025-11; low | Minutes per image; no semantics | Medium (scan cleaning, line fitting) |
| [HAWP](https://github.com/cherubicXN/hawp) v2/v3 | Wireframe parsing: line segments and junctions; v3 self-supervised | Raster → segments, junctions | MIT\[171\] – P | Yes; v2 on Wireframe/YorkUrban, v3 self-supervised; terms not verified | 2024-02; inactive | Natural images; needs retraining on renders | High (closest open substitute for GLSP; retrain) |
| [DeepLSD](https://github.com/cvg/DeepLSD) | Learned attraction fields + classical line detection | Raster → line segments | MIT; weights "released under an MIT license"\[172\] – P | Yes; Wireframe and MegaDepth | 2026-09; active | No junction types or semantics | Medium (wall-line candidates) |
| [RoomFormer](https://github.com/ywyue/RoomFormer) (CVPR 2023) | Room polygons as two-level queries | Density map → room polygons | MIT\[173\] – P | Yes (ETH polybox); Structured3D – NC, SceneCAD | 2025-04; low | 20-room cap; never trained on drawings | Medium (polygon decoder to retrain as fallback) |
| [PolyRoom](https://github.com/3dv-casia/PolyRoom) (ECCV 2024) | Room-aware transformer with angle-based vertex selection | Density map → polygons | No licence file\[51\] – U | Google Drive; re-upload on Hugging Face without a card\[174\]; Structured3D – NC | 2025-05; low | Needs RoomFormer and MMDetection | Low (method only) |
| [FRI-Net](https://github.com/Daisy-1227/FRI-Net) (ECCV 2024) | Room-wise implicit representation | Density map → polygons | No licence file\[50\] – U | Google Drive; Structured3D, SceneCAD – NC | 2025-05; low | PyTorch 1.9, CUDA 11.1 | Low |
| [HEAT](https://github.com/woodfrog/heat) | Edge attention transformer for planar graphs | Density map or aerial image → planar graph | "not allowed for commercial usage"; GPL-3.0 for research\[144\] – NC | Yes; Structured3D, outdoor buildings | 2025-03; low | NC | Low |
| [Floor-SP](https://github.com/woodfrog/floor-sp) | Mask R-CNN rooms + shortest-path polygon optimisation | Density map → polygons | MIT\[175\] – P | Yes (Google Drive); data not verified | 2023-10; inactive | Old stack | Low |
| [VecFloorSeg](https://github.com/DrZiji/VecFloorSeg) (CVPR 2023) | Two-stream graph attention for room segmentation on line arrangements | Vectorised plan + image → room regions | No licence file\[176\] – U | Processed CubiCasa5K and R2V data – NC/R | 2025-10; low | No licence | Low (idea: rooms from line graphs) |
| [ResPlan utilities](https://github.com/m-agour/ResPlan) | Polygon and graph tools, typed connectivity graphs (via_door, adjacency, via_window), baselines | Vector plan → NetworkX graph | Code MIT; data CC BY 4.0 (one LICENSE, two grants)\[103\] – P | – | 2026-07; active | Built for the ResPlan schema | High (edge schema and QA rules) |
| [MSD code](https://github.com/caspervanengelenburg/msd) | Access graphs (passage, door, front door) from Swiss Dwellings | Vector → graphs | No licence file\[177\] – U | – | 2025-08; low | No licence | Medium (read for the 0.04 m / 0.05 m edge rules; reimplement) |
| [Room-connectivity graphs](https://zenodo.org/records/22169669) (Chenna, 2026) | Door, cased-opening and shared-wall relations between rooms | Raster → typed room graph | CC BY 4.0 (code, weights, evaluation outputs)\[157\] – P | Yes: openings detector, room and wall segmenters; trained on CubiCasa5K, which is not redistributed – NC | Published 2026-08-30 | NC-trained weights under a CC BY tag; not independently reviewed | Medium (only open code for door-level connectivity from raster; retrain) |
| [floorplan-graph](https://github.com/abpaudel/floorplan-graph) (Paudel et al. 2021) | GNN room classification on the room graph | Graph → room types | GPL-3.0\[178\] – SA | Not verified | 2021-08; inactive | Residential vector input | Low |
| [floor-plan navigation graphs](https://github.com/IlliaRohalskyi/floor-plan) | Four methods to turn floor plans into navigation graphs, compared on real and synthetic plans | Raster or text → graph | MIT\[179\] – P | Not verified | 2026-02; low | Navigation focus | Low |

### 4.3 Vector CAD Symbol Spotting

All rows need vector primitives (SVG or DXF) and apply to DWGs, not scans ([literature review §6.1](literature-review.md#61-vector-symbol-spotting-and-primitive-graphs)). None has licence-clean weights.

| Implementation | Task | Input → Output | Code licence | Weights and training data | Activity | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [CADTransformer](https://github.com/VITA-Group/CADTransformer) (CVPR 2022) | Panoptic symbol spotting with HRNet embedding and transformer | Primitives + render → labelled primitives | MIT\[20\] – P | No trained weights in README; ImageNet HRNet backbone; FloorPlanCAD – NC | 2023-07; inactive | Superseded (PQ 68.9) | Low |
| [GAT-CADNet reimplementation](https://github.com/Liberation-happy/GAT-CADNet) | Graph attention spotting; author states the paper "did not provide official code" | Primitives → labels | No licence file\[180\] – U | Not verified; FloorPlanCAD – NC | 2024-12; low | Unofficial; no reported scores | Low |
| [SymPoint](https://github.com/nicehuster/SymPoint) (ICLR 2024) | Primitives as points; point transformer + mask head | Primitives → labelled primitives | IDEA non-commercial research licence\[143\] – NC | Yes (OneDrive); FloorPlanCAD – NC | 2024-03; inactive | NC code and weights | Low |
| [SymPoint-V2](https://github.com/nicehuster/SymPointV2) | SymPoint + layer-feature encoding | Primitives + layers → labels | No licence file\[181\] – U | Yes (Google Drive); FloorPlanCAD – NC | 2026-01; low | PyTorch 1.10, CUDA 11.1; gains depend on layers | Low |
| [VecFormer](https://github.com/WesKwong/VecFormer) (NeurIPS 2025) | Line-based primitives with branch fusion | Primitives → labelled lines | Apache-2.0\[31\] – P | None released (README, no GitHub releases); FloorPlanCAD training scripts | 2025-10; low | 8× A100; PyTorch 2.5; "built with reference to" SymPoint-V2 (no licence) | Medium (best permissive spotter for the optional DWG route; train on BBL DWGs) |
| [DPSS / ArchCAD](https://github.com/ArchiAI-LAB/ArchCAD) | Dual-pathway spotter (primitives + rendered image) with column classes | Primitives + raster → labels | "ACADEMIC USE LICENSE"\[26\] – R | Third-party weights on Hugging Face without a licence\[182\]; ArchCAD-40K, CC BY-NC 4.0, gated\[28\] | 2025-10; low | Academic-only code and data | Low (only spotter with a column class, but not usable) |
| [ArchCAD-gpu](https://github.com/Manavbangotra/ArchCAD-gpu) | DPSS and SymPoint pipeline for US PDF plan sets | Vector PDF → doors, windows, walls | ArchCAD academic licence copied – R | "not cleared for commercial use"\[33\] | 2026-09; active | Inherits academic licence | Low |
| [dwg-vision-object-recognition](https://github.com/ZhouBay-TF/dwg-vision-object-recognition) | DWG export via AutoCAD, SymPoint-V2 inference, geometry rules, visual review | DWG → JSON | No licence file\[34\] – U | Uses a SymPoint-V2 server | 2026-09; active | Needs AutoCAD; documentation in Chinese | Low (architecture reference) |
| [CADSpotting](https://github.com/yangfy2023/CADspotting) | Dense point sampling, PTv3 backbone, sliding-window aggregation | Primitives → labels | No licence file, empty "License" section\[146\]; based on OneFormer3D, CC BY-NC 4.0\[147\] – NC | Checkpoint script referenced, not checked; FloorPlanCAD – NC; LS-CAD not released | Created 2026-03, last push 2026-08; active | MMDet3D, spconv, flash-attn stack | Low (Medium if relicensed) |
| TextCAD\[4\], PolarSym | Text-aided and polar-attention spotting (2026 preprints) | Primitives (+ text) → labels | No public code found | – | – | – | Low |
| [Pang et al. CAD dataset](https://github.com/pangjunbiao/CAD-dataset) | Telecom equipment-room images with keypoint labels; no code | Raster | No licence file\[183\] – U | – | 2024-04; inactive | Not building plans | Low |

### 4.4 Multimodal and VLM Parsers, Agents and Benchmarks

| Implementation | Task | Input → Output | Code licence | Weights and training data | Activity | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [WAFFLE](https://github.com/TAU-VAILab/WAFFLE) | Dataset pipeline (LLM metadata, CLIP and ViT filtering) and fine-tuned models: DETR layout detector (plan, legend, scale, compass), CLIPSeg, ControlNet wall segmentation, Stable Diffusion | Raster + text → boxes, masks | No LICENSE file; README: "We release our code under the Wikimedia Commons license", which is not a software licence\[110\] – U | Yes (SharePoint); WAFFLE images, per-image Commons licences incl. CC BY-SA – U | 2025-04; low | Weak labels; SharePoint links | Medium (layout detector for sheet triage and style tests; ask the authors for a code licence) |
| [AECV-Bench](https://github.com/AECFoundry/AECV-Bench) | Counting and QA benchmark harness; Flask review app | Raster + prompts → scores | No LICENSE file\[127\] – U | – (calls OpenRouter, Cohere, Replicate) | 2026-08; active | Cloud providers; data inherits NC | Medium (protocol; reimplement against self-hosted VLMs) |
| [FloorplanQA](https://github.com/OldDeLorean/FloorplanQA) | Spatial-reasoning QA over JSON layouts, evaluation scripts | JSON → answers | MIT\[129\] – P | – | 2026-05; active | No perception | Medium (checks for area and path reasoning in a VLM step) |
| [AEC-Bench](https://github.com/nomic-ai/aec-bench) | Agentic tasks on drawings, schedules, specifications | Mixed → scores | Apache-2.0\[130\] – P | – | 2026-09; active | Drawing licences not checked | Low |
| [ArchPlanVQA](https://github.com/pengyang-Li/ArchPlanVQA) | Dataset link only | – | No licence file; README is a Google Drive link\[128\] – U | – | 2026-01; low | No code | Low |
| [Planning Drawing Validator](https://github.com/i-dot-ai/planning-drawing-validator) (UK Government Incubator for AI) | Classify drawing types and check planning requirements with any VLM via LiteLLM; CLI, REST, library | Drawing → JSON | MIT\[184\] – P | None (model-agnostic; default configuration uses a cloud model) | 2026-07; active | Validation, not geometry | Medium (public-sector pattern for sheet triage with a self-hosted VLM) |
| [floor-plan-document-intelligence](https://github.com/alifarzadjamali/floor-plan-document-intelligence) | CV and OCR pipeline to structured building information | Raster → JSON | MIT\[63\] – P | Committed checkpoint trained on CubiCasa5K – NC (README notes the dataset licence) | 2026-10; active | Prototype | Low |
| [floorplan-sketch2cad](https://huggingface.co/spaces/matthewishere/floorplan-sketch2cad) (Space) | SAM 2 + Qwen2.5-VL-7B detection + CAD solver | Raster sketch → CAD | README: code Apache-2.0, weights under their own licences\[185\] – P | facebook/sam2-hiera-large and Qwen2.5-VL-7B (both Apache-2.0) | 2026-09; active | Demo | Low (SAM + VLM pattern) |
| Sketch2BIM\[58\], Ayanzadeh & Oates\[64\] | Multi-agent sketch → BIM; agentic room graph | Raster → JSON, graph | No public code found | – | – | – | Low (patterns only) |

### 4.5 Hugging Face Models and Spaces

The Hub search returned several hundred repositories, mostly image-generation LoRAs, game "blueprint" tools and architecture-style classifiers; fewer than 50 models address floor plan recognition\[141\]. The table lists those with weights and a usable card. The "Code licence" column shows the licence stated on the Hub.

| Implementation | Task | Input → Output | Code licence | Weights and training data | Activity | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [Yytsi/floorplan-to-3d-walls](https://huggingface.co/Yytsi/floorplan-to-3d-walls) | U-Net/ResNet-34: floor, wall, door, window | Raster → masks | MIT | Safetensors (~98 MB); CubiCasa5K – NC\[45\] | 2026-05-13; 603 downloads/month | MIT tag on NC-trained weights | Low |
| [phungpx/RMBG-1.4-wall-segmentation-cubicassa](https://huggingface.co/phungpx/RMBG-1.4-wall-segmentation-cubicassa) | Binary wall segmentation | Raster → wall mask | "other"\[186\] | Base briaai/RMBG-1.4, "source-available model for non-commercial use"\[187\]; CubiCasa5K COCO conversion – NC | 2026-06-26; 1,856 downloads/month | Two NC layers | Low |
| [Patnev71/segformer-b0-finetuned-floorplan](https://huggingface.co/Patnev71/segformer-b0-finetuned-floorplan) | Room versus background | Raster → mask | "other"\[188\] | Base nvidia/mit-b0, NVIDIA SegFormer licence, "non-commercially … for research or evaluation purposes only"\[189\]; dataset not accessible | 2025-01 | – | Low |
| [OsamaMo/2dplan2strct](https://huggingface.co/OsamaMo/2dplan2strct) | RF-DETR boxes for wall, room, door, window | Raster → boxes JSON | "other": "non-commercial research and evaluation only"\[46\] | Checkpoint; training data not stated (2,964 validation images) – U | 2026-09-23 | Boxes only | Low |
| [Voix7/rtdetrv2-floorplan-v2](https://huggingface.co/Voix7/rtdetrv2-floorplan-v2) | RT-DETRv2 doors and windows | Raster → boxes | Apache-2.0\[190\] | Base PekingU/rtdetr_v2_r50vd (Apache-2.0); "unknown dataset" – U | 2026-07 | Data unknown | Low |
| [joshlyman/rtdetrv2-floorplancad-doors](https://huggingface.co/joshlyman/rtdetrv2-floorplancad-doors) | RT-DETRv2-R18 doors | Raster → boxes | Apache-2.0\[191\] | "unknown dataset"; name indicates FloorPlanCAD – NC | 2026-04 | – | Low |
| [mudasir13cs/floorcad-yolov8n-detect](https://huggingface.co/mudasir13cs/floorcad-yolov8n-detect) and `-seg` | YOLOv8n, 35 FloorPlanCAD classes | Raster → boxes, masks | AGPL-3.0\[47\] | FloorPlanCAD via the Voxel51 mirror – NC | 2026-08; 437 downloads/month | AGPL and NC | Low |
| [GreenMap YOLO11x blueprint detectors](https://huggingface.co/GreenMap/yolo11x-blueprint-layout-detector) (layout, wall OBB, legend layout) | Drawing area, legend block, title block; wall segments | Raster → boxes | Apache-2.0\[192\] | Ultralytics YOLO11x (AGPL-3.0)\[193\]; "architectural floor plans and construction drawings", source not stated – U | 2026-07 to 2026-08 | Tag conflicts with the framework licence | Low (layout classes match stage 0; retrain with RF-DETR) |
| [GreenMap Qwen3-VL blueprint extractors](https://huggingface.co/GreenMap/qwen3-vl-4b-ru-blueprint-extractor) (2B, 4B) | Field extraction from Russian blueprints | Raster → text, JSON | Apache-2.0\[194\] | Base Qwen3-VL (Apache-2.0); training data not stated – U | 2026-06 | Empty card | Low |
| [mudasir13cs/qwen25-vl-3b-floorplan-grpo](https://huggingface.co/mudasir13cs/qwen25-vl-3b-floorplan-grpo) and `-sft` (copies by minemaster01) | FloorplanVLM-style LoRA (SFT, GRPO) | Raster → JSON | Apache-2.0 tag; card text: "Intended non-commercial / research use"\[153\] | Base Qwen2.5-VL-3B, Qwen Research licence – NC\[151\]; CubiCasa5K – NC; metadata also lists Forceless/Zenodo10K, a PowerPoint corpus unrelated to plans\[195\] | 2026-08; 356 downloads/month | Contradictory metadata | Low (research reference) |
| [BDivyesh/boomi-stage3-vlm-cubicasa-research](https://huggingface.co/BDivyesh/boomi-stage3-vlm-cubicasa-research) | Room-type extraction LoRA | Raster → specification | CC BY-NC-SA 4.0\[196\] | Qwen2.5-VL-3B; CubiCasa5K – NC, labelled as such | 2026-06 | Low exact accuracy per the card | Low |
| [Barath/minicpmv4-floorplan-lora](https://huggingface.co/Barath/minicpmv4-floorplan-lora) | Structured symbol extraction from CAD-style plans | Raster → JSON boxes | Apache-2.0\[197\] | Base MiniCPM-V-4 (Apache-2.0 on its card); 3,281 FloorPlanCAD images – NC | 2026-06 | Hackathon model | Low |
| [rimashussain/gemma4-cubicasa-floorplan](https://huggingface.co/rimashussain/gemma4-cubicasa-floorplan) | Gemma 4 E4B fine-tune (GGUF) | Raster → text | Apache-2.0\[198\] | Base Gemma 4 E4B (Apache-2.0); data not stated, name indicates CubiCasa5K – NC | 2026-04 | – | Low |
| [mohansshf/dpss-archcad](https://huggingface.co/mohansshf/dpss-archcad) | DPSS weights | Primitives + raster → labels | None stated\[182\] | ArchCAD-40K – NC; upstream code academic-only | 2026-04 | – | Low |
| [v1nz/yolo11-floorplan-det](https://huggingface.co/v1nz/yolo11-floorplan-det); [mjlading/floorplan-annotator-models](https://huggingface.co/mjlading/floorplan-annotator-models) | YOLO11 wall, door, window; YOLO rooms and openings | Raster → boxes | CC BY-NC 4.0\[199\]\[200\] | CubiCasa5K (v1nz); a Roboflow project (mjlading) – NC | 2026-09; 2026-03 | AGPL framework as well | Low |
| [nabiullina-dstu/avito-floorplan-checkpoints](https://huggingface.co/nabiullina-dstu/avito-floorplan-checkpoints) | Mask R-CNN, RF-DETR-Seg, SegFormer-B2 and SAM fine-tunes for rooms; comparison report (Russian) | Raster → room masks | MIT\[201\] | ResPlan v2 + CubiCasa5K – NC; SegFormer licence – NC | 2026-08 | Reports that SAM fine-tuning did not help | Low (useful negative result) |
| [sabaridsnfuji/FloorPlanVisionAIAdaptor](https://huggingface.co/sabaridsnfuji/FloorPlanVisionAIAdaptor) | Llama 3.2 11B Vision LoRA describing plans | Raster → text | None stated\[202\] | Llama 3.2 Community Licence; "curated dataset" not named – U | 2024-12; 26 likes | – | Low |
| [TriDyme/florence2-floor-plans-large](https://huggingface.co/TriDyme/florence2-floor-plans-large), [Hyunwoo1605/mask2former-floorplan-instance-segmentation](https://huggingface.co/Hyunwoo1605/mask2former-floorplan-instance-segmentation) | Fine-tunes with template cards | Raster → text, masks | None stated | Data not stated – U | 2024 | Empty cards | Low |

Spaces:

| Implementation | Task | Input → Output | Code licence | Weights and training data | Activity | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [RasterScan/Automated-Floor-Plan-Digitalization](https://huggingface.co/spaces/RasterScan/Automated-Floor-Plan-Digitalization) | Raster-to-vector demo (57 likes) | Raster → vectors | MIT tag; app posts images to an external backend\[149\] | Closed SDK with licence activation\[148\] | 2025-12 | Not self-hostable as open source | Low |
| [Viraj2307/Floor-Plan-Detection](https://huggingface.co/spaces/Viraj2307/Floor-Plan-Detection) | Room outlines, doors, windows (15 likes) | Raster → JSON | Apache-2.0 tag; client for Roboflow's hosted inference API\[150\] | Hosted models; data not stated | 2024-11 | Cloud only | Low |
| [Dharini27/floorplan-vectorizer](https://huggingface.co/spaces/Dharini27/floorplan-vectorizer) | Mask R-CNN to COCO-style polygons (14 likes) | Raster → polygons | None stated\[203\] | Weights via Google Drive; data not stated | 2025-04 | – | Low |
| [rawanessam/FloorPlanTransformation](https://huggingface.co/spaces/rawanessam/FloorPlanTransformation), [rawanessam/DeepFloorPlan](https://huggingface.co/spaces/rawanessam/DeepFloorPlan), [IvanHahan/cubicasa](https://huggingface.co/spaces/IvanHahan/cubicasa) | Demos of Raster-to-Vector (PyTorch port), DeepFloorplan, CubiCasa5K | Raster → vectors, masks | None stated; upstream MIT, GPL-3.0, CC BY-NC 4.0\[204\] | Upstream weights – R / NC | 2024-12 to 2025-07 | Unofficial | Low (quick visual comparison) |
| [JL67/floorplan-door-detector](https://huggingface.co/spaces/JL67/floorplan-door-detector), [JL67/floorplan-symbol-detector](https://huggingface.co/spaces/JL67/floorplan-symbol-detector) | Demos of the RT-DETRv2 door and YOLOv8 FloorPlanCAD models | Raster → boxes | None stated | FloorPlanCAD – NC | 2026-09 | – | Low |
| [egorkoriakov/floor-plan-rooms](https://huggingface.co/spaces/egorkoriakov/floor-plan-rooms) (also `-contour`, `-structural`) | FastAPI YOLO service with a morphological room and wall algorithm | Raster → JSON | None stated\[205\] | `best.pt`; data not stated – U | 2026-07 | – | Low |
| [build-small-hackathon/Qwen-wall-segmentation](https://huggingface.co/spaces/build-small-hackathon/Qwen-wall-segmentation) | Wall segmentation by image editing with Qwen-Image-Edit-2511 | Raster → wall image | Apache-2.0\[206\] | Base Apache-2.0 | 2026-06 | Experimental | Low |

Datasets on the Hub are covered in [open datasets](datasets-open.md). The mirrors behind the fine-tunes above repeat its licence pattern: v1nz/cubicasa5k-yolo states CC BY-NC 4.0 correctly, phungpx/cubicassa5k-coco states "other", Claudio9701/cubicasa5k states no licence, and Voxel51/FloorPlanCAD states CC BY-SA 4.0 against the original's CC BY-NC 4.0.

### 4.6 OCR and Text in Drawings

PaddleOCR, docTR, kraken, Tesseract, Surya, TrOCR, Docling and Florence-2 are in [§5.5](#55-ocr-engines). This table adds what they do not cover.

| Implementation | Task | Input → Output | Code licence | Weights and training data | Activity | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [RapidOCR](https://github.com/RapidAI/RapidOCR) | ONNX Runtime and OpenVINO packaging of PP-OCR detection, orientation and recognition; used in pilot v2 | Raster → text boxes, strings | Apache-2.0\[154\] – P | PP-OCR models converted from PaddleOCR, redistributed under Apache-2.0 | 2026-10; active | Same model limits as PP-OCR | High (in use; keep) |
| [OnnxTR](https://github.com/felixdittrich92/OnnxTR) | docTR models in ONNX for CPU inference | Raster → text | Apache-2.0\[207\] – P | docTR weights | 2026-10; active | Not evaluated on plans | Medium (light second engine) |
| [EasyOCR](https://github.com/JaidedAI/EasyOCR) | CRAFT detection + CRNN recognition, 80+ languages incl. German, French, Italian | Raster → text | Apache-2.0\[208\] – P | Bundled; training data not documented (not verified) | 2025-12; low | Not evaluated on plans | Medium (second engine for voting) |
| [PARSeq](https://github.com/baudm/parseq) | Scene-text recogniser, fine-tunable | Text crops → strings | Apache-2.0\[209\] – P | Synthetic and public benchmark data; terms vary | 2024-05; inactive | Needs a detector | Medium (fine-tune on synthetic stamps and hand lettering) |
| [MMOCR](https://github.com/open-mmlab/mmocr) | Toolbox: DBNet, FCENet, ABINet, key information extraction | Raster → text | Apache-2.0\[210\] – P | Per model | 2024-11; low | OpenMMLab maintenance has slowed | Low |
| [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) | Character-region detection for rotated and curved text | Raster → text regions | MIT\[211\] – P | SynthText and ICDAR (terms not verified) | 2024-07; inactive | Detection only | Low |
| [eDOCr2](https://github.com/javvi51/edocr2) | Segmentation and OCR of mechanical engineering drawings (dimensions, tolerances, tables); synthetic recogniser training | Raster drawing → structured text | MIT\[212\] – P | Recognisers in GitHub releases; synthetic | 2025-01; low | Mechanical drawings, not plans | Medium (pattern for drawing-specific OCR) |
| [Text-Detection-on-Technical-Drawings](https://github.com/2Obe/Text-Detection-on-Technical-Drawings) | Generator of artificial technical drawings + TF detector and keras-ocr recogniser; matches \[134\], not confirmed as official | Synthetic drawings → text boxes | MIT\[213\] – P | Trained on generated drawings | 2022-04; inactive | TensorFlow Object Detection API | Medium (generator pattern for room stamps) |
| [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator) | Synthetic text-line images from fonts, backgrounds and distortions | Fonts + strings → labelled crops | MIT\[214\] – P | – | 2024-07; inactive | Lines only | High (synthetic stamp recognisers with the OFL fonts) |
| [SynthTIGER](https://github.com/clovaai/synthtiger) | Synthetic text image generator | Fonts + corpora → labelled images | MIT\[215\] – P | – | 2024-06; inactive | – | Medium |

### 4.7 General Frameworks

Pointers only. Mask2Former (MIT, archived), SAM 3, DINOv2/v3 and Grounding DINO are in [§5.3](#53-building-blocks-for-training-and-preprocessing).

| Implementation | Task | Input → Output | Code licence | Weights and training data | Activity | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) | U-Net, FPN, DeepLabV3+, SegFormer heads with 800+ encoders | Raster → masks | MIT, with per-file exceptions\[216\]\[155\] – P | ImageNet encoders; `mit_b*` NVIDIA NC, MobileOne Apple licence | 2026-10; active | Check per-encoder licence | High (in use) |
| [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) | Self-configuring U-Net training | Raster → masks | Apache-2.0\[217\] – P | Train from scratch | 2026-10; active | Medical defaults, needs adaptation of I/O | High (baseline comparator) |
| [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) | Segmentation toolbox | Raster → masks | Apache-2.0\[218\] – P | Model zoo, per-model data | 2024-08; inactive | Unmaintained since 2024 | Low |
| [detectron2](https://github.com/facebookresearch/detectron2) | Detection and segmentation library | Raster → boxes, masks | Apache-2.0\[219\] – P | COCO model zoo | 2026-09; active | Used by DPSS and WAFFLE | Medium |
| [RF-DETR](https://github.com/roboflow/rf-detr) | Real-time DETR detection and instance segmentation (DINOv2 backbone) | Raster → boxes, masks | Apache-2.0 for the package, N/S/M/L detection and Seg weights; PML 1.0 for XL, 2XL and A/F/P\[220\] – P (sizes) | COCO-pretrained | 2026-10; active | Boxes need conversion to geometry | High (openings, columns, stairs, stamps) |
| [RT-DETR](https://github.com/lyuwenyu/RT-DETR) / [D-FINE](https://github.com/Peterande/D-FINE) | Real-time DETR detectors | Raster → boxes | Apache-2.0\[221\]\[222\] – P | COCO/Objects365 | 2026-09; 2026-10; active | – | Medium (alternatives to RF-DETR) |
| [timm](https://github.com/huggingface/pytorch-image-models) | Backbones and pretrained weights | Raster → features | Apache-2.0\[223\] – P | Per weight; check each | 2026-10; active | Weight licences vary | Medium |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | Detection and segmentation | Raster → boxes, masks | AGPL-3.0\[193\] – SA | COCO | 2026-10; active | Network copyleft or paid licence | Low (avoid) |
| [CVAT](https://github.com/cvat-ai/cvat) / [Label Studio](https://github.com/HumanSignal/label-studio) | Self-hosted annotation (polygons, polylines, keypoints) | Images → labels | MIT\[224\]; Apache-2.0\[225\] – P | – | 2026-10; active | – | High (BBL gold set) |
| [supervision](https://github.com/roboflow/supervision) | Detection utilities: metrics, tracking, annotation | Predictions → metrics, overlays | MIT\[226\] – P | – | 2026-10; active | – | Medium |
| [vtracer](https://github.com/visioncortex/vtracer) | Raster to SVG tracing | Raster → SVG paths | MIT\[227\] – P | No learning | 2026-10; active | No semantics | Low (preview vectorisation of masks) |

### 4.8 Other Places Searched

- **Roboflow Universe.** Many floor plan datasets and hosted models, typically labelled CC BY 4.0, with undisclosed image provenance; some are relabelled CubiCasa5K or CVC-FP ([open datasets §5](datasets-open.md#5-licence-pitfalls-mirrors-and-derivatives)). Roboflow's own floor plan tutorial trains on a 504-image Universe set without naming its licence\[228\]. Hosted inference is cloud-only, and Universe pages refused automated access (HTTP 403), so no entries were catalogued individually.
- **Zenodo.** One relevant software record: the room-connectivity code and weights (§4.2). Floor Plan CIS, the dataset behind MitUNet, is a Zenodo dataset under CC BY 4.0 built from listing images\[162\].
- **GitLab.** No relevant floor plan recognition project; the hits were chip floorplanning, game tools and SDKs for commercial APIs.
- **Papers with Code.** Shut down in July 2025\[142\]. Hugging Face paper pages link code only for SymPoint and SymPoint-V2 among the 15 papers checked, and link the FloorplanVLM community adapters and the DPSS weights as models.

## 5. Pipeline Components

The components the pilot pipeline is built from: a recommended starter stack per pipeline step, then curated tables for the CAD and PDF toolchain, training and preprocessing building blocks, vision-language models, OCR engines and IFC generation. The tables use the relevance scale of §1; licences are as found in October 2026 and must be confirmed before use. Commercial products are in [closed solutions](solutions-closed.md).

### 5.1 Recommended Starter Stack

| Pipeline step | First choice | Licence | Alternatives / notes |
|---|---|---|---|
| Read DWG | ODA File Converter → ezdxf | Proprietary (free) / MIT | libredwg-web (GPL-3.0, entity gaps) |
| Validate CAD layers | plan-check | MIT (bundles GPL-3.0 libredwg-web) | Commercial validation product in use at BBL |
| Render DWG/PDF to raster | ezdxf drawing add-on; PyMuPDF | MIT; AGPL/commercial | – |
| Preprocess scans | OpenCV; UVDoc (dewarping) | Apache-2.0; MIT | Augraphy (MIT) to simulate scan defects for training |
| OCR | PaddleOCR PP-OCRv6 | Apache-2.0 | docTR (Apache-2.0); kraken for hand lettering |
| Read stamps, classify usage, triage | Qwen3-VL, self-hosted | Apache-2.0 for most sizes (verify) | Florence-2 (MIT) |
| Detect openings, stairs, symbols | RF-DETR (train on own renders) | Apache-2.0 | Avoid Ultralytics YOLO (AGPL-3.0) |
| Segment walls and rooms | Mask2Former or a U-Net (train on own renders) | MIT | DINOv2 backbone (Apache-2.0) |
| Geometry, rooms, connectivity | Shapely, GeoPandas, networkx | BSD | – |
| Vector symbol spotting (optional) | VecFormer | Apache-2.0 code; no weights released | – |
| Export | ezdxf (CAD-Richtlinie layers), IfcOpenShell | MIT; LGPL | FME (commercial, in-house skills) |
| Worst scans | Swiss digitising services ([closed solutions §3.7](solutions-closed.md#37-digitising-area-measurement-and-scan-to-plan-services)) | Service | – |

### 5.2 CAD and PDF Toolchain

| Model / tool | Task | Inputs | Code / weights | Training data and licence | Limitations | Relevance |
|---|---|---|---|---|---|---|
| [ODA File Converter](https://www.opendesign.com/guestfiles/oda_file_converter) | Convert DWG ↔ DXF (all versions) | DWG | Free binary (Windows/macOS/Linux) | No training; proprietary licence (read EULA for government use)\[12\] | Needs a virtual display (xvfb) on headless Linux; "frequently returns exit code 0 even when a specific drawing failed" – validate outputs\[12\]\[13\] | High (DWG → DXF before rendering) |
| [ODA Drawings SDK (Teigha)](https://www.opendesign.com/) | Programmatic DWG read/write | DWG | Commercial SDK | No training; commercial ODA membership | Membership cost and terms | Medium (most robust for production batch) |
| [GNU LibreDWG](https://www.gnu.org/software/libredwg/) | Open DWG read/convert | DWG | [Source](https://www.gnu.org/software/libredwg/) | No training; GPLv3+\[14\] | Work in progress; "lacks support for some DWG entities" (FreeCAD docs)\[15\] | Low |
| [libredwg-web](https://github.com/mlightcad/libredwg-web) | LibreDWG compiled to WebAssembly: parse DWG/DXF in the browser or Node.js; DWG → SVG example | DWG, DXF | [mlightcad/libredwg-web](https://github.com/mlightcad/libredwg-web) (npm `@mlightcad/libredwg-web`) | No training; GPL-3.0 | Inherits LibreDWG entity gaps\[15\] | Medium (local parsing and SVG rendering) |
| [plan-check](https://github.com/bbl-dres/plan-check) (BBL prototype) | Validate DWG floor plans against the CAD-Richtlinie BBL V1.0 (40 rules: layers, room polygons, AOID room stamps, hatches); runs locally in the browser, no upload | DWG (bundled parser has no DXF reader) | [bbl-dres/plan-check](https://github.com/bbl-dres/plan-check) | No training; MIT app bundling GPL-3.0 libredwg-web | Prototype; validation incomplete | High (QA gate for CV output; finds compliant DWGs to use as training labels) |
| [ezdxf](https://ezdxf.mozman.at/) | Parse DXF (layers, blocks, INSERT attributes, TEXT/MTEXT, hatches); render DXF to PNG, PDF or SVG with the drawing add-on | DXF (DWG via ODA) | [mozman/ezdxf](https://github.com/mozman/ezdxf) | No training; MIT | No direct DWG input; layer semantics depend on each author; use `recover.readfile()` for damaged files\[13\] | High (rendering + native text) |
| [Shapely](https://shapely.readthedocs.io/) / [GeoPandas](https://geopandas.org/) | Polygonise wall lines, close gaps, area checks, text-in-polygon joins; clean up polygons from raster model outputs | Vector geometry | [shapely/shapely](https://github.com/shapely/shapely), [geopandas/geopandas](https://github.com/geopandas/geopandas) | No training; BSD | Geometry only, no semantics | High |
| [FME](https://www.safe.com) (Safe Software) | Low-code DWG → GIS/IFC transformation and QA | DWG, DXF and many other formats | Commercial | No training; commercial; already used at BBL, with in-house expertise\[1\] | Rule logic still needed per layer profile | Low for extraction (Medium for output and integration) |
| [PyMuPDF](https://pymupdf.readthedocs.io/) / [pdfplumber](https://github.com/jsvine/pdfplumber) | Extract vector paths and text spans with coordinates; PyMuPDF also renders pages to raster | PDF | [pymupdf/PyMuPDF](https://github.com/pymupdf/PyMuPDF), [jsvine/pdfplumber](https://github.com/jsvine/pdfplumber) | No training; AGPL/commercial (PyMuPDF), MIT (pdfplumber) | PyMuPDF licence matters for internal tools; no layers/blocks; text may be outlined | Medium |
| [pdf2svg](https://github.com/dawbarton/pdf2svg) / Inkscape CLI | Convert vector PDF → SVG | Vector PDF | [dawbarton/pdf2svg](https://github.com/dawbarton/pdf2svg) | No training; GPL | Only needed for the vector symbol-spotting route (SVG input) | Low |

IfcOpenShell and Bonsai are covered in §5.6.

### 5.3 Building Blocks for Training and Preprocessing

| Model | Task | Inputs | Code / weights | Training data and licence | Limitations | Relevance |
|---|---|---|---|---|---|---|
| [RF-DETR](https://github.com/roboflow/rf-detr) | Real-time detection and instance segmentation (DINOv2 backbone) | Raster | [roboflow/rf-detr](https://github.com/roboflow/rf-detr); COCO-pretrained weights | Apache-2.0 (check per checkpoint size) | Boxes and masks; needs post-processing into geometry | High (doors, windows, columns, stairs, furniture) |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | Detection and segmentation | Raster | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 or paid Enterprise licence | Licence risk for internal federal tools | Low |
| [Mask2Former](https://github.com/facebookresearch/Mask2Former) | Universal segmentation (walls, rooms) | Raster | [facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former) | MIT | Repository archived | Medium |
| [SAM 3](https://github.com/facebookresearch/sam3) | Segmentation prompted by text or examples | Raster | [facebookresearch/sam3](https://github.com/facebookresearch/sam3); gated weights | Custom SAM licence | Licence review needed; no floor plan semantics | Medium (pre-labelling) |
| [DINOv2](https://github.com/facebookresearch/dinov2) / [DINOv3](https://github.com/facebookresearch/dinov3) | Self-supervised vision backbones | Raster | Weights (DINOv3 gated) | DINOv2 Apache-2.0; DINOv3 custom licence | Backbone only | Medium (DINOv2 for licence) |
| [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | Open-vocabulary detection | Raster + text | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Apache-2.0 | Little maintenance since 2024 | Low (pre-labelling) |
| [Deep Vectorization of Technical Drawings](https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings) (ECCV 2020) | Clean raster drawings and fit line/curve primitives | Raster | [Vahe1994/Deep-Vectorization-of-Technical-Drawings](https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings) | MPL-2.0 | Low activity | Medium (wall lines → vectors) |
| [Augraphy](https://github.com/sparkfish/augraphy) | Simulate print, scan, fax and copy degradation | Clean raster → degraded raster | [sparkfish/augraphy](https://github.com/sparkfish/augraphy) | MIT | Augmentation only | High (style and scan robustness) |
| [UVDoc](https://github.com/tanguymagne/UVDoc) | Document dewarping | Photo or scan → flat image | [tanguymagne/UVDoc](https://github.com/tanguymagne/UVDoc) | MIT | Research code, inactive since 2024 | Medium (folded or photographed sheets) |
| [CubiCasa5k-Next](https://github.com/Lqm1/CubiCasa5k-Next) | Clean-room PyTorch 2 reimplementation of the CubiCasa5K model | Raster | [Lqm1/CubiCasa5k-Next](https://github.com/Lqm1/CubiCasa5k-Next) | Apache-2.0 code; weights trained on CubiCasa5K stay non-commercial | Residential baseline | Medium (permissive code base) |

### 5.4 Vision-Language Models

| Model | Task | Inputs | Code / weights | Training data and licence | Reported results | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [Gemini 3 Pro](https://deepmind.google/models/gemini/) | General VLM: read room annotations, classify usage, count, QA | Raster | Closed; cloud API | Proprietary | AECV-Bench (independent): 0.51 mean exact-count accuracy, 16.0% MAPE; door 0.39, window 0.34\[55\] | Door-swing misreading, window/opening confusion, hallucinated fixtures\[55\]; cloud | Low (Medium for non-sensitive plans) |
| [GPT-5.2](https://platform.openai.com/docs/models) | As above | Raster | Closed; cloud API | Proprietary | AECV-Bench (independent): 0.49 mean exact-count accuracy, 19.4% MAPE\[55\] | As above; cloud | Low (Medium for non-sensitive plans) |
| [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) / [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) | Open VLM: read room annotations, classify usage, sheet triage; base for fine-tuning (FloorplanVLM uses Qwen2.5-VL) | Raster | Open weights; self-hostable | Apache-2.0 for most sizes – verify per checkpoint | See FloorplanVLM ([literature review §6.2](literature-review.md#62-raster-models-and-object-detection)) | Imprecise coordinates without fine-tuning; needs GPUs | High |
| [Florence-2](https://huggingface.co/microsoft/Florence-2-large) | Small VLM: OCR with regions, captioning, detection | Raster | Open weights | MIT | – | Not evaluated on plans | Medium |
| [PaliGemma](https://ai.google.dev/gemma/docs/paligemma) | Open VLM, fine-tunable | Raster | Open weights | Gemma terms | – | Custom licence; not evaluated on plans | Low |

Practical role for BBL:
- Use VLMs for reading and normalising room stamps (multilingual DE/FR/IT, abbreviations such as "Büro", "Bureau", "Ufficio", "WC D/H"), classifying room usage against a BBL taxonomy, sheet triage (plan type, floor, scale bar), and flagging anomalies.
- Do not use them as the source of coordinates.
- Self-hostable candidates: Qwen2.5-VL / Qwen3-VL (Apache-2.0 for most sizes; verify per checkpoint), Florence-2 (MIT), PaliGemma (Gemma terms). Cloud frontier models (GPT-5.x, Gemini 3, Claude) need a data-protection assessment before sending federal plans.

### 5.5 OCR Engines

| Model | Task | Inputs | Code / weights | Training data and licence | Reported results | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [PaddleOCR PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR) | Text detection + recognition; one model for 50 languages, including 46 Latin-script languages\[99\] | Raster | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR); weights on Hugging Face | Apache-2.0 | +4.6 points detection, +5.1 recognition over PP-OCRv5\[99\] | Not yet tested on plans; small rotated stamps need high-DPI tiling; model provenance review | High |
| [PaddleOCR-VL-1.6](https://github.com/PaddlePaddle/PaddleOCR) | Document VLM (0.9B) for page parsing | Raster | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR); weights on Hugging Face | Apache-2.0 | 96.3% on OmniDocBench v1.6\[99\] | Built for documents, not plans; model provenance review | Medium (fallback for difficult stamps) |
| [PaddleOCR PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR) | Text detection + recognition with text-line orientation; `latin_PP-OCRv5_mobile_rec` covers German, French, Italian (and Romansh)\[59\] | Raster | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR); ONNX: [monkt/paddleocr-onnx](https://huggingface.co/monkt/paddleocr-onnx)\[60\] | Apache-2.0 | 84.7% accuracy on a 3,111-image Latin-script set\[60\]\[61\] | Superseded by PP-OCRv6 | Medium (tested baseline) |
| [Docling](https://github.com/docling-project/docling) | Document conversion and layout analysis | PDF, raster | [docling-project/docling](https://github.com/docling-project/docling) | MIT | – | Page-level text; not built for small rotated room stamps | Medium (title blocks, legends, sheet triage) |
| [kraken](https://github.com/mittagessen/kraken) | Historical and handwritten text recognition | Raster | [mittagessen/kraken](https://github.com/mittagessen/kraken) | Apache-2.0 code; model licences vary | – | Needs fine-tuning on BBL hand lettering | Medium (hand-lettered historical plans) |
| [docTR](https://github.com/mindee/doctr) (Mindee) | Text detection + recognition; good on small text; rotation support | Raster | [mindee/doctr](https://github.com/mindee/doctr) | Apache-2.0 | – | Not evaluated on plans | Medium (second engine) |
| [Tesseract 5](https://github.com/tesseract-ocr/tesseract) | Text recognition (deu, fra, ita) | Raster | [tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract) | Apache-2.0 | – | Weak on rotated/small text without preprocessing | Low (baseline) |
| [Surya](https://github.com/datalab-to/surya) | Multilingual OCR, layout, reading order | Raster | [datalab-to/surya](https://github.com/datalab-to/surya) | GPL code; weights have commercial-use restrictions – verify | – | Licence | Low |
| [TrOCR](https://github.com/microsoft/unilm/tree/master/trocr) | Line-level text recognition, fine-tunable | Raster (text-line crops) | [microsoft/unilm](https://github.com/microsoft/unilm/tree/master/trocr) | MIT | – | Needs a separate text detector | Medium (fine-tune on room annotations) |

### 5.6 IFC Generation

| Model / tool | Task | Inputs | Code / weights | Training data and licence | Limitations | Relevance |
|---|---|---|---|---|---|---|
| [IfcOpenShell](https://ifcopenshell.org/) (`ifcopenshell.api`) | Script IFC: IfcWall from centreline + thickness + storey height; IfcSpace with name/number/area Psets; IfcDoor/IfcWindow via IfcOpeningElement | 2D vector model | [IfcOpenShell/IfcOpenShell](https://github.com/IfcOpenShell/IfcOpenShell) | No training; LGPL | Needs a clean 2D model; scripting effort | Medium (most controllable route) |
| [Bonsai](https://bonsaibim.org/) (ex-BlenderBIM) | Manual/assisted modelling and IFC QA on IfcOpenShell | IFC | Part of the IfcOpenShell project | No training; GPL | Manual | Low |
| [Archilyse pipeline](https://github.com/Archilyse/Archilyse) | IFC from Archilyse annotations\[82\] | Archilyse annotations | Repository removed (HTTP 404, October 2026); forks may exist | No training; AGPL\[7\] | Vendor bankrupt | Low |
| [FME](https://www.safe.com) IFC writer | IFC from GIS/CAD data | FME-readable vector data | Commercial | No training; commercial | Licence cost | Medium (existing BBL FME skills) |
| Bimify, WiseBIM, Archilogic, BIMxAI, AmpliFY ([closed solutions §3](solutions-closed.md#3-commercial-products-and-services)) | IFC export from plans | Plans | Closed | Proprietary | Cloud | Low |
| ArcGIS Indoors → IFC | Not native; via FME | Indoors model | Closed | Commercial | GIS-centric model | Low |

The recommended stance is to treat IFC as a derived product of a clean 2D vector model (GeoJSON/DXF with storey, height defaults and room attributes), not as a direct target of ML. A 2D-first canonical model keeps QA simple and keeps the IFC step deterministic.

## 6. Licence Pitfalls

| Item | Stated licence | Actual constraint | Source |
|---|---|---|---|
| CubiCasa5K repository | Shown as "NOASSERTION" by GitHub | CC BY-NC 4.0 covers code and data | \[5\] |
| SymPoint | "License for Non-commercial Scientific Research Purposes" | NC; also bans military and surveillance use | \[143\] |
| DPSS / ArchCAD, ArchCAD-gpu | "ACADEMIC USE LICENSE" | Commercial use, including commercial research, prohibited | \[26\]\[33\] |
| HEAT | GPL-3.0 file plus a custom LICENSE | Not for commercial use; GPL only for research | \[144\] |
| FP4S | CC BY-NC-SA 4.0 | Code is NC and share-alike | \[145\] |
| CADSpotting | No licence file | All rights reserved; base OneFormer3D is CC BY-NC 4.0 | \[146\]\[147\] |
| SymPoint-V2, PolyRoom, FRI-Net, VecFloorSeg, MSD code, AECV-Bench, ArchPlanVQA, GAT-CADNet reimplementation, FloorplanVLM reimplementations, most Hub Spaces | No licence file | All rights reserved: read, do not reuse | §4 |
| WAFFLE | "Wikimedia Commons license" | Not a software licence; images keep per-file licences, some CC BY-SA | \[110\] |
| Yytsi/floorplan-to-3d, OpenBIM-FloorPlan-AI, avito checkpoints, Barath, joshlyman, karanjaWakaba/Yolo_*_cubicasa (by name), Zenodo connectivity models | MIT, Apache-2.0 or CC BY 4.0 on the weights | Trained on CubiCasa5K or FloorPlanCAD: NC | \[45\]\[164\]\[201\]\[197\]\[191\]\[157\] |
| mudasir13cs Qwen adapters | Apache-2.0 tag | Card says non-commercial; base Qwen2.5-VL-3B is "for research or evaluation purposes only" | \[153\]\[151\] |
| RMBG-1.4 fine-tunes | "other" | Bria licence, non-commercial | \[187\] |
| SegFormer / MiT weights (incl. smp `mit_b*`, MitUNet, segformer-b0 fine-tunes) | MIT on the host repository (smp, MitUNet); "other" on Hub cards | NVIDIA Source Code License: "non-commercially … research or evaluation purposes only" | \[189\]\[155\] |
| GreenMap YOLO11 detectors | Apache-2.0 tag | Ultralytics framework and base weights AGPL-3.0, or a paid licence | \[192\]\[193\] |
| RF-DETR | Apache-2.0 | XL, 2XL and A/F/P under PML 1.0 | \[220\] |
| RasterScan Space | MIT tag | Front end only; backend is a closed, activated SDK | \[149\]\[148\] |
| Raster-to-Vector PyTorch port | MIT | `IP_gurobi.py` needs a commercial Gurobi licence; use the PuLP variant | \[169\] |
| ImageNet-pretrained backbones | BSD/Apache weights | ImageNet images are for "non-commercial research and educational purposes"; effect on weights unsettled | \[156\] |
| GPL-3.0 code (DeepFloorplan, TF2DeepFloorplan, Raster-to-Graph, floorplan-graph, FloorplanToBlender3d) | GPL-3.0 | Copyleft on distribution; internal use is unaffected | \[36\]\[37\]\[40\] |

Rules:
- The licence of the training data and of the base model limits the weights, whatever the weights' own tag says.
- No licence file means no reuse rights. Ask the authors or reimplement from the paper.
- Record repository URL, commit, licence and weight provenance for every third-party component the pilot adopts, as `SOURCE.md` does for datasets ([data/README](../data/README.md)).

## 7. Gaps

What no open-source implementation provides, and BBL has to build:
1. **Licence-clean pretrained weights** for any floor plan task. Every model must be trained on BBL's own renders and compliant DWGs.
2. **Room-stamp parsing**: reading multi-field stamps (AOID or number, usage, area) and assigning them to rooms. Only generic OCR and synthetic text generators exist.
3. **Official code for the most relevant methods**: GLSP, MuraNet, FloorplanVLM and FPBench-2K, TextCAD.
4. **Door-level connectivity from raster** with licence-clean weights; one NC-trained research release exists.
5. **Columns, stairs as flights, voids and gross floor area**: only DPSS has a column class, under an academic licence.
6. **An evaluation harness for BBL's metrics** (area error in m², Angle F1, connections found). FloorplanQA and AECV-Bench cover VLM reasoning only.
7. **Maintained, current code**: most research code needs porting to current PyTorch and CUDA before use.

## 8. Limitations of This Catalogue

- Not legal advice. NC, research-only, academic and missing licences need an opinion from BBL's legal service, also for internal benchmarking.
- Licences were read from LICENSE files and model cards on 7 October 2026. Statements about training data are the authors' own; no weights were downloaded and no code was run.
- Activity is the last push (`pushed_at`), which counts pushes to any branch; it is not the last release.
- The unauthenticated GitHub API (60 requests per hour, 10 searches per minute) limited the search to the top-ranked results per query. Roboflow Universe refused automated access, and the Papers with Code mirrors did not load.
- Hugging Face search matches repository names; models named otherwise are missed. Download counts are the Hub's figures for the last month.
- Not verified: the official status of the GAT-CADNet, CADSpotting and technical-drawing text repositories; the CADSpotting checkpoint; MitUNet weights; training data of HAWP, CRAFT, EasyOCR, Floor-SP and Deep Vectorization weights; the data behind the GreenMap, Voix7, OsamaMo and egorkoriakov models; whether VecFormer reuses SymPoint-V2 code.
- The field moves quickly: CubiCasa5k-Next and CADSpotting appeared in 2026, and FloorplanVLM's benchmark is announced. Re-check before adopting any component.

## 9. Sources

- Shared [source list](../research/sources.md); numbers 141–228 were added for this catalogue.
- Papers: [research index](../research/README.md); Markdown versions in `research/papers-md/` (local only).
- [Motivation and goals](motivation-goals.md) · [Literature review and state of the art](literature-review.md) · [Closed solutions](solutions-closed.md) · [Open datasets](datasets-open.md) · [Closed datasets](datasets-closed.md) · [Open building documentation](datasets-open-docu.md) · [Pipeline design](pipeline.md) · [Pilot v2](../pilot/v2-pipeline/README.md)
