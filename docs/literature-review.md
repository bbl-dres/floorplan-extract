# Literature Review and State of the Art

*Technical document · October 2026 · Methods, models, benchmarks and datasets for structured floor plan extraction, and the findings of the papers in the [research index](../research/README.md), mapped to the [pipeline design](pipeline.md). Ready-to-use tools are in [open](solutions-open.md) and [closed solutions](solutions-closed.md), training and test data in [open datasets](datasets-open.md). Bracketed numbers refer to the shared [source list](../research/sources.md).*

## 1. Scope, Method and Conventions

**Corpus.** The research index ([papers.json](../research/papers.json)) lists 81 entries: 78 papers and 3 practitioner pages (AxcelerateAI).
- **Read in full: 63 papers** with a local copy, converted to Markdown with Docling. 32 formed the first review; 31 were added in October 2026.
  - Most of the additions target the weak spots that [pilot v2](../pilot/v2-pipeline/README.md) measured: domain adaptation, synthetic data, topology, vector PDFs, room stamps, text in drawings, scale, annotation and historical drawings. A survey and three benchmarks complete them.
  - Twelve papers on sheet layout, title blocks, scale, CAD layers and grid axes came from the [domain review of inputs, masking and scale](reviews/2026-10-07-pipeline-target-inputs-masking-scale.md).
- **Summarised from abstract and index note: 10 papers.** These are marked † below; their figures come from the abstracts as reported in the index.
  - Open access but the download was blocked: Petitpierre & Guhennec 2023, eDOCr2 (2025), EXSCLAIM! (2023), Chang et al. 2025, Lin & Wang 2026.
  - Not open access: the surveys by Pizarro et al. 2022 and Khade et al. 2025, Schönfelder et al. 2024 (text on floor plans), Faltin et al. 2024 (scale inference) and Yin et al. 2020 (CAD layers).
- **Not reviewed: 5 papers** that are not open access: Macé et al. 2010, de las Heras et al. 2013 and 2015 (CVC-FP), Pizarro et al. 2023 (MLSTRUCT-FP) and Li et al. 2026 (ArchPlanVQA). Their datasets and benchmarks appear in §6 and §7 from their project pages.

**Surveys as context.** Three surveys frame the field. Rezvanifar et al. 2019 review symbol spotting in architectural drawings. Pizarro et al. 2022 review rule-based and learning-based analysis of raster plans from 1995 to 2021. Khade et al. 2025 sort floor plan models into nine application families.\[695\]\[696\]\[697\] The first two predate most of the panoptic symbol spotting, vector-PDF and VLM work reviewed here; Khade et al. are known only from their abstract. §3.1 relates their groupings to ours.

**Desk research.** The catalogue of approaches, models, code and datasets (§3, §6, §7) adds code repositories, Hugging Face checkpoints, dataset cards and vendor pages (October 2026), BBL's 2025 market screening\[1\] and BBL's vendor talks. Key figures and licences were checked against primary sources (papers, LICENSE files, dataset cards) where possible. The detailed licence checks are in [open solutions](solutions-open.md) and [open datasets](datasets-open.md).

**Method.** For each paper we recorded task, method, data, results as reported, stated and observed limitations, and relevance to pipeline stages 0–10.

**Quality of the evidence.** Several papers have internal inconsistencies, worth knowing before quoting them:
- PolarSym's tables contradict each other.
- CADSpotting's colour ablation is labelled the wrong way round.
- DeepFloorplan reports the best of several evaluations on the test set.
- Ahmed et al. 2012 judged room-label correctness themselves.
- Baseline numbers differ between papers. The FloorPlanCAD baseline, for example, is quoted as 55.3, 56.1 and 59.5 PQ.
- VectorGraphNET leads on weighted F1 (89.0), while its plain F1 (79.4) trails SymPoint's (86.8). Its comparison may also mix length-weighted and support-weighted F1.
- Two papers lack a clean held-out test:
  - Trivi reports scores on the 19 annotated drawings without describing a train/test split.
  - Guo et al.'s preprint splits flipped and rotated copies of the same plans at random across training and validation.
- Raster2Seq states three different CubiCasa5K splits. Semap's text and tables disagree on some gains. Chen et al. 2024's conclusion credits topology losses with "minor improvements" that its own table does not show.
- PP-OCRv6 is evaluated on in-house benchmarks only.
- Among the drawing-text and benchmark papers:
  - Ying et al. measure recognition on ground-truth text crops, not after detection.
  - Khan et al.'s "hallucination rate" is 100 minus precision, and their headline F1 covers one class.
  - Several open-model rows in MMArch's results table are exact linear transforms of other rows; quote only its headline scores.

The Docling conversion lost some formulas and garbled a few tables; these are marked where they matter.

**Conventions.**
- Results are the authors' own unless marked "independent", and are not comparable across datasets or rows.
- *Our evidence* marks results from [pilot v2](../pilot/v2-pipeline/README.md) and its [review](reviews/2026-10-07-pipeline-and-pilot-v2.md): one building with three sheets, plus public benchmarks. It is not literature and is labelled wherever it appears.
- Model and code tables use Model (linked) | Task | Inputs | Code / weights | Training data and licence | Limitations | Relevance, as in the solution catalogues; section-specific columns (reported results) sit before Limitations. Paper tables (§8) use Paper | Strategy | Contribution | Key result | Main limitation | Relevance.
- Relevance of papers: High = directly shapes a pipeline stage or the training data; Medium = useful ideas or baseline; Low = little transferable.
- Relevance of models, code and datasets: High = strong starting point for raster input, self-hostable, plausible licence (open source preferred). Medium = useful baseline or benchmark that needs major adaptation, or is limited by licence or hosting. Low = wrong domain or task, no code, superseded, or cloud-only.
- Licence shorthand as in [open datasets](datasets-open.md#1-scope-and-method): NC = non-commercial; R = research or academic use only.

## 2. Key Findings

1. **The field has converged on our representation.**
   - Raster-to-Vector, Chen et al. 2023, Raster-to-Graph, FloorplanVLM, CubiCasa5K and Sketch2BIM share it: typed wall junctions joined by wall segments, openings attached to walls, and rooms as closed cycles of the wall graph.
   - VectorFloorSeg reaches the same structure from vector input, with rooms as faces of the wall-line arrangement.
   - Zhang 2026 reads one network out in two ways: as an emitted coordinate sequence, or as detected junction and centre-line heatmaps assembled into a graph. On real CubiCasa5K scans detection wins by 2.7–5.1 wall F1 points, depending on the tolerance, and openings are better detected than emitted (opening F1 0.64 against 0.25).
   - This confirms the structural, graph-based design in [pipeline.md](pipeline.md).
2. **Nothing has been tested on BBL-like plans.**
   - Almost all methods are trained and tested on residential plans, at 256–512 px, mostly assuming axis-aligned (Manhattan) walls, and mostly East Asian.
   - The exceptions are narrow:
     - VectorGraphNET: 76 CAD-exported plans of ten university buildings, proprietary.
     - Versailles-FP: palace plans, walls only.
     - Trivi: 19 archival drawings of Roman monuments, labelled with materials; three of them are plans.
   - None evaluates office floors with room stamps, full-size sheets or area in m².
3. **Drawing-style shift is the dominant failure mode** (§5.1).
   - Trained on FloorPlanCAD and tested on office and campus plans, SymPoint falls from 83.3 to 33.2 PQ and CADSpotting from 87.4 to 52.9.
   - A CubiCasa5K model scores door IoU 0.099 on the multi-style historical WAFFLE set.
   - Wall segmentation transferred across styles drops from about 90 to 76 mean IoU.
   - Raster2Seq, trained on CubiCasa5K, keeps interior IoU 73.9 zero-shot on WAFFLE, against 60.5 for RoomFormer.
4. **Mixed-style and synthetic training data help, and our pilot shows it for drawings.**
   - Pooling styles restores wall IoU to about 91–93 (Dodge et al.).
   - Noisy data and exactly aligned re-renders are complementary. FloorplanVLM reaches 67% and 85% validity with each alone, 90% with both, and 96% after reinforcement learning.
   - Procedural synthesis with textures cut from real scans adds 5.1 mIoU on historical maps (Semap). This is with synthetic data mixed with real labels, followed by fine-tuning on real labels.
   - Style augmentation (Chen et al.) and synthetic scan degradation (Egiazarian et al.) are documented recipes.
   - *Our evidence:* a U-Net trained only on Swiss Dwellings, rendered in random styles, has seen no real plan ([review §2](reviews/2026-10-07-pipeline-and-pilot-v2.md#2-evidence)).
     - On 400 CubiCasa5K test plans it reaches zero-shot pixel IoU 0.57 / 0.49 / 0.57 for walls, doors and windows. CubiCasa's own in-domain model reaches 0.73 / 0.54 / 0.67.
     - On three unseen pilot sheets, wall IoU is 0.87 on a CAD print and 0.58 on a poché scan, against 0.36 for classical CV.
5. **Unlabelled target data can be used, but no paper does so on drawings.**
   - Three techniques close much of the synthetic-to-real gap on street scenes:
     - self-training with an EMA teacher and masked-image consistency (MIC);
     - multi-resolution crops (HRDA);
     - rare-class sampling.
   - Example: on GTA→Cityscapes, adding MIC to HRDA raises mIoU from 73.8 to 75.9.
   - They are the next lever after style randomisation (§5.4).
6. **Assigning room stamps to rooms is almost unstudied.**
   - Only Ahmed et al. 2012 assign OCR text to rooms: by containment, with a dictionary match, about 82% correct by the authors' own judgement.
   - TextCAD shows that text helps symbol spotting, but links it to geometry only implicitly.
   - No paper parses multi-field stamps (number, usage, area).
   - Word Beam Search is a better building block than fuzzy matching after OCR: it constrains words to a dictionary during decoding and leaves numbers free.
   - *Our evidence:* local OCR (PP-OCRv5 Latin) read 15 of 15 printed room names and 12 of 15 stamp areas on a CAD print, but no calligraphic name exactly ([pilot v2](../pilot/v2-pipeline/README.md#findings)).
7. **No recognition method outputs which two rooms a door connects.**
   - Only the datasets define it geometrically: MSD links two areas when a door polygon is within 0.05 m of both, and ResPlan has "via_door" edges.
   - The one recognition system that tries, Ayanzadeh & Oates, links each door to the two nearest room centroids. That rule is fragile for corridors.
   - *Our evidence:* pilot v2 links each opening to the rooms on its two sides. It found 12 of 12 reference connections on the CAD print and 8 of 12 on a survey scan.
8. **Stairs, voids and columns are barely covered.**
   - Stairs exist as one symbol class in a few datasets.
   - Voids are not handled anywhere.
   - Columns appear only in ArchCAD-400K, Raster-to-Vector and MSD. Talebi-Kalaleh & Mei detect them by glyph rules on vector framing plans: recall 0.922, precision 0.997, on generated drawings.
   - *Our evidence:* zero-shot column IoU is 0.09 and stair IoU 0.38 on CubiCasa5K.
9. **Area in m² is almost never reported, and scale recovery is rarely evaluated.**
   - Geometric metrics are IoU or F1 at pixel thresholds.
   - Guo et al. report one apartment's total area, within 0.81%, using a manually set scale.
   - Talebi-Kalaleh & Mei recover scale from dimension strings within 0.086% on generated framing plans. Lv et al. describe the same cue for raster plans but evaluate it only in their supplement.
   - Newer recipes for raster plans (Liang et al., Lin & Wang, Chang et al.) report OCR accuracy or whole-plan correctness (62 of 200 plans in Liang et al.), not scale error. Scale bars are read only in microscopy and map work (stage 1c).
   - *Our evidence:* the median room area error against the stamps is 2.7% on a CAD print, and 2.9% against the reference on CubiCasa5K.
10. **VLMs read but do not measure.**
    - AECV-Bench: text questions score up to 0.95, door and window counts 0.03–0.39 exact.
    - FloorplanQA: areas, unions and paths must be computed with libraries. Models do far better with structured JSON than with images (59–81% vs 19–40%).
    - MMArch: on figures from architecture and engineering papers, the best model answers 51.7% of questions and the best open model 29.9%, against 94.6% for human experts.
    - The 2026 designs restrict the VLM:
      - Talebi-Kalaleh & Mei let it act only through typed edit operations on a deterministic parse;
      - Sketch2BIM keeps a human in the loop.
11. **Vector symbol spotting scores are inflated by layer information.**
    - Wall-type ("stuff") PQ for SymPoint-V2 falls from 80.8 to 49.3 without layers.
    - Only CADSpotting and VecFormer are strong without layers, and all these methods need vector input.
    - Vector PDFs have no layers, and two routes exist for them:
      - classifying paths by geometry and stroke style (VectorGraphNET);
      - deterministic parsing of the PDF drawing operators (Talebi-Kalaleh & Mei).
12. **Topology losses are no shortcut.**
    - TopoMortar, on brick-wall images:
      - clDice gives the most topologically correct masks;
      - Skeleton Recall is the most robust to noisy labels;
      - augmentation plus self-distillation lifts plain cross-entropy + Dice past most topology losses.
    - Chen et al. 2024, on historical maps: topology losses did not beat plain binary cross-entropy, and a watershed on an edge map closed shapes best.
    - The choice has to be made on a plan benchmark scored by rooms.
13. **Little licence-clean data exists** (§5.6, §7).
    - Swiss Dwellings and MSD (CC BY 4.0) are Swiss, metric vector data.
    - ResPlan is CC BY 4.0 on GitHub but CC BY-NC-SA 4.0 on Kaggle, and has unclear provenance.
    - FloorPlanCAD, CubiCasa5K and ArchCAD-400K are non-commercial; Versailles-FP is for academic use under an agreement.
    - Small real sets go far: about a dozen labelled sheets can bootstrap a model in a new archive style (Petitpierre & Guhennec†), and Trivi trained on 19 drawings. BBL's own labelled sample therefore matters more than public volume.
14. **Isolating the drawing on the sheet is learnable and pays off, but no paper tests it on floor plan archives.**
    - Detectors trained on a few hundred to about 1,400 labelled sheets find title blocks at 0.97–1.00 and drawing regions at 0.94–1.00 in the respective metrics (Pang et al., Lombardi et al., FELD).
    - Detecting, cropping and then reading beats prompting a VLM with the whole sheet.
    - Cropping to the plan raised two baselines by 4–6 mIoU points (Lv et al.).
    - Several drawings per sheet, each with its own caption and scale, are handled on bridge drawings (Peng et al. 2025), not yet on plans (stage 1b).

## 3. Taxonomy of Approaches

### 3.1 Taxonomy of the Corpus

Every paper in [papers.json](../research/papers.json) has one primary class: an approach family or contribution type (L1) and a strategy (L2), defined in `meta.taxonomy`. Counts include the papers without a local copy (marked °); † marks the ten papers summarised from their abstracts. Section 8 summarises the papers in the same order.

- **Survey (3).** Reviews of the field or of an adjacent one; no new method.
  - Floor plan analysis (3): surveys of floor plan or architectural drawing analysis. Rezvanifar et al. 2019, Pizarro et al. 2022†, Khade et al. 2025†.
  - Adjacent field (0).
- **Classical and rule-based (4).** Hand-designed image processing and rules, no learned model for the main task.
  - Heuristic pipeline (3): line and text separation, morphology, gap closing, rooms from connected regions. Macé et al. 2010°, Ahmed et al. 2011 and 2012.
  - Statistical and structural recognition (1): patch classifiers or statistical models feeding structural (graph) reasoning. de las Heras et al. 2013°.
- **Raster parsing (18).** Learned parsing of drawing images (pixels, keypoints, regions, text), then conversion to geometry, sheet structure or scale.
  - Semantic segmentation (5): pixel classes, geometry derived afterwards. Dodge et al. 2017, DeepFloorplan, MuraNet, Guo et al. 2026, Trivi 2026.
  - Segmentation and vectorisation (3): segmentation or keypoints plus an explicit vectorisation step. Raster-to-Vector, Lv et al. 2021, Chang et al. 2025†.
  - Primitive vectorisation (1): cleaning raster drawings and fitting lines and curves, without building semantics. Egiazarian et al. 2020.
  - Symbol and keypoint detection (1): detectors or keypoint models for symbols, openings and layout regions. Pang et al. 2024.
  - Sheet layout and title blocks (4): detecting drawing views, title blocks, legends, notes and revision tables, and reading title blocks and view captions. Lombardi et al. 2025, Peng et al. 2025 (bridge views), Huang et al. 2026 (FELD), Peng et al. 2026 (title blocks).
  - Scale and dimension reading (4): metric scale from dimension strings, scale bars or scale notes; adjacent-field methods marked by domain. EXSCLAIM!† (microscopy scale bars), Faltin et al. 2024†, Lin & Wang 2026†, Liang et al. 2026.
- **Structured prediction (7).** Learned models that output graphs or polygons directly.
  - Graph-based (3): wall or room graphs predicted or classified with graph networks or autoregressive decoders. Paudel et al. 2021, Chen et al. 2023, Raster-to-Graph.
  - Polygon decoder (4): room polygons as query sets, sequences or implicit fields. RoomFormer, PolyRoom, FRI-Net, Raster2Seq.
- **Vector CAD parsing (11).** Models on vector primitives (lines, arcs, text, layers) of CAD drawings or vector PDFs, mostly learned.
  - Panoptic symbol spotting (8): class and instance per primitive, scored with panoptic quality. CADTransformer, GAT-CADNet, SymPoint, SymPoint-V2, CADSpotting, VecFormer, TextCAD, PolarSym.
  - Primitive graph segmentation (2): primitives as graph nodes classified into walls, rooms or regions. VectorFloorSeg, VectorGraphNET.
  - Layer and entity classification (1): classifying CAD layers, blocks or entities by their content rather than their names. Yin et al. 2020†.
- **Vision-language models and agents (5).** Multimodal language models fine-tuned or orchestrated for drawings.
  - Fine-tuned VLM (1): FloorplanVLM.
  - Agentic pipeline (4): LLM or VLM agents orchestrating tools, rules and humans. Sketch2BIM, Ayanzadeh & Oates, Talebi-Kalaleh & Mei, BlueprintAgent.
- **Text recognition (8).** Text detection and recognition, and its use on drawings.
  - OCR system (2): PaddleOCR 3.0, PP-OCRv6.
  - Decoding and lexicons (1): Word Beam Search.
  - Text in technical drawings (5): OCR and text parsing specialised for engineering or architectural drawings. Schlagenhauf et al. 2022, Schönfelder et al. 2024†, eDOCr2†, Khan et al. 2025, Ying et al. 2026.
- **Learning strategies (5).** Training and adaptation methods from adjacent fields that transfer to plan parsing.
  - Domain adaptation (2): MIC, DAFormer/HRDA.
  - Synthetic data and domain generalisation (1): Semap.
  - Topology-aware learning (1): TopoMortar.
  - Annotation strategy (1): Petitpierre & Guhennec 2023†.
- **Datasets and benchmarks (17).** Data or evaluation protocols as the primary contribution.
  - Raster plan dataset (5): CVC-FP°, CubiCasa5K, Versailles-FP, MLSTRUCT-FP°, WAFFLE.
  - Vector plan dataset (4): FloorPlanCAD, MSD, ResPlan, ArchCAD-400K.
  - Other drawing dataset (0).
  - Benchmark (8): ICDAR 2021 MapSeg (historical maps), Chen et al. 2024 (historical maps), FloorplanQA, AECV-Bench, ArchPlanVQA°, MMArch, FloorPlan-VLN, Zhang 2026 (readout comparison, ResPlan-FP).
- **Practitioner sources (3).** Grey literature, not peer-reviewed. Vendor pipeline (3): AxcelerateAI's tutorial, solution page and model guide.

The learning-strategy papers come from street scenes, historical maps and brick walls, and most sheet-layout and scale papers from façade, bridge, mechanical and structural drawings, maps and microscopy. They are included for what transfers to plans; that transfer is our inference, not their result.

**How this relates to the surveys' groupings.**
- Rezvanifar et al. split symbol spotting by description (pixel-based, usually statistical, vs vector-based, usually structural) and by matching strategy. The first split survives in ours as raster parsing vs vector CAD parsing.
- Pizarro et al. group work into rule-based and learning-based approaches and, within them, by task (walls, doors, rooms, vectorisation, modelling). Their rule-based group is our "Classical and rule-based". We split the learning-based group by what the model outputs: pixels converted afterwards (raster parsing), graphs or polygons directly (structured prediction), labelled vector primitives (vector CAD parsing) or text and JSON (VLMs). Tasks appear in our pipeline stages (§4), not in the taxonomy.
- Khade et al.'s nine families mix tasks and applications (room-based, dimensional, retrieval, area-based and others). They map onto our stages and contribution types rather than onto one L1 class.

### 3.2 Approach Families

Published work differs mainly in the input it assumes (vector primitives or pixels) and the output it produces (labelled primitives, pixel masks, boxes, polygons or graphs, text):

- **Rule-based CAD parsing.** Read layers, blocks and text entities; polygonise walls into rooms; assign text by point-in-polygon. Exact where drafting conventions hold, brittle where they do not.
- **Vector-PDF parsing.** Read paths, line weights, dash patterns and text spans from the PDF drawing operators, then apply rules (Talebi-Kalaleh & Mei) or classify the paths with a graph network (VectorGraphNET).
- **Panoptic symbol spotting on vector primitives.** Treat each line or arc as a graph node, point or token and label it with a class and an instance (CADTransformer, GAT-CADNet, SymPoint, VecFormer, DPSS). The standard benchmark is FloorPlanCAD, scored with panoptic quality (PQ).
- **Raster segmentation, then vectorisation.** A multi-task CNN predicts wall, room and opening masks (DeepFloorplan, CubiCasa5K); heuristics, optimisation or integer programming turn the masks into polygons (Raster-to-Vector, Lv et al.). Scored with pixel IoU and room-level F1.
- **Direct vector prediction.** Models emit graphs or polygon sequences directly:
  - on plan images: Raster-to-Graph, Raster2Seq, and FloorplanVLM (JSON from a fine-tuned VLM);
  - on point-cloud density maps rather than drawings: RoomFormer, PolyRoom and FRI-Net.\[2\]\[3\]
- **Line vectorisation.** Clean the raster and fit line and curve primitives (Deep Vectorization of Technical Drawings), then polygonise with rules or run vector methods on the result.
- **Object detection.** YOLO- or DETR-family detectors return boxes for doors, windows, furniture and stairs. Easy to fine-tune, but boxes are not geometry.
- **Sheet layout and scale reading.** Detect drawing views, title blocks and legends on the sheet; read captions and title blocks from crops; recover the scale per drawing from dimension strings, scale bars or scale notes (FELD, Lombardi et al., Peng et al., Lin & Wang, Liang et al.).
- **OCR plus association.** Text detection and recognition, followed by "text centre inside polygon" rules and dictionary matching to attach room annotations to rooms.\[62\]\[63\] TextCAD (2026) is one of few models that use CAD text jointly with geometry.\[4\]
- **Vision-language models.** General models prompted for reading, counting and QA, open models fine-tuned for direct vector output, or agents that call tools and rules.

Common practices across papers:
- Labels are derived from CAD layers and blocks where possible. ArchCAD-400K cut annotation effort "from 1,000 person-hours required for 16K data to 800 person-hours for 413K data".\[29\]
- Training data is mostly residential or small-scale: an average FloorPlanCAD drawing covers about 1,000 m², against 11,000 m² in ArchCAD-400K.\[29\] Many raster methods assume axis-aligned (Manhattan) layouts.
- Training on clean renders of vector data is common; simulated scan degradation (e.g. Augraphy) can narrow the gap to real scans.
- Vector models are often scored with layer or colour information as a "prior". Archive DWGs with inconsistent layers cannot supply it, so scores without priors are the realistic reference (§5.2).
- Results are self-reported on the authors' own benchmark and split, so numbers are rarely comparable across papers.

| Approach family | Typical input | Elements covered | Maturity | Main risk for BBL | Relevance |
|---|---|---|---|---|---|
| Rule-based CAD parsing (ezdxf, FME, ArcGIS Indoors import) | DWG/DXF | Room outlines, room attributes, walls; doors, windows and furniture via blocks | Production | Per-author layer mapping effort; breaks down where layers are inconsistent | Low (essential only for compliant DWGs: labels and native text) |
| Vector PDF parsing + rules (PyMuPDF, pdfplumber; operator parsing after Talebi-Kalaleh & Mei) | Vector PDF | Lines, line weights, text, hatches → walls, room attributes, scale from dimension strings; room outlines via polygonisation | Production libraries, custom logic | No layers or blocks; text may be outlined | Medium |
| Primitive graph segmentation (VectorFloorSeg, VectorGraphNET) | Vector primitives (SVG from DWG or PDF) | Walls, doors, windows as path classes; rooms as faces of the line arrangement | Research; code without licence or not released | Tested on clean or proprietary data only | Medium (vector-PDF branch) |
| CAD panoptic symbol spotting (SymPoint-V2, DPSS, VecFormer, CADTransformer, GAT-CADNet) | Vector primitives (SVG/DXF) | Walls, doors, windows, furniture, stairs; columns only with ArchCAD-400K classes | Research with code | Non-commercial data; GPU training; vector input only | Medium (DWG primitives only) |
| Raster segmentation / vectorisation (DeepFloorplan, CubiCasa5K, Raster-to-Vector, Raster-to-Graph, Lv et al.) | Raster | Room outlines, walls, doors, windows, some furniture icons | Research | Residential training data with few drawing styles; scans | High as an approach (pilot v2 follows it); existing open models Medium |
| Learned room-polygon decoders (Raster2Seq; RoomFormer, PolyRoom, FRI-Net on density maps) | Raster or density map | Room outlines and types; doors and windows as polygons (Raster2Seq) | Research with code | 256 px input; weights trained on NC data; misplaced openings | Medium (second room hypothesis) |
| Raster object detection (RF-DETR, YOLO) | Raster | Doors, windows, furniture, stairs (boxes) | Easy to fine-tune | Boxes not geometry; Ultralytics YOLO is AGPL-3.0 | High as an approach |
| Sheet layout detection and scale reading (FELD, Lombardi et al., Peng et al., Lin & Wang, Liang et al.) | Raster (whole sheet) | Drawing regions, title block, legend, notes; scale from dimensions, bars or notes | Research; no released code or data | Evidence from façade, bridge and mechanical sheets; Swiss title blocks differ | High as an approach (pipeline stages 1b and 1c) |
| VLM / multimodal LLM | Raster (rendered) | Room attributes, room types, counts, QA | Fast-moving | Geometric imprecision, hallucination; cloud models cannot see internal plans | Medium (High for room attributes with a self-hosted model) |
| OCR (PaddleOCR PP-OCRv6, docTR, Tesseract, kraken) | Raster or rendered vector | Room-annotation text (DE/FR/IT), dimension strings | Production | Rotated/small text, association logic | High |
| Commercial plan-to-vector / plan-to-BIM | All | Varies; usually walls, doors, windows, room outlines | Production (managed services) | Cloud-only, no CH hosting | Low (Medium for non-sensitive plans) |

Learning strategies (domain adaptation, synthetic data, topology losses, annotation protocols) are not extraction methods. They change how the raster models are trained (§5.3, §5.4).

### 3.3 Capability Matrix

Legend:
- Y = covered natively, P = partial or with post-processing, N = not covered, ? = unverified.
- Room attributes = number, usage and area read from room annotations.
- Stairs / ramps: no model or dataset with a documented ramp class was found, so Y and P refer to stairs.
- The rule-pipeline row assumes consistent layers and blocks. With inconsistent layers only native text (room attributes) stays Y.
- Floor outlines (GF), special polygons and room connections have no column: they are derived from walls, rooms and doors after CV ([pipeline stage 8](pipeline.md#8-derived-outputs)).
- Relevance matches the detailed tables in §6; commercial products are described in [closed solutions](solutions-closed.md).

| Model | Input | Room outlines | Room attributes | Walls | Doors | Windows | Columns | Furniture | Stairs / ramps | Output | Relevance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [ezdxf](https://ezdxf.mozman.at/) / [FME](https://www.safe.com) rule pipeline | DWG/DXF | Y (closed polylines or polygonised walls) | Y (TEXT/MTEXT/attributes) | Y (layer) | Y (blocks) | Y (blocks) | P (layer/hatch) | Y (blocks) | P | Any vector, IFC via IfcOpenShell | Low (essential for compliant DWGs) |
| [ArcGIS Indoors CAD import](https://pro.arcgis.com/en/pro-app/latest/help/data/indoors/import-cad-floor-plans-with-import-cad-to-indoor-dataset.htm) | DWG, DGN, RVT | Y | Y (annotation mapping) | Y | Y | Y | P | P | P | Indoors GDB / feature layers | Low |
| [VecFormer](https://arxiv.org/abs/2505.23395) / [SymPoint-V2](https://arxiv.org/abs/2407.01928) (FloorPlanCAD classes) | Vector primitives | N (wall "stuff" only) | N | Y | Y | Y | N (no column class in FloorPlanCAD)\[23\] | Y | Y | Labelled primitives | Medium (VecFormer; SymPoint-V2 has no licence) |
| [DPSS](https://arxiv.org/abs/2503.22346) (ArchCAD-400K classes) | Vector primitives + rendered image | N | N | Y | Y | Y | Y (column class)\[29\] | Y (one merged "furniture" class)\[29\] | Y | Labelled primitives | Low (only column class, but academic-only code and data) |
| [TextCAD](https://arxiv.org/abs/2607.12678) (2026) | Vector primitives + text | N | P (uses text) | Y | Y | Y | ? | Y | ? | Labelled primitives | Low |
| [VectorGraphNET](https://arxiv.org/abs/2410.01336) (2024) | Vector PDF → SVG paths | N | N (text removed) | Y (load-bearing, non-load-bearing, partition) | Y | Y | ? | ? | ? | Labelled paths | Low (no code; Medium as a method) |
| [VectorFloorSeg](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_VectorFloorSeg_Two-Stream_Graph_Attention_Network_for_Vectorized_Roughcast_Floorplan_Segmentation_CVPR_2023_paper.html) (2023) | Wall lines + render | Y (faces with room types) | N | P (input) | N | N | N | N | N | Labelled regions | Low (code without licence) |
| [DeepFloorplan](https://arxiv.org/abs/1908.11025) | Raster | Y (room-type masks) | N | Y | P (openings) | P | N | N | N | Raster masks | Medium |
| [CubiCasa5K model](https://arxiv.org/abs/1904.01920) | Raster | Y | N | Y | Y | Y | N | P (icons) | N | Raster masks → polygons | Medium |
| [Raster-to-Graph](https://doi.org/10.1111/cgf.15007) | Raster | Y | N | Y (graph) | P | P | N | N | N | Vector graph | Medium |
| [Raster2Seq](https://arxiv.org/abs/2602.09016) (2026) | Raster | Y (polygons with room types) | N | N (room boundaries only) | Y (polygons) | Y (polygons) | N | N | N | Labelled polygons | Medium |
| [FloorplanVLM](https://arxiv.org/abs/2602.06507) (2026) | Raster | Y | N | Y | Y | Y | N | N | N | JSON vectors | Medium |
| YOLO / RF-DETR checkpoints on Hugging Face ([open solutions §4.5](solutions-open.md#45-hugging-face-models-and-spaces)) | Raster | P (boxes) | N | P (boxes) | Y | Y | N | P | P | Boxes JSON | Low |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) / [docTR](https://github.com/mindee/doctr) | Raster | N | Y (text only) | N | N | N | N | N | N | Text + boxes | High |
| Frontier VLMs (GPT-5.x, Gemini 3, Claude, [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)) | Raster | P | Y (reading) | P | P | P | P | P | P | JSON (imprecise coords) | Medium (only self-hosted models for internal plans) |
| [Archilogic](https://www.archilogic.com/) | Raster, PDF, DWG, DXF | Y | Y (name/usage/area in model) | Y | Y | Y | ? | Y (assets) | ? | DXF, SVG, GeoJSON, IFC, IMDF\[9\] | Low (Medium for non-sensitive plans) |
| [Bimify](https://bimify.com/) | DWG, DXF, PDF, scans | Y | ? | Y | Y | Y | ? | ? | ? | IFC, RVT, DWG, DXF, CSV\[10\] | Low (Medium for non-sensitive plans) |
| [WiseBIM Plans2BIM](https://wisebim.fr/software/plans2bim/) | PDF, PNG, JPEG | Y | ? | Y | Y | Y | ? | N | ? | IFC, DXF, CSV/XLSX\[11\] | Low (Medium for non-sensitive plans) |
| [AmpliFY (INEX)](https://ify.inex.fr/) (BBL test) | PDF | Y | N (named rooms "Undefined"; usage guessed from symbols, often wrongly) | Y | N | Y | ? | P (WC, sinks) | P (stairwell as room type) | IFC, RVT, DWG | Low |

*Our evidence:* for comparison, pilot v2 covers room outlines, room attributes (OCR), walls, doors, windows and stairs, has a weak column class and no furniture ([pilot v2](../pilot/v2-pipeline/README.md)).

## 4. Findings by Pipeline Stage

Stage numbers follow [pipeline.md](pipeline.md) draft 0.3, which split stage 0 into normalisation (0a) and triage (0b) and added sheet layout (1b) and a scale proposal (1c) after the [domain review of inputs, masking and scale](reviews/2026-10-07-pipeline-target-inputs-masking-scale.md).

### Stage 0: Normalisation, Triage and Input Quality

- **WAFFLE:** separates plans from non-plans with an LLM pass over metadata, CLIP scoring and a ViT classifier.
- **ArchCAD-400K:** rejects drawings whose layer names deviate more than 5% from a reference table, keeping 5,538 of 11,917. This is a direct analogue of checking DWG compliance before using a DWG for labels.
- **Layer names are a weak feature.**
  - FloorPlanCAD's authors, working from real DWGs: "The layer name does not necessarily explains its content".
  - Yin et al.† classify CAD layers by their content instead of their names. On 94 drawings (elevations) they find nearly all floor levels and measure 88% of visible members perfectly.\[709\]
  - For archive DWGs, classify layers by content or render them; trust layer names only for DWGs that pass plan-check.
- **Gaps:** no paper triages by drawing style or scan quality.

### Stage 1: Preprocessing

- **Cleaning:** Egiazarian et al. clean scans with a U-Net trained on 20,000 synthetic pairs: vectors rendered onto 40 real scanned paper backgrounds, then blurred, distorted and noised. On real scans this gives IoU 92%, against 49% for a pretrained sketch cleaner.
- **Resolution:**
  - The learned graph methods work at 256–512 px, far too coarse for archive sheets.
  - Tile-and-stitch: Pang et al. tile; CADSpotting's sliding-window aggregation raises PQ on large sheets from 58.8 to 75.5.
  - Fine output stride matters for thin walls (Dodge et al.).
  - HRDA trains on two 512 px crops:
    - a detail crop at full resolution;
    - a context crop at half resolution, covering four times the area.
  - A learned scale attention fuses the two. On GTA→Cityscapes this scores 73.8 mIoU, against 65.1 for the detail crop alone. Gains are largest on thin classes: pole +7.5 and traffic sign +9.9 IoU over DAFormer.
  - Semap averages logits at full and half resolution on map pages beyond 10,000 px (+4.2 mIoU).
- **Orientation:** PaddleOCR includes text-line orientation and document unwarping. Unwarping must happen before every stage, not only OCR, or text and geometry no longer align.
- **Vector PDFs:** both papers below keep exact geometry that rasterising throws away.
  - Talebi-Kalaleh & Mei traverse the PDF drawing operators, tracking the transformation stack, line width, dash pattern and stroke and fill colours. They split line weights into hatch, medium and heavy classes per drawing.
  - VectorGraphNET converts PDF to SVG and flattens groups and transforms into plain paths.

### Stage 1b: Sheet Layout and Masking

- **Region-first parsing is learnable from a few hundred labelled sheets.**
  - Pang et al. detect nine region classes on rendered CAD sheets of telecom equipment rooms (300 training and 60 test images), then look for symbols only inside them. F1 at IoU > 0.5: main drawing 1.00, title bar 1.00, legend 1.00, scale text 0.97, north arrow 0.95.
  - WAFFLE fine-tunes a DETR, starting from a document-table detector, on 200 labelled images with four classes (floor plan, legend, scale, compass). It reports no detection accuracy. The boxes route OCR text to plan labels or legend entries.
  - Lombardi et al. detect title block, main content, legend and notes on 1,385 building drawings (Faster R-CNN). Accuracy at IoU ≥ 0.7: title block 0.971, main content 0.943, legend 0.86, notes 0.82.\[700\]
  - FELD (Huang et al.): 551 façade sheets with nine classes.\[699\]
    - RF-DETR is best: mAP50 0.949, mAP50:95 0.890, title block 0.989.
    - Qwen3-VL-8B with LoRA has the best F1 (0.911) but takes 6.1 s per sheet.
    - Pretraining on general document layouts (DocLayout-YOLO) hurt.
    - Inputs were downscaled to 1,344 × 800 px, which loses thin lines.
  - Peng et al. 2026 localise title blocks on German bridge drawings (64 test drawings): Qwen2.5-VL-3B with LoRA reaches F1 93.8%, YOLOv11-m 93.1% with tighter boxes. Zero-shot, the VLM did not localise at all.\[701\]
- **Detect, crop, then read.**
  - Lombardi et al.: rule baselines and table-structure models failed on hard drawings, and GPT-4o given the whole sheet missed the title block on 6 of 10. Cropped title blocks were read well.
  - Peng et al. 2026: after fine-tuning, the open VLM reads the title block's "Maßstab" field with word accuracy 0.990, against 0.572 zero-shot. The base model's licence is "qwen-research".
- **Several drawings on one sheet.**
  - Peng et al. 2025 detect views on 2,000 bridge drawings (YOLOv7, validation mAP50 about 0.92). They match view titles to views by containment, intersection or nearest centre, so each view is cropped and named by its caption, e.g. "Draufsicht M. 1 50".\[702\]
  - Document parsing handles the same problem: PaddleOCR 3.0's PP-StructureV3 adds a region model because one newspaper page holds several articles.
  - DWG layouts give the regions directly: each viewport has its own clip boundary and scale.
- **Cropping pays.**
  - Lv et al. detect the plan region first (YOLOv4, AP0.5 0.97), because the plan can cover "< 50%" of an image.
  - Adding the crop raised mIoU from 0.76 to 0.80 for Raster-to-Vector and from 0.76 to 0.82 for DeepFloorplan.
- **Masks.**
  - ICDAR 2021 MapSeg, task 2: segment each map sheet's content area, scored by the 95th-percentile Hausdorff distance (winner 19 px, others 85–126 px).
  - MapSeg, task 1: an input mask is supplied, and only predictions inside it are kept.\[703\]
  - Talebi-Kalaleh & Mei remove legend strokes and title-block rules by a rule (members without bearing at either end) instead of a fixed crop.
  - No floor plan paper masks non-drawing regions in training.
- **Gaps:**
  - No paper on floor plan sheets splits several drawings with a polygon each.
  - No openly licensed plan dataset annotates sheet regions, title blocks or captions.
  - FELD's and Lombardi et al.'s data are not released.

### Stage 1c: Scale Proposal per Drawing

- **Dimension strings paired with dimension lines** are the most precise cue:
  - Talebi-Kalaleh & Mei, on vector PDFs:
    - text and line are paired under explicit tolerances: text within 12 mm, direction within 6°, text projecting into the middle 90% of the line, ticks or arrows at both ends;
    - outliers are rejected by median absolute deviation, the inlier share is the confidence, and the result snaps to standard scales within 1.5%;
    - the maximum error is 0.086% on 50 generated plans.
  - Lv et al.: detect dimension-line endpoints and number regions, match them by bipartite matching, and take the largest k-means cluster of the resulting ratios, with a 0.9 m door as fallback. Scale accuracy is reported only in their supplement.
  - Liang et al.: take the median of OCR value over pixel gap, check each value against the geometry, match totals against cumulative chains and scale x and y separately. OCR is exact on 90.2% of dimension texts; 62 of 200 plans are reconstructed fully correctly.\[708\]
  - Lin & Wang†, on scanned historical drawings: YOLOv11n detects dimension annotations, endpoints and axis-grid markers (mAP50 0.98–0.99), PaddleOCR reads 89.3% of dimension texts, and RANSAC combines one estimate per dimension. Shown on one worked example.\[707\]
  - Chang et al.†: read the dimension strips around the plan and drop per-dimension scales outside the mean ± 1σ. They report "scale calculation accuracy exceeding 95%" on more than 5,000 plans, with the metric undefined.\[706\]
  - Faltin et al.† vote for the pixel resolution over dimension lines read with EasyOCR, and state only "promising results".\[705\]
  - Sketch2BIM fits separate x and y scales by least squares, regularised towards isotropy.
- **Scale bars.**
  - No floor plan paper reads them.
  - EXSCLAIM!† (microscopy figures): detects bars and labels, reads the label and pairs them by box-centre distance. Bar length error 5.4%; 82% of labels read on 440 figures. Code GPL-3.0.\[704\]
  - Pang et al. detect scale-text regions (F1 0.97); WAFFLE has a scale class.
- **Scale notes.**
  - The title block's "Maßstab" field is readable by a fine-tuned open VLM (Peng et al. 2026, stage 1b).
  - AECV-Bench's OCR-style questions include the drawing scale in the title block.
- **Other cues.**
  - A reference line drawn by the user (Guo et al.).
  - Training labels in mm per pixel (Chen et al. 2023).
  - *Our evidence:* a scale proposal from detected door widths works on plans with doors and fails on churches and palaces ([review §2.4](reviews/2026-10-07-pipeline-and-pilot-v2.md#24-benchmark-samples-in-the-viewer)).
- **Gap:** only Talebi-Kalaleh & Mei measure scale error, on generated drawings. The others report OCR or detection accuracy, or one example.

### Stage 2: Text Layer

- **Read and remove text first.** Egiazarian et al. assume text is removed before vectorisation. Ahmed et al. separate text from graphics first.
- **Merging characters into words:** Ahmed et al. merge characters into words by horizontal and vertical smearing, and rotate words to horizontal before OCR.
- **OCR engine:**
  - PaddleOCR (Apache-2.0) is the self-hostable candidate. Its 3.0 report covers Chinese, English and Japanese; German, French and Italian come from its Latin model (PP-OCRv5) and from PP-OCRv6.
  - PP-OCRv6 comes in three sizes, from 1.5M to 34.5M parameters. All three include French, German and Italian; 46 Latin-script languages are covered by adding about 200 diacritical characters.
  - The medium model reports +4.6 detection Hmean and +5.1 recognition accuracy over PP-OCRv5 server, on in-house benchmarks only.
  - Its recogniser decodes with CTC, so dictionary-constrained decoding can be added. "²" is not mentioned.
- **Constrained decoding (Word Beam Search).**
  - CTC output is decoded against a dictionary held in a prefix tree, while characters outside words (digits, punctuation) stay free.
  - On IAM handwriting with a realistic dictionary, word error rate falls from 29.1% (best path) to 23.7%; with an ideal dictionary, to 10.2%. The dictionary-only mode takes 77–289 ms per text line, including the network.
  - For stamps, DE/FR/IT room usages form the dictionary, and room numbers, AOIDs and areas are never "corrected" into words (stage 7).
- **Text in technical drawings.** Four papers train drawing-specific readers, all on mechanical or structural drawings, none on room stamps:
  - Schlagenhauf et al. train a Faster R-CNN detector and a Keras-OCR recogniser on generated drawings: random part geometries, dimensions and tolerances, and elements cut from real drawings. On 171 objects in 9 real drawings they reach a detection rate of 81.9% and recognition of 79.3%, against 77.2% and 46.8% for Keras-OCR out of the box. Text is handled only horizontally and vertically.\[134\]
  - Khan et al. locate nine annotation types with an oriented-box detector, then parse each crop to JSON with an OCR-free Donut model. One shared model beats nine class-specific ones (e.g. surface roughness F1 94.6 vs 79.1). The headline F1 of 97.3% is for one class (GD&T).\[135\]
  - Ying et al. pair a slimmed DBNet detector (F1 95.9, against 90.7 for DBNet) with a transformer recogniser for rebar annotations on Chinese drawings. The recogniser reaches 96.3% on ground-truth crops, against 94.6% for SVTR and 92.4% for PARSeq; vertical text is rotated before reading.\[136\]
  - eDOCr2† groups dimension text on mechanical drawings and trains custom recognisers on synthetic data (MIT code), a recipe for Swiss dimension notation.
  - Together they document synthetic training data, rotated text and crop-then-parse for drawing text. No end-to-end accuracy is reported for real detection plus reading.
- **Text grammars:**
  - TextCAD parses annotations with a type–attribute grammar (with a validity mask). It distinguishes text at three scales: primitive (dimensions), instance (door codes) and region (room names). Room stamps are region-level text.
  - AECV-Bench's prompts list room labels in several languages ("Chambre", "Zimmer", "Camera", "WC", "SDB").
- **Practitioner view:** AxcelerateAI reads text before walls and seeds rooms from room tags.

### Stage 3: Walls and Wall Graph

- **Junctions as keypoints.**
  - CubiCasa5K, Raster-to-Vector and Raster-to-Graph predict typed junction heatmaps (I/L/T/X shapes in each orientation), but connect only axis-aligned junctions.
  - Pang et al. add sub-pixel offset regression and an annealed Gaussian width (σ from 3 to 1) for precise keypoints.
- **Walls that are not axis-aligned.**
  - Chen et al. 2023 classify candidate segments over junctions with a graph neural network:
    - wall sAP 84.9, door 97.5, window 88.4 at 8 px;
    - inclined outer walls are kept via convex-hull edges;
    - performance still drops with many inclined walls.
  - Lv et al. start from Douglas–Peucker polygons and optimise boundary, IoU (differentiable rendering) and a soft orthogonality term, so inclined walls survive. Without the orthogonality term, wall-junction accuracy drops from 0.96 to 0.67. Curved walls still fail.
- **Closure.** Raster-to-Vector's integer programme enforces junction degrees, mutual exclusion within 10 px, closed room loops and openings on walls. It raises junction precision from 70.7 to 94.7. The solver used, Gurobi, is commercial and can be replaced.
- **Thin walls and gaps.** A one-pixel gap in a wall lets a room leak into its neighbour.
  - TopoMortar compares eight losses on brick walls, where mortar lines partition bricks as walls partition rooms:
    - clDice gives the lowest topology errors (in-distribution Betti-0 error 1.17, against 3.31 for cross-entropy + Dice);
    - Skeleton Recall is the most robust to noisy labels;
    - colour augmentation plus self-distillation cuts the out-of-distribution Betti-0 error of plain cross-entropy + Dice from 197 to 70.
  - Chen et al. 2024 found no topology loss better than binary cross-entropy on historical maps: PQ 47.1, against 45.6 (BALoss), 36.9 (TopoLoss) and 36.0 (MOSIN).
- **Historical and hatched walls.**
  - Versailles-FP:
    - It separates filled (poché) walls from hollow walls drawn as parallel strokes.
    - Its label generator scores Dice 91% on filled walls but 34% on hollow walls of CVC-FP; the authors blame part of the gap on CVC-FP's ground truth, which fills hollow walls.
    - A U-Net reaches wall IoU 88.1 on the 500 palace plans (5-fold cross-validation, images downscaled to 512 px).
  - Trivi segments 11 hatch codes for materials on archival drawings, with the same U-Net/ResNet-34 as pilot v2. It reports mIoU 0.71–0.73 on the three tabulated drawings, apparently not held out.
  - Neither labels "hatching that is not wall", pilot v2's largest false-positive source on CubiCasa5K's architectural sheets ([review §2.2](reviews/2026-10-07-pipeline-and-pilot-v2.md#22-cubicasa5k-zero-shot-benchmark-new)).
- **Wall properties.**
  - Thickness from the intensity profile of the wall mask (CubiCasa5K).
  - Double strokes merged into a centre line with explicit tolerances: angle ≤ 1°, offset ≤ 0.5 ft (Sketch2BIM).
  - Merging and snapping of collinear primitives (Egiazarian et al.).
  - Structural walls as those thicker than each plan's 60% wall-thickness quantile (MSD).
- **Target representation.** FloorplanVLM describes walls by start and end points, thickness and curvature, with rooms as cycles of wall IDs, and says this makes topology "consistent by design". It is the closest published match to our data model.
- **Vector input.**
  - VecFormer has the best wall ("stuff") PQ without layer information (85.9 on FloorPlanCAD), but outputs labelled lines only, no centre lines or thickness.
  - CADSpotting turns wall lines into polygons by rasterising them and taking connected components.
  - VectorGraphNET classifies the paths of 76 CAD-exported university plans by geometry and stroke style alone. It reaches F1 0.95–0.96 for load-bearing, non-load-bearing and partition walls (proprietary data, 14 test plans).
- **Columns.**
  - Talebi-Kalaleh & Mei detect columns on vector framing plans as closed glyphs 60–2,000 mm long with an aspect ratio of at most 6, then type them by their width profile. They reach recall 0.922 and precision 0.997 on 50 held-out generated plans; most misses are in rotated wings.
  - DAFormer's Rare Class Sampling raises rare classes (rider, train, bike), not thin ones (pole). Columns are both, so sampling alone may not be enough.

### Stage 4: Openings

- **Detection is good for common doors, poor for rare types.**
  - Doors: AP 96.0 (Dodge et al.), sAP 97.5 (Chen et al.), PQ above 90 on vector data.
  - Sliding, revolving and rolling doors and plain openings without a door leaf: AP 35.9 and PQ 0–53.
  - Rezvanifar et al.'s survey found that classical symbol spotters reach good recall but poor precision on dense drawings. Their clutter-tolerant template matcher finds bi-fold doors despite hatching at the same angle on a renovation drawing, but is evaluated quantitatively only on synthetic plans.
  - Zhang 2026: openings are "a local detection problem rather than a sequence-emission problem" (opening F1 0.64 detected, 0.25 emitted).
- **Constraints.** Every opening must lie on a wall:
  - Raster-to-Vector: parallel and within 10 px.
  - CubiCasa5K: reject endpoints outside the wall mask.
  - Sketch2BIM: host wall chosen by adjacency plus tangent direction, with minimum clearances from wall ends.
  - ResPlan: openings snapped to the wall band, aligned in 99.94% of plans.
- **Representation.**
  - FloorplanVLM nests each opening in its wall, with a width and an offset along the centre line.
  - CADSpotting derives the door hinge and orientation from the arc–line relationship.
  - Raster2Seq emits doors and windows as labelled polygons after the rooms (window and door F1 77.8 on CubiCasa5K, RoomFormer 78.5). It "occasionally fails to accurately localize windows and doors", producing openings that cross rooms, and suggests modelling them separately.
- **Swing and doors versus windows.**
  - RoomFormer cannot tell swing direction, or single from double doors.
  - Raster-to-Vector's "window if one side is outside" rule fails for interior windows and courtyards.

### Stage 5: Stairs and Voids

- **Stairs.**
  - A single class in FloorPlanCAD (PQ 60.8–84.8 across papers), ArchCAD-400K and Raster-to-Vector.
  - Lumped into "Others" in Chen et al.'s data (0.14% of rooms).
  - No flights, direction or void.
- **Voids.** Not handled by any method. The MSD paper's listing of Swiss Dwellings subtypes includes staircase, shaft, void, "air" (presumably air space) and ramp, so Swiss Dwellings is a candidate source of supervision.
- **Text cues.** TextCAD shows stair keywords in annotations help.
- **This stage has no ready method and must be developed.** *Our evidence:* stair IoU is 0.96 on held-out renders but 0.38 zero-shot on CubiCasa5K, and pilot v2 finds voids only where a void label is printed.

### Stage 6: Rooms

- **Rooms as faces.**
  - Raster-to-Graph takes the shortest cycles of the wall graph; 67% of plans are structurally perfect, against 14.8% for Raster-to-Vector.
  - Sketch2BIM takes the bounded faces of a half-edge planar graph, merging slivers under 2 ft.
  - DeepFloorplan treats doors and windows as room boundaries, so rooms close at openings.
  - VectorFloorSeg works on vector walls:
    - it extends the wall lines until they intersect;
    - it classifies line segments as room boundaries and the resulting regions as room types, with a two-stream graph network.
  - On CubiCasa5K, VectorFloorSeg reaches test mIoU 62.5 and room integrity 67.5, against 57.1 and 41.9 for an image-based OCRNet. Its input walls come from the annotations, not from real CAD.
- **Rooms as regions of an edge map.**
  - Chen et al. 2024 extract closed shapes from a U-Net edge probability map with a Meyer watershed rather than by thresholding: PQ 46.7, against 41.2 with connected components. Adding augmentation and joint tuning gives 51.1.
  - The watershed bridges small gaps but "cannot recover lost edges".
  - Seeded at room stamps, the same step can split open-plan faces.
- *Our evidence:* pilot v2 forms rooms as free-space regions between walls and closed openings, not as graph faces ([review §3.2](reviews/2026-10-07-pipeline-and-pilot-v2.md#32-where-document-and-pilot-disagree)).
  - On the CAD print it finds 15 of 15 rooms, with mean IoU 0.89 and a median area error of 2.7%. On CubiCasa5K it matches 65% of rooms.
  - With perfect labels, the same post-processing finds rooms with recall 0.77 and precision 0.80. The misses are open-plan spaces that no wall separates ([review §8.1](reviews/2026-10-07-pipeline-and-pilot-v2.md#81-engineering-foundation-step-10-7-october-2026)).
- **Learned polygon decoders.**
  - RoomFormer, PolyRoom and FRI-Net reach Room F1 97–99 on synthetic density maps, but are capped at 20 rooms per scene and never tested on drawings. Transfer between their datasets drops IoU to 74–85.
  - Raster2Seq decodes rooms as polygon sequences with learned anchors from plan images:
    - room F1 88.7 on CubiCasa5K, against 83.5 for RoomFormer;
    - the gap widens on plans with many rooms (RoomFormer and FRI-Net drop beyond 15 polygons or 150 corners);
    - input is fixed at 256 px, with sequences of 512 tokens.
  - Zhang 2026 compares emitting and detecting geometry with the same network:
    - an emitting decoder wins on clean renders in the style it was trained on (ResPlan, wall F1 0.968 against 0.915);
    - detection wins on real scans and on unseen styles;
    - combining a room-centric decoder with wall-first detection adds about 7 wall F1 points on CubiCasa5K. The paper attributes most of that gain to ensembling, not to the two representations complementing each other.
  - Use learned decoders only as a fallback or second hypothesis where the walls do not close.
- **Regularisation.**
  - Douglas–Peucker simplification alone gives good IoU but poor corner angles (Angle F1 49.6 and 55.8).
  - PolyRoom's vertex selection by corner probability and an angle threshold, followed by Douglas–Peucker, does much better.
  - FRI-Net fits axis-aligned directions first and oblique ones later.
- **Avoid** splitting rooms that carry several labels with axis-aligned cuts (Ahmed et al. 2012, Raster-to-Vector): it invents walls. Flag such rooms for review instead.
- **Polygonisation costs accuracy.** CubiCasa5K room IoU falls from 57.5 to 49.3 when segmentations are turned into polygons. This supports bounding rooms by the wall faces rather than vectorising room masks.

### Stage 7: Room Attributes

- **Ahmed et al. 2012** assign OCR text to rooms by containment, break ties by distance to the room centre, and match usage to a dictionary with Levenshtein distance.
  - 736 of 894 labels were correct (82.3%), judged by the authors.
  - They strip digits, which would destroy room numbers.
- **Recent pipelines** use three devices\[63\]\[64\]:
  - "text centre contained or conservatively nearby" rules;
  - fuzzy matching against label dictionaries (Levenshtein ratio ≥ 0.55 in Ayanzadeh & Oates);
  - multi-engine OCR fallback.

  Decoding against the dictionary (Word Beam Search, stage 2) avoids rewriting numbers after the fact. The resulting procedure for BBL is in [pipeline stages 2 and 7](pipeline.md#7-room-attributes).
- **Legends.** WAFFLE parses legend key–value pairs with an LLM plus regular expressions and grounds each key to its OCR box. The same pattern suits room stamps and legends.
- **Usage from text.** AECV-Bench shows usage classes anchored on label text are far more reliable than symbol-based ones (bedroom 0.76–0.91 against door 0.09–0.39 exact).
- **Usage from context.**
  - Paudel et al. infer room type from the room graph (graph neural network, 81% accuracy, residential, vector input). This is a fallback for rooms without stamps.
  - Lv et al. vote room types from detected text and symbol boxes, without reading the text.
- **Gap:** no method parses multi-field stamps, handles old room numbers, or links rooms to external registers.

### Stage 8: Derived Outputs

- **Connectivity schemas.**
  - **MSD:** a "passage" edge when two areas are within 0.04 m; a "door" edge when a door polygon is within 0.05 m of both areas; front doors and zones as attributes.
  - **ResPlan:** via_door, via_window, adjacency and direct edges. Shared-boundary detection reaches F1 0.971 and edge typing 0.85.
  - A BBL schema can combine these and add vertical links (stairs, lifts, ramps), which neither models.
  - Downstream use: FloorPlan-VLN guides a robot with plans that carry typed room regions; removing the plan cuts navigation success by 41%. Room polygons with usage and IDs are what such applications consume.
- **Linking storeys by grid axes.**
  - BlueprintAgent reads grid axes on 300 real scanned structural sheets from 20 projects: a vision probe finds label circles and lines, and a multimodal LLM reads the labels.\[710\]
  - Validators check span ratios and enforce consistent axes across floors. Axis F1 is 1.000, against 0.987 for a single zero-shot model.
  - Lin & Wang detect axis-grid markers with mAP50 0.993.
  - Shared axis labels are the natural anchor for registering storeys, and for BBL's optional `V_ACHSEN` layer.
- **Compute with libraries, not language models.** FloorplanQA shows LLMs add up overlapping areas instead of taking their union. Even LLM-written code fails on edge cases such as path planning. Use Shapely and networkx.
- **Gross floor area.** No paper computes it to a standard.
  - Ahmed et al.'s convex hull and CADSpotting's "largest connected component is the floor" are crude.
  - Guo et al. sum connected-component areas at a manually set scale (one apartment within 0.81%). A local LLM with retrieval then screens the rooms against design rules, evaluated only by example.
  - The GF rules in [pipeline.md](pipeline.md) (outer wall faces, voids cut out) have no counterpart in the literature.

### Stage 9: Scale Confirmation and QA

- **Scale confirmation** after rooms exist (the proposal is stage 1c):
  - pixel density from room areas read by OCR (Dodge et al.);
  - metres per pixel from the stated gross area (ResPlan);
  - overall dimensions checked against cumulative dimension chains (Liang et al.).
- **QA rules.**
  - ResPlan: door on a wall band; door connects at least two rooms; summed room area within ±25% of the stated total.
  - Ayanzadeh & Oates: connectivity by breadth-first search, door/edge consistency, symmetry, no isolated nodes, at most two automatic retries.
  - Sketch2BIM: schema and topology validator.
  - FloorplanVLM: validity checks (closed, watertight polygons).
  - PolarSym's idea of symmetry and repetition consistency could become a plausibility rule.
- **Metrics worth adopting** (§5.5): Angle F1 (PolyRoom, RoomFormer); Hausdorff and mean minimal deviation for vectors (Egiazarian et al.); length-weighted PQ for vector primitives (FloorPlanCAD); PQ for rooms as instances (Chen et al. 2024); Betti errors for wall masks (TopoMortar); exact match plus MAPE for counts (AECV-Bench); Hausdorff 95 of drawing-region boundaries (MapSeg); scale error per input class.

### Stage 10: Review and Export

- **Interactive correction.**
  - Sketch2BIM's natural-language edit loop, with a validator and deterministic IDs, converges in 3–4 iterations.
  - Raster-to-Graph conditions on a partial graph, which suits interactive correction.
  - Talebi-Kalaleh & Mei let a VLM act only through 12 typed operations, behind guards on ink coverage and confidence. A failed proposal leaves the layout unchanged ("fail closed"). This is a pattern for a self-hosted assistant, although the paper uses a cloud model.
  - BlueprintAgent re-reads only the local region that an engineering validator flags, instead of re-running the whole sheet. The same idea fits QA failures such as a stamp area that disagrees with its polygon.
- **Assist, don't automate.** AECV-Bench recommends counting features as "assistive tools that surface candidates and uncertainty, rather than as autonomous extractors".
- **Annotation practice.**
  - ArchCAD-400K: layer-based auto-labels plus correction in a vector editor by 10 drafters with automatic compliance checks; 800 person-hours for 413K chunks.
  - CubiCasa5K: two-stage QA of labels.
  - Raster-to-Vector: automatic constraint checks, then manual correction.
  - Versailles-FP: semi-automatic wall labels (steerable filters, then manual correction), about 10 times faster than labelling by hand.
  - Petitpierre & Guhennec† (from the abstract and index note):
    - use 2–3 visually homogeneous classes;
    - annotate only what is visible;
    - work iteratively;
    - about a dozen sheets bootstrap a model.

## 5. Cross-Cutting Themes

### 5.1 Generalisation Evidence

| Source | Setting | Result |
|---|---|---|
| Dodge et al. 2017 | Walls, trained on one style, tested on another (CVC→R-FP, R-FP→CVC) | Mean IoU 76.1 / 81.7; with pooled training 90.5 / 92.9 |
| CADSpotting | Trained on FloorPlanCAD, tested on office/campus plans (LS-CAD) | SymPoint 83.3 → 33.2 PQ; CADSpotting 87.4 → 52.9 (60.1 with sliding windows); joint training 75.5 |
| ArchCAD-400K | Trained and tested on diverse, mostly non-residential CAD | Best PQ without layer information 70.6, against 86.2 on FloorPlanCAD |
| WAFFLE | CubiCasa5K model on historical drawings in many styles | IoU walls 0.488, windows 0.202, doors 0.099 |
| Raster2Seq | Trained on CubiCasa5K, zero-shot on WAFFLE (interior IoU) | 73.9; RoomFormer 60.5; FRI-Net 56.7; CubiCasa5K model 46.1 |
| Zhang 2026 | Same network read out as emitted sequence or as detected graph, on real scans and on renders | Detection ahead by 2.7–5.1 wall F1 on real CubiCasa5K scans; emission ahead by 5.3–8.2 on renders in its training style |
| ResPlan | Room-type transfer between datasets (ResPlan↔MSD) | Accuracy 0.322 / 0.190 |
| RoomFormer, PolyRoom, FRI-Net | Synthetic (Structured3D) → real scans (SceneCAD) | IoU 74.0 / 85.2 / 80.6 |
| Raster-to-Vector | Other sources (Rent3D, web) | Geometry mostly holds; icon and room types often wrong (qualitative) |
| Semap | Historical maps from more than 50 institutions | mIoU 74.2 on a random test split; no held-out collection tested |
| MIC on HRDA | Synthetic street scenes → real (GTA→Cityscapes) | mIoU 75.9; supervised in-domain HRDA 81.6 |
| *Pilot v2 (our evidence)* | Style-randomised Swiss Dwellings renders → 400 CubiCasa5K test plans, zero-shot | Walls / doors / windows IoU 0.57 / 0.49 / 0.57 (CubiCasa's in-domain model 0.73 / 0.54 / 0.67); 65% of rooms matched, median area error 2.9% |
| *Pilot v2 (our evidence)* | Same model → three unseen pilot sheets | Wall IoU 0.87 (CAD print), 0.58 (poché scan), 0.29 (survey scan with ceiling ornament) |
| *Pilot v2 (our evidence)* | Held-out renders → real sheets | Walls 0.91 on renders; the best checkpoint on renders was not the best on the CAD print |

Reported in-domain scores are not planning targets. A held-out set of real BBL sheets, plus WAFFLE and CVC-FP as out-of-domain style tests, is mandatory. Synthetic validation shows that training works; it cannot select the model ([review §2.3](reviews/2026-10-07-pipeline-and-pilot-v2.md#23-synthetic-validation)).

### 5.2 Dependence on Layer Information (vector methods, FloorPlanCAD)

| Model | PQ without / with layers | Wall-type ("stuff") PQ without / with |
|---|---|---|
| SymPoint-V2 | 83.2 / 90.1 | 49.3 / 80.8 |
| DPSS (ArchCAD-400K) | 86.2 / 89.5 | 64.7 / 79.7 |
| CADSpotting | 87.4 / – | 71.5 / – |
| VecFormer | 88.4 / 91.1 | 85.9 / 90.4 |
| TextCAD (own split, own reimplementations) | 91.09 / 92.67 | 90.97 / 92.32 |

Two older spotters show the same pattern from the other side:
- GAT-CADNet uses geometry only and reaches PQ 73.7.
- CADTransformer's 68.9 includes a training augmentation that recombines CAD layers; without it, 67.3.

VectorGraphNET uses no layers either, but reports F1 rather than PQ (§6.1).

### 5.3 Training Data and Auto-Labelling

- **ArchCAD-400K's annotation engine is a blueprint for labelling BBL's compliant DWGs:**
  - map layer names to classes with regular expressions;
  - use CAD blocks as object instances;
  - reject drawings with more than 5% unmapped layers;
  - correct the result in a vector editor.

  It also warns that labels derived from layers alone are noisy.
- **Style augmentation (Chen et al.):** levels of detail, furniture on or off, ruler styles, recoloured lines, rooms, labels and background, hollow and filled walls, each image augmented five times. This is a checklist for the BBL renderer.
- **Synthetic scans (Egiazarian et al.):** render onto real scanned paper backgrounds, then blur, distort and add noise.
- **Procedural textures from real scans (Semap).**
  - What is randomised: fills, dot patterns, hatchings and texture masks cut from annotated real scans; colours sampled from a mixture fitted to real images; stroke widths; greyscale and JPEG artefacts.
  - Synthetic samples made up 90.9% of the training set, followed by fine-tuning on real labels only.
  - For BBL: cut poché, hatching and ornament from archive scans and composite them into both wall and non-wall regions. Pilot v2's renderer hatches only walls ([review §4.2](reviews/2026-10-07-pipeline-and-pilot-v2.md#42-renderer-synthpy)).
- **Data mix (FloorplanVLM):** noisy screenshots plus exactly aligned re-renders, sampled by clustering on outline shape and room adjacency to keep diversity.
- **Few real labels.** Trivi trained on 19 drawings. Petitpierre & Guhennec† report that about a dozen sheets bootstrap a model. Versailles-FP shows semi-automatic pre-labels with manual correction.
- **Licence-clean Swiss data:** Swiss Dwellings, the parent of MSD, includes office, meeting room, archive, staircase, shaft, void, ramp and column subtypes, according to the MSD paper. *Our evidence:* pilot v2 trains on Swiss Dwellings v3.0.0 only, and has prepared 72 storeys from permissively licensed IFC-Bench models as a non-residential source.

### 5.4 Domain Adaptation and Generalisation

Pilot v2's weak spots are hatching and ornament read as walls, thin walls and columns ([review §2.2](reviews/2026-10-07-pipeline-and-pilot-v2.md#22-cubicasa5k-zero-shot-benchmark-new)). They are the kind of errors that adaptation methods from street-scene segmentation address. None of these methods has been tested on drawings.

- **Self-training with masked consistency (MIC).**
  - How it works: an EMA teacher pseudo-labels the unlabelled target image. The student must predict the same labels on a copy with 70% of its 64 px patches masked, so it has to infer from context.
  - Results on GTA→Cityscapes: it adds 1.2–4.7 mIoU to every method it was combined with, CNN models included. For example, the DAFormer recipe on DeepLabV2 rises from 56.0 to 59.4.
  - Masking is the decisive component: without it, mIoU falls by 20 points.
  - Mask only the target domain: masking the synthetic source as well cost 2.7 points on GTA→Cityscapes, where synthetic context differs from real.
  - Cost: about 24% more training time; inference is unchanged.
  - For BBL: renders are the source and unlabelled archive scans the target. "Hatching inside a room is not wall" is a decision from context.
- **Multi-resolution training (HRDA).** Detail and context crops fused by learned attention help thin classes on large images (stage 1). Training took about 32 hours on one 24 GB GPU.
- **Training strategies (DAFormer).**
  - Rare Class Sampling adds 5.8 mIoU for adaptation; an ImageNet feature distance on object classes adds 3.5.
  - Rare Class Sampling lifts rare classes, not thin ones.
  - CNN encoders adapt worse than Transformer encoders: in the same setting, ResNet-101 reaches 50.9 mIoU after adaptation, MiT-B5 58.2. Pilot v2 uses a ResNet-34. MiT encoders in segmentation_models_pytorch are under NVIDIA's non-commercial licence ([open solutions](solutions-open.md#6-licence-pitfalls)).
- **Synthetic data for generalisation (Semap):** +5.1 mIoU from synthetic pretraining (74.2 against 69.1), but measured on a random split of the same collections (§5.1).
- *Our evidence:* style randomisation alone transfers (§5.1). Where it fails, the failures are context or rarity problems: ceiling ornament inside rooms, hatched roof slopes, columns. These are the cases the methods above target.
- **Review proposal:** self-training on the unlabelled archive, keeping only pseudo-labels on which flipped predictions agree, plus learning curves over 10, 50 and 200 labelled sheets ([review §4.3](reviews/2026-10-07-pipeline-and-pilot-v2.md#43-model-and-training)).
- **Open question:** whether these methods help on line drawings, where context is sparse and classes are defined by drawing convention.

### 5.5 Evaluation Metrics

| Metric | What it measures | Used by | Use for BBL |
|---|---|---|---|
| Pixel IoU / mIoU | Overlap per class | Most raster papers; pilot v2 | Walls, stairs; punishes thin walls (one pixel per face costs about 30% on a 6 px partition, [review §4.6](reviews/2026-10-07-pipeline-and-pilot-v2.md#46-evaluation-code)) |
| Panoptic quality (PQ = SQ × RQ) | Instance matching and segmentation quality in one number; catches merged and split instances | Symbol spotting on FloorPlanCAD (length-weighted); Chen et al. 2024 for closed shapes | Rooms as instances, walls as "stuff" |
| Room, corner and angle F1 | Matched room polygons, corners and corner angles | RoomFormer, PolyRoom, FRI-Net, Raster2Seq | Room outlines; Angle F1 for regularisation |
| Room integrity | Mean IoU of matched rooms × room F1, with one-to-one matching and the same room type | VectorFloorSeg | Rooms with usage |
| Betti-0 / Betti-1 error | Wrong number of connected components and enclosed regions | TopoMortar | Wall masks: a leak removes an enclosed region, a hatching loop adds one |
| Plain vs weighted F1 | Effect of class weighting | VectorGraphNET (weighted), GAT-CADNet (length-weighted) | Report plain F1 next to any weighted score |
| Junction and opening accuracy within half a wall width | Vector topology | Lv et al., Raster-to-Vector | Wall graph |
| Hausdorff distance, mean minimal deviation | Vector geometry error | Egiazarian et al. | Wall centre lines |
| CER / WER | Text recognition | Word Beam Search, OCR reports | Stamps, per field (number, usage, area) |
| Edit cost | Weighted sum of the moves, retypes, creations and deletions needed to turn the output into the reference | Zhang 2026 | Proxy for correction minutes per sheet, the decisive metric in [pipeline §8](pipeline.md#8-evaluation) |
| Exact match + MAPE | Counts | AECV-Bench | VLM checks |
| mAP per region class; Hausdorff 95 of region boundaries | Sheet-layout detection; drawing-region masks | FELD, Lombardi et al.; ICDAR 2021 MapSeg | Stage 1b, with the share of detections outside the drawing regions |
| Scale error; area error in m² | Metric accuracy | Talebi-Kalaleh & Mei (scale); Guo et al. (one plan) | Decisive for BBL; reported by almost no paper |

- **Averaging changes the picture.** Semap scores mIoU 74.2 averaged per sample but 57.4 per class over the dataset. Every score should state its averaging.
- **Reference quality changes it too.**
  - Zhang 2026 found that CubiCasa5K's room polygons rarely close into wall cycles, and that its derived centre lines carry spurs.
  - Re-scored against the corrected labels, one fixed model rises from wall F1 0.74 to 0.825 and opening F1 0.29 to 0.66.
  - Pilot v2's CubiCasa5K benchmark scores against the original labels, which the review already calls simplified redrawings ([review §2.2](reviews/2026-10-07-pipeline-and-pilot-v2.md#22-cubicasa5k-zero-shot-benchmark-new)). Re-scoring against the corrected labels would show how much of the gap to the in-domain model is label convention.
- **Pilot v2's harness** (review step 1c) is to add one-to-one matching, boundary metrics and area in m², next to the stage metrics in [pipeline §8](pipeline.md#8-evaluation).

### 5.6 Licences and Availability (as found)

| Resource | Status |
|---|---|
| Swiss Dwellings / MSD | CC BY 4.0 (Zenodo, 4TU) |
| ResPlan | CC BY 4.0 on GitHub, CC BY-NC-SA 4.0 on Kaggle\[103\]\[104\]; provenance is the authors' legal position, collection scripts not released |
| FloorPlanCAD | Annotations CC BY-NC 4.0; drawings not owned by the authors; the first release removed text, the November 2021 release keeps text, CAD layers, dimension chains and axes |
| ArchCAD-400K | CC BY-NC 4.0, gated, only a subset released; paper cites "academic fair use"; code under an "academic use" licence\[26\] |
| CubiCasa5K | CC BY-NC 4.0 for code and data on GitHub\[5\]; CC BY-NC-SA 4.0 on Zenodo |
| Raster-to-Vector / Raster-to-Graph data | LIFULL HOME'S via NII IDR, for universities and public research institutions only\[107\] |
| WAFFLE | Per-image Wikimedia Commons licences |
| Versailles-FP | "Available under agreement for the academic research community"\[112\] |
| Semap | Data and model on Zenodo; the paper names no licence; images keep their own rights |
| Chen et al. 2024 (maps) | Data on Zenodo; code repository without a licence |
| ICDAR 2021 MapSeg | Data CC BY 4.0 (Zenodo) |
| Sheet-layout and title-block data (FELD, Lombardi et al., Peng et al. 2026) | Not released; Peng et al.'s bridge drawings are confidential |
| LRFP (Chen et al. 2023), RFP (Lv et al.), Trivi's drawings | No URL, licence or release statement in the papers |
| VectorGraphNET university plans; FloorplanVLM training data | Proprietary |
| Code: VecFormer, PaddleOCR (PP-OCRv5, PP-OCRv6) | Apache-2.0 |
| Code: Raster2Seq, TopoMortar, Word Beam Search, eDOCr2 | MIT (repositories, per the research index); Raster2Seq's weights are trained on NC data |
| Code: MIC, HRDA | Mixed: some files under non-commercial licences (SegFormer, AdaptSegNet); weights for street scenes only. Reimplement |
| Code: SymPoint-V2, VectorFloorSeg, CADSpotting | No licence file\[181\]\[176\]\[146\] |
| Code: EXSCLAIM! | GPL-3.0 (per the research index); reimplement the scale-bar reader |
| No code | VectorGraphNET, Talebi-Kalaleh & Mei (benchmark promised), Lv et al., Guo et al., TextCAD, PolarSym, FloorplanVLM (official), FELD, Lombardi et al., Peng et al. 2025 and 2026, Liang et al., Lin & Wang, Chang et al., BlueprintAgent |

## 6. Models and Code by Family

The tables list what can be downloaded, trained or reimplemented, with licences as found in October 2026. Activity, weight provenance and the Hugging Face fine-tunes are in [open solutions §4](solutions-open.md#4-implementations-by-category). Ready-to-use VLMs and OCR engines are in [open solutions §5.4–5.5](solutions-open.md#54-vision-language-models).

### 6.1 Vector Symbol Spotting and Primitive Graphs

All rows take vector primitives as input, so they apply to DWGs and vector PDFs but not to scans. BBL's archive DWGs have inconsistent layers, so the scores without layer priors are the realistic expectation.

| Model | Task | Inputs | Code / weights | Training data and licence | PQ on FloorPlanCAD (without / with layer priors) | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [FloorPlanCAD baseline](https://arxiv.org/abs/2105.07147) (ICCV 2021) | Introduced panoptic symbol spotting (CNN-GCN)\[19\] | Primitives | Dataset site only | FloorPlanCAD; annotations CC BY-NC 4.0, drawings not owned by authors\[6\] | 59.5 / –\[22\] | Superseded | Low |
| [CADTransformer](https://github.com/VITA-Group/CADTransformer) (CVPR 2022) | Panoptic symbol spotting: primitive tokens with image features from a rendering | Primitives + render | [VITA-Group/CADTransformer](https://github.com/VITA-Group/CADTransformer), MIT\[20\]; no trained weights | FloorPlanCAD (NC) | 68.9 / –\[22\] (67.3 without layer-based augmentation) | Multi-GPU training; superseded | Low |
| [GAT-CADNet](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_GAT-CADNet_Graph_Attention_Network_for_Panoptic_Symbol_Spotting_in_CAD_CVPR_2022_paper.html) (CVPR 2022) | Panoptic symbol spotting: graph attention, instances as subgraphs | Primitives (geometry only) | No official code; unofficial reimplementation without licence\[180\] | FloorPlanCAD (NC) | 73.7 / –\[22\] | Graph size limited by GPU memory; superseded | Low |
| [SymPoint](https://arxiv.org/abs/2401.10556) (ICLR 2024) | Panoptic symbol spotting (primitives as points) | Primitives | [nicehuster/SymPoint](https://github.com/nicehuster/SymPoint) | FloorPlanCAD (NC); code under IDEA's non-commercial research licence\[143\] | 83.3 / –\[22\] | Superseded by SymPoint-V2; non-commercial code | Low |
| [SymPoint-V2](https://arxiv.org/abs/2407.01928) | SymPoint + layer-feature encoding | Primitives (+ layer information) | [nicehuster/SymPointV2](https://github.com/nicehuster/SymPointV2), trained models\[23\] | FloorPlanCAD (NC); no licence file in the repository\[181\] | 83.2 / 90.1\[22\]; 60.5 without priors on ArchCAD-400K\[24\] | Its gain depends on layer information that archive DWGs lack; no column class | Low (no licence) |
| [DPSS](https://arxiv.org/abs/2503.22346) (ArchCAD-400K baseline) | Panoptic symbol spotting fusing primitive and rendered-image features | Primitives + rendered image | [ArchiAI-LAB/ArchCAD](https://github.com/ArchiAI-LAB/ArchCAD), "ACADEMIC USE LICENSE"\[26\]; third-party weights without licence\[182\] | ArchCAD-400K (CC BY-NC 4.0, gated)\[28\]; FloorPlanCAD (NC) | 86.2 / 89.5\[22\]; 70.6 without priors on ArchCAD-400K\[29\] | 8× A800 training\[29\]; Chinese drafting conventions | Low (only spotter with a column class, but academic-only code and data) |
| [CADSpotting](https://arxiv.org/abs/2412.07377) | Panoptic symbol spotting on large drawings | Primitives | [yangfy2023/CADspotting](https://github.com/yangfy2023/CADspotting) (2026), no licence file; based on OneFormer3D, CC BY-NC 4.0\[146\]\[147\] | FloorPlanCAD (NC) + LS-CAD (45 office/campus plans ≥1,000 m², not released)\[30\] | – / 88.9\[22\] | Heavy 3D point-cloud stack | Low (Medium if relicensed) |
| [VecFormer](https://arxiv.org/abs/2505.23395) (NeurIPS 2025) | Panoptic symbol spotting with line-based primitives | Primitives | [WesKwong/VecFormer](https://github.com/WesKwong/VecFormer)\[31\] | FloorPlanCAD (NC); code Apache-2.0 | 88.4 / 91.1\[22\] | 8× A100, 500 epochs\[22\]; no column class; no weights released (October 2026) | Medium (best without priors; permissive code) |
| [TextCAD](https://arxiv.org/abs/2607.12678) (Jul 2026) | Panoptic symbol spotting using primitives and text annotations jointly\[4\] | Primitives + text | No public code found | FloorPlanCAD-V2, CubiCasa5K (NC)\[32\] | State of the art claimed on its own benchmarks\[4\] | Recent preprint | Low (Medium if released; uses room text) |
| [PolarSym](https://arxiv.org/abs/2608.11793) (Aug 2026) | CAD plan parsing with polar geometry-aware attention | Primitives | No public code found | Unnamed public CAD dataset | +1.73 PQ over a reproduced SymPoint-V2 | Recent preprint; inconsistent tables | Low |

Primitive graph segmentation classifies paths or faces instead of symbol instances:

| Model | Task | Inputs | Code / weights | Training data and licence | Reported results | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [VectorFloorSeg](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_VectorFloorSeg_Two-Stream_Graph_Attention_Network_for_Vectorized_Roughcast_Floorplan_Segmentation_CVPR_2023_paper.html) (CVPR 2023) | Room segmentation: wall lines as room boundaries, regions between them as room types (two-stream graph attention)\[684\] | Wall line segments + 256 px render | [DrZiji/VecFloorSeg](https://github.com/DrZiji/VecFloorSeg), no licence file\[176\] | CubiCasa5K (NC), R2V (R); walls taken from the annotations | CubiCasa5K test mIoU 62.5, room integrity 67.5 (OCRNet 57.1, 41.9) | Clean walls only; no openings; curves as polylines | Low (method reference for rooms from vectors) |
| [VectorGraphNET](https://arxiv.org/abs/2410.01336) (2024) | Path classification of PDF drawings converted to SVG; hierarchical labels; 1.3M weights\[685\] | Vector PDF → SVG paths | No code released | FloorPlanCAD (NC); 76 plans of 10 university buildings, proprietary, labels from CAD layers | FloorPlanCAD weighted F1 89.0, plain F1 79.4 (SymPoint 85.5 / 86.8); university plans accuracy 0.97, macro F1 0.82 | Text drawn as geometry causes errors; weighting differs between compared methods | Low (Medium as a method for vector PDFs) |

These models label primitives (lines and arcs), and VectorFloorSeg labels faces; none outputs room stamps. They are best used to recover doors, windows, furniture and stairs where blocks are exploded or unnamed. Columns need a model trained on ArchCAD-400K classes (DPSS), since FloorPlanCAD has no column class.

The ArchCAD-400K drop comes from larger, more diverse drawings within one dataset; the models were trained and tested on it. Transfer to unseen conventions, such as Swiss federal plans, has not been measured, so expect to fine-tune on BBL-labelled data. Practitioner integrations exist, e.g. a DWG → SymPointV2 → JSON pipeline and an "ArchCAD-gpu" repository for US PDF plan sets whose README states "A model trained on them is not cleared for commercial use".\[33\]\[34\]

### 6.2 Raster Models and Object Detection

No open raster model below is rated High: all are trained on residential plans in a few drawing styles, and their weights inherit non-commercial or research-only terms. The more promising route is training BBL's own models on vector data rendered in randomised graphical styles, which [pilot v2](../pilot/v2-pipeline/README.md) does (§5.1). The building blocks are in [open solutions §5.3](solutions-open.md#53-building-blocks-for-training-and-preprocessing); Hugging Face checkpoints, all trained on NC data or data of unknown provenance, are in [open solutions §4.5](solutions-open.md#45-hugging-face-models-and-spaces).

| Model | Task | Inputs | Code / weights | Training data and licence | Reported results | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [Raster-to-Vector](https://github.com/art-programmer/FloorplanTransformation) (Liu et al. 2017) | Junction detection + integer programming → vector plan | Raster | [art-programmer/FloorplanTransformation](https://github.com/art-programmer/FloorplanTransformation)\[169\] | Code MIT; data and weights R2V/LIFULL (research terms) | Junction precision 70.7 → 94.7 with IP | Manhattan assumption\[35\]; residential; old code; use the PuLP solver variant | Low (reuse the IP formulation) |
| [DeepFloorplan](https://arxiv.org/abs/1908.11025) (ICCV 2019) | Multi-task room-boundary + room-type segmentation\[36\] | Raster | [zlzeng/DeepFloorplan](https://github.com/zlzeng/DeepFloorplan) (pretrained); TF2 rewrite [zcemycl/TF2DeepFloorplan](https://github.com/zcemycl/TF2DeepFloorplan)\[37\] | Code GPL-3.0 (also the TF2 rewrite); data R2V, R3D (residential; research terms) | Mean IoU 0.66–0.76 | Pixel masks; no room attributes or columns | Medium (baseline) |
| [CubiCasa5K model](https://arxiv.org/abs/1904.01920) (2019) | Multi-task segmentation of rooms, walls, icons, openings\[38\] | Raster | [CubiCasa/CubiCasa5k](https://github.com/CubiCasa/CubiCasa5k)\[38\] | CubiCasa5K, CC BY-NC(-SA) 4.0, Finnish residential; the NC licence covers the code too\[5\]\[39\] | Room mean IoU 57.5 (49.3 polygonised) | Non-commercial; residential | Medium (benchmark baseline) |
| [Raster-to-Graph](https://doi.org/10.1111/cgf.15007) (EG 2024) | Autoregressive structural-graph prediction\[40\] | Raster | [SizheHu/Raster-to-Graph](https://github.com/SizheHu/Raster-to-Graph), GPL-3.0 | >10,000 residential plans from LIFULL; data on request\[40\] | Room F1 84.7; 67% structurally perfect | Residential, Manhattan, ≤ 50 nodes | Medium |
| [Raster2Seq](https://arxiv.org/abs/2602.09016) (SIGGRAPH 2026) | Rooms, doors and windows as one labelled polygon sequence; anchor-guided autoregressive decoder\[683\] | Raster (256 px) | [Cornell-VAILab/Raster2Seq](https://github.com/Cornell-VAILab/Raster2Seq), MIT; weights trained on NC/R data | Structured3D (NC), CubiCasa5K (NC), Raster2Graph/LIFULL (R) | Room F1 88.7 on CubiCasa5K (RoomFormer 83.5); WAFFLE zero-shot interior IoU 73.9 | 256 px; openings may cross rooms; 0.52 s per plan | Medium (retrain on own renders as a second room hypothesis) |
| [Zhang 2026](https://arxiv.org/abs/2608.25608) (readout comparison) | One network (Raster2Seq backbone) read out as an emitted wall sequence or as junction and centre-line heatmaps assembled into a graph; fusion with a room-centric decoder\[132\] | Raster (256 px) | [Cyprinus12138/fpvec-lab](https://github.com/Cyprinus12138/fpvec-lab): code, corrected CubiCasa5K labels, ResPlan-FP announced; the README declares MIT for code, CC BY 4.0 for ResPlan-FP and CC BY-NC-SA 4.0 for the corrected labels, but the repository holds only the README (no code, no LICENSE file; October 2026) | CubiCasa5K (NC), ResPlan, procedural residential plans | Fused wall F1 0.853 on CubiCasa5K; opening F1 0.64 detected vs 0.25 emitted | 256 px; residential; single author | Medium (edit-cost metric, corrected labels, test set) |
| [Lv et al.](https://openaccess.thecvf.com/content/CVPR2021/html/Lv_Residential_Floor_Plan_Recognition_and_Reconstruction_CVPR_2021_paper.html) (CVPR 2021) | Segmentation, scale from dimension strings, vectorisation allowing inclined walls\[687\] | Raster | No code found | RFP: 7,000 Chinese residential images; no release statement | mIoU 0.85; wall-junction accuracy 0.96 | Scale evaluated only in the supplement; curved walls fail | Low (method reference for scale and vectorisation) |
| [MuraNet](https://arxiv.org/abs/2309.00348) | Joint segmentation + detection of walls, doors, windows\[41\] | Raster | No public code found | CubiCasa5K (NC) | Wall IoU 78.4; door/window AP50 91.7 | One dataset | Low |
| [FloorplanVLM](https://arxiv.org/abs/2602.06507) (2026) | Fine-tuned VLM (SFT + GRPO) outputs vectors as JSON\[42\] | Raster | No official code; community reimplementation [manitocross/floorplan-vlm-training](https://huggingface.co/manitocross/floorplan-vlm-training) (Qwen2.5-VL-3B + LoRA on CubiCasa5K, both NC)\[43\]\[151\] | Proprietary Floorplan-HQ-300K\[44\] | 92.52% external-wall IoU on own FPBench-2K; opening F1 0.77 (Manhattan subset)\[42\] | Own benchmark; no columns, furniture or stairs | Medium (shows the fine-tuned-VLM route) |
| [Pixel-wise symbol spotting](https://arxiv.org/abs/2404.10985) (Pang et al. 2024) | Keypoint-based symbol spotting on images rendered from CAD | Raster (rendered CAD) | Dataset only\[183\] | Telecom equipment-room drawings | Region F1 0.95–1.00; keypoint F1 0.84 | Not building floor plans | Low |
| [Versailles-FP](https://arxiv.org/abs/2103.08064) U-Net (ICDAR 2021) | Wall segmentation of historical plans\[113\] | Raster (downscaled to 512 px) | No code link | Versailles-FP, academic agreement\[112\] | Wall IoU 88.1, Dice 93.3 (5-fold) | Walls only; one archive | Low (Medium as a test set) |
| [RoomFormer](https://arxiv.org/abs/2211.15658) (CVPR 2023) / [PolyRoom](https://arxiv.org/abs/2407.10439) (ECCV 2024) / [FRI-Net](https://arxiv.org/abs/2407.10687) (ECCV 2024) | Room-polygon reconstruction | Point-cloud density maps | [ywyue/RoomFormer](https://github.com/ywyue/RoomFormer) (MIT), [3dv-casia/PolyRoom](https://github.com/3dv-casia/PolyRoom), [Daisy-1227/FRI-Net](https://github.com/Daisy-1227/FRI-Net) (no licence files)\[50\]\[51\]\[52\] | Structured3D (non-commercial) / SceneCAD\[3\]\[54\] | FRI-Net room F1 99.1 on Structured3D\[53\] | Not trained on drawings; 20-room cap; pinned to PyTorch 1.9 / CUDA 11.1\[50\] | Low (RoomFormer as a polygon decoder to retrain) |

HEAT, Floor-SP and MonteFloor belong to the same density-map family. None of these has been shown to work on archive plans. Guo et al. (YOLOv8 symbols and a ResNet-34 U-Net) and Trivi (U-Net/ResNet-34 on hatch codes) release no code; they are summarised in §8.

### 6.3 Vision-Language Models: Benchmarks and Research Pipelines

Ready-to-use VLMs are listed in [open solutions §5.4](solutions-open.md#54-vision-language-models).

| Benchmark | Task | Inputs | Finding | Implication | Relevance |
|---|---|---|---|---|---|
| [AECV-Bench](https://arxiv.org/abs/2601.04819) (January 2026; [harness](https://github.com/AECFoundry/AECV-Bench)) | Counting and OCR-style QA on drawings; 120 floor plans, 10 models | Raster | Best mean exact-count accuracy 0.51 (Gemini 3 Pro); OCR-style questions up to 0.95\[55\] | Not reliable for counting/geometry; strong on text | High (reusable protocol) |
| [ArchPlanVQA](https://ascelibrary.com/doi/abs/10.1061/JCCEE5.CPENG-7571) (2026) | Semantic VQA on CAD-converted floor plans | Raster | General VLMs reach 33.03–37.88% semantic understanding\[56\] | Domain gap is large | Medium |
| [FloorplanQA](https://arxiv.org/abs/2507.07644) (ICML 2026) | Spatial reasoning over structured JSON layouts | JSON | LLMs fail on physical constraints (e.g. sum vs. union of areas)\[57\] | Even with vectors in hand, LLM spatial reasoning needs checks | Low |
| [MMArch](https://arxiv.org/abs/2608.09281) (2026) | Short-answer reasoning over figures from architecture and civil-engineering papers; 1,212 items in 10 subdomains | Raster (figures) | Best model 51.7% (GPT-5.5), best open model 29.9%, human experts 94.6%; most errors in combining evidence and applying principles\[131\] | VLMs remain far from expert reading of AEC figures | Low (not plan parsing) |
| [FloorPlan-VLN](https://arxiv.org/abs/2603.17437) (2026) | Robot navigation guided by floor plans with typed room regions; over 10,000 episodes in 72 Matterport3D scenes | Raster plan + camera video | A fine-tuned Qwen2.5-VL with the plan beside each frame reaches success rate 28.8% in unseen scenes, against 17.0% for a fine-tuned navigation baseline; masking the plan cuts the base variant from 20.9% to 12.3%\[133\] | Room polygons with usage have downstream value beyond area management | Low (a consumer of parsed plans) |

| Model | Task | Inputs | Code / weights | Training data and licence | Reported results | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [Sketch2BIM](https://arxiv.org/abs/2510.20838) | Multi-agent VLM pipeline with human in the loop: sketch → JSON → Revit\[58\] | Raster (hand sketches) | No public code found | Cloud GPT-5 | Converges in 3–4 iterations | Sketches, not archive plans; 10 sketches | Low (pattern for human-in-the-loop review) |
| [Ayanzadeh & Oates](https://arxiv.org/abs/2604.23970) (2026) | Agentic room graph with a self-critic\[64\] | Raster | No public code found | Not stated | 92% short-route success (one building) | Centroid door rule; graph not evaluated | Low (association heuristics) |
| [Talebi-Kalaleh & Mei](https://arxiv.org/abs/2608.17237) (2026) | Deterministic PDF-operator parsing, scale from dimension strings, drafting rules for columns and beams, then a VLM limited to typed, guarded edits\[688\] | Vector PDF (raster with a supplied scale) | No code; benchmark promised | 100 generated structural framing plans; cloud VLM | Scale error ≤ 0.086%; columns recall 0.922, precision 0.997 | One generator; structural plans; no raster evaluation | Medium (port the rules to vector PDFs; skip the cloud VLM) |
| [Guo et al.](https://doi.org/10.3390/app16136290) (2026) | YOLOv8 symbols, ResNet-34 U-Net walls and rooms, areas at a manual scale, local LLM with retrieval for design rules\[693\]\[694\] | Raster | No code | 101 residential plans; not released | mAP50 92.3 (doors 78.8, windows 72.1); one plan's area within 0.81% | Augmented copies of the same plans in training and validation (preprint) | Low |

### 6.4 Text Recognition

OCR engines (docTR, kraken, Tesseract, PaddleOCR-VL and others) are compared in [open solutions §5.5](solutions-open.md#55-ocr-engines); this table adds the papers.

| Model | Task | Inputs | Code / weights | Training data and licence | Reported results | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [PP-OCRv6](https://arxiv.org/abs/2606.13108) (2026) | Detection + CTC recognition in three sizes (1.5M–34.5M parameters)\[692\] | Raster | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), Apache-2.0\[99\] | In-house, synthetic and industrial data; French, German, Italian in all sizes | Medium: +4.6 detection Hmean, +5.1 recognition accuracy over PP-OCRv5 server (in-house) | In-house benchmarks only; not tested on plans; "²" not mentioned | High (planned upgrade) |
| [PaddleOCR 3.0](https://arxiv.org/abs/2507.05595) (PP-OCRv5) | OCR and document parsing toolkit; Latin model for DE/FR/IT\[59\] | Raster | Apache-2.0; ONNX via RapidOCR in pilot v2\[60\]\[154\] | – | 84.7% on a 3,111-image Latin-script set\[60\]\[61\] | Superseded by PP-OCRv6 | Medium (tested baseline) |
| [Word Beam Search](https://doi.org/10.1109/ICFHR-2018.2018.00052) (ICFHR 2018) | CTC decoding constrained to dictionary words, other characters free\[686\] | CTC output | [githubharald/CTCWordBeamSearch](https://github.com/githubharald/CTCWordBeamSearch), MIT | No training | IAM word error 29.1% → 23.7% (realistic dictionary) | Needs the recogniser's per-frame output; forecast modes only for small dictionaries | High (stamp decoding) |
| [eDOCr2](https://www.mdpi.com/2075-1702/13/3/254)† (2025) | Dimension and tolerance text on mechanical drawings; synthetic training of custom recognisers\[691\] | Raster drawings | [javvi51/edocr2](https://github.com/javvi51/edocr2), MIT\[212\] | Synthetic | Not read | Mechanical drawings; optional GPT-4o step | Medium (recipe for Swiss dimension notation) |
| [Schlagenhauf et al.](https://arxiv.org/abs/2205.02659) (2022) | Text detection (Faster R-CNN) and recognition (Keras-OCR) trained on generated technical drawings\[134\] | Raster drawings | [2Obe/Text-Detection-on-Technical-Drawings](https://github.com/2Obe/Text-Detection-on-Technical-Drawings), MIT\[213\] | Generated drawings; 9 real drawings for testing | Detection 81.9%, recognition 79.3% (Keras-OCR out of the box 77.2%, 46.8%) | Mechanical drawings; horizontal and vertical text only; tiny real test | Medium (generator pattern for room stamps) |
| [Khan et al.](https://arxiv.org/abs/2505.01530) (2025) | Oriented-box detection of nine annotation types, then OCR-free parsing of each crop to JSON (Donut)\[135\] | Raster drawings | No code found | 1,367 public mechanical drawings, labels not released | F1 97.3% for GD&T; one shared parser beats class-specific ones | Mechanical drawings; hallucination rate is 100 minus precision | Low (crop-then-parse pattern) |
| [Ying et al.](https://doi.org/10.3389/fbuil.2026.1839808) (2026) | Slimmed DBNet detector + transformer recogniser for rebar annotations\[136\] | Raster drawings | Release promised in the paper, but its data statement says "available on request" | 1,005 Chinese construction drawings, 7,108 annotations | Detection F1 95.9; recognition 96.3% on ground-truth crops | Chinese structural drawings; no end-to-end accuracy | Low (small detector-recogniser recipe) |

### 6.5 Training and Adaptation Code

| Model | Task | Inputs | Code / weights | Training data and licence | Reported results | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [MIC](https://arxiv.org/abs/2212.01322) (CVPR 2023) | Masked image consistency for unsupervised domain adaptation\[679\] | Raster (any segmenter) | [lhoyer/MIC](https://github.com/lhoyer/MIC); some files non-commercial; street-scene weights only | Synthetic → real street scenes | +1.2 to +4.7 mIoU over each base method (GTA→Cityscapes) | Untested on drawings; out-of-context objects | High (reimplement the loss in the pilot's training loop) |
| [DAFormer / HRDA](https://arxiv.org/abs/2304.13615) (TPAMI) | Multi-resolution crops, rare-class sampling, feature distance\[680\] | Raster | [lhoyer/HRDA](https://github.com/lhoyer/HRDA); same component caveat | As above | 73.8 mIoU GTA→Cityscapes; thin classes +7.5 to +9.9 IoU | Transformer encoder adapts best; about 32 h on one GPU | Medium |
| [Semap](https://arxiv.org/abs/2603.05037) (2026) | Procedural map synthesis; Mask2Former (Swin-L) with two-scale inference\[681\] | Raster maps | Data and model on Zenodo; synthesis code not found | 1,439 annotated + 12,122 synthetic patches; licence not named | +5.1 mIoU from synthesis; +4.2 from two scales | Maps; random test split | Medium (renderer recipe) |
| [TopoMortar](https://arxiv.org/abs/2503.03365) (BMVC 2025) | Loss benchmark: CE+Dice, clDice, Skeleton Recall, cbDice and others\[682\] | Raster photos | [jmlipman/TopoMortar](https://jmlipman.github.io/TopoMortar), MIT (per the research index) | 420 brick-wall images | clDice lowest Betti errors; Skeleton Recall best with noisy labels | Simple task; not plans | Medium (wall-loss A/B) |
| [Chen et al. 2024](https://doi.org/10.1371/journal.pone.0298217) | Edge maps + watershed closed-shape extraction; PQ evaluation\[689\] | Raster maps | Repository without a licence; data on Zenodo | Two Paris atlas sheets, 8,362 polygons | PQ 51.1 | One collection | Low (reimplement watershed and PQ) |

## 7. Datasets

Vector datasets can be rendered in randomised graphical styles (black-and-white or colour, solid or hatched wall fills, line weights) to train style-robust raster models. For BBL this matters more than building type, so renderable vector data ranks above single-style raster data. The full catalogue, with licence verdicts and access status, is in [open datasets](datasets-open.md#4-datasets-by-category).

| Dataset | Annotations | Format | Scale and building type | Licence | Limitations | Relevance |
|---|---|---|---|---|---|---|
| [FloorPlanCAD](https://floorplancad.github.io/) (mirror: [Voxel51/FloorPlanCAD](https://huggingface.co/datasets/Voxel51/FloorPlanCAD)) | Line-level panoptic labels; 30 thing + 5 stuff classes, no column class\[23\] | Vector SVG | 15,663 drawings (11,602 at first release); residential, schools, hospitals, malls, offices\[6\]\[19\]\[65\] | Annotations CC BY-NC 4.0; drawings not owned by the authors\[6\]. Mirror metadata stating CC BY-SA does not change this – NC | Chinese drafting; no room attributes | Medium (benchmark; non-residential) |
| [ArchCAD-400K](https://huggingface.co/datasets/jackluoluo/ArchCAD) ([paper](https://arxiv.org/abs/2503.22346)) | Line-grained labels, 27 categories incl. columns, beams, axis lines; furniture merged into one class\[29\] | Vector chunks | 413,062 chunks from 5,538 drawings\[29\]; mostly public/commercial, residential 14%; 40K samples public | CC BY-NC 4.0; gated with manual approval – NC\[28\]\[29\] | Chinese drafting; full set not public | Medium (benchmark; only dataset with columns) |
| [LS-CAD](https://arxiv.org/abs/2412.07377) (CADSpotting) | Large office/campus CAD\[30\] | Vector CAD | 45 plans ≥ 1,000 m²; non-residential | Announced; not released\[30\] | Tiny; availability unknown | Low (Medium if released) |
| [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) | Raster + SVG polygons, 80+ categories\[38\]; three style subsets (colorful, high_quality, high_quality_architectural) | Raster + SVG | 5,000; Finnish residential; every room labelled with its size in metres, so the scale is known (1 cm/px in the scaled images) | CC BY-NC 4.0 on GitHub\[5\], CC BY-NC-SA 4.0 on [Zenodo](https://doi.org/10.5281/zenodo.2613548); the stricter terms apply; derivatives relabelled MIT (e.g. Roboflow's cubicasa5k-2) do not change this\[48\]\[49\] – NC | Residential; multi-floor sheets | Medium (benchmark; [pilot v2](../pilot/v2-pipeline/README.md#public-benchmarks) zero-shot test) |
| [CVC-FP](https://dag.cvc.uab.es/dataset/cvc-fp-database-for-structural-floor-plan-analysis/) | Rooms, walls, doors, windows, room separations | Raster (scans) + SVG | 122 scanned plans in 4 subsets by origin and style; residential | Non-commercial, despite a CC BY name on its page – NC\[108\] | Small; door labels cover the swing area | Medium (real scans in several styles; pilot v2 samples) |
| [WAFFLE](https://arxiv.org/abs/2412.00955) (2024) | Building type, metadata, OCR and grounded legends; dense labels on 110 images | Raster | 18,556 plans (the abstract says nearly 20,000) across building types, regions and eras, incl. historical and public buildings\[100\] | Per-image Wikimedia Commons licences | Labels mostly automatic | High (test for unseen styles, historical and public buildings) |
| [Versailles-FP](https://www.etis-lab.fr/versailles-fp/en/) ([paper](https://arxiv.org/abs/2103.08064)) | Wall masks (filled and hollow walls) | Scans | 500 plans of the Château de Versailles, 1670–1790, up to 17,000 × 8,000 px\[111\]\[113\] | Academic use under agreement – R\[112\] | Walls only; one archive | Medium (historical wall test, via an academic partner) |
| [MLSTRUCT-FP](https://github.com/MLSTRUCT/MLStructFP) | Wall and slab polygons | Raster + JSON | 954 multi-unit plans; residential (Chile) | MIT loader code; dataset terms not stated; repository archived April 2026\[109\] | Residential; walls only | Medium (permissive wall labels once terms are clear) |
| [R2V / LIFULL HOME'S](https://github.com/art-programmer/FloorplanTransformation) | Raster + vector labels | Raster + vector | ~870 annotated; Japanese residential | Universities and public research institutions only – R\[107\] | Small; residential | Low |
| [R3D](https://github.com/zlzeng/DeepFloorplan) (DeepFloorplan) | Non-rectangular rooms | Raster | ~200+; residential | Research | Small; residential | Low |
| [Raster-to-Graph dataset](https://github.com/SizheHu/Raster-to-Graph) | Structural graph | Raster + graph | >10,000; Japanese residential | Request form; images under LIFULL terms – R | Residential | Low |
| RFP (Lv et al.) | Walls, openings, room types; subsets for text, symbols and scale | Raster | 7,000 Chinese residential images from the web | No release statement | Not available | Low |
| [RPLAN](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html) | Room layouts | Raster | ~80,000; Chinese residential | Research-only on request | Layouts, not drawings | Low |
| [ResPlan](https://arxiv.org/abs/2508.14006) | Walls, doors, windows, 17 room classes, typed connectivity graphs | Vector (pixel + metric coordinates) | 17,000; South Asian residential | CC BY 4.0 on GitHub, CC BY-NC-SA 4.0 on Kaggle – conflicting\[103\]\[104\] | Scraped listings; residential | Medium (renderable, metric; training only after licence confirmation) |
| [ResPlan-FP](https://github.com/Cyprinus12138/fpvec-lab) (Zhang 2026)\[132\] | Wall centre lines, openings and rooms; clean line renders without text or furniture | Raster (256 px) + vector | 16,998 plans rendered from ResPlan; residential; fixed 14,998/1,000/1,000 split | Released as CC BY 4.0, but inherits ResPlan's licence conflict | Low resolution; renders, not real drawings; no near-duplicate removal | Medium (redistributable vectorisation test set once ResPlan's licence is confirmed) |
| [Swiss Dwellings](https://doi.org/10.5281/zenodo.7788422) ([Archilyse page](https://archilyse.standfest.science/swiss-dwellings)) | Vector geometry, simulations\[7\]; "sourced from commercial clients of Archilyse AG" | Vector | "Over 45,000 apartments (370,000 rooms) in ~3,100 buildings" (v3.0.0); Swiss residential, some mixed use | CC BY 4.0\[7\]\[101\] | Residential; no plan images or room annotations | High (licence-clean, Swiss; pilot v2 training data) |
| [Modified Swiss Dwellings (MSD)](https://data.4tu.nl/datasets/e1d89cb5-6872-48fc-be63-aadd687ee6f9) | ML-ready graphs/images; door-level access graphs | Graphs + images | 5,372 floor plans, 18.9k apartments\[69\]\[70\]; residential only (filtering removed 2,305 plans with e.g. "waiting room") | 4TU: CC BY 4.0; Kaggle mirror: CC BY-SA 4.0 – use 4TU\[71\]\[72\] | Residential only; derived | Medium (connectivity reference; otherwise use Swiss Dwellings) |
| [Neufert 4.0](https://zenodo.org/records/14223942) | Audited Swiss Dwellings derivative (20,419 plans reviewed)\[73\] | Vector | 20k+; residential | CC BY 4.0 | Residential | Medium (cleaner Swiss Dwellings subset) |
| [IFC-Bench](https://huggingface.co/datasets/sylvainHellin/ifc-bench) | IFC models with walls, doors, windows, columns, stairs, named spaces | IFC | 51 models of 22 projects: offices, clinic, hospital, hotels, residential | Mostly CC BY 4.0 or MIT, four GPLv3; per model\[118\] | Few models; check each licence | High (non-residential render source; 17 models prepared in pilot v2) |
| [Structured3D](https://structured3d-dataset.org/) | Synthetic 3D + floorplans | Synthetic | 3,500 scenes; residential | Non-commercial | Point-cloud domain | Low |
| [ROBIN](https://github.com/gesstalt/ROBIN) | Symbol/room labels | Raster | Small (≈100s); residential | Academic; not verified | Small | Low |
| Semap\[681\] | Six classes on historical maps (built, road, water, boundary …) | Raster | 1,439 patches from more than 50 institutions + 12,122 synthetic | Zenodo; no licence named in the paper; images keep their own rights | Maps, not plans | Low (synthesis reference) |
| Paris atlas (Chen et al. 2024)\[689\] | Closed-shape polygons | Raster | 2 sheets, 8,362 polygons | Zenodo | Maps; one collection | Low (PQ protocol) |
| [ICDAR 2021 MapSeg](https://doi.org/10.5281/zenodo.4817662)\[703\] | Map content-area masks, building blocks, graticule intersections | Raster | Paris atlas sheets (1894–1937), up to about 10,000 px | CC BY 4.0 | Maps, not plans | Medium (public benchmark for drawing-region masks and their boundary metric) |
| TopoMortar\[682\] | Mortar masks with accurate, pseudo and noisy labels; 6 out-of-distribution test groups | Raster (photos) | 420 brick-wall images | Online with the code; licence not stated in the paper | Not drawings | Low (loss benchmark) |
| Trivi's archival drawings\[137\] | 11 material hatch classes | Scans | 19 drawings of Roman monuments (3 plans) | No release statement | Not available | Low |

No permissive, large, labelled dataset of administrative, office or public building plans with room stamps exists. Licence-clean training data comes from BBL's own compliant DWGs, Swiss Dwellings and permissively licensed IFC models, and from MLSTRUCT-FP once its dataset terms are verified. Non-commercial sets (FloorPlanCAD, CubiCasa5K, ArchCAD-400K) serve research benchmarks only, unless a legal review clears them.

Rendering compliant DWGs to raster and projecting their vector labels is the most efficient way to create in-domain training data. ArchCAD-400K used the same approach (§3.2). It kept only 5,538 of 11,917 candidate drawings after a layer-quality screening that discarded drawings "with over 5% deviation".\[29\] WAFFLE, CVC-FP and Versailles-FP are the best public tests for unseen styles.

Text in drawings has no usable public ground truth either:
- Khan et al.'s 1,367 annotated mechanical drawings are not released.
- Ying et al.'s rebar crops are promised in the paper but "available on request" in its data statement.
- Schlagenhauf et al.'s generator of artificial drawings is public.

Room stamps have to be synthesised or labelled by BBL ([open datasets §4.6](datasets-open.md#46-text-in-drawings)).

## 8. Paper Summaries

Grouped by the taxonomy of §3.1. Relevance as defined in §1. † = not read in full (summarised from the abstract and the index note).

### Survey

| Paper | Strategy | Contribution | Key result | Main limitation | Relevance |
|---|---|---|---|---|---|
| [Rezvanifar et al. 2019](https://doi.org/10.1186/s41074-019-0055-1)\[696\] | Floor plan analysis | Review of symbol spotting in architectural drawings by description (pixel-based vs vector-based) and matching strategy; adds template matching with a clutter-tolerant cross-correlation | Most methods have good recall but poor precision; the proposed matcher reaches F 0.987 on the synthetic GREC set (contest best 0.760) | Quantitative results on synthetic data only; predates deep learning for spotting | Medium (background for openings and symbols) |
| [Pizarro et al. 2022](https://doi.org/10.1016/j.autcon.2022.104348)†\[695\] | Floor plan analysis | Review of rule-based and learning-based analysis of raster floor plans, 1995–2021: walls, doors, rooms, vectorisation, modelling, with datasets, scopes and tasks | Most cited survey of the field | Not open access; raster images only; ends in 2021 | Medium (background) |
| [Khade et al. 2025](https://doi.org/10.1007/s10032-025-00528-8)†\[697\] | Floor plan analysis | Sorts floor plan models into nine families (dimensional, room-based, structural, retrieval, object detection, graph, map, residential, area-based) and compares them by dataset, year and reported metrics | Most recent journal survey | Not open access; known only from its abstract | Low |

### Classical and Rule-Based

| Paper | Strategy | Contribution | Key result | Main limitation | Relevance |
|---|---|---|---|---|---|
| [Ahmed et al. 2011](https://av.dfki.de/publications/improved-automatic-analysis-of-architectural-floor-plans/) | Heuristic pipeline | Text/graphics separation, line-thickness walls, gap closing, rooms as components | Room detection 89%, recognition 79% (80 scans) | One binarised style; thresholds not reported | Low |
| [Ahmed et al. 2012](https://av.dfki.de/?p=9839)\[62\] | Heuristic pipeline | OCR room labels by containment + Levenshtein dictionary | 82.3% labels correct (self-judged) | Strips digits; axis-aligned room splitting | Medium |

Not reviewed (no open copy): Macé et al. 2010 (rooms by recursive decomposition) and de las Heras et al. 2013 (patch-based wall segmentation, structural room recognition).

### Raster Parsing

| Paper | Strategy | Contribution | Key result | Main limitation | Relevance |
|---|---|---|---|---|---|
| [Dodge et al. 2017](https://www.mva-org.jp/Proceedings/2017USB/papers/10-01.pdf) | Semantic segmentation | FCN walls, door detection, scale from OCR'd room areas | Walls mean IoU 89.7–94.4; cross-style 76.1–81.7 | Pixel output; tiny test sets; cloud OCR | Medium |
| [Zeng et al. 2019 (DeepFloorplan)](https://arxiv.org/abs/1908.11025) | Semantic segmentation | Room-boundary-guided attention; rooms as bounded regions | Mean IoU 0.66–0.76 | Pixel output; small tests; best-of selection | Medium |
| [Huang et al. 2023 (MuraNet)](https://arxiv.org/abs/2309.00348) | Semantic segmentation | Joint wall segmentation + door/window detection | Wall IoU 78.4; door/window AP50 91.7 | One dataset; no code | Low |
| [Guo et al. 2026](https://doi.org/10.3390/app16136290)\[693\]\[694\] | Semantic segmentation | YOLOv8 symbols, ResNet-34 U-Net walls and rooms, areas at a manual scale, local LLM with retrieval for design guidance | mAP50 92.3 (doors 78.8, windows 72.1); one apartment's area within 0.81% | Preprint: augmented copies of the same plans in training and validation; 101 residential plans; no OCR or vectors. The journal version adds 10 held-out plans (per the index) | Low |
| [Trivi 2026](https://isprs-archives.copernicus.org/articles/XLVIII-2-W12-2026/487/2026/)\[137\] | Semantic segmentation | U-Net/ResNet-34 segmentation of 11 material hatch codes on archival drawings, from 19 annotated drawings | mIoU 0.71–0.73 (three tabulated drawings) | No held-out split; three plans; monuments | Medium (closest domain; same architecture as pilot v2) |
| [Liu et al. 2017 (Raster-to-Vector)](https://github.com/art-programmer/FloorplanTransformation) | Segmentation and vectorisation | Junction heatmaps + integer programming with topology constraints | Junction precision 70.7 → 94.7 with IP | Manhattan; 256 px; commercial solver | High |
| [Lv et al. 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Lv_Residential_Floor_Plan_Recognition_and_Reconstruction_CVPR_2021_paper.html)\[687\] | Segmentation and vectorisation | Plan-region detection, segmentation, scale from dimension strings (endpoint heatmaps, digit detection, bipartite matching, k-means of ratios), vectorisation by optimisation allowing inclined walls | mIoU 0.85; wall-junction accuracy 0.96; cropping to the plan adds 4–6 mIoU to two baselines | Scale accuracy only in the supplement; no OCR; curved walls fail; data not released | Medium |
| [Chang et al. 2025](https://doi.org/10.3390/buildings15071178)†\[706\] | Segmentation and vectorisation | Keypoint detection of walls, doors and windows with rule-based filtering; scale from dimension strips with outlier rejection; 3D viewer | Over 87% precision, 88% recall on more than 5,000 house plans; "scale calculation accuracy exceeding 95%" (metric undefined) | Residential; no code | Low |
| [Egiazarian et al. 2020](https://arxiv.org/abs/2003.05471) | Primitive vectorisation | Scan cleaning + primitive estimation + optimisation | IoU 79–86% on floor plans | No semantics; minutes per image | Medium |
| [Pang et al. 2024](https://arxiv.org/abs/2404.10985) | Symbol and keypoint detection | Sheet layout detection (nine region classes, main drawing included) + precise keypoints | Region F1 0.95–1.00 (main drawing 1.00); keypoint F1 0.84 | Telecom equipment-room drawings; 60 test images | Medium |
| [Lombardi et al. 2025](https://arxiv.org/abs/2504.08645)\[700\] | Sheet layout and title blocks | Title block, main content, legend and notes on building drawings; title blocks read from crops | Title-block accuracy 0.971, main content 0.943 (1,385 drawings); whole-sheet GPT-4o missed 6 of 10 hard title blocks | Cloud reader; no code or data | High (detect, crop, read) |
| [Peng et al. 2025](https://doi.org/10.17868/strath.00093312)\[702\] | Sheet layout and title blocks | Views detected on bridge drawings, named by their matched captions | View detection mAP50 about 0.92 (2,000 drawings) | Bridge drawings; no code | High (several drawings per sheet) |
| [Huang et al. 2026 (FELD)](https://arxiv.org/abs/2607.18997)\[699\] | Sheet layout and title blocks | Benchmark of detectors and VLMs for nine layout classes on façade sheets | RF-DETR mAP50 0.949 (title block mAP50:95 0.989); document-layout pretraining hurt | 551 sheets downscaled to 1,344 × 800; no data or code | High (closest analogue of stage 1b) |
| [Peng et al. 2026 (title blocks)](https://doi.org/10.35490/EC3.2026.252)\[701\] | Sheet layout and title blocks | Fine-tuned open VLM grounds and reads German title blocks | Localisation F1 93.8% (YOLOv11-m 93.1%); "Maßstab" field word accuracy 0.990 vs 0.572 zero-shot | 64 test drawings; base model "qwen-research"; data confidential | Medium |
| [Schwenker et al. 2023 (EXSCLAIM!)](https://doi.org/10.1016/j.patter.2023.100843)†\[704\] | Scale and dimension reading | Scale bars and labels in microscopy figures: detection, label reading, pairing | Bar length error 5.4%; 82% of labels read (440 figures) | Microscopy, not drawings; GPL-3.0 code | Medium (scale-bar reader to reimplement) |
| [Faltin et al. 2024](https://doi.org/10.1061/9780784485248.087)†\[705\] | Scale and dimension reading | Dimension lines (YOLOv7) read by EasyOCR, pixel resolution by voting | "Promising results" only | No figures; closed access | Low |
| [Lin & Wang 2026](https://doi.org/10.3390/buildings16051043)†\[707\] | Scale and dimension reading | Scanned historical drawings to CAD: dimensions, endpoints and grid markers detected, scale by RANSAC, rules snapped to a 50 mm module | Detection mAP50 0.98–0.99; 89.3% of dimension texts read | One worked example; AGPL detector; no code | Medium |
| [Liang et al. 2026](https://doi.org/10.35490/EC3.2026.212)\[708\] | Scale and dimension reading | Legacy plans to IFC with a metric grid rebuilt from dimension annotations: median scale, chain checks, separate x and y | OCR exact 90.2%; 62 of 200 plans fully correct | No code | High (robust dimension-string scale) |

### Structured Prediction

| Paper | Strategy | Contribution | Key result | Main limitation | Relevance |
|---|---|---|---|---|---|
| [Paudel et al. 2021](https://arxiv.org/abs/2108.05947) | Graph-based | Room type from room graph (GNN) | 81% accuracy | Residential vector input; adjacency ≠ access | Low |
| [Chen et al. 2023](https://arxiv.org/abs/2303.03851) | Graph-based | GNN over candidate segments; non-Manhattan; style augmentation | Wall/door/window sAP 84.9/97.5/88.4 | Residential; no rooms or thickness | High |
| [Hu et al. 2024 (Raster-to-Graph)](https://github.com/SizheHu/Raster-to-Graph) | Graph-based | Autoregressive wall-graph prediction; rooms as shortest cycles | Room F1 84.7; 67% structurally perfect | Manhattan; no openings; ≤ 50 nodes | Medium |
| [Yue et al. 2023 (RoomFormer)](https://arxiv.org/abs/2211.15658) | Polygon decoder | Room polygons as two-level queries | Room F1 97.3; transfer IoU 74.0 | Density maps; 20-room cap | Medium |
| [Liu et al. 2024 (PolyRoom)](https://arxiv.org/abs/2407.10439) | Polygon decoder | Segmentation-initialised polygons; angle-based vertex selection | Room F1 98.3; transfer IoU 85.2 | Density maps; segmenter-limited | Medium |
| [Xu et al. 2024 (FRI-Net)](https://arxiv.org/abs/2407.10687) | Polygon decoder | Rooms as unions of half-plane cells | Room F1 99.1 | Needs wall heights; point clouds | Low |
| [Phung & Averbuch-Elor 2026 (Raster2Seq)](https://arxiv.org/abs/2602.09016)\[683\] | Polygon decoder | Rooms, doors and windows as one labelled polygon sequence; learned anchors | Room F1 88.7 on CubiCasa5K (RoomFormer 83.5); WAFFLE zero-shot IoU 73.9 (RoomFormer 60.5) | 256 px; openings misplaced; no geometric constraints | Medium |

### Vector CAD Parsing

| Paper | Strategy | Contribution | Key result | Main limitation | Relevance |
|---|---|---|---|---|---|
| [Fan et al. 2022 (CADTransformer)](https://openaccess.thecvf.com/content/CVPR2022/html/Fan_CADTransformer_Panoptic_Symbol_Spotting_Transformer_for_CAD_Drawings_CVPR_2022_paper.html)\[20\] | Panoptic symbol spotting | Primitive tokens with image features from a rendering; local-attention transformer | PQ 68.9 (67.3 without layer-based augmentation) | Needs a rendering; superseded | Low |
| [Zheng et al. 2022 (GAT-CADNet)](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_GAT-CADNet_Graph_Attention_Network_for_Panoptic_Symbol_Spotting_in_CAD_CVPR_2022_paper.html) | Panoptic symbol spotting | Primitives as graph nodes; instances as subgraphs of a predicted adjacency matrix | PQ 73.7, geometry only | GPU memory limits drawing size; no official code | Low |
| [Liu et al. 2024 (SymPoint)](https://arxiv.org/abs/2401.10556) | Panoptic symbol spotting | Primitives as points | PQ 83.3 | Superseded; 1,000 epochs | Low |
| [Liu et al. 2024 (SymPoint-V2)](https://arxiv.org/abs/2407.01928) | Panoptic symbol spotting | Layer-feature enhancement | PQ 90.1 with layers, 83.2 without | Gains depend on layers; no licence | Low |
| [Yang et al. 2024 (CADSpotting)](https://arxiv.org/abs/2412.07377) | Panoptic symbol spotting | Coordinate-only sampling; sliding windows for large sheets; 3D walls | PQ 87.4; LS-CAD 75.5 | Vector only; tiny LS-CAD test | Medium |
| [Wei et al. 2025 (VecFormer)](https://arxiv.org/abs/2505.23395) | Panoptic symbol spotting | Line-based primitives; branch fusion refinement | PQ 88.4 without layers | One dataset; vector only | Medium |
| [Gong et al. 2026 (TextCAD)](https://arxiv.org/abs/2607.12678) | Panoptic symbol spotting | Text annotations fused with geometry | PQ 91.09 without layers (own split) | Implicit text association; no code | Medium |
| [Chen et al. 2026 (PolarSym)](https://arxiv.org/abs/2608.11793) | Panoptic symbol spotting | Polar geometry-aware attention | +1.73 PQ over reproduced SymPoint-V2 | Uses layers; inconsistent tables | Low |
| [Yang et al. 2023 (VectorFloorSeg)](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_VectorFloorSeg_Two-Stream_Graph_Attention_Network_for_Vectorized_Roughcast_Floorplan_Segmentation_CVPR_2023_paper.html)\[684\] | Primitive graph segmentation | Wall lines extended into an arrangement; lines classified as boundaries, regions as room types | CubiCasa5K test mIoU 62.5, room integrity 67.5 (image-based OCRNet 57.1, 41.9) | Walls from annotations, not real CAD; no licence | Medium |
| [Carrara et al. 2024 (VectorGraphNET)](https://arxiv.org/abs/2410.01336)\[685\] | Primitive graph segmentation | PDF → SVG paths as graph nodes with geometric and style features; hierarchical labels; 1.3M weights | FloorPlanCAD weighted F1 89.0, plain F1 79.4 (SymPoint 86.8); 76 university plans: accuracy 0.97, macro F1 0.82 | No code; proprietary data; weighting differs between compared methods | Medium |
| [Yin et al. 2020](https://doi.org/10.1016/j.autcon.2020.103082)†\[709\] | Layer and entity classification | CAD layers classified by content, not name; elevations, floor levels and openings recognised for façade BIM | Nearly all floor levels found; 88% of visible members measured perfectly (94 drawings) | Elevations, not plans; closed access | Medium (layer names are a weak feature) |

### Vision-Language Models and Agents

| Paper | Strategy | Contribution | Key result | Main limitation | Relevance |
|---|---|---|---|---|---|
| [Liu et al. 2026 (FloorplanVLM)](https://arxiv.org/abs/2602.06507) | Fine-tuned VLM | VLM outputs a wall graph with nested openings and room cycles | Outer IoU 0.925; opening F1 0.733 | Own benchmark; residential; model not released | High |
| [Ratul et al. 2025 (Sketch2BIM)](https://arxiv.org/abs/2510.20838) | Agentic pipeline | Multi-agent, human-in-the-loop sketch → BIM | Converges in 3–4 iterations | 10 sketches; cloud GPT-5 | Medium |
| [Ayanzadeh & Oates 2026](https://arxiv.org/abs/2604.23970) | Agentic pipeline | Agentic room graph with self-critic | 92% short-route success (one building) | Centroid door rule; graph not evaluated | Medium |
| [Talebi-Kalaleh & Mei 2026](https://arxiv.org/abs/2608.17237)\[688\] | Agentic pipeline | Deterministic PDF-operator parsing, dimension-string scale and drafting rules, then a VLM restricted to typed, guarded edits | Scale error ≤ 0.086%; columns recall 0.922, precision 0.997 (50 held-out generated plans) | Structural plans from one generator; cloud VLM; no code | High (vector-PDF and scale rules) |
| [Xu et al. 2026 (BlueprintAgent)](https://arxiv.org/abs/2609.07362)\[710\] | Agentic pipeline | Multimodal-LLM agent for scanned structural sheets; vision probe for grid labels; validators trigger targeted re-reads | Axis F1 1.000 vs 0.987 for one zero-shot model (300 sheets, 20 projects) | Structural frames, not rooms; no data release found | Medium (grid axes and storey linkage) |

### Text Recognition

| Paper | Strategy | Contribution | Key result | Main limitation | Relevance |
|---|---|---|---|---|---|
| [Cui et al. 2025 (PaddleOCR 3.0)](https://arxiv.org/abs/2507.05595) | OCR system | Self-hostable OCR and document parsing | Best on OmniDocBench (self-reported) | DE/FR/IT need the Latin / PP-OCRv6 models | High |
| [Zhang et al. 2026 (PP-OCRv6)](https://arxiv.org/abs/2606.13108)\[692\] | OCR system | Shared lightweight backbone for detection and CTC recognition, three sizes | +4.6 detection Hmean, +5.1 recognition over PP-OCRv5 server; French, German, Italian in all sizes | In-house benchmarks only | High |
| [Scheidl et al. 2018 (Word Beam Search)](https://doi.org/10.1109/ICFHR-2018.2018.00052)\[686\] | Decoding and lexicons | Dictionary prefix tree inside CTC beam search; non-word characters free; optional word bigrams | IAM word error 29.1% → 23.7% (realistic dictionary), 10.2% (ideal) | Handwriting lines, not drawings | High |
| [Schlagenhauf et al. 2022](https://arxiv.org/abs/2205.02659)\[134\] | Text in technical drawings | Detector and recogniser trained on a generator of artificial technical drawings | Detection 81.9%, recognition 79.3% on 9 real drawings (Keras-OCR 77.2%, 46.8%) | Mechanical drawings; tiny real test set | Medium |
| [Schönfelder et al. 2024](https://doi.org/10.1016/j.autcon.2023.105156)†\[698\] | Text in technical drawings | Text detector plus recogniser for architectural floor plans, chosen by model comparison (YOLOv7, PARSeq per citing papers); new floor plan dataset with text boxes; domain-specific synthetic data improves detection | Not read (closed access) | Unknown whether stacked multi-field stamps are handled | High (closest to reading room stamps) |
| [Villena Toro & Tarkian 2025 (eDOCr2)](https://www.mdpi.com/2075-1702/13/3/254)†\[691\] | Text in technical drawings | Dimension-text grouping and synthetic training of custom recognisers for mechanical drawings | Not read | Mechanical drawings; optional GPT-4o step | Medium |
| [Khan et al. 2025](https://arxiv.org/abs/2505.01530)\[135\] | Text in technical drawings | Annotation regions found by an oriented-box detector, then parsed to JSON by one fine-tuned Donut model | F1 97.3% for GD&T; shared model better than nine class-specific ones | Mechanical drawings; data not released; detector not evaluated | Low |
| [Ying et al. 2026](https://doi.org/10.3389/fbuil.2026.1839808)\[136\] | Text in technical drawings | Slimmed DBNet detector and transformer recogniser for rebar annotations | Detection F1 95.9; recognition 96.3% on ground-truth crops | Chinese structural drawings; contradictory release statements | Low |

### Learning Strategies

| Paper | Strategy | Contribution | Key result | Main limitation | Relevance |
|---|---|---|---|---|---|
| [Hoyer et al. 2023 (MIC)](https://arxiv.org/abs/2212.01322)\[679\] | Domain adaptation | EMA-teacher pseudo-labels + consistency on masked target images | +1.2 to +4.7 mIoU over each base method, CNNs included; 75.9 with HRDA (GTA→Cityscapes) | Street scenes; out-of-context objects; some non-commercial code files | High (route to unlabelled archive scans) |
| [Hoyer et al. 2023 (DAFormer, HRDA)](https://arxiv.org/abs/2304.13615)\[680\] | Domain adaptation | Transformer segmenter; rare-class sampling; feature distance; detail + context crops with scale attention | GTA→Cityscapes 73.8 mIoU (+16.3 over prior best); thin classes +7.5 to +9.9 IoU | Street scenes; CNN encoders adapt worse | Medium |
| [Petitpierre 2026 (Semap)](https://arxiv.org/abs/2603.05037)\[681\] | Synthetic data and domain generalisation | Historical-map benchmark; procedural synthesis with textures from real scans; two-scale inference | +5.1 mIoU from synthesis (74.2 vs 69.1); +4.2 from two scales | Maps; random test split, no held-out collection; thin lines weak | High (renderer recipe) |
| [Valverde et al. 2025 (TopoMortar)](https://arxiv.org/abs/2503.03365)\[682\] | Topology-aware learning | Brick-wall benchmark comparing eight losses under small data, noisy labels and shift | clDice lowest Betti errors; Skeleton Recall best with noisy labels; augmentation + self-distillation lifts CE+Dice past most topology losses | Simple task; photos, not drawings | Medium |
| [Petitpierre & Guhennec 2023](https://doi.org/10.1093/llc/fqad006)†\[690\] | Annotation strategy | Lessons from five cadastral vectorisation projects | 2–3 visually homogeneous classes; label what is visible; about a dozen sheets bootstrap a model | Not read in full | Medium (gold-set protocol) |

### Datasets and Benchmarks

| Paper | Strategy | Contribution | Key figures | Main limitation | Relevance |
|---|---|---|---|---|---|
| [Kalervo et al. 2019 (CubiCasa5K)](https://arxiv.org/abs/1904.01920) | Raster plan dataset | Multi-task junctions + segmentation; dataset in 3 styles | Room mean IoU 57.5 (49.3 polygonised) | Manhattan heuristics; residential; non-commercial | Medium |
| [Swaileh et al. 2021 (Versailles-FP)](https://arxiv.org/abs/2103.08064)\[113\] | Raster plan dataset | 500 historical palace plans; semi-automatic wall labels; U-Net detector | Wall IoU 88.1, Dice 93.3 (5-fold); labelling about 10× faster | Walls only; 512 px; academic agreement | Medium |
| [Ganon et al. 2024 (WAFFLE)](https://arxiv.org/abs/2412.00955) | Raster plan dataset | ~19K drawings in many styles, historical and public buildings | 110 densely labelled images | Weak labels; licences per image | Medium |
| [Fan et al. 2021 (FloorPlanCAD)](https://arxiv.org/abs/2105.07147) | Vector plan dataset | Defines panoptic symbol spotting and the benchmark | PQ 56.1 | No columns, rooms or text | Medium |
| [van Engelenburg et al. 2024 (MSD)](https://arxiv.org/abs/2407.10121) | Vector plan dataset | Swiss metric vector data with door-level graphs | 5,372 floors, 18.9K units | Residential subset; no stairs | High |
| [Abouagour et al. 2025 (ResPlan)](https://arxiv.org/abs/2508.14006) | Vector plan dataset | 17K metric vector plans with typed room graphs | Edge detection F1 0.971 | South Asian residential; provenance | Medium |
| [Luo et al. 2025 (ArchCAD-400K)](https://arxiv.org/abs/2503.22346) | Vector plan dataset | Layer-based annotation engine; non-residential data with columns | DPSS PQ 70.6 on ArchCAD | Non-commercial; vector only | High |
| [Chazalon et al. 2021 (ICDAR MapSeg)](https://arxiv.org/abs/2105.13265)\[703\] | Benchmark | Map competition: building blocks inside a given map-area mask; map content area per sheet; graticule intersections | Content-area task won with 19 px Hausdorff 95 (others 85–126 px) | Maps; Paris atlas only | Medium (drawing-region masks and metric) |
| [Chen et al. 2024 (historical maps)](https://doi.org/10.1371/journal.pone.0298217)\[689\] | Benchmark | Closed-shape extraction benchmark: edge detectors, losses, watershed, PQ | PQ 51.1 with U-Net edges + Meyer watershed; topology losses below binary cross-entropy | Maps; one collection; code without licence | Medium |
| [Zhang 2026 (readout comparison, ResPlan-FP)](https://arxiv.org/abs/2608.25608)\[132\] | Benchmark | Emitting vs detecting geometry with one network; edit-cost metric; corrected CubiCasa5K labels; ResPlan-FP test set | Detection +2.7 to +5.1 wall F1 on real scans; fusion +7.1, mostly from ensembling; corrected labels lift a fixed model's opening F1 from 0.29 to 0.66 | 256 px; residential; system-level comparison | High (evaluation design; supports detection plus graph) |
| [Rodionov et al. 2025 (FloorplanQA)](https://arxiv.org/abs/2507.07644) | Benchmark | Spatial reasoning over JSON layouts | Best 75.5%; images-only 19–40% | Single rooms; no perception | Medium |
| [Du et al. 2026 (MMArch)](https://arxiv.org/abs/2608.09281)\[131\] | Benchmark | 1,212 expert-verified questions on figures from AEC papers, 10 subdomains | Best model 51.7%, best open model 29.9%, experts 94.6% | Not plan parsing; items filtered against a solver model; several open-model rows look derived from others | Low |
| [Chen et al. 2026 (FloorPlan-VLN)](https://arxiv.org/abs/2603.17437)\[133\] | Benchmark | Floor-plan-guided navigation: plans with typed room regions as a global prior for a fine-tuned VLM | Success rate 28.8% in unseen scenes vs 17.0% for a fine-tuned baseline; masking the plan costs 41% | Plans drawn from Matterport3D region annotations, without walls or doors; single floors; simulation | Low |
| [Kondratenko et al. 2026 (AECV-Bench)](https://arxiv.org/abs/2601.04819) | Benchmark | VLM counting and QA benchmark on drawings | Counting exact 0.39–0.51 mean; text up to 0.95 | Whole-image prompting; residential | High |

Not reviewed (no open copy): CVC-FP (de las Heras et al. 2015), MLSTRUCT-FP (Pizarro et al. 2023) and ArchPlanVQA (Li et al. 2026). Their data and findings appear in §6.3 and §7.

### Practitioner Sources

| Source | Strategy | Content | Evidence | Main limitation | Relevance |
|---|---|---|---|---|---|
| AxcelerateAI, solution page | Vendor pipeline | Six stages, OCR before walls; room tags seed room polygons; output schema rooms/walls/doors/windows/text/symbols; on-premise deployment | Vendor claims | No metrics | Medium |
| AxcelerateAI, tutorial (16 Sep 2026) | Vendor pipeline | Denoise/deskew/tile → wall segmentation (U-Net/HRNet) → doors/windows (YOLOv8, oriented boxes) → OCR (DBNet) → room adjacency graph | Demo on a residential sample | No metrics or code | Medium |
| AxcelerateAI, model guide (Aug 2026) | Vendor pipeline | HRNet/U-Net walls, YOLO oriented-box symbols, Mask R-CNN door swings, DBNet+CRNN text, ViT+GNN topology | "98%+" unsupported | Licences not discussed | Low |

The comparison with pilot v2 is in the [review §5](reviews/2026-10-07-pipeline-and-pilot-v2.md#5-comparison-with-axcelerateai).

## 9. Gaps and Research Agenda

What the literature does not provide, and BBL has to build or test itself. Status as of 7 October 2026; step numbers refer to the [review's next steps](reviews/2026-10-07-pipeline-and-pilot-v2.md#7-recommended-next-steps).

| # | Gap | New literature | Status |
|---|---|---|---|
| 1 | **Benchmarks on BBL-like plans:** office and historical buildings, scans, hatched and coloured walls, full-size sheets, with area error in m² as a metric | VectorGraphNET (university plans, proprietary); Versailles-FP; Trivi; ResPlan-FP (residential renders) and an edit-cost metric (Zhang 2026) | Started: CubiCasa5K zero-shot benchmark (step 1a) and 35 public sample sheets in the viewer (1b). Evaluation harness (1c) and BBL gold set open |
| 2 | **Assigning room stamps to rooms:** multi-field stamps (AOID or number, usage, area) in DE/FR/IT, OCR errors, old room numbers matched to SAP | Word Beam Search and PP-OCRv6 as components; drawing-text readers trained on synthetic drawings (Schlagenhauf et al., Ying et al.) | Started: pilot v2 parses stamps, and area tokens with "m²" win over bare numbers. Dictionary decoding and SAP reconciliation (step 8) open |
| 3 | **Door-to-room connections** from recognition, not only from clean vector data | – | Started: pilot v2 links openings to the rooms on both sides (12 of 12 on the CAD print). Connectivity metric on public data (MSD) open |
| 4 | **Stairs and voids:** detecting flights and voids, and applying deduction rules | – | Open: voids are found only from printed labels; a void class is planned (step 5) |
| 5 | **Gross floor area** to SIA 416 rules, and scale recovery that is evaluated | Lv et al., Talebi-Kalaleh & Mei, Liang et al., Lin & Wang, Chang et al., Faltin et al. (dimension strings); EXSCLAIM! (scale bars, microscopy). Only Talebi-Kalaleh & Mei report scale error | Started: scale proposal from door widths (step 6a). Dimension strings, scale bar and scale note are in progress: open in the review (step 6b) and being added to pilot v2's scale module. Scale per drawing is now designed as pipeline stage 1c ([domain review §7](reviews/2026-10-07-pipeline-target-inputs-masking-scale.md#7-recommendations-and-status)). A GF reference to SIA 416 is open |
| 6 | **Walls that are not axis-aligned:** curved and oblique walls together with typed junctions on large sheets (most methods are Manhattan and 512 px) | Lv et al. (soft orthogonality) | Open |
| 7 | **Office-size floors:** learned room decoders cap at 20 rooms and graph methods at about 50 nodes | Raster2Seq handles more complex plans, within 512 tokens | Partly addressed: pilot v2's region approach has no room cap (more than 60 rooms on public historical plans) |
| 8 | **Robustness to drawing style:** no paper tests training on rendered vector data in randomised styles against real scans | Semap (maps); MIC, HRDA (street scenes) | Partly answered by our evidence: trained only on style-randomised renders, pilot v2 reaches 0.57 wall IoU zero-shot on 400 CubiCasa5K plans (in-domain model 0.73) and 0.58 on a poché scan ([review §2](reviews/2026-10-07-pipeline-and-pilot-v2.md#2-evidence)). Renderer v2 (step 4b), self-training and a permissive real validation set open |
| 9 | **Licence-clean pretrained weights:** almost all public weights are trained on non-commercial data | – | Started: pilot v2 trains on CC BY 4.0 data only. Its ImageNet-pretrained encoder is a question for the legal review ([review §6.4](reviews/2026-10-07-pipeline-and-pilot-v2.md#64-questions-for-bbls-legal-service)) |
| 10 | **Topology-aware training judged by rooms:** no paper measures the effect of topology losses on room recognition in plans | TopoMortar and Chen et al. 2024 disagree | Open |
| 11 | **Adaptation to an archive with few labels** and many unlabelled scans | MIC, HRDA, Petitpierre & Guhennec, Trivi | Open: self-training and learning curves proposed in the review (§4.3) |
| 12 | **Open-plan rooms** separated only by virtual lines | Chen et al. 2024 (watershed); Raster2Seq | Open: with perfect labels, post-processing finds 77% of rooms. Stamp-seeded watershed and a room-boundary head proposed (review §3.3, step 5) |
| 13 | **Several drawings on one sheet** | Partly answered outside floor plans: views named by their captions on bridge drawings (Peng et al. 2025); region models in document parsing (PP-StructureV3); DWG viewports with their own scale. Lv et al. detect one plan region only | Designed as pipeline stage 1b with a `drawings[]` level in the data model ([domain review §7](reviews/2026-10-07-pipeline-target-inputs-masking-scale.md#7-recommendations-and-status)). Implementation and a paper on floor plan sheets open |

## 10. Implications for the Pipeline Design

Proposed updates to [pipeline.md](pipeline.md).
- Items 1–7 come from the first review, 8–15 from the papers added in October 2026.
- Items 16–19 come from the [domain review of inputs, masking and scale](reviews/2026-10-07-pipeline-target-inputs-masking-scale.md). Its recommendations are implemented in pipeline.md draft 0.3; implementation status is in that review's §7.

Status: Partly (adopted in pilot v2 in part), Changed (pilot v2 took another route), Designed (in pipeline.md, not yet implemented), In progress or Open.

| # | Implication | After | Status |
|---|---|---|---|
| 1 | **Data model:** walls by centre line, thickness and curvature; openings nested in walls with width and offset; rooms as cycles of wall IDs | FloorplanVLM | Partly: pilot v2's typed model has wall centre lines and thickness, and openings with host wall, width and the two spaces they connect. Rooms are regions, not wall cycles (review §3.2) |
| 2 | **Stage 3:** typed junction keypoints with offset regression; segment classification over junction candidates that supports oblique walls; a constraint layer for degrees, exclusion, loops and openings on walls, with an open-source solver | Chen et al. 2023, Pang et al., Raster-to-Vector | Open: junction heads planned for model v2 (step 5) |
| 3 | **Stage 6:** rooms as bounded faces with sliver merging; regularisation by angle-based vertex selection plus Douglas–Peucker; learned polygon heads only as a fallback; room count checked against room stamps | Sketch2BIM, PolyRoom | Changed: pilot v2 uses free-space regions between walls and closed openings, which the review recommends keeping. Stamp checks are adopted (stamps per room, stamp vs polygon area); regularisation is open |
| 4 | **Stage 8:** an MSD/ResPlan-style edge schema (door, passage, adjacency, window) plus vertical links; each door linked to the two rooms on either side, not to the nearest centroids | MSD, ResPlan | Partly: doors and passages link the rooms on both sides. Edge types (front door, window adjacency) and vertical links open (review §3.3) |
| 5 | **Stage 9:** ResPlan's QA rules; scale from stamp areas; Angle F1, Hausdorff distance and area error in m² as metrics | ResPlan, Dodge et al. | Partly: stamp-area scale cue and QA checks (GF above the sum of rooms, hosted openings, reachability) adopted. Metrics in the harness (step 1c) open |
| 6 | **Training data:** an ArchCAD-400K-style annotation engine for compliant DWGs; Chen et al.'s augmentation list in the renderer; Egiazarian-style scan degradation; clustering-based sampling; Swiss Dwellings including its office and void subtypes | ArchCAD-400K, Chen et al. 2023, Egiazarian et al., FloorplanVLM | Partly: Swiss Dwellings rendered in random styles with basic scan defects; IFC-Bench storeys prepared (step 4a). Augraphy, void and office content in renderer v2 (step 4b). The DWG annotation engine needs BBL DWGs |
| 7 | **Evaluation:** a held-out set of real BBL sheets plus WAFFLE and CVC-FP as tests for unseen styles. Published in-domain scores are not planning targets. Add an edit-cost score as a proxy for correction time, and re-score CubiCasa5K against corrected labels | Zhang 2026 for the additions | Partly: CubiCasa5K zero-shot done; CVC-FP and WAFFLE samples in the viewer. Full runs (step 1c), the additions and the BBL gold set open |
| 8 | **Renderer:** composite poché, hatching and ornament cut from archive scans into wall and non-wall regions; oversample floors with columns and stairs | Semap, DAFormer (rare-class sampling) | Open (step 4b, [review §4.2](reviews/2026-10-07-pipeline-and-pilot-v2.md#42-renderer-synthpy)) |
| 9 | **Training:** self-training on unlabelled archive scans with masked consistency, masking only the scans; detail and context crops for thin walls on large sheets | MIC, HRDA | Open: the review proposes flip-agreement pseudo-labels (§4.3) |
| 10 | **Wall loss:** A/B of cross-entropy + Dice, clDice and Skeleton Recall on the wall channel, with augmentation and self-distillation as the cheap baseline, scored by rooms matched and Betti errors | TopoMortar, Chen et al. 2024 | Open |
| 11 | **Rooms:** a watershed on the wall probability map seeded at room stamps, to split open-plan faces and close small gaps; a retrained Raster2Seq as a second room hypothesis, with disagreements flagged | Chen et al. 2024, Raster2Seq | Open (review §3.3). Pilot v2 already uses a watershed to seal rooms that leak through an unclosed opening |
| 12 | **Stamps:** decode the recogniser's CTC output against a DE/FR/IT usage dictionary, leaving numbers free; move to PP-OCRv6; fine-tune detector and recogniser on rendered stamps and dimension strings, including rotated text | Word Beam Search, PP-OCRv6, Schlagenhauf et al., Ying et al. | Open: pilot v2 corrects names with a vocabulary after OCR |
| 13 | **Scale:** pair dimension strings with dimension lines under explicit tolerances, reject outliers by median absolute deviation, report the inlier share as confidence, snap to standard scales | Talebi-Kalaleh & Mei, Lv et al., Liang et al., Lin & Wang | In progress: open in the pilot review (step 6b) and being added to pilot v2's scale module; designed per drawing as stage 1c (item 18) |
| 14 | **Vector-PDF branch:** parse the PDF drawing operators (line weight, dash, fill) and classify paths, instead of rasterising CAD prints | Talebi-Kalaleh & Mei, VectorGraphNET, VectorFloorSeg | Open (step 7) |
| 15 | **Gold-set labelling:** few visually homogeneous classes, label only what is visible, about a dozen sheets per archive style, semi-automatic pre-labels | Petitpierre & Guhennec, Versailles-FP | Open ([review §6.2](reviews/2026-10-07-pipeline-and-pilot-v2.md#62-bbl-sample-data-to-request)) |
| 16 | **Inputs:** one canonical raster per page or DWG layout, with native text, paper size, dimension geometry and DWG viewports as side channels. Delivery DWGs are rendered as plotted; layer names are trusted only for compliant DWGs | DPSS, FloorPlanCAD, VectorFloorSeg (hybrid ablations); Yin et al. (layers by content) | Designed: stages 0a and 0b; implementation open |
| 17 | **Sheet layout (stage 1b):** drawing regions, title block, legend, scale bar and north arrow as polygons; masks in inference; ignore labels outside drawings in training; title blocks read by crop-then-parse | FELD, Lombardi et al., Peng et al. 2025 and 2026, Pang et al., Lv et al. (cropping), MapSeg (masks) | Designed: stage 1b and a `drawings[]` level in the data model; rule baseline and detector open |
| 18 | **Scale per drawing (stage 1c)** before segmentation, confirmed in stage 9. Drawing scale, resolution and print factor are kept apart. Dimension strings and scale bars measure metres per pixel directly, with a robust consensus and separate x and y fits | Talebi-Kalaleh & Mei, Liang et al., Lin & Wang, Chang et al., EXSCLAIM! | Designed: stages 1c and 9; per-region use in pilot v2 open |
| 19 | **Storey linkage by grid axes:** read axis labels and register storeys by shared labels; re-read only the regions a validator flags | BlueprintAgent, Lin & Wang | Open ([pipeline open questions](pipeline.md#9-open-questions)) |

## 11. Limitations of This Review

- **Not legal advice.** Licences for several repositories, weights and datasets could not be confirmed: DPSS weights, Raster-to-Graph data, R2V, WAFFLE images, the component licences of MIC and HRDA, Semap's data. The SymPoint-V2, VectorFloorSeg and CADSpotting repositories have no licence file, which grants no reuse rights. Verify each LICENSE file and dataset terms before use. Weights trained on CC BY-NC data should be assumed unusable for production federal work without legal review.
- **Self-reported metrics.** FloorPlanCAD PQ, Structured3D room F1, FPBench-2K IoU, AECV-Bench counts and street-scene mIoU measure different tasks and are not comparable.
- **Coverage.**
  - Five indexed papers could not be obtained, and ten were summarised from their abstracts.
  - The sheet-layout and scale evidence comes mostly from façade, bridge, mechanical and structural drawings, maps and microscopy. No paper evaluates sheet layout, masking or scale recovery on archive floor plans.
  - Docling garbled some tables; such figures are flagged or omitted.
  - The learning-strategy papers come from street scenes, maps and brick walls; their transfer to plans is our inference.
- **Vendors.** Vendor capabilities, prices and hosting statements come from vendor pages and may change.
- **Recent preprints.** Several 2026 papers are preprints with unverified code and reproducibility: TextCAD, PolarSym, FloorplanVLM, Semap, Talebi-Kalaleh & Mei, PP-OCRv6 and the Guo et al. preprint.
- **A fast-moving field.** OCR, detection and segmentation models were superseded several times during 2025–2026. Re-check versions before implementation.
- **Maintenance.** Maintenance status is not checked here. Many research repositories pin old CUDA/PyTorch versions (e.g. FRI-Net and RoomFormer: PyTorch 1.9, CUDA 11.1) and may need porting;\[50\] activity is in [open solutions](solutions-open.md).
- **Our evidence is thin.** No published model was run on BBL plans. Pilot v2 ran our own segmenter on three sheets of one floor, with post-processing rules shaped on those sheets, so its numbers are optimistic. The share of the archive in each input class and graphical style is unknown; recommendations should be revisited after the inventory step.

## 12. Sources

- Papers and practitioner pages: [research index](../research/README.md) and [papers.json](../research/papers.json); Markdown versions in `research/papers-md/` (local only).
- Shared [source list](../research/sources.md); numbers 679–710 were added for the papers integrated in October 2026.
- [Motivation and goals](motivation-goals.md) · [Pipeline design](pipeline.md) · [Open solutions](solutions-open.md) · [Closed solutions](solutions-closed.md) · [Open datasets](datasets-open.md) · [Closed datasets](datasets-closed.md) · [Pipeline and pilot v2 review](reviews/2026-10-07-pipeline-and-pilot-v2.md) · [Review of inputs, masking and scale](reviews/2026-10-07-pipeline-target-inputs-masking-scale.md) · [Pilot v1](../pilot/landgut-lohn-og1/README.md) · [Pilot v2](../pilot/v2-pipeline/README.md)
