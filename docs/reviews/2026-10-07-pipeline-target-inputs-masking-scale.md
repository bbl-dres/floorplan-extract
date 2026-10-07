# Review: Pipeline as Canonical Target – Inputs, Masking and Scale

*Domain review · 7 October 2026 · Scope: [docs/pipeline.md](../pipeline.md) draft 0.2 as the canonical target concept, measured against one requirement: users will upload any kind of plan (JPG, PNG, TIFF, PDF with vector and raster content, DWG produced for print), and the pipeline has to isolate the relevant drawing content and find its scale on its own. Evidence: the indexed papers in `research/papers-md`, a targeted search for what the corpus does not cover (49 new papers, 28 standards and tool sources; list in §8), [pilot v2](../../pilot/v2-pipeline/README.md) and the [pipeline review](2026-10-07-pipeline-and-pilot-v2.md); the full evidence is in the [evidence brief](../reports/2026-10-07-pipeline-inputs-masking-scale-evidence.md) and the [report on construction drawings and FM plans](../reports/2026-10-07-construction-and-fm-drawings.md). The recommendations are implemented in [pipeline.md draft 0.3](../pipeline.md); status in §7.*

## 1. Verdict

Draft 0.2 describes the stages for **one drawing at a known scale** well. As the canonical target for arbitrary uploads it has four structural gaps:

1. **No normalisation layer.** Inputs are routed by class into different extractors. Most real DWGs and PDFs are print products with noisy vectors, so the routing table's optimism (direct extraction, vector symbol spotting) does not match what BBL receives.
2. **No sheet-layout stage.** Nothing isolates the drawing from the title block (Plankopf), legend, key plan, notes and frame, and the data model has no level between the sheet and its elements. Sheets with several drawings at different scales cannot be represented.
3. **Scale sits in stage 9 only.** The segmenter needs the scale before stage 3; pilot v2 already estimates it in a pre-pass that the design does not mention. Scale is also a property of each drawing, not of the sheet.
4. **DWG handling is underspecified.** The draft does not say whether model space or the plotted layouts are rendered, and names one converter whose licence does not fit federal use without clarification.

Two smaller points: drawing types (construction drawings vs. facility-management plans) are not a stratum anywhere, and text is masked before segmentation although the learned path no longer needs that.

All four gaps have good evidence for a fix. None of the fixes changes stages 3–8.

## 2. Inputs: Raster-Canonical, Vector-Assisted

### 2.1 What the evidence says

The user's instinct, flatten noisy vectors to raster, is supported, with one qualification: flatten for perception, keep the vectors for what they do exactly.

- **Archive vectors are noisy, and vector models depend on the layers that make them clean.** FloorPlanCAD's authors found that "the layer name does not necessarily explain its content"; ArchCAD-400K kept 5,538 of 11,917 industry drawings (47%) after layer screening, and the screened labels still needed ten drafters to correct. Without layers, SymPoint-V2's wall PQ drops from 80.8 to 49.3.
- **Outlined text and unclosed outlines break vector parsing.** VectorGraphNET's errors concentrate on text exported as geometry; CADSpotting itself rasterises its vectors to close wall polygons. Pilot v2's CAD print had a text layer for dimensions only; its room stamps were outlines.
- **Image plus primitives beats either alone, in every ablation found:** DPSS 86.2 PQ against 81.7 (primitives) and 80.9 (image); FloorPlanCAD weighted F1 0.308 → 0.755 with render features; VectorFloorSeg 61.3 → 75.6 mIoU. These hybrids need vector-level labels, which only compliant BBL DWGs could supply.
- **Rendering works:** the pilot's rendered CAD print reached wall IoU 0.89 and 15/15 rooms.

### 2.2 Recommendation

**One canonical intermediate per page or DWG layout, the sheet package:** a raster at a recorded paper resolution, the paper size, optional vector side channels (paths with width, colour, layer; text runs with position, flagged when outlined or invisible), and the transforms source ↔ paper millimetres ↔ pixels ↔ plan metres. Every learned stage reads the raster, so one model, one renderer and one evaluation serve all input classes.

The vector side channel is used for five things only:

| Use | How |
|---|---|
| Text | Native strings and positions, merged with OCR of the render by overlap (the text layer is often partial) |
| Scale | Exact paper size; dimension text and lines in paper millimetres; DWG viewport scales; PDF measurement viewports (`/VP`, ISO 32000-1 §12.9) as a cue, never as truth |
| Geometry refinement | After segmentation, snap wall faces to vector segments that pass an "ink gate" both ways (Zhang 2026); rooms stay regions of raster-derived walls, so unclosed vector outlines never define a room |
| Region proposals | Frames, title-block tables, DWG viewports, PDF optional-content names (§3) |
| Layer hints | Only for DWGs that pass plan-check or match a known author profile (the analogue of ArchCAD-400K's screening) |

A learned hybrid (DPSS-style) stays a later branch, to be evaluated once BBL has vector-labelled DWGs.

### 2.3 Per input type

| Input | What goes wrong | Target handling |
|---|---|---|
| JPG, PNG | DPI missing or a 72/96 default; JPEG artefacts; photos with perspective | DPI trusted by source (scan header > paper-size match > metadata); never trust 72/96; photos and perspective flagged at triage |
| TIFF | Multi-page; 1-bit CCITT G4; very large | Split pages; expand 1-bit to 8-bit; tile at native resolution |
| PDF, vector | Outlined text (SHX fonts are always exported as geometry); partial text layer; single-precision coordinates; markups in annotations | Render per page at a resolution chosen from the smallest text height (the pilot needed 1200 dpi); render with and without annotations; detect outlined text and send it to OCR; text-layer coverage per role, not per file |
| PDF, raster or mixed | One page image, possibly with an invisible OCR layer; scans with vector redlines | Classify per page (vector, raster, mixed) by content; treat the image as a scan; keep vector redlines as overlay masks |
| DWG, compliant | Rare in the archive | Model space at 1:1 (plan-check requires mm, model space only, no xrefs, title block on `V_PLANLAYOUT`) → direct extraction, plus a render for review |
| DWG, delivery and print | Random layers, exploded blocks, unclosed rooms; content split over paper-space layouts and viewports at different scales; per-viewport frozen layers; missing xrefs; plot style files (CTB/STB) not in the upload | Render **each paper-space layout as plotted**: every viewport becomes a drawing region with an exact scale and clip boundary. Model space only when there are no layouts (then expect several plans side by side). Require the complete package (eTransmit) and flag missing xrefs; render with a default monochrome plot style and randomise line weights in training |

**DWG converter.** No permissive, production-grade DWG reader exists. The interface should be a swappable `DWG → DXF` step followed by ezdxf (MIT), which already reads DXF and renders layouts:

| Option | Licence | Note |
|---|---|---|
| LibreDWG | GPL-3.0 | Reads all versions with gaps; CVE history, so run sandboxed on untrusted uploads |
| QCAD Professional (command-line tools) | Proprietary; server licence (CHF 480 per server) explicitly allows use as a web service | Renders layouts to PDF or bitmap; viewport and plot-style fidelity to verify |
| ODA File Converter | Proprietary freeware, "for non-commercial applications only" for non-members | Needs ODA's written confirmation for internal federal use, or ODA membership |
| Cloud converters (Autodesk APS) | – | No Swiss region: excluded by design principle 7 |

Under Art. 9 EMBAG (guidance note EFD/BBL/KBB, 11.06.2025), software developed by or for the Confederation is published as open source. That makes copyleft dependencies (LibreDWG GPL, PyMuPDF AGPL) a question of BBL's own outbound licence rather than a blocker; the real licence risk is a proprietary converter. If BBL prefers a permissive licence for its code, pypdfium2 (Apache/BSD) and pdfplumber (MIT) replace PyMuPDF.

## 3. Masking: a Sheet-Layout Stage

### 3.1 What the evidence says

- **Region detection is learnable from a few hundred labelled sheets.** Pang et al. (300 training sheets): main drawing F1 1.00, title bar 1.00, legend 1.00, scale text 0.97, north arrow 0.95. Lombardi et al. (1,385 building drawings): title block 0.971, main content 0.943, legend 0.86 at IoU ≥ 0.7. FELD (551 façade sheets): RF-DETR mAP50:95 0.890 overall, title block 0.989, plan 0.969.
- **Cropping to the drawing helps downstream.** Lv et al. note that the plan "may only account for < 50% of the area" of an input picture; adding their region detector raised two baselines from 0.76 to 0.80 and 0.82 mIoU.
- **General document-layout pretraining hurts** on drawings (FELD ablation: DocLayout-YOLO's document weights performed worse than a COCO-pretrained baseline). Zero-shot VLMs fail to localise title blocks (Peng et al. 2026; GPT-4o missed 6 of 10 in Lombardi et al.), but read cropped title blocks well.
- **Maps already mask by content area.** The ICDAR 2021 MapSeg competition scores content-area masks by the 95th-percentile Hausdorff distance and ships input masks outside which predictions are discarded. No floor plan dataset does this.
- **Swiss conventions are strong priors.** SIA 400 puts the title block bottom right with plan number, scale(s), revision index, date and format, and allows key plan, north indication, drawing scales and legends in fields above it. CADexchange 4.3 (BBL is a member) makes a graphical scale, a north arrow and a key plan mandatory in the title block, requires a cut edge enclosing all plan content, and allows title blocks to be scaled down on small formats.

### 3.2 Recommendation

**Add stage 1b, sheet layout and masking.** It outputs `regions[]`: class, polygon (sheet pixels and paper millimetres), confidence, source (`dwg-viewport`, `pdf-vector`, `detector`, `rule`, `human`) and links (drawing ↔ caption; legend ↔ drawings; title block ↔ sheet).

- **Classes:** drawing (subtype floor plan, section, elevation, detail, site plan, key plan), title block, legend, scale bar, scale note, north arrow, notes, revision table, frame, stamp or seal, colour key, other.
- **Polygons, not boxes:** plans are L- or U-shaped, and title blocks often sit inside a drawing's bounding box. Refine each drawing box with the ink inside it (building components plus dimension chains, minus the other regions).
- **Sources in order:** regions from the file where it has them (DWG viewports, PDF frames and title-block tables), a detector on rasters, rules as a zero-label baseline and for pre-labels (frame as the largest rectangle; title block as a ruled table in a corner; drawings as ink clusters).
- **Detector:** RF-DETR with a mask head (Apache-2.0; already the candidate for openings; best on FELD), from COCO weights, with a second high-resolution pass for small objects (north arrows, scale bars, captions). Train on synthetic sheets (frames, Swiss-style title blocks, legends, several drawings at different scales, random crops as in Semap) plus 100–300 labelled BBL sheets across authors and eras.
- **Masks in inference:** run walls, openings, stairs and rooms only inside drawing polygons (plus a margin for dimension chains). OCR the sheet once and assign text roles by region: title block → sheet metadata, legend → legend entries, drawing → stamps and dimensions.
- **Masks in training:** on real labelled sheets, annotate only inside drawing regions and set the rest to the ignore label (255, as in Cityscapes and MMSegmentation); on synthetic sheets, render title blocks, legends, frames and key plans as explicit background, because their ruled lines and hatch swatches are what a wall model confuses with walls. Renderer v2 already does the latter.
- **Read title blocks by crop-then-parse:** detect, crop at full resolution, OCR plus key–value parsing; a fine-tuned self-hosted VLM read the "Massstab" field with 0.99 word accuracy (0.57 zero-shot).
- **Triage changes meaning:** "one floor per sheet" becomes "one floor per drawing region"; sheets with several drawings are normal input, not a defect.

## 4. Scale: per Drawing, Before Segmentation

### 4.1 What the evidence says

- **The factors must be kept apart.** Metres per pixel m = N · 25.4 / (1000 · d · r), with N the drawing scale (1:N), d the resolution in dpi and r the reproduction factor of the print (1 for the original, 0.5 for an A1 sheet copied to A3). Dimension strings, scale bars, area stamps and element sizes measure m directly; a scale note gives only N and needs d and r. SIA 400 expects reductions in √2 steps and requires them to be labelled; ISO 5455 warns that "the scale of a print may be different from that of the original drawing". All A formats share the 1:√2 aspect ratio, so the sheet outline cannot tell A1 from A3.
- **Dimension strings are the most precise cue.** Talebi-Kalaleh & Mei pair dimension text with dimension lines (text within 12 mm, direction within 6°, ticks at both ends), reject outliers by median absolute deviation and snap to standard scales: 0.086% maximum error, on generated drawings only. On rasters, Lv et al. and three 2025–2026 papers (Chang et al., Lin & Wang, Liang et al.) use the same pattern with OCR; Liang et al. fit x and y separately to catch anisotropic copies.
- **Scale bars** have no floor plan paper; microscopy and map work (EXSCLAIM!, Uni-AIMS, a Library of Congress map pipeline) detect the bar and its labels, re-crop at full resolution, read, and keep the arithmetic deterministic.
- **Only one paper evaluates scale recovery at all, on generated drawings.** VLMs read scale fields but must not measure: zero-shot room areas were off by about 43%.
- **Pilot v2 already implements most of this** in `fpx/scale.py`: five cues (scale note with DPI, dimension strings, scale bar, door widths, stamp areas), a precision-weighted consensus and flags. On 40 CVC-FP scans the consensus found a scale on 39, median error 2.9%; it diagnosed the pilot's CAD print as a reduced print (note 1:100, drawing measured 1:291 at 1,200 dpi). It works per sheet, not per drawing.

### 4.2 Recommendation

1. **Scale belongs to the drawing region.** Each region takes its own note from its caption ("Grundriss 1. OG 1:100", "Detail 1:20"); the title-block scale is only the sheet default. Never pool dimension strings across regions.
2. **Estimate early, confirm late:** stage 1c proposes the scale per drawing before resampling; stage 9 confirms it with stamp areas and element sizes once rooms exist, and reruns a drawing whose scale moves by more than a few per cent.
3. **Cues in trust order, all reported:** DWG viewport scale or model units (exact) > reviewer two-point calibration > dimension strings > scale bar > scale note × resolution × print factor > stamp areas > element-size priors (only to choose between discrete hypotheses) > sheet frame and page size > embedded PDF measurement scale.
4. **Consensus:** enumerate discrete hypotheses jointly (units m/cm/mm and Swiss cm with superscript mm; OCR factors of 10; print factor in √2 steps; N from the SIA/ISO list), score them by inliers among the direct cues, fit x and y separately, snap to a standard scale only when the print factor is trusted, auto-accept within about ±1%, otherwise ask the reviewer.
5. **Flags:** cues disagree; one cue or priors only; reduced or enlarged print suspected (note vs. direct cues off by a power of √2); anisotropy above 1%; several scales on the sheet; region without its own scale; "nicht massstäblich" / NTS; mixed units; DPI missing or a default.
6. **Metric:** scale error per input class against DWG-derived truth (paired DWG and scan sheets).

## 5. Other Sheet Features

| Feature | Why the pipeline needs it | Output | Priority |
|---|---|---|---|
| Storey label (EG, 1. OG, UG, rez-de-chaussée, piano terreno) | Assigns the drawing to a floor (GF per floor, SAP floors, AOID), orders sets | `drawing.storey` | High |
| Drawing title and type (Grundriss, Schnitt, Ansicht, Detail, Situation) | Only floor plans go to room extraction | `drawing.kind`, `drawing.title` | High |
| Scale note and scale bar per drawing | §4 | `drawing.scale` with cues | High |
| Title-block metadata (building, plan number, date, author, revision) | Links the sheet to the building and its AOID prefix; detects stale versions | `sheet.meta` | High |
| Legend entries (hatch and colour meaning) | Massive walls for `A_SCHRAFFUR`, zones, new/existing/demolished | `legend[]` with matched elements | High for wall types |
| Reference points "REF.PKT" with LV95 coordinates (CADexchange) | Two points georeference the plan; identical on all storeys | `drawing.refpoints[]` | High where present |
| North arrow and angle (mandatory in CADexchange title blocks) | Initial rotation for registration to the official survey (AV) footprint | `drawing.north` | Medium |
| Grid axes and labels | Registration across storeys and sheets; columns; `V_ACHSEN` export | `axes[]` | Medium (high for multi-storey) |
| Key plan | Must not become a second building; its highlight shows which wing is drawn | Excluded region, highlighted sector | Medium |
| Section and detail markers, revision clouds, stamps, colour overlays | Not walls or doors; occlusion; revisions may show a newer state | Masks and QA flags | Medium |
| Notes ("nicht massstäblich", area conventions) | Scale and area flags | QA flags | Low |

No openly licensed plan dataset annotates any of these. BBL will need to label its own sheets; renders can supply the synthetic part.

## 6. Drawing Types: Construction Drawings and FM Plans

BBL receives two families of plans, and they differ more than the training data suggests:

- **Construction drawings** (SIA 400 Ausführungs- and Werkpläne, typically 1:50): dense dimension chains, axis grids, level markers, section and detail markers, door and room tags, material hatching per wall type, furniture and fixtures, several drawings per sheet.
- **Facility-management plans** (area management, typically 1:100–1:200, following the CAD-Richtlinie BBL): room polygons with AOID stamps and areas, colour fills by usage, fewer annotations, but very heterogeneous in style and often with complex wall hatches.

Both differ from the residential plans that dominate public data, and the difference matters:

- **Domain shift is large and mostly recognition, not geometry.** Moved from FloorPlanCAD to large office, campus and hotel drawings, SymPoint falls from 83.3 to 33.2 PQ and CADSpotting from 87.4 to 52.9; recognition quality drops from about 91 to 40 while segmentation quality holds. On the mostly non-residential ArchCAD-400K the same methods reach 47.6–70.6. Pilot v2's weakest CubiCasa5K group is the most construction-like (`high_quality_architectural`, wall IoU 0.53 against 0.74).
- **The cause is clutter that looks like walls:** hatches outside walls, dimension and axis lines, leaders and text. Pooled or in-domain training closes most of the gap (Dodge et al. 76–82 → 90–93 mean IoU; joint training on LS-CAD 75.5 PQ; Zhang 2026 wall F1 0.64 → 0.97 after fine-tuning).
- **FM plans are a third style** that no paper tests: grey solid walls next to grey usage fills, magenta room-polygon lines on the wall faces, boxed room stamps. The [CAD-Richtlinie BBL V1.0](https://github.com/bbl-dres/plan-check) describes the target (15 layers, grey solid walls, magenta room polygons, one AOID per room, no furniture); legacy BBL plans look very different (boxed multi-attribute stamps, door-attribute blocks, dozens of dimensions, axes, section lines, line hatches over solid fills).
- **No construction-style or CAFM dataset is licence-clean for training.** ArchCAD-400K (CC BY-NC, gated, has axis and grid classes) and FloorPlanCAD (CC BY-NC; its November 2021 release keeps text, CAD layers, dimension chains and axes) serve evaluation only. Licence-clean real examples are US federal works (USACE construction documents, NARA working drawings, HABS), without labels: 15 are in `data/benchmark/construction-plans/`, and a 30-block FloorPlanCAD sample with six-class masks across schools, hospitals, offices, malls and residential towers is in `data/benchmark/floorplancad/sample/`.
- **BIM can generate construction-style training drawings.** IfcOpenShell (LGPL-3.0) draws clean storey plans with door arcs and room stamps from the permissively licensed IFC models, but no dimensions, axes, material hatches or tags; all 17 models carry wall materials and door tags, 16 named spaces and 3 grid axes. The annotations are best drawn by our own renderer from the IFC data.

Consequences for the design: drawing type is an evaluation stratum (§8 of pipeline.md) and a renderer target. **Renderer v3** should add, tied to the geometry: dimension chains and axis grids, opening, door and wall tags, level marks, section lines, SIA 400 material hatches and existing/new/demolition colours, a CAFM style (magenta polygons, usage fills with a legend, AOID or boxed stamps), office furniture, overlay plans, full sheets with title blocks, and blueprints (inverted polarity).

## 7. Recommendations and Status

Status: Done, Open or Declined.

| # | Recommendation | Where | Status |
|---|---|---|---|
| 1 | Canonical sheet package per page or layout: raster + paper size + vector side channels + transforms | pipeline.md §4, stage 0a; `fpx/inputs.py` | Done: design and implementation (rasters with DPI trust, PDF with page classification and outlined-text flags, DXF layouts and viewports) |
| 2 | Input handling per type (§2.3), including per-page PDF classification and text-layer coverage per role | pipeline.md stage 0a | Done in the design |
| 3 | DWG: render paper-space layouts as plotted; compliant DWGs via model space; swappable converter interface; no converter implemented yet | pipeline.md stages 0a/0b, §6; `fpx/inputs.py` | Done: DXF layouts rendered as plotted, viewports with scale and clip; `DwgConverter` interface (LibreDWG, QCAD, ODA adapters; none installed) |
| 4 | Decide the DWG converter (LibreDWG sandboxed, QCAD Professional server licence, or ODA with written confirmation) and BBL's outbound licence under Art. 9 EMBAG | pipeline.md §9 | Open (BBL decision) |
| 5 | Stage 1b sheet layout and masking: regions with polygons; masks in inference and training | pipeline.md stage 1b; `fpx/layout.py` (rule baseline) | Done: rule baseline finds the right number of drawings on 41 of 44 curated plans, and the pilot's CAD print without its hand-drawn region; learned detector Open (item 10) |
| 6 | Data model: `drawings[]` with polygon, kind, title, storey, scale, north, reference points; elements belong to a drawing | pipeline.md §4 | Done in the design |
| 7 | Scale per drawing, proposed in stage 1c before segmentation, confirmed in stage 9; factor model and flags | pipeline.md stages 1c and 9; `fpx/scale.py` per region | Done: per-drawing scale with the factor model; on the pilot's CAD print 0.36% from the calibration, reduced print (2^−1.5) identified; `fpx.pipeline.run_document` runs any upload per drawing |
| 8 | Title blocks by crop-then-parse; Swiss priors (SIA 400, CADexchange) | pipeline.md stage 1b | Done in the design |
| 9 | Text masking only for classical baselines; text roles assigned by region | pipeline.md stage 2 | Done in the design |
| 10 | Layout detector (RF-DETR seg) on synthetic sheets plus 100–300 labelled BBL sheets | pipeline.md §6–§7 | Open |
| 11 | Drawing type (construction, FM, historical) as evaluation stratum and renderer target | pipeline.md §7–§8 | Done in the design; benchmark data collected (§6) |
| 11a | Renderer v3: construction-drawing and CAFM content tied to the geometry, from Swiss Dwellings and IFC (§6) | `synth.py`, `ifc_prepare.py` | Open |
| 11b | Construction-drawing benchmarks: FloorPlanCAD sample in the evaluation harness; US federal drawings and FloorPlanCAD blocks in the curated set | `harness.py`, `data/curated/` | Open |
| 12 | Metrics for layout (mAP per class, HD95 of drawing boundaries, leakage) and scale (error per input class) | pipeline.md §8 | Done in the design |
| 13 | Index the ★★ papers of §8 | research/papers.json (81 entries; new strategies "Sheet layout and title blocks", "Scale and dimension reading", "Layer and entity classification") | Done |

## 8. Sources

Indexed papers are cited by name; their notes and local copies are in the [research index](../../research/README.md). New papers and standards found for this review (verification and licences in the evidence brief; ★★ = shapes stages 0–2 or 9):

**Sheet layout and title blocks**
- ★★ Huang et al. 2026, *Benchmarking deep learning approaches for AEC engineering drawing layout detection and information extraction* (FELD), EC3 2026, [arXiv:2607.18997](https://arxiv.org/abs/2607.18997)
- ★★ Lombardi et al. 2025, *Title block detection and information extraction for enhanced building drawings search*, EC3 2025, [arXiv:2504.08645](https://arxiv.org/abs/2504.08645)
- ★★ Peng et al. 2026, *Open-source VLMs for engineering drawing metadata: QWEN-VL title-block detection and information extraction*, EC3 2026, [doi:10.35490/EC3.2026.252](https://doi.org/10.35490/EC3.2026.252)
- ★★ Peng et al. 2025, *AI-based extraction and management of text and view information from 2D bridge engineering drawings*, EG-ICE 2025, [doi:10.17868/strath.00093312](https://doi.org/10.17868/strath.00093312)
- Carrara, Nousias, Borrmann 2025, *Content-based classification of construction drawings*, EG-ICE 2025, [doi:10.17868/strath.00093309](https://doi.org/10.17868/strath.00093309)
- ★★ Chazalon et al. 2021, *ICDAR 2021 competition on historical map segmentation*, [arXiv:2105.13265](https://arxiv.org/abs/2105.13265)

**Scale and dimensions**
- ★★ Faltin, Schönfelder, König 2024, *Towards a robust deep learning-based scale inference approach in construction drawings*, ASCE Computing in Civil Engineering, [doi:10.1061/9780784485248.087](https://doi.org/10.1061/9780784485248.087)
- ★★ Chang et al. 2025, *Raster image-based house-type recognition and three-dimensional reconstruction technology*, Buildings 15(7), [doi:10.3390/buildings15071178](https://doi.org/10.3390/buildings15071178)
- ★★ Lin & Wang 2026, *A hybrid deep learning and rule-based method for architectural drawing vectorization and CAD reconstruction*, Buildings 16(5), [doi:10.3390/buildings16051043](https://doi.org/10.3390/buildings16051043)
- ★★ Liang, Fukuda, Yabuki 2026, *Automated dimension-aware 2D-to-BIM reconstruction through cross-modal text-geometry alignment*, EC3 2026
- ★★ Schwenker et al. 2023, *EXSCLAIM!*, Patterns 4(11), [doi:10.1016/j.patter.2023.100843](https://doi.org/10.1016/j.patter.2023.100843)
- Degen et al. 2026, *Plan2Map*, [arXiv:2606.02747](https://arxiv.org/abs/2606.02747)
- Aliyev & Barsi 2026, *Evaluating vision-language models for zero-shot room area estimation in floor plan images*, [doi:10.3311/PPci.44138](https://doi.org/10.3311/PPci.44138)

**Inputs and CAD layers**
- ★★ Yin et al. 2020, *Automatic layer classification method-based elevation recognition in architectural drawings*, Automation in Construction 113, [doi:10.1016/j.autcon.2020.103082](https://doi.org/10.1016/j.autcon.2020.103082)
- Chen et al. 2026, *Evidence-gated multimodal parsing and vectorization of architectural floor plans* (SALI-FP), [arXiv:2609.25615](https://arxiv.org/abs/2609.25615)
- Not to be cited: Tang et al. 2017, *Automatic structural scene digitalization*, PLOS ONE, retracted on 28 May 2025.

**Other sheet features**
- ★★ Xu et al. 2026, *BlueprintAgent*, Findings of EMNLP 2026, [arXiv:2609.07362](https://arxiv.org/abs/2609.07362)

**Standards, guidelines and tools**
- SIA 400:2000, *Planbearbeitung im Hochbau* (title block, scales, graphic scale, labelling of reductions)
- CADexchange, *CAD-Basisrichtlinie 4.3* (October 2024), client version Kanton Basel-Stadt (title block contents, cut edge, reference points)
- ISO 5455 (scales), ISO 5457 (sheet layout), ISO 7200 (title blocks), ISO 129 (dimensioning), ISO 32000-1 §12.9 (PDF measurement)
- [plan-check rules for the CAD-Richtlinie BBL V1.0](https://github.com/bbl-dres/plan-check/blob/main/docs/pruefregeln-de.md)
- EFD / BBL / KBB, *Beschaffung von Software und Art. 9 EMBAG* (11.06.2025)
- Autodesk AutoCAD help on layouts, viewports, VPLAYER, xrefs, plot styles and PDF export; ezdxf documentation; ODA, LibreDWG, QCAD and Artifex licence pages
