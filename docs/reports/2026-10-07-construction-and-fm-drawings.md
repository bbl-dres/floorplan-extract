# Construction Drawings and FM Floor Plans: Evidence, Data and Renderer v3

*Status: report as delivered on 7 October 2026; its conclusions are summarised in the [pipeline review §6](../reviews/2026-10-07-pipeline-target-inputs-masking-scale.md#6-drawing-types-construction-drawings-and-fm-plans). The downloaded drawings are in `data/benchmark/construction-plans/` and the FloorPlanCAD sample in `data/benchmark/floorplancad/sample/` (both local, gitignored).*

*Research report · 7 October 2026 · Answers BBL's question whether construction drawings need their own data, and what Swiss construction and facility-management (FM) plans contain. Written for the project team.*

**Scope and method**
- Read first: the three dataset catalogues, the literature review (§2, §5.1, §7), the pilot v2 review (§2.2, §4.2, §6), the curated manifest, `synth.py` and `ifc_prepare.py`.
- Paper figures come from the Markdown copies in `research/papers-md/`.
- Three parallel research threads were run:
  - public datasets (about 120 candidate sources);
  - Swiss plan conventions (SIA 400 and 13 owners' CAD/CAFM guidelines);
  - the two DWGs in bbl-dres/plan-check, parsed locally.
- Downloads and trials were done directly.

**Limits**
- The shared web-search budget ran out during the dataset and Swiss threads; later checks used direct page reads.
- Roboflow, loc.gov, ScienceDirect and MDPI returned HTTP 403. Their entries rest on search snippets and are marked "(S)".

- SIA 400's German text was cross-checked on an unofficial mirror of unclear copyright status. Use a purchased copy for anything normative.

---

## Summary

1. **Yes, it makes a difference, and the evidence points the same way everywhere.** Models trained on simplified or residential plans lose a large share of their accuracy on construction-style and non-residential drawings:
   - FloorPlanCAD → large office/campus CAD (LS-CAD): PQ 83.3 → 33.2 (SymPoint) and 87.4 → 52.9 (CADSpotting).
   - Same methods trained in-domain on mostly non-residential ArchCAD-400K: 83.3 → 47.6 and 86.2 → 70.6 (DPSS).
   - Our pilot on CubiCasa5K's architect-style subset: wall IoU 0.53 vs 0.74 on high_quality.

   The drop sits mostly in *recognition* (RQ), not in segmentation quality. Clutter that looks like walls is the main named cause: hatching outside walls, dimension and axis lines, text, leader lines. A larger scale and the building type (cores, column grids, open plan, many small rooms) add to it.

   FM plans are a third style with its own traps:
   - solid grey walls next to grey usage fills;
   - magenta polygon lines on wall faces;
   - boxed room stamps;
   - legacy plans that keep axes, dimension chains and raised-floor grids.

   No paper tests FM plans. Pooled-style training and synthetic data do close much of the gap (Dodge: 76–82 → 90–93 mean IoU; CADSpotting joint training: 75.5 PQ).
2. **No public dataset of construction or FM plans is licence-clean for training.**
   - The most construction-like sets cannot be used for training:
     - ArchCAD-400K: CC BY-NC, gated, has an "Axis & Grid" class;
     - FloorPlanCAD v2: CC BY-NC. Our inspection of the release found it keeps CAD layers, dimension chains (90% of test blocks), axes (60%), hatches and tags. A 30-block construction-style sample with six-class masks is now in `data/benchmark/floorplancad/sample/` (§2.4);
     - Korea's AI Hub architectural drawing set: 48,033 drawings with OCR text, open only to Korean nationals;
     - LS-CAD: still unreleased.
   - No public CAFM dataset with room stamps exists.
   - Permissively licensed routes:
     - US public-domain drawing archives (HABS/HAER/HALS, NARA, NPS eTIC, USACE standard designs): real sheets, no labels;
     - rendering from IFC and Swiss Dwellings;
     - a few tiny CC BY / Apache sets.
3. **Fifteen licence-clean real construction drawings were downloaded** to `data/benchmark/construction-plans/`, all US federal public domain:
   - six vector USACE sheets: an office HQ with area stamps, enlarged plans with dimension chains and door tags, a life-safety plan;
   - four NARA working drawings, 1888–1911;
   - five HABS sheets, including a blueprint, a construction-stage overlay and a 2011 CAD measured drawing.

   Nine are proposed for the curated set (§6).
4. **What Swiss plans contain is now documented from primary sources.**
   - **SIA 400:2000 is still current.** Werkplan 1:50 content: 3–4 exterior dimension chains, opening sizes as width over height, level boxes, section flags, material hatches per Fig. 34, existing/new/demolition in black/red/yellow.
   - **SIA 2014 has been replaced by SIA 4006:2025.**
   - **The BBL CAD-Richtlinie V1.0 (01.01.2026) is public.** It allows 15 layers:
     - grey SOLID walls on `A_SCHRAFFUR`;
     - magenta `R_RAUMPOLYGON`;
     - an AOID text per room;
     - axes;
     - only two dimension lines;
     - no furniture.
   - **A real 2010 BBL Revisionsplan in plan-check looks very different.** It has boxed 9-attribute stamps, door-attribute blocks, 74 dimensions, axes with façade ticks, section lines, ANSI hatches over SOLID fills, and a raised-floor grid.
   - The renderer gap list is in §4.5.
5. **IFC → construction-style drawings is feasible, but the annotation layer has to be our own code.**
   - IfcOpenShell (LGPL-3.0, already installed, v0.9.0) cuts storeys into clean SVG plans with door arcs, stairs and room stamps (name, number, m²). A trial on `wbdg_office` worked once CSS was inlined.
   - It draws no dimension chains, axes, material hatches or tags.
   - Bonsai (GPL-3.0, a Blender add-on) can draw all of these, but interactively and per drawing, and its docs mark the drafting module "early development". It is not a batch generator.
   - All 17 permissive models carry wall materials and door tags, 16 have named spaces, and 3 have IfcGrid axes. This is enough to drive dimension chains, material hatches, door tags and grids from geometry in renderer v3.

---

## 1. Does It Make a Difference?

### 1.1 Evidence

| Source | Setting | Result | What it shows |
|---|---|---|---|
| CADSpotting (Yang et al., Table 8) | Train FloorPlanCAD (Chinese CAD, 10 m blocks) → test LS-CAD (45 office, campus and hotel plans ≥ 1,000 m²) | SymPoint PQ 83.3 → **33.2** (RQ 91.1 → 39.9; SQ 91.4 → 83.0). CADSpotting 87.4 → **52.9** (RQ 58.5); with sliding windows 60.1. Joint training: SymPoint 57.0, CADSpotting+SWA **75.5** | Large non-residential CAD: instances are *missed* (RQ collapses) while matched ones stay well segmented. Training on in-domain sheets recovers most of the loss |
| ArchCAD-400K (Luo et al.) | Same methods trained and tested in-domain: FloorPlanCAD vs ArchCAD (86% non-residential, axis grid, columns, beams) | SymPoint 83.3 vs **47.6**; SymPoint-V2 83.2 vs **60.5**; CADTransformer 68.9 vs 60.0; DPSS 86.2 vs **70.6**. Stuff PQ without layers: SymPoint-V2 49.3 on FloorPlanCAD | Construction-like CAD is intrinsically harder, even with in-domain training. Wall-type ("stuff") classes suffer most without layer information |
| WAFFLE (Ganon et al., Table 4) | CubiCasa5K model → multi-style internet plans (incl. public and historical buildings) | IoU walls **0.488** (P 0.737, R 0.590), doors **0.099**, windows 0.202, interior 0.461 | Style shift hits openings hardest |
| Pilot v2 on CubiCasa5K (our review §2.2) | Style-randomised Swiss Dwellings renders → 400 CubiCasa test plans, zero-shot | Walls/doors/windows IoU: colorful 0.69/0.59/0.70, high_quality **0.74**/0.63/0.74, high_quality_architectural **0.53**/0.46/0.52. Window recall 0.67 vs 0.85; rooms correct 0.66 vs 0.84 | Architect-style sheets are hardest. Main false positives: hatched roofs and terraces read as wall, single-line knee walls, double-line windows |
| Pilot v2, Landgut Lohn | Same model → three real sheets | Wall IoU 0.87–0.89 (CAD print), 0.58 (poché scan), 0.29 (survey scan with ornament) | Clutter inside rooms is the failure mode |
| Dodge et al. 2017 | FCN walls, CVC → R-FP and R-FP → CVC | Mean IoU 76.1 / 81.7; pooled training **90.5 / 92.9** | Mixing styles closes the gap |
| Raster2Seq (arXiv 2602.09016) (V) | CubiCasa-trained → WAFFLE, interior IoU | CubiCasa model 46.1; FRI-Net 56.7; RoomFormer 60.5; Raster2Seq 73.9 (88.7 room F1 in-domain) | Architecture choice matters, but the shift remains |
| Zhang 2026 (arXiv 2608.25608) (V) | CubiCasa + synthetic → ResPlan renders, wall F1 at tol 0.05 | Zero-shot 0.640 / 0.621; fine-tuned **0.968 / 0.915** | A small amount of in-domain data removes most of the loss |
| MitUNet (arXiv 2512.02413) (V) | 500 CIS real-estate plans | From scratch 87.91 mIoU; CubiCasa pretraining 88.53 (+0.6). Authors: "partition walls feature complex internal hatching … frequently misclassif[ied]"; dimension lines and text named as noise | Residential pretraining transfers little. Hatching inside walls is a failure mode |
| FGSSNet (arXiv 2507.10343) (V), counter-evidence | CubiCasa-trained → 10 industry plans | Wall IoU 78–80% vs 77–78% in-domain | Small, residential-like sample. A shift is not inevitable when styles are close |
| Aliyev & Barsi 2026 (Periodica Polytechnica) (V) | Colour-coded room plans (closest to FM plans) | VLM area estimation MAPE 41–44%; colour pixel counting **5.68%** per room | Use deterministic colour/vector processing for FM fills, not VLMs |
| AECV-Bench, DrawingVQA, AEC-Bench (V) | VLMs on plans and construction sheets | Door/window counts 0.28–0.39 exact; DrawingVQA 71.7% vs experts 94.9%; AEC-Bench callout accuracy at best 42.9% | VLMs read text but fail on symbols, leaders and geometry |

### 1.2 What Differs, and How Much It Matters

| Property | Simplified / residential (CubiCasa, Swiss Dwellings renders) | Construction drawings (Werkplan, USACE sheets) | FM / CAFM plans (BBL V1.0, legacy BBL, cantons) |
|---|---|---|---|
| Walls | Solid, outlined or hatched; one style per sheet | Material hatches per wall type (SIA 400 Fig. 34), insulation bands, partitions as double lines, colour-coded hatches; renovation colours black/red/yellow | Grey SOLID (ACI 8/252) or black; partitions sometimes unfilled; legacy: SOLID plus ANSI31/37 line hatch on top |
| Lines crossing rooms | Few | Axes, interior dimension chains, section lines, overhead dash-dot edges, leaders | Axes (often), façade-grid ticks (legacy), magenta polygon lines on wall faces, sometimes section lines (legacy) |
| Text | Room names, sometimes areas | Dimension figures everywhere, door and wall tags, level boxes, notes, room stamps with finishes | AOID or boxed stamps (number, name, area, height, B/W/D finishes), door numbers (legacy) |
| Fills outside walls | Rare | Floor finishes, terrain, stair treads, legend swatches | Usage colour fills (Kanton ZH ramps, UZH pastels), escape-route hatches (Stadt ZH), raised-floor grids |
| Sheet | One apartment | Whole floor at 1:50/1:100, title block, key plan, matchlines, several drawings | Whole floor at 1:100/1:200, title block with site overview map (BBL) |
| Building type | Residential | Offices, schools, hospitals: cores, column grids, open plan, many doors | Same, as built |

**Verdict.** Construction drawings and FM plans are both out of the distribution of CubiCasa-like data, in different ways.
- **Pilot v2 already covers part of the gap:** dimension chains (random, not geometry-bound), axes, material hatching, non-wall hatching, title blocks, colour zones and stamps. The CAD-print result (wall IoU 0.87) shows randomisation transfers.
- **What remains is mostly the annotation layer bound to geometry, and FM-specific overlays** (§4.5).
- **A real test set is required before any claim about either style.** Use the 15 USACE/NARA/HABS sheets here, a FloorPlanCAD and ArchCAD sample (evaluation only), and above all 10–20 BBL construction PDFs and FM DWGs (review §6.2).

---

## 2. Public Datasets

Licence marks:
- **(V):** read on the producer's page or API.
- **(S):** search snippet only.
- **(P):** taken from the paper.

Training means production weights; evaluation means internal benchmarking only.

### 2.1 New Datasets Found (not in `docs/datasets-open.md`)

**Construction drawings, working drawings and engineering sheets**

| Dataset | Content and labels | Size | Format | Licence (as stated) | Access | Type / region | Style | Verdict |
|---|---|---|---|---|---|---|---|---|
| [AI Hub 건축 도면 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71465) | Plans, sections, elevations, structural sheets; 8 structure, 12 space and 5 fixture classes; 304,462 OCR text boxes; boxes and segmentation | 48,033 images, about 25 GB | PNG + JSON | No open licence: "내국인만 데이터 신청이 가능합니다" (Korean nationals only) (V) | Gated | Housing; Korea | Highly construction-like | Not usable |
| [AEC-Bench](https://github.com/Nomic-ai/aec-bench) (catalogued; drawings now checked) | 196 agent tasks on real multi-discipline drawing packages (grids, callouts, title blocks) | 196 tasks | PDF via nomic-public-data.com | Repo "Apache License, Version 2.0" (V). Drawings "sourced from publicly available PDF documents on the web" (P); their own rights are not addressed | Direct | Public-sector projects | Full construction sets | Evaluation only |
| [DrawingVQA](https://huggingface.co/datasets/S2-MIND/DrawingVQA) | 92 expert QA pairs on 33 "Issued for Construction" structural sheets from 6 educational buildings | 33 sheets | Parquet | Questions "CC BY-NC-SA 4.0"; "the original drawing images are not fully open to the public" (V) | QA open | Schools; USA | Construction (structural) | Evaluation (QA only) |
| [masterpn RC series](https://www.kaggle.com/datasets/masterpn/rc-beams-dataset-v1-25-reinforced-concrete-beam) (Kaggle) | Real reinforcement-detailing sheets (beams, columns, walls, foundations), no labels | about 175 | PNG 1200 dpi + PDF | "CC-BY-NC-SA-4.0" (V) | Direct | European engineering office | Construction (structural) | Evaluation; ask the author for a licence |
| [ArchSIBench](https://huggingface.co/datasets/ArchSIBench/ArchSIBench) | 3,000 QA on plans, sections and photos from ArchDaily, Goood and Archiposition | 2.27 GB | Parquet | HF tag "apache-2.0" (V); the images are scraped (relabel) | Direct | Mixed | Presentation drawings | Evaluation only |
| Handoff-H1 ([arXiv 2608.15032](https://arxiv.org/abs/2608.15032)) | 10 residential blueprint sets with 2,009 expert take-off items | 10 sets | PDF | "available upon request for research use" (V) | Request | Residential; USA | Construction | Evaluation only |
| FELD ([arXiv 2607.18997](https://arxiv.org/abs/2607.18997)); title-block set (Lombardi et al., EC3 2025) | Sheet layout: title block, revision table, legend, notes, sections | 551 sheets; n/a | Raster | No release found | – | Facades; UK | Construction sheets | Ask the authors |
| [DELP / SkeySpot](https://github.com/HAIx-Lab/Skeyspot) | 45 scanned electrical layouts, 34 symbol classes | 45 | Scans | GitHub: "available for academic research upon request" (V); lab page: CC BY-NC-ND 4.0 (S) | Request | Housing; UK | MEP construction | Evaluation only |
| [devp1866 blueprints + BOM](https://www.kaggle.com/datasets/devp1866/engineering-blueprints-and-bill-of-materials-dataset) | Synthetic 11 × 17 sheets: title blocks, revision tables, notes, BOM; JSON | 500 | JPG + JSON | "CC-BY-SA-4.0" (V) | Direct | Synthetic | Sheet furniture only | Training (title-block OCR; share-alike) |
| [cadsy/cad-technical-drawings](https://huggingface.co/datasets/cadsy/cad-technical-drawings) | Synthetic mechanical sheets with labelled dimensions, callouts, title-block values | 130 | PNG + JSON | "Apache License 2.0" (V) | Direct | Mechanical | Dimension and title-block patterns | Training (auxiliary) |
| [plannotation](https://github.com/plannotation/plannotation) (tool) | IFC-aligned annotation layer inside PDF sheets: dimensions, grids, levels, tags | 3 + 1 sheets | PDF | Code "Apache-2.0"; sample "© Esplan OÜ, CC BY 4.0" (V) | Direct | Estonia | Construction | Route for synthetic labelled sheets |
| [Vandœuvre-lès-Nancy media library](https://www.data.gouv.fr/datasets/5ce7e5a78b4c415bdc600f33) | Interior plans of a municipal building | 7 DXF + 7 PDF | DXF / PDF | "Open Data Commons Open Database License (ODbL)" (V) | Direct | Public building; France | Real CAD (not inspected) | Tiny; real DXF test case (share-alike on databases) |
| [NPS eTIC / NPGallery](https://npgallery.nps.gov/etic) | NPS planning, design and construction drawings | >32,000 records (V) | JPG / PDF | "Digital assets without any copyright restrictions are public domain" (V); per asset | Direct | Visitor centres, admin buildings; USA | Modern working sheets | Training per PD-flagged asset (own labels) |
| HABS/HAER/HALS (Library of Congress; Commons mirror) | Measured drawings plus photocopies of original federal working drawings | >581,000 items; about 2,700 floor-plan sheets on Commons | TIFF / PNG / JPEG | NPS: "Materials created for HABS, HAER, or HALS are in the public domain." (V) | Commons API | Public, institutional, historic; USA | Construction-like (hand and CAD) | Training (own labels); style coverage |
| USACE MRSI library ([mrsi.erdc.dren.mil](https://mrsi.erdc.dren.mil/model-rfp/)) | Army standard designs: HQ, fire station, community service centre, company operations facility, etc.; vector D-size sheets | Dozens of PDFs | Vector PDF | Works of the U.S. Government (§105); no contractor named on the sheets checked (§3) | Direct | Offices, admin, training; USA | Modern CAD/Revit construction sheets | Training and evaluation (own labels) |

**FM, space-management, evacuation and indoor plans**

| Dataset | Content | Licence | Verdict |
|---|---|---|---|
| TUM German emergency plans (Hassaan et al., [Sensors 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10575354/)) | 403 photographed evacuation plans from German public buildings; 45 DIN/ISO 7010 symbol classes; about 5,000 synthetic | "available upon request from the corresponding authors" (V) | Evaluation, on request. Closest FM-like real set (DACH conventions) |
| [mazharrehan/floorplan](https://www.kaggle.com/datasets/mazharrehan/floorplan) (Kaggle) | 308 plans with rooms colour-coded by use, drawn in Visio by the author | "Attribution 4.0 International (CC BY 4.0)" (V) | Training (small, simplified) |
| University CAFM services (Toronto, CU Boulder, Hertfordshire CC) | Real CAFM plans with numbers, use codes, areas | "NOT FOR DISTRIBUTION" or similar (V) | Not usable |
| OSM indoor / TUM NavigaTUM import; IPIN maps (Zenodo, CC BY 4.0); Microsoft Indoor Location 2.0 (repo MIT) | Room polygons, floor images | ODbL / CC BY / MIT (V) | Render sources only (simplified) |

### 2.2 How Construction-Like the Catalogued Datasets Are

- **FloorPlanCAD.** Real Chinese CAD from "over 100 projects including residential buildings, schools, hospitals, and large shopping malls". Layers were split and cut into 10 m × 10 m blocks with only 30% kept. That removes title blocks, sheet context and long dimension chains.
  - **Text:** v1 "Texts are removed". The November 2021 release (15,663 drawings) "includes abundant textual annotations" (TextCAD, arXiv 2607.12678): door tags such as "FM B 1321", room tags, dimension text. The catalogue entry should say so.
  - **Classes:** 35, with no axis, dimension or hatch class. Unlabelled primitives are background.
  - **Verdict:** medium construction-likeness. Our own inspection of the test split is in §2.4.
- **ArchCAD-400K.** 5,538 drawings that "strictly conform to the layer-block organizational standards", cut into 14 m × 14 m chunks (413,062). Average drawing about 11,000 m²; 14% residential.
  - **Classes (HF card):** "Axis & Grid" (ID 0), four door types, stairs, elevator, sanitary, furniture, wall, glass, concrete and steel columns and beams, foundation, pile, rebar, fire hydrant, parking, "Others". Structural sheets are therefore included.
  - **Not separate classes:** text was anonymised; dimensions, hatches and level marks have no class (probably "Others").
  - **Verdict:** the most construction-like public set.
- **LS-CAD.** 45 whole office, campus and hotel plans with FloorPlanCAD labels. The journal version (Computational Visual Media 12(4), 22 Sep 2026) still says "will be publicly released … under a license waiver". The code repo says "The full floorplan dataset is not bundled". Unreleased.
- **MLSTRUCT-FP.** 954 Chilean multi-unit residential plans with wall and slab polygons. README crops show walls only; whether the raw rasters keep text and dimensions could not be verified. Repo archived 29 Apr 2026; dataset terms still not stated.
- **CubiCasa5K.** From "real estate marketing material conversions"; subsets 3,732 / 992 / 276. high_quality_architectural is architect-looking marketing redraws. They have hatched or solid walls, room abbreviations, furniture, and sometimes areas, dimensions or a title block (curated `cc-hqa-366` is a building-permit sheet). Grid axes, level and section marks are absent. The least construction-like, apart from a minority of permit sheets.

### 2.3 ArchCAD-400K: What It Would Add, and What Access Involves

- **What it adds:**
  - the only large set with axis/grid, column (concrete/steel), beam, foundation and pile classes;
  - mostly offices, industrial parks and public buildings;
  - five aligned modalities (raster, SVG, JSON, Q&A, point cloud).

  It would give an evaluation set for exactly the confusions BBL fears: axes, columns and hatches next to walls. It is the natural test for a vector (DXF/PDF) path.
- **Access:**
  - gated on Hugging Face (`jackluoluo/ArchCAD`), licence CC BY-NC 4.0, GitHub badge "ACADEMIC";
  - the requester logs in with their own account and confirms "I confirm that this dataset will be used for non-commercial purposes only";
  - manual approval, "may take up to 3 business days";
  - only the first round (40K curated samples, 16 Oct 2025) is released; no later round as of October 2026.
- **Not requested.** The non-commercial declaration needs BBL's legal service first (review §6.4, question 3).

### 2.4 FloorPlanCAD: Local Copy and Construction-Style Sample

**Download.**
- Source: the three official archives from the project site (test-00, train-00, train-01), extracted into `data/benchmark/floorplancad/` (test, train-00, train-01).
- `SOURCE.md` (written by me) records origin, licence, version, checksums and the file list.
- **Contents:**
  - test: 5,502 blocks (SVG + PNG + coco_vis);
  - train: 9,958 blocks in a first combined extraction.

  That makes 15,460 drawings, 203 fewer than the stated 15,663. At 19:05 the coordinator was re-extracting train into `train-00/` and `train-01/`; re-check the count when that is done.

**What the release actually contains:**
- **CAD layers and text are kept.** The SVGs keep the original CAD layers as Inkscape groups (`WALL`, `PUB_DIM`, `AXIS`, `柱网`, `PUB_HATCH`, `COLUMN`, `DIM_ELEV`, `DOOR_FIRE` …) and the text:
  - dimension figures in mm;
  - door and window tags (`M1324`, `FM乙1623`, `C1527`);
  - level marks (`H-0.050`, `12.100`);
  - room names with areas (`办公 9.63`).
- **Test-split shares:**

  | Content | Blocks |
  |---|---|
  | Dimension layers | 90% |
  | Axis layers | 60% |
  | Hatch layers | 56% (incl. floor-tile patterns inside rooms) |
  | Column layers | 51% |
  | Text | 96% |
  | Labelled furniture | 32% |

  This is far more construction-like than the paper suggests ("Texts are removed" refers to v1). Only sheet context is lost to the 10 m crops.
- **Scale.** 10 SVG units per metre: the viewBox 0 0 100 100 is a 10 m block. Checked against dimension text and door sizes (median single door 8.8 units, double door 12.0, window depth 2.0). About 90% of drawings fit; 18 test drawings use another unit, so check the scale per drawing.
- **Class ids differ from the papers.** The release's `semantic-id` order differs from the papers' tables and from SymPoint's class list, which reads a converted `semanticId` format:
  - 1 wall, 2 curtain wall;
  - 3–8 doors, 9–12 windows and opening symbol;
  - 13–29 furniture, appliances and sanitary;
  - 30 stairs, 31 elevator, 32 escalator, 33 railing, 34 row chairs, 35 parking.

  Verified against `coco_vis` and the CAD layer names. **Columns carry no label**; the `COLUMN` layers are background.

**Sample** (`data/benchmark/floorplancad/sample/`, 30 blocks, 4 MB, `manifest.json`):
- **Composition.** Six blocks each of school, hospital, office, mall and residential. The type is inferred from Chinese room names in each drawing's text and stated in the manifest.
- **Selection.** One block per drawing, de-duplicated (maximum image correlation 0.25), only drawings at 10 units/m.
- **Per block, three files:**
  - **Print-style render** at 100 px/m (1000 × 1000 px), black on white, everything kept.
  - **Filled six-class mask:**
    - wall: closing of the wall lines with 0.45 m;
    - window: closing with 0.30 m;
    - door and stairs: convex hull per instance (doors include the swing);
    - column: from the CAD column layers (layer-derived).
  - **Line-level mask:** the exact annotation, 3 px wide.
- **Mapping:**
  - wall ← 1;
  - window ← 2, 9, 10, 11;
  - door ← 3–8;
  - stairs ← 30, 32;
  - everything else background.
- **Content.** All 30 blocks have dimension chains and text, 23 axes, 24 hatches (incl. classroom and toilet floor patterns), 20 column layers, 28 furniture.
- **Caveats:**
  - The filled walls are an approximation: thin partitions that FloorPlanCAD leaves unlabelled stay background.
  - The door hulls include the swing, unlike our renderer's door-in-wall label, so score doors as instances.

**Recommended use.** Add the sample as a benchmark loader in `bench.py` (scale known: 100 px/m). It is the best available test of exactly the clutter the user describes: axes, dimensions, tags, floor hatches and furniture, in real CAD drawings of schools, hospitals, offices and malls.

---

## 3. Licence-Clean Real Construction Drawings (Downloaded)

**Location.** `data/benchmark/construction-plans/`:
- `SOURCE.md`: origin, licence basis per item, download date 7 Oct 2026, scales;
- `attribution.csv`: same columns as `commons-plans`;
- `pdf/`: two USACE PDFs;
- `images/`: cp01–cp06 rendered at 200 dpi, cp07–cp15 as 3000 px Commons renditions.

Total 28 MB. Commons was fetched at least 30 s apart.

| File | What it shows | Licence basis | Construction content |
|---|---|---|---|
| cp01.png | USACE ACSC "Electronic Security System – Large Size Facility" (2011), reduced 11 × 8.5 in exhibit | §105; USACE Huntsville title block, no contractor | Furniture, security symbols + legend, room names; printed 1:96 scale invalid on the reduced sheet |
| cp02.png | ACSC A-102 **Life Safety Plan**, 3/32" = 1' (61.5 px/m) | §105 | Red egress routes, fire-rated partitions, extinguisher and exit symbols, code summary |
| cp03.png | ACSC A-103 **Enlarged Floor Plan** Area A, 3/16" = 1' (123 px/m) | §105 | 71 dimension strings, door tags, section markers, steel columns, matchlines, key plan, cubicles |
| cp04.png | ACSC A-106 Enlarged Floor Plan Area D, 3/16" = 1' | §105 | Room name and number boxes, door tags, dimension chains, key plan |
| cp05.png | Brigade HQ **Small Brigade First Floor Plan** (Rev 7.0, Jun 2026), 1/8" = 1' (82 px/m) | §105; USACE Savannah District; path "OneDrive – US Army Corps of Engineers" | Office with **name + area stamps**, workstations, cores, overall dimensions, sheet grid, area table |
| cp06.png | Consolidated HQ **Second Floor Plan**, 1/16" = 1' (41 px/m) | §105 | About 100 rooms, open-office fields, tiny stamps |
| cp07.jpg | Executive Mansion Juneau, First Floor Plan, 1911 (Supervising Architect) | NARA RG 121, `{{PD-USGov}}` | Ink working drawing: dimensions, notes, interior elevations on the sheet |
| cp08.jpg | Post Office and Custom House El Paso, 1888 | NARA RG 121, `{{PD-USGov}}` | Poché walls, room sizes, approval signatures |
| cp09.jpg | Post Office Muskegon, 1904: plans on the site sheet | NARA RG 121, `{{PD-USGov}}` | Small plans on a coloured site plan |
| cp10.jpg | U.S. Naval Hospital San Diego, Contagious Ward, Drawing 101843 | HABS (Navy Bureau of Yards and Docks), `{{PD-USGov-NPS}}` | **Blueprint, inverted polarity**, dense dimensions |
| cp11.jpg | Old Post Office Building, Washington DC, **Sixth Floor Plan** | HABS (NPS drawing; NARA 121-BA-9373F) | Office floor, room numbers, scale bar, photo border |
| cp12.jpg | Post Office and Court House extension Portland, Drawing 61, 1904 | HABS (Supervising Architect) | Column grid, notes, details, photographed sheet |
| cp13.jpg | Letterman Army Hospital Building 1006, Main Laboratory | HABS (Army Post Engineer) | Laboratory wing; **low contrast** |
| cp14.jpg | NHDVS Mountain Branch **Administration Building**, HABS 2011 CAD measured drawing | HABS, `{{PD-USGov-NPS}}` | Dimension chains, section markers, room names with sizes, floor-pattern hatch, bilingual scale bars |
| cp15.jpg | Stamford Post Office, First Floor Plan 1939, with **1940–41 construction stages overlaid** | HABS (federal drawing photocopy) | Stage numbers, temporary partitions, X-crossings, notes |

**Also delivered (evaluation only, CC BY-NC 4.0):** the FloorPlanCAD construction-style sample, `data/benchmark/floorplancad/sample/`. It has 30 blocks of 10 m × 10 m with 100 px/m renders, filled and line-level six-class masks, and a manifest (§2.4). Distribution: school 6, hospital 6, office 6, mall 6, residential 6.

**Excluded.**
- NARA "Ground/Second Floor Plan" 1986 (proposed Reagan Library). Tagged PD-USGov, but probably by private architects.
- Pioneer Courthouse restoration sheets, 1971: authorship unclear.
- Alcatraz hospital 1940: the file is a details sheet, not a plan.
- Air Force CES standard design (AFCEC, 2018). Its colour-coded composite office plan is a useful FM analogue, but only a 650 px raster, and the drawings PDF was unreachable on WBDG.

**More of the same.** The best follow-up sources:
- the USACE MRSI library (fire stations, DES, company operations facilities, C2F Army HQ; vector);
- NPS eTIC public-domain assets (modern working drawings);
- Commons HABS categories (about 2,700 floor-plan sheets);
- NARA series 305855 on Commons.

None are Swiss. No Swiss or EU construction drawing with a clear open licence was found: every Swiss owner reserves rights in its plans (`datasets-open-docu.md`).

---

## 4. What Swiss Construction Drawings and FM Plans Contain

### 4.1 Standards and Plan Types (SIA)

- **SIA 400:2000 "Planbearbeitung im Hochbau"** is still current: revision 1 (2015), no successor.
- **SIA 2014 (CAD layers) is replaced by SIA 4006:2025** (valid from 1 Aug 2025; ISO 13567 based, EKG→eBKP translation). Cantonal guidelines (BS, ZG, SO, BL) still cite SIA 2014 EKG codes.

| Phase (SIA 112) | Scale | Typical floor-plan content |
|---|---|---|
| Vorprojekt (31) | 1:200 | Solid black walls, room organisation, few dimensions |
| Bauprojekt (32) | 1:100 | North arrow, room numbers and uses, fixtures; overall dimensions and wall thicknesses; datum "±0.00 = 423.82 m ü. M."; door widths; section flags. Example stamp "19 SÉJOUR / surf. S 29.3 m²" |
| Baugesuch (33) | 1:100 (ZH); 1:100 or 1:50 (BE) | As Bauprojekt; renovations existing **black**, new **red**, demolition **yellow** (SIA 400 B.8.11; ZH BVV §4) |
| Werkplan / Ausführungsplan (51/52) | 1:50 | Hatched cut walls incl. insulation; **3–4 exterior dimension chains** (overall, axes, rough openings, structure) plus interior chains; opening size **width over height** ("1.05" / "2.22"); BR/ST levels; room stamp with **level box "±0.00 / −0.16"** and finishes B/W/D; section and detail flags; axes; columns; cross-references |
| Schlitz- und Durchbruchpläne | 1:50 | Openings and slots with codes (WD, DD, BD, FD, WS, DS); yellow = horizontal, brown = vertical |
| Revisionspläne (53) | 1:100 / 1:50 | As built; Basel-Stadt: everything black, demolished elements deleted |
| Bestandes- / Bewirtschaftungs- / CAFM-Pläne | 1:100 (BBL, Stadt ZH), 1:200 print (Bern ISB), 1:50 (SG) | Defined by owners' guidelines (§4.2), not by SIA 400 |

**Drawing conventions (SIA 400):**
- **Hatches (Fig. 34):**
  - masonry: brick "/", sand-lime "\" wide, concrete block "\" in pairs;
  - concrete: X cross-hatch;
  - insulation: perpendicular lines or zig-zag;
  - steel: solid black;
  - artificial stone: grid.

  Colours are optional: brick red, concrete green, insulation pink.
- **Simplification:** at ≤ 1:100 uniform black walls are allowed.
- **Lines:** at most three weights (0.25/0.50/1.00 or 0.18/0.35/0.70 mm); dash-dot for axes (thin), overhead edges (medium) and section lines (thick).
- **Dimension figures:** millimetres often as superscripts ("2.96⁵"); Basel-Stadt forbids superscripts.
- **Not standardised:** a stair notation "n × Steigung/Auftritt" and a "90/210" door slash form are not in SIA 400. Legacy BBL door blocks use "85x210".

### 4.2 FM / CAFM Guidelines

| Owner | Room polygon | Stamp | Walls | Removed for FM |
|---|---|---|---|---|
| **BBL CAD-Richtlinie V1.0** (01.01.2026, public ZIP on bbl.admin.ch) | `R_RAUMPOLYGON` ACI 210 magenta, closed, no arcs, > 0.25 m²; `R_RAUMPOLYGON-ABZUG` 230; `R_GESCHOSSPOLYGON` 214 | Planner supplies only the AOID text (`WWWW.GG.EE.RRR`, e.g. `2011.DM.04.045`) on `R_AOID` at the room centre. The checking body generates `R_RAUMSTEMPEL`: name, Türschild-Nr, FB, RB, FL, AOID, "SIA: HNF 2.9", B/W/D | SOLID on `A_SCHRAFFUR` (ACI 8 = 128,128,128) for massive walls and columns | Only 15 layers: no furniture, finishes or general text; **one** overall length and width dimension; existing stamps and leaders removed; axes allowed (`V_ACHSEN`) |
| **BBL legacy** (2010 Revisionsplan in plan-check) | Polygons as blocks named AOID + usage letter (H/N/V/F) on `A1Z21---E-` | `bbl-raumstempel` boxed table, 9 attributes (FLAECHE, RAUMBEZ, RNR1, RNR2, FB, RB, B, W, D); door blocks with 14 attributes (Tuernummer "04.034-T1", Brandschutz "EI30", Groesse "85x210") | 303 SOLID fills **plus** ANSI31/37 line hatches | Nothing removed: axes with façade ticks, 74 dimensions, section and detail references, raised-floor grid, notes |
| CADexchange CAFM-Basisrichtlinie 2016 (+ V4 clauses in SO, BL, ZH) | `R_00104_R1_NGF-POLYGON` (210), GF-polygon (214), fill layer | Number, physical number, name, area; optional DIN 277 usage; "Ein Raumstempel pro NGF-Polygon" | "nur einfache Linienschraffuren oder Vollflächenfüllungen" | Furniture on its own layer |
| Basel-Stadt 4.3 (Dec 2024) | `A1ZD22_RAUMPOLYGON` (from 1.50 m clear height), `A1ZD21_GESCHOSSPOLYGON` | Room number `A.EG00.309`, name, SIA 416 room and window areas, clear height, levels, B/W/D | Solids without border | IBS CAFM set drops furniture, dimensions, axes, text, finishes |
| Stadt Bern ISB (2019/2020) | `RAUMPOLYGON` (210) | `RAUMBEZ`, `RNR2`, `Rh`; FL, EBF, SIA 416/DIN 277 | SOLID at 1:100 | Not shown: incomplete hatches, furniture, services, safety symbols |
| Kanton Zürich (V4 2018; Flächennachweis V1.1 2025) | EKG/eBKP per CADexchange | Number, area; Musterplan stamp Nr / Raumbez / RB / FB / RD / FD / BF / D / W / B | Line hatches or solid | **Only owner with published RGB per SIA d0165 code:** HNF1 oranges, HNF2 reds, HNF3 yellows, HNF4 greens, HNF5 cyans, HNF6 magentas, NNF7 blues, FF8 greys, VF9 yellows |
| Stadt Zürich IMMO (RFB V18.0; CAFM 2026) | `Z122_NGF-POLYGON` etc., at floor level, columns included | `Z263_RAUMSTEMPEL-IMMO`: name, Nr. (`EG00807`), BF, RH, Flächentyp ("VF 9.1"), boxed | "mit breiten Linien umrandet und mit grauer SOLID Schraffur" (ACI 252) | Furniture, finishes, detail dimensions removed; escape layers `Z291`–`Z299` (corridor and stair hatches) |
| UZH (2024) | `V010412_V_Flaechenpolygone` **red** (255,0,0) | MTEXT "Y12 / Büro / H-235 / 23m2 / 3.80m" | Simple line hatches | Furniture simplified; sample fills HNF #FFA8A8, VF #FFFFA8, FF #A8C0FF, NNF #FFD6A8 |
| Aargau, Solothurn, Basel-Landschaft, Zug, St. Gallen, Luzern | Magenta 210 / 214 polygon layers, or polygon layers split by NF/VF/FF (SG) | 4–9 field stamps ("M02 / 1.03 / Name / BF: 69.70 m2 / FF / RH / B / W / D") | SOLID, partitions sometimes unfilled (AG, ZG) | SG: "ohne Detailinformationen, Materialschraffuren, Bemassung, Textblöcke" |

- **Cross-owner rules:**
  - Arial everywhere, printed at least 1.5–2 mm;
  - polygons on the inner wall face, straight segments only;
  - no movable furniture;
  - few or no dimensions.
- **Not public:** armasuisse, ETH, EPFL, hospitals, SBB and Post publish no CAD guideline. The CADexchange V4.0 base guideline is behind a name/e-mail form (not obtained).

### 4.3 plan-check

[bbl-dres/plan-check](https://github.com/bbl-dres/plan-check) implements 40 rules from CAD-Richtlinie V1.0, chapters 4–5:
- **Layers:** required `R_RAUMPOLYGON`, `R_AOID`, `R_GESCHOSSPOLYGON`; warning on layers outside the 15.
- **Polygons:** closed, no arcs, LWPOLYLINE, ≥ 3 vertices, area ≥ 0.25 m², no self-intersection.
- **AOID:** exactly one per room, unique, regex `^\d{4}\.[A-Za-z0-9]{1,4}\.\d{2}\.\d{3}$`.
- **Geometry:** units mm; no MLINE, ELLIPSE, SPLINE or OLE.
- **Text:** Arial, on allowed layers.
- **Dimensions:** on `V_BEMASSUNG`.
- **Hatches:** SOLID on `A_SCHRAFFUR`.

It does not check wall geometry, room names or usage, door numbers or finishes.

Both sample DWGs are evidence for the renderer:
- **The V1.0 template:** a fictional administrative building, 5.OG. 30 polygons, AOIDs `9999.AA.05.001…`, a 13-line axis grid with bubbles, 2 dimensions, and a title block with a **site overview map of neighbouring buildings**.
- **The legacy 2010 Revisionsplan:** §4.2. Its geometry is in metres although `$INSUNITS` says mm, a vector-path pitfall.

### 4.4 Checklists

Frequencies are estimates: always > 90%, often 50–90%, sometimes 10–50%.

**Werkplan 1:50** (most harmful for extraction in bold):
- **Always:**
  - cut walls with material hatches and insulation bands;
  - doors (leaf + arc);
  - windows with BR/ST notes;
  - **3–4 exterior dimension chains** and **interior chains across rooms**;
  - level marks;
  - room stamps with level boxes;
  - stairs with walking line;
  - **section lines with flags**;
  - title block and legend (hatch swatches look like walls).
- **Often:**
  - partitions as unfilled double lines;
  - columns;
  - **axes with bubbles crossing the plan**;
  - opening dimensions and door numbers inside swings;
  - overhead dash-dot edges;
  - Aussparungen;
  - sanitary fixtures and kitchens;
  - shafts with crosses.
- **Sometimes:**
  - **floor-finish hatches in rooms**;
  - furniture;
  - fire-protection labels;
  - **Bestand/Neu/Abbruch colours** (yellow = demolished, not a wall);
  - exterior hatches.

**CAFM Bestandesplan** (BBL-style):
- **Always:**
  - massive walls in grey or black SOLID;
  - doors, windows, stairs, lifts, shafts, sanitary fixtures;
  - an AOID or a full stamp;
  - room polygons (in the DWG, sometimes printed **magenta on the wall faces**);
  - title block with **site overview map**.
- **Often:**
  - axes;
  - overall dimensions only;
  - fixed built-ins;
  - unfilled partitions;
  - reference points.
- **Sometimes:**
  - **usage colour fills** (grey FF ramp ≈ wall grey; magenta HNF6 ≈ polygon colour);
  - **escape-route hatches** in corridors and stairwells;
  - legacy material hatches over SOLID;
  - **raised-floor grids**;
  - section lines;
  - door-attribute blocks;
  - stamp leaders.
- **Rarely:** movable furniture.

**Extraction risks, most to least harmful:**
1. Lines crossing rooms (axes, dimension chains, section lines, overhead edges).
2. Door arcs and gaps.
3. Hatches outside walls.
4. Boxed stamps and level boxes (small closed rectangles).
5. Inconsistent wall depiction (filled vs unfilled; SOLID + hatch; renovation colours).
6. Usage fills vs wall greys.
7. Polygon lines on wall faces.
8. Windows and façade ticks.
9. Title-block site maps and legends.
10. Vector unit and block pitfalls.

### 4.5 Renderer v3: Gaps Against `synth.py`

`synth.py` v2 already draws:
- wall styles incl. Swiss material hatches by thickness (concrete grey/black/dots, masonry, sand-lime, stone, drywall cross) and an insulation zig-zag;
- door and window variants incl. casements;
- stairs and fixtures;
- multilingual stamps with areas, numbers and an AOID-like code;
- **random** dimension chains (not tied to walls; metres, 2 decimals);
- **random** unlabelled dash-dot axes, plus labelled column-grid axes (15%);
- non-wall hatches (terraces, floor finishes, context);
- ornament, reveals, niches, pilasters;
- colour zones (fire compartments, 8%);
- room fills (pastels, by room type);
- rubber stamps, red-pen revisions with deltas, hand annotations with leaders;
- title blocks (8%), legends (6%), key plans (4%), a second drawing (7%);
- scan defects.

Everything is drawn on 512 px crops at about 50 px/m.

**Gaps for construction drawings (Werkplan / Ausführungsplan):**
1. **Geometry-bound dimension chains.**
   - 3–4 exterior chains parallel to each façade: overall, axes, openings, structure.
   - Interior chains through rooms and wall thicknesses.
   - Extension lines touching wall corners; oblique ticks or dots.
   - Swiss figure formats ("4.25", "2.96⁵", "12⁵"), plus feet-inch for US-style sheets.
   - Today's chains are random lines whose figures do not match the walls.
2. **Opening annotations.** Width over height at each opening ("1.05" / "2.22"), BR/ST notes ("BR +1.10"), door numbers ("T 1.05", "04.034-T1") in ovals or hexagons, sometimes inside swings, EI30 labels. Also wall-type tags (USACE W1–W5 hexagons).
3. **Level marks.** Triangles (open = finished, filled = rough) with "+3.25" / "−0.10", level boxes "±0.00 / −0.16" in room stamps, and a datum note "±0.00 = 556.60 m ü.M." in the title block.
4. **Section and detail markers.** Thick dash-dot section lines across the whole building with pentagon or circle flags ("A │ 105"), detail callouts with leaders, matchlines.
5. **Axes bound to the structure.** Axis grids through column centres and wall axes, bubbles at both ends (numbers one way, letters the other), intersection labels ("H4"), façade-grid ticks at 1.25 m. Today grids appear only with generated columns (15%), and the other axes are random.
6. **Werkplan room stamps.** Boxed number, name, BF/FF, clear height "RH 2.60", level box, B/W/D finishes ("B: Hartsteinholz / W: Gipsglattstrich / D: Heraklith"), in DE/FR/IT.
7. **Material hatches per SIA 400 Fig. 34, incl. colour-coded variants.** Brick red, concrete green cross-hatch, insulation pink, artificial-stone grid; partitions as unfilled double lines; multi-layer exterior walls (structure + insulation + cladding), so the insulation does not read as a separate wall.
8. **Renovation modes.** Existing black/grey, new red, demolition yellow with X marks, closed openings filled. In greyscale, yellow becomes light grey; label yellow as *not wall*.
9. **Overhead and hidden edges.** Dashed or dash-dot beams, slab edges, skylights and upper stair flights crossing rooms.
10. **Aussparungen and Durchbrüche.** Rectangles with a heavy L-edge and codes ("DD 30/20"), in yellow or brown.
11. **Office content.** Procedural workstation clusters, conference tables with chairs, lockers, shelving; server-room raised-floor grids. IFC furniture footprints exist but are drawn as clutter only. cp05, cp06 and cp03 are the targets.
12. **Overlay plans.** Life-safety (red egress routes, rated partitions, extinguisher and exit symbols; cp02), security and electrical symbol layers with legend (cp01), construction-stage overlays (cp15).
13. **Sheet level.**
    - Whole sheets at a lower px/m (41–82 px/m at 200 dpi in the USACE sheets) with a sheet border grid (1–20 / A–P).
    - Title block with revision table, key plan with hatched area, matchlines, general-notes text blocks, scale bar.
    - Several drawings per sheet.
    - Scale notes that do not hold on reduced prints (cp01).
    - Raise title block, legend and key plan frequencies when a "sheet" mode is drawn.
14. **Line-weight hierarchy.** Three weights (cut thick, view medium, dimensions and hatches thin), dashed hidden lines.
15. **Blueprints and diazo prints.** Inverted polarity (cp10), blue-line prints, photographed sheets with black borders (cp11–cp13): add to the degradations.

**Gaps for FM / CAFM plans:**
1. **A CAFM style family.**
   - Grey SOLID walls (128,128,128 or ACI 252 grey) with a thin outline, partitions sometimes unfilled.
   - No material hatches, no furniture, only overall dimensions.
   - Simplified doors and windows.
   - Today grey walls exist, but not as a coherent family with these rules.
2. **Room polygons printed in colour on the inner wall faces.** Magenta (ACI 210) or red (UZH) thin lines, deduction polygons around columns, the floor polygon on the outer face (ACI 214). Label them background.
3. **Usage colour fills keyed to a legend.**
   - Kanton ZH RGB ramps per SIA d0165 code (incl. the grey FF8 ramp close to the wall grey, and magenta HNF6).
   - UZH pastels (HNF #FFA8A8, VF #FFFFA8, FF #A8C0FF, NNF #FFD6A8).
   - SAP function colours.

   Today's fills are random pastels with no legend and no grey or magenta traps.
4. **FM stamps.**
   - BBL V1.0 AOID-only text (`WWWW.GG.EE.RRR`).
   - The legacy boxed `bbl-raumstempel` table (number, AOID+usage letter, name, area, FB/RB, B/W/D).
   - Stadt ZH boxed stamp with "VF 9.1" type codes; UZH MTEXT stamp.
   - Small-room stamps outside the building with a leader (Stadt ZH).
   - Today's stamps are free text without boxes, without usage codes and without leaders.
5. **Escape-route and safety layers.** Corridor and stairwell hatches in colour, extinguisher and call-point symbols (Stadt ZH `Z296`–`Z299`).
6. **Legacy overlays.** ANSI31/37 hatches over SOLID walls, raised-floor grids in rooms, door-attribute text, façade-grid ticks, red section lines.
7. **Reference points and the BBL title block.**
   - "RFP 1 / X=… Y=…" symbols.
   - A title block with "Revisionsplan / Grundriss / 5. Obergeschoss", index table A–E, "Planformat 63/59.4", the federal logo, and a **site overview map** that must be labelled background.

**Targets to add.** Polygon-line, stamp-box and legend masks, so the model learns to ignore them. A usage class per room, for colour-fill reading.

---

## 5. Synthetic Construction Drawings from BIM

### 5.1 Tools and Licences

| Tool | Licence (verified) | What it produces | Fit |
|---|---|---|---|
| IfcOpenShell 0.9 (`ifcopenshell.draw`, IfcConvert SVG serializer) | LGPL-3.0 (GitHub API, file headers) | Storey cuts at a section height derived from the storeys; cut / projection classes per IFC type; door arcs; space names and areas; storey-height lines; scale and bounds in mm; elevations and sections | Good for geometry and stamps; no annotation layer |
| Bonsai (formerly BlenderBIM) | GPL-3.0 (file headers) | Drawings and sheets in SVG with dimensions, text and tags, grids, automatic hatch patterns by material, title blocks; needs Blender and Inkscape. Docs: "Work in Progress", drafting module "still in early development" | Interactive; not a batch generator. GPL covers the code, not the drawings it produces, but scripting it headless is effort |
| plannotation | Apache-2.0 | IFC-aligned annotation layer in PDF: dimensions, grids, levels, room tags | Format reference for labelled sheets |
| Own code (`ifc_prepare.py` + `synth.py`) | Ours | Already cuts the 17 models into 72 storeys with walls, doors, windows, columns, stairs, named spaces, furniture | Extend with materials, tags, grids, dimensions (§5.3) |

### 5.2 Trial

The trial outputs and scripts were not kept in the repository.

- **Model.** `wbdg_office`, CC BY 4.0: the USACE/NIBS Common BIM "Office building", a brigade HQ of the same programme as cp05.
- **Settings.** `ifcopenshell.draw` with `auto_floorplan`, `space_names`, `space_areas`, `door_arcs`, scale 1:100, A0 bounds.
- **A bug in the default path.** It crashes in its cell-filling step (`AttributeError: 'SwigPyObject' object has no attribute 'is_a'`, draw.py line 376) on 0.9.0 with Python 3.14. With `cells=False` it writes a 2.4 MB SVG in 71 s with three storeys (Level 1, Level 2, Roof).
- **Rendering.** MuPDF ignores the SVG's CSS classes, so everything renders black. After inlining the CSS per class (cut = dark grey fill, projection = thin lines, IfcSpace = light fill) the plan is clean.
- **What the plan shows:**
  - cut walls with door gaps;
  - door arcs, stairs, lift, toilets;
  - stamps with **name, number and m²** ("SERVER RM / 144 / 18.00m²"), exactly an FM-style stamp.
- **What is missing:** dimension chains, axes, material hatches, door tags, level marks.
- **Grid attempt.** A run with `include_curves=True` on `fantasy_office_building_3` (to draw IfcGrid axes) did not finish in 35 minutes and was stopped.

**Survey of the 17 permissive models**:

| Model | Storeys | Spaces (named) | Walls with material | Door tags | Columns | IfcGrid axes | Furniture |
|---|---|---|---|---|---|---|---|
| wbdg_office (office) | 3 | 99 (99) | 487/487 (stud, plasterboard, brick) | 102/102 | 0 | 0 | 7 |
| digital_hub (office) | 3 | 64 (64) | 178/178 (Ortbeton, Trockenbau, Dämmung, Leichtbeton) | 64/64 | 62 | 0 | 0 |
| fantasy_office_building_1/2/3 | 5/3/6 | 44/27/87 | all (StB, Kalksandstein, Gipskarton, Dämmung) | all | 34/44/20 | 0/0/**22** | 128/142/224 |
| fantasy_residential_building_1 | 4 | 57 | 104/104 | all | 28 | **52** | 276 |
| ac20 (KIT institute) | 5 | 82 | 121/121 (Kalksandstein) | 77/77 | 2 | 0 (113 IfcAnnotation) | 253 |
| smiley_west (KIT) | 5 | 140 | 281/281 (Kalksandstein, Beton, Gips, Isolierung, Holz) | 170/170 | 20 | 0 | 0 |
| dental_clinic | 4 | 269 | 1080/1080 | 254/254 | 0 | 0 | 118 |
| west_riverside_hospital | 8 | **0** | 1311/1440 | 440/440 | 349 | 0 | 402 |
| sixty5 (residential tower) | 19 | 110 | 7426 (Ytong, Kalkzandsteen, isolatie) | 1049 | 0 | 2 | 722 |
| fantasy_hotel_1/2, molio, city_house_munich, duplex, fzk_house | 2–15 | 7–117 | all or most | all | 0–86 | 0 | 0–324 |

**Best suited:**
- digital_hub, fantasy_office_building_3, smiley_west and ac20: offices and institutes, German material names that map directly to SIA 400 hatches, columns, named rooms;
- wbdg_office: office with FM-style stamps and a matching real USACE sheet style;
- dental_clinic: many small rooms;
- fantasy_residential_building_1: 52 grid axes.

west_riverside_hospital has no IfcSpace, so it serves for walls and columns only.

### 5.3 Recommended Approach

- **Keep the geometry and the drawing code our own; use ifcopenshell only as the IFC reader** (LGPL, fine as a library).
- **Extend `ifc_prepare.py` to also record per element:**
  - wall material layers (`ue.get_material`), mapped to SIA 400 hatch classes and colours;
  - IfcDoor/IfcWindow `Tag`/`Name` and overall width and height, for opening annotations and door tags;
  - IfcGrid axes, or axes inferred from column centres where no IfcGrid exists;
  - storey elevations, for level marks;
  - IfcSpace LongName/Name and area, for stamps and usage classes.
- **Let renderer v3 draw from these records:**
  - geometry-bound exterior and interior dimension chains;
  - axes and bubbles;
  - door, window and wall-type tags;
  - level marks;
  - section lines across the building;
  - SIA hatches by material;
  - CAFM mode (grey SOLID walls, magenta polygons, AOID or boxed stamps, usage fills with legend).

  Every annotation pixel is labelled background (plus a "text" target), so the model learns to ignore it.
- **Use IfcOpenShell's SVG** (with `cells=False` until the bug is fixed upstream) as an independent second renderer, a "real CAD export" style. Use Bonsai only for a handful of hand-made reference sheets, if at all.

---

## 6. Proposed Curated Entries

For the coordinator; `data/curated` was not touched.

**New vocabulary needed:**
- **source:** `US federal drawings`, for the USACE PDF renders. The Commons-hosted NARA and HABS files fit `Wikimedia Commons`.
- **category:** `Construction and FM drawings` ("Working and construction sheets: dimension chains, axes, tags, section markers, overlays; FM area plans").
- **content:** `tags` (door, window and wall-type tags), `section markers`, `key plan`, `notes` (general-notes text blocks).
- **challenges:**
  - `dimension chains` ("chains and extension lines crossing rooms and touching walls");
  - `inverted polarity` (blueprints);
  - `scale note invalid` (reduced prints).

| Proposed id | File (origin) | Licence / use | Scale (px/m) | Category / input / era / type | Content and challenges | Why |
|---|---|---|---|---|---|---|
| cp-usace-acsc-a103 | construction-plans/images/cp03.png | Public domain (§105) / public domain or CC0 | 123.0 (3/16" = 1' on a D sheet at 200 dpi) | Construction / vector PDF / CAD era / public (community service centre) | dimensions, tags, section markers, key plan, axes-like matchlines, furniture, title block; dimension chains, small rooms, columns, large sheet | Densest real dimension-chain and door-tag sheet; vector source |
| cp-usace-acsc-life-safety | cp02.png | PD | 61.5 | Construction / vector PDF / CAD era / public | colour zones, legend, notes, title block; overlaid symbols, colour fills | Brandschutzplan analogue: red routes and rated partitions over a grey plan |
| cp-usace-bde-hq-1f | cp05.png | PD | 82.0 | Construction / vector PDF / CAD era / office | room stamps, area values, furniture, dimensions, title block, notes; open plan, small rooms, large sheet | Office HQ with name + area stamps: closest public analogue to an FM area plan on a construction sheet |
| cp-usace-hq-2f | cp06.png | PD | 41.0 | Construction / vector PDF / CAD era / office | room stamps, area values, furniture; open plan, small rooms, low resolution text | Large office floor at low px/m with tiny stamps |
| cp-usace-acsc-security | cp01.png | PD | null (printed 1/8" = 1' does not hold on the reduced exhibit) | Construction / vector PDF / CAD era / public | furniture, legend, room stamps, title block; overlaid symbols, scale note invalid | Tests scale reading on reduced prints |
| cp-nara-juneau-1911 | cp07.jpg (NARA 12013985) | `{{PD-USGov}}` | null (scale note) | Construction / scan colour / 1900–1949 / residential (federal) | dimensions, notes, several drawings, title block; hand lettering, low contrast | Ink working drawing of the Supervising Architect's office |
| cp-habs-naval-hospital-blueprint | cp10.jpg | `{{PD-USGov-NPS}}` | null | Hospitals / photo / unknown / hospital | dimensions, title block; inverted polarity, small rooms | Blueprint (white on dark): untested polarity |
| cp-habs-nhdvs-admin-2011 | cp14.jpg | `{{PD-USGov-NPS}}` | null (scale bars in feet and metres) | Administrative / born-digital raster / CAD era / office/administrative | dimensions, section markers, room stamps, scale bar, north arrow, title block; hatching that is not wall, massive walls | Modern CAD measured drawing of an administration building |
| cp-habs-stamford-po-1939 | cp15.jpg | `{{PD-USGov-NPS}}` | null | Administrative / scan 1-bit / 1900–1949 / office/administrative | annotations, notes, title block, dimensions; overlaid symbols, thin walls | Construction-stage overlays and temporary partitions, like Umbau plans |

**Optional: two FloorPlanCAD blocks** (use `benchmark only`, licence CC BY-NC 4.0, scale 100 px/m known).
- **Proposed entries:**
  - `fpcad-0060-0042`: school classrooms with desks, brick-pattern floor hatch, axes, dimension chains;
  - `fpcad-0081-0038`: hospital wards with beds, door and window tags, axes.
- **New vocabulary needed:**
  - source: `FloorPlanCAD`;
  - reference: `walls+openings+stairs` ("FloorPlanCAD line labels rasterised and filled; columns layer-derived").
- **What they add:** the only curated sheets with real Chinese construction CAD clutter, with ground truth.

---

## 7. Recommendations

1. **Tell BBL plainly.** Construction and FM plans are different domains from the public residential sets, and there is no public, trainable substitute. Real BBL sheets are required for evaluation now, and for fine-tuning later:
   - 10–20 construction PDFs (Werkplan 1:50, Bauprojekt 1:100) from recent projects;
   - 10–20 FM DWGs/PDFs: V1.0-conformant and legacy Revisionspläne.

   Zhang 2026 and CADSpotting show that a modest in-domain set recovers most of the loss.
2. **Build renderer v3 around two style families,** Werkplan and CAFM (§4.5).
   - Priority 1: geometry-bound dimension chains and axes, level marks and section lines; CAFM mode with magenta polygons, usage fills with legend, boxed and AOID stamps; renovation colours.
   - Priority 2: office furniture, overlay plans, sheet mode, blueprint polarity.
3. **Extend `ifc_prepare.py`** with materials, door tags, grids and storey levels (§5.3), and oversample the office models (digital_hub, fantasy_office_*, smiley_west, ac20, wbdg_office).
4. **Benchmarks:**
   - add `construction-plans` (15 sheets) and the FloorPlanCAD sample (§2.4) to `bench.py`;
   - request ArchCAD-400K only after the legal review;
   - ask the TUM authors for the German evacuation plans (FM-like evaluation);
   - for vector input, the USACE PDFs (exact vector geometry, no labels) and plan-check's two DWGs are ready test cases.
5. **Fix the catalogues (for the docs owner):**
   - FloorPlanCAD v2 contains text;
   - LS-CAD is still unreleased;
   - SIA 2014 is replaced by SIA 4006:2025;
   - the BBL CAD-Richtlinie V1.0 ZIP is public.
6. **Licence-clean style material at scale.** USACE MRSI standard designs, NPS eTIC public-domain assets and HABS sheets on Commons. Unlabelled, but suitable for self-training and style statistics (review §4.3).

---

## 8. Sources

**Local**
- `research/papers-md/`: 2024-yang-cadspotting (Tables 6, 8), 2025-luo-archcad-400k (Tables 2, 4, 7), 2024-ganon-waffle (Table 4), 2017-dodge-parsing-floor-plan-images, 2021-fan-floorplancad, 2024-liu-sympoint (Table 5), 2019-kalervo-cubicasa5k.
- Project docs: `docs/reviews/2026-10-07-pipeline-and-pilot-v2.md` §2.2; `pilot/v2-pipeline/synth.py`, `ifc_prepare.py`.

**Datasets**
- AI Hub: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71465
- AEC-Bench: https://github.com/Nomic-ai/aec-bench and https://arxiv.org/html/2603.29199
- DrawingVQA: https://huggingface.co/datasets/S2-MIND/DrawingVQA
- ArchSIBench: https://huggingface.co/datasets/ArchSIBench/ArchSIBench
- masterpn RC drawings: https://www.kaggle.com/datasets/masterpn/rc-beams-dataset-v1-25-reinforced-concrete-beam
- devp1866: https://www.kaggle.com/datasets/devp1866/engineering-blueprints-and-bill-of-materials-dataset
- cadsy: https://huggingface.co/datasets/cadsy/cad-technical-drawings
- plannotation: https://github.com/plannotation/plannotation
- Vandœuvre-lès-Nancy: https://www.data.gouv.fr/datasets/5ce7e5a78b4c415bdc600f33
- NPS eTIC: https://npgallery.nps.gov/etic
- NPS heritage documentation: https://www.nps.gov/subjects/heritagedocumentation/collection.htm
- TUM emergency plans: https://pmc.ncbi.nlm.nih.gov/articles/PMC10575354/
- mazharrehan floor plans: https://www.kaggle.com/datasets/mazharrehan/floorplan
- ArchCAD: https://huggingface.co/datasets/jackluoluo/ArchCAD and https://github.com/ArchiAI-LAB/ArchCAD
- FloorPlanCAD: https://floorplancad.github.io/
- TextCAD: https://arxiv.org/html/2607.12678
- CADSpotting: https://github.com/yangfy2023/CADspotting and https://www.sciopen.com/article/10.26599/CVM.2026.9450573
- MLSTRUCT-FP: https://github.com/MLSTRUCT/MLSTRUCT-FP

**Domain-shift papers**
- Raster2Seq: https://arxiv.org/html/2602.09016v1
- Zhang 2026: https://arxiv.org/html/2608.25608
- MitUNet: https://arxiv.org/html/2512.02413v3
- FGSSNet: https://ar5iv.labs.arxiv.org/html/2507.10343
- AECV-Bench: https://arxiv.org/abs/2601.04819
- DrawingVQA paper: https://arxiv.org/abs/2607.15418
- Aliyev & Barsi: https://doi.org/10.3311/PPci.44138

**Downloads**
- USACE ACSC: https://rfpwizard.mrsi.erdc.dren.mil/MRSI/content/cos/hnc/acsc/Library/Standard%20Designs/Standard%20Drawings%20-%20ACSC%20(Apr%2011).pdf
- USACE BDE/BN HQ: https://mrsi.erdc.dren.mil/content/cos/sas/bn-bde-hq/Library/Standard%20Designs/BDE-BN%20HQ%20Standard%20Design%20Rev%207.0%20dated%2020260622.pdf
- Commons file pages: in `attribution.csv`.

**Swiss standards and guidelines**
- BBL: https://www.bbl.admin.ch/de/downloads-bauten
- SIA 400:2000 (French edition, copyright SIA; buy from the SIA shop)
- SIA shop: https://shop.sia.ch (400, 112, 2014, 4006 tables of contents)
- Basel-Stadt CAD-Richtlinie 4.3: https://media.bs.ch/original_file/bf5172ae8a03f9d7b1080444798e6d233158abb3/cad-richtlinie-sa-version-4-3-okt24-2-3410-0.pdf
- Stadt Bern ISB: https://www.bern.ch/politik-und-verwaltung/stadtverwaltung/prd/hochbau-stadt-bern/downloads-fur-planer/organisation-und-zusammenarbeit/CAD%20Richtlinien_ISB_20200709.pdf
- Kanton Zürich: https://www.zh.ch/content/dam/zhweb/bilder-dokumente/themen/planen-bauen/hochbau/planungsgrundlagen/cad/grundlagen/cad_richtlinie_v4_2018.pdf (and Richtlinie Flächennachweis 2025)
- Stadt Zürich RFB: https://www.stadt-zuerich.ch/content/dam/web/de/planen-bauen/bauvorschriften-und-planerische-grundlagen/dokumente/standards-richtlinien-immo/computer-aided-facility-management-cafm/richtlinie-flaechenerfassung-und-erstellung-bewirtschaftungsplaene.pdf
- UZH: https://www.ib.uzh.ch/dam/jcr:a28f48d7-aa7f-4eca-8420-4fd340976837/Richtlinien_Bauwerksdokumentation_UZH_V_2024_1.pdf
- Aargau: https://www.ag.ch/media/kanton-aargau/dfr/dokumente/immobilien/projekte/richtlinien-und-standards/richtliniecad.pdf
- Solothurn: https://so.ch/fileadmin/internet/bjd/bjd-hba/04-Ueber-uns/Zusatzrubrik/CAD_Richtlinien_HBASO_4-2_1_Revision_20190703.pdf
- Basel-Landschaft: https://bl-api.webcloud7.ch/politik-und-behorden/direktionen/bau-und-umweltschutzdirektion/hochbauamt/wichtige-dokumente/richtlinien/richtlinie-cad-hbabl-v2-0.pdf
- Zug: https://zg.ch/dam/jcr:0746cb70-776d-49d0-b128-9d103e7a340e/CAD_Richtlinie.pdf
- St. Gallen: https://www.sg.ch/bauen/hochbau/richtlinien-und-vorlagen/cad---planverwaltung.html
- CADexchange CAFM-Basisrichtlinie 2016: https://s5e30cb687f13cf3d.jimcontent.com/download/version/1606990015/module/10997028312/name/CAFM_Richtlinie_Neutral_2016_06_16.pdf
- plan-check: https://github.com/bbl-dres/plan-check

**Tools**
- IfcOpenShell (LGPL-3.0): https://github.com/IfcOpenShell/IfcOpenShell
- IfcConvert usage: https://docs.ifcopenshell.org/ifcconvert/usage.html
- Bonsai drawings guide: https://docs.bonsaibim.org/studio/guides/drawings/index.html
