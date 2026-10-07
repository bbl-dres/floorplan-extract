# Public Datasets for Floor Plan Extraction

*Research catalogue · October 2026 · Public datasets for training and evaluating structured floor plan extraction, with licence verdicts and recommended use for BBL. Methods, models and paper-by-paper findings are in the [literature review and state of the art](literature-review.md), tools in [solutions](solutions-closed.md). Bracketed numbers refer to the shared [source list](../research/sources.md).*

## 1. Scope and Method

**Corpus.** About 60 datasets and data sources in nine groups: vector CAD drawings, raster floor plans, vector layouts and graphs, historical and style-diverse plans, BIM/IFC models, text in drawings, synthetic data, VLM benchmarks, and 3D reconstruction. They come from three places:
- the datasets used or introduced in the 32 papers of the [research index](../research/README.md);
- the dataset table now in [literature review §7](literature-review.md#7-datasets), and an earlier survey draft, which lists no further datasets;
- a web search (October 2026) for what both miss, focused on historical and non-residential plans, permissively licensed BIM models, text in drawings and 2025–2026 releases.

**Method.** For each dataset we recorded content and annotations, format, scale, building type and region, licence, access and current status. Licences and availability were checked against primary sources in October 2026: LICENSE files, dataset cards and records on Zenodo, Kaggle, Hugging Face and 4TU, terms of use, and the papers. Rows marked "not re-checked" rely on the papers or earlier catalogues only.

**Conventions.**
- Tables use Dataset (linked) | Content and annotations | Format | Scale and building type | Licence | Access and status | Limitations | Relevance.
- Licence verdicts follow the stated licence: P = permissive, commercial use allowed with attribution; SA = share-alike; ND = no derivatives; NC = non-commercial; R = research or academic use only, by request or agreement; U = unclear or conflicting.
- Relevance: High = licence-clean training source, or key test set for BBL's building types and graphical styles. Medium = useful benchmark, test set or partial training source that needs a licence review or major adaptation. Low = wrong domain, too small, not available, or the licence rules out use.
- Training means data that may end up in production weights. Benchmark means internal evaluation only, never in production weights. Weights trained on NC or R data are treated as unusable for production federal work without legal review ([literature review §11](literature-review.md#11-limitations-of-this-review)).

**Quality of the evidence.**
- Mirrors and derivatives often state a looser licence than the original (§5).
- Producers contradict themselves. CVC-FP's page names "Creative Commons Attribution 4.0 International License (CC BY-NC 4.0)"; ResPlan is CC BY 4.0 on GitHub and CC BY-NC-SA 4.0 on Kaggle.
- Sizes differ between versions and papers: FloorPlanCAD 11,602 or 15,663 drawings, LS-CAD 45 or 50 plans, ZInD 1,524 or 1,575 homes, WAFFLE 18,556 plans or "nearly 20K" images.

## 2. Key Findings

1. **No public dataset fits BBL's core case.** None combines office, administrative or historical buildings, room stamps with number, usage and area, German, French or Italian text, and scanned or hand-drawn styles under a permissive licence. Training data has to come from BBL's own compliant DWGs, Swiss Dwellings and rendered models, with synthetic room stamps.
2. **Swiss Dwellings remains the backbone.** Version 3.0.0 (Zenodo, March 2023, CC BY 4.0) is still the latest version found: over 45,000 apartments in about 3,100 buildings, metric geometry including columns, and unit usage that includes commercial, public and janitor areas.\[101\] It is already in `data/public/` and feeds the pilot v2 segmenter.
3. **Non-residential plans and columns exist only in non-commercial CAD datasets.**
   - FloorPlanCAD (CC BY-NC 4.0) covers offices, schools, hospitals and malls, but has been frozen since the project shut down in early 2022.\[6\]
   - ArchCAD is mostly public and commercial buildings and has column classes, but is CC BY-NC 4.0, gated, and only 40K of its 413K samples are public. Identifiable text was removed.\[26\]\[28\]\[29\]
4. **Permissively licensed IFC models are the main untapped source.** IFC-Bench collects 51 IFC models from 22 projects, including offices, a dental clinic and a hospital, mostly under CC BY 4.0 or MIT.\[118\] The buildingSMART community samples are CC BY 4.0.\[115\] Rendered to plans in randomised styles, they give exact labels for walls, doors, windows, columns, stairs and named spaces. They add diversity, not volume.
5. **Several licences are stricter than catalogued so far.**
   - ResPlan: CC BY 4.0 on GitHub, CC BY-NC-SA 4.0 on Kaggle, built from scraped real-estate listings.\[103\]\[104\]
   - CVC-FP: non-commercial, despite a CC BY 4.0 name on its page.\[108\]
   - LIFULL HOME'S, the basis of R2V and Raster-to-Graph: "Only universities or public research institutions are eligible to be a User".\[107\]
   - ArchCAD: CC BY-NC 4.0 and gated; its code carries an "Academic License".\[26\]\[28\]
6. **Mirrors are not a licence source.** Copies of CubiCasa5K, CVC-FP, MLSTRUCT-FP and MSD on Kaggle, Roboflow, Hugging Face, figshare and Zenodo state licences that the originals do not grant (§5).
7. **Style-robustness tests rely on restricted data.** The public options are WAFFLE (Wikimedia, per-image licences), Versailles-FP (500 historical plans, academic use), CVC-FP (122 scans, NC) and MLSTRUCT-FP (dataset terms not stated). A held-out BBL gold set remains essential ([pipeline §8](pipeline.md#8-evaluation)).
8. **Text in drawings is not covered.** No public dataset has room stamps with ground truth, and this search found no OCR benchmark for architectural drawings. FloorPlanCAD text is Chinese and NC; WAFFLE's OCR output is pseudo-labels.
9. **VLM benchmarks are small and inherit restrictions.** AECV-Bench (120 plans) reuses CubiCasa5K and CVC-FP; FPBench-2K is announced but not released. Their protocols are reusable, their data is benchmark-only.

## 3. Recommended Use

| Use | Datasets | Licence | Notes |
|---|---|---|---|
| Training | Swiss Dwellings v3.0.0 | CC BY 4.0 | Core source; rendered in randomised graphical styles (pilot v2) |
| Training | IFC-Bench models under CC BY 4.0 or MIT; buildingSMART community samples | CC BY 4.0 / MIT | Non-residential topology, columns, stairs, named spaces; exclude the four GPLv3 models |
| Training | Neufert 4.0; MSD from 4TU | CC BY 4.0 | Cleaned Swiss Dwellings subsets; residential |
| Training | BBL DWGs that pass plan-check | Internal | Room polygons, AOIDs, GF; process in Switzerland only |
| Training | Synthetic room stamps (DE/FR/IT) on renders | Own output | Only source of stamp supervision; OFL fonts in `data/public/fonts/` |
| Training after review | ResPlan | Conflict: CC BY 4.0 vs CC BY-NC-SA 4.0 | Only after the authors confirm the licence in writing |
| Training after review | WAFFLE images under public domain, CC0 or CC BY | Per image | Attribution per image; CC BY-SA images bring share-alike |
| Training after review | MLSTRUCT-FP | Dataset terms not stated | Ask for the terms with the download request |
| Benchmark only | FloorPlanCAD, ArchCAD (40K), CubiCasa5K, CVC-FP, AECV-Bench | CC BY-NC 4.0 or inherited | Never in production weights; ArchCAD access requires declaring non-commercial use – check with BBL's legal service first |
| Style and scan robustness | WAFFLE, CVC-FP, Versailles-FP, BBL gold set | Mixed | Versailles-FP only through an academic partner |
| Not usable | LIFULL HOME'S, R2V, Raster-to-Graph, RPLAN, ZInD, Structured3D, 3D-FRONT | R / NC | Eligibility or non-commercial terms exclude BBL |

## 4. Datasets by Category

### 4.1 Vector CAD Drawings

These datasets label vector primitives and serve symbol spotting on DWGs. Rendered to raster, they could also train raster models, but their licences rule that out for production.

| Dataset | Content and annotations | Format | Scale and building type | Licence | Access and status | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [FloorPlanCAD](https://floorplancad.github.io/) | Line-level panoptic labels: 30 thing + 5 stuff classes (doors, windows, furniture, sanitary, stairs, walls); no column class\[23\]. The 15,663-drawing release adds the text entities that TextCAD uses as "FloorPlanCAD-V2"\[4\] | SVG + PNG | 15,663 drawings (11,602 at first release); residential towers, schools, hospitals, malls, offices; China\[6\]\[19\] | Annotations CC BY-NC 4.0; drawings not owned by the authors\[6\] – NC | Direct download; frozen since the project shut down in early 2022\[6\] | Chinese drafting; plans cut into 10 m × 10 m blocks with 30% kept for privacy\[19\]; no room stamps | Medium (benchmark; non-residential) |
| [ArchCAD](https://huggingface.co/datasets/jackluoluo/ArchCAD) ([paper](https://arxiv.org/abs/2503.22346), [code](https://github.com/ArchiAI-LAB/ArchCAD)) | Primitive-level semantic and instance labels: 27 categories in the paper, 30 classes plus "Others" on the dataset card, incl. concrete and steel columns, beams, axis grid, stairs, elevators; Q&A pairs\[28\]\[29\] | Raster, SVG, JSON, Q&A, 2D point cloud | Public: 40K samples (2.63 GB). Paper: 413,062 chunks from 5,538 drawings; residential 14%, mostly public and commercial; China\[28\]\[29\] | CC BY-NC 4.0, gated; code under an "Academic License"\[26\]\[28\] – NC | Request form, approval within 3 business days; first open-source round 16 Oct 2025, further releases planned, full set requested in an open issue\[26\]\[105\] | Only drawings meeting Chinese layer standards; identifiable text removed\[29\] | Medium (benchmark; only dataset with columns) |
| [LS-CAD](https://arxiv.org/abs/2412.07377) (CADSpotting) | FloorPlanCAD-style labels on large drawings | Vector CAD | 45 drawings of at least 1,000 m² (50 in an earlier preprint version): campuses, office complexes, hotels\[30\] | Release announced, no licence – U | No download found (October 2026) | Tiny | Low (Medium if released; contact the authors) |
| [SESYD](http://mathieu.delalandre.free.fr/projects/sesyd/) / FPLAN-POLY | Symbol instances on synthetic floor plans (SESYD, 16 classes) and on plans from the internet (FPLAN-POLY, 38 classes)\[19\] | Vector / raster | 1,000 synthetic documents; 48 plans\[19\] | Not re-checked – U | Classic symbol-spotting benchmarks | Synthetic or tiny | Low |
| [Pang et al. CAD dataset](https://github.com/pangjunbiao/CAD-dataset) | 15 symbol keypoint types | Raster rendered from CAD | 360 images (300 train, 60 test), 1,700–4,200 px wide; telecom equipment rooms | Not re-checked – U | GitHub | Not building floor plans | Low |

### 4.2 Raster Floor Plans

| Dataset | Content and annotations | Format | Scale and building type | Licence | Access and status | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) | Polygons for rooms, walls, openings and icons, 80+ categories; three style subsets\[38\] | Raster + SVG | 5,000 plans, 50–8,000 px; Finnish residential\[39\]; every room labelled with its size in metres (scaled images at 1 cm/px) | CC BY-NC 4.0 on GitHub\[5\], CC BY-NC-SA 4.0 on Zenodo – NC (apply the stricter) | Direct download; in `data/benchmark/` | Residential; relabelled mirrors (§5); multi-floor sheets show all floors in each floor image | Medium (benchmark; used for the pilot v2 zero-shot test) |
| [CVC-FP](https://dag.cvc.uab.es/dataset/cvc-fp-database-for-structural-floor-plan-analysis/) | Rooms, walls, doors, windows, parking doors, room separations, with structural relations as an attributed graph\[108\] | Scans + SVG | 122 scans in 4 subsets by origin and style; 905–7,383 px\[39\]\[108\]; residential | Page states "Creative Commons Attribution 4.0 International License (CC BY-NC 4.0)" and restricts commercial use\[108\] – NC | Direct download, with the SGT annotation tool; in `data/benchmark/` | Small; residential; door labels cover the swing area, not the opening | Medium (real scans in several styles) |
| [MLSTRUCT-FP](https://github.com/MLSTRUCT/MLStructFP) | Wall polygons (70,873) and slab polygons, JSON metadata | PNG 6,500–9,500 px + JSON | 954 floors; multi-unit residential, Chile\[109\] | Loader MIT; dataset licence not stated – U | Download link by request form; repository archived 29 April 2026\[109\] | Walls and slabs only | Medium (high-resolution wall labels; terms open) |
| [R2V](https://github.com/art-programmer/FloorplanTransformation) / [LIFULL HOME'S](https://www.nii.ac.jp/dsc/idr/en/lifull/) | R2V: vector labels for walls, doors, windows, room types, icons. LIFULL: rental data and floor plan images | Raster + vector | R2V: ~870 annotated plans. LIFULL: 5.31 million floor plan images; Japanese residential\[106\] | "Only universities or public research institutions are eligible to be a User", for "not-for-profit informatics-related researches"\[107\] – R | Application to NII, reviewed by LIFULL\[106\] | BBL not eligible | Low |
| [Raster-to-Graph dataset](https://github.com/SizheHu/Raster-to-Graph) | Wall-junction graphs with room semantics | Raster + graph | More than 10,000 plans from LIFULL; Japanese residential\[40\] | Annotations by request form; images need the LIFULL licence – R | Request | Inherits LIFULL terms | Low |
| [RPLAN](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html) | Room layouts | 256 × 256 raster | ~80,000; Chinese residential | Research-only on request (not re-checked) – R | Request | Layouts, not drawings | Low |
| Small early sets: [Rent3D](http://www.cs.toronto.edu/~fidler/projects/rent3D.html), [R3D](https://github.com/zlzeng/DeepFloorplan), R-FP, SydneyHouse, [ROBIN](https://github.com/gesstalt/ROBIN) | Walls, rooms, symbols; mostly semantic masks without vector ground truth | Raster | About 100–500 plans each; residential (London, Japan, Sydney) | Mostly research terms; not re-checked – U | R-FP and SydneyHouse availability unconfirmed | Small; one style each | Low |
| BRIDGE | Floor plan images with descriptions | Raster | 13,000 images | – | "Not publicly available" in 2021\[19\]; current status unconfirmed | – | Low |
| [Mendeley "floor plan dataset"](https://data.mendeley.com/datasets/ss25hm53cz/1) | Undocumented | Images | Undocumented | CC BY 4.0\[138\] – P | Direct download; not inspected | Provenance unknown | Low |

### 4.3 Vector Layouts and Graphs

Metric vector data with room types can be rendered in any graphical style, which matters more for BBL than building type. The Swiss Dwellings family is the only large permissive source.

| Dataset | Content and annotations | Format | Scale and building type | Licence | Access and status | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [Swiss Dwellings](https://zenodo.org/records/7788422) v3.0.0 | Areas with room types, walls, railings, columns, doors, windows and features (sanitary, kitchen) as WKT polygons in metres; unit usage residential, commercial, public, janitor; simulations\[101\] | CSV | Over 45,000 apartments (370,000 rooms) in ~3,100 buildings\[101\]; local copy: 3.26 M geometry rows, 13,905 floors, 3,184 buildings; Switzerland | CC BY 4.0\[101\] – P | Direct download (931.9 MB); in `data/public/`; no later version found | Sourced from Archilyse's commercial clients\[7\]; residential-dominated; geometry only, no drawing style or room stamps | High (licence-clean, Swiss, metric) |
| [Modified Swiss Dwellings (MSD)](https://data.4tu.nl/datasets/e1d89cb5-6872-48fc-be63-aadd687ee6f9) | Building-level floor plans, room types, structural walls, access graphs (passage, door, front door)\[69\] | Graphs + images + geometries | 5,372 floor plans, 18,900 units; residential only\[69\] | 4TU: CC BY 4.0 – P; Kaggle: CC BY-SA 4.0\[71\]\[72\] | Direct download; use the 4TU record | Filtering removed plans with non-residential rooms such as waiting rooms | Medium (door-level graphs; otherwise use Swiss Dwellings) |
| [MSD JSON](https://zenodo.org/records/17294451) | 4,572 room-based geometries with topological dual graphs, for TopologicPy | JSON (319.9 MB) | Switzerland; residential | CC BY 4.0, but derived from the Kaggle MSD database (v6)\[102\]\[72\] – U | Direct download; Cardiff University, October 2025 | Licence chain via a CC BY-SA copy | Low |
| [Neufert 4.0](https://zenodo.org/records/14223942) | Audited Swiss Dwellings derivative, 20,419 plans reviewed\[73\] | Vector | 20,000+; residential | CC BY 4.0 (not re-checked) – P | Zenodo | Residential | Medium (cleaner subset) |
| [ResPlan](https://github.com/m-agour/ResPlan) | Walls, doors, windows, 17 room classes; graphs with typed edges (via_door, adjacency, direct, via_window); metric coordinates | Shapely polygons + NetworkX graphs | 17,000 plans; South Asian residential\[103\] | GitHub: data CC BY 4.0, code MIT\[103\]; Kaggle: CC BY-NC-SA 4.0 (updated July 2026)\[104\] – U | Direct download; takedown policy in the repository | Scraped listings with withheld platforms; 6.9% near-duplicates | Medium (benchmark; training only after licence confirmation) |
| HouseExpo | 2D indoor layouts | Vector | 35,126 layouts from SUNCG | Not re-checked – U | GitHub | SUNCG was withdrawn; provenance risk | Low |

### 4.4 Historical, Scanned and Style-Diverse Plans

Drawing-style shift is the dominant failure mode in the literature: a CubiCasa5K model scores door IoU 0.099 on WAFFLE ([literature review §2](literature-review.md#2-key-findings)). These sets test it.

| Dataset | Content and annotations | Format | Scale and building type | Licence | Access and status | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [WAFFLE](https://github.com/TAU-VAILab/WAFFLE) ([paper](https://arxiv.org/abs/2412.00955)) | Plan images with building type, country, Wikipedia text, OCR and grounded legends; dense segmentation labels on 110 images | Raster (+ SVG where available) | 18,556 plans; more than 1,000 building types and 100 countries, incl. historical and public buildings\[100\] | Code "under the Wikimedia Commons license"\[110\]; each image keeps its own Commons licence (public domain, CC0, CC BY, CC BY-SA) – U | SharePoint download\[110\]: anonymous access fails; a browser download omits the original images (size limit). In `data/benchmark/`: metadata, 1,208 SVG plans, the 110-image segmentation benchmark, OCR outputs | Labels mostly automatic; licence per image, not in the metadata (query Commons) | High (best public test for unseen styles) |
| [Versailles-FP](https://www.etis-lab.fr/versailles-fp/en/) ([paper](https://arxiv.org/abs/2103.08064)) | Wall masks | Scans | 500 historical plans of the Château de Versailles, digitised by the VERSPERA project\[111\]\[113\] | "Academic usage of this dataset is free"; commercial use not addressed\[112\] – R | Agreement page; published at ICDAR 2021 | Walls only; one archive | Medium (historical wall test, via an academic partner) |
| Sapienza archive drawings | Semantic segmentation of analogue architectural drawings with hatch and texture conventions | Scans | Not stated | Release not stated – U | ISPRS Archives 2026\[137\] | Availability unknown | Low (contact the authors; hatched walls) |

CVC-FP (122 scans in 4 styles) and MLSTRUCT-FP (954 high-resolution plans) are listed in §4.2 and also serve as style tests.

### 4.5 BIM/IFC Models and Indoor Mapping

Cut and rendered as floor plans in randomised styles, IFC models give exact labels for walls, slabs, doors, windows, columns, stairs and spaces with names and numbers, in non-residential buildings. They are few, so they add diversity, not volume. Rendering creates a derivative, which rules out ND licences.

| Dataset | Content and annotations | Format | Scale and building type | Licence | Access and status | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [IFC-Bench](https://huggingface.co/datasets/sylvainHellin/ifc-bench) | 51 IFC models of 22 projects (architecture, structure, MEP); 1,027 QA pairs | IFC + JSON | Offices (several), dental clinic, hospital, hotels, residential\[118\] | QA CC BY 4.0; models keep their original licences, "majority … CC BY 4.0 or MIT", four GPLv3\[118\] – P (per model) | Hugging Face; 17 permissive models (CC BY 3.0/4.0, MIT) in `data/public/ifc-bench/` | Check each model's licence and origin; exclude the GPLv3 models; Schependomlaan is academic-only at its origin | High (non-residential render source) |
| [buildingSMART Community Sample Test Files](https://github.com/buildingsmart-community/Community-Sample-Test-Files) | Community IFC2x3, IFC4 and IFC4.3 samples | IFC | Not counted | CC BY 4.0\[115\] – P | GitHub | "NOT official buildingSMART examples"; most fail the validation service\[115\] | Medium (render source; check content) |
| [Open IFC Model Repository](https://openifcmodel.cs.auckland.ac.nz/) | Donated IFC models | IFC | Not counted | Per model; upload licence CC BY 3.0 per re3data\[119\]\[120\] – P (per model, unverified) | University of Auckland | Check each model | Medium |
| [Schependomlaan](https://github.com/openBIMstandards/Archive-DataSetSchependomlaan) | IFC design models, point clouds, event log | IFC | One Dutch residential project | "Permission to use the data for scientific and academic purposes"\[121\] – R | Archived 2020 | Residential; academic use | Low |
| [buildingSMART Sample-Test-Files](https://github.com/buildingSMART/Sample-Test-Files) | Certification datasets; the former sample files were purged in November 2024 | IFC | Small test models | "Copyright © buildingSMART International Ltd."\[116\] – U | GitHub | No reuse licence | Low |
| [IFC4.3.x sample models](https://github.com/buildingSMART/IFC4.3.x-sample-models) | Small concept examples | IFC | Snippets | CC BY-ND 4.0 after formal release\[117\] – ND | GitHub | Not whole buildings; ND | Low |
| [opensourceBIM IFC-files](https://github.com/opensourceBIM/IFC-files) | Community IFC files | IFC | Not counted | CC BY-ND 4.0 (not re-checked)\[139\] – ND | GitHub | ND | Low |
| [IFCNet](https://ifcnet.e3d.rwth-aachen.de/) | Single IFC objects in 65 classes | IFC / mesh | 19,613 objects | No licence provided\[140\] – U | Website | Objects, not plans | Low |
| [OpenStreetMap indoor](https://wiki.openstreetmap.org/wiki/Simple_Indoor_Tagging) | Rooms, corridors, doors and levels (Simple Indoor Tagging) | Vector | Sparse; mostly stations, malls, campuses | ODbL (not re-checked) – SA | Overpass or planet extracts | Uneven quality; no drawn walls | Low |

No Swiss federal or cantonal open dataset of indoor geometry was found.

### 4.6 Text in Drawings

No public dataset has room stamps (number, usage, area) with ground truth in any language, and this search found no public OCR benchmark for architectural or engineering drawings.

| Dataset | Content and annotations | Format | Scale and building type | Licence | Access and status | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| FloorPlanCAD text entities | CAD text in the 15,663-drawing release; TextCAD derives 166 annotation types from it\[4\] | SVG | Chinese drawings | CC BY-NC 4.0 – NC | With FloorPlanCAD | Chinese; no stamp ground truth | Low |
| ArchCAD Q&A | Generated questions and answers on drawing chunks\[28\] | JSON | Chinese drawings | CC BY-NC 4.0, gated – NC | With ArchCAD | Identifiable text removed | Low |
| WAFFLE OCR | OCR detections (pseudo-labels) on multilingual plans | JSON | 18,556 plans | Per image – U | With WAFFLE | Not ground truth | Low (multilingual test material) |
| [Text detection on technical drawings](https://arxiv.org/abs/2205.02659) | Generator of synthetic technical drawings with text, plus detection code\[134\] | Raster | Synthetic | Code licence not re-checked – U | GitHub | Technical drawings, not plans | Medium (generator pattern for room stamps) |
| [Engineering drawing parsing](https://arxiv.org/abs/2505.01530) | Title blocks, notes, measures; 9 categories\[135\] | Raster | 1,367 drawings | Release not stated – U | – | Not released | Low |
| [Rebar drawings](https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2026.1839808/full) | Rebar annotations\[136\] | Raster | 1,005 drawings, 7,108 annotations | "Available on request" – R | Request | Structural drawings | Low |

### 4.7 Synthetic Data and Generators

| Dataset | Content and annotations | Format | Scale and building type | Licence | Access and status | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| Own rendering ([pilot v2](../pilot/v2-pipeline/)) | Swiss Dwellings rendered in randomised graphical styles with synthetic lettering; IFC models and compliant DWGs can feed the same renderer | Raster + labels | Unlimited | Own output; inherits source licences (CC BY 4.0 needs attribution) – P | `synth.py`, `sd_prepare.py` | Realism depends on the style generator; add Augraphy for scan defects | High |
| [Google Fonts](https://github.com/google/fonts) (OFL subset) | 13 families for hand lettering, calligraphy, Fraktur, historical print and CAD styles | TTF | 13 families | SIL OFL 1.1 – P | In `data/public/fonts/` | Fonts only | High (room-stamp lettering) |
| FloorplanQA generator | Layouts in several rendering styles | JSON + images | 1,800 layouts generated with Gemini 2.5 Pro, 200 from HSSD\[57\] | Not re-checked – U | GitHub\[129\] | Single rooms with furniture | Low |
| [Infinigen Indoors](https://github.com/princeton-vl/infinigen) / [ProcTHOR](https://github.com/allenai/procthor) | Procedural 3D indoor scenes | 3D | Unlimited | Not re-checked – U | GitHub | Residential 3D scenes, not drawings | Low |
| Floorplan-2M / Floorplan-HQ-300K (FloorplanVLM) | Industrial plans and a re-rendered, pixel-aligned subset\[42\] | Raster + vectors | 2 M / 300K | Not released | – | Proprietary | Low |

SESYD (§4.1) is the classic synthetic symbol-spotting set.

### 4.8 VLM and QA Benchmarks

| Dataset | Content and annotations | Format | Scale and building type | Licence | Access and status | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [AECV-Bench](https://github.com/AECFoundry/AECV-Bench) | Counting doors, windows, bedrooms and toilets; OCR, counting and spatial QA | Raster + JSON | 120 plans + 192 QA pairs; plans from CubiCasa5K, CVC-FP and public drawings on the internet\[55\] | Repository licence not confirmed; data inherits NC\[127\] – NC | GitHub | Residential; small | High (protocol); data benchmark only |
| [ArchPlanVQA](https://github.com/pengyang-Li/ArchPlanVQA) | VQA on floor plans converted from CAD; general VLMs reach 33.03–37.88%\[56\] | Raster + QA | Not checked | Not re-checked – U | GitHub\[128\] | – | Medium |
| [FloorplanQA](https://github.com/OldDeLorean/FloorplanQA) | Spatial reasoning over structured layouts | JSON | 2,000 layouts\[57\] | Not re-checked; the HSSD part inherits HSSD terms – U | GitHub\[129\] | No perception | Low |
| FPBench-2K (FloorplanVLM) | Vectorisation benchmark with Manhattan and non-Manhattan plans | Raster + vectors | 2,000 plans\[42\] | Announced as open source – U | No download found | – | Low (Medium if released) |
| [AEC-Bench](https://github.com/nomic-ai/aec-bench) | Agentic tasks on drawings, schedules and specifications | Mixed | Not checked | Apache-2.0 (code)\[130\] – P | GitHub | Licence of the underlying drawings not checked | Low |
| MMArch, ResPlan-FP, FloorPlan-VLN | Architectural reasoning; rendered ResPlan plans; navigation on Matterport floor plans | Mixed | ResPlan-FP: 16,998 plans | Inherited (ResPlan conflict, Matterport terms) or not stated – U | arXiv 2026\[131\]\[132\]\[133\] | Not re-checked | Low |

### 4.9 3D Scans and Reconstruction

| Dataset | Content and annotations | Format | Scale and building type | Licence | Access and status | Limitations | Relevance |
|---|---|---|---|---|---|---|---|
| [ZInD](https://github.com/zillow/zind) | 360° panoramas with room layouts, windows, doors, openings and 2D floor plans | Panoramas + JSON | 1,575 homes, 67,448 panoramas, 2,737 floor plans; US residential\[114\] | Academic, non-commercial use only\[114\] – NC | Registration on the Bridge platform with verified academic affiliation; manual approval\[114\] | Not drawings; residential | Low |
| Structured3D, 3D-FRONT, HSSD, SceneCAD | Synthetic or scanned 3D residential scenes; density-map benchmarks for RoomFormer, PolyRoom and FRI-Net | 3D | Structured3D: 3,500 scenes | Structured3D non-commercial; others not re-checked – NC / U | Various | Point-cloud domain | Low |

## 5. Licence Pitfalls: Mirrors and Derivatives

| Copy | Stated licence | Original | Original licence |
|---|---|---|---|
| [Kaggle qmarva/cubicasa5k](https://www.kaggle.com/datasets/qmarva/cubicasa5k)\[122\] | CC BY-NC-SA 3.0 IGO | CubiCasa5K | CC BY-NC 4.0\[5\] |
| [Roboflow cubicasa5k-2](https://universe.roboflow.com/floorplan-recognition/cubicasa5k-2-qpmsa)\[48\] | MIT | CubiCasa5K | CC BY-NC 4.0 |
| [Floorplan Isovist Dataset](https://zenodo.org/records/13871782)\[123\] | CC BY 4.0 | CubiCasa5K | CC BY-NC 4.0 |
| [Roboflow CVC-FP subset](https://universe.roboflow.com/sandy-s-workspace/cvc-fp-floorplan) (88 images)\[124\] | CC BY 4.0 | CVC-FP | NC\[108\] |
| [figshare CVC-FP training data](https://figshare.com/articles/dataset/Data_and_codes_of_Indoor_mapping_and_modeling_by_parsing_floor_plan_images/12082131)\[125\] | CC BY 4.0 | CVC-FP | NC |
| [Hugging Face Space MLSTRUCT-FP](https://huggingface.co/spaces/rawanessam/MLSTRUCT-FP)\[126\] | MIT | MLSTRUCT-FP | MIT covers the loader only; dataset terms not stated\[109\] |
| [Voxel51/FloorPlanCAD](https://huggingface.co/datasets/Voxel51/FloorPlanCAD) | CC BY-SA in the metadata | FloorPlanCAD | CC BY-NC 4.0\[6\] |
| [Kaggle MSD](https://www.kaggle.com/datasets/caspervanengelenburg/modified-swiss-dwellings/data)\[72\] | CC BY-SA 4.0 | MSD on 4TU | CC BY 4.0\[71\] |
| [Zenodo MSD JSON](https://zenodo.org/records/17294451)\[102\] | CC BY 4.0 | Kaggle MSD (v6) | CC BY-SA 4.0 |
| [Kaggle ResPlan](https://www.kaggle.com/datasets/resplan/resplan)\[104\] | CC BY-NC-SA 4.0 | ResPlan on GitHub | CC BY 4.0\[103\] |
| [WAFFLE repository](https://github.com/TAU-VAILab/WAFFLE)\[110\] | "Wikimedia Commons license" | Wikimedia Commons images | Per image |

Rules:
- The producer's licence applies; a mirror cannot grant more rights than its source.
- Where a producer contradicts itself (ResPlan, CVC-FP), apply the stricter reading until the producer confirms otherwise in writing.
- Record the producer's licence, URL and version in each dataset's `SOURCE.md` ([data/README](../data/README.md)).

## 6. Gaps

What no public dataset provides, and BBL has to build or label itself:
1. **Office, administrative and public-building plans with room stamps** under a permissive licence.
2. **Room-stamp text** (AOID or number, usage, area) in German, French and Italian with ground truth, and an OCR benchmark for drawings.
3. **Historical and hand-drawn plans** with structural labels under a usable licence. Versailles-FP is academic-only and labels walls only.
4. **Scans with exact vector ground truth** at scale. CVC-FP has 122 and is non-commercial.
5. **Gross floor area polygons, voids and SIA 416 area rules** as labels; ramps; stairs as flights.
6. **Column grids** outside non-commercial Chinese CAD data. Swiss Dwellings and MSD have columns, but in residential buildings.
7. **Swiss open indoor geometry** of any kind.

Compliant BBL DWGs ([pipeline §7](pipeline.md#7-training-data)), rendered IFC models and synthetic room stamps address the training gaps; the BBL gold set ([pipeline §8](pipeline.md#8-evaluation)) addresses evaluation.

## 7. Limitations of This Catalogue

- Not legal advice. NC, ND, research-only and eligibility clauses need an opinion from BBL's legal service before any use, benchmarking included.
- Rows marked "not re-checked" rely on papers or earlier catalogues. This applies to SESYD, FPLAN-POLY, RPLAN, Rent3D, R3D, R-FP, SydneyHouse, ROBIN, HouseExpo, Neufert 4.0, Structured3D, 3D-FRONT, HSSD, SceneCAD, opensourceBIM, OpenStreetMap indoor, Infinigen, ProcTHOR, and the 2026 benchmarks MMArch, ResPlan-FP and FloorPlan-VLN.
- Licences and availability are as found in October 2026 and will change: ArchCAD plans further releases, and LS-CAD and FPBench-2K are announced.
- Sizes are quoted as stated by the producers and differ between versions (§1).
- The search found no Swiss or German-language plan datasets, but national and cantonal archives were not searched systematically.
- Downloaded and inspected (October 2026): Swiss Dwellings v3.0.0, the fonts and 17 IFC-Bench models in `data/public/`; CubiCasa5K, CVC-FP, WAFFLE (partial) and a sample of Swiss plans from Wikimedia Commons in `data/benchmark/`. All other entries are based on their documentation.

## 8. Sources

- Shared [source list](../research/sources.md); numbers 101–140 were added for this catalogue.
- Papers: [research index](../research/README.md), especially the datasets group and the papers that introduced each benchmark; Markdown versions in `research/papers-md/` (local only).
- Local dataset records: [data/README](../data/README.md) and the `SOURCE.md` files in `data/public/`.
- [Motivation and goals](motivation-goals.md) · [Literature review and state of the art](literature-review.md) · [Solutions](solutions-closed.md) · [Pipeline design](pipeline.md)
