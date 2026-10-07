# Centre-Line (BIM-Style) Floor Plan Reconstruction: Literature and a Recommendation for Pilot v2

*Status: analysis as delivered on 7 October 2026, prompted by the S1 stair case below; no pipeline change made yet. The recommendations (§10) are candidates for the next pilot iteration and for [pipeline.md](../pipeline.md) stages 3–6.*

*Analysis · 7 October 2026 · Question: should pilot v2 reconstruct plans the way a BIM authoring tool models them (walls as centre lines with thickness forming a junction graph; rooms as faces bounded by walls and room-separation lines; doors and windows hosted on the wall centre line; stairs and fixtures as objects inside rooms that never bound them)? If so, what should produce the graph, how are gaps closed, and how is the robustness of the current free-space room approach kept?*

*Method: read the method, representation, post-processing, results and limitation sections of 25 papers in `research/papers-md` (list in §12), plus targeted passages of eight more that touch walls, junctions, room faces or opening hosting; `research/papers.json` notes; `docs/literature-review.md`; `docs/pipeline.md` stages 3–6; `docs/reviews/2026-10-07-pipeline-and-pilot-v2.md` §3.2 and §4.4; the evidence reports in `docs/reports`; pilot v2's `fpx/walls.py`, `openings.py`, `stairs.py`, `rooms.py`, `config.py`, `sd_prepare.py`, `ifc_prepare.py`, and the harness results `pilot/v2-pipeline/data/harness/oracle.json` and `render.json` (both run 7 October 2026). No repository file was changed.*

*Conventions: figures are quoted from the papers as printed (their units, their benchmarks); "pp" are percentage points. Papers that are not local (SALI-FP, KIPPI, Wu et al. 2018, Chen & Tu 2025, Chang et al. 2025, Lin & Wang 2026) are cited second-hand from `docs/reports/2026-10-07-pipeline-inputs-masking-scale-evidence.md`, `docs/reports/2026-10-07-missing-papers.md` and `docs/literature-review.md`, and marked †. "Pilot" and "harness" figures are our own evidence, not literature. Statements about how Revit or IFC model rooms are practice knowledge, marked (practice), not paper findings.*

---

## 0. Short Answer

1. **The representation is right; adopt it as the canonical data model and the export.** The closest published models are FloorplanVLM (walls as start, end, thickness τ and curvature κ; openings "nested within their parent walls" with class, width ω and a centre offset δ "along the wall centerline"; rooms as "an ordered sequence of indices that reference walls … and form a closed topological cycle") and Zhang 2026's wall-first grammar, which adds what open plans need: rooms as "a cycle of wall IDs in which an edge may be marked open (no wall)". Sketch2BIM builds rooms from "a halfedge planar graph" and writes Revit; Lv 2021 derives a centre-line graph with wall widths and openings "located on specific walls". No method covers the whole BIM set (parametric walls, shared junctions, hosted openings with swing, separation edges, metric units, stairs and voids as objects); §3.
2. **On its own it does not fix missing walls.** A BIM tool fails the same way: a room in an unclosed boundary is not enclosed (practice). The literature measures the cost of rooms-from-walls: deriving rooms from walls "is global, since every cycle has to close, and loses information" (Zhang 2026); faces of a detected wall graph reach room F1 0.591 on CubiCasa5K against 0.784 for a room-polygon decoder with reconciliation (Zhang, Table 2); CubiCasa5K's own room IoU falls from 57.5 to 49.3 when segmentations become polygons; even Raster-to-Graph, on residential plans of 10–50 junctions at 512 px, gets 67.0% of plans "structurally perfect" (Raster-to-Vector 14.8%).
3. **What the graph does buy** is reasoning at the level of segments and junctions instead of pixels: a chord is accepted when the centre-line evidence covers a fraction of it (Zhang's readout, κ = 0.5), fragments are "chained across gaps up to 450 mm" and wall ends snap to the next support (Talebi-Kalaleh & Mei), junction degrees and closed loops are constraints (Raster-to-Vector: junction precision 70.7 → 94.7). Centre lines that continue through openings make doors close rooms by construction; stairs and fixtures can be declared non-bounding; room-separation lines become typed edges; door connections become the two faces of the host edge; IFC export becomes direct.
4. **Produce the graph by detection, not emission, and not only by skeletonising the wall mask.** On the same network, a training-free readout of junction and centre-line heatmaps beats autoregressive emission on real scans by 2.7 pp wall F1 at tolerance 0.05 and 5.1 pp at 0.015, closes rooms by construction (watertightness 0.956 vs 0.196), and its lead grows with plan size (+8.7 pp on plans with 32–70 walls) (Zhang 2026). Emission (FloorplanVLM, Raster2Seq, Zhang's decoder) degrades with sequence length and caps plan size; BBL office floors are large. Skeletonisation stays as the fallback and as the label generator (Zhang builds its ResPlan-FP centre lines exactly that way). CAD prints get a vector side channel (Talebi-Kalaleh & Mei; Sketch2BIM's double-stroke merge).
5. **Keep the region rooms and fuse, room by room.** Zhang 2026 found representation "secondary": a room-centric system with a deterministic reconciliation step produced no double walls, watertightness 0.92–0.94 and the lowest edit cost of the single systems, and fusing wall-detected walls into it added 7.1 pp wall F1. Our free-space regions are the room-centric hypothesis. Accept graph faces that agree with regions; where the graph does not close, derive the missing edges from the region boundary (Lv 2021, Zhang's weld/deduplicate/T-split); flag disagreements. Every room then ends as a face, and robustness is kept.
6. **Stairs must never bound rooms.** Model flights plus landings as a stair object, the stair eye as a void polygon, and let only walls, railings around voids and separation lines bound faces. Delete the convex-hull split; tread lines must be stair, not barrier. The literature barely covers this: Raster-to-Vector treats stairs as an icon box, Swiss Dwellings has `stairs` as a feature and `staircase` as an area type, ResPlan has `stair` as a space category, nobody models landings or voids. Our harness shows the cost: with perfect labels, 88 of 345 staircase areas are split (recall 0.661).
7. **Open plan needs a room-separation output.** It is the largest miss class on Swiss Dwellings (harness, perfect labels: 687 of 1,247 missed rooms are merges; kitchen recall 0.581, corridor 0.687). Evidence for a learnable boundary: VectorFloorSeg classifies extended wall lines as room boundaries (CubiCasa5K room integrity 67.51 vs 41.89 for OCRNet); Zhang's grammar and synthetic open-plan tier; MSD's "passage" edges between area polygons within 0.04 m. Labels come free from Swiss Dwellings area polygons.
8. **Openings:** store (host wall id, offset along the centre line, width, hinge side, swing side); project detections onto the host (Raster-to-Vector: parallel within 10 px; ResPlan: doors aligned to the wall band in 99.94% of plans; Sketch2BIM: ≥ 0.75 ft from wall ends, ≥ 0.50 ft apart); detect them with a local head, not a decoder (Zhang: opening F1 0.25 emitted vs 0.64 read from the heatmap).
9. **Areas:** no paper reports room area error against stamps. Compute net area from the face offset inwards by t/2 per wall edge (0 for separation edges) where walls are regular, and from pixel wall faces where they are not (poché, niches, pilasters). Our evidence: S1 median 2.7%; harness median absolute error 2.07% (perfect labels) and 2.95% (renders).
10. **New heads and labels:** centre line (continuous through openings), junctions with sub-pixel offset, room-separation lines, interior footprint, opening instances with hinge and swing, stair object (flights + landings), void, railing, column. All derivable from Swiss Dwellings plus the renderer; IFC-Bench could give true wall axes and space boundaries.

---

## 1. Context: What Pilot v2 Does and Where It Fails

**Code facts (stages 3–6).**
- *Walls* (`fpx/walls.py`): the cleaned wall mask is skeletonised into a graph of junction clusters and end points; short spurs are pruned; each edge gets thickness = 2 × median distance transform. The graph feeds centre lines, thickness, wall-gap passages and host walls, not rooms.
- *Openings* (`fpx/openings.py`): door and window components that touch the wall mask; host = nearest wall segment within `opening_host_max_dist`; passages by ray casting from wall ends (0.5–2.2 m).
- *Stairs* (`fpx/stairs.py`): stair outline = convex hull of flights closer than 1.5 m.
- *Rooms* (`fpx/rooms.py`): barrier = walls ∪ columns ∪ doors ∪ windows ∪ passage lines, closed by 0.3 m; rooms = connected free space inside the largest building component (1.6 m closing); a region that overlaps the stair hull is split along the hull when both parts exceed 3 m²; regions leaking to the sheet border are sealed (erosion 0.65 m, watershed).

**The observed case (Landgut Lohn S1, 2005 CAD print).** In the viewer, parts of the walls around the staircase are missing and the hall is cut into pieces: the pipeline returns the staircase as two rooms, "Treppe" (11.4 m²) and an unnamed stair piece (3.0 m²), where the 2005 reference has one staircase with a stamp of 17.98 m² plus the void; diagonal edges of the convex hull around the flights cut off small triangles of the landing. The room polygons have 25–133 vertices each (traced pixel outlines), not the few straight wall faces a CAD or BIM model would have.

**The two observed failures follow directly.** (1) A wall piece missing over more than the 0.3 m slit closing joins two rooms into one component or lets a room leak; nothing above pixel level can close it. (2) Stair flights are not barrier, so the hall and the stair form one region, which the convex hull then cuts; tread lines predicted as wall add barrier inside the hall.

**Harness (our evidence, 7 October 2026; one-to-one matching at IoU ≥ 0.5).**

| Run | Floors | Room recall | Precision | Missed: merged | outside building | split | oversized | missing |
|---|---|---|---|---|---|---|---|---|
| Oracle (perfect labels, post-processing only) | 295 | 0.848 | 0.835 | 687 | 310 | 170 | 53 | 27 |
| Render (style-randomised, full pipeline) | 97 | 0.791 | 0.746 | 215 | 201 | 70 | 14 | 12 |

- Merges are open plan: in the oracle run kitchens are merged 320 times (recall 0.581) and corridors 297 times (recall 0.687), mostly with LIVING_DINING, CORRIDOR and ROOM.
- Splits are stairs and corridors: STAIRCASE split 88 times (recall 0.661; 0.517 in the render run), CORRIDOR 68 times.
- "Outside building" are further wings dropped by the largest-component rule.
- Merges propagate: 861 door connections in the oracle run are wrong because a room on one side was merged.
- Wall centre lines are good where the mask is good: centre-line F 0.994 (oracle), 0.877 (render); wall pixel IoU 0.79 on renders.

---

## 2. Per-Paper Summary Tables

Columns: representation (walls · junctions · rooms · openings), how the graph is obtained, constraints, gap handling, rooms, openings, key figures (as printed), limitations.

### 2.1 Wall-Graph Methods on Raster Input

| Paper | Representation | Graph method | Constraints | Gap handling | Rooms | Openings | Key figures | Limitations |
|---|---|---|---|---|---|---|---|---|
| **Raster-to-Vector** (Liu 2017) | Walls = lines between wall junctions (no thickness); junctions typed I/L/T/X, "13 (= 4 + 4 + 4 + 1) types"; room type on each side of each wall; openings = lines between 4 opening-junction types; icons (incl. **stairs, column**) = axis-aligned boxes | CNN junction heatmaps (threshold 0.4, NMS) → axis-aligned candidates (10 px) → integer programme (Gurobi, "around 2s") → averaging, door-to-wall snap | One-hot semantics; junction degree = incident primitives; mutual exclusion within 10 px; loops for bedroom, bathroom, restroom, balcony, closet, pipe space, exterior ("allowing some walls that stick out"); opening on a parallel wall within 10 px | T junctions also hallucinated as two L and one X, IP picks one ("allow one mistake in the estimation of the degree"); no gap bridging | "all the closed polygons formed by walls"; multi-label polygons split "by either horizontal lines or a vertical lines" | Lines; "we move each door to align with its closest wall"; window if one side is outside | Wall-junction acc. 70.7 → 94.7 with IP (recall 95.1 → 91.7); opening acc. 67.9 → 91.9; room acc. 80.9 → 84.5, recall 78.5 → 88.4; loop constraints +2% room precision; 870 images, 256 × 256 | Manhattan only; "puts a wall which does not exist"; commercial solver; living room, kitchen, corridor not loop-constrained |
| **CubiCasa5K** (Kalervo 2019) | Wall polygons, width "by sampling along the wall lines and inspecting the intensity profile"; 21 heatmaps (wall junctions, icon corners, opening endpoints) | Pair junctions "vertically/horizontally aligned … with a joint facing each other" → wall skeleton "pruned based on the wall segmentation" | Axis alignment; opening endpoints inside the wall mask | None (missed junction → no polygon) | Cells from junction triplets spanning junction-free rectangles, merged if no separating wall and same label | Segment between aligned endpoints; "width of the opening is the same as the wall polygon"; endpoints outside the wall mask rejected | Room IoU (test) 57.5 raw → 49.3 polygonised; icons 55.7 → 41.6; on R2V data junction acc. 95.0 with TTA + IP | "if wall or icon junctions are missed or are not correctly located the polygons can not be created regardless the quality of the segmentation"; CC BY-NC-SA |
| **Lv 2021** | Walls "a line with a width"; doors, windows, doorways with start, end, thickness, "located on specific walls"; room polygons; centre-line graph (junctions, centre lines) | DeepLabv3+ masks + opening-endpoint heatmaps → room contours (DP, then optimisation of boundary, IoU by differentiable rendering, orthogonality) → centre-line graph initialised from the room polygons, junction optimisation and merging | Soft orthogonality and alignment; merge vertices closer than half the wall width; insert a vertex into an edge within half the wall width (T-split) | Inherited from segmentation (doorway is a boundary class) | Connected room-type regions → polygons; type refined by text and symbol votes | Endpoints from heatmaps; type by length after scale (door < 1.5 m single, …) | Wall-junction acc. 0.96, recall 0.94 (without orthogonality 0.67 / 0.71); opening 0.97 / 0.93; mIoU 0.85 with vectorisation | "For an open kitchen, the judgment of the room category will be wrong, because we assume that the room is separated by walls, doors, windows, and doorways"; "Curved walls cannot be vectorized well"; data not released |
| **Chen 2023** (GLSP) | Graph of endpoints and typed segments (wall, door, window): openings are edges of the same line graph; labels carry thickness; no rooms output | Junction heatmap (pixel resolution, offsets removed) → candidate segments (NSS/NDS suppression; "line segments of the convex hull are also added") → attention GNN classifies each candidate | Optional prior losses: penalise intersecting segments; penalise closed wall/window chains without a door | None | Count only: enclosed rooms 5.22 → 5.40 with prior losses | Typed segments between junctions | sAP8 wall 84.88, door 97.50, window 88.44; junction sAP8 89.38 | NDS: "performance is greatly reduced if many inclined walls appeared"; stairs and voids lumped into "Others" (2,871 of 2,053,535 rooms) |
| **Raster-to-Graph** (Hu 2024) | Nodes = wall junctions (13 R2V types) + room category in four quadrants; edges = wall segments (no thickness); no openings | Autoregressive Deformable-DETR: next-level nodes conditioned on image + drawn subgraph; edges from four-direction connection classes + line search (5 px tolerance) | Axis directions; probabilistic BFS sampling in training | None explicit | "all the shortest cycles in the structural graph"; category by node majority | Not modelled | Structure (all junctions and segments right) 67.0% vs R2V 14.8%, HEAT 64.4%; junction F1 98.0; region F1 95.6 (R2V 78.9); room F1 84.7 (R2V 69.8); 0.30 s | "Our method only recognizes axis-aligned walls"; plans with 10–50 nodes; 512 px; residential |
| **FloorplanVLM** (Liu 2026) | Wall (p_start, p_end, τ, κ); openings nested (c, ω, δ); room (label, cycle of wall indices) | Qwen2.5-VL-3B emits JSON, walls first; coordinates on a 1024 grid; SFT on 2M + 300K, then GRPO with validity, exterior-IoU and interior rewards | By serialisation ("each room definition is strictly grounded in the pre-defined wall skeleton"); watertightness reward | GRPO "forcing the model to 'snap' endpoints together" | Wall-ID cycles | Nested in walls | ρ_val 96.10%, IoU_ext 0.9252, IoU_room 0.8920, F1_room 0.8249, F1_op 0.7333; non-Manhattan 95.10% / 0.9027 / 0.8738 / 0.8101 / 0.6894; SFT only 90.20%; noisy data only 67.25% | About 3,000 tokens per plan; "quantizing coordinates to a fixed [0, 1024] integer grid"; no metric scale, swing, separation lines, stairs; proprietary data; 32 × H200 |
| **Zhang 2026** (readout) | Walls = centre-line segments with thickness and type; rooms = cycles of wall IDs with open edges; openings = intervals along a wall | One network, two readouts: emit (grammar-masked decoder, 32 bins, endpoints snapped to junction peaks within 0.06) or detect (junction peaks + centre-line chord coverage, no trained parameter) | Grammar mask (no duplicate IDs, no room referencing an unemitted wall); shared node instances → closure by construction | Chord accepted if centre-line coverage ≥ κ (0.5); fusion donors ink-gated (≥ 0.5 coverage within ± 3 px) | Faces of the planar graph (detect); or Raster2Seq rooms + reconciliation (weld 0.022, deduplicate, T-split) | Opening heatmap read separately; small door/window head (0.988 family accuracy) | CubiCasa5K: detect 0.818 / 0.787 wall F1 @.05 / @.015 vs emit 0.790 / 0.736; r_val 0.956 vs 0.196; rooms 0.591 (detect), 0.512 (emit), 0.784 (room-centric); fusion 0.853 / 0.811, edit cost 73.1 | 256 px; residential; readout thresholds need in-domain calibration (ResPlan zero-shot 0.621 frozen vs 0.677 calibrated); chaining collinear edges costs 22 pp; repository holds only a README (lit. review) |

### 2.2 Room-Polygon Decoders

| Paper | Representation | Graph method | Constraints | Gap handling | Rooms | Openings | Key figures | Limitations |
|---|---|---|---|---|---|---|---|---|
| **RoomFormer** (Yue 2023) | Rooms = ordered corner sequences; walls implicit: "the thickness of inner walls is implicitly provided by the distance between neighboring room polygons" | Two-level queries (rooms, corners), polygon matching | None ("without imposing any hard constraints") | n/a | Polygons | Doors/windows as 2-vertex "polygons" or a line decoder | Structured3D room F1 97.3; Structured3D → SceneCAD room IoU 74.0 (HEAT 52.5); door/window F1 81.7; CubiCasa5K room F1 83.5 (Raster2Seq's table) | 20 room queries (70 with openings); density maps; "we cannot predict towards which side a door opens and if it is a double door or a single door" |
| **PolyRoom** (Liu 2024) | Rooms = 40 uniformly sampled vertices with corner labels | Queries initialised from Mask2Former instance masks, refined per layer | Angle loss | n/a | Vertex selection by corner probability + angle threshold + Douglas–Peucker | Not modelled | Structured3D room / corner / angle F1 98.3 / 90.2 / 85.2; DP baseline angle F1 55.8 (Structured3D), 27.7 (SceneCAD); cross-data IoU 85.2; CubiCasa5K room F1 54.1 (Raster2Seq) | Misses rooms missing from the instance segmentation; overlooks thin walls; 20 rooms |
| **FRI-Net** (Xu 2024) | Room = union of convex primitives cut by predicted horizontal, vertical and diagonal lines (implicit occupancy) | Room-wise decoder (BSP-style) | Axis-aligned lines trained first | n/a | Zero level set → polygon | Supplementary only | Structured3D room F1 99.1, angle 86.9; cross-data IoU 80.6; without the staged training angle precision 89.6 → 68.2; CubiCasa5K room F1 77.1 (Raster2Seq) | "fails to accurately reconstruct the structure of the inner walls" |
| **Raster2Seq** (Phung 2026) | Labelled polygon sequences: rooms, doors and windows; no walls | Anchor-based autoregressive decoder, 256 px, 512 tokens | None ("does not directly enforce geometric constraints"); optional VLM refinement for shared edges | n/a | Polygons | Labelled polygons after the rooms | CubiCasa5K room F1 88.7 (RoomFormer 83.5), window & door 77.8 (78.5); R2G data room F1 97.0 (R2G 95.0); WAFFLE zero-shot IoU 73.9 (RoomFormer 60.5) | Cross-over windows and doors inside rooms; overlapping rooms; in Zhang 2026, 26.7% hard failures zero-shot on ResPlan |

### 2.3 Segmentation and Free-Space Room Methods

| Paper | Representation | Graph method | Constraints | Gap handling | Rooms | Openings | Key figures | Limitations |
|---|---|---|---|---|---|---|---|---|
| **Ahmed 2011 / 2012** | Wall polygons from thick and medium lines | Contours + polygonal approximation | – | Close gaps between short convex/concave wall edges within an empirical T_merge; convex hull + smearing for façade gaps; SURF door symbols close the rest | Connected components of the inverted image; multi-label rooms split horizontally or vertically between labels | Door symbols matched to gaps | Room detection rate 89% (85% with label splitting), recognition accuracy 79% (82%); 82.33% of labels correct (manual) | "only able to find the physical existent rooms"; thresholds empirical |
| **DeepFloorplan** (Zeng 2019) | Pixel classes: room boundary (wall, door, window) vs room type | Multi-task segmentation, boundary-guided attention | – | – | "connected regions bounded by the predicted room-boundary pixels", majority type | Boundary class only | mIoU R3D 0.63 (0.66 after post-processing), R2V 0.74 (0.76); handles non-uniform thickness and curved walls | Inside/outside confusion in "long and double-bended corridors"; compass icon read as wall; no vectors |
| **Egiazarian 2020** | Line segments and quadratic Béziers with width (stroke centre line + width) | U-Net cleaning, transformer per 64 px patch, optimisation, merging | – | Cleaning network inpaints gaps; merge "close and collinear enough" lines; cut dangling ends | – | – | Degraded scans IoU 79/82 vs 47 (CHD); cleaning IoU 92 vs 49 | No semantics (ink vectorisation); text removed beforehand |
| Chen 2024 (maps, supporting) | Edge probability map | U-Net edges + Meyer watershed | – | Watershed bridges small gaps but "cannot recover lost boundary fragments" | Closed shapes | – | PQ 46.7 watershed vs 41.2 connected components; 51.1 with augmentation; no topology loss beat BCE (47.1) | KIPPI judged heuristic and parameter-sensitive |

### 2.4 Vector-Input Methods

| Paper | Representation | Graph method | Constraints | Gap handling | Rooms | Openings | Key figures | Limitations |
|---|---|---|---|---|---|---|---|---|
| **VectorFloorSeg** (Yang 2023) | Primal graph of line endpoints and segments, including segments extended to intersection (flag c_e); dual graph of the partitioned regions | Two-stream GNN: segments classified as room boundary, regions as room types | – | "extending the line segments until intersection" (kinetic-style partition) | Faces of the arrangement, merged by label | – | CubiCasa5K test mIoU 62.49, room integrity 67.51 (OCRNet 57.13 / 41.89); R2V 81.38 / 86.20 (77.98 / 70.82) | Input walls from annotations; small regions mislabelled; "wall blocks and railings" confused; curves as polylines |
| **VectorGraphNET** (Carrara 2024) | PDF paths (via SVG) as graph nodes | Graph attention, path classification | – | – | "Room contour" is a path class (F1 0.90) | Door, window, opening path classes | TUM: load-bearing wall F1 0.95, non-load-bearing 0.96, partition 0.95, internal staircase 0.83; FloorPlanCAD F1 79.4 (SymPoint 86.8), wF1 89.0 | No centre lines or graph; proprietary data; no code |
| **CADSpotting** (Yang 2024) | Dense points on primitives; panoptic instances | Point transformer + sliding-window aggregation | – | "incomplete walls are automatically reconciled" by rasterising wall lines at 8K and taking connected components | Largest component = floor | Door pivot and orientation from "linear-arc spatial relationships" | Sliding windows +16.7 PQ over block partitioning on large office and campus drawings | Vector input; 3D in Blender, not BIM |
| **Talebi-Kalaleh & Mei 2026** | Deterministic primitives from PDF operators; walls as hatched or outlined bands; "the thinnest pair defines its centerline" | Ordered rules: grids → columns → walls before beams → topology | Bearing invariant (members bear on column, wall or beam); symbol consumption before chaining | "Fragments are chained across gaps up to 450 mm"; ends snap to a column within half its glyph extent + 600 mm, a wall within half its thickness + 600 mm, a beam within half its width + 600 mm | Slabs not from planar-graph faces: "such traversal produces one face per framing bay without distinguishing a slab from a light well" | Floor openings from X-marked rectangles or "stair and shaft boxes" | Scale within 0.086%; walls 1.000 / 1.000, columns 0.922 / 0.997, beams 0.886 / 0.990, openings 1.000 / 0.964 (recall / precision, held-out) | Generated framing plans; column recall 0.659 and 0.790 in families with wings rotated 15–27°; vector PDFs only; no code |

### 2.5 BIM, FE and Agent Pipelines

| Paper | Representation | Graph method | Constraints | Gap handling | Rooms | Openings | Key figures | Limitations |
|---|---|---|---|---|---|---|---|---|
| **Sketch2BIM** (Ratul 2025) | JSON walls (lines and arcs, thickness), openings "anchored to host walls", rooms; feet | GPT-5 agents; Hough + RANSAC; wall directions clustered with K ∈ [1, 8] by BIC/AIC; human feedback in natural language; Revit via RevitPythonShell | Merge if "orientation difference is ≤ 1°, endpoints are within 3.0 ft, perpendicular offset is ≤ 0.50 ft, and overlap is ≥ 60%"; stubs kept if ≥ 1.25 ft and turning ≥ 12°; validator checks connectivity | "Double-wall strokes that run parallel with a consistent gap are merged into a single wall centerline"; users "extend" and "connect" walls | "halfedge planar graph … Bounded faces are enumerated"; slivers under 2.0 ft merged | Host by "nearest adjacency and tangent consistency"; ≥ 0.75 ft from wall ends, ≥ 0.50 ft apart | Wall recall initially above 0.845; F1 near 1.0 after 3–4 feedback rounds; initial wall length RMSE 2.34 ft, MAE 0.620 ft; window MAE 1.301 ft | Ten sketches; no stairs; one slab for non-Manhattan plans; two initial extractions rerun |
| **Liang 2026** | IFC walls (baseline start, length), openings (base point) | OCR-guided denoising → U-Net wall/opening masks → metric grid from dimension text → IFC entities | Grid from dimension chains, x and y scaled separately | Not described | Region branch trained, not exported | Inferred "from raster geometry and host-wall context" | OCR 90.2% (85.7% on ResBIM); wall/opening IoU 92.1%; IFC count error 2.45%; dimension precision 82.4%; strict precision 71.1%; 62 of 200 plans fully correct | Drafting-convention assumptions; degrades below about 50% OCR accuracy; one benchmark |
| **BlueprintAgent** (Xu 2026) | Axis grid as skeleton; columns at grid intersections; beams as node sequences | MLLM reads, OCR/CV supply evidence; validators emit conflicts that trigger local re-reading | Axis, geometric-topological, annotation, 3D constraints | Targeted revisits of conflict regions | – | – | Beam F1 0.994 vs 0.301 (MLLM zero-shot) and 0.820 (fixed pipeline); same validators applied after the fact: 0.8505 vs 0.9609 | RC frames only; OpenSees, not IFC |

### 2.6 Datasets That Define the Target Representation

| Dataset | Walls | Rooms | Openings | Stairs, voids | Graph | Checks and limits |
|---|---|---|---|---|---|---|
| **MSD / Swiss Dwellings** (van Engelenburg 2024) | Polygons; separators: wall, railing, column; structure = walls thicker than the plan's 60% thickness quantile + columns | Area polygons with types (incl. staircase, shaft, void, air, lightwell) | Polygons: door, entrance door, window | `stairs` is a **feature**, `staircase` an **area**; MSD removes features | "passage" if area polygons ≤ 0.04 m apart; "door" if a door polygon is ≤ 0.05 m from both areas | MSD thins the structure mask into lines for conditioning |
| **ResPlan** (Abouagour 2025) | Polygons with one `wall_depth` per plan (median 21 cm) | Polygons, 17 categories incl. `stair` | Door and window bodies fill the gaps in the wall band (Zhang's description); post-processing "aligns openings to the nearest wall band" | `stair` as a space category | via_door, adjacency, direct, via_window | Doors on a wall band in 99.94%, connecting ≥ 2 rooms in 99.34%; windows 99.98%; summed room area within ± 25% of the listed gross area in 98.7%; wall thickness normalised per plan |

### 2.7 Papers Not Available Locally (†, second-hand)

| Paper | What the repository says |
|---|---|
| **SALI-FP** (Chen et al. 2026, arXiv 2609.25615)† | Evidence-gated edits, in the missing-papers report's words: "an automatic fix is accepted only if source ink supports it, under explicit thresholds, with logs"; quoted from the paper: "source-only affine registration to align predictions with linework"; production audit of 11,534 plans (752,510 valid polygon-bearing objects); CubiCasa5K mIoU 0.3596, room F1 0.445; no code, data restricted |
| **KIPPI** (Bauchet & Lafarge 2018)† | Kinetic extension of detected segments "until they meet each other" into a polygon partition; Chen 2024 calls such techniques heuristic and parameter-sensitive |
| **Wu, Meng, Liu 2018**† | Rule-based rooms from DXF "organized by layers" with doors and windows as blocks; "restore wall lines cut by openings in three cases to close rooms"; 3D building model |
| **Chen & Tu 2025**† | Closed-face extraction from a CAD layer into FloorspaceJS space objects for energy simulation; "equivalent to existing rules" |
| **Chang et al. 2025**† | Keypoint detection of walls, doors and windows with rule-based filtering; "over 87% precision, 88% recall" on more than 5,000 house plans; WebGL 3D viewer |
| **Lin & Wang 2026**† | Scanned historical drawings to editable CAD: dimension, endpoint and grid detection (mAP50 0.98–0.99), rules rebuild the line topology "snapped to a 50 mm module"; one worked example |

---

## 3. Representations (Question 1)

### 3.1 How Each Method Represents the Four Elements

| Method | Walls | Junctions | Rooms | Openings | Stairs, columns, voids |
|---|---|---|---|---|---|
| Raster-to-Vector | Lines, no thickness, Manhattan | Typed I/L/T/X, 13 orientations | Closed wall loops, type per wall side | Lines on walls | Boxes (icons) |
| CubiCasa5K | Polygons with width | Typed heatmaps | Merged grid cells | Segments inside wall polygons | Icon boxes |
| Lv 2021 | Centre line + width | Graph nodes (from optimisation) | Polygons from masks | On walls, start/end/thickness | – |
| Chen 2023 | Typed segments | Untyped junction peaks | – | Typed segments (collinear with walls) | "Others" room type |
| Raster-to-Graph | Graph edges, no thickness | Typed, room types per quadrant | Shortest cycles | – | – |
| FloorplanVLM | Start, end, τ, κ | Shared endpoint coordinates | Cycles of wall IDs | Nested: class, ω, δ | – |
| Zhang 2026 | Centre line, thickness, type | Shared node instances | Cycles of wall IDs + open edges | Intervals on walls | – |
| Sketch2BIM | Lines and arcs, thickness | Connectivity checked | Half-edge faces | Hosted, projected, clearances | Not supported |
| RoomFormer, PolyRoom, FRI-Net, Raster2Seq | Implicit (gap between rooms) | – | Polygons | Lines or polygons, unhosted | – |
| DeepFloorplan, Ahmed | Pixel mask / polygons | – | Free-space regions | Boundary pixels | – |
| VectorFloorSeg | Input lines + extensions | Line endpoints | Arrangement faces | – | Railings confused with walls |
| MSD, ResPlan | Polygons | – | Polygons | Polygons in the wall band | SD: stairs feature + staircase area; ResPlan: stair space |
| Liang 2026 | IFC baseline + length | – | – | IFC base point | – |
| **Pilot v2** | Skeleton centre lines + thickness (not used by rooms) | Skeleton junction clusters | Free-space regions | Host = nearest segment, width, sides | Flight polygons + convex hull; voids from labels |

### 3.2 BIM-Readiness

Criteria (practice): **W** parametric walls (axis, thickness, optional curvature), **J** shared junction topology, **O** openings hosted by a wall with offset and width, **R** rooms as faces or cycles of wall (and separation) edges, **M** metric units, **S** door hinge and swing, **X** stairs, voids and columns as objects.

| Method | W | J | O | R | M | S | X |
|---|---|---|---|---|---|---|---|
| FloorplanVLM | ✓ (+ κ) | ~ (coordinates) | ✓ | ✓ (no open edges) | – | – | – |
| Zhang 2026 wall-first | ✓ | ✓ | ✓ | ✓ + open edges | – | – | – |
| Sketch2BIM | ✓ (+ arcs) | ✓ | ✓ + clearances | ✓ faces | ✓ (feet) | – | – |
| Lv 2021 | ✓ | ✓ | ✓ | polygons | ✓ (scale) | – | – |
| Raster-to-Vector | lines | ✓ typed | ✓ | ✓ loops | – | – | boxes |
| Raster-to-Graph | lines | ✓ typed | – | ✓ cycles | – | – | – |
| Liang 2026 | ✓ (IFC) | – | ✓ (IFC) | – | ✓ | – | – |
| Pilot v2 | ✓ | ~ | ✓ (nearest) | regions | ✓ | – | partial |

- **No published method meets all seven.** The best matches are Zhang's grammar (W, J, O, R with open edges) and Sketch2BIM (W, J, O, R, M, and Revit export, but hand-drawn input and human feedback). FloorplanVLM's statement that defining "a shared wall skeleton first" lets it "enforce topological consistency by design" is about the serialisation, not about detection quality (see §6.2).
- **Typed I/L/T/X junctions exist only in Manhattan methods.** Methods that handle oblique walls use untyped junctions plus edge evidence (Zhang, Chen 2023) or continuous endpoints (FloorplanVLM, Sketch2BIM). For BBL's oblique and curved walls, junction *type* should be replaced by junction *degree and edge directions*.
- **Swing and hinge are not output by any method** except CADSpotting's door pivot on vector input; RoomFormer states it cannot.
- **Stairs, voids and columns as non-bounding objects** appear only implicitly: Raster-to-Vector's icon boxes, Swiss Dwellings' stairs feature, MSD's columns appended to the structure.

---

## 4. Getting the Graph from an Image (Question 2)

### 4.1 Families

1. **Junction heatmaps + integer programming** (Raster-to-Vector; CubiCasa5K with IP). Candidates are junction pairs aligned within 10 px; the IP enforces degrees, exclusion, loops and opening hosting. IP is what turns 70.7% junction precision into 94.7%. Gurobi ("around 2s"), Manhattan candidates, 256 px.
2. **Autoregressive graph decoding** (Raster-to-Graph). Nodes predicted level by level, conditioned on the drawn subgraph; edges by four-direction classes and a 5 px line search. Best structure rate (67.0%), but axis-aligned only and 10–50 nodes.
3. **Segment classification over candidates** (Chen 2023; HEAT as described by Raster-to-Graph and RoomFormer). Junction peaks, all plausible pairs, a GNN or edge classifier decides. Oblique walls survive only if candidate generation keeps them (convex-hull segments).
4. **Emit coordinates vs detect and assemble** (Zhang 2026). Same network: emitted walls (snapped to junction peaks within 0.06) vs a training-free readout: nodes = junction peaks (θ_J), edge (u, v) accepted if the centre-line heatmap covers ≥ κ of the chord (θ_C), no third node near the chord, chaining off. Rule: "detect on large real scans and decode on small ones; on clean renders, decode where the decoder has trained on the style, otherwise detect with in-domain-calibrated thresholds".
5. **Full-sequence emission by a VLM** (FloorplanVLM). Requires 2M pairs, 300K pixel-aligned pairs and RL to reach 96.10% validity; a frontier VLM zero-shot is "roughly right" in topology but coarse (Zhang: wall F1 0.811 at 0.05, 0.472 at 0.015).
6. **Skeletonisation** (pilot v2; MSD's thinning of the structure mask; Zhang's ground-truth conversion: wall and opening bodies "united into one continuous band", rasterised, "closed morphologically with a kernel of about half the wall depth, skeletonized, and traced into a graph", Douglas–Peucker, welded, snapped to axes within 0.35 of the wall depth; converter watertightness 0.995). On CubiCasa5K, skeleton centre lines "carried spurs and segments without ink under them"; spur pruning and an ink gate fixed them (skv4).
7. **Merging double strokes into centre lines** (Sketch2BIM: parallel strokes "with a consistent gap" merged under ≤ 1°, ≤ 0.50 ft offset, ≥ 60% overlap; Talebi-Kalaleh & Mei: "the thinnest pair defines its centerline"; Zhang's ink gate looks ± 3 px around the chord "which admits hollow and double-line wall styles whose centerline carries no ink").
8. **Primitive fitting** (Egiazarian: lines and Béziers with width, merged when collinear, dangling ends cut; Sketch2BIM: Hough + RANSAC, arcs by the sagitta method with error ≤ 0.10 ft; Lv: Douglas–Peucker then optimisation).
9. **Kinetic extension / line arrangement** (KIPPI†; VectorFloorSeg extends lines to intersection and lets a classifier decide which extensions are boundaries; Talebi-Kalaleh & Mei chain across 450 mm and snap ends to the next support).
10. **Regions → graph reconciliation** (Lv 2021: room polygons first, then a centre-line graph with vertices merged and T-junctions inserted within half the wall width; Zhang: weld 0.022 of the frame, deduplicate shared edges, T-split, drop edges < 0.002; "close to lossless" for room-bounding walls, 7.9 → 0 double walls per plan).

### 4.2 Constraints and Solvers

| Constraint | Where | How enforced |
|---|---|---|
| Junction degree = incident walls | Raster-to-Vector | IP equality (Gurobi) |
| Mutual exclusion of near-duplicate primitives (10 px) | Raster-to-Vector; Zhang (double-wall test: sine of angle ≤ 0.2, separation ≤ 28 units, overlap > 40%) | IP inequality; deterministic deduplication |
| Closed loops | Raster-to-Vector (selected room types, via local constraints at junctions); FloorplanVLM (validity reward); Chen 2023 (loss penalty) | IP; GRPO; training loss |
| Manhattan | Raster-to-Vector, CubiCasa5K, Raster-to-Graph (hard); Lv (soft orthogonality, 0.96 vs 0.67 junction accuracy without it); FRI-Net (axis lines first); Sketch2BIM (orientation clusters, not 0/90°) | Candidate generation; loss terms |
| Openings on walls | Raster-to-Vector (parallel, ≤ 10 px); CubiCasa5K (inside the wall mask); FloorplanVLM (nesting); Sketch2BIM (≥ 0.75 ft from ends); ResPlan (snap to the band) | IP; rejection; serialisation; validator |
| No duplicate IDs, rooms reference existing walls | Zhang | Grammar mask at every decoding step |
| Support (bearing) | Talebi-Kalaleh & Mei | Reject members unsupported at both ends |
| Engineering validators as triggers | BlueprintAgent (validators → local re-reading: 0.9609 vs 0.8505 post-hoc); Talebi (constructive edits need ink: unmarked span ≥ 0.60 axis coverage); SALI-FP† (fixes gated on ink) | Agent loop; admission tests |

---

## 5. Robustness (Question 3)

### 5.1 Missing Wall Pieces and Gaps

- **Coverage instead of connectivity.** Zhang's readout accepts a wall when the centre-line heatmap covers ≥ κ = 0.5 of the chord between two junction peaks: a gap shorter than half the wall is bridged, and the result is closed "by construction" (watertightness 0.956 on CubiCasa5K). The price: thresholds must be calibrated per domain (0.621 frozen vs 0.677 calibrated, ResPlan zero-shot), and chaining collinear edges costs 22 pp against split ground truth.
- **Extension to the next support.** Talebi-Kalaleh & Mei chain fragments over gaps up to 450 mm and snap free ends to a column, wall or beam within a fixed reach (half the target + 600 mm); KIPPI† extends segments "until they meet each other"; VectorFloorSeg extends all lines and classifies which extensions are room boundaries; Sketch2BIM merges collinear pieces whose ends are within 3.0 ft.
- **Constraints that make gaps visible.** A degree-1 junction is an anomaly in a closed plan; Raster-to-Vector's degree and loop constraints, Chen 2023's prior losses, and FloorplanVLM's validity reward all push toward closure. The IP needs candidates to exist: it can choose, not invent.
- **Pixel-level closing** (Ahmed T_merge; pilot slit closing 0.3 m; watershed in Chen 2024 and pilot sealing) works for short gaps but the watershed "cannot recover lost boundary fragments".
- **Training-side remedies.** Egiazarian's cleaning network inpaints line gaps; TopoMortar and Chen 2024 disagree on topology losses (lit. review: clDice best on bricks, no loss beats BCE on maps).
- **Validators that re-read.** BlueprintAgent: constraint failures as triggers for local re-reading beat the same constraints applied after the fact (0.9609 vs 0.8505 beam F1).

### 5.2 Non-Manhattan and Curved Walls

- Fail or excluded: Raster-to-Vector ("unable to detect walls which are neither horizontal nor vertical"), Raster-to-Graph ("only recognizes axis-aligned walls"), CubiCasa5K, Chen 2023 under NDS.
- Handled: FloorplanVLM (κ per wall; non-Manhattan subset 95.10% validity; 42.7% non-Manhattan training data), Sketch2BIM (arcs, orientation clusters), Lv (soft orthogonality; curved fails), DeepFloorplan (masks; curved walls fine as pixels), Egiazarian (Béziers).
- Partly: Talebi-Kalaleh & Mei lose columns and beams in wings rotated 15–27° (column recall 0.659 mixed, 0.790 skew-wing).
- Implication: snap per dominant direction (several directions), keep curvature as a wall parameter, fit arcs where the residual is large.

### 5.3 Thick Historical, Poché, Hollow and Double-Line Walls

- Versailles-FP: filled vs hollow walls; its label generator reaches Dice 90.75% on filled but 34.45% on hollow walls of CVC-FP (whose ground truth fills hollow walls).
- Double lines → one centre line: Sketch2BIM, Talebi-Kalaleh & Mei, Zhang's ink gate; Chen 2023 augments with "hollow lines to represent walls".
- Centre lines are ill-defined for poché masses, niches and pilasters (no paper addresses this; pilot: monumental plans wall IoU 0.05–0.11). For such walls the room face should come from pixels, not from centre line ± t/2 (§8).

### 5.4 Columns

- Objects: Raster-to-Vector icon "column"; MSD appends columns to the structure; Talebi-Kalaleh & Mei detect glyphs 60–2,000 mm with aspect ≤ 6 (recall 0.922, precision 0.997); VectorGraphNET column F1 0.96 (TUM, vector).
- Pilot: column IoU 0.09 zero-shot on CubiCasa5K, 0.4657 on renders (harness).
- In a BIM-style model a column touching walls bounds rooms, a free-standing column is a hole in the room (practice); both need the column as an object, not as wall.

### 5.5 Large Sheets and Many Rooms

- Caps: Raster-to-Graph 10–50 nodes; RoomFormer 20 room queries, RoomFormer and FRI-Net drop beyond 15 polygons or 150 corners (Raster2Seq); Raster2Seq 512 tokens; Zhang's benchmark rejects plans over "512 tokens or 80 walls"; FloorplanVLM about 3,000 tokens.
- Detection scales: Zhang's readout gains with plan size (−3.7 pp on 5–22 walls, +2.4 on 23–31, +8.7 on 32–70) while decoder F1 falls with length; CADSpotting's sliding windows +16.7 PQ on 1,000 m² office and campus drawings.
- Pilot regions have no cap (lit. review: more than 60 rooms on public historical plans).

### 5.6 Reported Failure Figures

| Measure | Value | Source |
|---|---|---|
| Plans with all junctions and segments right | 67.0% (R2G), 64.4% (HEAT), 14.8% (R2V) | Raster-to-Graph Table 2 |
| Junction precision without / with IP | 70.7 / 94.7 | Raster-to-Vector |
| Watertight plans: detect / emit / room-centric | 0.956 / 0.196 / 0.916 | Zhang Table 2 |
| Room F1: faces of detected graph / emitted / room-centric | 0.591 / 0.512 / 0.784 | Zhang Table 2 |
| Recall of walls that bound no room: room-centric / wall-first | 0.302 / 0.663 | Zhang Table 5 |
| Room IoU raw / polygonised | 57.5 / 49.3 | CubiCasa5K |
| IoU vs IFC strict precision on one plan | 90.92% IoU; 6/8 walls dimensionally right, 5/8 right in place and size | Liang 2026 |
| Room F1, Structured3D → CubiCasa5K | PolyRoom 98.3 → 54.1; FRI-Net 99.1 → 77.1 | PolyRoom; FRI-Net; Raster2Seq |

---

## 6. Where Rooms Come From (Question 4)

### 6.1 Evidence by Approach

| Approach | Evidence for | Evidence against |
|---|---|---|
| **Faces of the wall graph** | Raster-to-Graph structure 67.0% vs 14.8%; Sketch2BIM all ten plans valid (after feedback); VectorFloorSeg faces + boundary classification: room integrity 67.51 vs 41.89; Raster-to-Vector room recall 78.5 → 88.4 with IP | One missed wall merges two faces, one spurious wall splits one (global). Zhang: faces 0.591 vs 0.784; "deriving rooms from walls is global … and loses information"; the snap radius that closes rooms costs about 3 pp wall F1. CubiCasa5K polygonisation 57.5 → 49.3. Talebi-Kalaleh & Mei avoid face traversal for slabs |
| **Free-space regions** | DeepFloorplan, Ahmed; pilot S1 15/15 rooms, mean IoU 0.89, median area error 2.7%; harness recall 0.848 (perfect labels) and 0.791 (renders); no room cap | Cannot split what no barrier separates (open plan: 687 merges); one gap > 0.3 m merges rooms; Chen 2024: watershed "cannot recover lost boundary fragments" |
| **Learned room polygons** | Raster2Seq room F1 88.7 on CubiCasa5K, WAFFLE zero-shot IoU 73.9; Zhang: room-centric + reconciliation best rooms (0.784) and lowest single-system edit cost | No walls that bound no room (recall 0.302); gaps and overlaps need reconciliation (7.9 double walls per plan without it); caps on rooms and tokens; cross-domain hard failures (26.7%); trained on non-commercial data |

### 6.2 What the Controlled Comparison Says

Zhang 2026 is the only matched comparison. Its conclusions bear directly on the proposal:
- "With matched data and recipe, a room-centric system with a reconciliation step and a wall-first sequence model reach comparable wall quality, so the output representation matters less than is usually assumed."
- The asymmetry: "Deriving walls from rooms is a local operation and close to lossless … whereas deriving rooms from walls is global, since every cycle has to close, and loses information."
- The remaining difference is coverage: room polygons cannot carry dangling, stub and exterior walls.
- Output fusion helps (+7.1 pp), input conditioning does not (0 of 3 forms). The gain is mostly ensembling with a small cross-representation part (about 1 pp); "the high-precision reconciled output is the appropriate base and the high-recall detector the appropriate donor". A donor helps exactly when its purity exceeds half the base F1.

For us: region rooms are the high-precision room-centric base; a detected wall graph is the donor for non-bounding walls and for gap closing. The two fail differently, which is what fusion needs.

### 6.3 Open Plan

- **Explicit open edges:** Zhang's rooms carry edges "marked open (no wall)", watertightness allows them, and its edit cost charges a CONVERT (weight 10, the highest) for a phantom wall over an open boundary. Its synthetic open-plan tier leaves kitchen/living boundaries without walls.
- **Boundary classification:** VectorFloorSeg extends wall lines to intersection and classifies each segment as room boundary or not ("a line segment from the input is highly likely to represent a wall … an extended line segment is only a possible room boundary").
- **Selective closure:** Raster-to-Vector does not force living rooms, kitchens and corridors into loops and splits multi-label polygons with axis-aligned lines; Ahmed 2012 splits by labels (detection 89% → 85%, recognition 79% → 82%). The literature review warns that such cuts invent walls.
- **Failures acknowledged:** Lv 2021 (open kitchen), Ahmed 2011 ("only able to find the physical existent rooms").
- **Ground truth exists:** MSD "passage" edges (area polygons ≤ 0.04 m apart); CubiCasa5K virtual lines; CVC-FP "Separation" lines (used in our harness).
- **In BIM practice** these are room-separation lines: zero-thickness, room-bounding, not walls (practice).

### 6.4 Stairs, Voids and Landings

- **As objects:** Raster-to-Vector (stairs and column are icon boxes, a pixel can be "both a bathroom … and a bathtub icon"); FloorPlanCAD and VectorGraphNET (stair classes: internal staircase F1 0.83); Swiss Dwellings (`stairs` feature inside a `staircase` area).
- **As rooms:** ResPlan `stair` space; Chen 2023 "Others" (0.14% of rooms).
- **As slab openings:** Talebi-Kalaleh & Mei derive floor openings from "stair and shaft boxes clipped to the grid lattice".
- **Not at all:** Sketch2BIM ("does not include functions for generating architectural features such as staircases"), MSD (removes features).
- **Landings and voids:** no method. Pilot: the convex hull "swallows landings and corridor next to L-shaped or spiral stairs" (review §4.4); harness: staircase areas split 88 times (perfect labels).
- **BIM practice:** a stair (flights + landings) is not room-bounding; the stairwell room exists when walls or separation lines enclose it; the void is a slab opening deducted from area (practice; CAD-Richtlinie: voids over 5 m² cut out of the GF).

### 6.5 Stamps as Seeds and Checks

- Only Ahmed 2012 uses labels geometrically (split rooms with several function labels). Lv 2021 votes room types with text inside the polygon. AxcelerateAI's page says "room tags seed room polygons" (practitioner, unverified).
- In BIM practice a room is *placed* at a point and takes the enclosing face (practice). The stamp is that point: each stamp should fall into exactly one face; two stamps in one face point to a missing wall or separation; a stamp in an unclosed region is the "not enclosed" case. The stamp area against the face area tells merges (face ≈ sum of two stamps) from stale stamps.

---

## 7. Openings on the Centre Line (Question 5)

| Method | Parametrisation | Snapping to the host | Swing, type, connectivity |
|---|---|---|---|
| FloorplanVLM | (class, width ω, centre offset δ along the centre line), nested in the wall | By construction | – |
| Zhang 2026 | Door or window interval along a wall | Ground truth: "projected onto the nearest centerline as intervals" | Family from a 200K-parameter head (0.988) |
| Sketch2BIM | Hosted, span along the wall; default door 3.0 ft (2.5–3.5), window 4.5 ft (4.0–5.0) | Host by "nearest adjacency and tangent consistency"; projected orthogonally (lines) or as tangent chords (arcs); ≥ 0.75 ft from wall ends, ≥ 0.50 ft apart, no crossing of wall vertices; violations trimmed or relocated | Swing arcs used to detect, not stored |
| Raster-to-Vector | Line between opening junctions | Must lie on a wall "parallel to the opening with distance smaller than 10 pixels" (IP); then moved to the closest wall | Window if one side is outside |
| CubiCasa5K | Segment between aligned endpoints, width = wall width | Endpoints outside the wall mask rejected | Class from segmentation |
| Lv 2021 | Endpoints (heatmaps), thickness per wall part | Via segmentation | Type by length (door < 1.5 m, …; window < 0.8 m small, …) |
| Chen 2023 | Typed segments in the junction graph | Collinear by construction | – |
| ResPlan | Polygons filling the wall-band gap | "aligns openings to the nearest wall band": 99.94% (doors), 99.98% (windows) | via_door / via_window edges; doors connect ≥ 2 rooms in 99.34% |
| MSD | Polygons | – | Door edge if the door polygon is ≤ 0.05 m from both areas |
| RoomFormer | 2-vertex line | None | Cannot tell swing or single vs double |
| CADSpotting | Instance | – | Pivot and orientation from arc–line geometry |
| Raster2Seq | Polygon after the rooms | None ("cross-over windows") | – |
| Liang 2026 | IFC opening base point | "from raster geometry and host-wall context" | 20 mm evaluation tolerance |
| Pilot v2 | Centre, direction, width, depth | Host = nearest wall segment within a distance; sides sampled at depth/2 + offset | Exterior if one side lies outside the building |

- **Detection beats emission for openings** (Zhang: "a local detection problem rather than a sequence-emission problem"; 0.246 emitted vs 0.644 read from the heatmap, 0.799 ignoring the family).
- **Connectivity becomes topological.** With rooms as faces of the half-edge graph, a door on wall edge e at offset s connects the two faces on either side of e at s. No ray casting, no nearest-centroid rule (the literature review notes Ayanzadeh & Oates' nearest-centroid linking is fragile in corridors).
- **Swing and hinge are open in the literature.** The renderer draws swings, so it can label hinge and swing side; CADSpotting shows the geometry (pivot from arc and leaf).

---

## 8. Areas (Question 6)

- **Almost no paper measures room area.** ResPlan checks summed room areas against the listed gross area (± 25% for 98.7% of plans), a consistency check; Liang evaluates IFC dimensions (1 mm walls, 20 mm openings) but not areas; the literature review notes Guo et al. report one apartment's total within 0.81%.
- **Thickness and area in the representations.** RoomFormer: room polygons with gaps carry wall thickness implicitly; a planar graph (HEAT) "can approximate the true floorplan only up to the thickness of the walls". Zhang's reconciliation assigns "a constant thickness of 0.012 of the frame, since room polygons carry no thickness information". Lv keeps both: room contours (for area) and centre lines (for 3D).
- **Net area from the graph** (SIA 416 net = inner wall faces): offset each face inwards by t/2 per wall edge and 0 per separation edge (mitred at corners). In Revit terms, room area "at wall finish" (practice). Error budget: a thickness error δt on every edge changes the area by about perimeter × δt/2; for a 4 × 4 m room, 2 cm of thickness error is about 0.5% of the area.
- **Where centre line ± t/2 fails:** varying thickness, niches, radiator recesses, pilasters, poché masses. There the face should be snapped to the pixel wall face (the region approach's strength), with the graph carrying only topology.
- **Our evidence:** pilot v1, 10 cm snap to the wall face: median error 14.7% → 3.8%; pilot v2 region rooms: 2.7% (S1); harness median absolute error 2.07% (0.159 m²) with perfect labels, 2.95% (0.200 m²) on renders.

---

## 9. From 2D to BIM, IFC or FE Models (Question 7)

| Paper | Output | Representation used | Manual steps |
|---|---|---|---|
| Sketch2BIM | Revit model via RevitPythonShell: walls, doors, windows (generic families), one floor slab | JSON walls, hosted openings, half-edge rooms | 3–4 natural-language feedback rounds per plan (eight steps on plan 10); human validation of every model |
| Liang 2026 | IFC entities | Wall baseline and length, opening base point, metric grid from dimensions | None reported; about 6.5 minutes per plan |
| BlueprintAgent | Structural JSON + OpenSees .tcl (FE, not IFC) | Axis grid, columns, beams, sections, floors | Engineer review required; uncertain items flagged |
| Talebi-Kalaleh & Mei | Editable layout + 3D FE draft (walls as area elements) | Primitive geometry, bearing topology | Engineering review; guarded VLM edits |
| CADSpotting | 3D interior in Blender | Wall polygons, door pivots, window lines | – |
| Raster-to-Vector, Raster-to-Graph, DeepFloorplan, Lv | 3D pop-up models | Walls extruded, openings textured | – |
| Chang et al.† | WebGL 3D viewer | Keypoints → walls, doors, windows | – |
| Lin & Wang† | Editable CAD | Lines snapped to a 50 mm module | – |
| Wu et al. 2018† | 3D building model | Rule-based rooms from layered DXF | Requires layered DXF with block openings |
| Chen & Tu 2025† | FloorspaceJS space objects | Faces from closed curves in a CAD layer | – |
| FloorplanVLM | "render-ready JSON" | Wall-first JSON | – (no BIM export) |

**Mapping our model to IFC** (practice): wall → IfcWall with an Axis representation and a material layer set of the detected thickness; opening → IfcOpeningElement voiding the wall + IfcDoor/IfcWindow filling it at the offset; room → IfcSpace with its boundary (and space boundaries per wall edge); separation edges → space boundaries without a building element; stair → IfcStair (flights, landing); void → opening in the slab; column → IfcColumn. Every element keeps its pixel provenance and confidence.

---

## 10. Recommendation for Our Pipeline (Question 8)

### 10.1 Verdict on the Proposal

Adopt the BIM-style model as the **canonical representation and the export**, and use it to fix the stair problem and to close gaps at the graph level. Do not make the graph the **only** source of rooms: produce rooms by reconciling graph faces with the free-space regions, so that every final room is a face but a graph error never silently merges or splits a room. This is what the evidence supports:
- representation matters less than readout and reconciliation (Zhang §6.2);
- faces of a detected graph lose rooms on real scans (0.591 vs 0.784);
- fusion of two hypotheses with different errors helps (+7.1 pp), conditioning one on the other does not;
- pilot v2's regions are already a strong room-centric base (S1 15/15; harness recall 0.848 with perfect labels).

The proposal's real gains are specific: openings on centre lines close rooms at doors; stairs and fixtures are explicitly non-bounding; separation lines are first-class; door connections become topological; IFC follows directly. "Placing rooms like a BIM tool" is face lookup at a seed point. It inherits the BIM tool's failure (unclosed boundary), which is why the stamp checks and the region fallback stay.

### 10.2 Canonical Data Model

Extend `walls[]`, `openings[]`, `rooms[]` of pipeline.md §4 into one planar graph per drawing:
- **nodes:** id, position, degree, source, confidence;
- **edges:** id, node pair, kind (`wall`, `separation`, `railing`, `void-edge`), centre line (line or arc with curvature), thickness (walls), wall type (exterior, interior, load-bearing), source, evidence (heatmap coverage, ink coverage), confidence;
- **openings:** host edge, offset along the centre line, width, kind (door, window, passage, entrance), hinge side, swing side, the two faces it connects;
- **rooms:** face id, boundary half-edges (cycle of edge ids), polygon at inner wall faces (net), region IoU, stamps, area, flags;
- **objects (non-bounding):** stairs (outline of flights and landings, flights with direction, host face, void reference), columns (bounding if attached to walls, otherwise holes), fixtures;
- **voids:** polygon, bounding edges (railings), kind, deducted area, GF deduction (> 5 m²).

### 10.3 What Should Produce the Graph

| Source | Use | Evidence |
|---|---|---|
| **Learned junction + centre-line heatmaps, training-free readout** (primary for raster input) | Nodes from junction peaks with sub-pixel offsets; edges by chord coverage; thresholds calibrated per domain on the harness; chaining off; candidate pairs restricted to neighbours for office-size graphs | Zhang 2026 (detect > emit on real scans, closure by construction, scales with plan size); Pang et al. (sub-pixel offsets, annealed σ, lit. review); Chen 2023 (untyped junctions for oblique walls) |
| **Skeleton of the wall mask** (fallback, and today's baseline) | Until the heads exist; then as a second donor and as QA; spur pruning + ink gate | Pilot; Zhang's skv4 fixes; MSD thinning |
| **Region boundaries → edges** (always) | Where a region room is not a face: map its simplified boundary to existing wall edges (within t/2 + tolerance), add missing walls on wall-mask pixels and separation edges across free space, weld within half the wall thickness, T-split | Lv 2021; Zhang's reconciliation (7.9 → 0 double walls) |
| **Vector side channel** (CAD prints, vector PDFs, DWG) | Exact wall faces → centre lines (double strokes merged, thinnest pair); snapping of detected walls to vector segments that pass an ink check | Talebi-Kalaleh & Mei; Sketch2BIM; VectorFloorSeg; VectorGraphNET; pipeline.md stage 3 already plans snapping |

Not recommended: VLM or autoregressive emission of the graph (FloorplanVLM needs 2M + 300K training pairs and RL; frontier VLM zero-shot is coarse at 15 cm; decoders degrade with plan length). Typed I/L/T/X junctions (Manhattan only); predict degree and edge directions instead.

### 10.4 Closing Gaps Robustly

In order of preference, each step logged with its evidence (SALI-FP†-style gating):
1. **Train centre lines through openings.** Label the centre line of the union of wall and opening bodies (Zhang's ResPlan-FP recipe), so the head predicts a continuous wall across doors and windows; openings become intervals on it. Rooms then close at openings without virtual-wall heuristics.
2. **Coverage, not connectivity.** Accept a chord at κ coverage (start at Zhang's 0.5; sweep on the harness); short missing pieces inside a wall are bridged.
3. **Extend dangling ends.** For each degree-1 node, cast along the edge direction to the first wall within d_max (start between Talebi's 450 mm and Sketch2BIM's 3.0 ft, about 0.9 m; sweep). Accept if heatmap or ink coverage along the extension ≥ κ_gap, or if the extension splits a face holding two stamps into faces with one each. Otherwise keep the stub (exterior stubs and free-standing walls are real) and flag it.
4. **Check constraints and re-read.** Degree-1 nodes, faces with 0 or ≥ 2 stamps, stamp vs face area, rooms smaller than 0.25 m² and faces disagreeing with regions become conflicts that trigger a local re-segmentation at higher resolution (BlueprintAgent: triggers beat post-hoc filters, 0.9609 vs 0.8505). An IP over degrees and exclusion (Raster-to-Vector) is optional later, with an open-source solver.
5. **Keep pixel closing as the last resort**: slit closing and sealing stay in the region branch, flagged.

### 10.5 Keeping the Region Approach: Room-by-Room Reconciliation

1. Compute faces of the fused graph (polygonise the noded edges; drop the exterior face).
2. Match faces to region rooms one-to-one by IoU.
3. **Agree** (IoU above a threshold, e.g. 0.8; sweep): accept the face; area from §10.6.
4. **Several regions in one face** (graph missed a wall the pixels show): add the edges from the region boundaries (§10.3), recompute.
5. **Several faces in one region** (graph added a wall or separation the pixels do not show): keep the split only if the separating edge has wall-ink or separation-head support, or the stamps require it; otherwise merge.
6. **Region but no face** (graph did not close): derive the face from the region boundary; flag "closed from region".
7. **Neither closes** (leak): keep today's sealing, flag.
8. **Stamps:** each stamp in exactly one face; resolve or flag the rest (§6.5).

Every final room is a face with boundary edges, so connections, IFC and plan-check get a consistent topology, while each room keeps a provenance flag that shows which branch produced it.

### 10.6 Stairs, Voids, Columns and Fixtures

- **Delete the convex-hull split and keep stairs out of the barrier.** A stair is an object: outline of flights and landings, flights with direction, host face = the face it lies in.
- **A separate stair room exists only when room-bounding edges enclose it**: walls, railings around the void, or separation edges (learned, or required by a stamp such as "Treppe"). Landings belong to the stair object and hence to its host room.
- **Tread lines are stair, not wall.** Training: renderer v2 with more stair styles (stringers, spiral stairs, treads in outlined walls); post-processing fallback (ours): wall fragments fully inside the stair mask and parallel to the treads are relabelled stair and removed from the graph.
- **Voids:** a void class (Swiss Dwellings VOID, AIR, LIGHTWELL; the stair eye), railings as their bounding edges; voids are holes in faces, deducted from room area as the stamp convention requires (open question in pipeline.md §9) and from the GF above 5 m².
- **Columns:** bounding when they touch a wall, otherwise holes; never part of the wall mask.
- **Fixtures:** never in the barrier.

### 10.7 Model Heads and Training Labels

| Head | Label source (Swiss Dwellings + renderer) | Why |
|---|---|---|
| Wall body (existing) | Wall polygons | Thickness, pixel faces for area |
| **Centre line, continuous through openings** | Skeleton of wall ∪ door ∪ window polygons, closed by ½ wall depth, Douglas–Peucker, welded, snapped per dominant direction (Zhang's recipe); or per-wall minimum rotated rectangles (`synth.py` already computes centre, direction, length, thickness); IFC-Bench: IfcWall axes (`ifc_prepare.py` keeps only cut polygons today) | Graph readout; gap bridging |
| **Junctions with sub-pixel offset** (+ degree or direction bins) | Nodes of degree ≠ 2 and corners of the centre-line graph | Graph nodes for oblique walls |
| **Room-separation lines** | Shared boundaries of two areas (≤ 0.04 m, MSD's passage rule) minus walls, openings and railings buffered by t/2; evaluation also on CubiCasa5K virtual lines and CVC-FP "Separation" (non-commercial, benchmark only) | Open plan (687 of 1,247 oracle misses) |
| **Interior footprint** | Union of non-outdoor areas and walls | Building outline, further wings (310 "outside building" misses) |
| **Opening instances** with jambs, hinge, swing | Door/window polygons; the renderer knows hinge and swing because it draws them | Hosting, swing, connections; detector beats decoder |
| **Stair object** (flights + landings) | Swiss Dwellings stairs features (+ staircase area for landings, to verify); IFC IfcStair, IfcStairFlight, landing slabs | Non-bounding object |
| **Void, railing, column** | Swiss Dwellings VOID / AIR / LIGHTWELL areas, RAILING and COLUMN separators; renderer v2 void conventions | Holes, deductions, bounding edges |
| Text (planned) | Rendered stamps | Seeds and checks |

Resolution: 50 px/m makes an 8 cm partition 4 px wide; heatmaps with sub-pixel offsets keep junction precision, and the 100 px/m variant for drywall offices (review §3.4) applies. Topology losses (clDice, Skeleton Recall) on the centre-line channel are an A/B, scored by rooms (literature review item 10).

### 10.8 Evaluation and Experiments, in Order

1. **Oracle BIM test (no training).** From Swiss Dwellings perfect labels, build the centre-line graph (walls ∪ openings) and compute faces: (a) walls only, (b) plus separation edges from the area polygons. Compare with the region approach on the same 295 floors. (a) checks the label pipeline (watertightness, as Zhang's converter 0.995); (b) gives the ceiling the separation head can reach.
2. **Stairs as objects (no training).** Remove the hull split and stairs from the barrier; measure STAIRCASE and CORRIDOR splits and merges (oracle, render) and the S1 hall.
3. **Graph-level gap closing on today's masks.** Skeleton graph + extension rule + chord coverage on the wall probability + room-by-room reconciliation; measure on renders, CubiCasa5K, CVC-FP and S1–S3 with parameters frozen on the harness.
4. **Model v2 heads** (centre line, junctions, separation, interior, void, stair object, opening instances) on renderer v2; compare skeleton vs heatmap readout (Zhang's protocol), and fusion with region rooms, enabling the donor only where its measured purity exceeds half the base F1.
5. **IFC round trip** with IfcOpenShell and plan-check on the DXF.

Metrics: centre-line precision and recall within 5 cm (harness has them), wall F1 over a tolerance sweep, watertightness read with double-wall counts, the structure rate (all junctions and segments right), rooms one-to-one with miss reasons, area error in m², connectivity F1, stair-room correctness, and Zhang's edit cost including CONVERT.

### 10.9 Open Risks

- **Office and historical walls are not all straight bands.** Centre lines are ambiguous for poché masses, niches and pilasters; no paper evaluates this. Mitigation: area from pixel faces there, graph only for topology.
- **Separation lines are a convention.** Swiss Dwellings' area splits may not match BBL's AOID polygons; a learned head inherits the dataset's convention. Stamps and SAP room lists must arbitrate; CONVERT errors are the most expensive edits (Zhang).
- **Readout thresholds are domain-sensitive** (Zhang: 0.621 vs 0.677). They need a real calibration set (Landgut Lohn alone is too small; harness real benchmarks are non-commercial).
- **Global failure modes move, they do not vanish.** A spurious separation or wall splits a face; the reconciliation must default to the region when evidence is weak.
- **Non-Manhattan and curved walls** need curvature and multi-direction snapping; most evidence is residential and Manhattan.
- **Scale of the graph.** Office floors have hundreds of junctions; candidate pairs must be restricted (neighbours, directions) and the readout run on tiles with overlap (CADSpotting's sliding windows).
- **Swing and hinge** have no published method on raster plans; the instance head is new work.
- **Licences.** No model weights from these papers are usable; Zhang's code is announced but not released; Raster-to-Vector's solver is commercial. Everything is trained on our renders.
- **Evidence is thin where BBL differs most:** office floors, scans, stairs and voids, area in m². Our harness and the BBL gold set carry the burden of proof.

---

## 11. Notes on Repository Documents

- `docs/literature-review.md` stage 3 attributes "angle ≤ 1°, offset ≤ 0.5 ft" to Sketch2BIM's double-stroke merging. The paper states these tolerances for local merging ("orientation difference is ≤ 1°, endpoints are within 3.0 ft, perpendicular offset is ≤ 0.50 ft, and overlap is ≥ 60%"), then that parallel strokes "with a consistent gap" are merged into a centre line, without a separate gap tolerance.
- `research/papers.json` (BlueprintAgent) says "the paper states no data release"; the local copy's "Data availability" paragraph says the public release contains the 300 anonymised sheets, ground-truth JSON, OpenSees models and evaluation code.
- `docs/pipeline.md` stage 6 ("rooms are the faces of the planar wall graph") and review §3.2 ("keep the region approach") are reconciled by §10.5: faces in the output, regions as the base hypothesis.

---

## 12. Sources

Local papers read (`research/papers-md/`): 2017-liu-raster-to-vector, 2019-kalervo-cubicasa5k, 2021-lv-residential-floor-plan-recognition, 2023-chen-line-segments-gnn, 2024-hu-raster-to-graph, 2026-liu-floorplanvlm, 2026-zhang-readout-vectorization, 2022-yue-roomformer, 2024-liu-polyroom, 2024-xu-fri-net, 2026-phung-raster2seq, 2011-ahmed-improved-analysis, 2012-ahmed-room-detection-labeling, 2019-zeng-deepfloorplan, 2020-egiazarian-deep-vectorization, 2023-yang-vectorfloorseg, 2024-carrara-vectorgraphnet, 2024-yang-cadspotting, 2026-talebi-kalaleh-framing-plan-parsing, 2025-ratul-sketch2bim, 2026-liang-dimension-aware-bim, 2026-xu-blueprintagent, 2024-van-engelenburg-msd, 2025-abouagour-resplan; targeted passages of 2024-chen-historical-map-vectorization, 2021-swaileh-versailles-fp, 2024-pang-pixel-wise-symbol-spotting, 2025-valverde-topomortar, 2025-rodionov-floorplanqa, 2025-luo-archcad-400k, 2026-ayanzadeh-llm-agentic-parsing, 2021-fan-floorplancad.

Repository documents: `docs/pipeline.md` (§4, stages 3–6), `docs/reviews/2026-10-07-pipeline-and-pilot-v2.md` (§2, §3.2, §4.4, §8.1), `docs/literature-review.md` (§2, stages 3–6, §9, §10), `docs/reports/2026-10-07-pipeline-inputs-masking-scale-evidence.md` (§1.2, §6), `docs/reports/2026-10-07-missing-papers.md` (§4.3, §6), `research/papers.json`.

Pilot: `pilot/v2-pipeline/fpx/{walls,openings,stairs,rooms,config}.py`, `sd_prepare.py`, `ifc_prepare.py`, `harness.py`, `data/harness/oracle.json` and `render.json` (7 October 2026).
