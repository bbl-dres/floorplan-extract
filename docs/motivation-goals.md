# Motivation and Goals

*BBL · October 2026 · Why BBL explores computer-vision extraction of floor plan data, what it wants to achieve, what has been learned so far, and the recommended route. Bracketed numbers refer to the shared [source list](../research/sources.md).*

## 1. Background

The Federal Office for Buildings and Logistics (BBL) manages the real estate of the Swiss civil federal administration: more than 2,700 properties with roughly 3,000 buildings and about 6 million m², from administrative and court buildings to customs facilities, embassies, research institutes and cultural buildings.\[94\] Its area management, a CAFM system with room data in SAP, needs for every floor the room outlines with a unique room ID (AOID), usage and area, and the gross floor area (GF). These figures feed net-zero reporting, portfolio management, maintenance, transactions and workspace planning.

For many buildings this data is missing, outdated or of unknown quality. There are two reasons:

- **The archive is large and unstructured.** BBL surveys buildings regularly, but it cannot survey all of them. At the same time it holds, and keeps receiving, a great deal of documentation: plans as scans, PDFs and DWGs, construction-project deliveries, building documentation brochures (Bautendokumentationen). Most of it is unstructured, in no common format or drawing style, and hard to process by hand.
- **New deliveries are hard to get right.** Delivery teams must supply DWGs that follow the CAD-Richtlinie BBL V1.0, with fixed layers for room polygons and AOIDs. Compliance is checked with a commercial data-validation product and, as a prototype, with BBL's open-source [plan-check](https://github.com/bbl-dres/plan-check). Many teams find the requirements hard to meet, which slows the whole data flow.

The market offers no suitable solution (§4). This motivates an in-house approach: extract structured floor data directly from plans as they are archived or delivered, using computer vision (CV) where no clean CAD data exists.

## 2. Goals

1. **Unlock the archive (data mining).** Turn existing documentation into structured floor data for buildings that are not surveyed: rooms, areas, GF, room connections. Use it to fill gaps in SAP, and to decide where a survey is most needed (missing data, large deviations, outdated plans).
2. **Relieve delivery teams.** Accept plans in the form they are produced (construction drawings, PDFs, DWGs drawn for print) and generate the CAD-Richtlinie layers automatically, so that people only correct what plan-check flags.
3. **Keep the data trustworthy.** Every extracted value carries its source and a confidence; nothing is filled in silently. Plans are not the building, and recorded areas are often entered by hand, so results are checked against SAP, other documents or the site, and a person confirms what is uncertain.

**What we want to extract**, in order of priority. The elements follow the Industry Foundation Classes (IFC 4.3, ISO 16739-1:2024), so that every extracted element maps to an IFC class for export:

| Element | Details | IFC |
|---|---|---|
| Rooms (core) | Outline; from the tag: name or usage, AOID or room number, SIA category, height, materials. Three areas: net and gross from the geometry, and the tagged area | `IfcSpace`; `Qto_SpaceBaseQuantities` (NetFloorArea, GrossFloorArea, Height); tag values in a BBL property set |
| Doors | All types, including empty openings without a leaf; the two rooms each connects | `IfcDoor`; an empty door opening is an `IfcOpeningElement` in the wall without a filling; connected rooms via `IfcRelSpaceBoundary` |
| Windows | All types; the room each serves. Always separate from doors | `IfcWindow` |
| Walls | Exterior or interior; load-bearing where evident | `IfcWall`; `Pset_WallCommon` (IsExternal, LoadBearing) |
| Structure | Columns; beams where drawn (low priority, hard to detect) | `IfcColumn`, `IfcBeam` |
| Circulation | Stairs, ramps | `IfcStair`, `IfcRamp` |
| Voids in the floor | Shafts, staircase openings, air spaces, light wells; deducted from the GF above 5 m² | `IfcOpeningElement` in the `IfcSlab` |
| Building systems | Elevators, escalators and other large components | `IfcTransportElement` (elevator, escalator) |
| Furniture | Where drawn | `IfcFurniture` |
| Derived per floor | GF (required), room connections, EBF and zone proposals | `IfcBuildingStorey` with `Qto_BuildingStoreyBaseQuantities` (GrossFloorArea); zones as `IfcZone` |

**Outputs.** JSON following the [data model](pipeline.md#4-data-model) (a versioned schema will follow), DWG in predefined layers (to be defined in the next version), IFC 4.3 with the classes above, PDF and Excel.

**Use cases.**
- Area quantities for reporting, e.g. net zero.
- Cost estimation for facility management: cleaning, maintenance.
- Visualisation and workspace management.
- Fire protection planning: escape routes.
- Simulations of health in the workspace: visibility, acoustics, accessibility, daylight.

**Inputs.** Whatever users upload: raster images (JPG, PNG, TIFF, also scans and photos), PDFs (vector, raster or both), and DWGs. Construction drawings and facility-management plans dominate; historical plans follow no standard at all.

**Constraints.**
- Mostly office, administrative and historical buildings; few apartments.
- Internal plans are processed in Switzerland only; self-hosted, open-source components are preferred, and BBL publishes its own code as open source (Art. 9 EMBAG).
- Results feed the existing SAP-based area management; no second area-management system.
- Procurement follows public procurement law.

**Out of scope.** BIM authoring and digital-twin products; IFC is an export next to DWG, not an authoring target. Complete furniture inventories (furniture is extracted only where drawn) and building-condition assessment.

## 3. What We Have Learned

- **The in-house route works.** Two pilots on one historical building (Landgut Lohn) show it:
  - A segmentation model trained only on public data, rendered in random drawing styles, reads plans it has never seen: wall IoU 0.87 on a CAD print, 15 of 15 rooms with a median area error of 2.7%, all room names read locally ([pilot v2](../pilot/v2-pipeline/README.md)).
  - On 400 real plans of a public benchmark, never seen in training, it finds 88% of doors and 65% of rooms (median area error 2.9%).
  - Training a model costs one to two dollars of rented GPU time, on public data only.
- **Uploads can be handled as they come.** One normalisation step turns any image, PDF or DXF into a page image plus the vector data where it exists. A layout step finds each drawing on the sheet, its title block and its scale. On the pilot's CAD print it found the drawing on its own and recovered the scale within 0.4%, including the fact that the sheet was a reduced print ([pipeline design](pipeline.md)). DWGs still need a converter, which is a licence decision (§5).
- **Construction drawings are the main gap.** Published models lose half or more of their accuracy when moved from simple to dense, construction-style drawings (one example: 83 → 33 on the standard score), because hatches, dimension lines, axes and text look like walls ([report](reports/2026-10-07-construction-and-fm-drawings.md)). FM plans are a third style that no study covers. The cure is training data in these styles.
- **No public training data fits.** The large research datasets are non-commercial, and none contains Swiss construction or FM plans. BBL's own DWGs that pass plan-check are the best training source: rendered in many styles they give exact labels for rooms, AOIDs, GF and walls.
- **Some outputs are derived, not drawn.** GF and room connections follow from walls, rooms and doors. EBF and zones depend on information plans rarely show, so the pipeline proposes them and a person confirms.
- **Language models read but do not measure.** Vision-language models read text well (up to 0.95 on text questions) but count doors and windows poorly (0.39 and 0.34) and estimate areas badly.\[55\] They suit reading room stamps and title blocks, not geometry.
- **Recorded figures need care.** Areas in documentation and registers are often entered by hand and not derived from the geometry. Deviations between extracted and recorded areas are a finding in their own right, not automatically an extraction error.

## 4. Market

- **No product covers the chain** from archive plans to SAP, and no AI converter offers Swiss hosting. Vendor accuracy figures are unverified.
- **Archilogic** offers only its full product, which would create a second area-management system without SAP integration, and no conversion-only service; it is cloud-first. **Archilyse** has gone bankrupt.
- **On-premise options** are few (a custom build by AxcelerateAI, a Docker image from RasterScan) and can be tested on non-sensitive plans.
- **Swiss digitising services** remain the pragmatic choice for the worst scans, and laser scanning on site where no usable plan exists.

Details: [closed solutions](solutions-closed.md), [open solutions](solutions-open.md), BBL's 2025 market screening.\[1\]

## 5. Recommended Route and Next Steps

1. **Inventory the archive.** Scan it automatically per building and sheet: format, plan-check result, drawing type (construction, FM, historical), quality, drawings per sheet, scale cues. This sizes the work, shows which buildings have usable documentation, and points surveys to where documentation is missing.
2. **Use compliant DWGs twice.** Read them directly (exact), and render them in many styles as training data.
3. **Extend the training data to construction and FM styles.** Add dimension chains, axes, tags, material hatches, usage colours and full sheets to the renderer, and draw training plans from BIM models; label 100–300 BBL sheets for evaluation and fine-tuning.
4. **Run everything else through the pipeline** ([pipeline design](pipeline.md)): normalise, find the drawings and their scale, extract walls, doors, windows and rooms, read the room stamps, derive GF and connections, check plausibility.
5. **Write the CAD-Richtlinie layers and validate them with plan-check**; people correct only what is flagged, and their corrections become training data.
6. **Compare with SAP and prioritise.** Flag rooms and floors where extracted and recorded data disagree, and feed the list into survey planning.
7. **Measure what matters:** human correction time per sheet, per drawing type and style, on a stratified sample of 60–100 BBL sheets ([evaluation plan](pipeline.md#8-evaluation)).
8. **Decide the open points:** a DWG converter with a licence that fits federal use; a legal review of training data and pretrained weights; the budget for a correction tool and self-hosted GPUs.

**Open challenges** that remain beyond these steps: plans that differ from the built state; old scans, hand annotations and overlays; columns, stairs and voids; linking storeys and sheets of one building; licence-clean pretrained weights. The [literature review](literature-review.md) and the [reviews](reviews/README.md) track them.

## 6. Documents

- [Pipeline design](pipeline.md): the canonical target, stage by stage, with data model and evaluation plan
- [Literature review and state of the art](literature-review.md): methods, models, benchmarks and datasets
- [Open solutions](solutions-open.md) and [closed solutions](solutions-closed.md): what can be used or bought
- Datasets: [open](datasets-open.md), [open building documentation](datasets-open-docu.md) and [closed](datasets-closed.md)
- [Reviews](reviews/README.md) and [research reports](reports/README.md): findings, recommendations and their evidence
- Pilots: [Landgut Lohn v1](../pilot/landgut-lohn-og1/README.md) and [v2](../pilot/v2-pipeline/README.md)
