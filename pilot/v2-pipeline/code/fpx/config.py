"""All tunable thresholds of the pipeline in one place, in metres or square metres, with the reason for each value.

Values were set during pilot v2 while looking at the Landgut Lohn sheets (see the review). They are to be swept and
frozen on the public evaluation harness, not on BBL test sheets. Override per run with Config.from_file(path) (JSON).
Pixel-level constants of an algorithm (kernel sizes of a few pixels, Hough votes) stay in the code.
"""
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # resolution
    px_per_m: int = 50                  # segmenter working resolution (2 cm/px); training renders use the same
    ocr_px_per_m: int = 150             # OCR resolution: room lettering of ~0.25 m becomes ~40 px high
    line_simplify: float = 0.03         # tolerance when simplifying wall centre lines and stair outlines (m)
    poly_simplify: float = 0.005        # polygon simplification (m): under half a pixel keeps 1-px steps; 1 px tilted whole sides (-1 % area)
    contour_simplify: float = 0.012     # Douglas-Peucker on traced pixel-centre contours before the half-pixel offset (m, 0.6 px): the steps of a tilted edge deviate half a pixel from the chord and go, a one-pixel plateau along an edge (a real offset) deviates more and stays; 1 px cost 0.8 % of a room

    # stage 0-1: triage and preprocessing
    solid_ink_width: float = 0.15       # ink at least this thick counts as solid fill (poché walls)
    solid_ink_share: float = 0.35       # above this share of solid ink the walls are classed as poché
    skew_min_line: float = 2.0          # lines used to measure skew are at least this long (m)
    deskew_min_deg: float = 0.3         # smaller skews are left alone (resampling blurs thin lines)
    flat_scan_max_deg: float = 2.0      # larger skews are flagged in triage

    # stage 2: text layer
    ocr_tile: int = 1280                # OCR tile size in OCR pixels; detector input stays manageable
    ocr_overlap: int = 200              # tile overlap so that words on a tile border are read whole
    ocr_min_conf: float = 0.5           # below this, PP-OCR output is mostly noise from hatching and symbols (the one threshold: the engine's own filter is bypassed)
    ocr_rotated_scale: float = 0.7      # the detection pass for vertical text runs on the image rotated by 90 degrees and reduced by this factor: vertical dimension strings are horizontal text there, which the detector finds down to ~10 px characters (the upright pass alone misses a third of the vertical numbers); at 1.0 the pass costs as much as the upright one, at 0.5 the pre-pass's 16 px characters are lost; 0 skips it
    ocr_prepass_reuse: float = 0.8      # a drawing reuses the sheet pre-pass OCR when the pre-pass resolution is at least this share of ocr_px_per_m: the pre-pass puts characters at 16 px, as good as the 19 px of a 1:50 scan at 150 px/m; a 1:100 plan (37 px at 150 px/m) gets a second pass inside the drawing

    # stage 3: segmentation and walls
    seg_tile: int = 1024                # inference tile (px); the U-Net is fully convolutional
    seg_overlap: int = 128              # blended overlap avoids seams at tile borders
    seg_tta: bool = True                # average over four flips: steadier on unseen styles
    wall_min_area: float = 0.05         # wall blobs smaller than this are noise (m²)
    column_min_area: float = 0.02       # smallest column cross-section kept (m²): about 15 x 15 cm
    wall_segment_min_length: float = 0.2    # shorter centre-line pieces are skeleton artefacts (m)
    wall_min_thickness: float = 0.06    # an isolated wall piece thinner than this and than half the sheet's median wall thickness is a single stroke (dimension line, hatching, leader), not a wall: the thinnest partition is 0.08 m; on a coarse plan whose walls are all thin lines nothing is thinner than the median (m)
    stroke_max_area: float = 0.5        # ... and smaller than this: a 20 m dimension line of one stroke is 0.4 m², a building outline drawn thin is larger (m²)
    railing_max_thickness: float = 0.15 # an isolated piece surrounded by stair treads and thinner than this is a railing or stringer, not a wall (m)
    stair_ring_share: float = 0.6       # share of a wall piece's surroundings that is stair, from which an isolated thin piece counts as part of the stair
    wall_text_prob: float = 0.5         # a wall piece whose pixels lie in text boxes (model v2+ text head) at this mean probability or more is lettering read as wall
    fixture_reach: float = 1.0          # wall pixels within this distance of a fixture label (Kachelofen, Cheminée, ...) ... (m)
    fixture_outline_max: float = 0.12   # ... and thinner than this are the fixture's outline, not a wall: the walls a stove stands against are 0.15 m or more (m)
    wall_snap_angle: float = 6.0        # a centre-line piece within this angle of a dominant direction is snapped to it: skeleton wobble on a 0.3 m wall at 50 px/m reaches 5 degrees, real oblique walls differ by more (degrees)
    wall_simplify_share: float = 0.35   # Douglas-Peucker tolerance of a centre line as a share of its wall's thickness: the skeleton of a straight wall wanders by up to a third of the thickness at junctions and jambs
    gap_bridge_min: float = 0.06        # a wall end facing another wall across less than this is a skeleton seam, not a gap (m)
    gap_bridge_max: float = 2.0         # the longest missed wall piece bridged from the sheet's ink: longer gaps are rooms, not walls (m)
    gap_bridge_ink: float = 0.6         # share of the gap's length with ink in the wall band that proves a drawn wall: door jambs cover ~10 %, a swing arc at most half

    # stage 4: openings
    opening_min_area: float = 0.03      # smaller door/window blobs are noise (m²)
    opening_min_length: float = 0.4     # narrower than any real door or window (m)
    opening_host_max_dist: float = 0.6  # an opening's centre lies within this distance of its host wall line (m)
    opening_side_offset: float = 0.4    # probe beyond half the wall depth to find the spaces on either side (m)
    building_close: float = 1.6         # closing that bridges window and door gaps in the facade (m)
    passage_min_wall_area: float = 0.3  # only walls at least this large can end at an open passage (m²)
    passage_max_angle: float = 10.0     # a passage must follow the plan's dominant directions (degrees)
    passage_back: float = 0.5           # wall direction is measured over this length behind the wall end (m)
    passage_exit_max: float = 0.8       # maximum distance to leave the own wall body (m)
    passage_min: float = 0.5            # narrower gaps are not passages (m)
    passage_max: float = 2.2            # wider gaps are open-plan transitions, not passages (m)
    passage_dedupe: float = 0.3         # passages closer than this are duplicates (m)

    # stage 5: stairs and voids
    stair_min_area: float = 1.0         # smaller stair blobs are tread fragments (m²)
    stair_group_dist: float = 1.5       # flights closer than this belong to one stair (m)

    # stage 6: rooms
    slit_close: float = 0.3             # closes slits between walls and partly labelled openings (m)
    room_min_area: float = 0.25         # CAD-Richtlinie Kap. 5.8 (plan-check POLY_004): every room of 0.25 m² or more needs a polygon; smaller free regions are dropped (m²)
    stair_split_min_area: float = 3.0   # split a stair from a hall only if both parts are at least this large (m²)
    stair_hall_ratio: float = 1.0       # without a stamp on the stair, a stair is cut from its region only when the rest is at least this many times the stair outline (Landgut Lohn: an 18 m² open stair next to a 19 m² hall; a landing is smaller than its flights) ...
    stair_enclosure: float = 0.6        # ... and at least this share of the outline's boundary is wall (a stairwell open on one side: 0.66-0.79 on the Landgut scans)
    seal_erosion: float = 0.65          # erosion radius that separates a room from the outside through a window (m)
    seal_min_core: float = 1.0          # smallest room core kept by sealing (m²)
    stair_room_share: float = 0.5       # a room counts as stair if more than this share lies inside the stair outline
    void_min_area: float = 0.5          # smallest void kept (m²)
    separation_prob: float = 0.5        # model v2 boundary head: pixels at this probability or more cut a free region into open-plan areas (1.0: off). The head outlines every area, also where no wall stands: renders merged 297 -> 199, CVC-FP rooms recall 0.64 -> 0.72 at higher precision
    separation_min_area: float = 2.0    # every part of such a cut must reach this area, else the cut is noise around fixtures (m²)
    building_from_interior: bool = True     # model v2 interior head joins the enclosed-barrier rule as the building mask: outside-building misses 157 -> 76 on renders, 27 -> 2 on CVC-FP scans, GF IoU 0.913 -> 0.940; regions need a walled boundary (below), so a blob on empty paper is no room
    building_enclosure: float = 0.5     # with the interior head, a region counts as a room only when this share of its boundary is wall, door or window: a hallucinated blob on empty paper has none
    gf_close: float = 0.3               # closing of the GF outline over small gaps (m)
    gf_min_area: float = 5.0            # smaller outlines are not floors (m²)
    void_gf_deduction: float = 5.0      # CAD-Richtlinie: voids above this are cut out of the GF (m²)

    # stage 7: room attributes
    stamp_near_room: float = 0.5        # a stamp outside every room belongs to the nearest room within this distance (m)
    fragment_max_area: float = 1.5      # unlabelled regions below this are kept as rooms flagged for review (shaft, niche, gap), not dropped (m²)
    name_match_whole: float = 0.8       # OCR'd name snaps to a vocabulary entry at this similarity or more
    name_match_word: float = 0.65       # otherwise each word snaps at this similarity, so proper names survive

    # stage 8: derived outputs
    gf_inner_buffer: float = 0.05       # inset of the GF outline when testing which side of an opening is outside (m)
    exterior_probe: float = 0.3         # distance beyond the opening face where outside is tested (m)
    connect_probe: tuple = (0.3, 0.6)   # distances beyond the opening face where the two spaces are probed (m)
    interior_window_door_width: tuple = (0.0, 0.0)   # experiment, off: a window symbol in an interior wall of this width (e.g. 0.6-1.3 m) is a door, since casement windows and doors share the swing symbol (CubiCasa door regression of v2). Off because an unlabelled balcony stays inside the GF outline and its windows would become doors (oracle window recall 1.0 -> 0.78)
    opening_host_extension: float = 0.5   # a host wall's centre line is extended this far beyond its ends: centre lines stop at the jambs, so an opening's centre lies off the end by half its width (m)
    opening_host_angle: float = 15.0      # the host wall runs along the opening within this angle (degrees)

    # stage 9: QA
    stamp_tolerance_pct: float = 5.0    # stamp area vs polygon area: within this is "high" confidence
    stamp_low_pct: float = 15.0         # beyond this deviation the room is "low" confidence
    stamp_tolerance_m2: float = 0.3     # the stamp tolerance is at least this much: a 2-px band around a 5 m² WC is 0.25 m², discretisation, not a finding (m²)
    small_room_area: float = 2.0        # rooms below this are flagged as very small (m²)
    scale_cue_min_rooms: int = 3        # the stamp-area scale cue needs at least this many stamped rooms
    scale_cue_agreement: float = 0.03   # stamp-area scale cue agrees with the scale if within 3 % in length
    gap_max_width: float = 0.3          # a flagged small region narrower than this (inscribed circle) is guessed to be a gap between wall lines or behind a fixture, not a shaft or niche (m)

    # stage 10: export (DXF on the CAD-Richtlinie layers)
    massive_wall_min_thickness: float = 0.17    # heuristic: walls this thick are massive (SOLID hatch on A_SCHRAFFUR): metal-stud partitions are at most 0.15 m, plus a pixel of measurement error (m)
    solid_fill_share: float = 0.6       # heuristic: a wall core with this share of dark ink or more is drawn solid (poché), which marks masonry and concrete on such sheets
    massive_solid_min_thickness: float = 0.1    # a wall drawn solid counts as massive from this thickness; thinner solid strokes are lines, not walls (m)
    ifc_storey_height: float = 3.0      # IFC export: nominal storey height for spaces, walls and columns; a plan has no heights, the value is marked nominal in the file (m)
    ifc_door_height: float = 2.1        # IFC export: nominal door height (m)
    ifc_window_sill: float = 0.9        # IFC export: nominal window sill height (m)
    ifc_window_height: float = 1.4      # IFC export: nominal window height (m)
    ifc_stair_height: float = 1.5       # IFC export: nominal height of a stair flight solid, half a storey (m)
    ifc_slab_thickness: float = 0.3     # IFC export: nominal floor slab thickness below elevation 0 (m)

    # stage 9 (pre-pass on the native sheet): scale cues and consensus, fpx.scale
    scale_ocr_char_px: float = 16.0     # pre-pass OCR reduces larger characters to this height: digits read as well as at 26 px, 2-5x faster
    scale_ocr_small_px: float = 12.0    # characters below this height are enlarged for the pre-pass OCR: hand-lettered "1" read as "4" at 11 px
    scale_ocr_enlarge_px: float = 26.0  # ... to this height (at most 2x): small sheets, so the extra OCR time is a few seconds
    scale_ocr_max_side: int = 7000      # cap on the longer side of the pre-pass OCR image: OCR time grows with the area
    scale_ocr_tile: int = 2000          # pre-pass OCR tile: RapidOCR keeps up to 2000 px unscaled; long scale notes cut at 1280 px tile borders
    scale_ocr_overlap: int = 400        # pre-pass tile overlap: a title-block line of 20-40 characters fits whole in one tile
    scale_text_min: float = 0.05        # smallest plausible dimension-text box height in plan metres (2 mm lettering at 1:20, with margins)
    scale_text_max: float = 1.2         # largest plausible one (3.5 mm lettering at 1:200 with margins; 1:500 sheets rarely carry dimensions)
    scale_door_width: float = 0.9       # typical clear door width that the door-width search aims for (m)
    scale_tol_note: float = 0.015       # scale note with known dpi: exact up to paper shrink and scanner drift (about 1 %)
    scale_tol_dims: float = 0.03        # dimension strings: tick positions are read to a pixel or two on spans of 50 px and more
    scale_tol_bar: float = 0.04         # scale bar: label centres stand for the bar divisions to within a few pixels
    scale_tol_stamps: float = 0.05      # stamp areas vs room polygons: wall-face conventions and stale stamps shift areas by a few %
    scale_tol_doors: float = 0.2        # door widths vary from 0.7 to 1.0 m, so the door-width cue is good to about 20 %
    scale_snap: float = 0.03            # with a known dpi, snap to a standard or noted scale within 3 % (paper shrink, scanner drift)
    scale_print_step_tol: float = 0.04  # stage 1c: a print factor within 4 % of a sqrt(2) step is that reduction (SIA 400); fit-to-page prints and paper shrink move it by a few per cent
    scale_prior_building_m: float = 25.0    # stage 1c without any scale cue: of 1:50, 1:100 and 1:200 the scale that makes the drawing's longer side closest to this many metres is the prior (a house 12-30 m, an office wing 40-60 m); run at it and flagged for confirmation
    scale_tol_sizes: float = 0.05       # room sizes on stamps ("30'-6\" x 48'-6\"", "4.50 x 3.20") against the clear extent from the stamp to the walls: wall faces are hit to a pixel, but the stated size is the clear one and finishes or OCR'd inches differ by a few %

    # stage 0a: normalisation (fpx.inputs); sizes on paper in millimetres
    input_text_px: float = 24.0         # the smallest text is rendered with at least this font size in px: digits ~17 px high, inside the 12-21 px band the pre-pass OCR reads without resampling
    input_text_quantile: float = 0.1    # "smallest text" is this quantile of the text heights: a few stray superscripts do not drive the resolution
    input_dpi_min: int = 150            # never render below this: 0.18 mm lines stay a pixel wide
    input_dpi_max: int = 1200           # cap: the reduced S1 print needed 1200 dpi for its lettering; beyond that OCR gains little and memory grows fourfold per step
    input_dpi_default: int = 300        # vector pages without any text: the usual archive scan resolution
    input_max_mpx: float = 160.0        # cap on a rendered raster (megapixels): 160 MP RGB is 480 MB; an A4 page at 1200 dpi (139 MP) fits, A1 is reduced to ~500 dpi
    input_raster_page_share: float = 0.6    # PDF page classification: images covering this share of the page make it a raster (or mixed) page
    input_vector_min_paths: int = 50    # a page with fewer vector paths is not a vector drawing (a scan wrapper's frame or a few markups)
    outlined_glyph_min_mm: float = 0.5  # outlined-text glyphs are at least this large on paper: 1:100 lettering printed at A4 from A1 is ~0.7 mm
    outlined_glyph_max_mm: float = 8.0  # ... and at most this large (titles); larger paths are drawing geometry
    outlined_min_glyphs: int = 3        # a row of at least this many glyph-like paths without a text span counts as outlined text
    dxf_model_px_per_m: float = 100.0   # DXF model space (no paper): rendered at twice the segmenter resolution, so 0.25 m lettering is 25 px high for the layout stage and OCR

    # stage 1b: sheet layout and masking (fpx.layout), rule baseline; sizes on paper in millimetres
    layout_max_side: int = 3000         # the layout rules run on a copy reduced to this longer side: an A1 sheet keeps ~3.6 px/mm, enough for frames, tables and 2 mm text
    layout_gap_mm: float = 8.0          # ink closer than this belongs to one drawing: dimension chains sit 5-10 mm off the facade, separate drawings are usually 20 mm or more apart
    layout_gap_share: float = 0.012     # the same without a known resolution, as a share of the longer sheet side (10 mm on A1)
    layout_frame_min_share: float = 0.5 # a frame line spans at least this share of the sheet side (cut edge and frame of ISO 5457 run around the whole sheet; old scans break them up); it also needs a free margin inside
    layout_drawing_min_mm: float = 25.0 # ink clusters smaller than this (longer side) are symbols, stamps or notes, not drawings: a 5 m room at 1:200
    layout_drawing_min_share: float = 0.06  # the same without a known resolution, as a share of the longer side of the content area
    layout_minor_share: float = 0.2     # a cluster with less than this squared (4 %) of the largest drawing's area is a detached chain, label or symbol: joined to a drawing nearby, else "other"
    layout_fragment_solidity: float = 0.5   # a cluster covering less than this share of its convex hull is an open fragment (walls of a plan with door openings drawn as gaps), not a closed drawing
    layout_fragment_reach: float = 4.0  # ... and joins its nearest neighbour within this many gaps (a 1 m door gap at 1:100 is 10 mm, at most 1.25 gaps; 4 leaves room for wider openings)
    layout_text_share: float = 0.6      # an ink cluster whose ink lies at least this much inside text boxes is a text block (notes, title), not a drawing
    layout_title_block_max_share: float = 0.25  # a title block covers at most this share of the sheet (SIA 400: at most 180 mm wide on any format)
    layout_caption_mm: float = 30.0     # a caption lies within this distance of its drawing (titles sit directly below or above the drawing)
    layout_mask_margin_mm: float = 5.0  # mask = drawing polygon plus this margin: dimension text and chain ends just outside the inked cluster stay in

    @property
    def m(self):
        """Pixels per metre."""
        return self.px_per_m

    def px(self, metres):
        """Length in metres -> working pixels."""
        return metres * self.px_per_m

    def px2(self, square_metres):
        """Area in square metres -> working pixels."""
        return square_metres * self.px_per_m * self.px_per_m

    def m2(self, px_area):
        """Area in working pixels -> square metres."""
        return px_area / self.px_per_m / self.px_per_m

    @classmethod
    def from_file(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        if "connect_probe" in values:
            values["connect_probe"] = tuple(values["connect_probe"])
        return replace(cls(), **values)

    def to_dict(self):
        return asdict(self)


DEFAULT = Config()
