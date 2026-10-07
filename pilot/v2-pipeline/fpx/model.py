"""Data model: segmenter classes, the sheet that the stages fill in, and the shape of each result element.

Geometry is held in working pixels (y down) while the stages run and converted to plan metres (y up) on export.
"""
from dataclasses import dataclass, field
from typing import NotRequired, Optional, TypedDict

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Polygon

CLASSES = ["background", "wall", "door", "window", "column", "stairs"]
BG, WALL, DOOR, WINDOW, COLUMN, STAIRS = range(len(CLASSES))
# Extra sigmoid outputs of the v2 segmenter, one map each (renderer targets of the same names, see synth.py):
# interior = inside the building outline, boundary = outlines of room areas (also open-plan ones without a wall),
# void = voids (Luftraum, stairwell), text = boxes of text. Segment() writes them to Sheet.<name>_prob.
HEADS = ["interior", "boundary", "void", "text"]


class TextItem(TypedDict, total=False):
    text: str
    box: tuple                          # x0, y0, x1, y1 in working pixels
    conf: float
    source: str                         # "pdf" (native text layer) or "ocr"
    angle: int                          # 0 or 90 degrees
    height: float                       # character height in working pixels
    role: str                           # room stamp, number, dimension, fixture label, void label, other


class WallSegment(TypedDict):
    id: str
    line: LineString                    # centre line
    thickness: float                    # metres
    construction: NotRequired[str]      # "massive" or "lightweight": a heuristic (stage 10, fpx.export.wall_construction)
    construction_basis: NotRequired[str]    # why: thickness, or solid fill on the sheet
    fill_share: NotRequired[Optional[float]]    # share of dark ink in the wall core (None: wall too thin to tell)


class Opening(TypedDict, total=False):
    id: str
    kind: str                           # door, exterior door, window, interior opening, passage
    centre: np.ndarray
    along: np.ndarray                   # unit vector along the opening
    width: float                        # metres
    depth: float                        # metres (wall thickness at the opening)
    poly: Polygon
    score: float
    source: str                         # "segmenter" or "wall gap"
    line: tuple                         # passages only: the two wall ends
    host: Optional[str]                 # host wall segment id
    exterior: bool
    sides_px: list                      # probe points on either side
    connects: list                      # room ids (or "outside") on either side
    flag: str


class Void(TypedDict, total=False):
    px: int
    poly: Polygon
    label: str
    gf_deducted: bool


class Stamp(TypedDict, total=False):
    name: Optional[str]
    name_raw: str
    area: Optional[float]               # m², as written on the stamp
    aoid: Optional[str]
    box: tuple
    conf: float
    parts: list
    room: Optional[str]


class Room(TypedDict, total=False):
    id: str
    poly: Polygon
    px_label: int                       # label in Sheet.room_label
    stair: bool
    names: list
    names_raw: list
    name: Optional[str]
    area_stamp: Optional[float]
    aoid: Optional[str]                 # first AOID read in the room's stamps
    aoids: list                         # every AOID read in the room's stamps
    aoid_export: dict                   # {"written": bool, "aoid": str (if written), "reason": str (if not)}: R_AOID policy
    small_region: bool                  # unlabelled and below fragment_max_area: kept as a room, flagged for review
    review: Optional[dict]              # small regions: {"flag", "guess" (shaft, niche, gap), "width", "openings", "reason"}; else None
    voids: list
    usage: str
    area_deviation_pct: float
    confidence: str                     # high, medium, low
    reasons: list


class QAIssue(TypedDict):
    check: str
    severity: str
    element: str
    message: str


class Region(TypedDict, total=False):
    """Stage 1b: one region of a sheet (fpx.layout). Polygons in sheet raster pixels and paper millimetres."""
    id: str
    cls: str                            # in the JSON under "class": drawing, title block, legend, scale bar, scale note,
                                        # north arrow, notes, revision table, frame, key plan, stamp, colour key, caption, other
    polygon_px: list                    # [[x, y], ...] in sheet raster pixels (y down)
    polygon_mm: Optional[list]          # the same in paper millimetres from the top-left corner (None: resolution unknown)
    bbox_px: list
    confidence: str                     # high, medium, low
    reason: str
    source: str                         # dwg-viewport, pdf-vector, detector, rule, human
    links: dict                         # drawing <-> caption, title block <-> key plans, ...


class DrawingScale(TypedDict, total=False):
    """Stage 1c: the scale proposal of one drawing (fpx.scale.drawing_scale), confirmed in stage 9."""
    px_per_m: Optional[float]           # sheet raster pixels per plan metre
    metres_per_px: Optional[float]      # m = N 25.4 / (1000 d r)
    confidence: str
    note: Optional[dict]                # {scale: N, all, source: caption | title block (sheet default) | viewport}
    print_factor: Optional[dict]        # r and its sqrt(2) step hypothesis
    cues: dict
    consensus: dict
    flags: list                         # [QAIssue]
    confirmed: Optional[bool]           # stage 9: stamp areas agree


class Drawing(TypedDict, total=False):
    """Stage 1b: one drawing on a sheet; stages 1-9 run per drawing of kind floor plan."""
    id: str
    region: str                         # Region id
    kind: str                           # floor plan, section, elevation, detail, site plan, key plan
    title: Optional[str]                # caption text
    storey: Optional[str]               # EG, n. OG, n. UG, DG
    caption: Optional[dict]
    scale_note: dict                    # {scale: N or None, all: [N], source: caption | viewport | None}
    scale: DrawingScale                 # stage 1c
    polygon_px: list
    polygon_mm: Optional[list]
    mask: dict                          # {polygon_px, parts, margin_px, margin_mm, reason}
    north: Optional[float]              # north angle (not yet detected)
    refpoints: list                     # REF.PKT with LV95 coordinates (not yet read)
    source: str
    confidence: str
    transform: dict                     # sheet raster px -> working px of the drawing -> plan metres


@dataclass
class Sheet:
    # input, set by the loader
    id: str
    title: str
    source: str
    input_class: str
    img: np.ndarray                                     # RGB at px_per_m
    origin: tuple                                       # plan metres of pixel (0, 0); y points up
    scale: dict                                         # value and how it was obtained
    native_text: list = field(default_factory=list)     # vector PDF: [(text, x0, y0, x1, y1 in px, size_px, dir)]
    ocr_img: Optional[np.ndarray] = None                # scans: grey image at the OCR resolution
    meta: dict = field(default_factory=dict)
    px_per_m: float = 50

    # stage results
    text: list = field(default_factory=list)            # [TextItem]
    prob: Optional[np.ndarray] = None                   # class probabilities (C, H, W)
    label: Optional[np.ndarray] = None                  # class per pixel (H, W)
    interior_prob: Optional[np.ndarray] = None          # v2 segmenter only: P(inside the building outline) (H, W)
    boundary_prob: Optional[np.ndarray] = None          # v2 segmenter only: P(room-area outline, incl. open-plan) (H, W)
    void_prob: Optional[np.ndarray] = None              # v2 segmenter only: P(void: Luftraum, stairwell) (H, W)
    text_prob: Optional[np.ndarray] = None              # v2 segmenter only: P(inside a text box) (H, W)
    wall_mask: Optional[np.ndarray] = None
    column_mask: Optional[np.ndarray] = None
    wall_graph: Optional[nx.MultiGraph] = None
    wall_segments: list = field(default_factory=list)   # [WallSegment]
    wall_polys: list = field(default_factory=list)      # [Polygon]
    building_rough: Optional[np.ndarray] = None         # rough building mask, from walls and openings
    openings: list = field(default_factory=list)        # [Opening]
    stairs: list = field(default_factory=list)          # [Polygon], one per flight
    stair_mask: Optional[np.ndarray] = None
    stair_hull: Optional[np.ndarray] = None             # one outline per stair, including landings and stairwell
    voids: list = field(default_factory=list)           # [Void]
    room_label: Optional[np.ndarray] = None
    rooms: list = field(default_factory=list)           # [Room]
    gf: Polygon = field(default_factory=Polygon)        # floor outline (GF)
    stamps: list = field(default_factory=list)          # [Stamp]
    fragments: list = field(default_factory=list)       # [Room] small unlabelled regions, held from stage 7 to 9, then rooms (flagged)
    gf_area: float = 0.0
    ebf_proposal: dict = field(default_factory=dict)
    connectivity: nx.Graph = field(default_factory=nx.Graph)
    qa: list = field(default_factory=list)              # [QAIssue]
    # sheet layout (stage 1b, fpx.layout; filled by pipeline.run_document): the regions of the sheet this drawing
    # comes from and the drawing record(s); empty for sheets loaded whole (pipeline.run)
    regions: list = field(default_factory=list)         # [Region]
    drawings: list = field(default_factory=list)        # [Drawing]

    def to_plan(self, xy):
        """Working pixels -> plan metres."""
        xy = np.asarray(xy, float)
        return np.c_[self.origin[0] + xy[..., 0] / self.px_per_m, self.origin[1] - xy[..., 1] / self.px_per_m]
