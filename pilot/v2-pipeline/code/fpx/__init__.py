"""fpx: floor plan extraction, pipeline stages 0-10 of docs/pipeline.md.

One module per stage; thresholds in config.Config; the sheet and its result elements in model. Importing the package
is light: torch, OCR and CAD libraries load only with the modules that need them.

    inputs      0a normalisation: any upload (JPG/PNG/TIFF, PDF, DXF; DWG via a converter) -> sheet packages
    triage      0b triage, 1 preprocessing
    layout      1b sheet layout and masking: regions (title block, legend, scale bar, ...) and drawings
    scale       1c scale cues and consensus, per sheet and per drawing (confirmed in 9)
    text        2 text layer (native PDF text, OCR)
    segment     3a segmentation (U-Net, tiled, flip averaging)
    walls       3b wall graph
    openings    4 doors, windows, passages
    stairs      5 stairs
    rooms       6 rooms, voids, GF outline
    attributes  7 room stamps, names, areas, usage
    derived     8 GF area, exterior openings, EBF proposal, connectivity
    qa          9 scale cue and QA
    export      10 JSON and DXF
    conformance local checks of the DXF against plan-check's rules
    pipeline    runs the stages in order: run() on one sheet, run_document() on any upload, per drawing
"""
from .config import DEFAULT, Config
from .model import Sheet
