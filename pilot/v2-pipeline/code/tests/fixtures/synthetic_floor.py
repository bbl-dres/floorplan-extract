"""A synthetic floor for the golden-output test: a label map and a sheet image that exercise every post-processing
stage at once (stages 3b-10), with a known geometry so that the recorded output can be read against the drawing.

Sheet 17.3 x 11 m at 50 px/m (865 x 550 px), y down. Building outer faces x 1.0-17.3 (the east wall sits on the
sheet edge, as a cropped region of interest does), y 1.0-10.0; walls 0.3 m.

    north band y 1.3-4.7:  Büro x 1.3-5.7 | Sitzungszimmer x 6.0-10.4 | hall x 10.7-17.0 with a two-flight stair
    corridor   y 5.0-6.5:  across the whole building, a door into every room
    south band y 6.8-9.7:  Archiv x 1.3-8.0 | Lager x 8.3-17.0, joined by a 1.0 m wall gap (a passage)

Features (what each tests):
- a wall-gap passage between facing wall ends (south band, no door label);
- a stair in the hall, two flights, a "Treppe" stamp inside the outline cuts a stair room out of the hall;
- a LUFTRAUM label in the stair eye: a void of kind "stair eye", under 5 m², not deducted from the GF;
- a column island in the Archiv (a hole in the room polygon, one entry in structure[]);
- an interior window between Büro and Sitzungszimmer: reported as an interior opening, not a connection;
- an exterior door in the south facade of the Archiv;
- a 0.8 x 0.8 m shaft in the Lager: a small unlabelled region kept as a room and flagged for review;
- a hyphenated stamp ("Sitzungs-" / "zimmer") with an area in m² and an AOID, written to R_AOID;
- one opening leaking to the sheet border: an unlabelled door in the east wall at the sheet edge, drawn on the
  image as a leaf and swing only, so the hall is sealed along the swing instead of being dropped.
"""
import cv2
import numpy as np

from fpx import DEFAULT, Sheet
from fpx.model import CLASSES, COLUMN, DOOR, STAIRS, WALL, WINDOW

M = DEFAULT.px_per_m
W_M, H_M = 17.3, 11.0


def px(v):
    return int(round(v * M))


def synthetic_floor():
    lab = np.zeros((px(H_M), px(W_M)), np.uint8)
    img = np.full(lab.shape + (3,), 255, np.uint8)

    def rect(x0, y0, x1, y1, v):
        lab[px(y0):px(y1), px(x0):px(x1)] = v

    # exterior walls and the interior
    rect(1.0, 1.0, 17.3, 10.0, WALL)
    rect(1.3, 1.3, 17.0, 9.7, 0)
    # corridor walls
    rect(1.3, 4.7, 17.0, 5.0, WALL)
    rect(1.3, 6.5, 17.0, 6.8, WALL)
    # north band partitions
    rect(5.7, 1.3, 6.0, 4.7, WALL)
    rect(10.4, 1.3, 10.7, 4.7, WALL)
    # south band partition with a 1.0 m gap (no door label: a passage)
    rect(8.0, 6.8, 8.3, 9.7, WALL)
    rect(8.0, 7.4, 8.3, 8.4, 0)
    # doors from the corridor into every room
    for x0 in (3.0, 8.0, 15.5):
        rect(x0, 4.7, x0 + 0.9, 5.0, DOOR)
    for x0 in (5.0, 11.0):
        rect(x0, 6.5, x0 + 0.9, 6.8, DOOR)
    # windows in the north facade, an interior window between the two offices
    rect(2.5, 1.0, 3.7, 1.3, WINDOW)
    rect(7.5, 1.0, 8.7, 1.3, WINDOW)
    rect(5.7, 2.5, 6.0, 3.5, WINDOW)
    # exterior door in the south facade of the Archiv
    rect(2.5, 9.7, 3.4, 10.0, DOOR)
    # stair: two flights in the hall, the eye between them
    rect(11.5, 1.6, 12.5, 4.0, STAIRS)
    rect(13.7, 1.6, 14.7, 4.0, STAIRS)
    # column island in the Archiv
    rect(4.0, 7.5, 4.3, 7.8, COLUMN)
    # shaft in the Lager: 0.25 m walls around a 0.8 x 0.8 m cavity
    rect(14.0, 7.0, 15.3, 8.3, WALL)
    rect(14.25, 7.25, 15.05, 8.05, 0)
    # the leaking opening: a door in the east wall at the sheet edge that the segmenter missed, in the corner so
    # that the wall end faces the north wall's flank (no passage) and the hall's free space reaches the border
    rect(17.0, 1.3, 17.3, 2.2, 0)

    img[np.isin(lab, (WALL, COLUMN))] = 0
    # the missed door as drawn on the sheet: leaf hinged at (17.0, 2.2) into the hall, swing arc up to (17.0, 1.3)
    cv2.line(img, (px(17.0), px(2.2)), (px(16.1), px(2.2)), (0, 0, 0), 2)
    cv2.ellipse(img, (px(17.0), px(2.2)), (px(0.9), px(0.9)), 0, 180, 270, (0, 0, 0), 1)
    img[:, -2:] = 160                                   # the cut edge of the scan is grey (not ink), as on a real crop

    sheet = Sheet("golden", "synthetic floor", "test", "synthetic", img, (0.0, H_M), {"value": "1:50", "method": "test"})
    sheet.label, sheet.seg_label = lab, lab.copy()
    sheet.prob = (np.arange(len(CLASSES))[:, None, None] == lab[None]).astype(np.float32)

    def stamp(text, x, y, role, width=1.2):
        return {"text": text, "box": (px(x), px(y), px(x + width), px(y + 0.25)), "conf": 1.0, "source": "pdf",
                "angle": 0, "height": px(0.25), "role": role}

    sheet.text = [
        stamp("Büro", 3.0, 2.8, "room stamp"), stamp("15.0 m²", 3.0, 3.1, "number"),
        stamp("Sitzungs-", 7.6, 2.6, "room stamp"), stamp("zimmer", 7.6, 2.9, "room stamp"),
        stamp("14.5 m²", 7.6, 3.2, "number"), stamp("2051.ZZ.01.012", 7.6, 3.5, "room stamp", width=1.6),
        stamp("Treppe", 12.6, 1.8, "room stamp", width=1.0),            # inside the stair outline, on the eye
        stamp("LUFTRAUM", 12.6, 3.0, "void label", width=1.0),          # in the stair eye
        stamp("Halle", 15.2, 2.5, "room stamp"),
        stamp("Korridor", 8.5, 5.5, "room stamp"),
        stamp("Archiv", 2.0, 8.0, "room stamp"), stamp("19.4 m²", 2.0, 8.3, "number"),
        stamp("Lager", 10.0, 7.5, "room stamp"), stamp("23.5 m²", 10.0, 7.8, "number"),
    ]
    return sheet
