import pytest

from fpx.attributes import cluster_stamps, correct_name, stamp_area, usage


def item(text, box, role):
    return {"text": text, "box": box, "conf": 0.9, "source": "ocr", "angle": 0, "height": box[3] - box[1], "role": role}


@pytest.mark.parametrize("name, expected", [
    ("Bad / WC", "sanitary"), ("Obere Halle", "circulation"), ("Balkon", "outdoor"),
    ("Treppe", "stair"), ("Sitzungszimmer", "room"), ("Xyz", "unknown"),
])
def test_usage(name, expected):
    assert usage(name) == expected


def test_correct_name_keeps_proper_names():
    assert correct_name("Doudoir") == "Boudoir"
    assert correct_name("festibule") == "Vestibule"
    assert correct_name("Kreidolf-Timmer") == "Kreidolf-Zimmer"


def test_stamp_area_prefers_explicit_unit():
    parts = [item("Büro", (0, 0, 40, 10), "room stamp"), item("24.50 m2", (0, 12, 40, 22), "number"),
             item("1.12", (0, 24, 20, 34), "number")]          # room number after the area
    assert stamp_area(parts)[0] == 24.5
    assert stamp_area([item("Büro", (0, 0, 40, 10), "room stamp"), item("18,2", (0, 12, 30, 22), "number")])[0] == 18.2
    assert stamp_area([item("NF 24.50 m²", (0, 0, 60, 10), "room stamp")])[0] == 24.5
    assert stamp_area([item("Büro", (0, 0, 40, 10), "room stamp")]) == (None, None, None)
    assert stamp_area([item("Büro", (0, 0, 40, 10), "room stamp"), item("1.12", (0, 12, 30, 22), "number"), item("24.50", (0, 24, 30, 34), "number")]) == (24.5, item("24.50", (0, 24, 30, 34), "number"), "1.12")[:1] + (stamp_area([item("Büro", (0, 0, 40, 10), "room stamp"), item("1.12", (0, 12, 30, 22), "number"), item("24.50", (0, 24, 30, 34), "number")])[1], "1.12")
    assert stamp_area([item("Halle", (0, 0, 40, 10), "room stamp"), item("18 m2", (0, 12, 30, 22), "number")])[0] == 18.0


def test_cluster_stamps():
    items = [item("Sitzungs-", (0, 0, 60, 10), "room stamp"), item("zimmer", (0, 12, 45, 22), "room stamp"),
             item("NF 24.50 m²", (0, 24, 60, 34), "room stamp"),
             item("3.50", (300, 300, 330, 310), "number")]     # far away: a dimension
    (s,) = cluster_stamps(items)
    assert s["name"] == "Sitzungszimmer"
    assert s["area"] == 24.5
    assert items[3]["role"] == "number"
