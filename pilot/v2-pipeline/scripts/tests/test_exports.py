"""Excel and IFC exports of the two-room plan: counts, quantities and a round trip through IfcOpenShell."""
import json

import pytest

from fpx import DEFAULT, pipeline
from fpx.export import to_json
from test_pipeline import two_rooms

ifcopenshell = pytest.importorskip("ifcopenshell")
openpyxl = pytest.importorskip("openpyxl")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("exports")
    sheet = two_rooms()
    pipeline.run(sheet, stages=pipeline.AFTER_SEGMENTATION + ("export",), out_dir=out)
    data = json.loads((out / "t.json").read_text(encoding="utf-8"))
    return out, sheet, data


def test_xlsx_room_list(exported):
    out, sheet, data = exported
    assert data["exports"]["xlsx"] == "t.xlsx" and data["exports"]["xlsx_summary"]["Räume"] == 2
    wb = openpyxl.load_workbook(out / "t.xlsx")
    assert wb.sheetnames == ["Räume", "Geschoss", "Öffnungen", "QA", "Meta"]
    ws = wb["Räume"]
    head = [c.value for c in ws[1]]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    assert len(rows) == 2 and head[:6] == ["ID", "AOID", "Raumnummer", "Name", "Nutzung", "Netto m²"]
    west = next(r for r in rows if r[head.index("Name")] == "Büro")
    assert west[head.index("Stempel m²")] == 24.57 and isinstance(west[head.index("Netto m²")], float)
    assert west[head.index("Brutto m² (Wandanteil)")] > west[head.index("Netto m²")]
    assert wb["Geschoss"]["B1"].value == "Wert" and wb["Öffnungen"].max_row == 3            # door and window


def test_ifc_round_trip(exported):
    import ifcopenshell.util.element as ue
    out, sheet, data = exported
    summary = data["exports"]["ifc_summary"]
    assert summary["counts"] == {"spaces": 2, "walls": len(data["walls"]), "doors": 1, "windows": 1, "slabs": 1, "space boundaries": 2,
                                 "wall openings": 2}                                      # both openings are cut from their host wall
    f = ifcopenshell.open(str(out / "t.ifc"))
    assert f.schema == "IFC4X3"
    voids = f.by_type("IfcRelVoidsElement")
    assert len(voids) == 2 and all(v.RelatingBuildingElement.is_a("IfcWall") for v in voids)
    assert len(f.by_type("IfcRelFillsElement")) == 2
    spaces = f.by_type("IfcSpace")
    assert len(spaces) == 2 and {s.PredefinedType for s in spaces} == {"INTERNAL"}
    by_name = {s.Name: s for s in spaces}
    room = next(r for r in data["rooms"] if r["name"] == "Büro")
    q = ue.get_psets(by_name[room["id"]], qtos_only=True)["Qto_SpaceBaseQuantities"]
    assert q["NetFloorArea"] == pytest.approx(room["area_net"]) and q["Height"] == DEFAULT.ifc_storey_height
    p = ue.get_psets(by_name[room["id"]])
    assert p["Pset_BBL_Raum"]["StampArea"] == 24.57 and p["Pset_BBL_Raum"]["AreaGrossWallShare"] == room["area_gross"]
    assert by_name[room["id"]].LongName == "Büro"
    walls = f.by_type("IfcWall")
    kinds = {ue.get_psets(w).get("Pset_WallCommon", {}).get("IsExternal") for w in walls}
    assert True in kinds and False in kinds                                                  # facade and interior wall
    door = f.by_type("IfcDoor")[0]
    bounds = [b for b in f.by_type("IfcRelSpaceBoundary") if b.RelatedBuildingElement == door]
    assert {b.RelatingSpace.Name for b in bounds} == set(by_name) and all(b.InternalOrExternalBoundary == "INTERNAL" for b in bounds)
    assert f.by_type("IfcWindow")[0].ObjectPlacement.RelativePlacement.Location.Coordinates[2] == pytest.approx(DEFAULT.ifc_window_sill)
    assert f.by_type("IfcSlab")[0].PredefinedType == "FLOOR" and f.by_type("IfcBuildingStorey")[0].Elevation == 0.0
    # the storey contains every element, and writing again gives the same model: GUIDs, entity counts, quantities
    # (bytes differ only in the order of set-valued relationship attributes, which the API fills in hash order)
    contained = {e.GlobalId for rel in f.by_type("IfcRelContainedInSpatialStructure") for e in rel.RelatedElements}
    assert {w.GlobalId for w in walls} <= contained and door.GlobalId in contained
    from fpx.ifc import to_ifc
    to_ifc(data, out / "t2.ifc")
    g = ifcopenshell.open(str(out / "t2.ifc"))
    assert {e.GlobalId for e in f.by_type("IfcRoot")} == {e.GlobalId for e in g.by_type("IfcRoot")}
    assert {t: len(f.by_type(t)) for t in ("IfcSpace", "IfcWall", "IfcDoor", "IfcWindow", "IfcRelSpaceBoundary", "IfcPropertySet")} ==         {t: len(g.by_type(t)) for t in ("IfcSpace", "IfcWall", "IfcDoor", "IfcWindow", "IfcRelSpaceBoundary", "IfcPropertySet")}
    assert ue.get_psets(g.by_id(by_name[room["id"]].id()), qtos_only=True)["Qto_SpaceBaseQuantities"]["NetFloorArea"] == q["NetFloorArea"]


def test_ifc_validates(exported):
    import logging
    import ifcopenshell.validate
    out, sheet, data = exported
    logger = logging.getLogger("ifc-validate")
    logger.setLevel(logging.ERROR)
    errors = []

    class Catch(logging.Handler):
        def emit(self, record):
            errors.append(record.getMessage())
    logger.addHandler(Catch())
    ifcopenshell.validate.validate(str(out / "t.ifc"), logger, express_rules=False)
    assert errors == [], errors[:5]
