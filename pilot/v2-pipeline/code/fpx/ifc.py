"""Stage 10, IFC export: the sheet's JSON (fpx.export.to_json) as an IFC 4.3 model of one storey.

What is written, and how it maps to the data model (docs/motivation-goals.md, docs/pipeline.md §4):
- IfcProject / IfcSite / IfcBuilding / IfcBuildingStorey (elevation 0, a nominal storey height from the config);
- IfcSpace per room: the net polygon (inner wall faces) extruded by the storey height; Qto_SpaceBaseQuantities with
  NetFloorArea (voids excluded), GrossFloorArea (IFC's own meaning: the footprint, voids included) and Height;
  Pset_SpaceCommon.Reference = room number; Pset_BBL_Raum with the stamp area, the wall-share gross area, the area
  basis, usage, confidence, the raw stamp text and the source, so that BBL's own figures sit next to IFC's;
- IfcWall per wall segment (centre line swept by the thickness), Pset_WallCommon.IsExternal from the wall kind,
  LoadBearing left unset (unknown); Pset_BBL_Wand with the construction heuristic;
- IfcDoor per door of any kind (an empty door opening: PredefinedType USERDEFINED, ObjectType "empty opening"),
  IfcWindow per window, both as the opening polygon extruded to nominal heights, free-standing (hosting as
  IfcOpeningElement in the wall follows when openings are intervals on the wall graph);
- IfcStair per room with flights, aggregating one IfcStairFlight per flight polygon; IfcColumn per column;
- IfcSlab (FLOOR) with the GF polygon, one IfcOpeningElement per deducted void;
- IfcRelSpaceBoundary (first level) between every door and the two spaces it connects.
GUIDs are derived from the sheet id and the element id, so writing twice gives the same file; the header time stamp
is fixed for the same reason. Lengths in metres, plan frame of the drawing (x right, y up).
"""
import uuid

import numpy as np
from shapely.geometry import LineString, shape

from .config import DEFAULT

NAMESPACE = uuid.UUID("5f0f7f52-6c0d-4a1d-9c6e-2d8b1f0a4e21")     # fpx IFC GUID namespace
TIME_STAMP = "2026-01-01T00:00:00"


def guid(*parts):
    import ifcopenshell.guid
    return ifcopenshell.guid.compress(uuid.uuid5(NAMESPACE, "/".join(str(p) for p in parts)).hex)


def _ring(f, coords):
    pts = [f.createIfcCartesianPoint((float(x), float(y))) for x, y in coords[:-1]] if tuple(coords[0]) == tuple(coords[-1]) \
        else [f.createIfcCartesianPoint((float(x), float(y))) for x, y in coords]
    return f.createIfcPolyline(pts + [pts[0]])


def _profile(f, poly):
    """Closed profile of a shapely polygon (holes become voids of the profile)."""
    outer = _ring(f, list(poly.exterior.coords))
    if poly.interiors:
        return f.createIfcArbitraryProfileDefWithVoids("AREA", None, outer, [_ring(f, list(h.coords)) for h in poly.interiors])
    return f.createIfcArbitraryClosedProfileDef("AREA", None, outer)


def _largest(g):
    g = shape(g) if isinstance(g, dict) else g
    parts = [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and not p.is_empty]
    return max(parts, key=lambda p: p.area) if parts else None


class Writer:
    def __init__(self, project_name, sid, cfg=DEFAULT):
        import ifcopenshell
        import ifcopenshell.api.aggregate
        import ifcopenshell.api.context
        import ifcopenshell.api.feature
        import ifcopenshell.api.geometry
        import ifcopenshell.api.project
        import ifcopenshell.api.pset
        import ifcopenshell.api.root
        import ifcopenshell.api.spatial
        import ifcopenshell.api.unit
        self.api = ifcopenshell.api
        self.cfg = cfg
        self.guids = set()                                  # GUIDs derived from element ids (the rest is normalised at the end)
        self.f = ifcopenshell.api.project.create_file(version="IFC4X3")
        self.f.header.file_name.time_stamp = TIME_STAMP                    # fixed: writing twice gives the same bytes
        self.project = self.entity("IfcProject", project_name, sid, "project")    # units and contexts hang off the project
        ua = ifcopenshell.api.unit.assign_unit(self.f, length={"is_metric": True, "raw": "METERS"})
        ua.Units = tuple(sorted(ua.Units, key=lambda u: u.id()))   # the API fills the set in hash order: fix it for reproducible files
        model = ifcopenshell.api.context.add_context(self.f, context_type="Model")
        self.body = ifcopenshell.api.context.add_context(self.f, context_type="Model", context_identifier="Body",
                                                         target_view="MODEL_VIEW", parent=model)

    def entity(self, cls, name, *key, predefined_type=None):
        e = self.api.root.create_entity(self.f, ifc_class=cls, name=name, predefined_type=predefined_type)
        e.GlobalId = guid(*key)
        self.guids.add(e.GlobalId)
        return e

    def solid(self, product, poly, depth, z=0.0):
        """Extrude a polygon (plan metres) by depth, starting at height z."""
        rep = self.api.geometry.add_profile_representation(self.f, context=self.body, profile=_profile(self.f, poly), depth=float(depth))
        self.api.geometry.assign_representation(self.f, product=product, representation=rep)
        m = np.eye(4)
        m[2, 3] = z
        self.api.geometry.edit_object_placement(self.f, product=product, matrix=m, is_si=True)

    def pset(self, product, name, props):
        props = {k: v for k, v in props.items() if v is not None}
        if props:
            p = self.api.pset.add_pset(self.f, product=product, name=name)
            self.api.pset.edit_pset(self.f, pset=p, properties=props)

    def qto(self, product, name, props):
        q = self.api.pset.add_qto(self.f, product=product, name=name)
        self.api.pset.edit_qto(self.f, qto=q, properties={k: float(v) for k, v in props.items() if v is not None})


def to_ifc(data, path, cfg=DEFAULT):
    """Write the sheet's JSON (fpx.export.to_json) to path as IFC 4.3. Returns a summary of what was written."""
    sid = data["sheet"]["id"]
    w = Writer(data["sheet"]["title"] or sid, sid, cfg)
    f, api, project = w.f, w.api, w.project
    h, note = cfg.ifc_storey_height, "nominal height from the configuration, not read from the sheet"
    site = w.entity("IfcSite", "Site", sid, "site")
    building = w.entity("IfcBuilding", data["sheet"].get("source") or "Building", sid, "building")
    storey = w.entity("IfcBuildingStorey", str((data.get("drawings") or [{}])[0].get("storey") or sid), sid, "storey")
    storey.Elevation = 0.0
    api.aggregate.assign_object(f, products=[site], relating_object=project)
    api.aggregate.assign_object(f, products=[building], relating_object=site)
    api.aggregate.assign_object(f, products=[storey], relating_object=building)
    for e in (project, site, building, storey):
        api.geometry.edit_object_placement(f, product=e)
    w.pset(project, "Pset_BBL_Herkunft", {"Sheet": sid, "Source": data["sheet"].get("source"), "InputClass": data["sheet"].get("input_class"),
                                           "Scale": str((data["sheet"].get("scale") or {}).get("value")), "Generator": (data.get("generator") or {}).get("name"),
                                           "Note": "extracted automatically: unverified, check against the sheet"})
    elements, spaces, count = [], {}, {}

    def add(e, kind):
        elements.append(e)
        count[kind] = count.get(kind, 0) + 1

    for r in data["rooms"]:
        poly = _largest(r["geometry"])
        if poly is None:
            continue
        outdoor = r.get("usage") == "outdoor"
        s = w.entity("IfcSpace", r.get("aoid") or r["id"], sid, "room", r["id"], predefined_type="EXTERNAL" if outdoor else "INTERNAL")
        s.LongName = r.get("name") or None
        w.solid(s, poly, h)
        w.qto(s, "Qto_SpaceBaseQuantities", {"NetFloorArea": r.get("area_net", r.get("area")), "GrossFloorArea": r.get("area_polygon", r.get("area")), "Height": h})
        w.pset(s, "Pset_SpaceCommon", {"Reference": r.get("number"), "IsExternal": outdoor})
        w.pset(s, "Pset_BBL_Raum", {"StampArea": r.get("area_stamp"), "AreaGrossWallShare": r.get("area_gross"), "AreaBasis": r.get("area_basis"),
                                     "AreaDeviationPct": r.get("area_deviation_pct"), "Usage": r.get("usage"), "Confidence": r.get("confidence"),
                                     "NameRaw": r.get("name_raw"), "AoidsRead": ", ".join(r.get("aoids_read") or []) or None,
                                     "StairFlights": len(r.get("stair_flights") or []), "HeightNote": note})
        spaces[r["id"]] = s                               # spaces are aggregated into the storey, not contained
        count["spaces"] = count.get("spaces", 0) + 1
    api.aggregate.assign_object(f, products=list(spaces.values()), relating_object=storey)
    for wl in data["walls"]:
        line = shape(wl["centre_line"])
        if line.is_empty or line.length == 0:
            continue
        body = line.buffer(max(wl["thickness"], 0.02) / 2, cap_style="flat", join_style="mitre")
        e = w.entity("IfcWall", wl["id"], sid, "wall", wl["id"])
        w.solid(e, _largest(body), h)
        w.pset(e, "Pset_WallCommon", {"IsExternal": wl.get("kind") == "exterior"} if wl.get("kind") in ("exterior", "interior") else {})
        w.pset(e, "Pset_BBL_Wand", {"Kind": wl.get("kind"), "Thickness": wl.get("thickness"), "Construction": wl.get("construction"),
                                     "ConstructionBasis": wl.get("construction_basis"), "LoadBearing": "unknown"})
        add(e, "walls")
    doors = {}
    for o in data["openings"]:
        poly = _largest(o["geometry"])
        if poly is None:
            continue
        if o["kind"] in ("door", "exterior door", "passage"):
            kind = "passage" if o["kind"] == "passage" else "door"
            e = w.entity("IfcDoor", o["id"], sid, "door", o["id"], predefined_type="USERDEFINED" if kind == "passage" else "DOOR")
            if kind == "passage":
                e.ObjectType = "empty opening"
            w.solid(e, poly, cfg.ifc_door_height)
            w.pset(e, "Pset_DoorCommon", {"IsExternal": bool(o.get("exterior")), "Reference": o["id"]})
            doors[o["id"]] = (e, o)
            add(e, "doors")
        else:
            e = w.entity("IfcWindow", o["id"], sid, "window", o["id"], predefined_type="WINDOW")
            w.solid(e, poly, cfg.ifc_window_height, z=cfg.ifc_window_sill)
            w.pset(e, "Pset_WindowCommon", {"IsExternal": bool(o.get("exterior")), "Reference": o["id"]})
            add(e, "windows")
        w.pset(e, "Pset_BBL_Oeffnung", {"Kind": o["kind"], "Width": o.get("width"), "HostWall": o.get("host"), "Source": o.get("source"),
                                         "Connects": ", ".join(str(c) for c in (o.get("connects") or []) if c) or None})
    for k, s in enumerate(data.get("stairs", [])):
        poly = _largest(s)
        if poly is None:
            continue
        e = w.entity("IfcStairFlight", f"flight {k}", sid, "stair", k, predefined_type="STRAIGHT")
        w.solid(e, poly, cfg.ifc_stair_height)
        add(e, "stair flights")
    for k, c in enumerate(data.get("structure", [])):
        poly = _largest(c["geometry"])
        if poly is None:
            continue
        e = w.entity("IfcColumn", f"column {k}", sid, "column", k, predefined_type="COLUMN")
        w.solid(e, poly, h)
        add(e, "columns")
    gf = _largest(data["floor"]["gf"])
    if gf is not None:
        slab = w.entity("IfcSlab", "floor slab", sid, "slab", predefined_type="FLOOR")
        w.solid(slab, gf, cfg.ifc_slab_thickness, z=-cfg.ifc_slab_thickness)
        w.qto(slab, "Qto_SlabBaseQuantities", {"GrossArea": data["floor"]["gf_area"]})
        w.pset(slab, "Pset_BBL_Geschoss", {"GF": data["floor"]["gf_area"], "AGF": data["floor"].get("agf_area"), "SumNet": data["floor"].get("sum_net"),
                                            "SumGross": data["floor"].get("sum_gross")})
        add(slab, "slabs")
        for v in data.get("voids", []):
            poly = _largest(v["geometry"])
            if poly is None or not v.get("gf_deducted"):
                continue
            o = w.entity("IfcOpeningElement", v.get("label") or v.get("id"), sid, "void", v.get("id") or v["label"], predefined_type="OPENING")
            w.solid(o, poly, cfg.ifc_slab_thickness, z=-cfg.ifc_slab_thickness)
            api.feature.add_feature(f, feature=o, element=slab)
            count["slab openings"] = count.get("slab openings", 0) + 1
    api.spatial.assign_container(f, products=elements, relating_structure=storey)
    for did, (e, o) in doors.items():                      # first-level boundaries: a door bounds the two spaces it connects
        for rid in o.get("connects") or []:
            if rid in spaces:
                b = f.createIfcRelSpaceBoundary(guid(sid, "boundary", did, rid), None, None, None, spaces[rid], e, None, "PHYSICAL",
                                                "EXTERNAL" if o.get("exterior") else "INTERNAL")
                count["space boundaries"] = count.get("space boundaries", 0) + 1
    for e in f.by_type("IfcRoot"):                        # relationships and property sets got random GUIDs from the API
        if e.GlobalId not in w.guids:
            e.GlobalId = guid(sid, e.is_a(), e.id())        # entity ids follow the creation order, so the file is reproducible
    f.write(str(path))
    return {"file": str(path), "schema": "IFC4X3", "storey_height": h, "counts": count}
