"""Unit tests of the stage changes of the 2026-10-08 review: gap bridging, host walls, end-to-end passages, stairs as
objects, the building mask with wings, voids by kind, net and gross areas, wall kinds, QA severities."""
import numpy as np

from conftest import plan_image as plan, px
from fpx import DEFAULT, pipeline
from fpx.attributes import cluster_stamps, stamp_area
from fpx.model import DOOR, STAIRS, WALL, WINDOW
from fpx.openings import rough_building
from fpx.walls import contract_degree2


def run(sheet, stages=pipeline.AFTER_SEGMENTATION):
    pipeline.run(sheet, stages=stages)
    return sheet


def interior_wall(rect, gap=(3.5, 4.4), fill=0):
    rect(5.85, 1.3, 6.15, 6.7, WALL)
    rect(5.85, gap[0], 6.15, gap[1], fill)


def test_gap_bridged_only_where_the_sheet_shows_a_wall():
    # a 0.4 m gap in the interior wall that the drawing shows as wall (ink): bridged, two rooms
    s = run(plan(lambda r: interior_wall(r, (2.0, 2.4)), ink=lambda r: r(5.85, 2.0, 6.15, 2.4)))
    assert len(s.wall_bridges) == 1 and 0.3 <= s.wall_bridges[0]["length"] <= 0.5
    assert s.wall_mask[px(2.2), px(6.0)] and len(s.rooms) == 2
    assert any(i["check"] == "wall gap bridged" for i in s.qa)
    # the same gap left white on the sheet: an opening, not bridged, the rooms join
    s = run(plan(lambda r: interior_wall(r, (2.0, 2.4))))
    assert s.wall_bridges == [] and len(s.rooms) == 1


def test_openings_get_host_walls():
    s = run(plan(lambda r: (interior_wall(r, fill=DOOR), r(3, 1, 4.2, 1.3, WINDOW))))
    kinds = {o["kind"]: o for o in s.openings}
    assert set(kinds) == {"door", "window"}
    assert kinds["door"]["host"] is not None and kinds["window"]["host"] is not None
    host = next(w for w in s.wall_segments if w["id"] == kinds["door"]["host"])
    assert abs(host["line"].centroid.x - px(6.0)) < 3                  # the interior wall hosts the door
    assert not any(i["check"] == "opening hosted by a wall" for i in s.qa)


def test_passage_only_between_two_facing_wall_ends():
    s = run(plan(lambda r: interior_wall(r, (3.5, 4.4))))             # 0.9 m gap, no door label: an open passage
    assert [o["kind"] for o in s.openings] == ["passage"] and len(s.rooms) == 2
    assert list(s.connectivity.edges) and s.connectivity.edges[list(s.connectivity.edges)[0]]["type"] == "passage"
    s = run(plan(lambda r: r(5.85, 1.3, 6.15, 5.5, WALL)))            # a stub 1.2 m short of the south wall: a corner, no passage
    assert s.openings == [] and len(s.rooms) == 1


def test_stairs_are_objects_unless_stamps_demand_a_stair_room():
    stair = lambda r: r(1.3, 2.0, 2.5, 5.0, STAIRS)                   # one flight along the west wall of a 10 x 6 m hall
    s = run(plan(stair))
    assert len(s.rooms) == 1 and s.rooms[0]["stair_flights"] == [0] and not s.rooms[0]["stair"]
    s = run(plan(stair, text=[("Treppe", 1.4, 3.0, "room stamp"), ("Halle", 6.0, 3.0, "room stamp")]))
    assert len(s.rooms) == 2 and sum(r["stair"] for r in s.rooms) == 1
    assert {r["name"] for r in s.rooms} == {"Treppe", "Halle"}


def test_building_keeps_a_wing_with_a_door():
    barrier = np.zeros((400, 400), bool)
    barrier[50:200, 50:200] = True                                     # main block
    barrier[60:190, 60:190] = False
    barrier[250:350, 250:350] = True                                   # a second, detached block
    barrier[260:340, 260:340] = False
    anchors = np.zeros_like(barrier)
    assert not rough_building(barrier, DEFAULT)[300, 300]              # largest only, no anchor
    anchors[250:252, 300:310] = True                                   # a door label on the second block
    assert rough_building(barrier, DEFAULT, anchors=anchors)[300, 300]


def test_voids_by_kind():
    flights = lambda r: (r(1.3, 2.0, 2.3, 5.0, STAIRS), r(3.3, 2.0, 4.3, 5.0, STAIRS))   # two flights, the eye between them
    s = run(plan(flights, text=[("LUFTRAUM", 2.4, 3.4, "room stamp")]))
    assert [v["kind"] for v in s.voids] == ["stair eye"] and 2.5 < DEFAULT.m2(s.voids[0]["poly"].area) < 3.5
    assert not s.voids[0]["gf_deducted"] and len(s.rooms) == 1 and s.rooms[0]["voids"] == s.voids
    assert s.rooms[0]["area_net"] < s.rooms[0]["area_polygon"]
    s = run(plan(lambda r: interior_wall(r, fill=WALL), text=[("Luftraum", 8.0, 3.8, "room stamp")]))
    assert len(s.rooms) == 1 and [v["kind"] for v in s.voids] == ["air space"] and s.voids[0]["gf_deducted"]
    assert len(s.gf.interiors) == 1 and s.rooms[0]["poly"].centroid.x < px(6)


def test_net_gross_and_stamp_areas():
    s = run(plan(lambda r: interior_wall(r, fill=DOOR), text=[("Büro", 3.0, 3.8, "room stamp"), ("24.57 m2", 3.0, 4.1, "number")]))
    west = next(r for r in s.rooms if r["poly"].centroid.x < px(6))
    assert abs(west["area_net"] - 4.55 * 5.4) < 0.3 and west["area_stamp"] == 24.57 and west["area_basis"] == "net"
    assert 26.5 < west["area_gross"] < 28.0                             # half of each wall around it, to the centre lines
    assert abs(west["poly_gross"].bounds[2] - px(6.0)) <= 1.5          # the two rooms meet at the interior wall's centre
    assert abs(west["poly_gross"].bounds[0] - px(1.15)) <= 1.5         # the exterior wall is shared with the outside
    assert sum(r["area_gross"] for r in s.rooms) < DEFAULT.m2(s.gf.area)


def test_wall_kinds_and_qa_severities():
    s = run(plan(lambda r: interior_wall(r, fill=DOOR), text=[("Büro", 3.0, 3.8, "room stamp"), ("30.0 m2", 3.0, 4.1, "number")]))
    kinds = {w["id"]: w["kind"] for w in s.wall_segments}
    inner = next(w for w in s.wall_segments if abs(w["line"].centroid.x - px(6.0)) < 3)
    assert kinds[inner["id"]] == "interior" and sum(k == "exterior" for k in kinds.values()) >= 2
    assert {i["severity"] for i in s.qa} <= {"error", "warning", "info"}
    west = next(r for r in s.rooms if r["poly"].centroid.x < px(6))
    assert west["confidence"] == "low" and any(i["severity"] == "warning" and i["element"] == west["id"] for i in s.qa)


def test_stamp_numbers_and_void_names():
    item = lambda t, y, r: {"text": t, "box": (0, y, 40, y + 10), "conf": 1.0, "source": "ocr", "angle": 0, "height": 10, "role": r}
    area, part, number = stamp_area([item("Büro", 0, "room stamp"), item("1.12", 12, "number"), item("24.50", 24, "number")])
    assert area == 24.5 and part["text"] == "24.50" and number == "1.12"
    assert stamp_area([item("Halle", 0, "room stamp"), item("18 qm", 12, "number")])[0] == 18.0
    assert stamp_area([item("Halle", 0, "room stamp"), item("204", 12, "number")]) == (None, None, "204")
    parts = [item("Luft-", 0, "room stamp"), item("raum", 12, "room stamp")]
    assert cluster_stamps(parts) == [] and all(p["role"] == "void label" for p in parts)
    stamps = cluster_stamps([item("Raum", 0, "room stamp"), item("12.5 m²", 12, "number")])
    assert stamps[0]["name"] == "Raum" and stamps[0]["area"] == 12.5


def test_degree2_nodes_are_contracted():
    import networkx as nx
    G = nx.MultiGraph()
    G.add_node(0, xy=np.array([10.0, 30.0]), kind="end")
    G.add_node(1, xy=np.array([150.0, 30.0]), kind="junction")        # a junction whose spur was pruned
    G.add_node(2, xy=np.array([290.0, 30.0]), kind="end")
    G.add_edge(0, 1, path=np.array([[10.0, 30.0], [80.0, 30.0], [150.0, 30.0]]))
    G.add_edge(2, 1, path=np.array([[290.0, 30.0], [220.0, 30.0], [150.0, 30.0]]))   # stored the other way round
    contract_degree2(G)
    assert G.number_of_edges() == 1 and set(G.nodes) == {0, 2}
    (_, _, d), = G.edges(data=True)
    assert d["path"][0].tolist() == [10.0, 30.0] and d["path"][-1].tolist() == [290.0, 30.0] and len(d["path"]) == 5
