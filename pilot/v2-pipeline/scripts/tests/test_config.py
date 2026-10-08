import json
import re
from dataclasses import fields
from pathlib import Path

import pytest

from fpx.config import DEFAULT, Config


def test_units():
    assert DEFAULT.px(0.03) == 1.5
    assert DEFAULT.px2(0.05) == 125
    assert DEFAULT.m2(2500) == 1.0


def test_from_file(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"slit_close": 0.25, "connect_probe": [0.2, 0.5]}))
    cfg = Config.from_file(p)
    assert cfg.slit_close == 0.25 and cfg.connect_probe == (0.2, 0.5)
    assert cfg.room_min_area == DEFAULT.room_min_area
    p.write_text(json.dumps({"no_such_threshold": 1}))
    with pytest.raises(ValueError):
        Config.from_file(p)


def test_every_threshold_has_a_reason():
    """Each field is declared on one line with a comment that says why it has its value."""
    src = (Path(__file__).resolve().parents[1] / "fpx/config.py").read_text(encoding="utf-8")
    for f in fields(Config):
        line = re.search(rf"^\s+{f.name}: .*$", src, re.M)
        assert line and "#" in line.group(0), f.name
