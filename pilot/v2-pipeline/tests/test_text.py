import pytest

from fpx.text import merge_chars, role


@pytest.mark.parametrize("text, expected", [
    ("KACHELOFEN", "fixture label"),
    ("Kachel-", "fixture label"),
    ("LUFTRAUM", "void label"),
    ("2051.OG.01.003", "room stamp"),                 # AOID
    ("24.50 m2", "number"),
    ("18,2", "number"),
    ("Büro", "room stamp"),
    ("-", "other"),
])
def test_role(text, expected):
    assert role(text) == expected


def test_merge_chars_words_and_lines():
    # (text, x0, y0, x1, y1, size, direction): "Bad" and "WC" on one baseline with a word gap, "Halle" on the next line
    chars = [(c, 10 + 6 * i, 10, 16 + 6 * i, 20, 10, (1.0, 0.0)) for i, c in enumerate("Bad")]
    chars += [(c, 60 + 6 * i, 10, 66 + 6 * i, 20, 10, (1.0, 0.0)) for i, c in enumerate("WC")]
    chars += [(c, 10 + 6 * i, 40, 16 + 6 * i, 50, 10, (1.0, 0.0)) for i, c in enumerate("Halle")]
    words = sorted(w["text"] for w in merge_chars(chars))
    assert words == ["Bad", "Halle", "WC"]
    assert all(w["source"] == "pdf" and w["angle"] == 0 for w in merge_chars(chars))
