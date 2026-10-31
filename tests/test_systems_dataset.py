"""The first bar of a system cut from a dataset of systems: its signatures move into the context."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from systems_dataset import first_bar, safe, sample_page  # noqa: E402

FIRST = ("measure key:fifths:-2 time beats:3 beat-type:4 clef:G2 staff:1 clef:F4 staff:2 "
         "staff:1 C4 quarter staff:2 C3 half")


def test_a_continuation_moves_its_signatures_into_the_context() -> None:
    lmx, context = first_bar(FIRST, song_start=False)
    assert context == "key:fifths:-2 time beats:3 beat-type:4 clef:G2 staff:1 clef:F4 staff:2"
    assert lmx == "measure staff:1 C4 quarter staff:2 C3 half"


def test_a_song_start_keeps_its_lmx_and_has_no_context() -> None:
    assert first_bar(FIRST, song_start=True) == (FIRST, "")


def test_only_the_signatures_it_sets_are_moved() -> None:
    lmx, context = first_bar("measure clef:G2 staff:1 clef:F4 staff:2 staff:1 C4 quarter", song_start=False)
    assert context == "clef:G2 staff:1 clef:F4 staff:2"
    assert lmx == "measure staff:1 C4 quarter"
    assert first_bar("measure staff:1 C4 quarter", song_start=False) == ("measure staff:1 C4 quarter", "")


def test_a_later_signature_stays_in_the_bar() -> None:
    # a clef after the first note is a change within the bar, not part of the opening signatures
    text = "measure key:fifths:0 staff:1 C4 quarter clef:F4 staff:1 D4 quarter"
    lmx, context = first_bar(text, song_start=False)
    assert context == "key:fifths:0" and lmx == "measure staff:1 C4 quarter clef:F4 staff:1 D4 quarter"


def test_names() -> None:
    assert safe("grandstaff/chopin/p1 x.krn") == "grandstaff_chopin_p1_x_krn"
    assert sample_page("samples/123/p3-s2") == 3 and sample_page("beethoven/sonata/min3_up_m-1-4") == 0
