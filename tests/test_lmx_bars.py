"""What a bar of LMX leaves for the next one: the clefs, key and time, and the notes tied over."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdf2musicxml.lmx_bars import State, standalone, tied_over
from test_lmx_accuracy import BAR, CONTEXT

DATASET = Path(__file__).resolve().parents[1] / "data" / "bars_lmx" / "bars.jsonl"


def test_a_state_reads_back_as_its_tokens() -> None:
    s = State.parse(CONTEXT)
    assert s.key == "key:fifths:1" and s.time == ("beats:2", "beat-type:4")
    assert s.clefs == {"staff:1": "clef:G2", "staff:2": "clef:G2"} and not s.ties
    assert " ".join(s.tokens()) == CONTEXT


def test_a_bar_leaves_its_key_time_and_clefs_and_its_ties() -> None:
    bar = "measure key:fifths:-3 time beats:3 beat-type:4 C5 voice:1 half staff:1 clef:F4 staff:1 E3 quarter tied:start"
    after = State.parse(CONTEXT).after(bar.split())
    assert after.key == "key:fifths:-3" and after.time == ("beats:3", "beat-type:4")
    assert after.clefs == {"staff:1": "clef:F4", "staff:2": "clef:G2"}  # changed mid-bar
    assert after.ties == [("E3", "staff:1")]


def test_only_ties_over_the_barline_are_carried() -> None:
    assert tied_over(BAR.split()) == [("D6", "staff:1")]
    within = "measure C4 voice:1 quarter staff:1 tied:start C4 quarter tied:stop"
    assert tied_over(within.split()) == []
    # A tie on the other staff, at the same pitch, is another note's
    assert tied_over("measure C4 voice:1 quarter staff:1 tied:start C4 quarter staff:2 tied:stop".split()) == [("C4", "staff:1")]


def test_a_bar_alone_gets_its_state_in_front_and_keeps_what_it_sets_itself() -> None:
    bar = "measure key:fifths:-2 C4 voice:1 half staff:1".split()
    alone = standalone(bar, State.parse(CONTEXT))
    assert alone[:9] == ["measure", "key:fifths:-2", "time", "beats:2", "beat-type:4", "clef:G2", "staff:1",
                         "clef:G2", "staff:2"]
    assert alone[9:] == ["C4", "voice:1", "half", "staff:1"]


@pytest.mark.skipif(not DATASET.exists(), reason="the bar dataset is made by tools/bar_dataset.py")
def test_each_bars_state_is_what_the_bar_before_left() -> None:
    """In the bar dataset, each bar's context is the Linearizer's own state where the bar starts:
    what `State.after` makes of the bar before must be the same."""
    previous, checked = None, 0
    with DATASET.open(encoding="utf-8") as f:
        for k, line in enumerate(f):
            r = json.loads(line)
            if previous and (previous["piece"], previous["style"], previous["bar"] + 1) == (r["piece"], r["style"], r["bar"]):
                after = State.parse(previous["context"]).after(previous["lmx"].split(), ties=False)
                assert after == State.parse(r["context"]), (r["image"], previous["lmx"], r["context"])
                checked += 1
            previous = r
            if k > 20000:
                break
    assert checked > 1000
