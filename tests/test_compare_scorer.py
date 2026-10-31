"""The comparison scorer (evaluation/compare) on the cases a review of it turned up: a failed run of the pipeline stays in
the benchmark, measures align by sounding pitch, a third part's notes are counted, and the voices' order cannot change a
figure. Each case goes through real MusicXML read by music21, as a run does."""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "compare"))
import build_manifest  # noqa: E402
import f1_models as fm  # noqa: E402
import score_direct as sd  # noqa: E402

pytest.importorskip("music21")

TYPES = {4: "whole", 2: "half", 1: "quarter", 0.5: "eighth"}


def score(parts: list[list[list[tuple]]]) -> str:
    """A 4/4 MusicXML score; parts[p][m] is measure m of part p, a list of (step, alter, octave, quarters, voice). A change
    of voice backs up to the measure's start."""
    part_list = "".join(f'<score-part id="P{p}"><part-name>p{p}</part-name></score-part>' for p in range(1, len(parts) + 1))
    body = ""
    for p, measures in enumerate(parts, 1):
        body += f'<part id="P{p}">'
        for k, notes in enumerate(measures, 1):
            attrs = ("<attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
                     "<clef><sign>G</sign><line>2</line></clef></attributes>") if k == 1 else ""
            body += f'<measure number="{k}">{attrs}'
            voice = None
            for step, alter, octave, quarters, v in notes:
                if voice is not None and v != voice:
                    body += "<backup><duration>16</duration></backup>"
                voice = v
                body += (f"<note><pitch><step>{step}</step>{f'<alter>{alter}</alter>' if alter else ''}<octave>{octave}</octave></pitch>"
                         f"<duration>{int(quarters * 4)}</duration><voice>{v}</voice><type>{TYPES[quarters]}</type></note>")
            body += "</measure>"
        body += "</part>"
    return f'<?xml version="1.0" encoding="UTF-8"?><score-partwise version="3.1"><part-list>{part_list}</part-list>{body}</score-partwise>'


def write(folder: Path, name: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(text, encoding="utf-8")
    return folder / name


def f1s(read: list[list], truth: list[list]) -> dict[str, float]:
    r = fm.derive(fm.count(read, truth))
    return {c: r[f"{c}_F1"] for c in fm.CRITERIA}


C4 = ("C", 0, 4, 1, 1)


# ---------------------------------------------------------------------------------------------
# 1. A failed run of the pipeline stays in the benchmark, like a competitor's


def benchmark(tmp_path: Path) -> tuple[Path, Path]:
    """A benchmark folder of two engraved pieces, A and B. Every system was run on both and succeeded on A alone: the
    pipeline wrote A.musicxml and A.json, and on B only its B.json; homr wrote p001.musicxml on A and on B only its
    result.json; oemer was not run at all."""
    here, c13 = tmp_path / "bench", tmp_path / "comparison"
    write(c13, "pieces10.json", json.dumps({"pieces": []}))
    write(c13, "pieces.json", "{}")
    write(here, "pieces_new.json", json.dumps({"pieces": [{"id": p, "title": p, "period": "x"} for p in "AB"]}))
    gold = score([[[C4]]])
    for p in "AB":
        write(here / "pieces" / p, "score.musicxml", gold)
        write(here / "runs" / "ours" / p, f"{p}.json", "{}")
        write(here / "runs" / "homr" / p, "result.json", "{}")
    write(here / "runs" / "ours" / "A", "A.musicxml", gold)
    write(here / "runs" / "homr" / "A", "p001.musicxml", gold)
    return here, c13


def test_a_pipeline_run_that_wrote_nothing_is_an_entry_with_no_files(tmp_path: Path) -> None:
    here, c13 = benchmark(tmp_path)
    manifest = build_manifest.build(here, c13, tmp_path / "asap")
    ours, homr = manifest["systems"]["ours"], manifest["systems"]["homr"]
    assert sorted(ours) == sorted(homr) == ["A", "B"]
    assert ours["A"]["status"] == "ok" and ours["A"]["note_files"] == [str(here / "runs" / "ours" / "A" / "A.musicxml")]
    assert ours["B"]["status"] == "failed" and fm.system_files(ours["B"]) == []
    assert homr["B"]["status"] == "failed" and fm.system_files(homr["B"]) == []
    assert "oemer" not in manifest["systems"]  # never run: absent, not failed
    assert "ours (via its MusicXML)" in manifest["systems"] and sorted(manifest["systems"]["ours (via its MusicXML)"]) == ["A", "B"]


def test_a_run_with_no_folder_is_absent(tmp_path: Path) -> None:
    here, c13 = benchmark(tmp_path)
    (here / "runs" / "ours" / "B" / "B.json").unlink()
    (here / "runs" / "ours" / "B").rmdir()
    manifest = build_manifest.build(here, c13, tmp_path / "asap")
    assert list(manifest["systems"]["ours"]) == ["A"]
    assert any("ours " in s and "not run on 1: B" in s for s in build_manifest.summary(manifest))


def test_the_same_performance_pools_to_the_same_figure(tmp_path: Path) -> None:
    here, c13 = benchmark(tmp_path)
    manifest = build_manifest.build(here, c13, tmp_path / "asap")
    rows, not_run = fm.score_manifest(manifest, ["A", "B"], ["ours", "homr"], True)
    result = fm.build(manifest, ["A", "B"], rows, True, "m.json", not_run)
    for system in ("ours", "homr"):
        pooled = result["pooled"]["four"][system]
        assert pooled["n_pieces"] == 2 and not_run[system] == []
        assert pooled["micro"]["note_F1"] == pytest.approx(2 / 3)
        assert rows[system]["B"]["status"] == "failed" and rows[system]["B"]["note_F1"] == 0.0
        assert any(n.startswith("no output") for n in rows[system]["B"]["notes"])
    assert any(s.startswith("Ran and wrote nothing") and "ours on 1 piece (B)" in s and "homr on 1 piece (B)" in s for s in result["footnotes"])
    assert "B" in build_manifest.summary(manifest)[0] and "scored 0" in build_manifest.summary(manifest)[0]


# ---------------------------------------------------------------------------------------------
# 2. Measures align by sounding pitch and duration, not by spelling or printed note type


@pytest.mark.parametrize("second", [("C", 1, 4, 1, 1), ("D", -1, 4, 1, 1)], ids=["C sharp", "D flat"])
def test_a_c_sharp_aligns_with_the_measure_that_sounds_like_it_however_spelt(tmp_path: Path, second: tuple) -> None:
    read = sd.measures_of(write(tmp_path, "read.musicxml", score([[[("C", 1, 4, 1, 1)]]])), [])
    truth = sd.measures_of(write(tmp_path, "truth.musicxml", score([[[C4], [second]]])), [])
    assert fm.align(read, truth) == [(None, 0), (0, 1)]
    assert f1s(read, truth) == pytest.approx({c: 2 / 3 for c in fm.CRITERIA})


def test_the_alignment_features_are_sounding_pitches_and_musical_durations() -> None:
    e = lambda midi, dur, spelt="C4", kind="quarter": (1, Fraction(0), midi, Fraction(dur), False, spelt, kind)
    bars = [[e(61, 1, "C4"), e(61, 1, "D4"), e(60, Fraction(1, 3), "C4", "eighth"), e(60, Fraction(1, 2), "C4", "eighth")]]
    index = sd.features(bars)
    assert set(index) == {("pitch", 60), ("pitch", 61), ("duration", Fraction(1)), ("duration", Fraction(1, 3)), ("duration", Fraction(1, 2))}
    v = sd.vector(bars[0], index)
    assert v[index["pitch", 61]] == 2 and v[index["pitch", 60]] == 2
    assert v[index["duration", Fraction(1)]] == 1.0 and v[index["duration", Fraction(1, 3)]] == 0.5
    assert sd.similarity([[e(61, 1, "C4")]], [[e(61, 1, "D4")]])[0, 0] == pytest.approx(1.0)
    assert sd.similarity([[e(61, 1, "C4")]], [[e(60, 1, "C4")]])[0, 0] == pytest.approx(0.2)  # the duration alone in common
    assert sd.similarity([], [[e(60, 1)]]).shape == (0, 1) and sd.similarity([], []).shape == (0, 0)


# ---------------------------------------------------------------------------------------------
# 3. A third part's notes are events found


def test_a_third_part_costs_precision(tmp_path: Path) -> None:
    two = [[[C4]], [[("C", 0, 3, 1, 1)]]]
    truth = sd.measures_of(write(tmp_path, "truth.musicxml", score(two)), [])
    notes: list[str] = []
    read = sd.measures_of(write(tmp_path, "read.musicxml", score(two + [[[("G", 0, 5, 1, 1)]]])), notes)
    assert [e[0] for e in read[0]] == [1, 2, 3] and sum(map(len, read)) == 3 and sum(map(len, truth)) == 2
    assert f1s(read, truth) == pytest.approx({c: 0.8 for c in fm.CRITERIA})
    assert notes == ["read.musicxml: 3 parts with notes; staves 1 and 2 are the first two, the rest are extra staves whose events "
                     "count and match nothing under note, pitch and timed F1"]
    # onset F1 takes any staff: a third-staff note that sounds like a true note the first two staves lack does match there
    moved = sd.measures_of(write(tmp_path, "moved.musicxml", score([[[C4]], [[("E", 0, 4, 1, 1)]], [[("C", 0, 3, 1, 1)]]])), [])
    assert f1s(moved, truth) == pytest.approx({"note": 0.4, "pitch": 0.4, "timed": 0.4, "onset": 0.8})


def test_an_empty_part_among_three_does_not_shift_the_staves(tmp_path: Path) -> None:
    # an empty first part: the two parts with notes are staves 1 and 2, as before
    from music21 import converter

    read = sd.measures_of(write(tmp_path, "read.musicxml", score([[[C4]], [[("C", 0, 3, 1, 1)]]])), [])
    three = write(tmp_path, "three.musicxml", score([[[]], [[C4]], [[("C", 0, 3, 1, 1)]]]))
    assert len(converter.parse(str(three)).parts) == 3
    assert sd.measures_of(three, []) == read
    # one or two parts keep their places even when one is empty: the left hand alone is staff 2 (music21 fills the empty
    # upper measure with a whole-measure rest, which is a rest event on staff 1)
    lower = sd.measures_of(write(tmp_path, "lower.musicxml", score([[[]], [[("C", 0, 3, 1, 1)]]])), [])
    assert [e[0] for e in lower[0] if e[2] >= 0] == [2]


# ---------------------------------------------------------------------------------------------
# 4. The reading order does not depend on the voices' order in the file


def test_renumbering_the_voices_changes_no_figure(tmp_path: Path) -> None:
    a = sd.measures_of(write(tmp_path, "a.musicxml", score([[[C4, ("C", 0, 4, 2, 2)]]])), [])
    b = sd.measures_of(write(tmp_path, "b.musicxml", score([[[("C", 0, 4, 2, 1), ("C", 0, 4, 1, 2)]]])), [])
    assert [e[3] for e in a[0]] == [1, 2] and [e[3] for e in b[0]] == [2, 1]  # music21 keeps the file's order
    assert f1s(b, a) == f1s(a, b) == {c: 1.0 for c in fm.CRITERIA}


def test_the_order_key_holds_every_field_in_order_compares() -> None:
    e = lambda onset, midi, dur, grace=False: (1, Fraction(onset), midi, Fraction(0 if grace else dur), grace, "x", "y")
    assert sd.order(e(1, 60, 2)) == (Fraction(1), True, 60, Fraction(2))
    # by onset, a grace note first, then a chord from the bottom up, then by duration
    events = [e(1, 60, 1), e(0, 67, 1), e(0, 60, 2), e(0, 60, 1), e(0, 72, 1, grace=True)]
    assert sorted(events, key=sd.order) == [e(0, 72, 1, grace=True), e(0, 60, 1), e(0, 60, 2), e(0, 67, 1), e(1, 60, 1)]
    for found in ([e(0, 60, 1), e(0, 60, 2)], [e(0, 60, 2), e(0, 60, 1)]):
        assert sd.in_order(found, [e(0, 60, 1), e(0, 60, 2)], lambda x: (x[2], x[3], x[4])) == 2


def test_the_selftest_passes(capsys) -> None:
    assert fm.selftest() == 0
