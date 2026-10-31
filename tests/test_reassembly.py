"""Bars read one by one, stitched back into one score."""

from __future__ import annotations

from transcriber.lmx_bars import State, standalone
from transcriber.reassembly import to_musicxml
from test_lmx_accuracy import BAR, CONTEXT


def test_bars_stitch_into_one_score_that_reads_back_as_the_bars() -> None:
    from lmx.linearization.Linearizer import Linearizer

    second = BAR.replace("D5 voice:1 eighth stem:down staff:1 slur:stop", "D6 voice:1 eighth stem:down staff:1 tied:stop", 1)
    bars = [BAR.split(), second.split()]
    score, errors = to_musicxml(bars, State.parse(CONTEXT))
    assert not errors
    part = score.getroot().find("part")
    assert len(part.findall("measure")) == 2 and score.getroot().find("part-list") is not None
    again = Linearizer()
    again.process_part(part)
    # The first bar with the clefs, key and time it starts from in front, as a score's first bar has them
    assert again.output_tokens == standalone(bars[0], State.parse(CONTEXT)) + bars[1]


def test_a_first_bar_that_sets_its_own_key_keeps_it() -> None:
    bar = BAR.replace("measure", "measure key:fifths:3", 1).split()
    score, _ = to_musicxml([bar], State.parse(CONTEXT))
    assert score.getroot().find("part/measure/attributes/key/fifths").text == "3"


# --- repair: what MuseScore needs to open the file ---

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from transcriber.reassembly import repair, write

MUSESCORE = Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe")
TRIPLET = ("measure key:fifths:0 time beats:2 beat-type:4 clef:G2 staff:1 "
           "C5 voice:1 eighth 3in2 tuplet:start staff:1 D5 eighth 3in2 staff:1 "
           "E5 eighth 3in2 tuplet:stop staff:1 F5 quarter staff:1 G5 eighth staff:1")
LONE = "measure C5 voice:1 eighth 3in2 tuplet:start staff:1"


def _convert(path: Path) -> int:
    return subprocess.run([str(MUSESCORE), "-o", str(path.with_suffix(".mxl")), str(path)],
                          capture_output=True, timeout=300).returncode


def _durations(score) -> list[str]:
    return [n.findtext("duration") for n in score.getroot().iter("note")]


def test_measures_are_numbered_from_one() -> None:
    score, _ = to_musicxml([BAR.split(), BAR.split(), BAR.split()], State.parse(CONTEXT))
    assert [m.get("number") for m in score.getroot().find("part").findall("measure")] == ["1", "2", "3"]


def test_a_complete_triplet_keeps_its_tuplet() -> None:
    score, _ = to_musicxml([TRIPLET.split()], repair=False)
    counts = repair(score)
    assert counts["incomplete_tuplet_groups"] == 0
    assert len(list(score.getroot().iter("time-modification"))) == 3
    assert len(list(score.getroot().iter("tuplet"))) == 2


def test_a_bar_ending_after_one_triplet_eighth_makes_it_plain() -> None:
    raw, _ = to_musicxml([LONE.split()], repair=False)
    before = _durations(raw)
    score, _ = to_musicxml([LONE.split()])
    root = score.getroot()
    assert root.find(".//time-modification") is None and root.find(".//tuplet") is None
    # A plain eighth is half a quarter, a triplet eighth a third of one
    from fractions import Fraction
    divisions = lambda sc: Fraction(sc.getroot().findtext(".//divisions"))
    assert Fraction(_durations(score)[0]) / divisions(score) == Fraction(1, 2)
    assert Fraction(before[0]) / divisions(raw) == Fraction(1, 3)
    counts = repair(raw)
    assert counts["tuplet_groups_made_plain"] == 1 and counts["tuplet_groups"] == 1


@pytest.mark.skipif(not MUSESCORE.exists(), reason="MuseScore 4 is not installed")
def test_musescore_opens_a_bar_that_ends_in_a_triplet(tmp_path: Path) -> None:
    path = tmp_path / "lone.musicxml"
    write([LONE.split()], path)
    assert _convert(path) == 0


@pytest.mark.skipif(not MUSESCORE.exists(), reason="MuseScore 4 is not installed")
def test_musescore_opens_the_repaired_scriabin_score(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "score.musicxml"
    if not source.exists():
        pytest.skip("score.musicxml is not here")
    tree = ET.parse(source)
    repair(tree)
    path = tmp_path / "scriabin.musicxml"
    tree.write(str(path), encoding="utf-8", xml_declaration=True)
    assert _convert(path) == 0


# --- repair: what PageTurner's loader (partitura) and music21 need to load the file ---

# Both found in the 50-piece following test: a chord note straight after a rest (partitura: 'Rest' has no is_grace_chord)
# and a note with an accidental that partitura cannot name and no <alter> (KeyError 'flat-flat').
CHORD_AFTER_REST = ("measure key:fifths:0 time beats:2 beat-type:4 clef:G2 staff:1 "
                    "C5 voice:1 quarter staff:1 rest quarter staff:1 chord E5 quarter staff:1 chord G5 quarter staff:1")
DOUBLE_FLATS = ("measure key:fifths:0 time beats:2 beat-type:4 clef:G2 staff:1 B4 voice:1 quarter flat-flat staff:1 "
                "C5 quarter double-sharp staff:1")
TIE_THEN_FLAT_FLAT = ("measure key:fifths:0 time beats:2 beat-type:4 clef:G2 staff:1 "
                      "A4 voice:1 quarter staff:1 B4 quarter tied:start staff:1")
FLAT_FLAT_TIE_STOP = "measure B4 voice:1 quarter flat-flat tied:stop staff:1 C5 quarter double-sharp staff:1"


def _notes_of(path: Path) -> list[ET.Element]:
    return list(ET.parse(path).getroot().iter("note"))


def _chords_follow_pitched_notes(notes: list[ET.Element]) -> bool:
    """Every <chord/> note comes straight after a pitched note, or after another chord note of such a run."""
    prev = None
    for n in notes:
        if n.find("chord") is not None and (prev is None or prev.find("pitch") is None or n.find("pitch") is None):
            return False
        prev = n
    return True


def test_a_chord_note_after_a_rest_takes_the_rests_place(tmp_path: Path) -> None:
    bar = CHORD_AFTER_REST.split()
    raw, _ = to_musicxml([bar], repair=False)
    raw_notes = list(raw.getroot().iter("note"))
    assert raw_notes[1].find("rest") is not None and raw_notes[2].find("chord") is not None  # as the Delinearizer writes it
    assert not _chords_follow_pitched_notes(raw_notes)
    path = tmp_path / "chord_after_rest.musicxml"
    errors = write([bar], path)
    notes = _notes_of(path)  # parsed back from the file
    assert _chords_follow_pitched_notes(notes)
    assert [n.findtext("pitch/step") for n in notes] == ["C", "E", "G"]  # the rest gave way to the chord's first note
    assert [n.find("chord") is not None for n in notes] == [False, False, True]
    assert all(n.find("rest") is None for n in notes)
    assert [n.findtext("duration") for n in notes] == [raw_notes[0].findtext("duration")] * 3
    assert "took the rest's place" in errors


def test_a_chord_note_with_nothing_before_it_loses_its_chord_flag() -> None:
    bar = "measure key:fifths:0 time beats:2 beat-type:4 clef:G2 staff:1 chord E5 voice:1 quarter staff:1 G5 quarter staff:1".split()
    raw, _ = to_musicxml([bar], repair=False)
    assert raw.getroot().find(".//chord") is not None
    counts = repair(raw)
    assert raw.getroot().find(".//chord") is None and counts["chord_flags_dropped"] == 1


def test_a_chord_after_a_pitched_note_is_left_alone() -> None:
    raw, _ = to_musicxml([BAR.split()], State.parse(CONTEXT), repair=False)
    assert raw.getroot().find(".//chord") is not None
    raw.getroot().find("part/measure").set("number", "1")  # the one thing repair does to a bar like this
    before = ET.tostring(raw.getroot())
    counts = repair(raw)
    assert counts["chord_after_rest_made_first"] == counts["chord_flags_dropped"] == counts["alters_added"] == 0
    assert ET.tostring(raw.getroot()) == before


def test_a_double_flat_keeps_musicxmls_name_and_a_note_without_alter_is_given_one(tmp_path: Path) -> None:
    path = tmp_path / "double_flat.musicxml"
    errors = write([TIE_THEN_FLAT_FLAT.split(), FLAT_FLAT_TIE_STOP.split()], path)
    first, second, tied_stop, sharp2 = _notes_of(path)
    # MusicXML's accidental-value for a double flat is flat-flat (double-flat is not a member of it): not renamed
    assert tied_stop.findtext("accidental") == "flat-flat" and sharp2.findtext("accidental") == "double-sharp"
    # The tie makes the writer drop the second B's <alter> (a tied note has its first note's pitch): the note keeps that
    # pitch, now spelt out, so that a loader that reads the accidental when it finds no <alter> never has to
    assert tied_stop.findtext("pitch/alter") == "0" and tied_stop.findtext("pitch/step") == "B"
    assert sharp2.findtext("pitch/alter") == "2"
    assert "explicit <alter>0</alter>" in errors


def test_every_accidental_partitura_cannot_name_comes_with_an_alter(tmp_path: Path) -> None:
    from transcriber.reassembly import PARTITURA_ACCIDENTALS

    path = tmp_path / "accidentals.musicxml"
    write([TIE_THEN_FLAT_FLAT.split(), FLAT_FLAT_TIE_STOP.split(), DOUBLE_FLATS.split()], path)
    for n in _notes_of(path):
        accidental = n.findtext("accidental")
        if accidental is not None and accidental not in PARTITURA_ACCIDENTALS:
            assert n.find("pitch/alter") is not None, accidental


def test_music21_loads_the_repaired_reading(tmp_path: Path) -> None:
    m21 = pytest.importorskip("music21")
    path = tmp_path / "reading.musicxml"
    write([CHORD_AFTER_REST.split(), TIE_THEN_FLAT_FLAT.split(), FLAT_FLAT_TIE_STOP.split(), DOUBLE_FLATS.split()], path)
    score = m21.converter.parse(str(path))
    names = [p.nameWithOctave for n in score.flatten().notes for p in (n.pitches if n.isChord else [n.pitch])]
    assert "C5" in names and "E5" in names and "G5" in names  # the chord that followed a rest is there
    assert "B--4" in names and "C##5" in names  # flat-flat is a double flat, double-sharp a double sharp
    assert "B4" in names  # the tied B keeps the pitch it was tied to


def test_partitura_loads_the_repaired_reading(tmp_path: Path) -> None:
    partitura = pytest.importorskip("partitura")  # PageTurner's loader; installed in WSL ~/ptvenv, not in the Windows venv
    path = tmp_path / "reading.musicxml"
    write([CHORD_AFTER_REST.split(), TIE_THEN_FLAT_FLAT.split(), FLAT_FLAT_TIE_STOP.split(), DOUBLE_FLATS.split()], path)
    assert len(partitura.load_musicxml(str(path)).parts) == 1
