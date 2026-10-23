"""Reassembly: bars read one by one stitched back into one score of MusicXML.

Each bar's LMX starts with `measure`, so a score's LMX is its bars' tokens one after another; the
first bar is given the state the score starts from in front (`lmx_bars.standalone`), in case it
does not set its own clefs, key and time. LMX's own Delinearizer makes the part of it, and
`part_to_score` the score round it.
"""

from __future__ import annotations

import io
import math
from fractions import Fraction
import xml.etree.ElementTree as ET
from pathlib import Path

from .lmx_bars import State, standalone


def stitch(bars: list[list[str]], start: State | None = None) -> list[str]:
    """A score's LMX, from its bars in order."""
    tokens: list[str] = []
    for k, bar in enumerate(bars):
        bar = bar if bar[:1] == ["measure"] else ["measure", *bar]
        tokens += standalone(bar, start or State()) if k == 0 else bar
    return tokens


def _tuplet_marks(note: ET.Element) -> tuple[bool, bool]:
    kinds = {t.get("type") for t in note.iter("tuplet")}
    return "start" in kinds, "stop" in kinds


def _make_plain(note: ET.Element) -> None:
    """Take the tuplet off a note, its duration made the plain value."""
    tm = note.find("time-modification")
    for t in list(note.iter("tuplet")):
        for parent in note.iter():
            if t in list(parent):
                parent.remove(t)
                break
    for notations in note.findall("notations"):
        if len(notations) == 0:
            note.remove(notations)
    if tm is None:
        return
    actual, normal = int(tm.findtext("actual-notes")), int(tm.findtext("normal-notes"))
    note.remove(tm)
    d = note.find("duration")
    if d is not None:
        v = Fraction(d.text) * actual / normal
        d.text = str(int(v)) if v.denominator == 1 else str(v)


def _group_is_whole(group: list[ET.Element]) -> bool:
    """Uniform and complete: one ratio, one note type and dots, as many notes as the ratio's actual-notes."""
    ratios = set()
    for n in group:
        tm = n.find("time-modification")
        if tm is None:
            return False
        ratios.add((tm.findtext("actual-notes"), tm.findtext("normal-notes")))
    primary = [n for n in group if n.find("chord") is None]
    shapes = {(n.findtext("type"), len(n.findall("dot"))) for n in primary}
    return len(ratios) == 1 and len(shapes) == 1 and len(primary) == int(next(iter(ratios))[0])


def repair(score: ET.ElementTree) -> dict:
    """Make the MusicXML importable (MuseScore refuses it otherwise); counts of what was changed.

    Every measure gets a number, from 1 in each part. In each measure the notes are walked in order: a
    tuplet group runs from a note that starts a tuplet to the note that stops it. A closed group is kept
    only if uniform and complete; an unclosed one, a note stopping and starting at once, and a note with a
    time modification outside any bracket are made plain.
    """
    counts = {"measures_numbered": 0, "tuplet_groups": 0, "tuplet_groups_made_plain": 0,
              "stray_notes_made_plain": 0, "incomplete_tuplet_groups": 0}

    def close(group: list[ET.Element]) -> None:
        counts["tuplet_groups"] += 1
        if not _group_is_whole(group):
            counts["tuplet_groups_made_plain"] += 1
            for n in group:
                _make_plain(n)

    for part in score.getroot().iter("part"):
        for k, measure in enumerate(part.findall("measure"), 1):
            if measure.get("number") != str(k):
                measure.set("number", str(k))
                counts["measures_numbered"] += 1
            group: list[ET.Element] | None = None
            for note in measure.findall("note"):
                starts, stops = _tuplet_marks(note)
                if group is None:
                    if starts and not stops:
                        group = [note]
                    elif starts or stops or note.find("time-modification") is not None:
                        counts["stray_notes_made_plain"] += 1
                        _make_plain(note)
                elif starts and stops:
                    close(group)
                    group = None
                    counts["stray_notes_made_plain"] += 1
                    _make_plain(note)
                elif starts:
                    counts["tuplet_groups"] += 1
                    counts["tuplet_groups_made_plain"] += 1
                    for n in group:
                        _make_plain(n)
                    group = [note]
                else:
                    group.append(note)
                    if stops:
                        close(group)
                        group = None
            if group is not None:
                counts["tuplet_groups"] += 1
                counts["tuplet_groups_made_plain"] += 1
                for n in group:
                    _make_plain(n)
        _scale_divisions(part)
    counts["incomplete_tuplet_groups"] = counts["tuplet_groups_made_plain"] + counts["stray_notes_made_plain"]
    return counts


def _scale_divisions(part: ET.Element) -> None:
    """Make every duration a whole number of divisions again, after plain durations came out fractional."""
    scale = 1
    for d in part.iter("duration"):
        scale = math.lcm(scale, Fraction(d.text).denominator)
    if scale == 1:
        return
    for d in part.iter("duration"):
        d.text = str(int(Fraction(d.text) * scale))
    for d in part.iter("divisions"):
        d.text = str(int(Fraction(d.text) * scale))


_repair = repair


def to_musicxml(bars: list[list[str]], start: State | None = None, repair: bool = True) -> tuple[ET.ElementTree, str]:
    """The score as MusicXML (repaired to be importable), and whatever the Delinearizer objected to."""
    from lmx.linearization.Delinearizer import Delinearizer
    from lmx.symbolic.part_to_score import part_to_score

    err = io.StringIO()
    d = Delinearizer(errout=err)
    d.process_text(" ".join(stitch(bars, start)))
    score = part_to_score(d.part_element)
    if repair:
        _repair(score)
    return score, err.getvalue()


def write(bars: list[list[str]], path: Path | str, start: State | None = None, repair: bool = True) -> str:
    """Write the score; the Delinearizer's objections, then (when repaired) a line of what was changed."""
    score, errors = to_musicxml(bars, start, repair=False)
    if repair:
        counts = _repair(score)
        if counts["incomplete_tuplet_groups"]:
            errors += f"{counts['incomplete_tuplet_groups']} incomplete tuplet groups made plain\n"
    score.write(str(path), encoding="utf-8", xml_declaration=True)
    return errors
