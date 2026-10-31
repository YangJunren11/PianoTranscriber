"""MusicXML to LMX bars for the comparison with other OMR systems: the repairs that make a competitor's MusicXML readable by
LMX's Linearizer, the merge of two parts into one piano part, and `bars_of_file`. Taken unchanged from the research
repository's experiments/13_comparison/compare.py (the functions from _duration_at_end to revoice)."""

from __future__ import annotations

import copy
import re
import sys
import xml.etree.ElementTree as ET
from fractions import Fraction
from math import lcm  # noqa: F401
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pdf2musicxml  # noqa: E402,F401  (puts third_party/lmx on the path)


def _duration_at_end(measure: ET.Element) -> int:
    """Where the voices stand at the measure's end, in its divisions (the sum of forward/backup/notes)."""
    t = 0
    for e in measure:
        if e.tag == "note" and e.find("chord") is None and e.find("grace") is None:
            t += int(float(e.findtext("duration") or 0))
        elif e.tag == "forward":
            t += int(float(e.findtext("duration") or 0))
        elif e.tag == "backup":
            t -= int(float(e.findtext("duration") or 0))
    return t


def _set_staff(note: ET.Element, staff: int) -> None:
    existing = note.find("staff")
    if existing is not None:
        existing.text = str(staff)
        return
    new = ET.Element("staff")
    new.text = str(staff)
    for k, child in enumerate(note):
        if child.tag in ("beam", "notations", "lyric", "play"):
            note.insert(k, new)
            return
    note.append(new)


def merge_parts(upper: ET.Element, lower: ET.Element, warnings: list[str]) -> ET.Element:
    """Two parts of one staff each as one part on two staves: the first is the right hand (staff 1), the second
    the left (staff 2, its voices numbered 5 and up as MuseScore numbers them)."""
    part = copy.deepcopy(upper)
    lowers = lower.findall("measure")
    div1 = div2 = 1
    for k, m1 in enumerate(part.findall("measure")):
        for d in m1.iter("divisions"):
            div1 = int(float(d.text))
        for n in m1.findall("note"):
            _set_staff(n, 1)
        attrs = m1.find("attributes")
        if k == 0 or attrs is not None:
            if attrs is None:
                attrs = ET.Element("attributes")
                m1.insert(0, attrs)
            if attrs.find("staves") is None:
                st = ET.Element("staves")
                st.text = "2"
                at = [i for i, c in enumerate(attrs) if c.tag in ("divisions", "key", "time")]
                attrs.insert(at[-1] + 1 if at else 0, st)
            for c in attrs.findall("clef"):
                c.set("number", "1")
        if k >= len(lowers):
            continue
        m2 = lowers[k]
        for d in m2.iter("divisions"):
            div2 = int(float(d.text))
        scale = Fraction(div1, div2)
        if scale.denominator != 1:
            warnings.append(f"measure {k + 1}: divisions {div1} and {div2} cannot be put together")
            scale = Fraction(1)
        s = int(scale)
        end = _duration_at_end(m1)
        if end:
            back = ET.SubElement(m1, "backup")
            ET.SubElement(back, "duration").text = str(end)
        for e in m2:
            if e.tag == "attributes":
                target = m1.find("attributes")
                if target is None:
                    target = ET.Element("attributes")
                    m1.insert(0, target)
                for c in e.findall("clef"):
                    c = copy.deepcopy(c)
                    c.set("number", "2")
                    target.append(c)
            elif e.tag in ("note", "backup", "forward"):
                e = copy.deepcopy(e)
                d = e.find("duration")
                if d is not None and s != 1:
                    d.text = str(int(float(d.text)) * s)
                if e.tag == "note":
                    _set_staff(e, 2)
                    v = e.find("voice")
                    if v is not None and (v.text or "").isdigit():
                        v.text = str(int(v.text) + 4)
                m1.append(e)
    return part


def only_measures(part: ET.Element) -> ET.Element:
    """The part with nothing but its measures (oemer writes a <sound tempo> before the first one, which LMX's Linearizer
    asserts against)."""
    clean = ET.Element(part.tag, part.attrib)
    clean.extend(m for m in part if m.tag == "measure")
    FIXES["measure_rests_unmarked"] += fix_measure_rests(clean)
    FIXES["types_added"] += add_missing_types(clean)
    FIXES["chords_reordered"] += sort_chords(clean)
    if clean.find(".//accidental") is None:  # only a file with no printed signs at all (homr, oemer): ours and Audiveris's are left as read
        FIXES["accidentals_added"] += add_accidentals(clean)
    return clean


FIXES = {"measure_rests_unmarked": 0, "types_added": 0, "chords_reordered": 0, "accidentals_added": 0}  # what the conversion had to repair in the file being read

ACCIDENTAL_OF = {1: "sharp", -1: "flat", 0: "natural", 2: "double-sharp", -2: "flat-flat"}
ALTER_OF = {"sharp": 1, "flat": -1, "natural": 0, "double-sharp": 2, "flat-flat": -2, "natural-sharp": 1, "natural-flat": -1}
BEFORE_ACCIDENTAL = ("time-modification", "stem", "notehead", "notehead-text", "staff", "beam", "notations", "lyric", "play")


def add_accidentals(part: ET.Element) -> int:
    """An <accidental> given to each note whose <alter> the key and the bar's earlier accidentals do not already make
    (FAIRNESS.md). LMX writes only the printed sign and its Delinearizer works the pitch out again from the key and the signs
    (lmx.symbolic.PitchAlternator, mimicked here: an accidental holds for its staff, step and octave from its onset to the
    bar's end), so a note with the right <alter> and no <accidental> (homr and oemer write none at all) came back with the
    key's pitch. A file whose signs already make its pitches (ours, Audiveris's) is left as it is. How many were added."""
    sharps, flats = "FCGDAEB", "BEADGCF"
    fifths, added = 0, 0
    for measure in part.findall("measure"):
        onset, timed = Fraction(0), []  # (onset, document order, note)
        for e in measure:
            if e.tag == "attributes" and e.findtext("key/fifths"):
                fifths = int(float(e.findtext("key/fifths")))
            d = e.findtext("duration")
            if e.tag == "note":
                chord = e.find("chord") is not None
                dur = Fraction(int(float(d))) if d else Fraction(0)
                timed.append((onset - dur if chord else onset, len(timed), e))
                if not chord:
                    onset += dur
            elif e.tag in ("backup", "forward") and d:
                onset += Fraction(int(float(d))) * (-1 if e.tag == "backup" else 1)
        signs: dict[tuple, list[tuple[Fraction, int]]] = {}
        for at, _, note in sorted(timed, key=lambda x: (x[0], x[1])):
            pitch = note.find("pitch")
            if pitch is None or pitch.findtext("step") is None or pitch.findtext("octave") is None:
                continue
            step = pitch.findtext("step").strip()
            where = (note.findtext("staff") or "1", step, pitch.findtext("octave").strip())
            sign = note.find("accidental")
            if sign is not None:
                if sign.text in ALTER_OF:
                    signs.setdefault(where, []).append((at, ALTER_OF[sign.text]))
                continue
            if any(t.get("type") == "stop" for t in note.findall("tie")):
                continue  # the Delinearizer gives a tied-over note its first note's pitch
            carried = [a for o, a in signs.get(where, []) if o <= at]
            expected = carried[-1] if carried else (1 if step in sharps[:max(fifths, 0)] else -1 if step in flats[:max(-fifths, 0)] else 0)
            alter = int(float(pitch.findtext("alter") or 0))
            if alter == expected or alter not in ACCIDENTAL_OF:
                continue
            new = ET.Element("accidental")
            new.text = ACCIDENTAL_OF[alter]
            k = next((i for i, c in enumerate(note) if c.tag in BEFORE_ACCIDENTAL), len(note))
            note.insert(k, new)
            signs.setdefault(where, []).append((at, alter))
            added += 1
    return added

STEPS = "CDEFGAB"


def sort_chords(part: ET.Element) -> int:
    """Each chord's notes put bottom up (FAIRNESS.md): LMX's Linearizer refuses a bar whose chord is written in any other
    order ("Chord notes must have ascending pitches"), though MusicXML leaves the order free and it means nothing; homr and
    oemer write chords top down, and without this most of their bars were scored as empty. The first note of the chord
    keeps no <chord/>, the others get one. How many chords were reordered."""
    def height(note: ET.Element) -> tuple:
        step, octave = note.findtext("pitch/step"), note.findtext("pitch/octave")
        if step is None or octave is None or step.strip() not in STEPS:
            return (99, 0)  # a rest or an unpitched note: left at the top
        return (int(octave), STEPS.index(step.strip()))

    reordered = 0
    for measure in part.findall("measure"):
        children = list(measure)
        k = 0
        while k < len(children):
            e = children[k]
            if e.tag != "note" or e.find("chord") is not None:
                k += 1
                continue
            group = [e]
            while k + len(group) < len(children) and children[k + len(group)].tag == "note" \
                    and children[k + len(group)].find("chord") is not None:
                group.append(children[k + len(group)])
            ordered = sorted(group, key=height)  # stable: notes of one height keep their order
            if len(group) > 1 and [id(n) for n in ordered] != [id(n) for n in group]:
                reordered += 1
                for n in group:
                    for c in n.findall("chord"):
                        n.remove(c)
                for j, n in enumerate(ordered):
                    if j:
                        at = next((i for i, c in enumerate(n) if c.tag not in ("grace", "cue")), len(n))
                        n.insert(at, ET.Element("chord"))
                first = list(measure).index(group[0])
                for n in group:
                    measure.remove(n)
                for j, n in enumerate(ordered):
                    measure.insert(first + j, n)
                children = list(measure)
            k += len(group)
    return reordered

TYPES = {Fraction(8): "breve", Fraction(4): "whole", Fraction(2): "half", Fraction(1): "quarter", Fraction(1, 2): "eighth",
         Fraction(1, 4): "16th", Fraction(1, 8): "32nd", Fraction(1, 16): "64th", Fraction(1, 32): "128th"}
AFTER_TYPE = ("dot", "accidental", "time-modification", "stem", "notehead", "staff", "beam", "notations", "lyric")


def add_missing_types(part: ET.Element) -> int:
    """A note with no <type> (oemer writes none) is given the one its duration says, with its dots, or a triplet's
    (3 in the time of 2); LMX's Linearizer needs it. How many were added; a duration that fits none is left, and the
    Linearizer then refuses that bar."""
    divisions, added = None, 0
    for measure in part.findall("measure"):
        for d in measure.iter("divisions"):
            divisions = int(float(d.text))
        for note in measure.findall("note"):
            duration = note.find("duration")
            if note.find("type") is not None or duration is None or not divisions or note.find("grace") is not None:
                continue
            q = Fraction(int(float(duration.text)), divisions)
            dots = len(note.findall("dot"))
            base, tuplet = q / (2 - Fraction(1, 2 ** dots)), False
            if base not in TYPES and base * Fraction(3, 2) in TYPES and not dots:
                base, tuplet = base * Fraction(3, 2), True
            if base not in TYPES:
                continue
            kind = ET.Element("type")
            kind.text = TYPES[base]
            at = next((k for k, c in enumerate(note) if c.tag in AFTER_TYPE), len(note))
            note.insert(at, kind)
            if tuplet and note.find("time-modification") is None:
                mod = ET.Element("time-modification")
                ET.SubElement(mod, "actual-notes").text = "3"
                ET.SubElement(mod, "normal-notes").text = "2"
                note.insert(at + 1 + len(note.findall("dot")), mod)
            added += 1
    return added


def fix_measure_rests(part: ET.Element) -> int:
    """A rest marked measure="yes" is kept only where it lasts the whole bar (the time signature in force and the divisions say
    how long that is); oemer marks every rest so, and LMX's Linearizer refuses a bar with such a rest. How many were unmarked."""
    divisions = beats = beat_type = None
    fixed = 0
    for measure in part.findall("measure"):
        for d in measure.iter("divisions"):
            divisions = int(float(d.text))
        for t in measure.iter("time"):
            if t.findtext("beats") and t.findtext("beat-type") and t.findtext("beats").strip().isdigit():
                beats, beat_type = int(t.findtext("beats")), int(t.findtext("beat-type"))
        whole = divisions * 4 * beats // beat_type if divisions and beats and beat_type else None
        for note in measure.findall("note"):
            rest, duration = note.find("rest"), note.find("duration")
            if rest is not None and rest.get("measure") == "yes" and (
                    whole is None or duration is None or int(float(duration.text)) != whole):
                del rest.attrib["measure"]
                fixed += 1
    return fixed


def piano_part_of(root: ET.Element, warnings: list[str]) -> ET.Element | None:
    """The score's piano part as tools/bar_dataset.linearize wants it: one part on two staves. A part with one
    staff only, or two parts of one staff, are handled and said."""
    parts = [p for p in root.findall("part") if p.find(".//note") is not None]
    if not parts:
        return None
    if len(parts) == 1:
        staves = {s.text.strip() for s in parts[0].iter("staves")}
        if "2" not in staves:
            warnings.append("one part with one staff: the left hand is not there")
        return only_measures(parts[0])
    if len(parts) > 2:
        warnings.append(f"{len(parts)} parts: the first two are taken as the right and the left hand")
    else:
        warnings.append("two parts: merged as the right and the left hand")
    return merge_parts(only_measures(parts[0]), only_measures(parts[1]), warnings)


def kern_to_xml(path: Path) -> ET.Element:
    """Humdrum **kern (SMT writes **ekern, "bekern") as MusicXML, by kern_to_musicxml.convert_file (music21 with the
    spine paths resolved first, as for the collection's kern scores); plain music21 if that fails."""
    text = re.sub(r"\*\*e?kern(_[\d.]+)?", "**kern", path.read_text(encoding="utf-8", errors="replace"))
    import tempfile

    plain = Path(tempfile.mkdtemp(prefix="kern_")) / (path.stem + ".krn")  # never beside the system's own files
    plain.write_text(text, encoding="utf-8")
    try:
        import kern_to_musicxml

        root, _ = kern_to_musicxml.convert_file(plain, path.stem)
        return root
    except Exception:  # noqa: BLE001
        import warnings as w

        w.simplefilter("ignore")
        from music21 import converter

        target = plain.with_suffix(".musicxml")
        converter.parse(str(plain), format="humdrum").write("musicxml", fp=str(target))
        return ET.parse(target).getroot()


def bars_of_file(path: Path, fmt: str, warnings: list[str]) -> list[str | None]:
    """The bars of one output file as LMX (None: LMX cannot write the bar)."""
    suffix = path.suffix.lower()
    if suffix in (".lmx", ".txt") or (fmt == "lmx" and suffix not in (".mxl", ".musicxml", ".xml", ".krn", ".kern", ".bekern", ".ekern")):
        tokens = path.read_text(encoding="utf-8", errors="replace").split()
        bars, cur = [], []
        for t in tokens:
            if t == "measure" and cur:
                bars.append(" ".join(cur))
                cur = []
            cur.append(t)
        if cur:
            bars.append(" ".join(cur))
        return [b if b.startswith("measure") else "measure " + b for b in bars]
    from bar_dataset import linearize, read

    if suffix in (".krn", ".kern", ".bekern", ".ekern") or fmt == "kern":
        root = kern_to_xml(path)
    else:
        root = read(path)
    FIXES.update(measure_rests_unmarked=0, types_added=0, chords_reordered=0, accidentals_added=0)
    part = piano_part_of(root, warnings)
    if part is None:
        warnings.append(f"{path.name}: no notes")
        return []
    if any(FIXES.values()):
        warnings.append(f"{path.name}: conversion repaired " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in FIXES.items() if v))
    bars = [lmx for lmx, _ in linearize(part)]
    failed = [k for k, lmx in enumerate(bars) if lmx is None]
    if failed:  # a bar LMX cannot write: tried again with its voices one after the other (FAIRNESS.md)
        again = copy.deepcopy(part)
        measures = again.findall("measure")
        for k in failed:
            if k < len(measures):
                revoice(measures[k])
        second = [lmx for lmx, _ in linearize(again)]
        mended = [k for k in failed if k < len(second) and second[k] is not None]
        for k in mended:
            bars[k] = second[k]
        if mended:
            warnings.append(f"{path.name}: {len(mended)} of {len(failed)} bars LMX could not write were written with their voices one after the other")
    return bars


def revoice(measure: ET.Element) -> None:
    """A measure rewritten voice by voice (FAIRNESS.md): homr writes a measure time slice by time slice, stepping back and
    forth between the voices with <backup>s of a tuplet's length, which LMX cannot write ("Duration could not be split up to
    note types for forward/backup"), and the whole bar was then scored as empty. Each note keeps its start, length, pitch,
    staff and voice; only the order of the elements and the backups and forwards between them change. Used only on a bar
    LMX cannot write as it is."""
    notes, others, onset, last = [], [], Fraction(0), Fraction(0)
    for k, e in enumerate(list(measure)):
        d = e.findtext("duration")
        dur = Fraction(int(float(d))) if d else Fraction(0)
        if e.tag == "note":
            chord, grace = e.find("chord") is not None, e.find("grace") is not None
            at = last if chord else onset
            notes.append((at, k, e, dur, chord, grace))
            if not chord and not grace:
                last, onset = onset, onset + dur
        elif e.tag in ("backup", "forward"):
            onset += dur if e.tag == "forward" else -dur
        else:
            others.append(e)
    events: dict[tuple, list] = {}  # (staff, voice): [(onset, order, [notes], length)]
    for at, k, e, dur, chord, grace in notes:
        key = (int(e.findtext("staff") or 1), e.findtext("voice") or "1")
        group = events.setdefault(key, [])
        if chord and group:
            group[-1][2].append(e)
        else:
            group.append((at, k, [e], Fraction(0) if grace else dur))
    for e in list(measure):
        measure.remove(e)
    head = [e for e in others if e.tag != "barline" or e.get("location") == "left"]
    tail = [e for e in others if e not in head]
    for e in head:
        measure.append(e)

    def step(tag: str, length: Fraction, staff: int, voice: str) -> None:
        el = ET.SubElement(measure, tag)
        ET.SubElement(el, "duration").text = str(int(length))
        if tag == "forward":
            ET.SubElement(el, "voice").text = voice
            ET.SubElement(el, "staff").text = str(staff)

    cursor = Fraction(0)
    for (staff, voice), group in sorted(events.items()):
        if cursor:
            step("backup", cursor, staff, voice)
            cursor = Fraction(0)
        for at, _, members, length in sorted(group, key=lambda g: (g[0], g[1])):
            if at > cursor:
                step("forward", at - cursor, staff, voice)
            elif at < cursor:
                step("backup", cursor - at, staff, voice)
            for e in members:
                measure.append(e)
            cursor = at + length
    for e in tail:
        measure.append(e)
