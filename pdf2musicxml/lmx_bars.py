"""Bars in Linearized MusicXML (LMX, github.com/OMR-Research/lmx), one at a time.

LMX writes a part as one sequence of tokens, each bar starting with `measure`; a bar's own tokens give
the clefs, key and time only where they change, and a note tied over the barline only as `tied:stop`
on its far side. Read bar by bar, each bar starts from a state: the clefs, key and time in force, and
the notes tied over to it from the bar before. This module keeps that state (`State`, and what a bar
leaves for the next, `State.after`), gives a bar back as MusicXML on its own with its state in front
(`standalone`, `measure_of`), and lists its notes (`notes_of`): what a musician reads from it, with
none of the tokens that only say how it is drawn (voices, stems, beams).
"""

from __future__ import annotations

import copy
import io
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from fractions import Fraction

from lmx.linearization.vocabulary import CLEF_TOKENS, KEY_TOKENS, PITCH_TOKENS, STAFF_TOKENS

KEYS, CLEFS, PITCHES, STAVES = set(KEY_TOKENS), set(CLEF_TOKENS), set(PITCH_TOKENS), set(STAFF_TOKENS)


@dataclass
class State:
    """What a bar starts from."""

    key: str | None = None  # "key:fifths:-2"
    time: tuple[str, str] | None = None  # ("beats:4", "beat-type:4")
    clefs: dict[str, str] = field(default_factory=dict)  # "staff:1" -> "clef:G2"
    ties: list[tuple[str, str | None]] = field(default_factory=list)  # (pitch, staff) tied over to the bar

    @classmethod
    def parse(cls, text: str) -> State:
        """A state written as tokens (`tokens`), as the bar dataset's context is."""
        return State().after(text.split(), ties=False).with_ties(tied_over(text.split()))

    def with_ties(self, ties: list[tuple[str, str | None]]) -> State:
        out = copy.deepcopy(self)
        out.ties = list(ties)
        return out

    def tokens(self, ties: bool = True) -> list[str]:
        out = [self.key] if self.key else []
        if self.time:
            out += ["time", *self.time]
        for staff in sorted(self.clefs):
            out += [self.clefs[staff], staff]
        for pitch, staff in self.ties if ties else []:
            out += [pitch, *([staff] if staff else []), "tied:start"]
        return out

    def after(self, bar: list[str], ties: bool = True) -> State:
        """The state the next bar starts from: every clef, key and time this bar sets, and the notes
        it ties over its barline."""
        out = copy.deepcopy(self)
        k = 0
        while k < len(bar):
            t = bar[k]
            if t in KEYS:
                out.key = t
            elif t == "time" and k + 2 < len(bar) and bar[k + 1].startswith("beats:") \
                    and bar[k + 2].startswith("beat-type:"):
                out.time = (bar[k + 1], bar[k + 2])
                k += 2
            elif t in CLEFS and k + 1 < len(bar) and bar[k + 1] in STAVES:
                out.clefs[bar[k + 1]] = t
                k += 1
            k += 1
        out.ties = tied_over(bar) if ties else []
        return out


def tied_over(bar: list[str]) -> list[tuple[str, str | None]]:
    """The notes a bar ties over its barline: those with a tie starting that no later note of the
    same pitch on the same staff stops within the bar. A note's staff is the last staff token, as
    LMX writes a staff only where it changes."""
    staff, notes = None, []
    for t in bar:
        if t == "measure":
            staff = None
        elif t in PITCHES or t == "rest":
            notes.append([t, staff, False, False])
        elif t in STAVES:
            staff = t
            if notes:
                notes[-1][1] = t
        elif t == "tied:start" and notes:
            notes[-1][2] = True
        elif t == "tied:stop" and notes:
            notes[-1][3] = True
    open_ties = []
    for k, (pitch, on, start, _) in enumerate(notes):
        if start and pitch != "rest" and not any(p == pitch and s == on and stop for p, s, _, stop in notes[k + 1:]):
            open_ties.append((pitch, on))
    return open_ties


def true_states(records: list[dict]) -> list[State]:
    """The state each bar of the bar dataset (tools/bar_dataset.py) truly starts from: its context,
    and the notes tied over to it from the bar before, where that bar is in the dataset too."""
    by = {(r["piece"], r["style"], r["bar"]): r for r in records}
    out = []
    for r in records:
        before = by.get((r["piece"], r["style"], r["bar"] - 1))
        out.append(State.parse(r["context"]).with_ties(tied_over(before["lmx"].split()) if before else []))
    return out


def standalone(bar: list[str], state: State) -> list[str]:
    """A bar's tokens with its state in front as the clefs, key and time it starts with, so it can
    be read alone: those it sets itself at its start take their place."""
    k = 1 if bar[:1] == ["measure"] else 0
    s = State(state.key, state.time, dict(state.clefs))
    while k < len(bar):
        t = bar[k]
        if t in KEYS:
            s.key, k = t, k + 1
        elif t == "time" and k + 2 < len(bar) and bar[k + 1].startswith("beats:") and bar[k + 2].startswith("beat-type:"):
            s.time, k = (bar[k + 1], bar[k + 2]), k + 3
        elif t in CLEFS and k + 1 < len(bar) and bar[k + 1] in STAVES:
            s.clefs[bar[k + 1]], k = t, k + 2
        else:
            break
    return ["measure", *s.tokens(ties=False), *bar[k:]]


def sets_state(bar: list[str]) -> bool:
    """Whether a bar's tokens write a key, a time signature or a clef: the bar sets the state itself."""
    return any(t.startswith(("key:", "clef:")) or (t == "time" and k + 2 < len(bar) and bar[k + 1].startswith("beats:")
                                                   and bar[k + 2].startswith("beat-type:"))
               for k, t in enumerate(bar))


def reset_state(state: State, mode: bool | str | None) -> State:
    """The state a bar that opens a system is read from, by the recognizer's system-reset `mode` (the
    checkpoint's variant "system_reset"): True reads it from the tied-over notes alone; "key-clefs"
    keeps the time too, as a time is not printed at a system start. Falsy: the state as it is."""
    if mode == "key-clefs":
        return State(time=state.time, ties=state.ties)
    return State(ties=state.ties) if mode else state


def reset_tokens(bar: list[str], state: State, mode: bool | str | None) -> list[str]:
    """What a recognizer with system-reset `mode` is to write for a bar that opens a system: the bar
    with the signatures it reads in front. True: key, time and clefs; "key-clefs": key and clefs (and
    a time the bar sets itself at its start). Falsy: the bar as it is."""
    if mode == "key-clefs":
        return standalone(bar, State(key=state.key, clefs=dict(state.clefs)))
    return standalone(bar, state) if mode else bar


def measure_of(bar: list[str], state: State) -> tuple[list[ET.Element], str]:
    """A bar as MusicXML, by LMX's own Delinearizer: its <measure> elements (one, for a bar that is
    one), and whatever the Delinearizer objected to."""
    from lmx.linearization.Delinearizer import Delinearizer

    err = io.StringIO()
    d = Delinearizer(errout=err)
    try:
        d.process_text(" ".join(standalone(bar, state)))
    except Exception as e:  # noqa: BLE001: a bar a network made up can be anything
        return [], f"{type(e).__name__}: {e}"
    return d.part_element.findall("measure"), err.getvalue()


@dataclass(frozen=True)
class Note:
    measure: int  # 0, but for a bar written as several
    staff: int
    onset: Fraction  # in quarter notes from the bar's start
    pitch: str  # "F#4", "Bb3", or "rest"
    duration: Fraction  # in quarter notes; 0 for a grace note
    grace: bool


def notes_of(bar: list[str], state: State) -> tuple[list[Note], bool]:
    """A bar's notes and rests, each where it starts and on which staff, with its pitch as sounded
    (the key and accidentals applied) and its length; and whether the bar reads back as one bar of
    MusicXML without the Delinearizer objecting."""
    measures, errors = measure_of(bar, state)
    notes, unreadable = [], False
    for m, measure in enumerate(measures):
        divisions, time, last = 1, Fraction(0), Fraction(0)
        for e in measure:
            if e.tag == "attributes" and e.findtext("divisions"):
                divisions = int(e.findtext("divisions"))
            elif e.tag in ("backup", "forward"):
                d = Fraction(int(e.findtext("duration") or 0), divisions)
                time += d if e.tag == "forward" else -d
            elif e.tag == "note":
                grace = e.find("grace") is not None
                duration = Fraction(0) if grace else Fraction(int(e.findtext("duration") or 0), divisions)
                chord = e.find("chord") is not None
                onset = last if chord else time
                step, octave = e.findtext("pitch/step"), e.findtext("pitch/octave")
                if e.find("rest") is not None:
                    pitch = "rest"
                elif step is None or octave is None:  # tokens that make no note
                    pitch, unreadable = "?", True
                else:
                    alter = int(float(e.findtext("pitch/alter") or 0))
                    pitch = step + {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}.get(alter, "?") + octave
                notes.append(Note(m, int(e.findtext("staff") or 1), onset, pitch, duration, grace))
                if not chord and not grace:
                    last, time = time, time + duration
    return notes, len(measures) == 1 and not errors and not unreadable
