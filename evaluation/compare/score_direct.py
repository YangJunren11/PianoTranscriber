"""A scorer with no LMX in the loop: each system's MusicXML (or .mxl) read as it is by music21 and compared with
the ASAP truth note by note.

compare.py turns every file into LMX with LMX's Linearizer before scoring it, and the Linearizer refuses some
MusicXML that competitors write (FAIRNESS.md). Here both sides are read by music21 instead, and only what LMX
itself does not compare is ignored (stems, beams, slurs, voice numbers, printed note types, spelling).

Per measure, each file becomes a list of events: (staff, onset, pitch, duration, grace), with

- staff: the position of the part (or of the PartStaff music21 makes of a part on two staves): 1 and 2 for a file of
  one or two parts, even when one is empty (a reading of the left hand alone is the lower staff). A file of more
  parts numbers the parts that hold notes 1, 2, 3, ... in order: every event is kept, and those on a third or later
  staff count as events found but can match nothing of a two-staff truth under the figures that need the staff (note,
  pitch and timed F1; onset F1 takes any staff) (`measures_of`);
- onset: quarter notes from the measure's start (voices flattened, chords expanded into one event a note);
- pitch: the MIDI number (step, alter and octave as written in <pitch>), or -1 for a rest (rests count, as in the
  current scorer; hidden rests, which music21 makes of <forward>, do not);
- duration: quarter notes, tuplets included; 0 for a grace note;
- ties: a tied continuation is a note of its own, on both sides (as in the current scorer);
- times written in a grid that cannot hold them are put back on the note values (`snap`): the ASAP file of
  Schubert op.142/3 counts in 1024ths of a quarter, so its triplet eighths are 341 or 342 units long and their
  onsets drift by a unit; music21 keeps those raw lengths (341/1024, not 1/3), and no reading that writes exact
  triplets could match one of them in length (VERIFY_direct_scorer.md).

Measures are aligned as transcribe.align does: one to one when the counts are equal, else Needleman-Wunsch with a gap
cost of 0.3 over the cosine similarity of the measures' notes (`vector`: a count per sounding pitch, the MIDI number,
rests as one more pitch, and half a count per musical duration in quarters). The features are the ones the F1s compare,
so what is scored by sounding pitch is aligned by sounding pitch: a C sharp aligns with a D flat, and a spelling can
change no figure (the research repository's scan_dataset.notes counted written steps, letter and octave without the
alteration, and printed note types instead, so an enharmonic spelling could pair a bar with the wrong measure). The
price: a reading that writes a triplet as plain eighths has other durations than the truth and is aligned by its pitches
alone as far as those notes go; the printed types matched such a bar. Then per aligned pair, pooled over the piece
(micro-averaged):

- note F1: staff by staff, notes in reading order (onset, grace first, pitch upwards, then duration, so that two
  voices' simultaneous notes of one pitch fall in one order whichever voice is written first), longest common run of
  (pitch, duration, grace);
- pitch F1: the same with the pitch alone;
- timed F1: the same staff, onset, pitch, duration and grace (a multiset intersection), as the current scorer's
  "timed note F1";
- onset F1 (extra): pitch and onset only, any staff, any length;
- bar length right: aligned pairs whose last event ends at the same point in both.

Unaligned measures and bars are left out of the F1s, as in transcribe.accuracy.

Reduced from the research repository's experiments/13_comparison/score_direct.py to the functions f1_models.py uses
(`measures_of`, `score`); its command line and the runs over stored outputs are not carried. Three things are changed
from that code, after a review of the scorer: extra parts are kept (they were cut to two); the alignment features are
sounding pitches (accidentals counted, enharmonic spellings merged, rests counted as a pitch) and exact durations in
quarters (they were written steps, letter and octave only, and printed note types); and the reading order breaks ties
by duration (it depended on the voices' order in the file). `score` equalled experiments/13_comparison/direct_scores.json
before these changes, and differs from it wherever one of them changes a pair.
"""

from __future__ import annotations

import argparse
import collections
import functools
import glob
import json
import sys
import time
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np

GAP = 0.3  # as transcribe.GAP

Event = tuple  # (staff, onset, pitch, duration, grace, step_octave, printed type); the last two are kept for the record only


def q(x) -> Fraction:
    """A music21 quarter length (float or Fraction) as an exact Fraction."""
    return x if isinstance(x, Fraction) else Fraction(x).limit_denominator(3840)


# Every note value's length and start, in quarters, has a denominator dividing this: down to 128ths of a quarter, with
# triplets, nonuplets, quintuplets and septuplets (and their products) on top
MUSICAL = 2**7 * 3**2 * 5 * 7
_DENOMINATORS = [d for d in range(1, MUSICAL + 1) if MUSICAL % d == 0]


@functools.lru_cache(maxsize=None)
def snap(x: Fraction) -> Fraction:
    """A time as a note value has it. A time already on the note values' grid (its denominator divides MUSICAL), or with a
    small denominator of its own (a 13-tuplet: 1/13), is kept. Any other comes from a file whose <divisions> cannot hold
    the value (341/1024 for a third of a quarter): it is moved to the musical value with the smallest denominator within
    one unit of its own denominator (1/3), or kept if there is none."""
    d = x.denominator
    if MUSICAL % d == 0 or d <= 128:
        return x
    for k in _DENOMINATORS:
        if k >= d:
            break
        y = Fraction(round(x * k), k)
        if abs(x - y) <= Fraction(1, d):
            return y
    return x


def parse(path: Path):
    warnings.simplefilter("ignore")
    from music21 import converter

    return converter.parse(str(path), forceSource=True)


def staves_of(parts: list) -> list:
    """The parts of a file in the order their events are numbered as staves 1, 2, 3, ...: one or two parts keep their
    places even when one is empty (a piano part's two staves, or two parts: a reading of the left hand alone is the
    lower staff, not the upper one; VERIFY_direct_scorer.md); of more, the ones that hold notes count, in order, and
    every one of them is kept: a third staff's events are events found that match nothing of a two-staff truth under
    note, pitch and timed F1 (onset F1 takes any staff)."""
    if len(parts) <= 2:
        return parts
    return [p for p in parts if p.recurse().notes.first() is not None] or parts


def measures_of(path: Path, notes: list[str]) -> list[list[Event]]:
    """The file's measures as lists of events, staves merged measure by measure."""
    from music21 import stream

    score = parse(path)
    every = list(score.parts) if hasattr(score, "parts") else []
    parts = staves_of(every)
    if len(parts) > 2:
        notes.append(f"{path.name}: {len(parts)} parts with notes; staves 1 and 2 are the first two, the rest are extra "
                     "staves whose events count and match nothing under note, pitch and timed F1")
    if sum(p.recurse().notes.first() is not None for p in parts) == 1:
        notes.append(f"{path.name}: one staff only")
    staves = []
    for s, part in enumerate(parts, 1):
        bars = []
        for m in part.getElementsByClass(stream.Measure):
            ev = []
            for el in m.recurse().notesAndRests:
                if el.isRest and (el.style.hideObjectOnPrint or getattr(el, "hideObjectOnPrint", False)):
                    continue  # a <forward>, not a rest on the page
                onset = snap(q(el.getOffsetInHierarchy(m)))
                grace = el.duration.isGrace
                dur = Fraction(0) if grace else snap(q(el.duration.quarterLength))
                if el.isRest:
                    ev.append((s, onset, -1, dur, grace, "rest", el.duration.type))
                else:
                    for p in el.pitches:
                        ev.append((s, onset, int(round(p.ps)), dur, grace, f"{p.step}{p.octave}", el.duration.type))
            bars.append(ev)
        staves.append(bars)
    n = max((len(b) for b in staves), default=0)
    if staves and len({len(b) for b in staves}) > 1:
        notes.append(f"{path.name}: staves have {[len(b) for b in staves]} measures")
    return [[e for b in staves if k < len(b) for e in b[k]] for k in range(n)]


# ---------------------------------------------------------------------------------------------
# Alignment (transcribe.align, with the measures' similarity computed from the events)


def features(bars: list[list[Event]]) -> dict:
    """The index of every sounding pitch (MIDI number, -1 a rest) and every musical duration (quarters; 0 a grace note)
    found in the bars: the features a bar's `vector` counts. Built from both sides' bars together, so that the vectors
    of a reading and of the truth lie in one space."""
    pitches = sorted({e[2] for b in bars for e in b})
    durations = sorted({e[3] for b in bars for e in b})
    return {**{("pitch", p): k for k, p in enumerate(pitches)},
            **{("duration", d): k + len(pitches) for k, d in enumerate(durations)}}


def vector(bar: list[Event], index: dict) -> np.ndarray:
    """A bar as the alignment sees it: a count for each sounding pitch (a rest is one more pitch), and half a count for
    each musical duration. The same facts the F1s compare (pitch as a MIDI number, duration in quarters), none of the
    spelling or the printed note type, so that a C sharp and a D flat align alike. A bracket drawn or not changes no
    duration, so it changes nothing here. A reading that writes a triplet as plain eighths (drops the tuplet) has other
    durations than the truth and is aligned by its pitches alone as far as those notes go; the printed-type feature of the
    research code matched such a bar. That is the trade-off of aligning by what the F1s compare."""
    v = np.zeros(len(index), np.float32)
    for e in bar:
        v[index["pitch", e[2]]] += 1
        v[index["duration", e[3]]] += 0.5
    return v


def similarity(read: list[list[Event]], truth: list[list[Event]]) -> np.ndarray:
    """The cosine similarity of every bar read with every true measure, by `vector`."""
    index = features(read + truth)
    m = np.stack([vector(b, index) for b in read + truth]) if read or truth else np.zeros((0, 0), np.float32)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    return m[: len(read)] @ m[len(read):].T


def align(read: list[list[Event]], truth: list[list[Event]]) -> list[tuple[int | None, int | None]]:
    if len(read) == len(truth):
        return [(k, k) for k in range(len(read))]
    sim = similarity(read, truth)
    n, m = len(read), len(truth)
    score = np.zeros((n + 1, m + 1))
    score[:, 0] = -GAP * np.arange(n + 1)
    score[0, :] = -GAP * np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score[i, j] = max(score[i - 1, j - 1] + sim[i - 1, j - 1], score[i - 1, j] - GAP, score[i, j - 1] - GAP)
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if i and j and np.isclose(score[i, j], score[i - 1, j - 1] + sim[i - 1, j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and np.isclose(score[i, j], score[i - 1, j] - GAP):
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]


# ---------------------------------------------------------------------------------------------
# Scoring


def common_run(a: list, b: list) -> int:
    try:
        from rapidfuzz.distance import LCSseq

        return LCSseq.similarity(a, b)
    except ImportError:
        row = [0] * (len(b) + 1)
        for x in a:
            prev = 0
            for j, y in enumerate(b, 1):
                prev, row[j] = row[j], prev + 1 if x == y else max(row[j], row[j - 1])
        return row[-1]


def order(e: Event) -> tuple:
    """Reading order within a staff: by onset, a grace note before the note it leads to, a chord from the bottom up, and
    simultaneous notes of one pitch by duration. Every field `in_order` compares is in the key, so the order does not
    depend on how the file lists its voices: two events that sort alike are alike."""
    return (e[1], not e[4], e[2], e[3])


def in_order(found: list[Event], truth: list[Event], key) -> int:
    """How many events two bars have in common, staff by staff, each staff's events in reading `order`: the longest run
    of `key`s they share."""
    run = 0
    for staff in {e[0] for e in found} | {e[0] for e in truth}:
        a = [key(e) for e in sorted((e for e in found if e[0] == staff), key=order)]
        b = [key(e) for e in sorted((e for e in truth if e[0] == staff), key=order)]
        run += common_run(a, b)
    return run


def multiset(found: list[Event], truth: list[Event], key) -> int:
    return sum((collections.Counter(map(key, found)) & collections.Counter(map(key, truth))).values())


def length(bar: list[Event]) -> Fraction:
    return max((e[1] + e[3] for e in bar), default=Fraction(0))


def score(read: list[list[Event]], truth: list[list[Event]], rests: bool = True) -> dict:
    if not rests:
        read = [[e for e in b if e[2] >= 0] for b in read]
        truth = [[e for e in b if e[2] >= 0] for b in truth]
    pairs = align(read, truth)
    t = collections.Counter()
    for b, m in pairs:
        if b is None or m is None:
            continue
        found, true = read[b], truth[m]
        t["bars"] += 1
        t["found"] += len(found)
        t["true"] += len(true)
        t["right"] += in_order(found, true, lambda e: (e[2], e[3], e[4]))
        t["pitch"] += in_order(found, true, lambda e: e[2])
        t["timed"] += multiset(found, true, lambda e: (e[0], e[1], e[2], e[3], e[4]))
        t["onset"] += multiset(found, true, lambda e: (e[1], e[2]))
        t["length_right"] += length(found) == length(true)
    f1 = lambda r: 2 * r / (t["found"] + t["true"]) if t["found"] + t["true"] else 1.0
    return {"bars_read": len(read), "measures": len(truth), "matched": t["bars"],
            "bars_unmatched": sum(m is None for _, m in pairs), "measures_unmatched": sum(b is None for b, _ in pairs),
            "events_true": t["true"], "events_found": t["found"],
            "note_F1": f1(t["right"]), "pitch_F1": f1(t["pitch"]), "timed_F1": f1(t["timed"]), "onset_F1": f1(t["onset"]),
            "bar_length_right": t["length_right"], "bar_length_counted": t["bars"]}
