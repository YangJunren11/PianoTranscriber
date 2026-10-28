"""How right a bar's LMX is: a bar the recognizer read, against the bar's true LMX.

Two ways of looking at it, as tokens and as music:

- **Tokens.** The edit distance between the two token sequences (insertions, deletions and
  substitutions) over the true bar's tokens: the token error rate, the symbol error rate of OMR
  papers. A bar is exact when every token is right.
- **Music.** Both bars are read back as MusicXML by LMX's own Delinearizer, each with the state the
  bar starts from in front (`lmx_bars`), and their notes and rests compared. Voices, stems and beams,
  which only say how the bar is drawn, do not count: a bar with its voices numbered otherwise or its
  stems turned is musically right though its tokens are not.

  A note read wrong is one mistake, though in MusicXML it can move many: a length read wrong moves
  every note after it in its voice, and LMX goes from voice to voice by stepping back a length
  (`backup`), so a voice that comes out too long moves the whole of the next voice too. So notes are
  counted two ways. Read in order (note F1), staff by staff, the notes of each bar in the order a
  musician reads them, matched as their longest common run: same pitch as sounded and same length,
  each note misread counted once. Pitches are counted the same way without lengths (pitch F1). And
  in time (timed note F1): same staff, same start, same pitch and same length, so a note moved counts
  as wrong, as it would be heard. The clefs, key and time the bar leaves for the next bar, and the
  notes it ties over, are compared too.

A prediction that does not read back as one bar of MusicXML is invalid; its notes are whatever
the Delinearizer could make of it.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

from .lmx_bars import Note, State, notes_of


def edit_distance(a: list[str], b: list[str]) -> int:
    try:
        from rapidfuzz.distance import Levenshtein

        return Levenshtein.distance(a, b)
    except ImportError:  # the same, slowly
        row = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            prev, row[0] = row[0], i
            for j, y in enumerate(b, 1):
                prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
        return row[-1]


def common_run(a: list, b: list) -> int:
    """The length of the longest common subsequence of two sequences."""
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


def in_order(found: list[Note], truth: list[Note], key) -> int:
    """How many notes two bars have in common, staff by staff, each staff's notes in reading order
    (by start, a grace note before the note it leads to, a chord from the bottom up): the longest
    run of `key`s they share."""
    order = lambda n: (n.measure, n.onset, not n.grace, n.pitch)
    run = 0
    for staff in {n.staff for n in found} | {n.staff for n in truth}:
        a = [key(n) for n in sorted((n for n in found if n.staff == staff), key=order)]
        b = [key(n) for n in sorted((n for n in truth if n.staff == staff), key=order)]
        run += common_run(a, b)
    return run


def paired(found: list[Note], truth: list[Note], key=lambda n: (n.pitch, n.duration, n.grace)) -> tuple[set[int], set[int]]:
    """Which notes of each bar `in_order` pairs up (indices into each list): the rest are the
    mistakes, notes read that are not there and notes there that were not read."""
    order = lambda n: (n.measure, n.onset, not n.grace, n.pitch)
    hit_found, hit_truth = set(), set()
    for staff in {n.staff for n in found} | {n.staff for n in truth}:
        a = sorted((k for k, n in enumerate(found) if n.staff == staff), key=lambda k: order(found[k]))
        b = sorted((k for k, n in enumerate(truth) if n.staff == staff), key=lambda k: order(truth[k]))
        # The longest common subsequence, and one way of pairing it up
        table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a) - 1, -1, -1):
            for j in range(len(b) - 1, -1, -1):
                table[i][j] = table[i + 1][j + 1] + 1 if key(found[a[i]]) == key(truth[b[j]]) \
                    else max(table[i + 1][j], table[i][j + 1])
        i = j = 0
        while i < len(a) and j < len(b):
            if key(found[a[i]]) == key(truth[b[j]]):
                hit_found.add(a[i]), hit_truth.add(b[j])
                i, j = i + 1, j + 1
            elif table[i + 1][j] >= table[i][j + 1]:
                i += 1
            else:
                j += 1
    return hit_found, hit_truth


@dataclass
class BarScore:
    tokens: int  # in the true bar
    edits: int  # to make the prediction the true bar
    exact: bool
    valid: bool  # the prediction reads back as one bar of MusicXML
    notes: int  # in the true bar, notes and rests
    found: int  # in the prediction
    right: int  # read in order: the same pitch and length
    pitches_right: int  # read in order: the same pitch
    timed_right: int  # the same staff, start, pitch and length
    state_right: bool  # the same clefs, key and time left for the next bar, and the same ties over


def score_bar(predicted: str, truth: str, state: State | str, read_from: State | None = None) -> BarScore:
    """A bar read, against the true bar, both starting from `state` (a `State`, or its tokens as
    the bar dataset's context holds them). A bar read from another state, as when the bars before it
    were read wrong (`read_from`), sounds as that state makes it: a key misread three bars back
    turns this bar's notes too, though its tokens are right."""
    state = State.parse(state) if isinstance(state, str) else state
    read_from = state if read_from is None else read_from
    p, t = predicted.split(), truth.split()
    if p == t and read_from == state:  # read back once: the same bar both times
        true_notes, valid = notes_of(t, state)
        n = len(true_notes)
        return BarScore(len(t), 0, True, valid, n, n, n, n, n, True)
    found, valid = notes_of(p, read_from)
    true_notes, _ = notes_of(t, state)
    timed = lambda n: (n.measure, n.staff, n.onset, n.pitch, n.duration, n.grace)
    timed_right = sum((collections.Counter(map(timed, found)) & collections.Counter(map(timed, true_notes))).values())
    return BarScore(len(t), edit_distance(p, t), p == t, valid, len(true_notes), len(found),
                    in_order(found, true_notes, lambda n: (n.pitch, n.duration, n.grace)),
                    in_order(found, true_notes, lambda n: n.pitch), timed_right,
                    read_from.after(p) == state.after(t))


@dataclass
class Tally:
    """Bars' scores added up: micro-averaged, so a long bar counts for its tokens and notes."""

    bars: int = 0
    tokens: int = 0
    edits: int = 0
    exact: int = 0
    valid: int = 0
    notes: int = 0
    found: int = 0
    right: int = 0
    pitches_right: int = 0
    timed_right: int = 0
    state_right: int = 0
    by: dict = field(default_factory=dict)  # a Tally for each group, when added with one

    def add(self, s: BarScore, group: str | None = None) -> None:
        for t in (self, self.by.setdefault(group, Tally())) if group is not None else (self,):
            t.bars += 1
            t.tokens += s.tokens
            t.edits += s.edits
            t.exact += s.exact
            t.valid += s.valid
            t.notes += s.notes
            t.found += s.found
            t.right += s.right
            t.pitches_right += s.pitches_right
            t.timed_right += s.timed_right
            t.state_right += s.state_right

    @staticmethod
    def f1(right: int, found: int, true: int) -> float:
        return 2 * right / (found + true) if found + true else 1.0

    def summary(self) -> dict[str, float]:
        n = max(self.bars, 1)
        return {"bars": self.bars,
                "token error rate": self.edits / max(self.tokens, 1),
                "bars exact": self.exact / n,
                "note F1": self.f1(self.right, self.found, self.notes),
                "note precision": self.right / max(self.found, 1),
                "note recall": self.right / max(self.notes, 1),
                "pitch F1": self.f1(self.pitches_right, self.found, self.notes),
                "timed note F1": self.f1(self.timed_right, self.found, self.notes),
                "state right": self.state_right / n,
                "valid": self.valid / n}


COLUMNS = [("token error rate", "token err", "{:.2%}"), ("bars exact", "exact", "{:.1%}"), ("note F1", "note F1", "{:.2%}"),
           ("pitch F1", "pitch F1", "{:.2%}"), ("timed note F1", "timed F1", "{:.2%}"), ("state right", "state", "{:.1%}"),
           ("valid", "valid", "{:.1%}")]


def table(tallies: dict[str, Tally], groups: bool = True) -> str:
    """Models side by side: each model's line, and under it a line for each group of bars."""
    head = f"{'':34} {'bars':>7} " + " ".join(f"{label:>10}" for _, label, _ in COLUMNS)
    lines = [head]

    def line(name: str, t: Tally) -> str:
        s = t.summary()
        return f"{name:34} {s['bars']:>7} " + " ".join(f"{fmt.format(s[k]):>10}" for k, _, fmt in COLUMNS)

    for name, t in tallies.items():
        lines.append(line(name, t))
        if groups and len(t.by) > 1:
            for g in sorted(t.by):
                lines.append(line("  " + g, t.by[g]))
    return "\n".join(lines)
