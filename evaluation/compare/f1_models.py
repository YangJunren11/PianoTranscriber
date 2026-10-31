"""Music-level F1 for OMR systems, scored from each system's own MusicXML, over the whole piece.

Scoring a benchmark by SER over LMX tokens makes a system that does not write LMX pay
for the conversion into it. Here every system's output file, as it hands it to a user, is read by music21 the same way as
the gold (evaluation/compare/score_direct.py: measures_of, in_order, multiset) and compared note by note. Nothing goes
through LMX, MuseScore or a normaliser.

What is scored for a system on a piece
  files   the manifest's "note_files" where there are some (Zeus: its score.musicxml; ours: its own .musicxml), else
          "files" (homr, oemer and img2Mxml: their page .musicxml files; Audiveris: its .mxl; SMT: its smt_score.musicxml,
          the kern already turned into MusicXML by the SMT runner). Pages in the manifest's order, their measure lists
          concatenated. A file music21 cannot read, or that is not MusicXML (.lmx, .krn), is noted and left out.
  gold    the manifest's "truth" file, read the same way. Nothing is excluded from it (no bar has to be writable as LMX).
  cut     the manifest's "cut" (experiment 13 scored a system that read only the first pages against the first so many
          measures) is NOT used for the headline figures: the whole piece is the denominator, so the pages a system did not
          read are misses. It is recorded in the row as "cut_ignored", and used only for the "direct_*" continuity figures.
          EXCEPT for a system named in --apply-cut: for it the TRUE measures are truncated to the first `cut` before the
          alignment and the counting (a piece with no cut is scored whole). The row records "cut_applied" (the cut, or None
          for a piece with none) and "cut_ignored" is None. Bars read beyond the cut still count as events found, and being
          aligned to no measure they are unmatched bars, so a system is charged for what it wrote past the cut but not for
          the measures it did not attempt. Whole-piece rows and cut rows are separate rows (use --rename to name them).
  absent  a system with no manifest entry for a piece (never run on it) is left out of that piece and of the pooled
          sums, and listed under "not_run" (printed, and shown as [n of m pieces]). A system that ran and wrote nothing
          has an entry with an empty file list (build_manifest.py writes one, with "status": "failed", for every system
          it finds a run folder and no output for, the pipeline here included) and counts as 0 on that piece; the row
          notes it, and a footnote names those pieces. Only the manifest decides: a system is never dropped from a
          piece because its output is missing.

An event is the tuple score_direct.measures_of makes, one for each note of a chord and each rest, per measure, staves
merged (the parts of the file, or the staves of one part, in order):
    (staff, onset, pitch, duration, grace, step_octave, printed_type)
  staff: the position of the part (the first part is staff 1) or of the staff within a part, so a system that writes
  the bass part first has its notes on the wrong staff. A file of more than two parts numbers the parts that hold notes
  1, 2, 3, ... and keeps them all: the events of a third staff are events found (they cost precision) and can match
  nothing of a two-staff truth under note, pitch and timed F1; onset in quarters from the measure's start (voices
  flattened, snapped to note values); pitch the MIDI number, -1 for a rest (RESTS COUNT unless --no-rests, with their
  durations; hidden <forward> rests never); duration in quarters, 0 for a grace note; grace a bool; step_octave ("C4",
  "rest") and printed_type ("quarter") are kept for the record and used by nothing.

Bars are aligned to measures by Needleman-Wunsch, always (gap cost 0.3 over the cosine similarity of the measures' notes,
score_direct's dynamic programme; a measure's features are its sounding pitches and its musical durations, the facts the
F1s compare, so a C sharp aligns with a D flat and no spelling can change a figure). score_direct.align pairs bars by
position instead when the counts happen to be equal, which scores 0 for a system that dropped the first measure and added
a spurious one; here the same alignment is run for every system and piece, whatever its bar count. Then, over the aligned
pairs only, the matches are counted as score_direct.score counts them:
  note   in_order, per staff: longest common run of the events sorted by (onset, grace first, pitch, duration), compared
         by (pitch, duration, grace); the sort key holds every field compared, so the voices' order in the file cannot
         change the count
  pitch  in_order, the same sort, compared by pitch alone
  onset  multiset intersection, any staff, compared by (onset, pitch)
  timed  multiset intersection, compared by (staff, onset, pitch, duration, grace)
So note, pitch and timed F1 need the right staff; onset F1 does not. tp(criterion) is that count summed over the aligned
pairs. Then, for the WHOLE piece:
  events_true  = events of ALL true measures, events_found = events of ALL bars read
  precision    = tp / events_found      (0.0 if no event was read)
  recall       = tp / events_true       (0.0 if the gold has no event)
  F1           = 2 P R / (P + R) = 2 tp / (events_found + events_true)    (0.0 if P + R = 0, i.e. also when both sides
                 are empty and when nothing matched)
A measure no bar was aligned to ("missed") adds its events to events_true only, so it costs recall; a bar aligned to no
measure ("unmatched") adds its events to events_found only, so it costs precision. Pooled figures over a set of pieces are
micro (sums of tp, events_found, events_true, then the same formulas) and the mean over the pieces' own figures is given
beside them.
Two continuity figures, neither of them a headline:
  matched_only_*  the F1 over the aligned pairs of THIS alignment only (found and true counted over those pairs; 1.0 if
                  they hold no event, and "matched_only_empty" says so): what the score would be if missed measures and
                  unmatched bars cost nothing.
  direct_*        score_direct.score itself, as experiment 13 ran it: the truth cut to the manifest's "cut" where there is
                  one, score_direct's own alignment (positional when the counts are equal). The figure of the research
                  repository's experiments/13_comparison/direct_scores.json, so this is the continuity figure with
                  experiment 13, except where one of score_direct's fixes bites (a third part, an enharmonic spelling
                  that changed the alignment, simultaneous notes of one pitch in two voices; see score_direct). Per
                  piece and as a mean only: an F1 cannot be pooled from counts of a different alignment.
matched_only_* is NOT score_direct.score's figure: it equals direct_* except (a) where experiment 13 cut the truth (the
rows with a "cut_ignored"), and (b) where the bar count equals the measure count and the
content alignment is not the diagonal, so that score_direct paired by position and this tool by content. Rows of kind (b)
carry "realigned": true and are named in the footnotes.
Rests: they are events with durations and count unless --no-rests, as in direct_scores.json. A visible rest shortened
before a note changes the rest (note and timed F1 drop) and moves the note after it (onset and timed F1 drop); with
--no-rests only the move is left. Say so beside any table of these figures (the footnotes do).
Structure: measures (true), measures_matched (aligned to a bar), measures_missed, bars_read, bars_unmatched,
measures_found = measures_matched / measures; measures_exact (extra) = aligned pairs whose events agree exactly under the
timed criterion. Caution: measures_found counts a measure as found when a bar was aligned to it, however the bar looks,
and the alignment pairs off min(bars, measures) of them when the counts differ, so it is a count of bars read against
measures, not of bars read right; measures_exact is the stricter one.

Usage (from the repository root):
  python -I -X utf8 evaluation/compare/f1_models.py run MANIFEST.json --out RESULT.json [--systems A B] [--pieces X Y] [--no-rests]
                                                       [--apply-cut A] [--rename "OLD=NEW"]
  python -I -X utf8 evaluation/compare/f1_models.py merge BASE.json EXTRA.json [EXTRA.json ...] --out MERGED.json
  python -I -X utf8 evaluation/compare/f1_models.py selftest
--apply-cut SYSTEM [SYSTEM ...]  (names among the --systems, matched without regard to case) scores those systems against
the first `cut` true measures where the manifest has a cut. --rename OLD=NEW (repeatable; OLD among the --systems) gives a
row another name in the result, so that the same system can be run twice (whole piece, cut) and merged. merge adds the rows
of the EXTRA files (same manifest, same pieces, same rests setting, no row name already in BASE) to BASE and pools again.
The manifest is the one build_manifest.py writes (pieces: "truth" and the rest; systems -> piece -> files / note_files / cut)
with two more lists, "ten" and "engraved" (piece ids; either may be empty). Systems "ours 26 Oct ..." and the duplicate "ours (via its MusicXML)" are left out unless
named with --systems; names match without regard to case. Pooled sets: "ten", "four" (engraved) and "fourteen"; a set is
left out when the pieces selected for it are those of a set already made (--pieces col0141 gives "ten" only). No GPU.
Prints ASCII only.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # this folder: score_direct

import score_direct as sd  # noqa: E402

XML_SUFFIXES = {".musicxml", ".xml", ".mxl"}
CRITERIA = {  # name -> (counting function, key of an event)
    "note": (sd.in_order, lambda e: (e[2], e[3], e[4])),
    "pitch": (sd.in_order, lambda e: e[2]),
    "onset": (sd.multiset, lambda e: (e[1], e[2])),
    "timed": (sd.multiset, lambda e: (e[0], e[1], e[2], e[3], e[4])),
}
# the integers that add up over pieces
COUNTS = ["measures", "measures_matched", "measures_missed", "measures_exact", "bars_read", "bars_unmatched",
          "events_true", "events_found", "matched_events_found", "matched_events_true"] + [f"tp_{c}" for c in CRITERIA]
MEANED = ([f"{c}_{x}" for c in CRITERIA for x in ("P", "R", "F1")]
          + ["measures_found", "measures_exact_share"] + [f"matched_only_{c}_F1" for c in CRITERIA])
DIRECT = [f"direct_{c}_F1" for c in CRITERIA]  # score_direct.score's own figures; averaged only, never pooled from counts

DEFINITIONS = {
    "event": "(staff, onset, pitch, duration, grace, step_octave, printed_type) as score_direct.measures_of makes it; "
             "every part's events are kept, the last two fields are for the record only",
    "alignment": "Needleman-Wunsch for every system and piece, whatever the bar count (score_direct.align's dynamic "
                 "programme without its equal-count shortcut, which pairs bars by position), over the cosine similarity of "
                 "the measures' sounding pitches and musical durations (not their spelling or printed note types)",
    "rests": "rests are events with durations and count by default (--no-rests leaves them out of both sides): a "
             "shortened visible rest is a changed event (note and timed F1 drop) and moves the notes after it (onset and "
             "timed F1 drop); hidden <forward> never counts",
    "staff": "staff = position of the part (first part is staff 1) or of the staff within a part; note, pitch and timed "
             "F1 need the right staff, onset F1 does not (a system that writes the bass part first scores 0 on the three). "
             "A file of more than two parts numbers the parts with notes 1, 2, 3, ...: the events of a third staff count "
             "as events found and match nothing of a two-staff truth under the three",
    "note": "in_order per staff, events sorted by (onset, grace first, pitch, duration), compared by (pitch, duration, "
            "grace); the order does not depend on the voices' order in the file",
    "pitch": "in_order per staff, the same sort, compared by pitch",
    "status": "the manifest's run status of the system on the piece: ok (it wrote output), failed (it ran and wrote "
              "nothing: an empty file list, scored 0 and kept in the pooled figures)",
    "onset": "multiset, any staff, compared by (onset, pitch)",
    "timed": "multiset, compared by (staff, onset, pitch, duration, grace)",
    "precision": "tp / events_found of ALL bars read (0.0 if none)",
    "recall": "tp / events_true of ALL true measures (0.0 if none)",
    "F1": "2 tp / (events_found + events_true), 0.0 when that is 0",
    "tp": "the criterion's matches summed over the aligned (bar, measure) pairs only",
    "measures_found": "measures_matched / measures; matched = a bar was aligned to the measure (bars read against "
                      "measures, not bars read right)",
    "measures_exact": "aligned pairs whose events agree exactly under the timed criterion (extra)",
    "matched_only": "F1 over the aligned pairs of this alignment only (1.0 if they hold no event): what the score would "
                    "be if missed measures and unmatched bars cost nothing. Not score_direct.score's figure: it differs "
                    "where experiment 13 cut the truth (cut_ignored) and where realigned is true; see direct",
    "realigned": "True when the bar count equals the measure count and the content alignment is not the diagonal: "
                 "score_direct pairs those bars by position, this tool by content, so direct_* and matched_only_* differ",
    "direct": "score_direct.score as experiment 13 ran it (truth cut to the manifest's cut where there is one, score_direct's "
              "own alignment): the figure of experiments/13_comparison/direct_scores.json, except where score_direct's fixes "
              "bite (extra parts kept, alignment by sounding pitch, reading order by duration too); per piece and mean only",
    "pooled": "micro = sums of tp / events_found / events_true over the pieces; mean = mean of the pieces' own figures. "
              "Over the pieces a system has a manifest entry for (see not_run); pieces_in_set is the whole set",
    "cut_ignored": "the page cut experiment 13 used (a system that read only the first pages); NOT applied here",
}

CUT_APPLIED_DEFINITION = ("the manifest's cut applied (--apply-cut): the true measures were truncated to the first `cut` before the "
                          "alignment and the counting, so measures, events_true and the recall are over what the system attempted; "
                          "bars read beyond the cut stay events found and unmatched bars; None for a piece with no cut (scored whole)")


# ---------------------------------------------------------------------------------------------
# Alignment


def align(read: list[list[sd.Event]], truth: list[list[sd.Event]]) -> list[tuple[int | None, int | None]]:
    """score_direct.align without its equal-count shortcut: Needleman-Wunsch (gap cost sd.GAP over the cosine similarity
    of the measures' notes, sd.similarity) for every pair of lists, so that a system whose bar count happens to equal the
    measure count is aligned by content like any other (equal counts used to mean 'bar k is measure k'). The recursion
    and the traceback are score_direct.align's own; the selftest checks they agree wherever that one does not shortcut."""
    sim = sd.similarity(read, truth)
    n, m = len(read), len(truth)
    score = np.zeros((n + 1, m + 1))
    score[:, 0] = -sd.GAP * np.arange(n + 1)
    score[0, :] = -sd.GAP * np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score[i, j] = max(score[i - 1, j - 1] + sim[i - 1, j - 1], score[i - 1, j] - sd.GAP, score[i, j - 1] - sd.GAP)
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if i and j and np.isclose(score[i, j], score[i - 1, j - 1] + sim[i - 1, j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and np.isclose(score[i, j], score[i - 1, j] - sd.GAP):
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]


# ---------------------------------------------------------------------------------------------
# Counting


def count(read: list[list[sd.Event]], truth: list[list[sd.Event]], rests: bool = True) -> dict:
    """The additive counts of one system on one piece (see COUNTS), whole piece in the denominator."""
    if not rests:
        read = [[e for e in b if e[2] >= 0] for b in read]
        truth = [[e for e in b if e[2] >= 0] for b in truth]
    pairs = align(read, truth)
    c = dict.fromkeys(COUNTS, 0)
    c["measures"], c["bars_read"] = len(truth), len(read)
    # not a count (never pooled): would score_direct have paired these bars by position, and does this alignment differ?
    c["realigned"] = len(read) == len(truth) and pairs != [(k, k) for k in range(len(read))]
    c["events_true"], c["events_found"] = sum(map(len, truth)), sum(map(len, read))
    for b, m in pairs:
        if b is None:
            c["measures_missed"] += 1
        elif m is None:
            c["bars_unmatched"] += 1
        else:
            found, true = read[b], truth[m]
            c["measures_matched"] += 1
            c["matched_events_found"] += len(found)
            c["matched_events_true"] += len(true)
            for name, (fn, key) in CRITERIA.items():
                n = fn(found, true, key)
                c[f"tp_{name}"] += n
                if name == "timed":
                    c["measures_exact"] += n == len(found) == len(true)
    return c


def ratio(a: float, b: float, empty: float = 0.0) -> float:
    return a / b if b else empty


def derive(c: dict) -> dict:
    """The counts and the figures made of them: precision, recall and F1 of each criterion over the whole piece (or
    pooled counts), the matched-only F1s, and the structural shares."""
    r = dict(c)
    for name in CRITERIA:
        tp = c[f"tp_{name}"]
        p, q = ratio(tp, c["events_found"]), ratio(tp, c["events_true"])
        r[f"{name}_P"], r[f"{name}_R"] = p, q
        r[f"{name}_F1"] = ratio(2 * p * q, p + q)
        # the F1 over the aligned pairs, 1.0 if there are none (score_direct's convention)
        r[f"matched_only_{name}_F1"] = ratio(2 * tp, c["matched_events_found"] + c["matched_events_true"], 1.0)
    r["matched_only_empty"] = c["matched_events_found"] + c["matched_events_true"] == 0
    r["measures_found"] = ratio(c["measures_matched"], c["measures"])
    r["measures_exact_share"] = ratio(c["measures_exact"], c["measures"])
    return r


def score_events(read: list[list[sd.Event]], truth: list[list[sd.Event]], rests: bool = True, cut: int | None = None,
                 apply_cut: bool = False) -> dict:
    """One system on one piece: the whole-piece figures, and the direct_* figures of score_direct.score on the truth cut
    to `cut` measures (as experiment 13 scored a system that read the first pages only), `cut` None or 0 for no cut.
    With apply_cut the true measures are truncated to the first `cut` before the alignment and the counting, so the
    figures themselves are over the cut (bars read beyond it are unmatched bars); `cut` None or 0 changes nothing."""
    kept = truth[:cut] if cut else truth
    r = derive(count(read, kept if apply_cut else truth, rests))
    d = sd.score(read, kept, rests)
    r["direct_truth_measures"] = len(kept)
    for name in CRITERIA:
        r[f"direct_{name}_F1"] = float(d[f"{name}_F1"])
    return r


def pool(rows: list[dict]) -> dict:
    """Micro pooling (sums of the counts, then the formulas) and the mean of the rows' own figures."""
    keys = [k for k in MEANED + DIRECT if all(k in r for r in rows)]
    return {"n_pieces": len(rows), "micro": derive({k: sum(r[k] for r in rows) for k in COUNTS}),
            "mean": {k: sum(r[k] for r in rows) / len(rows) for k in keys}}


# ---------------------------------------------------------------------------------------------
# Running over a manifest


def system_files(entry: dict) -> list[str]:
    return list(entry.get("note_files") or entry.get("files") or [])


def default_systems(manifest: dict) -> list[str]:
    names = [s for s in manifest["systems"] if not s.startswith("ours 26 Oct")]
    if "ours" in names:  # the same file as "ours (via its MusicXML)"
        names = [s for s in names if s != "ours (via its MusicXML)"]
    return names


def pick(wanted: list[str], available: list[str], what: str) -> list[str]:
    low = {a.lower(): a for a in available}
    out = []
    for w in wanted:
        if w.lower() not in low:
            raise SystemExit(f"unknown {what} {w!r}; available: {available}")
        out.append(low[w.lower()])
    return out


def read_files(files: list[str], notes: list[str]) -> list[list[sd.Event]]:
    bars: list[list[sd.Event]] = []
    for f in files:
        p = Path(f)
        if p.suffix.lower() not in XML_SUFFIXES:
            notes.append(f"{p.name}: not MusicXML, left out")
            continue
        try:
            bars += sd.measures_of(p, notes)
        except Exception as e:  # noqa: BLE001
            notes.append(f"{p.name}: music21 could not read it ({type(e).__name__}: {str(e)[:120]})")
    return bars


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:5.1f}"


def line(name: str, r: dict) -> str:
    return (f"{name:42} note {pct(r['note_F1'])}  pitch {pct(r['pitch_F1'])}  onset {pct(r['onset_F1'])}  "
            f"timed {pct(r['timed_F1'])}  measures found {r['measures_matched']}/{r['measures']} ({pct(r['measures_found'])})  "
            f"exact {r['measures_exact']}  bars {r['bars_read']}  matched-only note {pct(r['matched_only_note_F1'])}")


def score_manifest(manifest: dict, piece_ids: list[str], systems: list[str], rests: bool, log=None, save=None,
                   apply_cut: set[str] | None = None, rename: dict[str, str] | None = None):
    """(rows, not_run): rows[label][piece] the scored row, not_run[label] the pieces the system has no entry for.
    apply_cut: the systems (manifest names) scored against the first `cut` true measures; rename: manifest name -> row label."""
    apply_cut, rename = apply_cut or set(), rename or {}
    truths: dict[str, list] = {}
    rows: dict[str, dict] = {}
    not_run: dict[str, list[str]] = {}
    for system in systems:
        label = "ours" if system == "ours (via its MusicXML)" and "ours" not in manifest["systems"] else system
        label = rename.get(system, label)
        if label in rows:
            raise SystemExit(f"two rows would be called {label!r}")
        rows[label], not_run[label] = {}, []
        for piece in piece_ids:
            entry = manifest["systems"][system].get(piece)
            if entry is None:
                not_run[label].append(piece)  # never run on this piece: left out, not scored 0
                continue
            started = time.time()
            if piece not in truths:
                truths[piece] = sd.measures_of(Path(manifest["pieces"][piece]["truth"]), [])
            notes: list[str] = []
            files = system_files(entry)
            status = entry.get("status") or ("ok" if files else "failed")
            if not files:
                notes.append("no output: the system ran and wrote nothing (scored 0)")
            read = read_files(files, notes)
            applied = system in apply_cut
            r = score_events(read, truths[piece], rests, entry.get("cut"), applied)
            r.update(files=files, status=status, notes=notes, cut_ignored=None if applied else entry.get("cut"))  # no wall-clock time in the row: the file must be reproducible
            if applied:
                r["cut_applied"] = entry.get("cut")
            rows[label][piece] = r
            if log:
                log(label, piece, r, round(time.time() - started, 1))
        if save:
            save(rows, not_run)
    return rows, not_run


def footnotes(rows: dict[str, dict], not_run: dict[str, list[str]], rests: bool) -> list[str]:
    """The sentences a table of these figures needs beside it (ASCII)."""
    out = [
        "Every system's own MusicXML is read by music21 the same way as the gold; the whole piece is the denominator: "
        "events of measures no bar was aligned to count as false negatives, events of bars aligned to no measure as false "
        "positives. Bars are aligned to measures by Needleman-Wunsch for every system, whatever its bar count, on the "
        "measures' sounding pitches and durations (not their spelling), the same facts the F1s compare.",
        ("Rests count as events with durations in every figure, so a shortened visible rest costs note and timed F1 and, "
         "by moving the notes after it, onset and timed F1 (--no-rests: pitched notes only)." if rests else
         "Rests are left out of both sides (--no-rests): pitched notes only."),
        "Note, pitch and timed F1 need the right staff (the first part, or the upper staff of one part, is staff 1); onset "
        "F1 (pitch and position in the bar) does not. Every note a system writes counts, those on a third or later staff "
        "included (they match nothing of a two-staff gold under the three that need the staff). F1 is blind to clefs, keys, "
        "time signatures, voices, beams, "
        "slurs and articulations except where they change a note.",
        "measures found = bars read against measures (a bar is aligned to a measure whatever it holds); measures exact = "
        "bars whose events all agree under the timed criterion, the stricter count.",
    ]
    applied = {s: sorted((p, r["cut_applied"]) for p, r in by.items() if r.get("cut_applied")) for s, by in rows.items()}
    applied = {s: p for s, p in applied.items() if p}
    if applied:
        out.append("Scored against the first measures only (the manifest's cut: the pages the system read or finished): the unread "
                   "measures are not charged, so recall is over what the system attempted; bars read beyond the cut count as events "
                   "found and as unmatched bars. " + "; ".join(
                       f"{s} on {len(p)} piece{'s' if len(p) != 1 else ''} ({', '.join(f'{q} {c}' for q, c in p)})"
                       for s, p in applied.items()) + ". Pieces of such a row with no cut are scored whole. The same system's "
                   "whole-piece row, where there is one, charges the unread pages as misses.")
    cut = {s: sorted(p for p, r in by.items() if r.get("cut_ignored")) for s, by in rows.items()}
    cut = {s: p for s, p in cut.items() if p}
    if cut:
        out.append("Read only part of the piece (a first page or some pages; experiment 13 scored these against the first "
                   "measures only, here the unread measures are misses): "
                   + "; ".join(f"{s} on {len(p)} piece{'s' if len(p) != 1 else ''} ({', '.join(p)})" for s, p in cut.items()) + ".")
    failed = {s: sorted(p for p, r in by.items() if not r.get("files")) for s, by in rows.items()}
    failed = {s: p for s, p in failed.items() if p}
    if failed:
        out.append("Ran and wrote nothing, scored 0 and kept in the pooled figures: "
                   + "; ".join(f"{s} on {len(p)} piece{'s' if len(p) != 1 else ''} ({', '.join(p)})" for s, p in failed.items()) + ".")
    absent = {s: p for s, p in not_run.items() if p}
    if absent:
        out.append("Pooled figures cover the pieces a system has a manifest entry for; left out, not scored 0: "
                   + "; ".join(f"{s} on {', '.join(p)}" for s, p in absent.items())
                   + ". A system that ran and wrote nothing has an entry with an empty file list and counts as 0.")
    realigned = {s: sorted(p for p, r in by.items() if r.get("realigned")) for s, by in rows.items()}
    realigned = {s: p for s, p in realigned.items() if p}
    if realigned:
        out.append("Aligned by content where score_direct (experiment 13) would have paired the bars by position, because "
                   "bar and measure counts are equal: " + "; ".join(f"{s} on {', '.join(p)}" for s, p in realigned.items())
                   + ". On these rows matched_only_* differs from direct_*.")
    out.append("matched_only_* = the same F1 over the aligned pairs only (what the score would be if missed measures and "
               "unmatched bars cost nothing); it is not experiment 13's figure on the rows read only in part (cut) or "
               "realigned (above). direct_* = experiments/13_comparison/direct_scores.json's figure (score_direct.score, "
               "truth cut where experiment 13 cut it): the continuity figure.")
    return out


def build(manifest: dict, piece_ids: list[str], rows: dict[str, dict], rests: bool, source: str,
          not_run: dict[str, list[str]] | None = None) -> dict:
    not_run = {s: list(ps) for s, ps in (not_run or {}).items()}
    sets = {"ten": manifest["ten"], "four": manifest["engraved"], "fourteen": manifest["ten"] + manifest["engraved"]}
    pooled: dict = {}
    made: list[list[str]] = []
    for name, whole in sets.items():
        members = [p for p in whole if p in piece_ids]
        if not members or members in made:  # no piece selected for it, or the pieces of a set already made
            continue
        made.append(members)
        for system, by_piece in rows.items():
            got = [by_piece[p] for p in members if p in by_piece]
            if got:  # pieces_in_set is the whole set, so that [n of m pieces] shows when pieces were not selected or not run
                pooled.setdefault(name, {})[system] = {"pieces": [p for p in members if p in by_piece], "pieces_in_set": list(whole),
                                                       "pieces_not_run": [p for p in members if p in not_run.get(system, [])],
                                                       **pool(got)}
    pieces = {p: {k: manifest["pieces"][p].get(k) for k in ("title", "period", "split", "set")} for p in piece_ids}
    definitions = dict(DEFINITIONS)
    if any("cut_applied" in r for by in rows.values() for r in by.values()):
        definitions["cut_applied"] = CUT_APPLIED_DEFINITION
    return {"manifest": source, "rests_count": rests, "definitions": definitions, "pieces": pieces, "systems": rows,
            "not_run": not_run, "pooled": pooled, "footnotes": footnotes(rows, not_run, rests)}


def merge_results(manifest: dict, base: dict, extras: list[dict]) -> dict:
    """BASE with the rows of the EXTRA results added (and pooled again): same manifest, pieces and rests setting, no row
    name twice. Merging a base with no extra gives the base back, byte for byte once written."""
    rows = {s: dict(by) for s, by in base["systems"].items()}
    not_run = {s: list(p) for s, p in base["not_run"].items()}
    for k, e in enumerate(extras, 1):
        for key in ("manifest", "rests_count"):
            if e[key] != base[key]:
                raise SystemExit(f"extra {k}: {key} differs from the base ({e[key]!r} against {base[key]!r})")
        if list(e["pieces"]) != list(base["pieces"]):
            raise SystemExit(f"extra {k}: other pieces than the base")
        for s, by in e["systems"].items():
            if s in rows:
                raise SystemExit(f"extra {k}: row {s!r} is already in the base")
            rows[s], not_run[s] = dict(by), list(e["not_run"][s])
    return build(manifest, list(base["pieces"]), rows, base["rests_count"], base["manifest"], not_run)


def merge(args) -> int:
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    extras = [json.loads(Path(f).read_text(encoding="utf-8")) for f in args.extras]
    manifest = json.loads(Path(base["manifest"]).read_text(encoding="utf-8"))
    result = merge_results(manifest, base, extras)
    Path(args.out).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"merged {len(extras)} result file(s) into {args.base}: rows {list(result['systems'])}\nwritten {args.out}")
    return 0


def run(args) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    piece_ids = pick(args.pieces, list(manifest["pieces"]), "piece") if args.pieces else list(manifest["pieces"])
    systems = pick(args.systems, list(manifest["systems"]), "system") if args.systems else default_systems(manifest)
    rests = not args.no_rests
    apply_cut = set(pick(args.apply_cut, systems, "system among --systems for --apply-cut")) if args.apply_cut else set()
    rename = {}
    for spec in args.rename or []:
        old, eq, new = spec.partition("=")
        if not eq or not new.strip():
            raise SystemExit(f"--rename wants OLD=NEW, got {spec!r}")
        rename[pick([old], systems, "system among --systems for --rename")[0]] = new.strip()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def log(label, piece, r, seconds):
        print(f"{label[:22]:22} {piece:13} note {pct(r['note_F1'])}  onset {pct(r['onset_F1'])}  timed {pct(r['timed_F1'])}  "
              f"found {r['measures_matched']}/{r['measures']}  bars {r['bars_read']}  events {r['events_found']}/{r['events_true']}  "
              f"{seconds}s  {'cut applied ' + str(r['cut_applied']) + '  ' if r.get('cut_applied') else ''}{'; '.join(r['notes'])[:160]}", flush=True)

    def save(rows, not_run):
        out.write_text(json.dumps(build(manifest, piece_ids, rows, rests, str(args.manifest), not_run), indent=1, default=str),
                       encoding="utf-8")

    rows, not_run = score_manifest(manifest, piece_ids, systems, rests, log, save, apply_cut, rename)
    result = build(manifest, piece_ids, rows, rests, str(args.manifest), not_run)
    out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    for name, by_system in result["pooled"].items():
        print(f"\n== {name}: micro over the pieces scored (whole piece in the denominator; rests {'count' if rests else 'do not count'})")
        for system, p in by_system.items():
            extra = "" if p["n_pieces"] == len(p["pieces_in_set"]) else f"  [{p['n_pieces']} of {len(p['pieces_in_set'])} pieces]"
            print(line(system, p["micro"]) + extra)
        print(f"-- {name}: mean over pieces (direct = experiment 13's matched-only figure, truth cut where it cut)")
        for system, p in by_system.items():
            m = p["mean"]
            print(f"{system:42} note {pct(m['note_F1'])}  pitch {pct(m['pitch_F1'])}  onset {pct(m['onset_F1'])}  "
                  f"timed {pct(m['timed_F1'])}  measures found {pct(m['measures_found'])}  direct note {pct(m.get('direct_note_F1'))}")
    absent = {s: p for s, p in result["not_run"].items() if p}
    if absent:
        print("\nNOT RUN (no manifest entry; left out of the rows and the pooled sums, not scored 0):")
        for s, p in absent.items():
            print(f"  {s}: {', '.join(p)}")
    print("\nNotes for the table")
    for sentence in result["footnotes"]:
        print(" -", sentence)
    print("\nwritten", out)
    return 0


# ---------------------------------------------------------------------------------------------
# Self-test: made-up event lists through the same counting code, and a few tiny MusicXML files through a whole run


WHITE = [(s, o, 12 * (o + 1) + p) for o in range(2, 9) for s, p in zip("CDEFGAB", (0, 2, 4, 5, 7, 9, 11))]  # (step, octave, midi)


def ev(degree: int, onset: float = 0.0, dur: float = 1.0, staff: int = 1, grace: bool = False, kind: str = "quarter"):
    """An event on the white key `degree`, its step_octave spelt as its pitch sounds (for the record: nothing uses it)."""
    s, o, midi = WHITE[degree]
    return (staff, sd.Fraction(onset), midi, sd.Fraction(0 if grace else dur), grace, f"{s}{o}", kind)


def bar(k: int, n: int = 4) -> list:
    """The k-th made-up measure: n quarters on white keys no other measure shares (k up to 11)."""
    return [ev(3 * k + i, onset=i) for i in range(n)]


TYPES = {4: "whole", 2: "half", 1: "quarter", 0.5: "eighth"}  # quarters -> printed note type


def toy_score(parts: list[list[list[tuple]]]) -> str:
    """A 4/4 MusicXML score of several parts: parts[p][m] is measure m of part p, a list of notes
    (step, alter, octave, quarters, voice); a change of voice backs up to the measure's start, so that the voices'
    notes sound together."""
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


def toy_musicxml(measures: list[list[tuple[str, int]]]) -> str:
    """A one-staff 4/4 MusicXML score with a quarter note for each (step, octave) of each measure."""
    return toy_score([[[(s, 0, o, 1, 1) for s, o in notes] for notes in measures]])


def selftest() -> int:
    bad: list[str] = []

    def check(name: str, got, want, tol: float = 1e-9) -> None:
        ok = abs(got - want) <= tol if isinstance(want, float) else got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}" + ("" if ok else f" (expected {want!r})"))
        if not ok:
            bad.append(name)

    gold = [bar(k) for k in range(10)]  # 10 measures of 4 events

    print("identical gold and prediction")
    r = derive(count(gold, gold))
    for c in CRITERIA:
        check(f"{c} F1", r[f"{c}_F1"], 1.0)
        check(f"{c} P + R", r[f"{c}_P"] + r[f"{c}_R"], 2.0)
    check("measures_found", r["measures_found"], 1.0)
    check("measures_exact", r["measures_exact"], 10)
    check("events found, true", (r["events_found"], r["events_true"]), (40, 40))

    print("one measure missed (the fourth of ten)")
    read = gold[:3] + gold[4:]
    r = derive(count(read, gold))
    for c in CRITERIA:
        check(f"{c} precision unchanged", r[f"{c}_P"], 1.0)
        check(f"{c} recall = 36/40", r[f"{c}_R"], 0.9)
        check(f"{c} F1 = 72/76", r[f"{c}_F1"], 72 / 76)
        check(f"{c} matched-only F1 stays 1.0", r[f"matched_only_{c}_F1"], 1.0)
    check("measures_missed, bars_unmatched", (r["measures_missed"], r["bars_unmatched"]), (1, 0))
    check("measures_found", r["measures_found"], 0.9)

    print("one extra bar of 3 events (unlike any measure)")
    extra = [ev(33 + i, onset=i) for i in range(3)]
    read = gold[:5] + [extra] + gold[5:]
    r = derive(count(read, gold))
    for c in CRITERIA:
        check(f"{c} precision = 40/43", r[f"{c}_P"], 40 / 43)
        check(f"{c} recall unchanged", r[f"{c}_R"], 1.0)
        check(f"{c} matched-only F1 stays 1.0", r[f"matched_only_{c}_F1"], 1.0)
    check("bars_unmatched, measures_missed", (r["bars_unmatched"], r["measures_missed"]), (1, 0))
    check("measures_found", r["measures_found"], 1.0)

    print("one pitch changed (second event of the third measure)")
    read = [list(b) for b in gold]
    read[2][1] = ev(30, onset=1)
    r = derive(count(read, gold))
    for c in CRITERIA:
        check(f"{c} F1 = 78/80", r[f"{c}_F1"], 78 / 80)
    check("measures_found, measures_exact", (r["measures_found"], r["measures_exact"]), (1.0, 9))

    print("one duration changed: pitch and onset F1 stay 1, note and timed drop")
    read = [list(b) for b in gold]
    read[4][2] = ev(3 * 4 + 2, onset=2, dur=0.5)
    r = derive(count(read, gold))
    check("pitch F1", r["pitch_F1"], 1.0)
    check("onset F1", r["onset_F1"], 1.0)
    check("note F1 = 78/80", r["note_F1"], 78 / 80)
    check("timed F1 = 78/80", r["timed_F1"], 78 / 80)

    print("one onset changed (last event of a measure moved later): pitch and note F1 stay 1")
    read = [list(b) for b in gold]
    read[6][3] = ev(3 * 6 + 3, onset=3.5)
    r = derive(count(read, gold))
    check("pitch F1", r["pitch_F1"], 1.0)
    check("note F1", r["note_F1"], 1.0)
    check("onset F1 = 78/80", r["onset_F1"], 78 / 80)
    check("timed F1 = 78/80", r["timed_F1"], 78 / 80)

    print("nothing read")
    r = derive(count([], gold))
    check("note precision, recall, F1", (r["note_P"], r["note_R"], r["note_F1"]), (0.0, 0.0, 0.0))
    check("measures_found, measures_missed", (r["measures_found"], r["measures_missed"]), (0.0, 10))
    check("matched_only_empty", r["matched_only_empty"], True)

    print("the same matched-only figures as score_direct.score (where it does not shortcut or cut)")
    changed = [list(b) for b in gold]
    changed[2][1] = ev(30, onset=1)
    changed[5][0] = ev(3 * 5, onset=0, dur=0.5)
    for name, read in {"identical": gold, "missed": gold[:3] + gold[4:], "extra": gold[:5] + [extra] + gold[5:],
                       "changed": changed}.items():
        a, b = derive(count(read, gold)), sd.score(read, gold)
        for c in CRITERIA:
            check(f"{name}: matched-only {c} = score_direct", a[f"matched_only_{c}_F1"], float(b[f"{c}_F1"]))
        check(f"{name}: bars_unmatched, measures_unmatched", (a["bars_unmatched"], a["measures_missed"]),
              (b["bars_unmatched"], b["measures_unmatched"]))

    print("alignment is Needleman-Wunsch for every bar count (the equal-count shortcut is not used)")
    for name, read in {"missed": gold[:3] + gold[4:], "extra": gold[:5] + [extra] + gold[5:], "nothing read": [],
                       "first dropped, two more": gold[1:] + [extra, extra]}.items():
        check(f"{name}: same pairs as score_direct.align", align(read, gold) == sd.align(read, gold), True)
    check("equal counts, identical: the diagonal", align(gold, gold), [(k, k) for k in range(10)])
    check("both empty", align([], []), [])
    shifted = gold[1:] + [extra]  # first measure dropped, one spurious bar added: 10 bars against 10 measures
    check("equal counts, first measure dropped and a bar added: gap, shifted diagonal, gap",
          align(shifted, gold), [(None, 0)] + [(k - 1, k) for k in range(1, 10)] + [(9, None)])
    r = derive(count(shifted, gold))
    for c in CRITERIA:
        check(f"{c} F1 = 72/79 (9 of 10 measures read right, 3 spurious events)", r[f"{c}_F1"], 72 / 79)
    check("measures_missed, bars_unmatched, measures_exact", (r["measures_missed"], r["bars_unmatched"], r["measures_exact"]), (1, 1, 9))

    print("equal counts, content alignment off the diagonal: flagged 'realigned', direct_* (positional) and matched_only_* differ")
    r = score_events(shifted, gold)
    check("realigned", r["realigned"], True)
    check("matched-only note F1 = 1.0 (the 9 aligned pairs agree)", r["matched_only_note_F1"], 1.0)
    check("direct (positional) note and pitch F1 = 18/79 (one shared pitch per pair)", (r["direct_note_F1"], r["direct_pitch_F1"]), (18 / 79, 18 / 79))
    check("direct (positional) onset and timed F1 = 0", (r["direct_onset_F1"], r["direct_timed_F1"]), (0.0, 0.0))
    check("not flagged: identical, equal counts on the diagonal", score_events(gold, gold)["realigned"], False)
    check("not flagged: a measure missed (unequal counts)", score_events(gold[:3] + gold[4:], gold)["realigned"], False)
    check("not flagged: nothing read", score_events([], gold)["realigned"], False)
    check("not flagged: one pitch changed, equal counts, diagonal", score_events([list(b) for b in gold[:2]] + [gold[2][:1] + [ev(30, onset=1)] + gold[2][2:]] + gold[3:], gold)["realigned"], False)
    check("realigned is not a count: absent from the pooled micro", "realigned" in pool([r, r])["micro"], False)

    print("rests count by default and not with --no-rests")
    rest = lambda k, d=1.0: (1, sd.Fraction(k), -1, sd.Fraction(d), False, "rest", "quarter")
    with_rests = [b + [rest(4)] for b in gold]
    r_on, r_off = derive(count(gold, with_rests, True)), derive(count(gold, with_rests, False))
    check("recall with rests = 40/50", r_on["timed_R"], 0.8)
    check("recall without rests", r_off["timed_R"], 1.0)

    print("a shortened visible rest is a changed event: note and timed F1 drop with rests, nothing drops without")
    short = [list(b) for b in with_rests]
    short[3][-1] = rest(4, 0.5)
    r_on, r_off = derive(count(short, with_rests, True)), derive(count(short, with_rests, False))
    check("rests: note F1 = 98/100", r_on["note_F1"], 0.98)
    check("rests: timed F1 = 98/100", r_on["timed_F1"], 0.98)
    check("rests: pitch F1 and onset F1 stay 1", (r_on["pitch_F1"], r_on["onset_F1"]), (1.0, 1.0))
    check("no rests: note, pitch, onset, timed F1 all 1", tuple(r_off[f"{c}_F1"] for c in CRITERIA), (1.0,) * 4)
    check("no rests: events counted", (r_off["events_found"], r_off["events_true"]), (40, 40))

    print("a visible rest shortened BEFORE a note: the rest changes and the note moves earlier (the 'rest then A5' case)")
    rested = [[ev(3 * k, onset=0), ev(3 * k + 1, onset=1), rest(2), ev(3 * k + 2, onset=3)] for k in range(10)]
    moved = [list(b) for b in rested]
    moved[5] = [rested[5][0], rested[5][1], rest(2, 0.5), ev(3 * 5 + 2, onset=2.5)]
    r_on, r_off = derive(count(moved, rested, True)), derive(count(moved, rested, False))
    check("rests: pitch F1 stays 1", r_on["pitch_F1"], 1.0)
    check("rests: note F1 = 78/80 (the rest's duration)", r_on["note_F1"], 78 / 80)
    check("rests: onset F1 = 78/80 (the note moved)", r_on["onset_F1"], 78 / 80)
    check("rests: timed F1 = 76/80 (both)", r_on["timed_F1"], 76 / 80)
    check("no rests: note and pitch F1 stay 1", (r_off["note_F1"], r_off["pitch_F1"]), (1.0, 1.0))
    check("no rests: onset and timed F1 = 58/60 (only the move is left)", (r_off["onset_F1"], r_off["timed_F1"]), (58 / 60, 58 / 60))
    check("no rests: events counted", (r_off["events_found"], r_off["events_true"]), (30, 30))
    check("this case is not realigned", r_on["realigned"], False)

    print("staff identity is positional: swapped staves cost note, pitch and timed F1, not onset F1")
    two = [[ev(3 * k + i, onset=i, staff=1 + i % 2) for i in range(4)] for k in range(10)]
    swapped = [[(3 - e[0],) + e[1:] for e in b] for b in two]
    r = derive(count(swapped, two))
    check("note, pitch, timed F1 = 0", (r["note_F1"], r["pitch_F1"], r["timed_F1"]), (0.0, 0.0, 0.0))
    check("onset F1 = 1", r["onset_F1"], 1.0)
    check("the right staves score 1", derive(count(two, two))["timed_F1"], 1.0)

    print("alignment by sounding pitch: a C sharp read against a truth of C then C sharp, or C then D flat, scores alike")
    c_sharp = lambda spelt: (1, sd.Fraction(0), 61, sd.Fraction(1), False, spelt, "quarter")  # MIDI 61 however it is spelt
    c_natural = (1, sd.Fraction(0), 60, sd.Fraction(1), False, "C4", "quarter")
    for spelt in ("C4", "D4"):
        truth_ = [[c_natural], [c_sharp(spelt)]]
        check(f"truth spelt C, {spelt}: the one bar read aligns with the second measure", align([[c_sharp("C4")]], truth_), [(None, 0), (0, 1)])
        r = derive(count([[c_sharp("C4")]], truth_))
        for c in CRITERIA:
            check(f"truth spelt C, {spelt}: {c} F1 = 2/3", r[f"{c}_F1"], 2 / 3)
    check("the alignment features are pitches and durations, not spellings or printed types",
          sorted(set(sd.features([[c_sharp("C4"), c_natural]]))), [("duration", sd.Fraction(1)), ("pitch", 60), ("pitch", 61)])
    check("a bar's vector: a count per pitch, half a count per duration",
          sd.vector([c_sharp("C4"), c_natural], sd.features([[c_sharp("C4"), c_natural]])).tolist(), [1.0, 1.0, 1.0])
    check("a triplet eighth and an eighth are different durations to the alignment",
          len(sd.features([[ev(0, dur=1 / 3), ev(0, dur=0.5)]])), 3)

    print("a third part is kept: its events count as events found and match nothing of a two-staff truth under note, pitch and timed F1")
    three = [[ev(3 * k, onset=0, staff=1), ev(3 * k + 1, onset=0, staff=2), ev(3 * k + 2, onset=0, staff=3)] for k in range(10)]
    two_staves = [[e for e in b if e[0] <= 2] for b in three]
    r = derive(count(three, two_staves))
    check("events found 30, true 20", (r["events_found"], r["events_true"]), (30, 20))
    for c in CRITERIA:
        check(f"{c} F1 = 40/50", r[f"{c}_F1"], 0.8)
    check("the truth with a third staff the reading lacks: recall 20/30", derive(count(two_staves, three))["note_R"], 2 / 3)
    check("all measures still aligned", (r["measures_matched"], r["measures_exact"]), (10, 0))

    print("the reading order is the voices' order in the file no longer: simultaneous notes of one pitch sort by duration")
    quarter = (1, sd.Fraction(0), 60, sd.Fraction(1), False, "C4", "quarter")
    half = (1, sd.Fraction(0), 60, sd.Fraction(2), False, "C4", "half")
    for name, read in {"same order": [[quarter, half]], "voices swapped": [[half, quarter]]}.items():
        r = derive(count(read, [[quarter, half]]))
        check(f"{name}: all four F1 = 1", tuple(r[f"{c}_F1"] for c in CRITERIA), (1.0,) * 4)
    check("the order key holds every field in_order compares", sd.order(half), (sd.Fraction(0), True, 60, sd.Fraction(2)))
    check("a grace note still comes first, a chord still from the bottom up",
          [e[2] for e in sorted([ev(5, onset=0), ev(3, onset=0, grace=True), ev(1, onset=0)], key=sd.order)], [WHITE[3][2], WHITE[1][2], WHITE[5][2]])

    print("pooling: micro (sums) against the mean of the pieces")
    zero = dict.fromkeys(COUNTS, 0)
    one = {**zero, "measures": 10, "measures_matched": 10, "measures_exact": 10, "bars_read": 10, "events_true": 10,
           "events_found": 10, "matched_events_found": 10, "matched_events_true": 10, **{f"tp_{c}": 10 for c in CRITERIA}}
    two_ = {**zero, "measures": 90, "measures_missed": 90, "events_true": 90}
    p = pool([derive(one), derive(two_)])
    check("micro recall = 10/100", p["micro"]["timed_R"], 0.1)
    check("mean recall = (1 + 0) / 2", p["mean"]["timed_R"], 0.5)
    check("micro measures_found = 10/100", p["micro"]["measures_found"], 0.1)
    check("mean measures_found", p["mean"]["measures_found"], 0.5)
    check("no direct_* in a mean of rows that have none", any(k.startswith("direct_") for k in p["mean"]), False)

    print("the page cut: whole-piece figures ignore it, direct_* apply it as experiment 13 did")
    mixed = [ev(3 * 1 + 0, onset=0)] + [ev(3 * 7 + i, onset=i) for i in range(1, 4)]  # one note of measure 1, three of 7
    read = [gold[0], mixed]
    r = score_events(read, gold, True, cut=2)
    d = sd.score(read, gold[:2])
    for c in CRITERIA:
        check(f"direct {c} = score_direct.score on the first 2 measures", r[f"direct_{c}_F1"], float(d[f"{c}_F1"]))
    check("direct note F1 = 10/16", r["direct_note_F1"], 10 / 16)
    check("direct_truth_measures", r["direct_truth_measures"], 2)
    check("whole-piece note recall = 7/40 (the second bar is paired with measure 7)", r["note_R"], 7 / 40)
    check("whole-piece note precision = 7/8", r["note_P"], 7 / 8)
    check("matched-only note F1 = 14/16 differs from direct on a cut row", r["matched_only_note_F1"], 14 / 16)
    check("measures_found = 2/10", r["measures_found"], 0.2)
    r = score_events(read, gold, True)
    check("without a cut direct = matched-only (same alignment, unequal counts)", r["direct_note_F1"], r["matched_only_note_F1"])
    check("direct_truth_measures without a cut", r["direct_truth_measures"], 10)
    p = pool([score_events(read, gold, True, 2), score_events(gold, gold, True)])
    check("mean direct note F1 over the two pieces", p["mean"]["direct_note_F1"], (10 / 16 + 1.0) / 2)

    print("apply_cut: the true measures truncated to the first `cut` before the alignment and the counting")
    unread = gold[:9]  # the tenth measure not attempted
    whole = score_events(unread, gold, True, cut=9)
    cutr = score_events(unread, gold, True, cut=9, apply_cut=True)
    check("without apply_cut the unread measure is a miss: note recall 36/40, measures_found 0.9", (whole["note_R"], whole["measures_found"]), (0.9, 0.9))
    check("cut 9 drops the missed measure: note recall, F1, measures_found = 1.0", (cutr["note_R"], cutr["note_F1"], cutr["measures_found"]), (1.0, 1.0, 1.0))
    check("cut 9: all four F1 = 1.0", tuple(cutr[f"{c}_F1"] for c in CRITERIA), (1.0,) * 4)
    check("cut 9: measures, measures_missed, measures_exact, events_true", (cutr["measures"], cutr["measures_missed"], cutr["measures_exact"], cutr["events_true"]), (9, 0, 9, 36))
    check("cut 9: direct_* equal the whole-piece row's (score_direct on the cut truth either way)",
          all(cutr[f"direct_{c}_F1"] == whole[f"direct_{c}_F1"] for c in CRITERIA) and cutr["direct_truth_measures"] == 9, True)
    holey = gold[:3] + gold[4:]  # the fourth measure missed
    check("cut = the measure count changes nothing (a row with a missed measure: the whole dict is equal)",
          score_events(holey, gold, True, cut=10, apply_cut=True) == score_events(holey, gold, True, cut=10), True)
    check("apply_cut with no cut (None, 0) changes nothing",
          score_events(holey, gold, True, None, True) == score_events(holey, gold) == score_events(holey, gold, True, 0, True), True)
    r = score_events(holey, gold, True, cut=8, apply_cut=True)
    check("a measure missed INSIDE the cut still costs: measures_missed 1, recall 28/32", (r["measures_missed"], r["note_R"]), (1, 28 / 32))
    check("and the two bars beyond the cut are unmatched: precision 28/36", (r["bars_unmatched"], r["note_P"]), (2, 28 / 36))
    r = score_events(gold, gold, True, cut=9, apply_cut=True)
    check("a bar beyond the cut is left unmatched: bars_unmatched 1, measures_missed 0", (r["bars_unmatched"], r["measures_missed"]), (1, 0))
    check("it is the last bar: the alignment ends (9, None)", align(gold, gold[:9])[-1], (9, None))
    check("events found 40, events true 36", (r["events_found"], r["events_true"]), (40, 36))
    check("precision = 36/40", r["note_P"], 36 / 40)
    check("recall = 1.0", r["note_R"], 1.0)
    check("F1 = 72/76", r["note_F1"], 72 / 76)
    check("measures_found 1.0 (9 of 9), bars_read 10", (r["measures_found"], r["bars_read"]), (1.0, 10))
    check("matched-only F1 stays 1.0", r["matched_only_note_F1"], 1.0)
    check("the same read, cut ignored: precision 1.0, recall 1.0, nothing unmatched", tuple(score_events(gold, gold, True, cut=9)[k] for k in ("note_P", "note_R", "bars_unmatched")), (1.0, 1.0, 0))
    check("pick matches names without regard to case", pick(["a CUT"], ["A", "a cut"], "x"), ["a cut"])
    try:
        pick(["nope"], ["A"], "system")
        check("pick refuses an unknown name", False, True)
    except SystemExit:
        check("pick refuses an unknown name", True, True)

    print("sets and absent systems: build() with made-up rows")
    pieces = {k: {"title": k, "period": "x", "split": "x", "set": "x"} for k in ("a", "b", "e")}
    manifest = {"pieces": pieces, "ten": ["a", "b"], "engraved": ["e"]}
    full, half = score_events(gold, gold), score_events([], gold)
    systems = {"S": {"a": full, "b": full, "e": full}, "T": {"a": full, "b": half}, "U": {"a": full}}
    notrun = {"S": [], "T": ["e"], "U": ["b", "e"]}
    res = build(manifest, ["a", "b"], systems, True, "m.json", notrun)
    check("only the ten pieces selected: pooled sets", list(res["pooled"]), ["ten"])
    check("pieces_in_set stays the whole set", res["pooled"]["ten"]["S"]["pieces_in_set"], ["a", "b"])
    check("a system missing a piece: n_pieces 1 of 2", (res["pooled"]["ten"]["U"]["n_pieces"], len(res["pooled"]["ten"]["U"]["pieces_in_set"])), (1, 2))
    check("a system that wrote nothing counts 0: micro recall 20/40 for T", res["pooled"]["ten"]["T"]["micro"]["note_R"], 0.5)
    check("a system left out is listed as not run", res["not_run"]["U"], ["b", "e"])
    check("pieces_not_run in the pooled block", res["pooled"]["ten"]["U"]["pieces_not_run"], ["b"])
    check("the footnote names what was left out", any("U on b, e" in s for s in res["footnotes"]), True)
    res = build(manifest, ["a", "b", "e"], systems, True, "m.json", notrun)
    check("ten and engraved selected: sets", list(res["pooled"]), ["ten", "four", "fourteen"])
    check("fourteen is 3 pieces for S, 1 of 3 for U, not 'ten'", (res["pooled"]["fourteen"]["S"]["n_pieces"], res["pooled"]["fourteen"]["U"]["n_pieces"]), (3, 1))
    check("U has no piece of the four: not in that block", "U" in res["pooled"]["four"], False)
    res = build(manifest, ["e"], {"S": {"e": full}}, True, "m.json", {"S": []})
    check("only the engraved piece selected: four, not a copy as fourteen", list(res["pooled"]), ["four"])
    res = build(manifest, ["a"], {"S": {"a": full}}, True, "m.json", {"S": []})
    check("one of the ten selected: [1 of 2]", (list(res["pooled"]), res["pooled"]["ten"]["S"]["n_pieces"], len(res["pooled"]["ten"]["S"]["pieces_in_set"])), (["ten"], 1, 2))

    print("a whole run over tiny MusicXML files: real music21 reading, empty file list, no entry, a page cut")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        measures = [[("C", 4), ("D", 4), ("E", 4), ("F", 4)], [("G", 4), ("A", 4), ("B", 4), ("C", 5)],
                    [("D", 5), ("E", 5), ("F", 5), ("G", 5)], [("A", 5), ("B", 5), ("C", 6), ("D", 6)]]
        files = {}
        for name, ms in {"gold": measures, "first2": measures[:2], "dropped": measures[1:] + [[("E", 3)] * 2]}.items():
            files[name] = str(tmp / f"{name}.musicxml")
            Path(files[name]).write_text(toy_musicxml(ms), encoding="utf-8")
        lmx = tmp / "bars.lmx"
        lmx.write_text("measure C4", encoding="utf-8")
        # two staves, and the same with a third part of one note a measure
        bass = [[("C", 0, 3, 1, 1)] for _ in measures]
        files["two"] = str(tmp / "two.musicxml")
        Path(files["two"]).write_text(toy_score([[[(s, 0, o, 1, 1) for s, o in m] for m in measures], bass]), encoding="utf-8")
        files["three"] = str(tmp / "three.musicxml")
        Path(files["three"]).write_text(toy_score([[[(s, 0, o, 1, 1) for s, o in m] for m in measures], bass, [[("E", 0, 5, 1, 1)] for _ in measures]]), encoding="utf-8")
        p = lambda t, truth="gold": {"title": t, "period": "x", "split": "x", "set": "ten", "truth": files[truth]}
        manifest = {"pieces": {"p1": p("p1"), "p2": p("p2"), "p3": p("p3", "two")}, "ten": ["p1", "p2", "p3"], "engraved": [],
                    "systems": {"A": {"p1": {"files": [files["gold"]]}, "p2": {"files": [files["gold"]]}},
                                "B": {"p1": {"files": [], "status": "failed"}, "p2": {"files": [files["gold"]], "status": "ok"}},
                                "C": {"p1": {"files": [files["gold"]]}},
                                "D": {"p1": {"files": [files["first2"]], "cut": 2}, "p2": {"files": [files["first2"]], "cut": 2}},
                                "E": {"p1": {"files": [files["dropped"]]}, "p2": {"files": [str(lmx)], "note_files": [files["gold"]]}},
                                "F": {"p1": {"files": [str(lmx)]}},
                                "G": {"p3": {"files": [files["three"]]}},
                                "H": {"p3": {"files": [files["two"]]}}}}
        rows, not_run = score_manifest(manifest, ["p1", "p2", "p3"], ["A", "B", "C", "D", "E", "F", "G", "H"], True)
        res = build(manifest, ["p1", "p2", "p3"], rows, True, "toy.json", not_run)
        a = rows["A"]["p1"]
        check("A: all four F1 = 1, 4/4 measures, 4 exact", (a["timed_F1"], a["note_F1"], a["measures_found"], a["measures_exact"]), (1.0, 1.0, 1.0, 4))
        check("A: status ok (no status in the manifest: it has files)", a["status"], "ok")
        b = rows["B"]["p1"]
        check("B: empty file list scores 0 and is kept", (b["note_F1"], b["events_found"], b["measures_missed"]), (0.0, 0, 4))
        check("B: status failed, noted in the row", (b["status"], any(n.startswith("no output") for n in b["notes"])), ("failed", True))
        check("B: pooled over 2 pieces, micro recall 16 of 32 events", (res["pooled"]["ten"]["B"]["n_pieces"], res["pooled"]["ten"]["B"]["micro"]["note_R"]), (2, 0.5))
        check("B: the footnote names the piece it wrote nothing on", any(s.startswith("Ran and wrote nothing") and "B on 1 piece (p1)" in s for s in res["footnotes"]), True)
        check("no such footnote when every system wrote something", any("Ran and wrote nothing" in s for s in footnotes({"A": rows["A"]}, {"A": []}, True)), False)
        g, h = rows["G"]["p3"], rows["H"]["p3"]
        check("H: two staves against two staves: all four F1 = 1, 20 events", (tuple(h[f"{c}_F1"] for c in CRITERIA), h["events_found"]), ((1.0,) * 4, 20))
        check("G: a third part of 4 notes is counted: 24 events found, 20 true", (g["events_found"], g["events_true"]), (24, 20))
        check("G: note and timed F1 = 40/44, the third staff's notes matching nothing", (g["note_F1"], g["timed_F1"]), (40 / 44, 40 / 44))
        check("G: measures still all aligned, none exact", (g["measures_found"], g["measures_exact"]), (1.0, 0))
        check("G: the note says what became of the third part", any("3 parts with notes" in n and "match nothing" in n for n in g["notes"]), True)
        # spellings and voices through music21: files that sound alike score alike
        sharp = str(tmp / "sharp.musicxml")
        Path(sharp).write_text(toy_score([[[("C", 0, 4, 1, 1)], [("C", 1, 4, 1, 1)]]]), encoding="utf-8")
        flat = str(tmp / "flat.musicxml")
        Path(flat).write_text(toy_score([[[("C", 0, 4, 1, 1)], [("D", -1, 4, 1, 1)]]]), encoding="utf-8")
        one_sharp = str(tmp / "one_sharp.musicxml")
        Path(one_sharp).write_text(toy_score([[[("C", 1, 4, 1, 1)]]]), encoding="utf-8")
        read_ = sd.measures_of(Path(one_sharp), [])
        r_sharp, r_flat = derive(count(read_, sd.measures_of(Path(sharp), []))), derive(count(read_, sd.measures_of(Path(flat), [])))
        check("a C sharp read: the same figures against a truth spelt C sharp and one spelt D flat",
              [r_sharp[f"{c}_F1"] for c in CRITERIA], [r_flat[f"{c}_F1"] for c in CRITERIA])
        check("and they are 2/3, the bar aligned with the measure that sounds the same", r_flat["note_F1"], 2 / 3)
        v12 = str(tmp / "v12.musicxml")
        Path(v12).write_text(toy_score([[[("C", 0, 4, 1, 1), ("C", 0, 4, 2, 2)]]]), encoding="utf-8")
        v21 = str(tmp / "v21.musicxml")
        Path(v21).write_text(toy_score([[[("C", 0, 4, 2, 1), ("C", 0, 4, 1, 2)]]]), encoding="utf-8")
        e12, e21 = sd.measures_of(Path(v12), []), sd.measures_of(Path(v21), [])
        check("two voices' simultaneous C4s, quarter and half: music21 lists them in the file's order", [e[3] for e in e21[0]], [sd.Fraction(2), sd.Fraction(1)])
        check("the voices renumbered: all four F1 = 1", tuple(derive(count(e21, e12))[f"{c}_F1"] for c in CRITERIA), (1.0,) * 4)
        check("C: no entry for p2 and p3: not scored, not run", (list(rows["C"]), not_run["C"], res["pooled"]["ten"]["C"]["n_pieces"]), (["p1"], ["p2", "p3"], 1))
        d = rows["D"]["p1"]
        check("D: whole piece, first 2 of 4 measures read: recall 0.5, precision 1, found 2/4", (d["note_R"], d["note_P"], d["measures_found"]), (0.5, 1.0, 0.5))
        check("D: cut recorded, applied only to direct_*: direct 1.0 on 2 measures", (d["cut_ignored"], d["direct_note_F1"], d["direct_truth_measures"]), (2, 1.0, 2))
        check("D: matched-only 1.0", d["matched_only_note_F1"], 1.0)
        e = rows["E"]["p1"]
        check("E: 4 bars against 4 measures, first dropped, one spurious: aligned by content, note F1 = 24/30 (12 events right of 14 found, 16 true)", e["note_F1"], 24 / 30)
        check("E: measures_missed 1, bars_unmatched 1", (e["measures_missed"], e["bars_unmatched"]), (1, 1))
        check("E: note_files preferred over files", (rows["E"]["p2"]["note_F1"], rows["E"]["p2"]["files"]), (1.0, [files["gold"]]))
        check("F: an .lmx-only entry scores 0 with a note", (rows["F"]["p1"]["note_F1"], any("not MusicXML" in n for n in rows["F"]["p1"]["notes"])), (0.0, True))
        check("fourteen not made when only the ten ran", "fourteen" in res["pooled"], False)
        check("the footnote on cuts names D", any("D on 2 pieces" in s for s in res["footnotes"]), True)
        check("E p1 is realigned, A p1 and D p1 are not", (e["realigned"], a["realigned"], d["realigned"]), (True, False, False))
        check("E p1: direct (positional) differs from matched-only", e["direct_note_F1"] != e["matched_only_note_F1"], True)
        notes_ = res["footnotes"]
        check("footnote names the realigned row: 'E on p1'", any(s.startswith("Aligned by content where score_direct") and "E on p1" in s for s in notes_), True)
        check("no realigned footnote when nothing is realigned", any("Aligned by content where" in s for s in footnotes({"A": rows["A"]}, {"A": []}, True)), False)
        check("footnote: alignment is Needleman-Wunsch for every system", any("Needleman-Wunsch for every system, whatever its bar count" in s for s in notes_), True)
        check("footnote: rests count, with the durations sentence", any(s.startswith("Rests count as events with durations in every figure") for s in notes_), True)
        check("footnote with --no-rests says pitched notes only", any(s.startswith("Rests are left out of both sides") for s in footnotes(rows, not_run, False)), True)
        check("footnote: staff convention and what F1 cannot see", any("need the right staff" in s and "blind to clefs" in s for s in notes_), True)
        check("footnote: matched_only_* and the continuity figure direct_*", any("matched_only_*" in s and "continuity figure" in s for s in notes_), True)
        check("footnote: measures found against measures exact", any(s.startswith("measures found = bars read against measures") for s in notes_), True)
        check("the definitions carry realigned and the rests/staff sentences", all(k in DEFINITIONS for k in ("realigned", "rests", "staff", "alignment", "matched_only", "direct")), True)
        check("JSON-able", bool(json.dumps(res, default=str)), True)
        # --apply-cut and --rename through score_manifest, and merge
        rows_c, nr_c = score_manifest(manifest, ["p1", "p2", "p3"], ["D"], True, apply_cut={"D"}, rename={"D": "D, scored on the cut"})
        dc = rows_c["D, scored on the cut"]["p1"]
        check("rename: the row has the new label only", list(rows_c), ["D, scored on the cut"])
        check("D cut: truth truncated to 2 measures: recall 1.0, measures 2, found 2/2, exact 2", (dc["note_R"], dc["measures"], dc["measures_found"], dc["measures_exact"]), (1.0, 2, 1.0, 2))
        check("D cut: cut_applied 2, cut_ignored None", (dc["cut_applied"], dc["cut_ignored"]), (2, None))
        check("D cut: direct_* the same as D whole-piece's", all(dc[f"direct_{c}_F1"] == d[f"direct_{c}_F1"] for c in CRITERIA), True)
        check("a row not named in apply_cut has no cut_applied key (so a run without --apply-cut is byte-identical to before)", "cut_applied" in d, False)
        res_c = build(manifest, ["p1", "p2", "p3"], rows_c, True, "toy.json", nr_c)
        check("the cut row's pooled ten: 2 pieces, 4 measures, found 4", (res_c["pooled"]["ten"]["D, scored on the cut"]["n_pieces"], res_c["pooled"]["ten"]["D, scored on the cut"]["micro"]["measures"], res_c["pooled"]["ten"]["D, scored on the cut"]["micro"]["measures_matched"]), (2, 4, 4))
        check("cut_applied is defined in a result that has such a row, not in one that has none", ("cut_applied" in res_c["definitions"], "cut_applied" in res["definitions"]), (True, False))
        check("the footnote names the cut row and its cuts", any("D, scored on the cut on 2 pieces (p1 2, p2 2)" in x for x in res_c["footnotes"]), True)
        check("the cut-ignored footnote is not written for a row whose cut was applied", any("experiment 13 scored these" in x for x in res_c["footnotes"]), False)
        merged = merge_results(manifest, res, [res_c])
        check("merge: every row of the base and of the extra, base first", list(merged["systems"]), list(res["systems"]) + ["D, scored on the cut"])
        check("merge: the extra row is pooled in the sets, with D beside it", ("D" in merged["pooled"]["ten"], "D, scored on the cut" in merged["pooled"]["ten"]), (True, True))
        check("merge: the base rows are unchanged", json.dumps(merged["pooled"]["ten"]["A"], default=str) == json.dumps(res["pooled"]["ten"]["A"], default=str), True)
        check("merge of a base with no extra is the base (as written)", json.dumps(merge_results(manifest, res, []), indent=1, default=str) == json.dumps(res, indent=1, default=str), True)
        try:
            merge_results(manifest, res_c, [res_c])
            check("merge refuses a row name twice", False, True)
        except SystemExit:
            check("merge refuses a row name twice", True, True)

    print("selftest", "FAILED: " + ", ".join(bad) if bad else "passed")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="score a manifest's systems on its pieces")
    r.add_argument("manifest")
    r.add_argument("--out", required=True)
    r.add_argument("--systems", nargs="+")
    r.add_argument("--pieces", nargs="+")
    r.add_argument("--no-rests", action="store_true", help="leave rests out of both sides (default: they count, as in direct_scores.json)")
    r.add_argument("--apply-cut", nargs="+", metavar="SYSTEM",
                   help="score these systems (among --systems, any case) against the first `cut` true measures where the manifest has a cut")
    r.add_argument("--rename", action="append", metavar="OLD=NEW", help="name the row of system OLD (among --systems) NEW; repeatable")
    m = sub.add_parser("merge", help="add the rows of EXTRA result files to a BASE result and pool again")
    m.add_argument("base")
    m.add_argument("extras", nargs="*")
    m.add_argument("--out", required=True)
    sub.add_parser("selftest", help="made-up event lists through the counting code, and tiny MusicXML files through a whole run")
    args = ap.parse_args(argv)
    return {"run": run, "merge": merge}.get(args.command, lambda _: selftest())(args)


if __name__ == "__main__":
    sys.exit(main())
