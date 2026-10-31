"""System-level symbol error rate (SER) of any OMR system's MusicXML, as the OLiMPiC paper scores it: the bars of one
printed system joined, one Levenshtein distance over LMX tokens for the system, the edits summed over the systems and
divided by the gold tokens.

(evaluation/system_ser.py does the same for the bar recognizer's own dump on a bar dataset. This tool is for whole-piece
output of other systems: a MusicXML file or a few, a score on paper, no bar boxes to go by.)

  python -I -X utf8 evaluation/compare/system_ser_models.py run MANIFEST.json --out RESULT.json [--systems A B] [--pieces X Y] [--dump-systems DIR]
  python -I -X utf8 evaluation/compare/system_ser_models.py selftest

What it does, for a piece and a system:

1. gold: the piece's true MusicXML as LMX measure by measure (`align.load_truth`); a measure LMX cannot
   write (None) has no gold and is left out for every system alike (counted in the result).
2. prediction: the system's MusicXML files (pages in order) as LMX measure by measure with our converter
   (`evaluation/compare/omr_convert.bars_of_file`: evaluation/bar_dataset.linearize, after the repairs that make other
   systems' valid MusicXML writable: chords put bottom up, accidentals made explicit, voices stepped one after the other
   where a backup of a tuplet's length defeats the Linearizer; experiments/13_comparison/FAIRNESS.md). A measure the
   converter still refuses is an empty measure for the score (`measure` alone), and is counted: `refused`. The row
   `ser_without_refused` leaves such a measure out of both sides, to show how much of a system's score is the converter's.
3. alignment: bars to true measures with `align.align` (Needleman-Wunsch over the notes each holds, gap 0.3, bars
   may be inserted and measures missed), the alignment of experiments/13_comparison.
4. systems: the partition of the true measures into printed systems comes from the manifest ("partition": the JSON that
   `pdf2musicxml.transcribe` wrote for the piece, its per_bar records with page and system, aligned to the gold like any
   system's bars). A true measure our own cut missed takes the system of the measure before it. Every system under test
   is scored on that one partition: the predicted bars aligned to a system's measures are that system's prediction; a
   bar aligned to no measure joins the system of the measure before it (an insertion); a measure with no bar is a
   deletion.
5. joined sequences: the gold of a system is its measures' tokens one after the other, its first one with the key and
   clefs in front (`lmx_bars.reset_tokens`, "key-clefs": a system's start shows them, the time only where it changes);
   the prediction the same: its first bar with the key and clefs of the state its own earlier bars left, in front.
   Every system under test is treated alike, so a system that reads a key signature once and not again at each line start
   is not punished for it.
6. SER (`ser`): Levenshtein over tokens (rapidfuzz) of the joined prediction against the joined gold, per system; edits summed
   over the systems of the piece (or the set) over the gold tokens. A system with no output for a piece scores every
   gold token as an edit. A second figure, `ser_nostem`, leaves the stem and beam tokens (how a note is drawn, not what it is)
out of both sides: some systems write no stems at all (experiments/13_comparison/TEDN.md).

Also per piece: the bar-level figures for continuity with experiments/13_comparison (`bar_ser_matched`: edits over
tokens of the aligned pairs only, the true bars as they are, the 'token error rate' of its result.json; `bar_ser_all`:
the same with the bars read that match no measure counted as insertions and the measures missed as deletions) and note
F1 from the direct scorer (`score_direct`, MusicXML read by music21, no LMX).

Manifest: {"pieces": {ID: {"title", "period", "truth": PATH, "partition": PATH.json}},
           "systems": {NAME: {ID: {"files": [PATHS in page order], "native_json": PATH (optional; the bars are the
                                    JSON's per_bar lmx, no MusicXML in between), "cut": N (optional: truth measures),
                                    "note_files": [PATHS] (optional: MusicXML for note F1 where "files" are LMX)}}}
"files" may also be LMX files (.lmx: split at `measure`, used as they are: Zeus writes LMX).
A system with no entry for a piece was not run on it (left out); an entry with no files was run and wrote nothing (every gold token is an edit).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # the repository: pdf2musicxml, third_party/lmx
sys.path.insert(0, str(HERE.parent))  # evaluation/: bar_dataset
sys.path.insert(0, str(HERE))  # this folder: omr_convert, score_direct

KEY_CLEFS = "key-clefs"
DRAWING = ("stem:", "beam:")  # tokens that only say how a note is drawn; the second SER (ser_nostem) leaves them out of both sides


def edits(a: list[str], b: list[str]) -> int:
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.distance(a, b)


# ---------------------------------------------------------------------------------------------
# Reading


def read_gold(truth_path: str) -> list[tuple[str | None, str]]:
    from align import load_truth

    return load_truth(truth_path)


def read_bars(files: list[str], notes: list[str]) -> list[list[str] | None]:
    """The files' bars as tokens, None for a bar the converter refuses. Unreadable files are noted and skipped."""
    import omr_convert as compare

    out: list[list[str] | None] = []
    for f in files:
        try:
            for lmx in compare.bars_of_file(Path(f), "musicxml", notes):
                out.append(lmx.split() if lmx is not None else None)
        except Exception as e:  # noqa: BLE001 - a competitor's file can be anything
            notes.append(f"{Path(f).name}: unreadable ({type(e).__name__}: {str(e)[:150]})")
    return out


PITCH_TOKEN = re.compile(r"^[A-G][0-9]$")


def bars_with_counts(path: str, notes: list[str]) -> list[tuple[list[str] | None, int]]:
    """One file's bars as (tokens or None if refused, pitched notes of the bar). Counts: <note> elements with <pitch> of the
    measure of the piano part the converter reads (grace notes included); for a LMX file, its pitch tokens."""
    import omr_convert as compare
    from bar_dataset import read

    lmx = compare.bars_of_file(Path(path), "musicxml", notes)
    if Path(path).suffix.lower() in (".lmx", ".txt"):
        return [(b.split(), sum(bool(PITCH_TOKEN.match(t)) for t in b.split())) for b in lmx]
    part = compare.piano_part_of(read(Path(path)), [])
    counts = [sum(n.find("pitch") is not None for n in m.iter("note")) for m in part.findall("measure")] if part is not None else []
    if len(counts) != len(lmx):
        raise ValueError(f"{len(counts)} measures counted, {len(lmx)} bars written")
    return [(b.split() if b is not None else None, c) for b, c in zip(lmx, counts)]


def merge_bars(norm: list, raw: list) -> tuple[list, dict]:
    """Per bar: the normalised bar if the converter accepts it and its pitched-note count equals the raw bar's; else the raw bar
    if the converter accepts it; else None (empty). Both lists are (tokens or None, count) of equal length."""
    st = dict(normalised=0, fallback_notes=0, fallback_refused=0, empty=0)
    out = []
    for (nb, nc), (rb, rc) in zip(norm, raw):
        if nb is not None and nc == rc:
            out.append(nb)
            st["normalised"] += 1
        elif rb is not None:
            out.append(rb)
            st["fallback_notes" if nb is not None else "fallback_refused"] += 1
        else:
            out.append(None)
            st["empty"] += 1
    return out, st


def read_bars_fallback(files: list[str], fallbacks: list[str], notes: list[str]) -> tuple[list, dict]:
    """The normalised files' bars with the raw files' as fallback (merge_bars), file by file. A file whose raw twin is
    itself (MuseScore failed on it), or whose bar count differs from its raw twin's, is read raw throughout."""
    total = dict(normalised=0, fallback_notes=0, fallback_refused=0, empty=0, raw_files_failed=0, raw_files_count_differs=0, raw_whole_bars=0, raw_whole_refused=0, files=len(files))
    out: list = []
    for f, fb in zip(files, fallbacks):
        try:
            raw = bars_with_counts(fb, notes)
        except Exception as e:  # noqa: BLE001
            notes.append(f"{Path(fb).name}: unreadable ({type(e).__name__}: {str(e)[:150]})")
            continue
        rawonly = [b for b, _ in raw]
        if f == fb:
            total["raw_files_failed"] += 1
            bars = rawonly
        else:
            try:
                norm = bars_with_counts(f, notes)
            except Exception as e:  # noqa: BLE001
                notes.append(f"{Path(f).name}: normalised file unreadable ({type(e).__name__}: {str(e)[:100]}); raw used")
                norm = []
            if len(norm) != len(raw):
                total["raw_files_count_differs"] += 1
                notes.append(f"{Path(f).name}: {len(norm)} bars against {len(raw)} raw; raw file used whole")
                bars = rawonly
            else:
                bars, st = merge_bars(norm, raw)
                for k, v in st.items():
                    total[k] += v
                out += bars
                continue
        total["raw_whole_bars"] += sum(b is not None for b in bars)
        total["raw_whole_refused"] += sum(b is None for b in bars)
        out += bars
    return out, total


def merge_gold(norm: list, raw: list) -> tuple[list, dict]:
    """Gold: the normalised measure if LMX can write it, else the raw one if it can, else excluded (None)."""
    st = dict(normalised=0, raw=0, excluded=0)
    out = []
    for n, r in zip(norm, raw):
        if n[0] is not None:
            out.append(n)
            st["normalised"] += 1
        elif r[0] is not None:
            out.append(r)
            st["raw"] += 1
        else:
            out.append(n)
            st["excluded"] += 1
    return out, st


def read_gold_fallback(truth: str, fallback: str) -> tuple[list, dict]:
    norm, raw = read_gold(truth), read_gold(fallback)
    if len(norm) != len(raw):
        st = dict(normalised=0, raw=len(raw), excluded=sum(m is None for m, _ in raw), measure_count_differs=True)
        st["raw"] -= st["excluded"]
        return raw, st
    return merge_gold(norm, raw)


NOTE_TYPES = frozenset(["breve", "long", "whole", "half", "quarter", "eighth", "16th", "32nd", "64th", "128th"])


def strip_hidden_rests(tokens: list[str]) -> tuple[list[str], int, int]:
    """The token-level step applied to both sides (a system's joined tokens) before the Levenshtein distance. MuseScore pads a voice it
    considers incomplete with invisible rests (`print-object:no rest ... whole`, each after a `backup <type>`); they are not music that was
    read or missed, and the gold is padded much less than a prediction. Removes (1) every hidden-rest group: `print-object:no` directly
    followed by `rest`, then the tokens after it while they are a note type, `dot`, `rest:measure`, `voice:*` or `staff:*`;
    (2) the two tokens before the group when they are `backup <type>` (the backup only served the hidden voice);
    (3) afterwards every `backup <type>` / `forward <type>` pair followed by `measure`, another `backup`/`forward`, or the end.
    Returns (tokens, groups removed, pairs removed in step 3)."""
    out: list[str] = []
    groups = 0
    i, n = 0, len(tokens)
    while i < n:
        if tokens[i] == "print-object:no" and i + 1 < n and tokens[i + 1] == "rest":
            j = i + 2
            while j < n and (tokens[j] in NOTE_TYPES or tokens[j] in ("dot", "rest:measure") or tokens[j].startswith(("voice:", "staff:"))):
                j += 1
            if len(out) >= 2 and out[-2] == "backup" and out[-1] in NOTE_TYPES:
                del out[-2:]
            groups += 1
            i = j
        else:
            out.append(tokens[i])
            i += 1
    pairs = 0
    changed = True
    while changed:
        changed = False
        k = 0
        res: list[str] = []
        while k < len(out):
            if out[k] in ("backup", "forward") and k + 1 < len(out) and out[k + 1] in NOTE_TYPES and (
                    k + 2 >= len(out) or out[k + 2] in ("measure", "backup", "forward")):
                k += 2
                pairs += 1
                changed = True
            else:
                res.append(out[k])
                k += 1
        out = res
    return out, groups, pairs


def native_bars(json_path: str) -> list[list[str]]:
    d = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return [r["lmx"].split() for r in d["per_bar"]]


def partition_of(partition_json: str, gold: list[tuple[str | None, str]]) -> tuple[list[tuple[int, int]], dict]:
    """The printed system of every true measure: (page, system) of the bar of our own cut aligned to it; a measure with
    no bar takes the system of the measure before it (the first ones, of the first measure's bar after them)."""
    from align import align

    d = json.loads(Path(partition_json).read_text(encoding="utf-8"))
    recs = d["per_bar"]
    bars = [r["lmx"].split() for r in recs]
    truth = [m for m, _ in gold]
    mine: list[tuple[int, int] | None] = [None] * len(truth)
    for b, m in align(bars, truth):
        if b is not None and m is not None:
            mine[m] = (recs[b]["page"], recs[b]["system"])
    missed = sum(x is None for x in mine)
    last = None
    for m, x in enumerate(mine):
        if x is None:
            mine[m] = last
        else:
            last = x
    nxt = None
    for m in range(len(mine) - 1, -1, -1):  # the first measures, if our cut missed them
        if mine[m] is None:
            mine[m] = nxt
        else:
            nxt = mine[m]
    return mine, {"measures": len(truth), "measures_with_no_bar_of_ours": missed,
                  "systems": len({x for x in mine if x is not None}), "bars_of_ours": len(bars)}


# ---------------------------------------------------------------------------------------------
# Scoring


def chain(bars: list[list[str]]):
    from pdf2musicxml.lmx_bars import State

    states, state = [], State()
    for bar in bars:
        states.append(state)
        state = state.after(bar)
    return states


def score_piece(read: list[list[str] | None], gold: list[tuple[str | None, str]], systems_of: list, drop_refused: bool = False,
                strip_hidden: bool = False) -> dict:
    """Everything the tool reports for one system on one piece. `read`: the bars (None: refused by the converter);
    `systems_of[m]`: the printed system of true measure m."""
    from pdf2musicxml.lmx_bars import State, reset_tokens, tied_over
    from align import align

    truth = [m for m, _ in gold]
    refused = [b is None for b in read]
    bars = [b if b is not None else ["measure"] for b in read]
    pairs = align(bars, truth) if (bars or truth) else []
    states = chain(bars)

    def true_tokens(m: int, first: bool) -> list[str]:
        t = truth[m].split()
        if not first:
            return t
        state = State.parse(gold[m][1]).with_ties(tied_over(truth[m - 1].split()) if m and truth[m - 1] else [])
        return reset_tokens(t, state, KEY_CLEFS)

    def read_tokens(b: int, first: bool) -> list[str]:
        return reset_tokens(bars[b], states[b], KEY_CLEFS) if first else bars[b]

    # which system each pair belongs to, in alignment order
    keys = [x for x in systems_of if x is not None]
    here = keys[0] if keys else (0, 0)
    seq = []  # (system, bar or None, measure or None)
    for b, m in pairs:
        if m is not None:
            here = systems_of[m] if systems_of[m] is not None else here
        seq.append((here, b, m))
    drop_bar = set()
    if drop_refused:  # leave a refused bar and the measure it was aligned to out of both sides
        drop_bar = {b for _, b, m in seq if b is not None and refused[b]}
    per: dict = {}
    order: list = []
    matched_e = matched_t = 0
    all_e = 0
    gold_total = 0
    unwritable_aligned = inserted_tokens = deleted_tokens = 0
    for key, b, m in seq:
        if key not in per:
            per[key] = {"pred": [], "gold": [], "first_gold": True, "first_pred": True}
            order.append(key)
        s = per[key]
        if b is not None and b in drop_bar:
            continue
        if m is not None and truth[m] is None:
            unwritable_aligned += b is not None
            continue
        if m is not None:
            t = true_tokens(m, s["first_gold"])
            s["first_gold"] = False
            s["gold"].extend(t)
            gold_total += len(truth[m].split())
        if b is not None:
            p = read_tokens(b, s["first_pred"])
            s["first_pred"] = False
            s["pred"].extend(p)
        # bar-level figures, on the bars as they are
        if b is not None and m is not None:
            e = edits(bars[b], truth[m].split())
            matched_e += e
            matched_t += len(truth[m].split())
            all_e += e
        elif b is not None:
            all_e += len(bars[b])
            inserted_tokens += len(bars[b])
        else:
            all_e += len(truth[m].split())
            deleted_tokens += len(truth[m].split())
    rows, total_e, total_g, ns_e, ns_g = [], 0, 0, 0, 0
    hidden = {"hidden_groups_pred": 0, "hidden_groups_gold": 0, "backup_pairs_pred": 0, "backup_pairs_gold": 0}
    for key in order:
        s = per[key]
        if strip_hidden:  # both sides alike (strip_hidden_rests); the bar-level figures above stay on the bars as they are
            s["pred"], g1, p1 = strip_hidden_rests(s["pred"])
            s["gold"], g2_, p2_ = strip_hidden_rests(s["gold"])
            hidden["hidden_groups_pred"] += g1
            hidden["hidden_groups_gold"] += g2_
            hidden["backup_pairs_pred"] += p1
            hidden["backup_pairs_gold"] += p2_
        e = edits(s["pred"], s["gold"])
        rows.append({"system": list(key), "gold_tokens": len(s["gold"]), "pred_tokens": len(s["pred"]), "edits": e,
                     "gold": s["gold"], "pred": s["pred"]})
        total_e += e
        total_g += len(s["gold"])
        g2, p2 = [t for t in s["gold"] if not t.startswith(DRAWING)], [t for t in s["pred"] if not t.startswith(DRAWING)]
        ns_e += edits(p2, g2)
        ns_g += len(g2)
    return {"gold_tokens": total_g, "edits": total_e, "edits_nostem": ns_e, "gold_tokens_nostem": ns_g, "systems": len(rows), "systems_with_gold": sum(r["gold_tokens"] > 0 for r in rows),
            "bar_gold_tokens": gold_total, "bar_edits_matched": matched_e, "bar_tokens_matched": matched_t,
            "bar_edits_all": all_e, "inserted_tokens": inserted_tokens, "deleted_tokens": deleted_tokens,
            "bars_read": len(read), "bars_refused": sum(refused), "pairs_with_unwritable_gold": unwritable_aligned,
            "measures": len(truth), "measures_unwritable": sum(t is None for t in truth),
            "bars_aligned": sum(b is not None and m is not None for b, m in pairs),
            "bars_unaligned": sum(m is None for b, m in pairs if b is not None),
            "measures_missed": sum(b is None for b, m in pairs if m is not None and truth[m] is not None),
            "per_system": rows, **hidden}


def evaluate(manifest: dict, systems: list[str] | None, pieces: list[str] | None, dump: Path | None = None, strip_hidden: bool = True) -> dict:
    warnings.simplefilter("ignore")
    out = {"pieces": {}, "systems": {}}
    for pid, p in manifest["pieces"].items():
        if pieces and pid not in pieces:
            continue
        gold_source = None
        if p.get("fallback_truth"):
            gold, gold_source = read_gold_fallback(p["truth"], p["fallback_truth"])
        else:
            gold = read_gold(p["truth"])
        sysof, info = partition_of(p["partition"], gold)
        out["pieces"][pid] = {**{k: p.get(k) for k in ("title", "period")}, "partition": info}
        if gold_source:
            out["pieces"][pid]["gold_source"] = gold_source
        for name, per_piece in manifest["systems"].items():
            if systems and name not in systems:
                continue
            if pid not in per_piece:  # never run on this piece
                continue
            spec = per_piece[pid]
            notes: list[str] = []
            fb_stats = None
            if spec.get("native_json"):
                read: list = native_bars(spec["native_json"])
            elif spec.get("files") and spec.get("fallback_files"):
                read, fb_stats = read_bars_fallback(spec["files"], spec["fallback_files"], notes)
            elif spec.get("files"):
                read = read_bars(spec["files"], notes)
            else:
                read = []
                notes.append("no output")
            r = score_piece(read, gold, sysof, strip_hidden=strip_hidden)
            r2 = score_piece(read, gold, sysof, drop_refused=True, strip_hidden=strip_hidden)
            res = {k: v for k, v in r.items() if k != "per_system"}
            res.update(ser=100 * r["edits"] / r["gold_tokens"] if r["gold_tokens"] else None,
                       ser_without_refused=100 * r2["edits"] / r2["gold_tokens"] if r2["gold_tokens"] else None,
                       ser_nostem=100 * r["edits_nostem"] / r["gold_tokens_nostem"] if r["gold_tokens_nostem"] else None,
                       edits_without_refused=r2["edits"], gold_tokens_without_refused=r2["gold_tokens"],
                       bar_ser_matched=100 * r["bar_edits_matched"] / r["bar_tokens_matched"] if r["bar_tokens_matched"] else None,
                       bar_ser_all=100 * r["bar_edits_all"] / r["bar_gold_tokens"] if r["bar_gold_tokens"] else None,
                       no_output=not read, notes=notes[:6], files=spec.get("files") or spec.get("native_json"))
            if fb_stats is not None:
                res["fallback"] = fb_stats
            res["note_F1"] = note_f1(spec, p["truth"], read)
            if spec.get("cut") and read:  # the first so many measures only, as experiments/13_comparison scored a system that read the first pages
                rc = score_piece(read, gold[: spec["cut"]], sysof[: spec["cut"]])
                res["bar_ser_matched_cut"] = 100 * rc["bar_edits_matched"] / rc["bar_tokens_matched"] if rc["bar_tokens_matched"] else None
            out["systems"].setdefault(name, {})[pid] = res
            if dump is not None:
                dump.mkdir(parents=True, exist_ok=True)
                (dump / f"{name.replace(' ', '_')}__{pid}.json").write_text(json.dumps(r["per_system"]), encoding="utf-8")
    return out


def note_f1(spec: dict, truth_path: str, read: list) -> float | None:
    """The direct scorer's note F1 (music21, no LMX) of the system's files on the piece, over the first `cut` measures if given."""
    files = spec.get("note_files") or spec.get("files")
    if not files or not read:
        return None
    import score_direct

    notes: list[str] = []
    bars: list = []
    for f in files:
        try:
            bars += score_direct.measures_of(Path(f), notes)
        except Exception:  # noqa: BLE001
            pass
    if not bars:
        return None
    truth = score_direct.measures_of(Path(truth_path), notes)
    if spec.get("cut"):
        truth = truth[: spec["cut"]]
    return 100 * score_direct.score(bars, truth)["note_F1"]


def pooled(res: dict, ids: list[str], key_e="edits", key_g="gold_tokens") -> float | None:
    rows = [res[i] for i in ids if i in res]
    g = sum(r[key_g] for r in rows)
    return 100 * sum(r[key_e] for r in rows) / g if g else None


# ---------------------------------------------------------------------------------------------


def selftest() -> int:
    from pdf2musicxml.lmx_bars import State

    # a hand example of the system join: three measures, the first two on one system, the third on the next
    gold = [("measure key:fifths:0 time beats:4 beat-type:4 clef:G2 staff:1 clef:F4 staff:2 C4 quarter staff:1", ""),
            ("measure D4 quarter staff:1", "key:fifths:0 time beats:4 beat-type:4 clef:G2 staff:1 clef:F4 staff:2"),
            ("measure E4 quarter staff:1", "key:fifths:0 time beats:4 beat-type:4 clef:G2 staff:1 clef:F4 staff:2")]
    sysof = [(1, 0), (1, 0), (1, 1)]
    same = [m.split() for m, _ in gold]
    r = score_piece(same, gold, sysof)
    assert r["edits"] == 0 and r["systems"] == 2, r["edits"]
    # one note wrong in the second measure: one substitution
    wrong = [list(b) for b in same]
    wrong[1][1] = "F4"
    r = score_piece(wrong, gold, sysof)
    assert r["edits"] == 1, r["edits"]
    # the third measure missed: its tokens are deleted (reset: key and clefs in front of a system's first measure)
    r = score_piece(same[:2], gold, sysof)
    assert r["edits"] == len(r["per_system"][1]["gold"]) > 0 and r["per_system"][0]["edits"] == 0, r["edits"]
    # no output: every gold token is an edit
    r = score_piece([], gold, sysof)
    assert r["edits"] == r["gold_tokens"], (r["edits"], r["gold_tokens"])
    # a system that does not repeat the key and clefs at the start of the next line is not punished: its bar there is
    # read from the state the bars before it left
    r = score_piece(same, gold, sysof)
    assert r["edits"] == 0
    # hidden-rest step (strip_hidden_rests): group removed with its backup, then dangling backup/forward pairs; both sides alike
    toks = "measure C4 quarter staff:1 backup whole print-object:no rest voice:6 whole staff:2 measure D4 quarter backup quarter forward half measure".split()
    out_, g_, p_ = strip_hidden_rests(toks)
    assert out_ == "measure C4 quarter staff:1 measure D4 quarter measure".split() and (g_, p_) == (1, 2), (out_, g_, p_)
    out_, g_, p_ = strip_hidden_rests("measure print-object:no rest voice:2 whole staff:1 dot E4 half".split())  # no backup before it: only the group goes
    assert out_ == "measure E4 half".split() and g_ == 1, out_
    assert strip_hidden_rests("measure rest whole staff:1 backup half".split())[0] == "measure rest whole staff:1".split()  # a visible rest stays; the trailing backup goes
    sg = [("measure key:fifths:0 time beats:4 beat-type:4 clef:G2 staff:1 clef:F4 staff:2 C4 quarter staff:1 backup whole print-object:no rest voice:6 whole staff:2", "")]
    plain = sg[0][0].split()[:-7]
    r = score_piece([plain], sg, [(1, 0)])
    assert r["edits"] > 0, r["edits"]
    r = score_piece([plain], sg, [(1, 0)], strip_hidden=True)
    assert r["edits"] == 0 and r["hidden_groups_gold"] == 1 and r["hidden_groups_pred"] == 0, r["edits"]
    # the lossless-or-raw rule: normalised bar if accepted and the pitched-note counts agree, else the raw bar if accepted, else empty
    A, B, C = ["measure", "C4"], ["measure", "D4"], ["measure", "E4", "F4"]
    norm = [(A, 1), (B, 1), (None, 2), (C, 2), (C, 2)]
    raw = [(A, 1), (A, 2), (A, 2), (None, 3), (B, 2)]
    out, st = merge_bars(norm, raw)
    # 1 same count: normalised; 2 count differs: raw; 3 normalised refused: raw; 4 count differs and raw refused: empty; 5 same count: normalised
    assert out == [A, A, A, None, C], out
    assert st == dict(normalised=2, fallback_notes=1, fallback_refused=1, empty=1), st
    g, gst = merge_gold([("a", "x"), (None, "y"), (None, "z")], [("a2", "x"), ("b2", "y"), (None, "z")])
    assert [m for m, _ in g] == ["a", "b2", None] and gst == dict(normalised=1, raw=1, excluded=1), (g, gst)
    assert PITCH_TOKEN.match("C4") and not PITCH_TOKEN.match("sharp") and not PITCH_TOKEN.match("quarter")
    print("selftest ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["run", "selftest"])
    ap.add_argument("manifest", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--systems", nargs="*")
    ap.add_argument("--pieces", nargs="*")
    ap.add_argument("--keep-hidden-rests", action="store_true", help="do not remove MuseScore's hidden-rest padding (strip_hidden_rests) from both sides")
    ap.add_argument("--dump-systems", help="a folder for every system's joined gold and prediction tokens")
    a = ap.parse_args(argv)
    if a.what == "selftest":
        return selftest()
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    res = evaluate(manifest, a.systems, a.pieces, Path(a.dump_systems) if a.dump_systems else None, strip_hidden=not a.keep_hidden_rests)
    text = json.dumps(res, indent=1, ensure_ascii=True, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    for name, per in res["systems"].items():
        for pid, r in per.items():
            print(f"{name:24} {pid:14} SER {r['ser'] if r['ser'] is None else round(r['ser'], 1):>6}  bar-matched "
                  f"{r['bar_ser_matched'] if r['bar_ser_matched'] is None else round(r['bar_ser_matched'], 1)}  refused {r['bars_refused']}/{r['bars_read']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
