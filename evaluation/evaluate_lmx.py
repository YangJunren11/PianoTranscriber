"""How right bar recognizers' LMX is, side by side, on the bar dataset's validation bars (or another
split): each bar read scored against its true LMX by transcriber/lmx_accuracy.py, as tokens (token
error rate, bars exact) and as music (notes read in order, pitches, notes in time, the state left for
the next bar, and whether the bar reads back as MusicXML at all), for every model and every house
style.

A model trained with a system reset (system reset) reads a bar that opens a system from
the notes tied over to it alone, and writes the key, time and clefs it prints: such a bar is scored
against its true LMX with those in front (`lmx_bars.standalone`), so tokens are compared like with like.

Each model reads every bar from its true state (what the bar before truly left), so each bar is
scored on its own; with --chained, from the state its own reading of the bar before left, through
each piece in each house style as the pipeline reads a score, so a clef, key or tie misread carries
on into the bars after it; with --first-model too, each piece's first bar, read with no state in
front, is read by that model and the bars after it by the others. With --starts-model , a signature
specialist: every bar that opens a system (record field "first") is read by that model, from the state its
variant makes of the state before, and scored against the truth for its reset mode; the --models read the
rest, and a bar of those whose reading writes a key, time or clef (`lmx_bars.sets_state`) is read again by
the specialist from the same state (no reset: it opens no system), whose reading replaces the first. Bars read some other way can be scored too (--predictions: a JSON line a
bar, {"image": ..., "lmx": ...}); and --oracle scores the true LMX against itself, which must come
to full marks: a check of the test itself.

  python evaluation/evaluate_lmx.py data/bars_lmx --models runs/a.pt runs/b.pt [--predictions read.jsonl]
      [--split val] [--limit 2000] [--chained [--first-model runs/first.pt]] [--starts-model runs/starts.pt] [--oracle] [--by style]
      [--out scores.jsonl]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transcriber.lmx_accuracy import Tally, score_bar, table  # noqa: E402
from transcriber.lmx_bars import State, reset_state, reset_tokens, sets_state, true_states  # noqa: E402


def resets(reader, row: dict) -> bool | str:
    """How the reader reads this bar without the state's signatures, or False: it opens a system, and
    the reader was trained to read the signatures printed there (True: all; "key-clefs": not the time)."""
    return (reader.variant.get("system_reset") or False) if reader is not None and row.get("first") else False


def truth_of(reader, row: dict, state: State) -> str:
    """The LMX the reader is to write for a bar: the bar's, with its signatures in front if it resets."""
    return " ".join(reset_tokens(row["lmx"].split(), state, resets(reader, row)))


def route(reader, starts, row: dict):
    """The reader of a bar: the starts model for a bar that opens a system, if there is one."""
    return starts if starts is not None and row.get("first") else reader


def count_routed(stats: dict | None, rows: int, at_start: int, signature: int) -> None:
    if stats is not None:
        stats["rows"] = stats.get("rows", 0) + rows
        stats["at_start"] = stats.get("at_start", 0) + at_start
        stats["signature"] = stats.get("signature", 0) + signature


def read_routed(reader, starts, root: Path, rows: list[dict], states: list[State], size: int,
                stats: dict | None = None) -> tuple[list[str], list[State], list[str]]:
    """Each bar read from its true state by its reader (`route`), after the reset of that reader's
    variant; then a bar `reader` read that writes a key, time or clef is read again by `starts` from the
    same state, and the second reading is kept (its truth is the bar's own, as it opens no system).
    Also the states given and the LMX each was to be read as."""
    models = [route(reader, starts, r) for r in rows]
    given = [reset_state(s, resets(m, r)) for m, r, s in zip(models, rows, states)]
    truths = [truth_of(m, r, s) for m, r, s in zip(models, rows, states)]
    out = [""] * len(rows)
    for model in {id(m): m for m in models}.values():
        mine = [k for k, m in enumerate(models) if m is model]
        for k, bar in zip(mine, read_bars(model, root, [rows[k] for k in mine], [given[k] for k in mine], size)):
            out[k] = bar
    if starts is not None:
        again = [k for k, m in enumerate(models) if m is reader and sets_state(out[k].split())]
        for k, bar in zip(again, read_bars(starts, root, [rows[k] for k in again], [given[k] for k in again], size)):
            out[k] = bar
        count_routed(stats, len(rows), sum(m is starts for m in models), len(again))
    return out, given, truths


def read_bars(reader, root: Path, rows: list[dict], states: list[State], size: int) -> list[str]:
    """Each bar read from the state given for it, in batches of about the same width."""
    order = sorted(range(len(rows)), key=lambda k: rows[k]["mask"][2])
    out = [""] * len(rows)
    for k in range(0, len(order), size):
        part = order[k:k + size]
        crops = [(np.asarray(Image.open(root / rows[i]["image"]).convert("L")), rows[i]["mask"]) for i in part]
        for i, bar in zip(part, reader.read(crops, [states[i] for i in part])):
            out[i] = " ".join(bar)
    return out


def read_chained(reader, root: Path, rows: list[dict], states: list[State], size: int,
                 first=None, starts=None, stats: dict | None = None) -> tuple[list[str], list[State], list[str]]:
    """Each piece in each house style read bar after bar, each bar from the state the reading of the
    bar before left; a bar whose bar before is not in the dataset starts again from its true state.
    The pieces are read side by side, a bar of each at a time; a piece's first bar by `first`, if
    given, a bar that opens a system by `starts`, and a bar `reader` read that writes a key, time or
    clef read again by `starts` from the same state (see `read_routed`), if given. Also the states each bar was read from, and the LMX each was to be read as (see `truth_of`)."""
    runs = collections.defaultdict(list)
    for k, r in enumerate(rows):
        runs[(r["piece"], r["style"])].append(k)
    runs = [sorted(v, key=lambda k: rows[k]["bar"]) for v in runs.values()]
    out, used, truths = [""] * len(rows), list(states), [r["lmx"] for r in rows]
    left: dict[int, State] = {}  # the state each run's last reading left
    for step in range(max(map(len, runs))):
        live = [(n, run[step]) for n, run in enumerate(runs) if step < len(run)]
        for j in range(0, len(live), size):
            part = live[j:j + size]
            for n, k in part:
                before = runs[n][step - 1] if step else None
                if before is not None and rows[before]["bar"] == rows[k]["bar"] - 1:
                    used[k] = left[n]
            models = [first if first is not None and rows[k]["bar"] == 0 else route(reader, starts, rows[k])
                      for _, k in part]
            for (n, k), model in zip(part, models):
                if resets(model, rows[k]):
                    used[k] = reset_state(used[k], resets(model, rows[k]))
                truths[k] = truth_of(model, rows[k], states[k])
            crops = [(np.asarray(Image.open(root / rows[k]["image"]).convert("L")), rows[k]["mask"]) for _, k in part]
            read: list[list[str]] = [[] for _ in part]
            for model in {id(m): m for m in models}.values():
                mine = [i for i, m in enumerate(models) if m is model]
                for i, bar in zip(mine, model.read([crops[i] for i in mine], [used[part[i][1]] for i in mine]) if mine else []):
                    read[i] = bar
            if starts is not None:
                again = [i for i, m in enumerate(models) if m is reader and sets_state(read[i])]
                for i, bar in zip(again, starts.read([crops[i] for i in again], [used[part[i][1]] for i in again]) if again else []):
                    read[i] = bar
                count_routed(stats, len(part), sum(m is starts for m in models), len(again))
            for (n, k), bar in zip(part, read):
                out[k] = " ".join(bar)
                left[n] = used[k].after(bar)
    return out, used, truths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="the bar dataset (evaluation/systems_dataset.py or the bar dataset format)")
    ap.add_argument("--models", nargs="*", default=[], help="bar recognizers (training/train_recognizer.py)")
    ap.add_argument("--predictions", nargs="*", default=[], help="bars read some other way: JSON lines, image and lmx")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0, help="only so many bars, drawn at random")
    ap.add_argument("--chained", action="store_true", help="each bar from the state the model's reading of the bar before left")
    ap.add_argument("--first-model", default="", help="chained: a piece's first bar read by this model")
    ap.add_argument("--starts-model", default="",
                    help="a bar that opens a system is read by this model (a system-start specialist); the --models read the rest")
    ap.add_argument("--oracle", action="store_true", help="the true LMX against itself too: a check of the test")
    ap.add_argument("--by", default="style", choices=["style", "piece", "none"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="", help="every bar's scores, a JSON line a bar and model")
    args = ap.parse_args()
    root = Path(args.data)
    with (root / "bars.jsonl").open(encoding="utf-8") as f:
        rows = [r for r in map(json.loads, f) if r["split"] == args.split]
    if args.limit and args.limit < len(rows):
        if args.chained:  # whole pieces in a style, so their bars follow one another
            runs = sorted({(r["piece"], r["style"]) for r in rows})
            random.Random(0).shuffle(runs)
            keep, n = set(), 0
            for run in runs:
                if n >= args.limit:
                    break
                keep.add(run)
                n += sum((r["piece"], r["style"]) == run for r in rows)
            rows = [r for r in rows if (r["piece"], r["style"]) in keep]
        else:
            rows = random.Random(0).sample(rows, args.limit)
    states = true_states(rows)
    print(f"{len(rows)} {args.split} bars, from {len({r['piece'] for r in rows})} pieces", flush=True)
    readings: dict[str, tuple[list[str], list[State], list[str]]] = {}
    if args.oracle:
        readings["true LMX (oracle)"] = ([r["lmx"] for r in rows], states, [r["lmx"] for r in rows])
    for path in args.predictions:
        with open(path, encoding="utf-8") as f:
            read = {p["image"]: p["lmx"] for p in map(json.loads, f)}
        readings[Path(path).stem] = ([read.get(r["image"], "") for r in rows], states, [r["lmx"] for r in rows])
    if args.models:
        from transcriber.bar_recognizer import Reader

        first = Reader(args.first_model) if args.first_model else None
        starts = Reader(args.starts_model) if args.starts_model else None
        stats: dict = {}
        for path in args.models:
            started = time.time()
            reader = Reader(path)
            if args.chained:
                name = Path(path).stem + (f" (chained, {Path(args.first_model).stem} first)" if first else " (chained)")
                if starts is not None:
                    name = name.removesuffix(")") + ", " + f"{Path(args.starts_model).stem} at system starts)"
                stats.clear()
                readings[name] = read_chained(reader, root, rows, states, args.batch, first, starts, stats)
            elif starts is not None:
                stats.clear()
                readings[f"{Path(path).stem} ({Path(args.starts_model).stem} at system starts)"] = read_routed(
                    reader, starts, root, rows, states, args.batch, stats)
            else:
                given = [reset_state(s, resets(reader, r)) for r, s in zip(rows, states)]
                readings[Path(path).stem] = (read_bars(reader, root, rows, given, args.batch), given,
                                             [truth_of(reader, r, s) for r, s in zip(rows, states)])
            if starts is not None:
                a, g = stats.get("at_start", 0), stats.get("signature", 0)
                print(f"{a + g} of {stats.get('rows', 0)} bars read by the specialist: {a} at system starts, "
                      f"{g} for a signature the main model wrote", flush=True)
            print(f"{path}: {len(rows)} bars read in {time.time() - started:.0f} s", flush=True)
    tallies = {}
    out = open(args.out, "w", encoding="utf-8") if args.out else None
    for name, (read, used, truths) in readings.items():
        tally = Tally()
        for r, bar, state, true_state, truth in zip(rows, read, used, states, truths):
            # Read from the state the reading before left (chained), a bar sounds as that state
            # makes it, as it will in the score put together: a key misread carries on
            s = score_bar(bar, truth, true_state, read_from=state)
            tally.add(s, None if args.by == "none" else r[args.by])
            if out:
                out.write(json.dumps({"model": name, "image": r["image"], "read": bar, **asdict(s),
                                      "state_given_right": state == true_state or state == reset_state(true_state, True)
                                                           or state == reset_state(true_state, "key-clefs")}) + "\n")
        tallies[name] = tally
    if out:
        out.close()
    print(table(tallies, groups=args.by != "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
