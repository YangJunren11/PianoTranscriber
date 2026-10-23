"""A PDF of piano music read into MusicXML, with a JSON of data about every bar read besides.

The JSON holds, for every bar read, what can be told without the answer: where the bar counter's peak
was and whether a barline lay under it, how many tokens the bar has, whether they read back as one bar
of MusicXML, whether the notes fill the time signature, and how sure the recognizer was of each token
(one more pass over the bar's tokens, teacher-forced with what it just wrote).

  python -m pdf2musicxml.transcribe score.pdf --out score.musicxml [--json score.json] [--pages 1-3,7]
      [--lmx score.lmx] [--threads N] [--no-batch]

With no model flags the models in `models/` are used: the bar counter, the main recognizer and the
specialist that reads the bars opening a system. Another can be given with --bar-counter, --recognizer,
--starts-model (empty for none) and --first-model.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from .lmx_bars import State, notes_of, reset_state

DOUBTFUL = 0.5  # a bar with a token the recognizer gave less probability than this is doubtful
BATCH = 16  # bars through the recognizer together, for their probabilities


def page_numbers(text: str) -> list[int] | None:
    """Pages as the command line takes them: '1-3,7' is 1, 2, 3, 7; nothing is all."""
    return sorted({n for part in text.split(",") if part
                   for n in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)}) or None


def plain(value):
    """Anything a JSON holds: numpy numbers and dataclasses made plain, floats rounded to 4 places."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 4)
    return value


# ---------------------------------------------------------------------------------------------
# Reading


def reset(model, state: State, first: bool) -> State:
    """The state a bar is read from: the state the bar before left, or for a bar that opens a system
    (by a recognizer trained for it) only the notes tied over to it."""
    return reset_state(state, resets(model, first))


def confidences(reader, crops: list, prefixes: list, bars: list[list[str]]) -> list[dict]:
    """How sure the recognizer was of each bar: one teacher-forced pass over the prefix, the tokens
    read and the end, and the probability it gave each token read (and the end). `prefixes` are
    the states the bars were read from, or their prefixes as token ids."""
    import torch

    from .bar_recognizer import EOS, PAD, batch, prepare

    out = []
    for k in range(0, len(bars), BATCH):
        part = slice(k, k + BATCH)
        starts = [reader.vocabulary.prefix(p) if isinstance(p, State) else p for p in prefixes[part]]
        ids = [p + reader.vocabulary.encode(b) + [EOS] for p, b in zip(starts, bars[part])]
        tokens = torch.full((len(ids), max(len(i) for i in ids)), PAD, dtype=torch.long)
        for row, i in enumerate(ids):
            tokens[row, :len(i)] = torch.tensor(i)
        images, padding = batch([prepare(image, mask) for image, mask in crops[part]])
        with torch.no_grad():
            memory, memory_padding = reader.model.encode(images.to(reader.device), padding.to(reader.device))
            logits = reader.model.decode(memory, memory_padding, tokens[:, :-1].to(reader.device))
            logprob = torch.log_softmax(logits.float(), -1).cpu()
        for row, (p, i) in enumerate(zip(starts, ids)):
            at = torch.arange(len(p) - 1, len(i) - 1)  # where the bar's tokens and the end are predicted
            lp = logprob[row, at, torch.tensor(i[len(p):])]
            out.append({"mean_logprob": float(lp.mean()), "min_prob": float(lp.exp().min())})
    return out


def resets(model, first: bool) -> bool | str:
    """How `model` reads a bar that opens a system without the state's signatures, or False: True from
    a state with only the ties, "key-clefs" with the time kept too."""
    return (getattr(model, "variant", {}).get("system_reset") or False) if first else False


def fills_time(bar: list[str], state: State, opening: bool) -> bool | None:
    """Whether the notes of each staff add up to the bar's time signature; None where no time is
    known or the bar may be a pickup (the score's first)."""
    time_ = state.after(bar).time
    if time_ is None or opening:
        return None
    notes, valid = notes_of(bar, state)
    if not valid or not notes:
        return False
    length = Fraction(4 * int(time_[0].split(":")[1]), int(time_[1].split(":")[1]))
    ends: dict[int, Fraction] = {}
    for n in notes:
        ends[n.staff] = max(ends.get(n.staff, Fraction(0)), n.onset + n.duration)
    return all(e == length for e in ends.values())


def transcribe(pdf, counter, reader, first=None, pages: list[int] | None = None, starts=None, batch: bool = True):
    """A PDF read bar after bar, each from the state the bar before left (the score's first bar by
    `first`, a bar that opens a system by `starts`, and a bar the reader wrote a key, time or clef in read
    again by `starts`, if given: see `read.read_bar`). With `batch`, a system's bars are read together and
    only those that depend on the bar before them are read again (`read.read_system`): the same bars
    as without it. The bars' tokens, a record for each bar read, and one for each page."""
    from .bar_cropper import crop
    from .page import pdf_pages
    from .read import read_system, systems

    bars, records, info = [], [], []
    state, number, before = State(), 0, None
    for page_number, page in zip(pages or range(1, 10 ** 6), pdf_pages(pdf, pages)):
        cropped = crop(page.pixels, counter.peaks(page.pixels), page_number, number)
        read = [b for b in cropped.bars if not b.carried]
        mine = []  # this page's bars: (model, crop, state read from, tokens)
        for group in systems(read, before):
            opens = (group[0].page, group[0].system) != before
            reads = read_system(reader, first, starts, [(b.image, b.mask) for b in group], state, len(bars), opens, batch)
            for k, (b, (model, start, tokens)) in enumerate(zip(group, reads)):
                opens_bar = opens and not k
                notes, valid = notes_of(tokens, start)
                records.append({"number": b.number, "page": b.page, "system": b.system, "first": opens_bar, "reset": resets(model, opens_bar), "by_starts_model": starts is not None and model is starts, "sure": b.sure,
                                "snapped": b.snapped, "box": list(b.box), "tokens": len(tokens), "lmx": " ".join(tokens),
                                "valid": valid, "notes": len(notes), "fills_time": fills_time(tokens, start, not bars),
                                "read_from": " ".join(start.tokens())})
                mine.append((model, (b.image, b.mask), start, tokens))
                bars.append(tokens)
                before, state = (b.page, b.system), start.after(tokens)
        for model in {id(m): m for m, *_ in mine}.values():  # the first model's bar and the reader's, apart
            rows = [k for k, row in enumerate(mine) if row[0] is model]
            sure = confidences(model, [mine[k][1] for k in rows], [mine[k][2] for k in rows], [mine[k][3] for k in rows])
            for k, c in zip(rows, sure):
                records[len(records) - len(mine) + k].update(c)
        info.append({"number": page_number, "systems": len(cropped.level.systems),
                     "staves": [len(s.staves) for s in cropped.level.systems], "bars": len(read),
                     "carried": len(cropped.bars) - len(read), "stray_peaks": len(cropped.stray),
                     "bars_not_snapped": sum(not b.snapped for b in read), "scale": page.scale})
        print(f"page {page_number}: {len(read)} bars read, {number} to {cropped.next - 1}", flush=True)
        number = cropped.next
    return bars, records, info


# ---------------------------------------------------------------------------------------------


def summary(records: list[dict], info: list[dict], models: dict, variant: dict, elapsed: float) -> dict:
    n = max(len(records), 1)
    fills = [r["fills_time"] for r in records if r["fills_time"] is not None]
    return {"bars": len(records), "pages": len(info), "valid_share": sum(r["valid"] for r in records) / n,
            "mean_logprob": sum(r["mean_logprob"] for r in records) / n,
            "bars_min_prob_under_half": sum(r["min_prob"] < DOUBTFUL for r in records),
            "bars_not_snapped": sum(not r["snapped"] for r in records),
            "stray_peaks": sum(p["stray_peaks"] for p in info),
            "fills_time_share": sum(fills) / len(fills) if fills else None,
            "models": models, "variant": variant, "elapsed_seconds": elapsed}


def run(pdf, out, counter, reader, opener, starts, pages, json_path, lmx_path, models, started,
        batch: bool = True) -> list[list[str]]:
    """A PDF read with the models given, the MusicXML written to `out`, the JSON of data about every bar to
    `json_path` and the LMX, a bar a line, to `lmx_path` if given. `models` names the models for the JSON's summary and `started` is when the work began."""
    from .reassembly import write

    bars, records, info = transcribe(pdf, counter, reader, opener, pages, starts, batch)
    errors = write(bars, out)
    if lmx_path:
        Path(lmx_path).write_text("\n".join(" ".join(b) for b in bars) + "\n", encoding="utf-8")
    data = {"per_bar": records, "pages": info,
            "summary": summary(records, info, models, getattr(reader, "variant", {}), time.time() - started)}
    if json_path:
        Path(json_path).write_text(json.dumps(plain(data), indent=1) + "\n", encoding="utf-8")
    print(json.dumps(plain(data["summary"])))
    print(f"{len(bars)} bars written to {out}" + (f"; the Delinearizer objected: {errors[:300]}" if errors else ""))
    return bars


def main(argv: list[str] | None = None) -> int:
    from .api import DEFAULT_BAR_COUNTER, DEFAULT_RECOGNIZER, DEFAULT_STARTS_MODEL, Transcriber

    ap = argparse.ArgumentParser(description="A PDF of piano music read into MusicXML.")
    ap.add_argument("pdf")
    ap.add_argument("--out", required=True, help="the MusicXML to write")
    ap.add_argument("--json", default="", help="data about every bar read, to write as JSON")
    ap.add_argument("--pages", default="", help="pages to read, e.g. 1-3,7 (default: all)")
    ap.add_argument("--lmx", default="", help="the score's LMX too, a bar a line")
    ap.add_argument("--bar-counter", default=DEFAULT_BAR_COUNTER, help="the bar counter's weights (default: %(default)s)")
    ap.add_argument("--recognizer", default=DEFAULT_RECOGNIZER, help="a trained bar recognizer (default: %(default)s)")
    ap.add_argument("--starts-model", default=DEFAULT_STARTS_MODEL,
                    help="a bar that opens a system read by this recognizer instead; empty for none (default: %(default)s)")
    ap.add_argument("--first-model", default="", help="the score's first bar read by this recognizer instead")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-batch", action="store_true", help="read the bars one at a time, not a system's bars together")
    args = ap.parse_args(argv)
    Transcriber(args.recognizer, args.starts_model, args.bar_counter, threads=args.threads or None,
                first_model=args.first_model).transcribe(
        args.pdf, args.out, page_numbers(args.pages), args.json or None, args.lmx or None, batch=not args.no_batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
