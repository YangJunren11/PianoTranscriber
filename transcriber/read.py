"""How the bars of a system are read: each from the state the bar before left (`read_bar`), and a
system's bars together (`read_system`). `transcribe` uses these in the whole pipeline:

  page, scaled so a staff space is 9 pixels   (page)
   -> where each bar ends                      (bar counter, its network)
   -> the systems, their tops and bottoms      (staff counter)
   -> each bar cut out, with its mask          (bar cropper)
   -> each bar read as LMX, from the state the bar before left   (bar recognizer)
   -> the bars stitched into MusicXML          (reassembly)

A bar that runs on from one system to the next is cut twice, its start at the end of the first
system (`Bar.carried`); only its end, on the next system, is read for now, and a crop past a system's
last barline holding only a courtesy key, time or clef is left unread.

The main recognizer is trained with a system reset: it reads a bar that opens a system, one whose
(page, system) differs from the bar before's, from the notes tied over to it alone, and reads the key,
time and clefs printed there. A signature specialist (`starts`) can read every such bar instead; the main
recognizer reads the others, and a bar of those that writes a key, time or clef is read again by the
specialist, whose reading replaces it. `opener` keeps the score's first bar.

The time signature is printed once, so a wrong one read there (or a change that was missed) is carried
to every bar after it, and a recognizer obeys the state it is given. `TimeResolver` guards against this
at decoding time: the bars read since the time was last judged are checked for whether their voices add
up to the time in force (`transcribe.fills_time`); once at least MIN_BARS bars can be told, and fewer than
half fill it, they are read again from the same start with each of TIMES in the state, the time under
which most of them fill is kept (ties to the recognizer's own mean log probability) and is written into
the first bar as a time token. `transcribe`'s --no-resolve-time reads as before.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction
from typing import Callable

TIMES = ("2/4", "3/4", "4/4", "2/2", "3/8", "6/8", "9/8", "12/8", "5/4", "7/8", "6/4")  # the times tried, in this order
MIN_BARS = 4  # bars whose fill is known that a time is judged on
TRIGGER = Fraction(1, 2)  # a time is doubted when fewer than this share of the bars fill it


def read_bar(reader, opener, starts, crop, state, bars_read: int, opens: bool):
    """One bar read: (the model whose reading is kept, the state it was read from, its tokens). The
    score's first bar is read by `opener`, if given. Otherwise `reader` reads every bar, except one that
    opens a system, which goes to `starts` straight away (read from the state by that model's reset); a
    bar `reader` read that writes a key, time or clef is read again by `starts` from the same state
    (no reset: it opens no system), and that reading replaces the first."""
    from .lmx_bars import reset_state, sets_state

    def resets(model):
        return (getattr(model, "variant", {}).get("system_reset") or False) if opens else False

    if opener is not None and not bars_read:
        model = opener
    else:
        model = starts if starts is not None and opens else reader
    start = reset_state(state, resets(model))
    tokens = model.read([crop], [start])[0]
    if model is reader and starts is not None and sets_state(tokens):
        model, start = starts, state
        tokens = model.read([crop], [start])[0]
    return model, start, tokens


def systems(bars: list, before=None) -> list[list]:
    """Bars (with `page` and `system`) in order as one list for each system: runs with the same (page,
    system). `before` is the (page, system) of the bar before the first, if any."""
    out, key = [], before
    for b in bars:
        if (b.page, b.system) != key or not out:
            out.append([])
        out[-1].append(b)
        key = (b.page, b.system)
    return out


def read_system(reader, opener, starts, crops: list, state, bars_read: int, opens: bool, batch: bool = True,
                log: list | None = None, force_time: tuple | None = None) -> list[tuple]:
    """A system's bars read, each as `read_bar` would read it chained from `state`: a list of (model,
    state read from, tokens). With `batch`, the first bar is read as `read_bar` reads it and the others
    in one batch by `reader` (its encoder apart for each, so that every bar is read as alone), each from the state the first bar left (the ties it leaves dropped from the
    third bar on, as those are not known yet); then, walking the system with the true state, a bar whose
    true state differs from the one it was batch-read from (ties over, or a key, time or clef changed
    before it) is read again one at a time with `read_bar`, and so is one `starts` is to read for a
    signature `reader` wrote. `log` gets the index of each bar read again. `force_time` (("beats:3",
"beat-type:4")) is the time the first bar leaves in the state whatever it wrote itself: the bars after it
are read as if it had written that time (the tokens returned are as read: see `with_time`)."""
    from .lmx_bars import sets_state

    out = [read_bar(reader, opener, starts, crops[0], state, bars_read, opens)]
    state = out[0][1].after(out[0][2] if force_time is None else with_time(out[0][2], force_time, state))
    batched = None
    if batch and len(crops) > 1:
        guessed = [state] + [state.with_ties([])] * (len(crops) - 2)
        batched = getattr(reader, "read_apart", reader.read)(crops[1:], guessed)
    for k in range(1, len(crops)):
        if batched is not None and state == guessed[k - 1]:
            if starts is None or not sets_state(batched[k - 1]):
                out.append((reader, state, batched[k - 1]))
            else:  # the signature `reader` wrote is read again by `starts`, as read_bar does
                out.append((starts, state, starts.read([crops[k]], [state])[0]))
        else:
            if log is not None:
                log.append(k)
            out.append(read_bar(reader, opener, starts, crops[k], state, bars_read + k, False))
        state = out[-1][1].after(out[-1][2])
    return out


def time_of(name: str) -> tuple[str, str]:
    """A time signature as the state holds it: "12/8" is ("beats:12", "beat-type:8")."""
    beats, beat_type = name.split("/")
    return f"beats:{beats}", f"beat-type:{beat_type}"


def time_name(time) -> str | None:
    """The state's time as "12/8", or None for none."""
    return f"{time[0].split(':')[1]}/{time[1].split(':')[1]}" if time else None


def fill_share(flags: list) -> Fraction | None:
    """The share of the bars whose voices fill the time (`transcribe.fills_time`: True, False, or None for a
    bar it cannot be told of, which is left out); None if none can be told."""
    known = [f for f in flags if f is not None]
    return Fraction(sum(known), len(known)) if known else None


def needs_time(flags: list, min_bars: int = MIN_BARS, trigger: Fraction = TRIGGER) -> bool:
    """Whether the time in force is doubted: at least `min_bars` bars can be told of and fewer than `trigger`
    of those fill it."""
    known = [f for f in flags if f is not None]
    return len(known) >= min_bars and Fraction(sum(known), len(known)) < trigger


def bar_length(time) -> Fraction:
    """A bar's length in quarter notes, from the state's time or a name like "12/8"."""
    beats, beat_type = (time.split("/") if isinstance(time, str) else [t.split(":")[1] for t in time])
    return Fraction(4 * int(beats), int(beat_type))


def choose_time(current: Fraction | None, tried: dict, logprob: Callable[[str], float],
                length: Callable[[str], object] | None = None) -> str | None:
    """The time to change to, or None to keep the one in force. `current` is the share of bars that fill the
    time in force and `tried` the share for each time read again. A time is taken only if it beats the one in
    force; of those that fill the most bars, `logprob(time)` (the recognizer's mean log probability of the
    bars read with it, asked only for the tied ones) picks, and then the order of `tried`. Whether bars fill
    depends on the time's bar length alone, so with `length(time)` given, times of one length (12/8 and 6/4,
    3/4 and 6/8, 4/4 and 2/2) are one candidate, as good as its best share, and `logprob` picks among all of
    them: how many bars each happened to fill is no more than the recognizer's chance."""
    if not tried:
        return None
    score = tried
    if length is not None:
        best: dict = {}
        for t, share in tried.items():
            best[length(t)] = max(best.get(length(t), share), share)
        score = {t: best[length(t)] for t in tried}
    top = max(score.values())
    if current is not None and top <= current:
        return None
    leaders = [t for t, share in score.items() if share == top]
    return leaders[0] if len(leaders) == 1 else max(leaders, key=logprob)


def with_time(bar: list[str], time, before) -> list[str]:
    """`bar` with `time` as its time signature: one it wrote is given that time, else the token is put where LMX
    puts it (after `measure` and the key, before the clefs) if the state `before` it had another."""
    from .lmx_bars import KEYS

    bar = list(bar)
    for k in range(len(bar) - 2):
        if bar[k] == "time" and bar[k + 1].startswith("beats:") and bar[k + 2].startswith("beat-type:"):
            bar[k + 1:k + 3] = time
            return bar
    if before.time == time:
        return bar
    at = 1 if bar[:1] == ["measure"] else 0
    while at < len(bar) and bar[at] in KEYS:
        at += 1
    return bar[:at] + ["time", *time] + bar[at:]


@dataclasses.dataclass
class Pending:
    """A system read: what is passed on (`meta`, which is the caller's own), its crops, the state it was read
    from, the number of bars before it, whether it opens a system, the bars (model, state read from, tokens)
    and the tokens as the model wrote them (a first bar given a time differs)."""

    meta: object
    crops: list
    state: object
    bars_before: int
    opens: bool
    reads: list
    raw: list


@dataclasses.dataclass
class Done:
    """A system settled: as `Pending`, and the time it was read again with, if it was (else None)."""

    meta: object
    opens: bool
    reads: list
    raw: list
    resolved: str | None


class TimeResolver:
    """Systems read one after another, their time judged as they come. A system's bars are held until the
    bars held (they may be several systems of a page) have at least `min_bars` that can be told
    of whether they fill the time; if fewer than `trigger` of them do, they are read again from the state they
    started from with each of `times` in place of its time, the first bar's time forced (`read_system`'s
    `force_time`), and the time that makes the most bars fill is kept: written into the first bar if the state
    changed (`with_time`) and carried on, for the bars after them are read from it. A time must beat the one
    in force to be taken. With `by_length` (the default), a time of the same bar length as the one in force is
    not tried, as no count of bars that fill can tell them apart, and times of one length are told apart by the
    recognizer's log probability alone (`choose_time`); without it, every time is tried and the one that
    happened to fill most bars wins. With `on` False nothing is held or read again.

    Use: `read_system(..., self.state, self.count, ...)` for each system, `add` its bars and handle the
    systems `add` returns, and `flush` at the end of each page (bars too few to judge a time on are kept
    as read: a page's bars are judged on that page). `windows` has a record for each set of bars read again."""

    def __init__(self, reader, opener, starts, batch: bool = True, on: bool = True, times=TIMES,
                 min_bars: int = MIN_BARS, trigger: Fraction = TRIGGER, say: Callable[[str], None] = print,
                 by_length: bool = True):
        from .lmx_bars import State

        self.reader, self.opener, self.starts, self.batch, self.on = reader, opener, starts, batch, on
        self.times, self.min_bars, self.trigger, self.say, self.by_length = times, min_bars, trigger, say, by_length
        self.state, self.count = State(), 0  # what the next system is read from, and how many bars came before it
        self.held: list[Pending] = []
        self.windows: list[dict] = []

    def add(self, meta, crops: list, reads: list, opens: bool) -> list[Done]:
        """A system's bars read from `self.state`; the systems that are settled now, in order (none while
        the time cannot yet be judged)."""
        item = Pending(meta, crops, self.state, self.count, opens, reads, [t for *_, t in reads])
        self.state, self.count = reads[-1][1].after(reads[-1][2]), self.count + len(reads)
        if not self.on:
            return [Done(meta, opens, reads, item.raw, None)]
        self.held.append(item)
        flags = self._flags(self.held)
        return self._settle(judge=True) if sum(f is not None for f in flags) >= self.min_bars else []

    def flush(self) -> list[Done]:
        """The systems still held, settled as they are: too few bars to judge the time on."""
        return self._settle(judge=False)

    def _flags(self, held: list[Pending]) -> list:
        from .transcribe import fills_time

        return [fills_time(tokens, start, p.bars_before + k == 0)
                for p in held for k, (_, start, tokens) in enumerate(p.reads)]

    def _settle(self, judge: bool) -> list[Done]:
        held, self.held = self.held, []
        resolved = None
        if held and judge and needs_time(self._flags(held), self.min_bars, self.trigger):
            held, resolved = self._reread(held)
        return [Done(p.meta, p.opens, p.reads, p.raw, resolved) for p in held]

    def _reread(self, held: list[Pending]) -> tuple[list[Pending], str | None]:
        first = held[0]
        now = first.reads[0][1].after(first.reads[0][2]).time  # the time in force after the first bar
        share = fill_share(self._flags(held))
        tried, results = {}, {}
        for name in self.times:
            if time_of(name) != now and not (self.by_length and now and bar_length(name) == bar_length(now)):
                results[name] = self._read(held, time_of(name))
                tried[name] = fill_share(self._flags(results[name])) or Fraction(0)
        logprobs: dict[str, float] = {}

        def logprob(name: str) -> float:
            if name not in logprobs:
                logprobs[name] = self._logprob(results[name])
            return logprobs[name]

        chosen = choose_time(share, tried, logprob, bar_length if self.by_length else None)
        bars = sum(len(p.reads) for p in held)
        self.windows.append({"first_bar": first.bars_before, "bars": bars, "systems": len(held),
                             "time_before": time_name(now), "share_before": float(share or 0),
                             "tried": {n: float(s) for n, s in tried.items()}, "mean_logprob_of_tied": logprobs,
                             "chosen": chosen})
        self.say(f"time: bars {first.bars_before} to {first.bars_before + bars - 1}: {float(share or 0):.2f} fill "
                 f"{time_name(now)}; read again with {len(tried)} times, " +
                 (f"kept {time_name(now)}" if chosen is None else f"chose {chosen} ({float(tried[chosen]):.2f} fill)"))
        if chosen is None:
            return held, None
        done = results[chosen]
        self.state = done[-1].reads[-1][1].after(done[-1].reads[-1][2])
        return done, chosen

    def _read(self, held: list[Pending], time) -> list[Pending]:
        """The held systems read again in a row from the state the first started from, with `time` in it."""
        state, out = dataclasses.replace(held[0].state, time=time), []
        for k, p in enumerate(held):
            reads = read_system(self.reader, self.opener, self.starts, p.crops, state, p.bars_before, p.opens,
                                self.batch, force_time=time if not k else None)
            raw = [t for *_, t in reads]
            if not k:
                model, start, tokens = reads[0]
                reads[0] = (model, start, with_time(tokens, time, p.state))
            out.append(Pending(p.meta, p.crops, state, p.bars_before, p.opens, reads, raw))
            state = reads[-1][1].after(reads[-1][2])
        return out

    def _logprob(self, held: list[Pending]) -> float:
        """The recognizer's mean log probability of the bars it read (`transcribe.confidences`, on the tokens as
        it wrote them), averaged over the bars."""
        from .transcribe import confidences

        rows = [(model, crop, start, raw) for p in held for crop, (model, start, _), raw in zip(p.crops, p.reads, p.raw)]
        values = []
        for model in {id(r[0]): r[0] for r in rows}.values():
            mine = [r for r in rows if r[0] is model]
            values += [c["mean_logprob"] for c in confidences(model, [r[1] for r in mine], [r[2] for r in mine],
                                                              [r[3] for r in mine])]
        return sum(values) / len(values)
