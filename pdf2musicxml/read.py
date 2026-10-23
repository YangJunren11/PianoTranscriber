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
"""

from __future__ import annotations


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
                log: list | None = None) -> list[tuple]:
    """A system's bars read, each as `read_bar` would read it chained from `state`: a list of (model,
    state read from, tokens). With `batch`, the first bar is read as `read_bar` reads it and the others
    in one batch by `reader` (its encoder apart for each, so that every bar is read as alone), each from the state the first bar left (the ties it leaves dropped from the
    third bar on, as those are not known yet); then, walking the system with the true state, a bar whose
    true state differs from the one it was batch-read from (ties over, or a key, time or clef changed
    before it) is read again one at a time with `read_bar`, and so is one `starts` is to read for a
    signature `reader` wrote. `log` gets the index of each bar read again."""
    from .lmx_bars import sets_state

    out = [read_bar(reader, opener, starts, crops[0], state, bars_read, opens)]
    state = out[0][1].after(out[0][2])
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
