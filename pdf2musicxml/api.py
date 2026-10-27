"""PDF to MusicXML from Python: `pdf_to_musicxml` reads one PDF, `Transcriber` keeps the models loaded
to read many. Both do what `python -m pdf2musicxml.transcribe` does, with the models the repository
ships in `models/` as defaults (a path that is not absolute is taken from the repository's root)."""

from __future__ import annotations

import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECOGNIZER = "models/main.pt"
DEFAULT_STARTS_MODEL = "models/specialist.pt"
DEFAULT_BAR_COUNTER = "models/barcounter.pt"


def resolve(path, what: str) -> Path:
    """`path` as a file that exists, taken from the repository's root if not absolute, or a
    FileNotFoundError naming what it was for and where it looked."""
    found = Path(path)
    if not found.is_absolute():
        found = ROOT / found
    if not found.exists():
        raise FileNotFoundError(f"{what} not found: {found} (give another with the {what.replace(' ', '_')} argument)")
    return found.resolve()


def pages_of(pages) -> list[int] | None:
    """Pages as a sorted list of 1-based numbers, from such a list or a string like '1-3,7'; nothing is all."""
    from .transcribe import page_numbers

    if pages is None:
        return None
    return page_numbers(pages) if isinstance(pages, str) else sorted(set(int(p) for p in pages)) or None


class Transcriber:
    """The models that read a PDF, loaded once: the bar counter, the main recognizer, the state specialist
    that reads the bars opening a system (None to go without) and optionally a recognizer for the score's
    first bar. `counter`, `reader`, `starts` and `opener` stand in for the loaded models, and then no
    path is looked at."""

    def __init__(self, recognizer=DEFAULT_RECOGNIZER, starts_model=DEFAULT_STARTS_MODEL,
                 bar_counter=DEFAULT_BAR_COUNTER, device=None, threads=None, first_model=None,
                 *, counter=None, reader=None, starts=None, opener=None):
        # every path is looked for before anything is loaded, so that a wrong one costs nothing
        if counter is None:
            bar_counter = resolve(bar_counter, "bar counter")
        if reader is None:
            recognizer = resolve(recognizer, "recognizer")
        if starts is None and starts_model:
            starts_model = resolve(starts_model, "starts model")
        if opener is None and first_model:
            first_model = resolve(first_model, "first model")
        if threads:
            import torch

            torch.set_num_threads(threads)
        if counter is None:
            from .bar_counter import BarCounter

            counter = BarCounter(bar_counter, threads or None)
        if reader is None or (starts is None and starts_model) or (opener is None and first_model):
            from .bar_recognizer import Reader
        if reader is None:
            reader = Reader(recognizer, device)
        if starts is None and starts_model:
            starts = Reader(starts_model, device)
        if opener is None and first_model:
            opener = Reader(first_model, device)
        self.counter, self.reader, self.starts, self.opener = counter, reader, starts, opener
        self.models = {"bar_counter": str(bar_counter), "recognizer": str(recognizer),
                       "first_model": str(first_model) if first_model else None,
                       "starts_model": str(starts_model) if starts_model else None}

    def transcribe(self, pdf, out, pages=None, json_path=None, lmx_path=None, batch=True,
                   resolve_time=True) -> Path:
        """`pdf` read into MusicXML at `out`, which is returned. `pages` (1-based, a list or '1-3,7') limits
        the pages read; `json_path` gets a JSON of data about every bar read, `lmx_path` the LMX a bar a line.
        With `resolve_time` (the default), a time signature that the bars do not fill is replaced by the one
        that makes them fill (bars read again, `read.TimeResolver`); False carries the time as it was read."""
        from .transcribe import run

        started = time.time()
        run(pdf, out, self.counter, self.reader, self.opener, self.starts, pages_of(pages), json_path, lmx_path,
            self.models, started, batch, resolve_time)
        return Path(out)


def pdf_to_musicxml(pdf, out, *, pages=None, json_path=None, lmx_path=None,
                    recognizer=DEFAULT_RECOGNIZER, starts_model=DEFAULT_STARTS_MODEL,
                    first_model=None, bar_counter=DEFAULT_BAR_COUNTER,
                    device=None, threads=None, batch=True, resolve_time=True) -> Path:
    """One PDF of piano music read into MusicXML at `out`, which is returned. `pages` (1-based, a list or
    '1-3,7') limits the pages read, `json_path` and `lmx_path` ask for the JSON of data about every bar and
    the LMX. The models are loaded for this one call: to read many PDFs, make a `Transcriber` and call its
    `transcribe`. `resolve_time=False` carries the time signature as it was read (see `Transcriber.transcribe`)."""
    return Transcriber(recognizer, starts_model, bar_counter, device, threads, first_model).transcribe(
        pdf, out, pages, json_path, lmx_path, batch, resolve_time)
