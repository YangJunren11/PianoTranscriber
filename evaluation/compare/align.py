"""Bars read aligned to the true measures of a score, and the true measures as LMX. Taken unchanged from the research
repository: `align` and `load_truth` from pdf2musicxml/transcribe.py, and `notes`, `note_matrix`, `unit_rows` and
`similarity` from tools/scan_dataset.py (the published pdf2musicxml/transcribe.py does not carry them)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bar_dataset import linearize, piano_part, read  # noqa: E402

GAP = 0.3  # the price of a bar read with no measure, or a measure with no bar, in aligning them
PITCH = re.compile(r"^[A-G]\d$")
LENGTHS = ["breve", "whole", "half", "quarter", "eighth", "16th", "32nd", "64th", "128th"]
INDEX = {t: k for k, t in enumerate([f"{s}{o}" for s in "ABCDEFG" for o in range(9)] + LENGTHS)}


def notes(tokens: list[str], index: dict[str, int]) -> np.ndarray:
    """A bar as the notes it holds: a count for each pitch as written, and half that for each length."""
    v = np.zeros(len(index), np.float32)
    for t in tokens:
        if PITCH.match(t) and t in index:
            v[index[t]] += 1
        elif t in LENGTHS:
            v[index[t]] += 0.5
    return v


def note_matrix(rows: list[list[str]]) -> np.ndarray:
    """The notes() of each bar (a row of tokens): one row a bar."""
    return np.stack([notes(r, INDEX) for r in rows]) if rows else np.zeros((0, len(INDEX)), np.float32)


def unit_rows(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def similarity(read: list[list[str]], truth: list[str | None]) -> tuple[np.ndarray, np.ndarray]:
    """How alike each bar read and each measure of the score are (cosine of their notes), and which
    measures LMX could write at all."""
    a, b = unit_rows(note_matrix(read)), unit_rows(note_matrix([m.split() if m else [] for m in truth]))
    return a @ b.T, np.array([m is not None for m in truth])


def align(read: list[list[str]], truth: list[str | None]) -> list[tuple[int | None, int | None]]:
    """Bars read and true measures paired up, as (bar, measure) with None for a side that has none: the best
    global alignment (Needleman-Wunsch) of bars to measures by the notes they hold (`similarity`),
    a gap costing GAP. Always, whatever the counts: equal counts do not make bar k measure k, for a reading
    that misses a barline in one place and invents one in another (Schubert op. 142/1: 250 bars for 250
    measures, most of them paired off by one) has them equal. When every bar agrees with its measure the
    alignment is the one to one pairing anyway, and it costs a second or so at a thousand bars."""
    if not read or not truth:
        return [(k, None) for k in range(len(read))] + [(None, k) for k in range(len(truth))]
    sim, _ = similarity(read, truth)
    n, m = sim.shape
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


def load_truth(path) -> list[tuple[str | None, str]]:
    """The true MusicXML's measures as (LMX, context), or a SystemExit saying why they cannot be had:
    asked for before the long reading, so that a wrong file costs nothing."""
    try:
        part = piano_part(read(Path(path)))
    except Exception as e:  # a missing file, or one that is not MusicXML
        raise SystemExit(f"{path}: cannot be read as MusicXML ({type(e).__name__}: {e})") from e
    if part is None:
        raise SystemExit(f"{path}: not a score for piano (one part, on two staves)")
    return linearize(part)
