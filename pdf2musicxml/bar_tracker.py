"""The bar tracker: where each bar on a page ends, by rules, from the staves the staff counter finds.

The bar counter (github.com/YangJunren11/Bar-counter-for-PDF-Piano-Scores) does this with a network,
because by rules alone barlines are hard to tell from stems, beams and the lines that open systems,
and a system's staves hard to group (its `training/count_bars.py` counts 9 of 47 real scanned pages
exactly). Here the staff counter has already found every staff line and grouped the staves into
systems, and a barline is held to that:

  1. it runs from the top line of a staff to its bottom line, unbroken, in every staff of its system,
     at the same column: a stem is in one staff, and is shorter than one;
  2. at each end of a staff it stops, or carries on unbroken to the next staff (a piano's barlines
     cross the gap between its staves; a song's stop at the voice): a stem that covers a staff runs
     on past it, unbroken, to its notehead or beam, while a barline stops though a bar number or a
     fermata may stand just past it;
  3. it is a thin line all the way down: in every row of each staff, off the staff lines, the ink
     through it is no wider than a barline, as it is not through a stem's notehead or a time
     signature's figures, while a note or accidental beside it, however close, leaves a gap;
  4. lines within a space of each other are one barline: a double bar, a final bar, a repeat;
  5. the line that opens a system ends no bar, and nor does a repeat sign that starts one.

So a bar is counted where it ends, as the bar counter counts it: a bar carried over to the next
system, or the next page, is counted where its barline is.

  python -m pdf2musicxml.bar_tracker score.pdf [--pages 1-3] [--draw out/] [--json out.json]
"""

from __future__ import annotations

import json

import numpy as np
from scipy import ndimage

from .staff_counter import INK, Layout, Level, Staff, look

COVER = 0.85  # the least share of a staff's height a barline covers, or of a gap it crosses
ON = 0.5  # spaces past a staff's end line that a line runs on unbroken, to be no barline that stops there
THIN = 0.7  # spaces: the widest a barline's ink is across, row by row
STRAIGHT = 0.85  # the least share of a staff's rows, off its lines, where a barline is that thin
GROUP = 1.2  # spaces: lines this close are one barline
OPENING = 1.5  # spaces: no barline this near the start of a system's staff lines, where its opening line is
START = 12.0  # spaces: a repeat sign this near the start of a system starts its first bar


def track(page: np.ndarray, space: float | None = None) -> Layout:
    """The systems on a page (`staff_counter.count`), each with where its bars end (`System.bars`,
    the x of each bar's barline halfway down its staff lines)."""
    level = look(page, space)
    find_bars(level)
    return level.layout()


def lines_down(level: Level, dark: float = INK) -> list[np.ndarray]:
    """For each system, the columns of the turned page where a line darker than `dark` runs down
    every staff of the system, from its top line to its bottom line, and stops there or carries on
    to the next staff: the tests a barline passes surely (1 and 2 above), where the others can turn
    real ones away."""
    if not level.systems:
        return []
    d = level.space
    ink = level.dark >= dark
    h, w = ink.shape
    # A barline may lean a pixel either way over a staff, on a page not quite level
    wide = ndimage.maximum_filter1d(ink, 3, axis=1)
    rows = np.concatenate([np.zeros((1, w), np.int32), np.cumsum(wide, axis=0, dtype=np.int32)])
    cols = np.arange(w)

    def share(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
        """For every column, the share of rows top to bottom (bottom left out) that are ink."""
        top, bottom = np.clip(top, 0, h), np.clip(bottom, 0, h)
        return (rows[bottom, cols] - rows[top, cols]) / np.maximum(bottom - top, 1)

    staves = [s for system in level.systems for s in system.staves]
    staves.sort(key=lambda s: s.top(s.middle))
    tops = [np.round(np.interp(cols, s.xs, s.tops)).astype(int) for s in staves]
    bottoms = [np.round(np.interp(cols, s.xs, s.tops) + 4 * s.space).astype(int) for s in staves]
    on = max(3, int(round(ON * d)))
    # Each staff, column by column: covered from its top line to its bottom line, and ending at
    # each of those, or carried on to the staff above or below. A line ends at a staff line if it
    # does not run on unbroken past it; the line itself, and a barline drawn a pixel long, are the
    # first row or two
    covered, ends = {}, {}
    for k, staff in enumerate(staves):
        covered[id(staff)] = share(tops[k], bottoms[k] + 1) >= COVER
        stops_above = share(tops[k] - on, tops[k] - 1) < 1
        stops_below = share(bottoms[k] + 2, bottoms[k] + on + 1) < 1
        if k > 0:
            stops_above |= share(bottoms[k - 1] + 1, tops[k]) >= COVER
        if k + 1 < len(staves):
            stops_below |= share(bottoms[k] + 1, tops[k + 1]) >= COVER
        ends[id(staff)] = stops_above & stops_below
    found = []
    for system in level.systems:
        line = np.ones(w, bool)
        for staff in system.staves:
            line &= covered[id(staff)] & ends[id(staff)]
        found.append(line)
    return found


def find_bars(level: Level) -> None:
    """Each system's barlines, as `System.bars`, on the turned page."""
    if not level.systems:
        return
    d = level.space
    ink = level.dark >= INK
    cols = np.arange(ink.shape[1])
    for system, line in zip(level.systems, lines_down(level)):
        x0, x1 = system.x0, system.x1
        line = line & (cols > x0 + OPENING * d) & (cols <= x1 + 0.5 * d)
        xs = np.flatnonzero(line)
        # Runs of columns, then the runs within a space or so of each other as one barline
        runs = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1) if len(xs) else []
        runs = [r for r in runs if r[-1] - r[0] <= 0.8 * d]  # wider: a block of ink, not a line
        groups: list[list[np.ndarray]] = []
        for r in runs:
            if groups and r[0] - groups[-1][-1][-1] <= GROUP * d:
                groups[-1].append(r)
            else:
                groups.append([r])
        bars = []
        for g in groups:
            left, right = int(g[0][0]), int(g[-1][-1])
            if not all(thin(ink, staff, left, right, d) for staff in system.staves):
                continue
            bars.append((left, right))
        # A repeat sign at the head of a system, after its clef and key, starts the first bar
        if bars and bars[0][0] < x0 + START * d and repeat_start(ink, system.staves[0], bars[0][1], d):
            bars = bars[1:]
        system.bars = [(left + right) / 2 for left, right in bars]


def thin(ink: np.ndarray, staff: Staff, left: int, right: int, d: float) -> bool:
    """A thin line down a staff somewhere in columns left to right: in most of the staff's rows,
    off its lines, the run of ink through that column is no wider than a barline."""
    x = (left + right) / 2
    top = staff.top(x)
    rows = np.arange(int(round(top)), int(round(top + 4 * staff.space)) + 1)
    # The middle of each space: a scan's staff lines can be a third of a space thick, blurred
    off = np.ones(len(rows), bool)
    for k in range(5):
        off &= np.abs(rows - (top + k * staff.space)) > max(1.5, 0.3 * d)
    rows = rows[off]
    widest = max(2, int(round(THIN * d)))
    reach = widest + 1
    lo, hi = max(left - reach, 0), min(right + reach, ink.shape[1] - 1)
    patch = ink[rows][:, lo:hi + 1]
    for c in range(left - lo, right - lo + 1):
        on = patch[:, c]
        # Each row's run of ink through column c: as far left and right as it goes unbroken
        run_left = np.array([c - (np.flatnonzero(~row[:c + 1])[-1] if (~row[:c + 1]).any() else -1) - 1 for row in patch])
        run_right = np.array([(np.flatnonzero(~row[c:])[0] if (~row[c:]).any() else len(row) - c) - 1 for row in patch])
        width = np.where(on, run_left + run_right + 1, 0)
        if np.mean(on & (width <= widest)) >= STRAIGHT:
            return True
    return False


def repeat_start(ink: np.ndarray, staff: Staff, x: int, d: float) -> bool:
    """Dots in the two middle spaces of a staff just right of x, as a repeat sign has."""
    top, space = staff.top(x), staff.space
    lo, hi = x + int(0.3 * d), x + int(1.6 * d)
    dots = 0
    for upper, lower in ((1, 2), (2, 3)):
        y0, y1 = int(top + upper * space + 0.25 * space), int(top + lower * space - 0.25 * space)
        region = ink[y0:y1 + 1, lo:hi + 1]
        dots += bool(region.size and region.mean() > 0.12)
    return dots == 2


def draw(page: np.ndarray, layout: Layout):
    """The staff counter's drawing (`staff_counter.draw`), with each bar's end marked down its
    system."""
    from PIL import ImageDraw

    from .staff_counter import draw as draw_systems

    im = draw_systems(page, layout)
    pen = ImageDraw.Draw(im)
    for system in layout.systems:
        for k, x in enumerate(system.bars):
            top, bottom = system.lines_top(x), system.lines_bottom(x)
            pen.line([(x, top - 6), (x, bottom + 6)], fill="#e6007e", width=3)
            pen.text((x - 4, top - 18), str(k + 1), fill="#e6007e")
    return im


def main() -> int:
    import argparse
    import itertools
    from pathlib import Path

    from .page import gray, normalise, pdf_pages

    ap = argparse.ArgumentParser(description="Where each bar ends on each page of a PDF or picture.")
    ap.add_argument("source", help="a PDF, or a picture of a page")
    ap.add_argument("--pages", default="", help="for a PDF: pages to look at, e.g. 1-3,7")
    ap.add_argument("--draw", default="", help="a folder to draw what was found into")
    ap.add_argument("--json", default="", help="a file to write what was found to")
    args = ap.parse_args()
    source = Path(args.source)
    if source.suffix.lower() == ".pdf":
        numbers = sorted({n for part in args.pages.split(",") if part
                          for n in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)}) or None
        pages = zip(numbers or itertools.count(1), pdf_pages(source, numbers))
    else:
        pages = [(1, normalise(gray(source)))]
    out = Path(args.draw) if args.draw else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    results, start = [], 0
    for number, page in pages:
        layout = track(page.pixels)
        bars = [len(s.bars) for s in layout.systems]
        print(f"page {number}: {sum(bars)} bars, starts with bar {start}"
              f" ({' + '.join(map(str, bars)) or 'no systems'})")
        start += sum(bars)
        results.append({"page": number, **layout.as_dict()})
        if out:
            draw(page.pixels, layout).save(out / f"p{number:03d}.png")
    if args.json:
        Path(args.json).write_text("[\n" + ",\n".join(json.dumps(r) for r in results) + "\n]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
