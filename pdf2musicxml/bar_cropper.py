"""The bar cropper: every bar on a page cut out for the bar recognizer, from where the bar counter's
network says bars end and the systems the staff counter finds.

  1. each peak of the network (a bar's end, at its barline halfway down its system) is put in the
     system it falls in, and moved onto the barline under it: the network's grid is four pixels,
     and the columns where a line runs down every staff of the system and stops there
     (`bar_tracker.lines_down`) are exact; a double or final bar's lines are one barline, and the
     bar ends at their middle;
  2. a system's bars run from barline to barline, its first from where its staff lines start, so
     it has the clef and key; staff lines going on past a system's last barline hold the start of
     a bar that ends on the next system, or only a courtesy key, time or clef for the next system,
     cut out too and marked as carried (which, the bar recognizer tells);
  3. each bar is cut from the page turned level, so its staff lines are straight across, from the
     system's top to its bottom (straight lines outside everything printed with the system: every
     bar of a system as tall, its staff lines as far down), with margins all round; a mask marks
     the bar itself, as against the margins, where the bars either side and the systems above and
     below can show.

Bars are numbered from 0 through the score as the bar counter numbers them, by where they end: a
bar carried over to the next system has the number of the bar it is the start of.

  python -m pdf2musicxml.bar_cropper score.pdf [--pages 1-3] --out crops/
  python -m pdf2musicxml.bar_cropper page1.png page2.png ... --out crops/
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from .bar_counter import THRESHOLD, Peak
from .bar_tracker import GROUP, lines_down
from .page import SPACE
from .staff_counter import GREY, Level, look, turn

SNAP = 6  # pixels: how far a peak is moved onto the barline it marks, a cell and a half of the network's grid
MARGIN_X = 2.0  # spaces either side of a bar
MARGIN_Y = 1.0  # spaces above its system's top and below its bottom
TAIL = 3.0  # spaces: staff lines going on this far past a system's last barline hold a bar's start


@dataclass
class Bar:
    number: int  # from 0 through the score, by where bars end
    page: int
    system: int  # on its page, from 0
    left: float  # on the turned page: the barline before it, or where its system's staff lines start
    right: float  # its barline, or where its system's staff lines end, for a bar carried on
    top: float  # its system's top and bottom
    bottom: float
    carried: bool  # after its system's last barline: a bar's start, ending on the next system, or a courtesy signature
    snapped: bool  # its barline found under the network's peak
    sure: float  # the network's peak; 0 for a carried start
    box: tuple[int, int, int, int]  # the crop on the turned page: left, top, right, bottom
    image: np.ndarray = field(repr=False)  # the crop, grey, 0 black and 255 white
    mask: np.ndarray = field(repr=False)  # 1 over the bar, 0 over the margins

    def as_dict(self) -> dict:
        r = lambda v: round(float(v), 1)
        return {"number": self.number, "page": self.page, "system": self.system, "left": r(self.left),
                "right": r(self.right), "top": r(self.top), "bottom": r(self.bottom), "carried": self.carried,
                "snapped": self.snapped, "sure": round(self.sure, 3), "box": list(self.box)}


@dataclass
class Cropped:
    """What the bar cropper makes of a page."""

    bars: list[Bar]
    level: Level  # the page turned level, and its systems
    picture: np.ndarray  # the page turned level, grey
    stray: list[Peak]  # peaks over the threshold in no system
    next: int  # the number of the first bar on the next page


def crop(pixels: np.ndarray, peaks: list[Peak], page: int = 1, first: int = 0,
         threshold: float = THRESHOLD) -> Cropped:
    """Every bar on a page (grey, 0 black and 255 white, a staff space 9 pixels), from the network's
    peaks on it; `first` is the number of the page's first bar."""
    level = look(pixels)
    layout = level.layout()
    d = level.space or SPACE
    picture = pixels if not level.turned else (
        255 - np.clip(turn((255 - pixels.astype(np.float32)) / 255, level.turned), 0, 1) * 255).astype(np.uint8)
    # Where the network has found a barline, a pale line will do, on a blurred scan
    downs, pale = lines_down(level), lines_down(level, GREY)
    ends: list[list[tuple[float, float, bool]]] = [[] for _ in level.systems]
    stray = []
    for p in peaks:
        if p.height < threshold:
            continue
        k = layout.system_at(p.x, p.y)
        if k is None:
            stray.append(p)
            continue
        x, _ = level.forth(p.x, p.y)
        x, snapped = snap(downs[k], x, d)
        if not snapped:
            x, snapped = snap(pale[k], x, d)
        ends[k].append((x, p.height, snapped))
    bars = []
    number = first
    for k, system in enumerate(level.systems):
        # Two peaks on one barline are one bar's end: the surer
        merged: list[tuple[float, float, bool]] = []
        for e in sorted(ends[k]):
            if merged and e[0] - merged[-1][0] < d:
                merged[-1] = max(merged[-1], e, key=lambda m: m[1])
            else:
                merged.append(e)
        start = system.x0
        for x, height, snapped in merged:
            bars.append(cut(picture, level, k, number, page, start, x, False, snapped, height))
            number += 1
            start = x
        if system.x1 - start > TAIL * d:
            bars.append(cut(picture, level, k, number, page, start, system.x1, True, False, 0.0))
    return Cropped(bars, level, picture, stray, number)


def snap(line: np.ndarray, x: float, d: float) -> tuple[float, bool]:
    """The middle of the barline nearest x, within SNAP pixels, from the lines down the system: lines
    within GROUP spaces of each other are one, a double or final bar; x itself if none."""
    reach = SNAP + int(math.ceil(GROUP * d))
    lo = max(0, int(math.floor(x - reach)))
    cols = np.flatnonzero(line[lo:int(math.ceil(x + reach)) + 1]) + lo
    groups = np.split(cols, np.flatnonzero(np.diff(cols) > GROUP * d) + 1) if len(cols) else []
    near = [g for g in groups if g[0] - SNAP <= x <= g[-1] + SNAP]
    if not near:
        return x, False
    middle = min(((g[0] + g[-1]) / 2 for g in near), key=lambda m: abs(m - x))
    return float(middle), True


def cut(picture: np.ndarray, level: Level, k: int, number: int, page: int, left: float, right: float,
        carried: bool, snapped: bool, sure: float) -> Bar:
    system = level.systems[k]
    d = level.space or SPACE
    h, w = picture.shape
    top = min(system.top(left), system.top(right))
    bottom = max(system.bottom(left), system.bottom(right))
    box = (max(int(math.floor(left - MARGIN_X * d)), 0), max(int(math.floor(top - MARGIN_Y * d)), 0),
           min(int(math.ceil(right + MARGIN_X * d)), w), min(int(math.ceil(bottom + MARGIN_Y * d)), h))
    image = picture[box[1]:box[3], box[0]:box[2]]
    mask = np.zeros(image.shape, np.uint8)
    mask[max(int(round(top)) - box[1], 0):int(round(bottom)) - box[1] + 1,
         max(int(round(left)) - box[0], 0):int(round(right)) - box[0] + 1] = 1
    return Bar(number, page, k, left, right, top, bottom, carried, snapped, sure, box, image, mask)


# ---------------------------------------------------------------------------------------------
# Drawing what was cut


COLOURS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def shown(bar: Bar):
    """A bar's crop as the recognizer gets it, the margins tinted so the bar stands out."""
    from PIL import Image

    grey = np.repeat(bar.image[:, :, None], 3, axis=2).astype(np.float32)
    tint = np.array([200, 220, 255], np.float32)
    out = np.where(bar.mask[:, :, None] == 1, grey, grey * 0.55 + tint * 0.45)
    return Image.fromarray(out.astype(np.uint8))


def sheet(cropped: Cropped, width: int = 1800):
    """Every bar of a page, system by system, each crop as the recognizer gets it."""
    from PIL import Image, ImageDraw

    rows: list[list[Bar]] = []
    for bar in cropped.bars:
        if not rows or rows[-1][0].system != bar.system:
            rows.append([])
        rows[-1].append(bar)
    gap, label = 10, 16
    lines = []  # rows of the sheet: (system, crops), wrapped to the width
    for row in rows:
        line, used = [], 0
        for bar in row:
            im = shown(bar)
            if line and used + im.width + gap > width:
                lines.append((row[0].system, line))
                line, used = [], 0
            line.append((bar, im))
            used += im.width + gap
        lines.append((row[0].system, line))
    height = sum(max(im.height for _, im in line) + label + gap for _, line in lines) + gap
    out = Image.new("RGB", (width, max(height, 1)), "white")
    pen = ImageDraw.Draw(out)
    y = gap
    for system, line in lines:
        x = gap
        colour = COLOURS[system % len(COLOURS)]
        for bar, im in line:
            name = f"bar {bar.number}" + (" (after the last barline: its start, or a courtesy signature)" if bar.carried else "")
            name += "" if bar.snapped or bar.carried else " (no barline found under the network's peak)"
            pen.text((x, y), name, fill=colour)
            out.paste(im, (x, y + label))
            pen.rectangle([x - 1, y + label - 1, x + im.width, y + label + im.height], outline=colour)
            x += im.width + gap
        y += max(im.height for _, im in line) + label + gap
    return out


def overview(cropped: Cropped):
    """The page turned level, each bar's box drawn on it, and peaks in no system ringed."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(cropped.picture).convert("RGB")
    pen = ImageDraw.Draw(im)
    for bar in cropped.bars:
        colour = COLOURS[bar.number % len(COLOURS)]
        pen.rectangle([bar.left, bar.top, bar.right, bar.bottom], outline=colour, width=2)
        pen.text((bar.left + 3, bar.top + 2), str(bar.number) + (" carried" if bar.carried else ""), fill=colour)
    for p in cropped.stray:
        x, y = cropped.level.forth(p.x, p.y)
        pen.ellipse([x - 8, y - 8, x + 8, y + 8], outline="#e6007e", width=3)
    return im


def main() -> int:
    import argparse
    import itertools
    from pathlib import Path

    from PIL import Image

    from .api import DEFAULT_BAR_COUNTER, resolve
    from .bar_counter import BarCounter
    from .page import gray, normalise, pdf_pages

    ap = argparse.ArgumentParser(description="Every bar of a PDF or of pictures of its pages cut out, with its mask.")
    ap.add_argument("source", nargs="+", help="a PDF, or pictures of pages in order")
    ap.add_argument("--bar-counter", default=DEFAULT_BAR_COUNTER, help="the bar counter's weights (default: %(default)s)")
    ap.add_argument("--pages", default="", help="for a PDF: pages to cut, e.g. 1-3,7")
    ap.add_argument("--first", type=int, default=0, help="the number of the first bar")
    ap.add_argument("--out", required=True, help="a folder for the crops, their masks and drawings")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    sources = [Path(s) for s in args.source]
    if sources[0].suffix.lower() == ".pdf":
        numbers = sorted({n for part in args.pages.split(",") if part
                          for n in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)}) or None
        pages = zip(numbers or itertools.count(1), pdf_pages(sources[0], numbers))
    else:
        pages = ((k + 1, normalise(gray(source))) for k, source in enumerate(sources))
    counter = BarCounter(resolve(args.bar_counter, "bar counter"), args.threads or None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index, first = [], args.first
    for number, page in pages:
        cropped = crop(page.pixels, counter.peaks(page.pixels), number, first)
        folder = out / f"p{number:03d}"
        folder.mkdir(exist_ok=True)
        for bar in cropped.bars:
            name = f"bar{bar.number:04d}" + ("-start" if bar.carried else "")
            Image.fromarray(bar.image).save(folder / f"{name}.png")
            Image.fromarray(bar.mask * 255).save(folder / f"{name}-mask.png")
            index.append({**bar.as_dict(), "image": f"{folder.name}/{name}.png", "mask": f"{folder.name}/{name}-mask.png"})
        sheet(cropped).save(out / f"p{number:03d}-bars.png")
        overview(cropped).save(out / f"p{number:03d}-page.png")
        ends = [b for b in cropped.bars if not b.carried]
        print(f"page {number}: {len(ends)} bars, {first} to {cropped.next - 1}"
              f"{f'; {len(cropped.bars) - len(ends)} carried on' if len(ends) < len(cropped.bars) else ''}"
              f"{f'; {len(cropped.stray)} peaks in no system' if cropped.stray else ''}"
              f"{f'; {sum(not b.snapped for b in ends)} with no barline found under the peak' if any(not b.snapped for b in ends) else ''}")
        first = cropped.next
    (out / "bars.json").write_text("[\n" + ",\n".join(json.dumps(b) for b in index) + "\n]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
