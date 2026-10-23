"""A page of a PDF as the rest of the pipeline sees it: grey, 0 black and 255 white, scaled so a staff
space (the gap between two staff lines) is 9 pixels. The bar counter
(github.com/YangJunren11/Bar-counter-for-PDF-Piano-Scores) was trained on pages at this size, and
the staff counter works on the same pixels, so a bar the one finds and a system the other finds can
be matched by position alone.

The staff space is measured from the page itself, as the bar counter measures it
(`training/omr_ml.py`, `BarCounting.staffSpace`): from the top of one staff line to the top of the
next is the commonest step between black runs down the page's columns. A PDF page is drawn twice,
at 3 pixels a point to measure it and again at the size wanted, which keeps engraved scores sharp.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

SPACE = 9.0  # pixels per staff space, as the pipeline sees a page


def staff_space(page: np.ndarray) -> float | None:
    """The distance between staff lines in pixels, or None for a page with too few lines to tell,
    such as a title page. The same measure as the bar counter's, step for step."""
    ink = page < 160
    cols = ink[:, :: max(1, ink.shape[1] // 300)]
    steps = np.zeros(160, np.int64)
    for c in cols.T:
        edges = np.flatnonzero(np.diff(np.concatenate([[0], c.astype(np.int8), [0]])))
        if len(edges) < 6:
            continue
        starts = edges[::2]  # where each black run starts
        steps += np.bincount(np.minimum(np.diff(starts), 159), minlength=160)[:160]
    steps[:3] = steps[159] = 0
    if steps.sum() < 50:
        return None
    mode = int(np.argmax(steps))
    # To a fraction of a pixel: the steps around the commonest, weighted by how common
    lo, hi = max(mode - 1, 3), min(mode + 2, 159)
    return float(np.average(np.arange(lo, hi), weights=steps[lo:hi] + 1e-9))


def scale_for(space: float | None, width: int) -> float:
    """How much to scale a page `width` pixels wide whose staff space measured `space`. A page
    without staves is given the size of a typical printed page."""
    scale = SPACE / space if space else 1240 / width * SPACE / 10.3
    return float(np.clip(scale, 0.4, 2.5))


@dataclass
class Page:
    pixels: np.ndarray  # grey, 0 black and 255 white, a staff space SPACE pixels
    scale: float  # pixels here per pixel of the picture it was made from (per point, for a PDF)
    measured_space: float | None  # the staff space measured there, in its pixels (points)


def normalise(picture: np.ndarray) -> Page:
    """A page already drawn or scanned, scaled so its staff space is SPACE pixels."""
    space = staff_space(picture)
    scale = scale_for(space, picture.shape[1])
    h, w = round(picture.shape[0] * scale), round(picture.shape[1] * scale)
    resized = Image.fromarray(picture).resize((w, h), Image.BILINEAR if scale > 1 else Image.LANCZOS)
    return Page(np.asarray(resized), scale, space)


def gray(path: Path | str) -> np.ndarray:
    """A picture file as 8-bit grey on white, transparency taken as paper."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        paper = Image.new("RGB", im.size, "white")
        paper.paste(im, mask=im.split()[-1])
        im = paper
    return np.asarray(im.convert("L"))


def pdf_pages(path: Path | str, numbers: list[int] | None = None) -> Iterator[Page]:
    """Each page of a PDF (1-based `numbers`, or all), drawn at 3 pixels a point to measure its
    staff space and again so that space is SPACE pixels. `Page.scale` is then pixels a point."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(path))
    try:
        for number in numbers or range(1, len(document) + 1):
            page = document[number - 1]
            first = np.asarray(page.render(scale=3, grayscale=True).to_pil().convert("L"))
            space = staff_space(first)
            scale = 3 * scale_for(space, first.shape[1])
            pixels = np.asarray(page.render(scale=scale, grayscale=True).to_pil().convert("L"))
            yield Page(pixels, scale, space and space / 3)
    finally:
        document.close()
