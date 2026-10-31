"""The bar cropper on pages drawn here, from peaks put where the bar counter's network would put them."""

from __future__ import annotations

import numpy as np
import pytest

from transcriber.bar_counter import Peak
from transcriber.bar_cropper import MARGIN_X, crop
from transcriber.page import SPACE
from test_staff_counter import LEFT, RIGHT, grand_staff, page

BARS = [LEFT + k * (RIGHT - LEFT) / 4 for k in range(1, 5)]


def systems(tops=(120, 420, 720)):
    im, pen = page()
    lines = [grand_staff(pen, top) for top in tops]
    return im, pen, lines


def test_bars_run_from_barline_to_barline_numbered_through_the_page() -> None:
    im, _, lines = systems()
    # The network's grid is four pixels: its peaks a few pixels off the barlines
    peaks = [Peak(x + 3, (top + bottom) / 2, 0.9) for top, bottom in lines for x in BARS]
    cropped = crop(np.asarray(im), peaks, first=7)
    assert [b.number for b in cropped.bars] == list(range(7, 19)) and cropped.next == 19
    assert not cropped.stray and not any(b.carried for b in cropped.bars)
    for k, (top, bottom) in enumerate(lines):
        bars = [b for b in cropped.bars if b.system == k]
        assert [b.right for b in bars] == pytest.approx(BARS, abs=1.5)
        assert [b.left for b in bars] == pytest.approx([LEFT] + BARS[:-1], abs=3)
        assert all(b.snapped for b in bars)
        for b in bars:
            assert b.top <= top and b.bottom >= bottom
            assert b.image.shape == b.mask.shape == (b.box[3] - b.box[1], b.box[2] - b.box[0])
            # The bar itself, with margins either side of it
            cols = np.flatnonzero(b.mask.any(axis=0))
            assert cols[0] + b.box[0] == pytest.approx(b.left, abs=1)
            assert cols[-1] + b.box[0] == pytest.approx(b.right, abs=1)
            assert b.left - b.box[0] == pytest.approx(MARGIN_X * SPACE, abs=2)


def test_faint_peaks_and_peaks_in_no_system_are_no_bars() -> None:
    im, _, lines = systems()
    top, bottom = lines[0]
    peaks = [Peak(x, (top + bottom) / 2, 0.9) for x in BARS] + [Peak(BARS[0], (top + bottom) / 2 + 300, 0.2),
                                                               Peak(500, 1300, 0.9)]
    cropped = crop(np.asarray(im), peaks)
    assert len([b for b in cropped.bars if not b.carried]) == 4
    assert [(p.x, p.y) for p in cropped.stray] == [(500, 1300)]


def test_a_peak_with_no_barline_under_it_stays_where_it_is() -> None:
    im, _, lines = systems()
    top, bottom = lines[0]
    cropped = crop(np.asarray(im), [Peak(200, (top + bottom) / 2, 0.9)])
    first = cropped.bars[0]
    assert first.right == pytest.approx(200, abs=1) and not first.snapped


def test_staff_lines_past_the_last_bar_end_are_carried_to_the_next_system() -> None:
    im, _, lines = systems()
    top, bottom = lines[0]
    cropped = crop(np.asarray(im), [Peak(x, (top + bottom) / 2, 0.9) for x in BARS[:2]])
    first = [b for b in cropped.bars if b.system == 0]
    assert [b.carried for b in first] == [False, False, True]
    # Numbered as the bar it starts, which ends further on
    assert first[-1].number == 2 and cropped.next == 2
    assert first[-1].left == pytest.approx(BARS[1], abs=1.5) and first[-1].right == pytest.approx(RIGHT, abs=3)


def test_a_double_bar_is_one_barline_ending_at_its_middle() -> None:
    im, pen, lines = systems(tops=(120,))
    top, bottom = lines[0]
    x = BARS[1]
    pen.line([(x + 5, top), (x + 5, bottom)], fill=0, width=4)
    cropped = crop(np.asarray(im), [Peak(x + 5, (top + bottom) / 2, 0.9), Peak(x - 1, (top + bottom) / 2, 0.6)])
    ends = [b.right for b in cropped.bars if not b.carried]
    assert ends == [pytest.approx(x + 2.5, abs=1.5)]
