"""The staff counter on pages drawn here, where every staff, system and mark is where it was put."""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from transcriber.page import SPACE, normalise, pdf_pages, staff_space
from transcriber.staff_counter import count

W, H = 1000, 1400
LEFT, RIGHT = 80, 940


def staff(pen: ImageDraw.ImageDraw, top: float, space: float = SPACE, left: int = LEFT, right: int = RIGHT) -> None:
    for k in range(5):
        y = round(top + k * space)
        pen.line([(left, y), (right, y)], fill=0, width=2)


def note(pen: ImageDraw.ImageDraw, x: float, y: float, stem: float = 0) -> None:
    """A filled notehead at (x, y), its stem `stem` pixels up (or down, if negative)."""
    pen.ellipse([x - 6, y - 4, x + 6, y + 4], fill=0)
    if stem:
        pen.line([(x + 5 if stem > 0 else x - 5, y), (x + 5 if stem > 0 else x - 5, y - stem)], fill=0, width=2)


def grand_staff(pen: ImageDraw.ImageDraw, top: float, gap: float = 7 * SPACE, bars: int = 4) -> tuple[float, float]:
    """Two staves joined by a line at the left and by barlines: a piano's system. Its staff lines'
    top and bottom."""
    staff(pen, top)
    lower = top + 4 * SPACE + gap
    staff(pen, lower)
    bottom = lower + 4 * SPACE
    pen.line([(LEFT, top), (LEFT, bottom)], fill=0, width=2)
    for k in range(1, bars + 1):
        x = LEFT + k * (RIGHT - LEFT) / bars
        pen.line([(x, top), (x, bottom)], fill=0, width=2)
    return top, bottom


def page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("L", (W, H), 255)
    return im, ImageDraw.Draw(im)


def test_staff_space_is_the_step_between_lines() -> None:
    im, pen = page()
    for top in (100, 400, 700):
        staff(pen, top, space=12)
    assert staff_space(np.asarray(im)) == pytest.approx(12, abs=0.05)


def test_a_blank_page_has_no_systems() -> None:
    layout = count(np.full((H, W), 255, np.uint8))
    assert layout.systems == [] and layout.space is None


def test_grand_staves_are_systems_of_two() -> None:
    im, pen = page()
    expected = [grand_staff(pen, top) for top in (120, 420, 720, 1020)]
    layout = count(np.asarray(im))
    assert [len(s.staves) for s in layout.systems] == [2, 2, 2, 2]
    for system, (top, bottom) in zip(layout.systems, expected):
        assert system.lines_top(500) == pytest.approx(top, abs=1)
        assert system.lines_bottom(500) == pytest.approx(bottom, abs=1)
        assert system.x0 == pytest.approx(LEFT, abs=3) and system.x1 == pytest.approx(RIGHT, abs=3)


def test_staves_joined_by_nothing_are_systems_of_their_own() -> None:
    im, pen = page()
    for top in (150, 400, 650):
        staff(pen, top)
    assert [len(s.staves) for s in count(np.asarray(im)).systems] == [1, 1, 1]


def test_notes_high_on_ledger_lines_make_no_staff_of_their_own() -> None:
    """Note after note on five ledger lines, all along a staff: a ladder of short lines a space
    apart over it, which is not a staff."""
    im, pen = page()
    top, _ = grand_staff(pen, 300)
    for x in range(140, RIGHT - 20, 3 * int(SPACE)):
        for k in range(1, 6):
            pen.line([(x - 8, top - k * SPACE), (x + 8, top - k * SPACE)], fill=0, width=2)
        note(pen, x, top - 5 * SPACE)
    layout = count(np.asarray(im))
    assert [len(s.staves) for s in layout.systems] == [2]
    assert layout.systems[0].lines_top(500) == pytest.approx(top, abs=1)


def test_a_voice_over_a_piano_is_one_system_joined_at_the_left() -> None:
    """A song: the voice's barlines stop at its staff, but the line at the left joins all three."""
    im, pen = page()
    voice, top = 150, 150 + 4 * SPACE + 9 * SPACE
    staff(pen, voice)
    _, bottom = grand_staff(pen, top)
    pen.line([(LEFT, voice), (LEFT, bottom)], fill=0, width=2)
    layout = count(np.asarray(im))
    assert [len(s.staves) for s in layout.systems] == [3]


def test_a_turned_page_is_found_where_it_lies() -> None:
    im, pen = page()
    expected = [grand_staff(pen, top) for top in (150, 550, 950)]
    note(pen, 500, 150 - 3 * SPACE)  # three spaces over the first system's top line
    angle = 1.0
    turned = im.rotate(-angle, resample=Image.BICUBIC, fillcolor=255)  # clockwise: lines fall to the right
    layout = count(np.asarray(turned))
    assert layout.angle == pytest.approx(angle, abs=0.1)
    assert [len(s.staves) for s in layout.systems] == [2, 2, 2]
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    cx, cy = (W - 1) / 2, (H - 1) / 2
    for system, (top, _) in zip(layout.systems, expected):
        for x in (200, 800):  # the drawn top line at x, turned as the page was
            xt, yt = cx + c * (x - cx) - s * (top - cy), cy + s * (x - cx) + c * (top - cy)
            assert system.lines_top(xt) == pytest.approx(yt, abs=1.5)
    # Its top runs straight along its staff lines, three spaces and a notehead over them
    first = layout.systems[0]
    assert first.above == pytest.approx(3 * SPACE + 4, abs=2)
    assert first.top(200) - first.lines_top(200) == pytest.approx(first.top(800) - first.lines_top(800))


def test_a_system_holds_what_is_printed_above_and_below_it() -> None:
    """Notes on ledger lines, a dynamic under the bass, a word over the treble: the system's top
    and bottom are straight lines along it, above and below all of them, not its staff lines'."""
    im, pen = page()
    top, bottom = grand_staff(pen, 300)
    high = top - 4 * SPACE
    for k in (1, 2, 3, 4):  # ledger lines up to a high note
        pen.line([(395, top - k * SPACE), (425, top - k * SPACE)], fill=0, width=2)
    note(pen, 410, high, stem=-3 * SPACE)
    pen.text((600, top - 3 * SPACE), "dolce", fill=0)
    pen.rectangle([300, bottom + 3 * SPACE, 316, bottom + 4.5 * SPACE], fill=0)  # a dynamic
    grand_staff(pen, 800)
    layout = count(np.asarray(im))
    first, second = layout.systems
    for x in (LEFT, 410, 700, RIGHT):  # the same all along, however little is printed there
        assert first.top(x) == pytest.approx(high - 4, abs=2)
        assert first.bottom(x) == pytest.approx(bottom + 4.5 * SPACE, abs=2)
    # Nothing printed beyond the second's staff lines: its top and bottom are theirs
    assert second.above == pytest.approx(0, abs=1) and second.below == pytest.approx(0, abs=1)


def test_a_low_note_reaches_past_the_middle_of_the_gap_below() -> None:
    """Systems may overlap: a note hanging far below one system is that system's, even past the
    middle of the gap, while the next system keeps its own staff lines."""
    im, pen = page()
    top1, bottom1 = grand_staff(pen, 150)
    top2, bottom2 = grand_staff(pen, bottom1 + 10 * SPACE)
    low = bottom1 + 6 * SPACE  # past the middle of the ten-space gap
    for k in range(1, 7):
        pen.line([(495, bottom1 + k * SPACE), (525, bottom1 + k * SPACE)], fill=0, width=2)
    note(pen, 510, low, stem=6 * SPACE)
    layout = count(np.asarray(im))
    first, second = layout.systems
    assert first.bottom(510) >= low + 3 and first.bottom(RIGHT) >= low + 3
    assert second.top(510) == pytest.approx(top2, abs=2)
    assert layout.system_at(510, (top1 + bottom1) / 2) == 0
    assert layout.system_at(510, (top2 + bottom2) / 2) == 1


def test_a_page_of_any_size_is_scaled_to_nine_pixel_spaces() -> None:
    im = Image.new("L", (1600, 2200), 255)
    pen = ImageDraw.Draw(im)
    for top in (200, 700, 1200):
        staff(pen, top, space=16, left=100, right=1500)
    scaled = normalise(np.asarray(im))
    assert scaled.scale == pytest.approx(SPACE / 16, abs=0.01)
    assert staff_space(scaled.pixels) == pytest.approx(SPACE, abs=0.3)


def test_a_pdf_page_is_drawn_at_nine_pixel_spaces(tmp_path) -> None:
    im, pen = page()
    for top in (120, 420, 720, 1020):
        grand_staff(pen, top)
    pdf = tmp_path / "score.pdf"
    im.save(pdf, resolution=150)
    (drawn,) = list(pdf_pages(pdf))
    assert staff_space(drawn.pixels) == pytest.approx(SPACE, abs=0.3)
    assert [len(s.staves) for s in count(drawn.pixels).systems] == [2, 2, 2, 2]
