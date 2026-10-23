"""The staff counter: the staves on a page, the systems they make, and each system's top and bottom.

A system is the staves read together, joined at their left end by a line (a piano's two staves by a
line and a brace), and everything printed with them: notes on ledger lines far above or below,
their stems and beams, slurs, dynamics, fingerings, pedal marks, words. Its top and bottom are two
straight lines along the system, parallel to its staff lines: as far above its first staff's top
line and below its last staff's bottom line as anything printed with it reaches, anywhere along it.
So every bar of a system is cropped alike, its staff lines as far from the crop's top and bottom,
whatever is in it. They may reach past the staff lines of the system above or below it, as a low
bass and the next system's high treble can share the gap between them. Its staff lines' top and
bottom, where its barlines run, are kept too.

The bar counter marks each bar at its barline halfway down its staff lines; the bar cropper takes
the system that mark falls in, and crops the bar between the barline before it and its own, from
the system's top to its bottom.

On a page scaled so a staff space is about 9 pixels (`pdf2musicxml.page`), though the space is
measured again here and any size will do:

  1. thin strokes: rows of ink twice as dark as the paper a little above and below them, in
     horizontal runs at least a space long. Staff lines are kept, however faint; stems, noteheads,
     beams and most text are not.
  2. level: the angle at which those strokes stack into the sharpest rows is the one at which the
     staff lines are level, and the page is turned by it.
  3. staves, strip by strip: the page is cut into strips eight spaces wide, and in each, a staff is
     five rows of thin strokes evenly spaced, the page's staff space apart (measured from those
     strokes), with nothing as strong a space above or below (a staff is five lines, not part of a
     ruling, nor of a ladder of ledger lines). A staff is then followed from strip to strip, so a
     scanned page that is not quite flat bends no staff out of reach, and put on the five rows
     least often bare paper along it: a staff line is hidden under beams and notes, but never
     missing, as ledger lines are between notes.
  4. where each staff starts and ends: followed on through strips too thick with notes to find it
     in, then out to the last columns where three of its five lines are.
  5. systems: two staves one above the other are one system if a line joins them at their left
     end, or if barlines run from the top of the one to the bottom of the other.
  6. what each system holds: the ink with the staff lines taken out falls into pieces (a note with
     its ledger lines, stem and beam; a slur; a word). A piece crossing a system's staff lines is
     that system's; one clear of every system's is the system's whose pieces are nearest, within
     six spaces; and one within a space and a half of a piece already taken goes with it, as a
     word goes letter by letter and an 8va line dash by dash, as far as twelve spaces from the
     staff lines. A system's top and bottom are as far out as its furthest piece.

Everything is given back in the pixels of the page as it was given, not as it was turned.

  python -m pdf2musicxml.staff_counter score.pdf [--pages 1-3] [--draw out/] [--json out.json]
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .page import SPACE, staff_space

INK = 0.3  # darker than this (0 paper, 1 ink) is ink
GREY = 0.15  # darker than this is not paper
RIDGE = 0.08  # how much darker than the paper above and below it a staff line is, at the least
STRIP = 8.0  # strip width, in spaces
LINE = 0.25  # the least share of a strip's columns in which every one of a staff's five lines is found
FOLLOW = 0.15  # and in which the middle of them is, where a staff found already is followed on
JOIN = 0.9  # the least share of a gap a line must cross to join two staves
REACH = 6.0  # spaces: how far from a system's staff lines a piece clear of every system's is its
CHAIN = 1.5  # spaces: how near a piece already taken another must be to go with it
FAR = 12.0  # spaces: and how far from the staff lines that can carry on
BOTH = 2.0  # a piece between two systems is both's when one is no more than this many times as far, and a space


@dataclass
class Staff:
    """Five lines. `xs` and `tops` follow the top line along the staff, so a staff on a page that
    is turned or bent is followed as it is; between them it is taken to be straight."""

    x0: float  # where its lines start
    x1: float  # and end
    space: float  # from one line to the next
    xs: list[float]
    tops: list[float]

    def top(self, x: float) -> float:
        return float(np.interp(x, self.xs, self.tops))

    def bottom(self, x: float) -> float:
        return self.top(x) + 4 * self.space

    def line(self, k: int, x: float) -> float:
        """The height of line k at x, 0 the top line and 4 the bottom."""
        return self.top(x) + k * self.space

    @property
    def middle(self) -> float:
        return (self.x0 + self.x1) / 2

    def as_dict(self) -> dict:
        r = lambda v: round(float(v), 2)
        return {"x0": r(self.x0), "x1": r(self.x1), "space": r(self.space),
                "top": [[r(x), r(y)] for x, y in zip(self.xs, self.tops)]}


@dataclass
class System:
    """Staves read together, and everything printed with them: as far as `above` pixels above its
    first staff's top line, and `below` below its last staff's bottom line."""

    staves: list[Staff]
    above: float = 0.0
    below: float = 0.0
    bars: list[float] = field(default_factory=list)  # where each bar ends, from the bar tracker

    @property
    def x0(self) -> float:
        """Where its staff lines start."""
        return min(s.x0 for s in self.staves)

    @property
    def x1(self) -> float:
        """And end."""
        return max(s.x1 for s in self.staves)

    def lines_top(self, x: float) -> float:
        """Its first staff's top line at x: where its barlines start."""
        return self.staves[0].top(x)

    def lines_bottom(self, x: float) -> float:
        """Its last staff's bottom line at x: where its barlines end."""
        return self.staves[-1].bottom(x)

    def top(self, x: float) -> float:
        """Its top at x: a straight line along it, parallel to its staff lines, above everything
        printed with it."""
        return self.lines_top(x) - self.above

    def bottom(self, x: float) -> float:
        """Its bottom at x, below everything printed with it."""
        return self.lines_bottom(x) + self.below

    def as_dict(self) -> dict:
        r = lambda v: round(float(v), 1)
        return {"x0": r(self.x0), "x1": r(self.x1), "above": r(self.above), "below": r(self.below),
                "top": [[r(x), r(self.top(x))] for x in (self.x0, self.x1)],
                "bottom": [[r(x), r(self.bottom(x))] for x in (self.x0, self.x1)],
                "staves": [s.as_dict() for s in self.staves], **({"bars": [r(x) for x in self.bars]} if self.bars else {})}


@dataclass
class Layout:
    """What the staff counter finds on a page."""

    width: int
    height: int
    space: float | None  # the page's staff space, in its pixels; None where there are no staves
    angle: float  # degrees the staff lines fall from left to right (rise, if negative)
    systems: list[System] = field(default_factory=list)

    @property
    def staves(self) -> list[Staff]:
        return [s for system in self.systems for s in system.staves]

    def system_at(self, x: float, y: float) -> int | None:
        """The system a point is in: the one whose staff lines' middle is nearest, among those whose
        staff lines at x it falls between with a space to spare. A bar counter's peak, halfway down
        its system's staff lines, finds its system so."""
        best, distance = None, math.inf
        for k, system in enumerate(self.systems):
            top, bottom = system.lines_top(x), system.lines_bottom(x)
            margin = self.space or SPACE
            if top - margin <= y <= bottom + margin and abs(y - (top + bottom) / 2) < distance:
                best, distance = k, abs(y - (top + bottom) / 2)
        return best

    def as_dict(self) -> dict:
        return {"width": self.width, "height": self.height,
                "space": None if self.space is None else round(self.space, 3), "angle": round(self.angle, 3),
                "systems": [s.as_dict() for s in self.systems]}


# ---------------------------------------------------------------------------------------------
# Finding them


def count(page: np.ndarray, space: float | None = None) -> Layout:
    """The staves and systems on a page: grey, 0 black and 255 white."""
    return look(page, space).layout()


@dataclass
class Level:
    """A page as the staff counter works on it: its darkness turned so its staff lines are level,
    and the systems found there. The bar tracker works on it too."""

    width: int  # of the page as given
    height: int
    dark: np.ndarray  # 0 paper to 1 ink, turned
    space: float | None
    angle: float  # radians the staff lines fall to the right, on the page as given
    turned: float  # radians the page was turned by: none, where that is too little to move a row
    systems: list[System]  # on the turned page

    def back(self, x: float, y: float) -> tuple[float, float]:
        """A point of the turned page, where it is on the page as given."""
        return unturned(x, y, self.turned, self.width, self.height)

    def forth(self, x: float, y: float) -> tuple[float, float]:
        """A point of the page as given, where it is on the turned page."""
        return unturned(x, y, -self.turned, self.width, self.height)

    def layout(self) -> Layout:
        """What was found, on the page as given."""
        w, h = self.width, self.height
        systems = []
        for system in self.systems:
            middle = (system.lines_top(system.x0) + system.lines_bottom(system.x0)) / 2
            systems.append(System([unturn(s, self.turned, w, h) for s in system.staves], system.above, system.below,
                                  [self.back(x, middle)[0] for x in system.bars]))
        space = float(np.median([s.space for x in systems for s in x.staves])) if systems else self.space
        return Layout(w, h, space, math.degrees(self.angle), systems)


def look(page: np.ndarray, space: float | None = None) -> Level:
    """The staves and systems on a page, on the page turned so its staff lines are level."""
    h, w = page.shape
    dark = darkness(page)
    # The staff space, from the staff lines' own strokes: measured from black runs, as a page is
    # scaled, it can miss lines a scaling has left pale grey
    guess = space or staff_space(page) or SPACE
    d = space or spacing(thin_strokes(dark, guess))
    if d is None:
        return Level(w, h, dark, None, 0.0, 0.0, [])
    angle = skew(thin_strokes(dark, d))
    turned = angle if abs(angle) >= math.radians(0.02) else 0.0  # less, and no row moves a pixel
    if turned:
        dark = turn(dark, turned)
    lines = thin_strokes(dark, d)
    staves = sorted(find_staves(lines, dark >= GREY, d), key=lambda s: s.top(s.middle))
    systems = group(dark >= GREY, staves, d)  # a pale line joins staves as well as a black one
    gather(dark >= INK, lines, systems, d)
    return Level(w, h, dark, d, angle, turned, systems)


VERTICAL = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], bool)


def darkness(page: np.ndarray) -> np.ndarray:
    """0 for the paper and 1 for the ink, whatever shade either is: the paper is the commonest
    grey, and the ink the darkest hundredth of the rest."""
    counts = np.bincount(page.ravel(), minlength=256)
    paper = int(np.argmax(counts))
    darker = np.cumsum(counts[:paper])
    black = int(np.searchsorted(darker, 0.01 * darker[-1])) if paper > 0 and darker[-1] else 0
    if paper - black < 64:  # too little ink to tell its shade
        black = max(paper - 128, 0)
    return np.clip((paper - page.astype(np.float32)) / max(paper - black, 1), 0, 1)


def thin_strokes(dark: np.ndarray, d: float) -> np.ndarray:
    """Rows of ink twice as dark as the paper a little above and below them, in horizontal runs at
    least a space long: staff lines, and not stems, noteheads, beams or most text. A line drawn
    small, or a scan at a low resolution scaled up, is grey and blurred over two rows, and a light
    scan's lines can be far fainter than its notes, which this still finds. How far above and
    below is the staff lines' weight: the commonest height of a vertical run of anything greyer
    than the paper, as every column crosses five lines a staff."""
    runs, _ = ndimage.label(dark >= GREY, structure=VERTICAL)
    size = np.bincount(runs.ravel())
    heights = np.bincount(size[1:], minlength=int(d))[: max(2, int(d / 2) + 1)]
    heights[0] = 0
    weight = int(np.argmax(heights)) if heights.sum() else 1
    k = max(2, (weight + 1) // 2 + 1)
    above = np.pad(dark, ((k, 0), (0, 0)))[:-k]
    below = np.pad(dark, ((0, k), (0, 0)))[k:]
    ridge = dark - np.maximum(above, below)
    thin = (ridge >= RIDGE) & (ridge >= 0.5 * dark)  # however faint the line, twice as dark as beside it
    # A scanned line is broken here and there; a gap of a pixel or two is closed
    thin = ndimage.binary_closing(thin, np.ones((1, 3), bool))
    return ndimage.binary_opening(thin, np.ones((1, max(3, int(round(d)))), bool))


def spacing(lines: np.ndarray) -> float | None:
    """The staff space from thin strokes: the commonest step from the top of one to the top of the
    next, down the page's columns, to a fraction of a pixel. None where there are too few."""
    cols = lines[:, :: max(1, lines.shape[1] // 300)]
    steps = np.zeros(160, np.int64)
    for c in cols.T:
        tops = np.flatnonzero(np.diff(np.concatenate([[0], c.astype(np.int8)])) == 1)
        if len(tops) >= 3:
            steps += np.bincount(np.minimum(np.diff(tops), 159), minlength=160)[:160]
    steps[:3] = steps[159] = 0
    if steps.sum() < 50:
        return None
    mode = int(np.argmax(steps))
    lo, hi = max(mode - 1, 3), min(mode + 2, 159)
    return float(np.average(np.arange(lo, hi), weights=steps[lo:hi] + 1e-9))


def skew(lines: np.ndarray) -> float:
    """The angle, in radians, at which the thin strokes stack into the sharpest rows: the slope of
    the staff lines, positive where they fall to the right."""
    ys, xs = np.nonzero(lines)
    if len(ys) < 100:
        return 0.0
    if len(ys) > 400_000:
        pick = np.random.default_rng(0).choice(len(ys), 400_000, replace=False)
        ys, xs = ys[pick], xs[pick]
    xs = xs - lines.shape[1] / 2

    def sharpness(angle: float) -> float:
        rows = np.round(ys - xs * math.tan(angle)).astype(np.int64)
        return float(np.sum(np.bincount(rows - rows.min()).astype(np.float64) ** 2))

    def sharpest(angles: np.ndarray) -> float:
        """The middle of the angles found sharpest: a level page is as sharp a twentieth of a
        degree either way, where no row moves by half a pixel."""
        values = np.array([sharpness(a) for a in angles])
        return float(np.median(angles[values >= values.max() * (1 - 1e-9)]))

    best = sharpest(np.radians(np.round(np.arange(-4.0, 4.001, 0.1), 2)))
    return sharpest(best + np.radians(np.round(np.arange(-0.1, 0.1001, 0.01), 2)))


def turn(dark: np.ndarray, angle: float) -> np.ndarray:
    """A page's darkness turned so lines falling at `angle` are level, about its centre."""
    c, s = math.cos(angle), math.sin(angle)
    centre = np.array([(dark.shape[0] - 1) / 2, (dark.shape[1] - 1) / 2])
    # For each pixel of the turned page, (row, column), where it is on the page as given
    matrix = np.array([[c, s], [-s, c]])
    return ndimage.affine_transform(dark, matrix, centre - matrix @ centre, order=1, cval=0.0)


def unturned(x: float, y: float, angle: float, w: int, h: int) -> tuple[float, float]:
    """A point of the turned page, where it is on the page as given."""
    c, s = math.cos(angle), math.sin(angle)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    return cx + c * (x - cx) - s * (y - cy), cy + s * (x - cx) + c * (y - cy)


def unturn(staff: Staff, angle: float, w: int, h: int) -> Staff:
    """A staff found on the turned page, where it is on the page as given."""
    if angle == 0:
        return staff
    points = [unturned(x, y, angle, w, h) for x, y in zip(staff.xs, staff.tops)]
    middle = 2 * staff.space  # the ends are where the middle line starts and ends
    x0 = unturned(staff.x0, staff.top(staff.x0) + middle, angle, w, h)[0]
    x1 = unturned(staff.x1, staff.top(staff.x1) + middle, angle, w, h)[0]
    return Staff(x0, x1, staff.space, [p[0] for p in points], [p[1] for p in points])


@dataclass
class Found:
    strip: int
    y: int  # the top line's row, to a pixel
    space: float
    score: float


def find_staves(lines: np.ndarray, inked: np.ndarray, d: float) -> list[Staff]:
    """The staves on a page from its thin strokes, `lines`, and from where it is not bare paper."""
    h, w = lines.shape
    width = max(16, int(round(STRIP * d)))
    step = width // 2
    starts = list(range(0, max(w - width, 0) + 1, step))
    if starts[-1] + width < w:
        starts.append(w - width)

    def strips(mask: np.ndarray) -> list[np.ndarray]:
        """For each strip, the share of its columns in which each row is in the mask."""
        total = np.concatenate([np.zeros((h, 1), np.int32), np.cumsum(mask, axis=1, dtype=np.int32)], 1)
        return [(total[:, min(a + width, w)] - total[:, a]) / width for a in starts]

    profiles = strips(lines)
    found = [f for j, p in enumerate(profiles) for f in in_strip(p, j, d)]
    covered = strips(ndimage.maximum_filter1d(inked, 3, axis=0))  # a row, or a row beside it, not bare paper
    measured = []
    for chain in settle(follow(found, d), covered, d):
        staff = measure(chain, starts, width, profiles, lines, inked, d)
        if staff is not None:
            measured.append((sum(f.score for f in chain), staff))
    # Should two staves still overlap, the one found more strongly
    staves: list[Staff] = []
    for _, staff in sorted(measured, key=lambda m: -m[0]):
        if not any(overlap(staff, s) for s in staves):
            staves.append(staff)
    return staves


def overlap(a: Staff, b: Staff) -> bool:
    lo, hi = max(a.x0, b.x0), min(a.x1, b.x1)
    if hi <= lo:
        return False
    x = (lo + hi) / 2
    return a.top(x) < b.bottom(x) + 0.5 * b.space and b.top(x) < a.bottom(x) + 0.5 * a.space


def in_strip(profile: np.ndarray, strip: int, d: float) -> list[Found]:
    """Staves in one strip, from its row profile: the share of its columns where each row has a
    thin stroke. Five rows a space apart all well filled, and the rows a space above and below
    less so."""
    h = len(profile)
    near = ndimage.maximum_filter1d(profile, 3)  # a line may sit a pixel off the comb
    # A space past the outermost lines, at the page's spacing whatever the comb's, and two rows
    # either way: a comb spaced a little wrong over a ladder of ledger lines still finds the staff
    # line under them
    around = ndimage.maximum_filter1d(profile, 5)
    best = np.zeros(h)
    best_space = np.zeros(h)
    # The page's own staff space, a little either way, as a scan can stretch it
    for space in np.arange(0.94 * d, 1.06 * d + 0.01, 0.25):
        offsets = np.round(np.arange(5) * space).astype(int)
        beyond = int(round(d))
        lo, hi = beyond, h - offsets[-1] - beyond
        if hi <= lo:
            continue
        rows = np.arange(lo, hi)
        stack = np.stack([near[rows + o] for o in offsets])
        weakest = stack.min(0)
        outside = np.maximum(around[rows - beyond], around[rows + offsets[-1] + beyond])
        score = np.where(outside < 0.8 * weakest, weakest, 0.0)
        better = score > best[rows]
        best[rows[better]] = score[better]
        best_space[rows[better]] = space
    reach = int(round(1.5 * d))
    top = ndimage.maximum_filter1d(best, 2 * reach + 1)
    found, last = [], -10 * reach
    for y in np.flatnonzero((best == top) & (best >= LINE)):
        if y - last > reach:  # where a peak is flat, the first of it
            found.append(Found(strip, int(y), float(best_space[y]), float(best[y])))
            last = y
    return found


def follow(found: list[Found], d: float) -> list[list[Found]]:
    """Each staff's finds in strip after strip: a find continues the staff whose last find, at most
    three strips back, is level with it and as widely spaced."""
    chains: list[list[Found]] = []
    for f in sorted(found, key=lambda f: (f.strip, -f.score)):
        options = [c for c in chains if 1 <= f.strip - c[-1].strip <= 3
                   and abs(c[-1].y - f.y) <= max(2.0, 0.3 * f.space) + (f.strip - c[-1].strip - 1)
                   and abs(c[-1].space - f.space) <= 0.12 * d]
        if options:
            min(options, key=lambda c: abs(c[-1].y - f.y)).append(f)
        else:
            chains.append([f])
    # Lost for longer, as across a passage thick with notes, and found again level with where it was
    chains.sort(key=lambda c: c[0].strip)
    merged: list[list[Found]] = []
    for c in chains:
        before = [m for m in merged if m[-1].strip < c[0].strip and abs(m[-1].y - c[0].y) <= max(2.0, 0.3 * d)]
        if before:
            max(before, key=lambda m: m[-1].strip).extend(c)
        else:
            merged.append(c)
    return [sorted(c, key=lambda f: f.strip) for c in merged]


def settle(chains: list[list[Found]], covered: list[np.ndarray], d: float) -> list[list[Found]]:
    """One chain of finds for each staff. Where ledger lines crowd above or below a staff and a
    beam lies along its far line, a strip can find the staff a line too high or low, and the staff
    is then followed as two or more chains a line apart; where every note has ledger lines and a
    beam lies along the line all the way, it is followed a line out from end to end. A staff line
    is hidden here and there, under a beam or a note, but never missing: wherever it should be
    there is ink of some kind, never bare paper, while between ledger lines there is paper. The
    chains in one band are put on the lines that are least often bare paper across every strip of
    the band, and taken as one staff. `covered` is, for each strip, the share of its columns in
    which each row is not bare paper."""

    near = {}

    def strength(chain: list[Found], strips: list[int]) -> float:
        """The share of the strips' columns in which the barest of the chain's five lines is inked.
        A find's top row can be a pixel or two out, and its spacing a quarter of a pixel, so each
        line is looked for two rows either way, still well inside the space between lines."""
        space = float(np.median([f.space for f in chain]))
        ys = np.interp(strips, [f.strip for f in chain], [f.y for f in chain])
        for j in strips:
            if j not in near:
                near[j] = ndimage.maximum_filter1d(covered[j], 3)
        return min(float(np.mean([near[j][int(np.clip(round(y + k * space), 0, len(near[j]) - 1))]
                                  for j, y in zip(strips, ys)])) for k in range(5))

    def shifted(chain: list[Found], lines: int) -> list[Found]:
        return [Found(f.strip, f.y + int(round(lines * f.space)), f.space, f.score) for f in chain]

    bands: list[list[list[Found]]] = []
    for c in sorted(chains, key=lambda c: -sum(f.score for f in c)):
        y = np.median([f.y for f in c])
        band = next((b for b in bands if any(abs(np.median([f.y for f in k]) - y) < 4.5 * d for k in b)), None)
        if band is None:
            bands.append([c])
        else:
            band.append(c)
    settled = []
    for band in bands:
        strips = sorted({f.strip for c in band for f in c})
        if len(strips) < 2:
            continue
        # Each chain as found, and a line up or down from it: one followed a line out all the way
        # has no other chain to set it right
        best = max((shifted(c, k) for c in band for k in (0, -1, 1)), key=lambda c: strength(c, strips))
        space = float(np.median([f.space for f in best]))
        at = lambda j: float(np.interp(j, [f.strip for f in best], [f.y for f in best]))
        chain: dict[int, Found] = {f.strip: f for f in best}
        for c in band:
            for f in c:
                if f.strip in chain:
                    continue
                target = at(f.strip)
                y = f.y + int(round(round((target - f.y) / space) * space))
                if abs(y - target) <= 0.4 * space:  # not half a space out, where it could be either
                    chain[f.strip] = Found(f.strip, y, f.space, f.score)
        if len(chain) >= 2:
            settled.append([chain[j] for j in sorted(chain)])
    return settled


def measure(chain: list[Found], starts: list[int], width: int, profiles: list[np.ndarray], lines: np.ndarray,
            inked: np.ndarray, d: float) -> Staff | None:
    """A staff's top line to a fraction of a pixel in each strip it was found in, its spacing, and
    where its lines start and end."""
    h, w = lines.shape
    xs, tops, spaces = [], [], []
    for f in chain:
        a = starts[f.strip]
        profile = profiles[f.strip]
        centres = []
        for k in range(5):
            y = f.y + int(round(k * f.space))
            rows = np.arange(max(y - 2, 0), min(y + 3, h))
            weights = profile[rows]
            centres.append(float(np.average(rows, weights=weights)) if weights.sum() > 0 else float(y))
        space, top = np.polyfit(np.arange(5), centres, 1)
        xs.append(a + width / 2)
        tops.append(float(top))
        spaces.append(float(space))
    order = np.argsort(xs)
    xs, tops = np.asarray(xs)[order], np.asarray(tops)[order]
    if len(tops) >= 3:  # one strip's top a pixel out, where a beam lay on a line, is the other's
        tops = np.r_[tops[0], ndimage.median_filter(tops, 3, mode="nearest")[1:-1], tops[-1]]
    space = float(np.median(spaces))
    # It runs at least from its first strip to its last, however thick the notes between. Its ends:
    # from those strips outwards, as far as columns with three of its five lines go on, with no
    # gap wider than a clef
    cols = np.arange(w)
    rows = np.round(np.interp(cols, xs, tops)[None, :] + np.arange(5)[:, None] * space).astype(int)
    present = np.zeros(w, int)
    for k in range(5):
        hit = np.zeros(w, bool)
        for dy in (-1, 0, 1):
            r = np.clip(rows[k] + dy, 0, h - 1)
            hit |= lines[r, cols]
        present += hit
    there = np.flatnonzero(present >= 3)
    if not len(there):
        return None
    # Strips beyond the first and last it was found in, thick with notes, where its lines are now
    # known: as far as the middle of its five lines is still there
    def still(j: int, top: float) -> bool:
        near = ndimage.maximum_filter1d(profiles[j], 3)
        rows = np.clip(np.round(top + np.arange(5) * space).astype(int), 0, h - 1)
        return float(np.median(near[rows])) >= FOLLOW

    reached = [chain[0].strip, chain[-1].strip]
    for end, way in ((0, -1), (1, 1)):
        j, missed = reached[end] + way, 0
        while 0 <= j < len(starts) and missed <= 1:
            if still(j, tops[0] if way < 0 else tops[-1]):
                reached[end], missed = j, 0
            else:
                missed += 1
            j += way
    first, last = starts[reached[0]], starts[reached[1]] + width - 1
    inner = there[(there >= first) & (there <= last)]
    if not len(inner):
        return None
    reach = 3.5 * d
    x0 = inner[0]
    for x in there[there < inner[0]][::-1]:
        if x0 - x > reach:
            break
        x0 = x
    # Between a system's opening line and its clef, its lines can be shorter than a space, too short
    # for a thin stroke: on, column by column, while three of the five rows are still ink, over a
    # break of a pixel in a worn scan's lines (not two: a brace can stand that close)
    def inked_at(x: int) -> int:
        """How many of the five lines are ink at column x, or a row either side."""
        return sum(bool(inked[max(rows[k][x] - 1, 0):rows[k][x] + 2, x].any()) for k in range(5))

    while True:
        on = [x for x in range(x0 - 1, max(x0 - 3, -1), -1) if inked_at(x) >= 3]
        if not on:
            break
        x0 = on[0]
    x1 = inner[-1]
    for x in there[there > inner[-1]]:
        if x - x1 > reach:
            break
        x1 = x
    x0, x1 = float(x0), float(x1)
    if x1 - x0 < 4 * d:
        return None
    # The top line from end to end: the strips' tops within the staff, and its ends carried on
    # from the nearest two
    inside = (xs >= x0) & (xs <= x1)
    if inside.sum() < 1:
        return None
    xs, tops = list(xs[inside]), list(tops[inside])
    slope = np.polyfit(xs, tops, 1)[0] if len(xs) >= 2 else 0.0
    if xs[0] > x0:
        xs.insert(0, x0)
        tops.insert(0, tops[0] - slope * (xs[1] - x0))
    if xs[-1] < x1:
        tops.append(tops[-1] + slope * (x1 - xs[-1]))
        xs.append(x1)
    return Staff(x0, x1, space, [float(x) for x in xs], [float(t) for t in tops])


def group(ink: np.ndarray, staves: list[Staff], d: float) -> list[System]:
    """Staves into systems, top to bottom: a staff joins the one above it if a line joins them at
    their left end, or if barlines run from the top of the one to the bottom of the other."""
    if not staves:
        return []
    h, w = ink.shape
    lean = 2 * max(1, int(round(0.08 * d))) + 1  # a line may lean a pixel or two over a gap
    wide = ndimage.maximum_filter1d(ink, lean, axis=1)
    rows = np.concatenate([np.zeros((1, w), np.int32), np.cumsum(wide, axis=0, dtype=np.int32)])
    # A column inked down most of the page is the edge of a scan or a frame, not a line in the music
    edge = rows[-1] >= 0.8 * h
    systems = [[staves[0]]]
    for upper, lower in zip(staves, staves[1:]):
        if joined(rows, edge, upper, lower, d):
            systems[-1].append(lower)
        else:
            systems.append([lower])
    return [System(s) for s in systems]


def cover(rows: np.ndarray, xs: np.ndarray, top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """For each column x, the share of rows top to bottom (bottom left out) that are ink."""
    h = rows.shape[0] - 1
    top, bottom = np.clip(top, 0, h), np.clip(bottom, 0, h)
    span = np.maximum(bottom - top, 1)
    return (rows[bottom, xs] - rows[top, xs]) / span


def joined(rows: np.ndarray, edge: np.ndarray, upper: Staff, lower: Staff, d: float) -> bool:
    w = rows.shape[1]
    # Side by side rather than one above the other: not one system
    if min(upper.x1, lower.x1) - max(upper.x0, lower.x0) < 0.5 * min(upper.x1 - upper.x0, lower.x1 - lower.x0):
        return False
    tops = lambda s, xs: np.round([s.top(x) for x in xs]).astype(int)
    bottoms = lambda s, xs: np.round([s.bottom(x) for x in xs]).astype(int)
    # The line at the left end, with a brace or bracket further left. Where the staves are found to
    # start can be a clef's width past it: the few pixels of line before a clef are too short to
    # count as line
    xs = np.arange(max(int(min(upper.x0, lower.x0) - 6 * d), 0), min(int(max(upper.x0, lower.x0) + d), w - 1) + 1)
    gap = cover(rows, xs, bottoms(upper, xs) + 2, tops(lower, xs) - 1)
    if np.any((gap >= JOIN) & ~edge[xs]):
        return True
    # Barlines from the top of the one to the bottom of the other
    xs = np.arange(int(max(upper.x0, lower.x0) + 2 * d), int(min(upper.x1, lower.x1)) + 1)
    if not len(xs):
        return False
    through = (cover(rows, xs, tops(upper, xs), bottoms(lower, xs) + 1) >= JOIN) & ~edge[xs]
    groups = np.split(xs[through], np.flatnonzero(np.diff(xs[through]) > 1) + 1) if through.any() else []
    return sum(1 for g in groups if len(g) <= 2 * d) >= 2


def gather(ink: np.ndarray, lines: np.ndarray, systems: list[System], d: float) -> None:
    """How far above and below its staff lines everything printed with each system reaches
    (`System.above` and `below`), on the turned page."""
    if not systems:
        return
    h, w = ink.shape
    cols = np.arange(w)
    # Each system's staff lines' top and bottom in every column, carried on level past their ends
    tops = np.stack([np.interp(cols, s.staves[0].xs, s.staves[0].tops) for s in systems])
    bottoms = np.stack([np.interp(cols, s.staves[-1].xs, s.staves[-1].tops) + 4 * s.staves[-1].space
                        for s in systems])
    # Where a system's pieces can be: instrument names stand to the left of it
    beside = np.stack([(cols >= s.x0 - 8 * d) & (cols <= s.x1 + 3 * d) for s in systems])
    # The staff lines taken out, so what is printed on them falls apart into its own pieces
    on_lines = np.zeros_like(ink)
    for system in systems:
        for staff in system.staves:
            xs = cols[int(max(staff.x0, 0)):int(min(staff.x1, w - 1)) + 1]
            top = np.interp(xs, staff.xs, staff.tops)
            for k in range(5):
                for dy in (-1, 0, 1):
                    on_lines[np.clip(np.round(top + k * staff.space).astype(int) + dy, 0, h - 1), xs] = True
    labels, n = ndimage.label(ink & ~(lines & on_lines), structure=np.ones((3, 3), bool))
    if n == 0:
        return
    sizes = np.bincount(labels.ravel())
    boxes = [(b[0].start, b[0].stop - 1, b[1].start, b[1].stop - 1) if b else None for b in ndimage.find_objects(labels)]
    owner = np.full(n + 1, -1)
    other = np.full(n + 1, -1)  # a second system a piece between two is as much the one's as the other's
    shared: list[int] = []
    clear: list[int] = []
    small: list[int] = []
    for i, box in enumerate(boxes, 1):
        if box is None:
            continue
        r0, r1, c0, c1 = box
        if sizes[i] < max(4.0, 0.08 * d * d):
            small.append(i)  # a speck, or a dash of an 8va line or a crescendo
            continue
        middle = (c0 + c1) // 2
        near = beside[:, c0] | beside[:, c1]
        crossing = near & (r0 <= bottoms[:, middle]) & (r1 >= tops[:, middle])
        if crossing.sum() == 1:
            owner[i] = int(np.argmax(crossing))
        elif crossing.sum() == 2:
            shared.append(i)  # a piece joining two systems, as a low note and a high one touching
        elif crossing.sum() == 0 and near.any():
            clear.append(i)
        # Across three systems or more: the edge of a scan, or a frame round the page

    def outlines() -> tuple[np.ndarray, np.ndarray]:
        """Each system's highest and lowest in every column: its staff lines, the pieces it has, and
        each pixel of a piece shared with another system, the nearer system's."""
        high = np.full((len(systems), w), np.inf)
        low = np.full((len(systems), w), -np.inf)
        taken, also = owner[labels], other[labels]
        for k, system in enumerate(systems):
            mine = (taken == k) | (also == k)
            has = mine.any(0)
            high[k, has] = np.argmax(mine, 0)[has]
            low[k, has] = h - 1 - np.argmax(mine[::-1], 0)[has]
            inside = (cols >= system.x0) & (cols <= system.x1)
            high[k, inside] = np.minimum(high[k, inside], tops[k, inside])
            low[k, inside] = np.maximum(low[k, inside], bottoms[k, inside])
        for i in shared:
            r0, r1, c0, c1 = boxes[i - 1]
            ys, xs = np.nonzero(labels[r0:r1 + 1, c0:c1 + 1] == i)
            ys, xs = ys + r0, xs + c0
            apart = np.maximum(tops[:, xs] - ys, ys - bottoms[:, xs]).clip(0)
            nearest = np.argmin(apart, 0)
            for k in np.unique(nearest):
                on = nearest == k
                np.minimum.at(high[k], xs[on], ys[on])
                np.maximum.at(low[k], xs[on], ys[on])
        return high, low

    # A piece clear of every system's staff lines is the system's whose own pieces are nearest above
    # or below it: a slur hung under one system's beams is that system's, though it dips as far
    # towards the next system's staff lines. One as near the one as the other, words or a singer's
    # name between two systems, is both's: which it belongs to takes reading it
    high, low = outlines()
    reach_high = np.where(np.isfinite(high), high, np.where(beside, tops, np.inf))
    reach_low = np.where(np.isfinite(low), low, np.where(beside, bottoms, -np.inf))
    waiting = []
    for i in clear:
        r0, r1, c0, c1 = boxes[i - 1]
        above = reach_high[:, c0:c1 + 1].min(1) - r1
        below = r0 - reach_low[:, c0:c1 + 1].max(1)
        lines_gap = np.maximum(tops[:, (c0 + c1) // 2] - r1, r0 - bottoms[:, (c0 + c1) // 2])
        gap = np.where(above > 0, above, np.where(below > 0, below, 0)).astype(float)
        gap[(lines_gap > FAR * d) | ~(beside[:, c0] | beside[:, c1])] = np.inf
        nearest, second = np.argsort(gap)[:2] if len(gap) > 1 else (int(np.argmin(gap)), None)
        if gap[nearest] <= REACH * d:
            owner[i] = nearest
            if second is not None and gap[second] <= min(REACH * d, BOTH * gap[nearest] + d):
                other[i] = second
        elif np.isfinite(gap[nearest]):
            waiting.append(i)
    # Pieces near ones already taken go with them, and pieces near those, as far as FAR from the
    # staff lines: a word goes letter by letter, and a dashed line dash by dash
    for i in small:
        r0, r1, c0, c1 = boxes[i - 1]
        middle = (c0 + c1) // 2
        lines_gap = np.maximum(tops[:, middle] - r1, r0 - bottoms[:, middle])
        if np.any((lines_gap <= FAR * d) & (beside[:, c0] | beside[:, c1])):
            waiting.append(i)
    newest = np.flatnonzero(owner >= 0)
    while waiting and len(newest):
        b = np.array([boxes[i - 1] for i in newest])
        still, added = [], []
        for i in waiting:
            r0, r1, c0, c1 = boxes[i - 1]
            dy = np.maximum(0, np.maximum(b[:, 0] - r1, r0 - b[:, 1]))
            dx = np.maximum(0, np.maximum(b[:, 2] - c1, c0 - b[:, 3]))
            near = (dy <= CHAIN * d) & (dx <= CHAIN * d)
            if near.any():
                owner[i] = owner[newest[np.argmax(near)]]
                # Near pieces of two systems, or of one that is both's: both's
                others = {*owner[newest[near]], *other[newest[near]]} - {owner[i], -1}
                if others:
                    other[i] = min(others, key=lambda k: abs(k - owner[i]))
                added.append(i)
            else:
                still.append(i)
        waiting, newest = still, np.array(added, int)
    # As far out as its furthest piece, measured from its staff lines where that piece is
    high, low = outlines()
    for k, system in enumerate(systems):
        some = np.isfinite(high[k])
        system.above = float(max(0.0, (tops[k] - high[k])[some].max())) if some.any() else 0.0
        system.below = float(max(0.0, (low[k] - bottoms[k])[some].max())) if some.any() else 0.0


# ---------------------------------------------------------------------------------------------
# From the command line


def draw(page: np.ndarray, layout: Layout):
    """The page with what was found drawn on it: each system's staff lines, its top and bottom, and
    a bar down the left from the one to the other."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(page).convert("RGB")
    pen = ImageDraw.Draw(im)
    colours = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    for k, system in enumerate(layout.systems):
        colour = colours[k % len(colours)]
        for staff in system.staves:
            xs = np.linspace(staff.x0, staff.x1, 24)
            for line in range(5):
                pen.line([(x, staff.line(line, x)) for x in xs], fill=colour, width=1)
        for edge in (system.top, system.bottom):
            pen.line([(system.x0, edge(system.x0)), (system.x1, edge(system.x1))], fill=colour, width=2)
        x = system.x0 - 8
        pen.line([(x, system.top(system.x0)), (x, system.bottom(system.x0))], fill=colour, width=3)
        pen.text((x - 22, (system.lines_top(system.x0) + system.lines_bottom(system.x0)) / 2 - 6),
                 f"{k + 1}:{len(system.staves)}", fill=colour)
    return im


def main() -> int:
    import argparse
    import itertools
    from pathlib import Path

    from .page import gray, normalise, pdf_pages

    ap = argparse.ArgumentParser(description="The systems on each page of a PDF or picture, top and bottom.")
    ap.add_argument("source", help="a PDF, or a picture of a page")
    ap.add_argument("--pages", default="", help="for a PDF: pages to count, e.g. 1-3,7")
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
    results = []
    for number, page in pages:
        layout = count(page.pixels)
        sizes = [len(s.staves) for s in layout.systems]
        print(f"page {number}: {len(sizes)} systems of {', '.join(map(str, sizes)) or 'no'} staves"
              f" (turned {layout.angle:+.2f} degrees)")
        results.append({"page": number, **layout.as_dict()})
        if out:
            draw(page.pixels, layout).save(out / f"p{number:03d}.png")
    if args.json:
        Path(args.json).write_text("[\n" + ",\n".join(json.dumps(r) for r in results) + "\n]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
