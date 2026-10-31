"""Converts KernScores' Humdrum **kern files to MusicXML for piano on two staves, measure by measure.

The real-scan manifest (data/real_scans/manifest.jsonl) names the works its scans hold in GrandStaff's
folder names ("chopin/mazurkas/mazurka59-1", "mozart/piano-sonatas/sonata01",
"scarlatti-d/keyboard-sonatas/L001K514"). Their whole-piece kern files are in Craig Sapp's repositories,
cloned shallow (git clone --depth 1 https://github.com/craigsapp/<repo>) under
data/downloads/kernscores/<repo>/kern/ (untrusted data: read here as text only, never run).

  python -I tools/kern_to_musicxml.py --all [--manifest data/real_scans/manifest.jsonl]
        [--clones data/downloads/kernscores] [--out data/real_scans/scores/kern] [--check] [--workers N]
  python -I tools/kern_to_musicxml.py chopin/mazurkas/mazurka59-1 ... [--clones DIR] [--out DIR] [--log FILE]
  (later sets: bach/wtc/wtc1f01, bach/inventions/sinfo01, beethoven/piano-sonatas/sonata01, haydn/piano-sonatas/sonata12,
   scriabin/op08/op8_no07, chopin/first-editions/032-1-Sm-001; see SETS and SCRIABIN)

writes <out>/<work>.musicxml and <out>/conversion.jsonl (one line a work: the kern measures, the
measures written, and with --check the bars bar_dataset.py's linearize makes of it and how many LMX
cannot write). It keeps going when a work fails and says why on stderr and in conversion.jsonl.

How a work is converted. music21 parses kern, but its Humdrum spine paths fail on nested splits and
three-way merges (a Chopin hand written as two or three voices), and it reads no slurs, so the
spine paths are resolved here first: each staff's voices become flat **kern columns that start and
end with the piece, and music21 reads that. The voices of a staff are then put back together in one
measure as music21 Voices (the right-hand child of a split keeps voice 1, as MuseScore numbers them;
voices 1-4 sit on the upper staff, 5-8 on the lower), slurs and arpeggios are added from the kern, and
the two staves are written as one part with <staves>2</staves> (what bar_dataset.piano_part
accepts). Details:

  * the upper staff is *staff1 (or the rightmost kern spine when the file gives no *staff); a file
    with three or more staves (prelude 28-08) puts the F-clef spines on the lower staff and the rest
    on the upper; two spines of one staff (prelude 28-01) become two voices;
  * **dynam and other non-kern spines are dropped, and so are *I instruments (a cembalo score would
    be named for a harpsichord, which piano_part refuses): the part is named Piano;
  * a hidden rest (4ryy) and the start of a voice that opens part-way through a measure are gaps
    (<forward>), not rests; a staff that plays nothing in a bar gets the bar's rest, as a printed
    score shows it; a rest that fills its bar alone is written as a whole-bar rest
    (<rest measure="yes"/>);
  * nothing is expanded: repeats stay as written (their signs are not written: LMX ignores barlines),
    a pickup is its own measure, and every measure of the kern is one of the score, empty ones too;
  * a grace note that ends a measure in the kern is written at the start of the next one, where
    MuseScore puts it; a chord's notes are written lowest first (LMX needs it); elided slurs (&( &))
    and a slur that opens and closes on one note are left out; stems are written only where the
    kern gives them (the / and backslash marks), not guessed;
  * voices are numbered as MuseScore does: 1, 2, ... on the upper staff, 5, 6, ... on the lower;
  * a work of several kern files (a Mozart sonata's movements, a variation set's variations) is the
    files in name order one after the other, measures renumbered from 1 and each opening with the
    key, time and clefs (conversion.jsonl says where each file starts).

Checks it makes and writes to conversion.jsonl (and calls "problems" when they fail): the measures
written equal the kern's, the notes and rests equal the kern's (less the hidden rests, plus the bar
rests of silent staves), the pitches of every measure equal the kern's (graces left out), every slur
is written, and with --check bar_dataset.read, piano_part and linearize all work and linearize makes
a bar of every measure. Not problems but listed: irregular_measures (a voice that is not the
time signature's length: pickups, the end of a repeated section) and overfull_measures (longer than
the bar: the kern has no barline for a cadenza or a meter it only comments on), empty_measures
(the kern has the bar without notes), and the bars LMX cannot write.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import re
import sys
import time
import traceback
import warnings
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

ROOT = Path.cwd()

REPOS = {
    "chopin/mazurkas": "chopin-mazurkas",
    "chopin/preludes": "chopin-preludes",
    "mozart/piano-sonatas": "mozart-piano-sonatas",
    "scarlatti-d/keyboard-sonatas": "scarlatti-keyboard-sonatas",
    "hummel/preludes": "hummel-preludes",
    "joplin/joplin": "joplin",
}
# Works of the later sets, named <composer>/<set>/<work> (the work is a kern file's name
# without .krn, or the start of the names of its movements): "<composer>/<set>" -> (repository, folders of kern
# files under it, in the order they are searched). Scriabin's sets are opNN folders (see SCRIABIN).
SETS = {
    "bach/wtc": ("bach-wtc", ["kern"]),
    "bach/inventions": ("inventions", ["kern"]),
    "beethoven/piano-sonatas": ("beethoven-piano-sonatas", ["kern"]),
    "haydn/piano-sonatas": ("haydn-piano-sonatas", ["kern"]),
    "chopin/first-editions": ("chopin-first-editions", ["kern"]),
}
SCRIABIN = re.compile(r"scriabin/(op\d+)/")  # scriabin/op08/op8_no07: repository scriabin, folders op08 and ccarh_kern
MAX_SLOTS = 4  # voices on one staff: MusicXML voices 1-4 (upper staff) and 5-8 (lower)
PAD_PITCH = "CCCCCCCC"  # a note no score holds (C in octave -4): marks a gap, and is taken out again
SLUR_MARK = "@slur:"  # on a lyric: where a slur starts or stops, until place_slurs writes it
EMPTY = f"256{PAD_PITCH}"  # a 256th of the marker: all an otherwise empty measure of a column holds


class Failure(Exception):
    """A work that cannot be converted, with the cause."""

    def __init__(self, cause: str, message: str):
        super().__init__(f"{cause}: {message}")
        self.cause = cause


# ---------------------------------------------------------------------------- finding the kern files


def kern_files(work: str, clones: Path) -> list[Path]:
    """The kern files of a work in the order they are played. `work` is a GrandStaff folder name."""
    parts = work.split("/")
    if parts[0] == "joplin" and len(parts) in (2, 3):  # GrandStaff: joplin/joplin/<rag>
        key, name = "joplin/joplin", parts[-1]
    elif len(parts) == 3:
        key, name = "/".join(parts[:2]), parts[2]
    else:
        raise Failure("unmapped", f"{work}: not <composer>/<set>/<work>")
    if key in REPOS:
        folders = [clones / REPOS[key] / "kern"]
    elif key in SETS:
        folders = [clones / SETS[key][0] / f for f in SETS[key][1]]
    elif SCRIABIN.match(work):
        folders = [clones / "scriabin" / parts[1], clones / "scriabin" / "ccarh_kern"]
    else:
        raise Failure("unmapped", f"{work}: no repository known for {key}")
    for kern in folders:
        if (kern / f"{name}.krn").is_file():
            return [kern / f"{name}.krn"]
    # a Mozart sonata: sonata01 is sonata01-1, sonata01-2, sonata01-3 (sonata06-3a ... the variations); a
    # Scriabin sonata op19 is op19_no01, op19_no02; a Chopin sonata 058-1-BH is 058-1-BH-001 ... 004
    for kern in folders:
        if not kern.is_dir():
            raise Failure("missing-clone", f"{kern} is not there")
        movements = sorted(list(kern.glob(f"{name}-*.krn")) + list(kern.glob(f"{name}_*.krn")))
        if movements:
            return movements
    raise Failure("unmapped", f"{work}: no {name}.krn or {name}-*.krn in {[str(k) for k in folders]}")


def manifest_works(manifest: Path) -> list[str]:
    works = set()
    for line in manifest.read_text(encoding="utf8").splitlines():
        if line.strip():
            works.update(json.loads(line).get("works", []))
    return sorted(works)


# ---------------------------------------------------------------------------- kern tokens

RECIP = re.compile(r"(\d+)(?:%(\d+))?(\.*)")


def token_duration(token: str) -> Fraction:
    """Whole notes a kern token lasts (a chord by its first note); 0 for a grace note."""
    first = token.split(" ")[0]
    if re.search(r"[qQ]", first):
        return Fraction(0)
    m = RECIP.search(first)
    if not m:
        return Fraction(0)
    digits, per, dots = m.group(1), m.group(2), len(m.group(3))
    if per:
        base = Fraction(int(per), int(digits))
    elif set(digits) == {"0"}:
        base = Fraction(2 ** len(digits))  # 0 a breve, 00 a long
    else:
        base = Fraction(1, int(digits))
    return base * (2 - Fraction(1, 2 ** dots))


def is_hidden_rest(token: str) -> bool:
    return "r" in token and "y" in token and " " not in token and RECIP.search(token) is not None


def undot_ratio(token: str) -> str:
    """A dotted ratio duration (20%3.) as an undotted one (40%9): music21 drops the dot after a %."""

    def one(sub: str) -> str:
        m = RECIP.search(sub)
        if not m or not m.group(2) or not m.group(3):
            return sub
        d = Fraction(int(m.group(2)), int(m.group(1))) * (2 - Fraction(1, 2 ** len(m.group(3))))
        new = str(d.denominator) if d.numerator == 1 else f"{d.denominator}%{d.numerator}"
        return sub[:m.start()] + new + sub[m.end():]

    return " ".join(one(sub) for sub in token.split(" "))


def clean_token(token: str) -> str:
    """A note token music21 can read: no slur or phrase marks (the slurs are read here), no accidental
    on a rest, a chord's stray number dropped and a note with no duration given its chord's."""
    subs = [s for s in token.split(" ") if s]
    recip = next((m.group(0) for m in (RECIP.search(s) for s in subs) if m), None)
    out = []
    for sub in subs:
        sub = re.sub(r"[(){}&]", "", sub)
        if "r" in sub:
            sub = re.sub(r"[#\-n]", "", sub)
        if not re.search(r"[a-gA-Gr]", sub):
            continue
        if not RECIP.search(sub) and recip:
            sub = recip + sub
        out.append(sub)
    if len(out) > 1 and any(re.search(r"[qQ]", x) for x in out) and not all(re.search(r"[qQ]", x) for x in out):
        out = [x for x in out if not re.search(r"[qQ]", x)]  # a grace note written into a chord is lost
    if any("%" in x for x in out):
        return undot_ratio(" ".join(out))
    return " ".join(out)


def rest_tokens(duration: Fraction) -> list[str]:
    """Kern tokens for a gap of `duration` whole notes, each the marker note (never a rest or a score's note)."""
    if duration.denominator & (duration.denominator - 1):  # not a power of two: tuplet units
        return [f"{duration.denominator}{PAD_PITCH}"] * duration.numerator
    out = []
    while duration > 0:
        n = 1
        while Fraction(1, n) > duration:
            n *= 2
        piece, dots = Fraction(1, n), 0
        while dots < 3 and piece * (2 - Fraction(1, 2 ** (dots + 1))) <= duration:
            dots += 1
        out.append(f"{n}{'.' * dots}{PAD_PITCH}")
        duration -= piece * (2 - Fraction(1, 2 ** dots))
    return out


def follow_paths(kinds: list[str], tokens: list[str]) -> list[str]:
    """The spine kinds left to right after a line of spine-path interpretations."""
    out, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "*^":
            out += [kinds[i], kinds[i]]
        elif tok == "*v":
            j = i
            while j < len(tokens) and tokens[j] == "*v":
                j += 1
            out.append(kinds[i])
            i = j - 1
        elif tok == "*x" and i + 1 < len(tokens):
            out += [kinds[i + 1], kinds[i]]
            i += 1
        elif tok == "*+":
            out += [kinds[i], "**new"]
        elif tok != "*-":
            out.append(kinds[i])
        i += 1
    return out


def count_kern(text: str) -> tuple[int, int, int, int]:
    """(measures, note events, hidden rests, empty measures) of a kern file: the segments between
    barlines (the one before the first barline, a pickup, and the one after the last count only if
    they hold data; the rest count even if empty), and every note and rest of every **kern spine (a
    chord's notes each), of which the hidden rests (4ryy) are counted again."""
    segments: list[bool] = []  # does the segment hold data
    has, events, hidden = False, 0, 0
    kinds: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("!"):
            continue
        tokens = line.split("	")
        if line.startswith("**"):
            kinds = tokens
        elif line.startswith("*"):
            kinds = follow_paths(kinds, tokens)
        elif line.startswith("="):
            segments.append(has)
            has = False
        else:
            has = True
            for kind, tok in zip(kinds, tokens):
                if kind == "**kern" and tok != ".":
                    events += len([x for x in tok.split(" ") if x])
                    hidden += is_hidden_rest(tok)
    segments.append(has)
    if len(segments) == 1:
        return int(segments[0]), events, hidden, 0
    middle = segments[1:-1]
    return int(segments[0]) + len(middle) + int(segments[-1]), events, hidden, middle.count(False)


# ---------------------------------------------------------------------------- resolving spine paths


class Spine:
    """One spine as the layout holds it now: a leaf of the splits and merges so far."""

    __slots__ = ("kind", "staff", "slot", "acc")

    def __init__(self, kind, staff=None, slot=None, acc=Fraction(0)):
        self.kind, self.staff, self.slot, self.acc = kind, staff, slot, acc


class Flat:
    """A kern file with its spine paths resolved."""

    def __init__(self):
        self.text = ""
        self.staff_of: dict[int, int] = {}  # column -> staff (1 upper, 2 lower)
        self.columns = 0
        self.slurs: list[tuple[int, int]] = []  # (event that opens, event that closes)
        self.arpeggios: list[int] = []  # events marked as arpeggiated
        self.where: dict[int, tuple[int, int]] = {}  # event -> (column, line of the flat text)
        self.placed: dict[tuple[int, int], tuple[bool, str]] = {}  # (column, line) -> (a rest, the token)
        self.stats = {"slur_closes_unmatched": 0, "slurs_left_open": 0, "tokens_lost": 0}


def renumber(token: str, k: int) -> str:
    """A barline token numbered k whatever number or letter it had: music21 drops the grace notes of a
    measure its barline gives no number (a pickup after a repeat sign), and the kern's own numbers
    repeat (41a, 41b) or restart."""
    if token.startswith("=="):
        return token
    return f"={k}" + re.sub(r"[0-9a-z\-]", "", token[1:])


def trailing_graces(out: list, start: int) -> list:
    """Takes the grace notes that end a column's measure out of rows[start:] and returns them in
    order as (column, cell): they belong to the next measure, whose first note they lead into."""
    closed: set = set()
    taken = []
    for k in range(len(out) - 1, start - 1, -1):
        kind, row = out[k]
        if kind != "data":
            continue
        for c in list(row):
            tok = row[c][0]
            if c in closed:
                continue
            if re.search(r"[qQ]", tok) and not tok.endswith(PAD_PITCH):
                taken.append((c, row.pop(c)))
            else:
                closed.add(c)
    taken.reverse()
    return taken


def staves_of(lines: list[str]) -> tuple[list[str], dict[int, int]]:
    """The first line's spine kinds, and the staff (1 upper, 2 lower) of each **kern spine by its column."""
    kinds = None
    tags: dict[int, list[int]] = {}
    clefs: dict[int, str] = {}
    for line in lines:
        if line.startswith("!") or not line.strip():
            continue
        tokens = line.split("\t")
        if line.startswith("**"):
            kinds = tokens
        elif line.startswith("*") and kinds:
            if any(t in ("*^", "*v", "*x", "*+", "*-") for t in tokens):
                break
            for i, tok in enumerate(tokens):
                m = re.fullmatch(r"\*staff(\d+)", tok)
                if m and i < len(kinds) and kinds[i] == "**kern":
                    tags.setdefault(i, []).append(int(m.group(1)))
                if tok.startswith("*clef") and i not in clefs:
                    clefs[i] = tok
        else:
            break
    if kinds is None:
        raise Failure("not-kern", "no **kern line")
    kerns = [i for i, k in enumerate(kinds) if k == "**kern"]
    if len(kerns) < 2:
        raise Failure("not-two-staves", f"{len(kerns)} **kern spine")
    numbers = {i: tags[i][0] for i in kerns if i in tags}
    if len(numbers) == len(kerns) and len(set(numbers.values())) == 2:
        low = max(numbers.values())
        return kinds, {i: 2 if numbers[i] == low else 1 for i in kerns}
    # no (or too many) *staff tags: the F-clef spines are the lower staff, the rest the upper
    staff = {i: 2 if clefs.get(i, "").startswith("*clefF") else 1 for i in kerns}
    if len(set(staff.values())) == 2:
        return kinds, staff
    if len(kerns) == 2 and not numbers:  # kern lists the lowest spine first
        return kinds, {kerns[0]: 2, kerns[1]: 1}
    raise Failure("not-two-staves", f"cannot tell the staves apart: tags {numbers}, clefs {clefs}")


def flatten(text: str) -> Flat:
    """Kern with its spine paths resolved: flat columns, each one voice of one staff (a staff's voices
    in slots 0, 1, ...), plus the slurs and arpeggios read from the kern."""
    lines = [ln.rstrip("\r\n") for ln in text.split("\n")]
    kinds, staff_of = staves_of(lines)
    cols: list[tuple[int, Spine]] = []
    used: dict[int, set[int]] = {1: set(), 2: set()}
    for i in sorted(range(len(kinds)), key=lambda i: -i):  # roots of one staff take slots from the right
        if kinds[i] == "**kern":
            s = staff_of[i]
            slot = min(set(range(MAX_SLOTS)) - used[s])
            used[s].add(slot)
            cols.append((i, Spine("kern", s, slot)))
        else:
            cols.append((i, Spine(kinds[i])))
    spines = [p for _, p in sorted(cols, key=lambda c: c[0])]
    # rows: ("bar", token) / ("tandem", {(staff, 0): token}) / ("data", {(staff, slot): (token, event or None)})
    out: list[tuple[str, object]] = []
    col_acc: dict[tuple[int, int], Fraction] = {}
    max_slot = {1: 0, 2: 0}
    flat = Flat()
    # slurs belong to a staff (one may open in one voice and close in another); LMX only marks where
    # they start and stop, so which open slur a ) closes does not matter
    open_slurs: dict[int, list[int]] = {1: [], 2: []}  # staff -> the events that opened the slurs still open
    next_event = 0
    segment = 0  # where the current measure's rows start in `out`
    bars = 0  # barlines so far
    started = ended = False
    # a double barline (==) ends the piece only if it is the file's last barline: a section or movement may end
    # with one in the middle of a file (a minuet before its trio), and music21 would stop reading there
    last_bar = max((n for n, ln in enumerate(lines) if ln.startswith("=")), default=-1)
    for at, line in enumerate(lines):
        if not line.strip() or line.startswith("!") or ended:
            continue
        tokens = line.split("\t")
        if line.startswith("**"):
            if started:
                raise Failure("kern-syntax", "a second exclusive-interpretation line")
            started = True
            continue
        if len(tokens) != len(spines):
            raise Failure("kern-syntax", f"{len(tokens)} tokens for {len(spines)} spines: {line[:60]!r}")
        if line.startswith("="):
            first = next((t for t, p in zip(tokens, spines) if p.kind == "kern"), tokens[0])
            final = first.startswith("==") and at == last_bar
            moved = [] if final else trailing_graces(out, segment)
            bars += 1
            out.append(("bar", renumber(first, bars) if final else f"={bars}" + re.sub(r"[0-9a-z\-]", "", first[1:].lstrip("="))))
            out.extend(("data", {c: cell}) for c, cell in moved)
            segment = len(out) - len(moved)
            for p in spines:
                p.acc = Fraction(0)
            col_acc = {}
            ended = final
        elif line.startswith("*"):
            # clefs, keys and time signatures, once for each staff, from its lowest voice that has one
            found: dict[str, dict[int, str]] = {"clef": {}, "key": {}, "meter": {}}
            for tok, p in sorted(zip(tokens, spines), key=lambda tp: tp[1].slot if tp[1].slot is not None else 9):
                if p.kind != "kern":
                    continue
                kind = ("clef" if tok.startswith("*clef") else "key" if re.match(r"\*k\[", tok)
                        else "meter" if re.match(r"\*M\d+/\d+", tok) else None)
                if kind:
                    found[kind].setdefault(p.staff, tok)
            for by_staff in found.values():
                if by_staff:
                    out.append(("tandem", {(s, 0): tok for s, tok in by_staff.items()}))
            new_spines: list[Spine] = []
            pads: list[tuple[int, int, Fraction]] = []
            i = 0
            while i < len(tokens):
                tok, p = tokens[i], spines[i]
                if tok == "*^":
                    right = Spine(p.kind, p.staff, p.slot, p.acc)
                    left = Spine(p.kind, p.staff, p.slot, p.acc)
                    if p.kind == "kern":
                        free = set(range(MAX_SLOTS)) - used[p.staff]
                        if not free:
                            raise Failure("too-many-voices", f"more than {MAX_SLOTS} voices on staff {p.staff}")
                        left.slot = min(free)
                        used[p.staff].add(left.slot)
                        max_slot[p.staff] = max(max_slot[p.staff], left.slot)
                        pads.append((left.staff, left.slot, left.acc))
                    new_spines += [left, right]
                    i += 1
                elif tok == "*v":
                    j = i
                    while j < len(tokens) and tokens[j] == "*v":
                        j += 1
                    group = spines[i:j]
                    if len({g.kind == "kern" for g in group}) != 1:
                        raise Failure("kern-syntax", "a merge of kern and other spines")
                    merged = Spine(group[0].kind, group[0].staff, None, max(g.acc for g in group))
                    if merged.kind == "kern":
                        # the merged voice goes on in the first spine's staff (a voice may have crossed over)
                        merged.slot = min(g.slot for g in group if g.staff == merged.staff)
                        for g in group:
                            used[g.staff].discard(g.slot)
                        used[merged.staff].add(merged.slot)
                        pads.append((merged.staff, merged.slot, merged.acc))
                    new_spines.append(merged)
                    i = j
                elif tok == "*-":
                    if p.kind == "kern":
                        used[p.staff].discard(p.slot)
                    i += 1
                elif tok == "*x":
                    if i + 1 >= len(tokens) or tokens[i + 1] != "*x":
                        raise Failure("kern-syntax", "an unpaired *x")
                    new_spines += [spines[i + 1], p]
                    i += 2
                elif tok == "*+":
                    raise Failure("kern-syntax", "*+ (a spine added) is not handled")
                else:
                    new_spines.append(p)
                    i += 1
            spines = new_spines
            for s, slot, acc in pads:  # a voice born part-way through a measure opens with a gap
                gap = acc - col_acc.get((s, slot), Fraction(0))
                if gap > 0:
                    for tok in rest_tokens(gap):
                        out.append(("data", {(s, slot): (tok, None)}))
                    col_acc[(s, slot)] = acc
        else:
            row: dict[tuple[int, int], tuple[str, int | None]] = {}
            for tok, p in zip(tokens, spines):
                if p.kind != "kern" or tok == ".":
                    continue
                note = clean_token(tok)
                dur = token_duration(note)
                p.acc += dur
                event = next_event
                next_event += 1
                for at, ch in enumerate(tok):  # slurs: ( opens one and ) closes the latest one still open on the staff
                    if at and tok[at - 1] == "&":
                        continue  # &( and &) are elided slurs, not drawn (often on one note)
                    if ch == "(":
                        open_slurs[p.staff].append(event)
                    elif ch == ")":
                        if open_slurs[p.staff]:
                            start = open_slurs[p.staff].pop()
                            if start != event:  # (4c) is no slur
                                flat.slurs.append((start, event))
                        else:
                            flat.stats["slur_closes_unmatched"] += 1
                if ":" in tok:
                    flat.arpeggios.append(event)
                if not note:
                    flat.stats["tokens_lost"] += 1
                    continue
                if is_hidden_rest(note):
                    note = RECIP.search(note).group(0) + PAD_PITCH  # a hidden rest is a gap, not a rest
                    event = None
                row[(p.staff, p.slot)] = (note, event)
                col_acc[(p.staff, p.slot)] = col_acc.get((p.staff, p.slot), Fraction(0)) + dur
                max_slot[p.staff] = max(max_slot[p.staff], p.slot)
            out.append(("data", row))
    if not started:
        raise Failure("not-kern", "no **kern line")
    flat.stats["slurs_left_open"] = sum(len(v) for v in open_slurs.values())

    columns = [(s, slot) for s in (2, 1) for slot in range(max_slot[s] + 1)]
    index = {c: n for n, c in enumerate(columns)}
    flat.columns = len(columns)
    flat.staff_of = {n: c[0] for n, c in enumerate(columns)}
    lines_out = ["\t".join(["**kern"] * len(columns))]
    seen: set[tuple[int, int]] = set()  # the columns that hold an event in this measure so far

    def place(cells: list[str], c: tuple[int, int], tok: str, event: int | None) -> None:
        n = index[c]
        cells[n] = tok
        line = len(lines_out)  # the line this row will be (music21 calls it the event's humdrumPosition)
        if event is not None:
            flat.where[event] = (n, line)
        flat.placed[(n, line)] = ("r" in tok, tok)

    def fill() -> None:
        """A column with no event in a measure gets a marker, so music21 makes the measure in it too."""
        if seen:
            empty = [c for c in columns if c not in seen]
            if empty:
                cells = ["."] * len(columns)
                for c in empty:
                    place(cells, c, EMPTY, None)
                lines_out.append("\t".join(cells))
        seen.clear()

    for kind, row in out:
        if kind == "bar":
            fill()
            lines_out.append("\t".join([row] * len(columns)))
        elif kind == "tandem":
            cells = ["*"] * len(columns)
            for c, tok in row.items():
                if c in index:
                    cells[index[c]] = tok
            lines_out.append("\t".join(cells))
        else:
            cells = ["."] * len(columns)
            for c, (tok, event) in row.items():
                place(cells, c, tok, event)
                seen.add(c)
            lines_out.append("\t".join(cells))
    fill()
    lines_out.append("\t".join(["*-"] * len(columns)))
    flat.text = "\n".join(lines_out) + "\n"
    return flat


# ---------------------------------------------------------------------------- music21 to MusicXML


def parse_flat(text: str):
    """music21's score of a flat kern text, and what music21 said about events it could not read."""
    from music21 import converter

    said = io.StringIO()
    with contextlib.redirect_stderr(said), contextlib.redirect_stdout(said), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = converter.parseData(text, format="humdrum")
    return score, [ln.strip() for ln in said.getvalue().splitlines() if ln.strip()]


def offset_in(measure, el):
    """Where an element is in a measure (not where its last-used container puts it)."""
    try:
        return measure.elementOffset(el)
    except Exception:
        return el.getOffsetInHierarchy(measure)


def is_marker(el) -> bool:
    from music21 import note

    return isinstance(el, note.Note) and el.pitch.octave is not None and el.pitch.octave < -2


def measure_lengths(columns: list) -> list[float]:
    """Quarter notes each measure lasts: as far as any column goes (the markers of an empty
    measure do not count)."""
    count = min(len(c) for c in columns)
    lengths = [0.0] * count
    for measures in columns:
        for i in range(count):
            for el in measures[i].notesAndRests:
                if is_marker(el) and el.quarterLength == 1 / 64:
                    continue
                lengths[i] = max(lengths[i], float(offset_in(measures[i], el) + el.quarterLength))
    return lengths


def staff_part(slot_parts: list, staff: int, lengths: list[float], silent: list[int]):
    """One PartStaff: the voices of a staff (slot 0 first) put together measure by measure."""
    from music21 import clef, instrument, key, meter, note, stream

    measures = [list(p.getElementsByClass(stream.Measure)) for p in slot_parts]
    ps = stream.PartStaff()
    ps.partName = "Piano"
    ps.partAbbreviation = "Pno"
    ps.insert(0, instrument.Piano())
    bar = None
    for i in range(len(measures[0])):
        lead = measures[0][i]
        m = stream.Measure(number=lead.number)
        for attr in ("numberSuffix", "paddingLeft", "paddingRight"):
            with contextlib.suppress(Exception):
                setattr(m, attr, getattr(lead, attr))
        if lead.leftBarline is not None:
            m.leftBarline = copy.deepcopy(lead.leftBarline)
        if lead.rightBarline is not None:
            m.rightBarline = copy.deepcopy(lead.rightBarline)
        for el in lead.elements:
            if isinstance(el, (clef.Clef, key.KeySignature, meter.TimeSignature)):
                m.insert(lead.elementOffset(el), copy.deepcopy(el))
                if isinstance(el, meter.TimeSignature):
                    bar = el.barDuration.quarterLength
        for slot, slot_measures in enumerate(measures):
            voice = stream.Voice()
            voice.id = str(slot + 1 + (0 if staff == 1 else 4))
            for el in slot_measures[i].notesAndRests:
                if not is_marker(el):  # a marker is a gap: the voice just goes on later
                    if el.isChord:
                        el.sortDiatonicAscending(inPlace=True)  # LMX wants a chord's notes ascending
                    voice.insert(offset_in(slot_measures[i], el), el)  # the same object: its slurs follow
            if len(voice):
                m.insert(0, voice)
        voices = list(m.getElementsByClass(stream.Voice))
        if not voices and lengths[i] > 0:  # nothing played on this staff: a printed score shows the bar's rest
            voice = stream.Voice()
            voice.id = str(1 if staff == 1 else 5)
            voice.insert(0, note.Rest(quarterLength=lengths[i]))
            m.insert(0, voice)
            voices = [voice]
            silent[0] += 1
        # a rest that is the staff's only event and fills the bar is the bar's rest
        events = list(voices[0].notesAndRests) if len(voices) == 1 else []
        if (len(events) == 1 and isinstance(events[0], note.Rest) and bar
                and events[0].quarterLength == bar):
            events[0].fullMeasure = True
        ps.append(m)
    return ps


def whole_score(flat: Flat):
    """(a music21 Score of one piano part on two staves, what music21 said, silent staff-bars given a
    rest, slurs and arpeggios added, events music21 did not read)."""
    from music21 import expressions, layout, stream

    score, said = parse_flat(flat.text)
    by_column = {}
    for p in score.parts:
        m = re.fullmatch(r"spine_(\d+)", str(p.id))
        if m:
            by_column[int(m.group(1))] = p
    columns = {s: sorted(c for c, st in flat.staff_of.items() if st == s) for s in (1, 2)}
    missing = [c for cs in columns.values() for c in cs if c not in by_column]
    if missing:
        raise Failure("music21-parse", f"no part for columns {missing}; music21 said {said[:3]}")
    measures = {c: list(by_column[c].getElementsByClass(stream.Measure)) for cs in columns.values() for c in cs}
    counts = {c: len(m) for c, m in measures.items()}
    if len(set(counts.values())) != 1:
        raise Failure("measure-count", f"voices disagree on the measures: {counts}")
    lengths = measure_lengths(list(measures.values()))
    silent = [0]
    staves = {s: staff_part([by_column[c] for c in columns[s]], s, lengths, silent) for s in (1, 2)}

    # slurs and arpeggios: music21 gives each note the line of the flat text it was read from
    elements: dict[tuple[int, int], object] = {}
    for c, ms in measures.items():
        for m in ms:
            for el in m.notesAndRests:
                elements[(c, getattr(el, "humdrumPosition", None))] = el
    lost = [(c, line, tok) for (c, line), (rest, tok) in flat.placed.items()
            if (c, line) not in elements and not tok.endswith(PAD_PITCH)]
    added = {"slurs": 0, "arpeggios": 0}
    for sid, (start, end) in enumerate(flat.slurs):
        a, b = elements.get(flat.where.get(start)), elements.get(flat.where.get(end))  # an event may be lost: no place
        if (a is not None and b is not None and not a.isRest and not b.isRest
                and flat.staff_of[flat.where[start][0]] == flat.staff_of[flat.where[end][0]]):
            # music21 numbers slurs 1 to 6 round and round, whether they overlap or not, so the slurs
            # are written by place_slurs afterwards: a lyric marks each end for it to find
            a.addLyric(f"{SLUR_MARK}{sid}:start")
            b.addLyric(f"{SLUR_MARK}{sid}:stop")
            added["slurs"] += 1
    for event in flat.arpeggios:
        el = elements.get(flat.where.get(event))
        if el is not None and el.isChord:
            el.expressions.append(expressions.ArpeggioMark())
            added["arpeggios"] += 1
    out = stream.Score()
    out.insert(0, staves[1])
    out.insert(0, staves[2])
    out.insert(0, layout.StaffGroup([staves[1], staves[2]], name="Piano", abbreviation="Pno",
                                    symbol="brace", barTogether=True))
    return out, said, silent[0], added, lost


def musicxml_of(score) -> ET.Element:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        fp = score.write("musicxml", fp=str(Path(tmp) / "x.musicxml"))
        return ET.parse(fp).getroot()


NOTE_STEPS = 14  # LMX writes a forward or backup as notes of 32 quarters (a maxima) down to 1/256


def lmx_can_write(duration: int, divisions: int) -> bool:
    """LMX splits a forward or backup into note types, binary lengths from a maxima down; the
    rest must come to nothing."""
    step = divisions * 32
    for _ in range(NOTE_STEPS):
        if step <= duration:
            duration -= step
        step //= 2
    return duration == 0


def tidy_xml(root: ET.Element) -> dict:
    """Takes the hidden rests music21 fills the gaps of a voice with out of the part: one that comes
    before more of its voice's notes becomes a <forward>, one that ends its voice is dropped, and the
    <backup>s are set again to the length of the voice before them."""
    stats = {"forwards": 0, "trailing_dropped": 0, "backup_anomalies": 0, "left_hidden": 0, "hidden_kept": 0}
    divisions = int(next(root.iter("divisions")).text)

    def dur(e):
        d = e.find("duration")
        return int(d.text) if d is not None else 0

    def span(e):
        return dur(e) if e.tag == "forward" or (e.tag == "note" and e.find("chord") is None) else 0

    def filler(e):
        return (e.tag == "note" and e.get("print-object") == "no" and e.find("rest") is not None
                and e.find("chord") is None)

    for measure in root.iter("measure"):
        runs, backups, cur = [], [], []
        for e in list(measure):
            if e.tag == "backup":
                runs.append(cur)
                backups.append(e)
                cur = []
            else:
                cur.append(e)
        runs.append(cur)
        if not any(filler(e) for run in runs for e in run):
            continue
        for run, backup in zip(runs, backups):
            if sum(span(e) for e in run) != dur(backup):
                stats["backup_anomalies"] += 1
        new_runs, counted = [], dict(stats)
        for run in runs:
            last = max((k for k, e in enumerate(run) if e.tag in ("note", "forward") and not filler(e)), default=-1)
            new, pend = [], []

            def flush():
                if pend:
                    fw = ET.Element("forward")
                    ET.SubElement(fw, "duration").text = str(sum(dur(e) for e in pend))
                    for tag in ("voice", "staff"):
                        v = pend[0].find(tag)
                        if v is not None:
                            ET.SubElement(fw, tag).text = v.text
                    new.append(fw)
                    stats["forwards"] += 1
                    pend.clear()

            for k, e in enumerate(run):
                if filler(e):
                    if k < last:
                        pend.append(e)
                    else:
                        stats["trailing_dropped"] += 1
                    continue
                flush()
                new.append(e)
            new_runs.append(new)
        lengths = [sum(span(e) for e in run) for run in new_runs]
        forwards = [int(e.find("duration").text) for run in new_runs for e in run if e.tag == "forward"]
        if not all(lmx_can_write(d, divisions) for d in lengths + forwards):  # a tuplet gap: keep the hidden rests
            stats.update(counted)
            stats["left_hidden"] += 1
            stats["hidden_kept"] += sum(1 for run in runs for e in run if filler(e))
            continue
        for e in list(measure):
            measure.remove(e)
        for n, run in enumerate(new_runs):
            if n and sum(span(e) for e in new_runs[n - 1]) > 0:
                back = ET.Element("backup")
                ET.SubElement(back, "duration").text = str(sum(span(e) for e in new_runs[n - 1]))
                measure.append(back)
            measure.extend(run)
    return stats


def place_slurs(root: ET.Element) -> dict:
    """Writes the slurs whose ends whole_score marked with lyrics: <slur> in each note's <notations>,
    numbered so that slurs open at once in the document never share a number (1 to 16)."""
    marks: dict[int, dict[str, int]] = {}
    carriers: list[tuple[ET.Element, str, int]] = []  # (note, start or stop, slur), in document order
    order = 0
    for note in root.iter("note"):
        for lyric in list(note.findall("lyric")):
            text = lyric.findtext("text") or ""
            if text.startswith(SLUR_MARK):
                sid, kind = text[len(SLUR_MARK):].split(":")
                note.remove(lyric)
                marks.setdefault(int(sid), {})[kind] = order
                carriers.append((note, kind, int(sid)))
                order += 1
    stats = {"slurs_placed": 0, "slurs_backwards": 0, "slur_numbers_wrapped": 0}
    numbers: dict[int, int] = {}
    active: list[tuple[int, int]] = []  # (document position it ends at, number)
    for sid in sorted(marks, key=lambda i: min(marks[i].values())):
        ends = marks[sid]
        if len(ends) != 2:
            continue
        lo, hi = min(ends.values()), max(ends.values())
        stats["slurs_backwards"] += ends["stop"] < ends["start"]
        active = [(e, n) for e, n in active if e > lo]
        taken = {n for _, n in active}
        free = [n for n in range(1, 17) if n not in taken]
        if not free:
            stats["slur_numbers_wrapped"] += 1
            free = [1]
        numbers[sid] = free[0]
        active.append((hi, free[0]))
        stats["slurs_placed"] += 1
    for note, kind, sid in carriers:
        if sid not in numbers:
            continue
        notations = note.find("notations")
        if notations is None:
            notations = ET.SubElement(note, "notations")
        ET.SubElement(notations, "slur", {"type": kind, "number": str(numbers[sid])})
    return stats


def renumber_voices(root: ET.Element) -> None:
    """Numbers the voices as MuseScore does: in each measure the upper staff's voices 1, 2, ... in the
    order they are written and the lower staff's 5, 6, ... (music21 numbers them one after another
    across the staves, so the lower staff's first voice is 2 and clashes with the upper staff's)."""
    for measure in root.iter("measure"):
        runs, cur = [], []
        for e in list(measure):
            if e.tag == "backup":
                runs.append(cur)
                cur = []
            elif e.tag in ("note", "forward"):
                cur.append(e)
        runs.append(cur)
        taken = {"1": 0, "2": 0}
        for run in runs:
            if not run:
                continue
            staff = run[0].findtext("staff") or "1"
            number = int(staff == "2") * 4 + 1 + taken.setdefault(staff, 0)
            taken[staff] += 1
            for e in run:
                voice = e.find("voice")
                if voice is None:
                    voice = ET.Element("voice")
                    # <voice> comes after <duration> (and <tie>) in a note, after <duration> in a forward
                    at = max((k for k, c in enumerate(e) if c.tag in ("duration", "tie", "instrument", "footnote",
                                                                      "level", "chord", "pitch", "rest", "unpitched",
                                                                      "grace", "cue")), default=-1)
                    e.insert(at + 1, voice)
                voice.text = str(number)


def irregular_measures(root: ET.Element) -> tuple[list[int], list[int]]:
    """(measures, overfull measures), numbered from 0, whose longest voice is not the time
    signature's length: pickups, the ends of repeated sections and the like, and the overfull ones
    where the kern has no barline for a cadenza or a change of meter it only comments on."""
    out, over, divisions, bar = [], [], None, None
    for n, measure in enumerate(root.find("part").findall("measure")):
        for e in measure.iter("divisions"):
            divisions = int(e.text)
        for t in measure.iter("time"):
            if t.find("beats") is not None and t.find("beat-type") is not None:
                bar = Fraction(int(t.find("beats").text) * 4, int(t.find("beat-type").text))
        longest, pos = 0, 0
        for e in measure:
            if e.tag == "backup":
                pos -= int(e.find("duration").text)
            elif e.tag == "forward" or (e.tag == "note" and e.find("chord") is None):
                d = e.find("duration")
                pos += int(d.text) if d is not None else 0
            longest = max(longest, pos)
        if divisions and bar and Fraction(longest, divisions) != bar:
            out.append(n)
            if Fraction(longest, divisions) > bar:
                over.append(n)
    return out, over


def kern_pitches(text: str) -> list:
    """For each measure of a kern file (as count_kern counts them), a Counter of the pitches (step,
    alter, octave) of its notes, grace notes left out: read from the kern spines directly, with no
    thought of staves or voices, to check the converted score against."""
    import collections

    segments: list = []
    cur, has, kinds = collections.Counter(), False, []
    for line in text.splitlines():
        if not line.strip() or line.startswith("!"):
            continue
        tokens = line.split("	")
        if line.startswith("**"):
            kinds = tokens
        elif line.startswith("*"):
            kinds = follow_paths(kinds, tokens)
        elif line.startswith("="):
            segments.append((cur, has))
            cur, has = collections.Counter(), False
        else:
            has = True
            for kind, tok in zip(kinds, tokens):
                if kind != "**kern" or tok == ".":
                    continue
                for sub in tok.split(" "):
                    m = re.search(r"([a-gA-G]+)", sub)
                    if not m or "r" in sub or re.search(r"[qQ]", sub):
                        continue
                    letters = m.group(1)
                    octave = 3 + len(letters) if letters[0].islower() else 4 - len(letters)
                    cur[(letters[0].upper(), sub.count("#") - sub.count("-"), octave)] += 1
    segments.append((cur, has))
    if not segments[0][1]:
        segments = segments[1:]
    if segments and not segments[-1][1]:
        segments = segments[:-1]
    return [c for c, _ in segments]


def xml_pitches(root: ET.Element) -> list:
    import collections

    out = []
    for measure in root.find("part").findall("measure"):
        c = collections.Counter()
        for n in measure.findall("note"):
            p = n.find("pitch")
            if p is not None and n.find("grace") is None:
                a = p.find("alter")
                c[(p.findtext("step"), int(float(a.text)) if a is not None else 0, int(p.findtext("octave")))] += 1
        out.append(c)
    return out


def clean(root: ET.Element, title: str) -> None:
    """The names of the part and its instrument, and a title, all piano."""
    for e in root.iter():
        if e.tag in ("part-name", "instrument-name"):
            e.text = "Piano"
        elif e.tag in ("part-abbreviation", "instrument-abbreviation"):
            e.text = "Pno"
    for e in list(root):
        if e.tag in ("work", "movement-title", "identification"):
            root.remove(e)
    work = ET.Element("work")
    ET.SubElement(work, "work-title").text = title
    root.insert(0, work)


def convert_file(path: Path, title: str) -> tuple[ET.Element, dict]:
    """(MusicXML root, stats) of one kern file."""
    text = path.read_text(encoding="utf8", errors="replace")
    kern_measures, kern_events, hidden, kern_empty = count_kern(text)
    flat = flatten(text)
    score, said, silent, added, lost = whole_score(flat)
    root = musicxml_of(score)
    tidy = tidy_xml(root)
    renumber_voices(root)
    placed = place_slurs(root)
    clean(root, title)
    parts = root.findall("part")
    measures = len(parts[0].findall("measure")) if len(parts) == 1 else None
    irregular, overfull = irregular_measures(root) if measures else ([], [])
    wanted, got = kern_pitches(text), xml_pitches(root)
    pitch_mismatch = [n for n, (a, b) in enumerate(zip(wanted, got)) if a != b] if len(wanted) == len(got) else None
    empty = [n for n, m in enumerate(parts[0].findall("measure")) if m.find("note") is None] if measures else []
    stats = {"kern": path.name, "kern_measures": kern_measures, "measures": measures,
             "kern_events": kern_events, "kern_empty_measures": kern_empty, "empty_measures": empty[:60],
             "hidden_rests": hidden, "silent_staff_bars": silent,
             "events": sum(1 for _ in root.iter("note")), "columns": flat.columns,
             "slurs_in_kern": len(flat.slurs), "slurs_written": added["slurs"],
             "arpeggios_written": added["arpeggios"], "events_music21_lost": [f"{t} (column {c}, line {n})" for c, n, t in lost[:10]],
             "flatten": flat.stats, "tidy": tidy, "slurs_placed": placed, "irregular_measures": irregular[:30],
             "pitch_mismatch_measures": pitch_mismatch, "irregular_count": len(irregular), "overfull_measures": overfull[:30], "music21": said[:5]}
    return root, stats


def join(roots: list[ET.Element]) -> ET.Element:
    """The movements one after the other as one score, measures numbered from 1."""
    first = roots[0]
    part = first.find("part")
    for other in roots[1:]:
        for m in other.find("part").findall("measure"):
            part.append(m)
    for n, m in enumerate(part.findall("measure"), 1):
        m.set("number", str(n))
    return first


def check_output(path: Path) -> dict:
    """What bar_dataset.py makes of the file: read, piano_part, linearize."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import bar_dataset

    score = bar_dataset.read(path)
    part = bar_dataset.piano_part(score)
    if part is None:
        return {"piano_part": False}
    bars = bar_dataset.linearize(part)
    bad = [i for i, b in enumerate(bars) if b[0] is None]
    return {"piano_part": True, "linearized": len(bars), "bars_lmx_cannot_write": len(bad),
            "unwritable_bars": bad[:60]}


def convert(work: str, clones: Path, out: Path, check: bool) -> dict:
    started = time.time()
    try:
        files = kern_files(work, clones)
        stats = {"work": work, "files": [f.name for f in files]}
        roots, per_file = [], []
        for f in files:
            try:
                root, st = convert_file(f, work)
            except Failure as e:
                raise Failure(e.cause, f"{f.name}: {e}") from e
            roots.append(root)
            per_file.append(st)
        stats["parts"] = per_file
        stats["kern_measures"] = sum(s["kern_measures"] for s in per_file)
        for key in ("kern_events", "hidden_rests", "silent_staff_bars"):
            stats[key] = sum(s[key] for s in per_file)
        stats["starts"] = [sum(s["measures"] for s in per_file[:n]) for n in range(len(per_file))]
        root = join(roots) if len(roots) > 1 else roots[0]
        ET.indent(root)
        dest = out / f"{work}.musicxml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root).write(dest, encoding="utf-8", xml_declaration=True)
        stats["measures"] = len(root.find("part").findall("measure"))
        stats["events"] = sum(1 for _ in root.iter("note"))
        try:
            stats["output"] = dest.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            stats["output"] = dest.as_posix()
        if check:
            stats.update(check_output(dest))
        problems = []
        if stats["measures"] != stats["kern_measures"]:
            problems.append(f"{stats['measures']} measures written for {stats['kern_measures']} in kern")
        expected = (stats["kern_events"] - stats["hidden_rests"] + stats["silent_staff_bars"]
                    + sum(p["tidy"]["hidden_kept"] for p in per_file))
        if stats["events"] != expected:
            problems.append(f"{stats['events']} notes and rests written, {expected} expected from kern")
        if any(p["tidy"]["backup_anomalies"] for p in per_file):
            problems.append("music21's backups did not return to the bar's start")
        if any(p["pitch_mismatch_measures"] is None for p in per_file):
            problems.append("the notes of the kern and of the score could not be compared measure by measure")
        elif any(p["pitch_mismatch_measures"] for p in per_file):
            problems.append("notes differ from the kern's in measures " + str(
                [m + st for p, st in zip(per_file, stats["starts"]) for m in p["pitch_mismatch_measures"]][:12]))
        if any(p["slurs_placed"]["slurs_placed"] != p["slurs_in_kern"] for p in per_file):
            problems.append("slurs written: " + "/".join(
                f"{p['slurs_placed']['slurs_placed']} of {p['slurs_in_kern']}" for p in per_file))
        if check and stats.get("piano_part") is not True:
            problems.append("bar_dataset.piano_part refuses it")
        elif check and stats.get("linearized") != stats["kern_measures"]:
            problems.append(f"{stats.get('linearized')} bars linearized for {stats['kern_measures']} in kern")
        stats["problems"] = problems
    except Failure as e:
        stats = {"work": work, "failure": e.cause, "message": str(e)}
    except Exception as e:  # music21 or anything else: the work is logged and the rest go on
        stats = {"work": work, "failure": "exception", "message": f"{type(e).__name__}: {e}",
                 "trace": traceback.format_exc()[-1500:]}
    stats["seconds"] = round(time.time() - started, 2)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("works", nargs="*", help="works such as chopin/mazurkas/mazurka59-1")
    ap.add_argument("--all", action="store_true", help="every work in the manifest")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/real_scans/manifest.jsonl")
    ap.add_argument("--clones", type=Path, default=ROOT / "data/downloads/kernscores")
    ap.add_argument("--out", type=Path, default=ROOT / "data/real_scans/scores/kern")
    ap.add_argument("--check", action="store_true", help="also run bar_dataset's read, piano_part and linearize")
    ap.add_argument("--workers", type=int, default=1, help="works converted at once")
    ap.add_argument("--log", type=Path, default=None, help="where the per-work lines go (default <out>/conversion.jsonl)")
    args = ap.parse_args()
    works = manifest_works(args.manifest) if args.all else args.works
    if not works:
        ap.error("name works or give --all")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            futures = [pool.submit(convert, w, args.clones, args.out, args.check) for w in works]
            for n, f in enumerate(futures, 1):
                results.append(f.result())
                report(n, len(works), results[-1])
    else:
        for n, work in enumerate(works, 1):
            results.append(convert(work, args.clones, args.out, args.check))
            report(n, len(works), results[-1])
    log = args.log or args.out / "conversion.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    if not args.all:  # keep the other works' lines when only some are redone
        old = {}
        if log.exists():
            for line in log.read_text(encoding="utf8").splitlines():
                d = json.loads(line)
                old[d["work"]] = d
        old.update({r["work"]: r for r in results})
        results = [old[k] for k in sorted(old)]
    log.write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf8")
    failed = [r for r in results if "failure" in r]
    print(f"{len(results) - len(failed)} converted, {len(failed)} failed, {time.time() - started:.0f} s", file=sys.stderr)


def report(n: int, total: int, stats: dict) -> None:
    if "failure" in stats:
        print(f"[{n}/{total}] {stats['work']}: FAILED {stats['message']}", file=sys.stderr, flush=True)
    else:
        note = "; ".join(stats["problems"]) or "ok"
        print(f"[{n}/{total}] {stats['work']}: {stats['measures']} measures, {note}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
