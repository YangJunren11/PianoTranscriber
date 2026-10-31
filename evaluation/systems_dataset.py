"""Bar datasets from datasets of piano systems that come with LMX and MusicXML: OLiMPiC (the
synthetic and the scanned subset) and GrandStaff, each system image cut into bars by the bar
counter's network and the bar cropper, and each bar paired with the LMX of the MusicXML measure
that stands in the same place. The result is in the format of the bar dataset format (see its
docstring), one folder a source, so it can be trained on together with the others.

  python evaluation/systems_dataset.py olimpic OLIMPIC_DIR data/bars_olimpic \\
      --bar-counter models/barcounter.pt [--subset synthetic|scanned|all] [--workers 8] [--limit N]
  python evaluation/systems_dataset.py grandstaff GRANDSTAFF_DIR data/bars_grandstaff \\
      --bar-counter models/barcounter.pt [--subset clean|distorted|all] [--workers 8] [--limit N]

A sample is one system: an image, its MusicXML and its LMX. It is padded with white all round (about
four staff spaces: the staff counter and the network expect margins), scaled as the pipeline scales
a page, and cut; it is paired only if the cropper finds exactly one system of two staves, no stray
peak, and as many bars (the crops not carried) as the MusicXML has measures and the LMX has
`measure` tokens. Bar k is then measure k, and a bar LMX cannot write is left out.

A system's first measure shows the key, time and clefs it starts with. Of a system that continues a
song they go into the context and are taken off the front of the bar's LMX, as bars_lmx does for the
first bar of a system; a song's first system (OLiMPiC's p1-s1) keeps them and has no context.

--limit N takes the first N samples of each split in the dataset's list order (GrandStaff: in the order
of the crc of their names, which has no lists). The output folder is
cleared first, never appended to.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcriber.lmx_bars import CLEFS, KEYS, STAVES, State  # noqa: E402

VAL = 20  # without a partition list of its own: one sample in VAL held out for validation, by name
PAD = 4.0  # staff spaces of white all round
OLIMPIC = "CC BY-SA 4.0 (OLiMPiC)"
GRANDSTAFF = "GrandStaff; LMX CC BY-SA 4.0 (OLiMPiC)"


def first_bar(lmx: str, song_start: bool) -> tuple[str, str]:
    """A system's first measure as the bar dataset's (lmx, context). For a song start, the LMX as it
    is and no context; for a continuation, the key, time and clefs it opens with (as
    lmx_bars.standalone parses them) moved off its front into the context."""
    if song_start:
        return lmx, ""
    tokens = lmx.split()
    k = 1 if tokens[:1] == ["measure"] else 0
    start = k
    while k < len(tokens):
        t = tokens[k]
        if t in KEYS:
            k += 1
        elif t == "time" and k + 2 < len(tokens) and tokens[k + 1].startswith("beats:") \
                and tokens[k + 2].startswith("beat-type:"):
            k += 3
        elif t in CLEFS and k + 1 < len(tokens) and tokens[k + 1] in STAVES:
            k += 2
        else:
            break
    context = " ".join(State().after(["measure", *tokens[start:k]], ties=False).tokens(ties=False))
    return " ".join(["measure", *tokens[k:]]), context


def safe(text: str) -> str:
    return re.sub(r"\W+", "_", text).strip("_")


def sample_page(name: str) -> int:
    """The page number in a sample's name (p3-s2), else 0."""
    m = re.search(r"(?:^|/)p(\d+)-", name)
    return int(m.group(1)) if m else 0


def val_by_name(piece: str) -> str:
    return "val" if zlib.crc32(piece.encode()) % VAL == 0 else "train"


# ---- the datasets ------------------------------------------------------------------------------

def olimpic_samples(root: Path, subset: str, limit: int) -> list[dict]:
    """OLiMPiC's samples: root holds olimpic-1.0-synthetic and olimpic-1.0-scanned (or is one)."""
    parts = {"synthetic": root / "olimpic-1.0-synthetic", "scanned": root / "olimpic-1.0-scanned"}
    for name in parts:
        if root.name.endswith(name):
            parts = {name: root}
    out = []
    for name, folder in parts.items():
        if subset not in (name, "all") or not (folder / "samples").is_dir():
            continue
        for lst, split in (("train", "train"), ("dev", "val"), ("test", "test")):
            path = folder / f"samples.{lst}.txt"
            if not path.exists():
                continue
            lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
            for line in lines[:limit or None]:
                score, _, tail = line.removeprefix("samples/").partition("/")
                piece = f"olimpic_{score}_{tail}"
                stem = str(folder / line)
                out.append({"image": stem + ".png", "musicxml": stem + ".musicxml", "lmx": stem + ".lmx",
                            "piece": piece, "style": f"olimpic_{name}",
                            "page": sample_page(line), "split": split, "licence": OLIMPIC,
                            "song_start": tail == "p1-s1"})
    return out


def grandstaff_samples(root: Path, images: str, limit: int) -> list[dict]:
    """GrandStaff's samples: root holds the .krn, .jpg and _distorted.jpg of grandstaff.tgz, in its
    composer folders, and the LMX and MusicXML in grandstaff-lmx/ beside them, one .lmx and one
    .musicxml for each .krn. The dataset has no partition lists: a whole score's folder (all its
    excerpts, transposed or not, clean or distorted) is held out for validation, one in VAL by name.
    Samples are ordered by the crc of their name, so --limit takes a mixed lot."""
    found = []
    for lmx in sorted((root / "grandstaff-lmx").glob("**/*.lmx")):
        rel = lmx.relative_to(root / "grandstaff-lmx").with_suffix("").as_posix()
        folder = rel.rpartition("/")[0]
        split = val_by_name(folder)
        for style, suffix, wanted in (("grandstaff", ".jpg", "clean"), ("grandstaff_distorted", "_distorted.jpg", "distorted")):
            image = root / (rel + suffix)
            if images in (wanted, "all") and image.exists():
                found.append({"image": str(image), "musicxml": str(lmx.with_suffix(".musicxml")), "lmx": str(lmx),
                              "piece": "grandstaff_" + safe(rel), "style": style, "page": 0, "split": split,
                              "licence": GRANDSTAFF, "song_start": False})
    found.sort(key=lambda s: (zlib.crc32((s["piece"] + s["style"]).encode()), s["piece"], s["style"]))
    taken, out = collections.Counter(), []
    for s in found:
        if not limit or taken[s["split"]] < limit:
            taken[s["split"]] += 1
            out.append(s)
    return out


# ---- one sample --------------------------------------------------------------------------------

_counter = None


def init(repo: str) -> None:
    global _counter
    import torch

    torch.set_num_threads(1)
    from transcriber.bar_counter import BarCounter

    _counter = BarCounter(repo, threads=1)


def process(job) -> tuple[list[dict], collections.Counter, float]:
    from bar_dataset import linearize, piano_part, read
    from transcriber.bar_cropper import crop
    from transcriber.page import gray, normalise, staff_space

    s, out = job
    started = time.time()
    stats: collections.Counter = collections.Counter()
    try:
        picture = gray(s["image"])
        space = staff_space(picture) or 10.0
        pad = round(PAD * space)
        picture = np.pad(picture, pad, constant_values=255)
        page = normalise(picture)
        cropped = crop(page.pixels, _counter.peaks(page.pixels))
    except Exception:  # noqa: BLE001: an unreadable picture
        stats["skipped: image failed"] += 1
        return [], stats, time.time() - started
    layout = cropped.level.layout()
    if len(layout.systems) != 1:
        stats["skipped: not one system"] += 1
        return [], stats, time.time() - started
    if len(layout.systems[0].staves) != 2:
        stats["skipped: not two staves"] += 1
        return [], stats, time.time() - started
    if cropped.stray:
        stats["skipped: stray peaks"] += 1
        return [], stats, time.time() - started
    bars = [b for b in cropped.bars if not b.carried]
    try:
        part = piano_part(read(Path(s["musicxml"])))
        if part is None:
            stats["skipped: not piano"] += 1
            return [], stats, time.time() - started
        measures = linearize(part)
        n_lmx = Path(s["lmx"]).read_text(encoding="utf-8").split().count("measure")
    except Exception:  # noqa: BLE001
        stats["skipped: MusicXML or LMX unreadable"] += 1
        return [], stats, time.time() - started
    if not (len(bars) == len(measures) == n_lmx):
        stats["skipped: count mismatch"] += 1
        stats["  (MusicXML measures differ from LMX's)" if len(measures) != n_lmx
              else f"  (bars - measures = {len(bars) - len(measures):+d})"] += 1
        return [], stats, time.time() - started
    folder = out / s["piece"] / s["style"]
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for k, (bar, (lmx, context)) in enumerate(zip(bars, measures)):
        if lmx is None:
            stats["bars LMX cannot write"] += 1
            continue
        if k == 0:
            lmx, context = first_bar(lmx, s["song_start"])
        name = f"p{s['page']:02d}-b{k:04d}.png"
        Image.fromarray(bar.image).save(folder / name)
        ys, xs = np.nonzero(bar.mask)
        records.append({"image": f"{s['piece']}/{s['style']}/{name}",
                        "mask": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                        "lmx": lmx, "context": context, "piece": s["piece"], "style": s["style"],
                        "page": s["page"], "bar": k, "system": 0, "first": k == 0,
                        "split": s["split"], "licence": s["licence"]})
        stats["bars"] += 1
    stats["paired samples"] += 1
    return records, stats, time.time() - started


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["olimpic", "grandstaff"])
    ap.add_argument("folder", help="the downloaded and extracted dataset")
    ap.add_argument("out")
    ap.add_argument("--bar-counter", required=True, help="the bar counter weights (models/barcounter.pt)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N samples of each split, to try it")
    ap.add_argument("--subset", default="all", choices=["synthetic", "scanned", "clean", "distorted", "all"],
                    help="olimpic: synthetic, scanned or all; grandstaff: its clean images, its distorted ones or all")
    args = ap.parse_args()
    out = Path(args.out)
    samples = (olimpic_samples if args.dataset == "olimpic" else grandstaff_samples)(
        Path(args.folder), args.subset, args.limit)
    if not samples:
        print("no samples found")
        return 1
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    stats, splits, busy = collections.Counter(), collections.Counter(), 0.0
    started = time.time()
    with ProcessPoolExecutor(args.workers, initializer=init, initargs=(args.bar_counter,)) as pool, \
            (out / "bars.jsonl").open("w", encoding="utf-8") as f:
        for k, (records, s, took) in enumerate(pool.map(process, ((s, out) for s in samples), chunksize=4), 1):
            stats.update(s)
            busy += took
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                splits[r["split"]] += 1
            if k % 500 == 0:
                print(f"{k} of {len(samples)} samples ({time.time() - started:.0f} s, {dict(splits)} bars)", flush=True)
    for name, n in sorted(stats.items()):
        print(f"{name}: {n}")
    print(f"samples: {len(samples)}, paired: {stats['paired samples']} "
          f"({100 * stats['paired samples'] / len(samples):.1f} %)")
    print(f"bars written: {dict(splits)}")
    print(f"{busy / len(samples):.2f} s per sample per worker; {time.time() - started:.0f} s in all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
