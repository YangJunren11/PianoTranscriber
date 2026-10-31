"""Trains the bar recognizer (transcriber/bar_recognizer.py) on the bar dataset (tools/bar_dataset.py).

Each bar is given with the state it truly starts from (`lmx_bars.true_states`): its context in the
dataset, and the notes tied over from the bar before. The crops are clean engravings, so each is made
to look scanned at random as it is drawn (thickened or thinned, blurred, noisy, thresholded,
speckled, a little bigger or smaller), as the bar counter's training pages were. Bars are drawn in
batches of about the same width, so little of a batch is padding. The loss is on the bar's tokens
only, not on the state in front of them.

Every --check steps the checkpoint is saved, and the validation bars are scored: their loss, and
--sample of them read as the recognizer reads (each from its true state, the likeliest token at a
time) and scored by transcriber/lmx_accuracy.py. With --patience, training stops once the
validation loss (or with --stop-on, a score of the bars read, the loss breaking ties) has not
improved for so many checks, and the checkpoint kept is the best. With
--init, training goes on from a trained recognizer's weights; with --guard, another dataset's
validation bars are read at each check too, to see what the training costs there.

This is the experiments' copy of tools/train_bar_recognizer.py, with what the designs in
experiments/PLAN.md need:

- several datasets at once (each a root with bars.jsonl and its crops; one that does not exist is
  skipped with a warning), and --val-data for the validation bars (by default the first root's);
- --system-reset (design 01): a bar that opens a system shows its key, time and clefs printed, so it
  is trained to read them: its prefix holds only the notes tied over to it, and its target is
  `lmx_bars.standalone` (measure, the key, time and clefs in force, then the bar's own tokens). Every
  other bar is as before. The checkpoint says so in "variant", which `Reader` exposes, so that
  inference resets the state at system starts too. With --reset-keeps-time (variant "key-clefs") the
  time stays in the prefix and out of the target, as it is not printed at a system start;
- --only first|later|state (design 11, a signature specialist): train and validate on the bars that open a
  system (record field "first" true), on all the others only, or on the bars that set the state: those
  that open a system or whose LMX holds a key, time or clef (`lmx_bars.sets_state`); recorded in the
  variant as "only". The reset applies to the bars that open a system only, the others train as ordinary bars;
- --repeat ROOT:N (repeatable): a dataset's training bars are repeated so that they occur N times in all, to weigh a
  small set of the kind of page the model is to read (real scans) against a large one. ROOT is one of the datasets
  given, as on the command line; it applies after --only and before --oversample, validation is untouched, and the
  checkpoint says so in "variant" ("repeat");
- the validation loss on a random 1600 validation bars (--loss-bars; the same ones at every check),
  not the narrowest;
- --module PATH (a design's model.py: experiments/PLAN.md, "The model contract", e.g. experiments/05_hires/model.py): the
  network is the module's `build(vocabulary, **config)` from its CONFIG (--d-model and --layers replace the CONFIG's
  d_model and layers if it has them; --pretrained starts the encoder from ImageNet's weights, as for the baseline), and the
  crops are made ready as the module says (`make_prepare(**config)` or `prepare(image, mask)`), after the augmentation,
  for training and validation bars alike. The checkpoint keeps the module's path from the repository root under "module" and
  the config, so `Reader` builds the same network and prepares crops the same way; --init from such a checkpoint uses its
  module. Without --module (and an --init without one) nothing is different from the baseline.

  python experiments/common/train.py data/bars_lmx [data/bars_pdmx ...] --out runs/recognizer.pt [--steps 100000]
  python experiments/common/train.py data/bars_lmx --out runs/reset.pt --init runs/full.pt --system-reset --lr 1e-4 \
      --patience 4 --stop-on "note F1" --check 1000 --sample 512
  python experiments/common/train.py data/bars_asap2 data/bars_scans --repeat data/bars_scans:3 --out runs/scans.pt
  python experiments/common/train.py data/bars_lmx --module experiments/05_hires/model.py --pretrained --out runs/hires.pt
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transcriber.bar_recognizer import (EOS, MAX_TOKENS, PAD, ROOT, BarRecognizer, Vocabulary, batch,  # noqa: E402
                                         load_module, prepare, preparer)
from transcriber.lmx_accuracy import Tally, score_bar, table  # noqa: E402
from transcriber.lmx_bars import State, reset_state, reset_tokens, sets_state, true_states  # noqa: E402
from transcriber.page import SPACE  # noqa: E402


def records(root: Path, split: str) -> list[dict]:
    """The split's bars of one dataset, each knowing its root (`root` / `image` opens the crop)."""
    with (root / "bars.jsonl").open(encoding="utf-8") as f:
        return [{**r, "root": str(root)} for r in map(json.loads, f) if r["split"] == split]


def gather(roots: list[str], split: str) -> list[dict]:
    """The split's bars of every dataset that exists; one that does not is skipped, with a warning."""
    rows = []
    for root in roots:
        if (Path(root) / "bars.jsonl").exists():
            rows += records(Path(root), split)
        else:
            print(f"warning: {root} has no bars.jsonl, skipped", flush=True)
    return rows


def starts_of(rows: list[dict]) -> list[State]:
    """The state each bar truly starts from, dataset by dataset (pieces may share names across them)."""
    out: list[State | None] = [None] * len(rows)
    for root in dict.fromkeys(r["root"] for r in rows):
        mine = [k for k, r in enumerate(rows) if r["root"] == root]
        for k, state in zip(mine, true_states([rows[k] for k in mine])):
            out[k] = state
    return out


def only(rows: list[dict], which: str) -> list[dict]:
    """The bars that open a system ("first"), all the others ("later"), or those that set the state
    ("state": the first and those whose LMX holds a key, time or clef); every bar if `which` is empty."""
    if not which:
        return rows
    if which == "state":
        return [r for r in rows if r.get("first") or sets_state(r["lmx"].split())]
    return [r for r in rows if bool(r.get("first")) == (which == "first")]


def _has_time(tokens: list[str]) -> bool:
    return any(a == "time" and b.startswith("beats:") and c.startswith("beat-type:")
               for a, b, c in zip(tokens, tokens[1:], tokens[2:]))


def _has_key(tokens: list[str]) -> bool:
    return any(t.startswith("key:") for t in tokens)


OVERSAMPLE = {"first_time": lambda t: _has_time(t), "first_key": lambda t: _has_key(t),
              "first_change": lambda t: _has_time(t) or _has_key(t)}


def oversample(rows: list[dict], specs: list[str]) -> tuple[list[dict], list[tuple[str, int, int]]]:
    """Repeat the bars that open a system (not a piece's first bar) and print a time or key: each spec
    "<which>:<times>" (first_time, first_key, first_change) adds the matches so that they occur `times` times
    in all. Returns the rows and, per spec, (spec, matches, times). A bar matching two specs is repeated for each."""
    out = list(rows)
    done = []
    for spec in specs:
        which, _, times = spec.partition(":")
        if which not in OVERSAMPLE or not times.isdigit() or int(times) < 1:
            raise ValueError(f"bad --oversample {spec!r}: want <{'|'.join(OVERSAMPLE)}>:<times>")
        hit = [r for r in rows if r.get("first") and r.get("bar", 0) > 0 and OVERSAMPLE[which](r["lmx"].split())]
        out += hit * (int(times) - 1)
        done.append((spec, len(hit), int(times)))
    return out, done


def repeat(rows: list[dict], specs: list[str], roots: list[str]) -> tuple[list[dict], list[tuple[str, int, int]]]:
    """Repeat a dataset's bars: each spec "<root>:<times>" (a root among `roots`, compared as resolved paths) adds
    that root's bars so that they occur `times` times in all. Returns the rows and, per spec, (spec, bars, times)."""
    known = {Path(root).resolve() for root in roots}
    where = {name: Path(name).resolve() for name in dict.fromkeys(r["root"] for r in rows)}
    out = list(rows)
    done = []
    for spec in specs:
        root, _, times = spec.rpartition(":")
        if not root or not times.isdigit() or int(times) < 1:
            raise ValueError(f"bad --repeat {spec!r}: want <data root>:<times>, times at least 1")
        target = Path(root).resolve()
        if target not in known:
            raise ValueError(f"bad --repeat {spec!r}: {root} is not one of the datasets to train on ({', '.join(roots)})")
        hit = [r for r in rows if where[r["root"]] == target]
        out += hit * (int(times) - 1)
        done.append((spec, len(hit), int(times)))
    return out, done


def reset(row: dict, state: State, system_reset: bool | str) -> tuple[State, list[str]]:
    """The state a bar is read from and the tokens it is to be read as. With a system reset, a bar that
    opens a system is read from its ties alone and writes the signatures it shows; with "key-clefs"
    the time stays in the state and only the key and clefs are written."""
    tokens = row["lmx"].split()
    if system_reset and row.get("first"):
        return reset_state(state, system_reset), reset_tokens(tokens, state, system_reset)
    return state, tokens


def scanned(image: np.ndarray, mask: list[int], rng: random.Random) -> tuple[np.ndarray, list[int]]:
    """A clean crop made to look scanned, and its mask moved with it."""
    import cv2

    ink = 1 - image.astype(np.float32) / 255
    s = rng.uniform(0.85, 1.15)
    if abs(s - 1) > 0.01:
        ink = cv2.resize(ink, (max(1, round(ink.shape[1] * s)), max(1, round(ink.shape[0] * s))),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        mask = [round(v * s) for v in mask]
    r = rng.random()
    if r < 0.25:
        ink = cv2.dilate(ink, np.ones((2, 2), np.uint8))
    elif r < 0.35:
        ink = cv2.erode(ink, np.ones((1, 2) if rng.random() < 0.5 else (2, 1), np.uint8))
    if rng.random() < 0.6:
        ink = cv2.GaussianBlur(ink, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < 0.5:
        ink = ink + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0.02, 0.15), ink.shape).astype(np.float32)
    if rng.random() < 0.3:
        ink = (ink > rng.uniform(0.3, 0.6)).astype(np.float32)
    if rng.random() < 0.2:
        speck = np.random.default_rng(rng.randrange(1 << 30)).random(ink.shape) < rng.uniform(0.0005, 0.003)
        ink = np.maximum(ink, speck.astype(np.float32))
    if rng.random() < 0.3:
        ink = ink * rng.uniform(0.6, 1.0) + rng.uniform(0.0, 0.15)
    return (255 * (1 - np.clip(ink, 0, 1))).astype(np.uint8), mask


def design_path(path: str) -> str:
    """A design's model.py as a checkpoint keeps it: from the repository root (absolute if it lies outside)."""
    file = Path(path)
    if not file.is_absolute():
        file = Path.cwd() / file if (Path.cwd() / file).exists() else ROOT / file
    file = file.resolve()
    if not file.is_file():
        raise FileNotFoundError(f"no model module at {path}")
    try:
        return file.relative_to(ROOT).as_posix()
    except ValueError:
        return file.as_posix()


@functools.lru_cache(maxsize=None)
def preparation(module: str, config: str):
    """How a crop is made ready: the baseline's `prepare`, or the design's (`bar_recognizer.preparer`). The key is
    two strings, the module's path and the config as JSON, so that a data-loading worker (a process of its own, on
    Windows started afresh) finds the function again from them: a module loaded by its path cannot be pickled."""
    return preparer(load_module(module), json.loads(config)) if module else prepare


class Bars(torch.utils.data.Dataset):
    """Bars to draw. Each data-loading worker gets a copy of this, so it holds only what drawing a
    bar needs, the token ids packed small; the bars' records are kept only to score validation bars.
    With a `module` (a design's model.py, see `design_path`) and its `config`, crops are made ready as it says."""

    def __init__(self, rows: list[dict], vocabulary: Vocabulary, train: bool, system_reset: bool | str = False,
                 module: str = "", config: dict | None = None):
        self.train, self.system_reset = train, system_reset
        self.module, self.config = module, json.dumps(config or {}, sort_keys=True)
        self.items = []
        self.rows, self.states = [], []  # the validation bars kept, and the states they truly start from
        for r, state in zip(rows, starts_of(rows)):
            read_from, tokens = reset(r, state, system_reset)
            prefix = vocabulary.prefix(read_from)
            ids = prefix + vocabulary.encode(tokens) + [EOS]
            if len(ids) <= MAX_TOKENS + 1:
                self.items.append((str(Path(r["root"]) / r["image"]), r["mask"], np.array(ids, np.uint16), len(prefix)))
                if not train:
                    self.rows.append(r)
                    self.states.append(state)

    def __len__(self) -> int:
        return len(self.items)

    def width(self, k: int) -> int:
        return self.items[k][1][2] + round(2 * SPACE)  # the mask's right edge and the margin after it

    def __getitem__(self, k: int):
        name, mask, ids, start = self.items[k]
        image = np.asarray(Image.open(name).convert("L"))
        if self.train:
            image, mask = scanned(image, mask, random.Random())
        return preparation(self.module, self.config)(image, mask), ids, start


def collate(items):
    crops, ids, starts = zip(*items)
    images, padding = batch(list(crops))
    tokens = torch.full((len(ids), max(map(len, ids))), PAD, dtype=torch.long)
    for k, i in enumerate(ids):
        tokens[k, :len(i)] = torch.from_numpy(i.astype(np.int64))
    inputs, labels = tokens[:, :-1], tokens[:, 1:].clone()
    labels[labels == PAD] = -100
    for k, s in enumerate(starts):
        labels[k, :s - 1] = -100  # the state is given, not learned: only the bar's tokens, from <bos> on
    return images, padding, inputs, labels


class SimilarWidths(torch.utils.data.Sampler):
    """Batches of bars of about the same width, the batches in random order, for ever."""

    def __init__(self, data: Bars, size: int, seed: int):
        self.data, self.size, self.rng = data, size, random.Random(seed)

    def __iter__(self):
        while True:
            order = list(range(len(self.data)))
            self.rng.shuffle(order)
            batches = []
            for k in range(0, len(order), 64 * self.size):
                chunk = sorted(order[k:k + 64 * self.size], key=self.data.width)
                batches += [chunk[j:j + self.size] for j in range(0, len(chunk), self.size)]
            self.rng.shuffle(batches)
            yield from batches


def check(model, data: Bars, sample: int, device, size: int, loss_bars: int = 1600) -> tuple[float, Tally]:
    """The validation bars' loss, and a sample of them read and scored."""
    model.eval()
    losses = []
    with torch.no_grad():
        order = sorted(random.Random(0).sample(range(len(data)), min(loss_bars, len(data))), key=data.width)
        for k in range(0, len(order), size):
            images, padding, inputs, labels = collate([data[i] for i in order[k:k + size]])
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(images.to(device), padding.to(device), inputs.to(device))
            losses.append(torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), labels.to(device).flatten(),
                                                            ignore_index=-100).item())
        tally = Tally()
        picked = sorted(random.Random(0).sample(range(len(data)), min(sample, len(data))), key=data.width)
        for k in range(0, len(picked), size):
            items = [data[i] for i in picked[k:k + size]]
            images, padding = batch([c for c, _, _ in items])
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                read = model.read(images.to(device), padding.to(device), [ids[:start].tolist() for _, ids, start in items])
            for i, ids in zip(picked[k:k + size], read):
                row, state = data.rows[i], data.states[i]
                read_from, tokens = reset(row, state, data.system_reset)
                tally.add(score_bar(" ".join(data.vocabulary.decode(ids)), " ".join(tokens), state,
                                    read_from=read_from if read_from != state else None), row["style"])
    model.train()
    return float(np.mean(losses)), tally


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+", help="bar datasets (tools/bar_dataset.py): each a folder with bars.jsonl and the crops")
    ap.add_argument("--val-data", nargs="*", default=None,
                    help="datasets whose validation bars are checked against (default: the first one's)")
    ap.add_argument("--system-reset", action="store_true",
                    help="a bar that opens a system is read from its ties alone and writes its key, time and clefs")
    ap.add_argument("--reset-keeps-time", action="store_true",
                    help="with --system-reset: the time stays carried over (it is not printed at a system start); "
                         "only the key and clefs are reset and written (variant \"key-clefs\")")
    ap.add_argument("--only", choices=["first", "later", "state"], default="",
                    help="train and validate on only the bars that open a system (first), only the others (later), or those that "
                         "set the state: open a system or hold a key, time or clef (state)")
    ap.add_argument("--oversample", action="append", default=[], metavar="SPEC",
                    help="repeat training bars that open a system (bar > 0) and print a time (first_time), a key (first_key) "
                         "or either (first_change): SPEC is <which>:<times>; repeatable; applied after --only; validation untouched")
    ap.add_argument("--repeat", action="append", default=[], metavar="ROOT:N",
                    help="repeat the training bars of one of the datasets so that they occur N times in all; repeatable; "
                         "applied after --only and before --oversample; validation untouched")
    ap.add_argument("--out", required=True, help="the checkpoint to write")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--module", default="",
                    help="a design's model.py (experiments/PLAN.md, 'The model contract'): its network and its crop "
                         "preparation replace the baseline's; the checkpoint keeps its path")
    ap.add_argument("--d-model", type=int, default=None, help="default 256, or the module's CONFIG")
    ap.add_argument("--layers", type=int, default=None, help="default 4, or the module's CONFIG")
    ap.add_argument("--pretrained", action="store_true", help="start the encoder from ImageNet's ResNet-18")
    # Each worker is a process of its own, and importing PyTorch alone takes over a gigabyte in each
    ap.add_argument("--workers", type=int, default=6, help="data-loading processes")
    ap.add_argument("--prefetch", type=int, default=2, help="batches each worker keeps ready")
    # PyTorch keeps what it has used, and on Windows a card that fills up spills into the computer's memory
    # and slows to a crawl: anything else on the card (the desktop, another network) would tip it over
    ap.add_argument("--gpu-memory", type=float, default=0.85, help="the share of the card training may hold")
    ap.add_argument("--check", type=int, default=5000, help="score the validation bars and save every so many steps")
    ap.add_argument("--sample", type=int, default=256, help="validation bars read at each check")
    ap.add_argument("--loss-bars", type=int, default=1600, help="random validation bars the loss is taken on")
    ap.add_argument("--limit", type=int, default=0, help="train on only so many bars, to try it")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init", default="", help="go on from this trained recognizer's weights")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop when the validation loss has not improved for so many checks, keeping the best")
    ap.add_argument("--stop-on", default="loss",
                    help="what --patience watches: the validation loss, or a score of the bars read, such as 'state right'")
    ap.add_argument("--guard", default="", help="another bar dataset whose validation bars are read at each check too")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.init, map_location="cpu") if args.init else None
    vocabulary = Vocabulary(saved["vocabulary"]) if saved else Vocabulary()
    if args.reset_keeps_time and not args.system_reset:
        ap.error("--reset-keeps-time needs --system-reset")
    # A design with a network of its own: --module, or the one the --init checkpoint was trained with
    module = ""
    if args.module:
        try:
            module = design_path(args.module)
        except FileNotFoundError as e:
            ap.error(str(e))
    if saved and saved.get("module"):
        if module and module != saved["module"]:
            ap.error(f"--init {args.init} was trained with the module {saved['module']}, not {module}")
        module = saved["module"]
    design_config = None
    if module:
        design = load_module(module)
        design_config = {**design.CONFIG, **(saved["config"] if saved else {})}
        for key, value in (("d_model", args.d_model), ("layers", args.layers)):
            if value is not None and not saved:
                if key in design.CONFIG:
                    design_config[key] = value
                else:
                    print(f"warning: --{key.replace('_', '-')} {value} ignored, {module} does not have it in its CONFIG", flush=True)
        # (what the network is built with: --pretrained decides, as below, not the CONFIG's "pretrained")
        print(f"design {module}: { {**design_config, 'pretrained': bool(args.pretrained and not saved)} }", flush=True)
    system_reset = "key-clefs" if args.system_reset and args.reset_keeps_time else args.system_reset
    variant = {**(saved.get("variant", {}) if saved else {}), "system_reset": system_reset}
    if args.only:
        variant["only"] = args.only
    else:
        variant.pop("only", None)
    train_rows = only(gather(args.data, "train"), args.only)
    if args.only:
        print(f"--only {args.only}: {len(train_rows)} training bars remain after the filter", flush=True)
    if args.repeat:
        try:
            train_rows, done = repeat(train_rows, args.repeat, args.data)
        except ValueError as e:
            ap.error(str(e))
        for spec, n, times in done:
            print(f"--repeat {spec}: {n} bars repeated {times} times", flush=True)
        variant["repeat"] = list(args.repeat)
    else:
        variant.pop("repeat", None)
    if args.oversample:
        try:
            train_rows, done = oversample(train_rows, args.oversample)
        except ValueError as e:
            ap.error(str(e))
        for spec, n, times in done:
            print(f"--oversample {spec}: {n} bars repeated {times} times", flush=True)
        variant["oversample"] = list(args.oversample)
    else:
        variant.pop("oversample", None)
    if args.limit:
        train_rows = train_rows[:args.limit]
    val_roots = args.data[:1] if args.val_data is None else args.val_data
    train = Bars(train_rows, vocabulary, True, system_reset, module, design_config)
    val_rows = only(gather(val_roots, "val"), args.only)
    if args.only:
        print(f"--only {args.only}: {len(val_rows)} validation bars remain after the filter", flush=True)
    val = Bars(val_rows, vocabulary, False, system_reset, module, design_config)
    train.vocabulary = val.vocabulary = vocabulary
    print(f"{len(train)} bars to train on, {len(val)} to check against; {len(vocabulary)} tokens", flush=True)
    guard = None
    if args.guard:
        guard = Bars(only(gather([args.guard], "val"), args.only), vocabulary, False, system_reset, module, design_config)
        guard.vocabulary = vocabulary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory)
    if module:
        # As for the baseline, --pretrained decides (the CONFIG's "pretrained" is only what build() makes without being told)
        model = design.build(len(vocabulary), **{**design_config, "pretrained": args.pretrained and not saved})
        config = {**design_config, "pretrained": False}  # as for the baseline: the checkpoint's weights are the ones to use
    else:
        config = saved["config"] if saved else {"d_model": 256 if args.d_model is None else args.d_model,
                                                "layers": 4 if args.layers is None else args.layers, "pretrained": False}
        model = BarRecognizer(len(vocabulary), **{**config, "pretrained": args.pretrained and not saved})
    if saved:
        model.load_state_dict(saved["model"])
    model = model.to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, on {device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    shape = lambda s: min(1.0, s / args.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, shape)
    loader = torch.utils.data.DataLoader(train, batch_sampler=SimilarWidths(train, args.batch, args.seed),
                                         collate_fn=collate, num_workers=args.workers,
                                         persistent_workers=args.workers > 0,
                                         prefetch_factor=args.prefetch if args.workers else None)
    model.train()
    started, running, best, since = time.time(), None, None, 0
    for step, (images, padding, inputs, labels) in enumerate(loader, 1):
        images, padding, inputs, labels = (t.to(device, non_blocking=True) for t in (images, padding, inputs, labels))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(images, padding, inputs)
        loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100,
                                                 label_smoothing=0.1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
        if step % 100 == 0:
            print(f"step {step} loss {running:.3f} lr {sched.get_last_lr()[0]:.2e} "
                  f"({(time.time() - started) / step:.2f}s a step)", flush=True)
        if step % args.check == 0 or step == args.steps:
            val_loss, tally = check(model, val, args.sample, device, args.batch, args.loss_bars)
            tallies = {f"step {step}": tally}
            if guard is not None:
                tallies[f"  {Path(args.guard).name}"] = check(model, guard, args.sample, device, args.batch, args.loss_bars)[1]
            score = (-val_loss,) if args.stop_on == "loss" else (tally.summary()[args.stop_on], -val_loss)
            improved = best is None or score > best
            if improved or not args.patience:
                torch.save({"config": config, "model": model.state_dict(), "vocabulary": vocabulary.tokens, "step": step,
                            "variant": variant, **({"module": module} if module else {})}, out)
            best, since = (score, 0) if improved else (best, since + 1)
            print(f"step {step}: validation loss {val_loss:.3f}{' (best, kept)' if improved and args.patience else ''}; "
                  f"{tally.bars} bars read:\n" + table(tallies, groups=False), flush=True)
            if args.patience and since >= args.patience:
                kept = f"validation loss {-best[0]:.3f}" if args.stop_on == "loss" else f"{args.stop_on} {best[0]:.1%}"
                print(f"stopped: no better for {since} checks; kept step {step - since * args.check} ({kept})", flush=True)
                break
        if step >= args.steps:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
