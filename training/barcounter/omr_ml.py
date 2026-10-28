"""A learned bar counter: a small fully convolutional network that marks where every bar on a page
ends, one peak per bar, at the barline halfway down its system. The bars on a page are the peaks.

Barlines are easy to see and hard to tell from stems, beams and the lines that open systems by
rules alone, and a system's staves hard to group by rules on real scans (training/count_bars.py
finds the staves on 91% of AudioLabs' real pages, and the bars on 13%). The network learns both
from pages whose bars are known: MuseScore's engravings in five house styles (training/omr_synth.py)
and AudioLabs v2's real scans, annotated by hand.

Every page is first scaled so a staff space is 9 pixels, measured from the page itself (the
commonest black run down a column plus the commonest white run is one staff space), so a page
reaches the network at the same size whatever its edition, scan or resolution.

  python training/omr_ml.py prepare                 # data/omr_ml/{pages/, index.json}
  python training/omr_ml.py train [--steps 12000]   # data/omr_ml/model.pt
  python training/omr_ml.py evaluate [--model ...]  # counts on the held-out pages
  python training/omr_ml.py export --model data/omr_ml/model_v2.pt   # model/BarCounter.mlpackage
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
# The prepared pages; OMR_ML_WORK for another set of them (read again by each worker that loads pages)
WORK = Path(os.environ.get("OMR_ML_WORK", DATA / "omr_ml"))
SPACE = 9.0  # pixels per staff space, as the network sees a page
STRIDE = 4  # the heatmap's cell, in pixels
CROP_H, CROP_W = 640, 1344

# Real scans held out for testing, by work: AudioLabs' Beethoven sonatas (the only solo piano there)
# and the second half of Winterreise
AUDIOLABS_TEST = ("Beethoven_", "Schubert_D911-13", "Schubert_D911-14", "Schubert_D911-15", "Schubert_D911-16",
                  "Schubert_D911-17", "Schubert_D911-18", "Schubert_D911-19", "Schubert_D911-20",
                  "Schubert_D911-21", "Schubert_D911-22", "Schubert_D911-23", "Schubert_D911-24")


# ---------------------------------------------------------------------------------------------
# Pages


def gray(path: Path) -> np.ndarray:
    """A page as 8-bit grey on white; MuseScore's PNGs are black on transparent."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[-1])
        im = bg
    return np.asarray(im.convert("L"))


def staff_space(page: np.ndarray) -> float | None:
    """The distance between staff lines, from runs down the page's columns: staff lines are the
    commonest black runs, and the paper between them the commonest white runs."""
    ink = page < 160
    cols = ink[:, :: max(1, ink.shape[1] // 300)]
    # A black run and the white run after it: from one staff line's top to the next line's top
    pairs = np.zeros(160, np.int64)
    for c in cols.T:
        edges = np.flatnonzero(np.diff(np.concatenate([[0], c.astype(np.int8), [0]])))
        if len(edges) < 6:
            continue
        starts = edges[::2]  # where each black run starts
        steps = np.diff(starts)
        pairs += np.bincount(np.minimum(steps, 159), minlength=160)[:160]
    pairs[:3] = pairs[159] = 0
    if pairs.sum() < 50:
        return None
    mode = int(np.argmax(pairs))
    # To a fraction of a pixel: the steps around the commonest, weighted by how common
    lo, hi = max(mode - 1, 3), min(mode + 2, 159)
    return float(np.average(np.arange(lo, hi), weights=pairs[lo:hi] + 1e-9))


def snap(ink: np.ndarray, boxes: list[list[float]], reach: float = 5, need: float = 0.8) -> list[list[float]]:
    """Each bar's right edge moved onto its barline. MuseScore's box for a bar that ends a system
    takes in any courtesy key or time signature after the line, and ends past it: the line is the
    column nearest the edge, up to five spaces before it, dark down most of the bar's height.
    AudioLabs' boxes, drawn by hand, stop a pixel or three short; and a vocal score's barlines
    break between the voices and the piano, so there the line is looked for within a space either
    side, dark down half the height."""
    out = []
    for l, t, r, b in boxes:
        top, bottom = int(max(t, 0)), int(min(b, ink.shape[0]))
        x0, x1 = int(max(r - reach * SPACE, l + 1, 0)), int(min(r + SPACE, ink.shape[1]))
        if bottom - top < 4 or x1 <= x0:
            out.append([l, t, r, b])
            continue
        cover = ink[top:bottom, x0:x1].mean(0)
        lines = np.flatnonzero(cover >= need)
        if len(lines):
            xs = lines + x0
            r = float(xs[np.argmin(np.abs(xs - r))])
        out.append([l, t, r, b])
    return out


def prepare_one(item: dict) -> dict | None:
    """One page scaled to SPACE, saved as grey PNG, with its bars' ends in the scaled pixels."""
    page = gray(Path(item["path"]))
    space = staff_space(page)
    scale = SPACE / space if space else 1240 / page.shape[1] * SPACE / 10.3
    scale = float(np.clip(scale, 0.4, 2.5))
    h, w = round(page.shape[0] * scale), round(page.shape[1] * scale)
    out = WORK / "pages" / f"{item['key']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    small = Image.fromarray(page).resize((w, h), Image.BILINEAR if scale > 1 else Image.LANCZOS)
    small.save(out)
    boxes = [[v * scale for v in b] for b in item["boxes"]]
    if item["set"] == "train":
        # Test pages keep their boxes as given: only the count is scored there
        by_hand = item["key"].startswith("al_")
        boxes = snap(np.asarray(small) < 128, boxes, reach=1 if by_hand else 5, need=0.45 if by_hand else 0.8)
    return {"key": item["key"], "set": item["set"], "work": item["work"], "png": str(out.relative_to(WORK)),
            "size": [h, w], "scale": scale, "space": space, "boxes": boxes, "licence": item.get("licence", "")}


FONTS = ["/System/Library/Fonts/Supplemental/Times New Roman.ttf", "/System/Library/Fonts/Supplemental/Georgia.ttf",
         "/System/Library/Fonts/Supplemental/Baskerville.ttc", "/System/Library/Fonts/Supplemental/Didot.ttc",
         "/System/Library/Fonts/Palatino.ttc", "/System/Library/Fonts/Supplemental/Bodoni 72.ttc",
         "/System/Library/Fonts/Supplemental/Courier New.ttf", "/System/Library/Fonts/Supplemental/Charter.ttc"]
if not Path(FONTS[0]).exists():  # not a Mac: Windows' nearest
    FONTS = [str(Path("C:/Windows/Fonts") / f) for f in ("times.ttf", "georgia.ttf", "BASKVILL.TTF", "BOD_R.TTF",
                                                         "pala.ttf", "BOOKOS.TTF", "cour.ttf", "constan.ttf")]
WORDS = ("sonate allegro andante adagio presto menuetto trio rondo finale variationen serie werke pianoforte "
         "opus in dur moll erste ausgabe verlag leipzig preface the edition of this work was engraved from "
         "autograph manuscript first print revised by notes on performance tempo pedal fingering contents "
         "page table index no. breitkopf härtel peters schirmer henle complete works volume part book").split()


def text_page(k: int) -> dict:
    """A page with no music on it: a title page, a preface, a table of contents or a catalogue, as
    scans of complete editions are full of. Columns ruled with vertical lines, and frames, are what
    a bar counter could mistake for barlines."""
    from PIL import ImageDraw, ImageFont
    rng = random.Random(10_000 + k)
    w, h = rng.randint(1000, 1250), rng.randint(1350, 1700)
    im = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(im)
    font = lambda size: ImageFont.truetype(rng.choice(FONTS), size)
    y = rng.randint(60, 160)
    if rng.random() < 0.4:  # a frame round the page
        m = rng.randint(30, 70)
        for off in range(rng.choice([1, 2])):
            d.rectangle([m + 6 * off, m + 6 * off, w - m - 6 * off, h - m - 6 * off], outline=0, width=rng.choice([1, 2, 3]))
    if rng.random() < 0.6:  # a title
        size = rng.randint(40, 90)
        title = " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 3))).upper()
        d.text((w // 2 - len(title) * size // 4, y), title, font=font(size), fill=0)
        y += size + rng.randint(30, 80)
    cols = rng.choice([1, 1, 2, 3])
    rules = rng.random() < 0.5
    size = rng.randint(14, 26)
    col_w = (w - 160) // cols
    f = font(size)
    for c in range(cols):
        x = 80 + c * col_w
        yy = y
        while yy < h - 120:
            if rng.random() < 0.08:
                yy += size
                continue
            line = " ".join(rng.choice(WORDS) for _ in range(rng.randint(2, max(3, col_w // (size * 3)))))
            if rng.random() < 0.3:
                line = f"{rng.randint(1, 300):>4}  " + line
            d.text((x, yy), line, font=f, fill=0)
            yy += int(size * rng.uniform(1.2, 1.6))
        if rules and c:
            d.line([(x - 12, y), (x - 12, h - 120)], fill=0, width=rng.choice([1, 2]))
    if rules and rng.random() < 0.5:  # a number column ruled off on the left
        d.line([(130, y), (130, h - 120)], fill=0, width=1)
    out = WORK / "pages" / f"text_{k}.png"
    im.save(out)
    return {"key": f"text_{k}", "set": "train", "work": f"text_{k}", "png": str(out.relative_to(WORK)),
            "size": [h, w], "scale": 1.0, "space": None, "boxes": [], "licence": "generated"}


def gather() -> list[dict]:
    """Every page with bars known: synthetic (train), AudioLabs (train and test), results/omr (test)."""
    items = []
    for js in sorted((DATA / "omr_synth").glob("*/*.json")):
        if js.name.startswith("job-"):  # a batch MuseScore is still working through
            continue
        d = json.loads(js.read_text())
        for k, pg in enumerate(d["pages"]):
            items.append({"key": f"synth_{d['style']}_{d['source']['id']}_{k}", "set": "train",
                          "work": d["source"]["id"], "path": str(js.parent / pg["png"]), "boxes": pg["boxes"],
                          "licence": d["source"]["licence"]})
    for js in sorted((DATA / "audiolabs_v2").glob("*/json/*.json")):
        d = json.loads(js.read_text())
        work = js.parent.parent.name
        boxes = [[b["left"], b["top"], b["left"] + b["width"], b["top"] + b["height"]] for b in d["system_measures"]]
        items.append({"key": f"al_{js.stem}", "set": "test_real" if work.startswith(AUDIOLABS_TEST) else "train",
                      "work": work, "path": str(js.parent.parent / "img" / (js.stem + ".png")), "boxes": boxes,
                      "licence": "AudioLabs v2 (mixed)"})
    import xml.etree.ElementTree as ET
    for truth in sorted((ROOT / "results" / "omr").glob("*/truth.json")):
        t = json.loads(truth.read_text())
        # At MuseScore's default export resolution the .mpos units are 96 to a pixel at 150 dpi
        pages: dict[int, list] = {}
        for e in ET.parse(truth.parent / "score.mpos").getroot().iter("element"):
            x, y, w, h = (float(e.get(k)) / 96 for k in ("x", "y", "sx", "sy"))
            pages.setdefault(int(e.get("page")), []).append([x, y, x + w, y + h])
        for variant in ("clean", "scan"):
            for k, png in enumerate(sorted((truth.parent / variant).glob("p*.png"))):
                items.append({"key": f"omr_{variant}_{truth.parent.name}_{k}", "set": f"test_{variant}",
                              "work": truth.parent.name, "path": str(png), "boxes": pages.get(k, [])})
    return items


def prepare(workers: int) -> None:
    items = gather()
    with ProcessPoolExecutor(workers) as pool:
        rows = [r for r in pool.map(prepare_one, items, chunksize=8) if r]
        rows += list(pool.map(text_page, range(600), chunksize=8))
    (WORK / "index.json").write_text(json.dumps(rows))
    import collections
    print(collections.Counter(r["set"] for r in rows))


# ---------------------------------------------------------------------------------------------
# Targets and augmentation


def ends(boxes: list[list[float]]) -> np.ndarray:
    """(x, y) of each bar's end: its right edge, halfway down its system."""
    if not boxes:
        return np.zeros((0, 2), np.float32)
    b = np.asarray(boxes, np.float32)
    return np.stack([b[:, 2], (b[:, 1] + b[:, 3]) / 2], 1)


def systems(boxes: list[list[float]]) -> list[tuple[float, float]]:
    """The vertical extent of each system: bars sharing a top."""
    spans: list[list[float]] = []
    for _, t, _, b in sorted(boxes, key=lambda b: b[1]):
        if spans and abs(t - spans[-1][0]) < 4 * SPACE:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([t, b])
    return [(a, b) for a, b in spans]


def heatmap(points: np.ndarray, h: int, w: int) -> np.ndarray:
    hm = np.zeros((h // STRIDE, w // STRIDE), np.float32)
    ys, xs = np.mgrid[0:hm.shape[0], 0:hm.shape[1]]
    for x, y in points / STRIDE:
        if not (0 <= x < hm.shape[1] and 0 <= y < hm.shape[0]):
            continue
        x0, x1, y0, y1 = int(max(x - 4, 0)), int(min(x + 5, hm.shape[1])), int(max(y - 7, 0)), int(min(y + 8, hm.shape[0]))
        g = np.exp(-((xs[y0:y1, x0:x1] - x) ** 2 / (2 * 0.8 ** 2) + (ys[y0:y1, x0:x1] - y) ** 2 / (2 * 2.0 ** 2)))
        hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g)
        cy, cx = int(round(y)), int(round(x))
        if 0 <= cy < hm.shape[0] and 0 <= cx < hm.shape[1]:
            hm[cy, cx] = 1.0
    return hm


def degrade(ink: np.ndarray, rng: random.Random) -> np.ndarray:
    """Make a clean page look like a scan: blur, noise, bilevel thresholds, thicker or thinner
    strokes, speckle, grey paper. ink is 0 (paper) to 1 (black)."""
    import cv2
    r = rng.random()
    if r < 0.3:
        ink = cv2.dilate(ink, np.ones((rng.choice([2, 3]),) * 2, np.uint8))
    elif r < 0.4:
        ink = cv2.erode(ink, np.ones((1, 2) if rng.random() < 0.5 else (2, 1), np.uint8))
    if rng.random() < 0.7:
        ink = cv2.GaussianBlur(ink, (0, 0), rng.uniform(0.4, 1.3))
    if rng.random() < 0.6:
        ink = ink + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0.03, 0.2), ink.shape).astype(np.float32)
    if rng.random() < 0.4:
        ink = (ink > rng.uniform(0.3, 0.65)).astype(np.float32)
    if rng.random() < 0.25:
        g = np.random.default_rng(rng.randrange(1 << 30))
        speck = g.random(ink.shape) < rng.uniform(0.0005, 0.004)
        ink = np.maximum(ink, cv2.dilate(speck.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(np.float32))
    if rng.random() < 0.3:
        ink = ink * rng.uniform(0.6, 1.0) + rng.uniform(0.0, 0.15)
    return np.clip(ink, 0, 1)


def sample(row: dict, rng: random.Random, train: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A crop of a page as ink (1 is black), its heatmap and the weight of each heatmap cell. A
    system cut by the crop's top or bottom is weighed zero: whether its middle is in view is not
    something the crop can show."""
    import cv2
    page = np.asarray(Image.open(WORK / row["png"]).convert("L"), np.float32)
    ink = 1.0 - page / 255.0
    pts = ends(row["boxes"])
    spans = systems(row["boxes"])
    if train:
        # A little bigger or smaller than the scaled page, and turned a little
        s = rng.uniform(0.85, 1.2)
        a = rng.uniform(-1.2, 1.2)
        h, w = ink.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), a, s)
        nw, nh = int(w * s), int(h * s)
        m[0, 2] += (nw - w) / 2
        m[1, 2] += (nh - h) / 2
        ink = cv2.warpAffine(ink, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
        if len(pts):
            pts = np.c_[pts, np.ones(len(pts))] @ m.T
        spans = [((t - h / 2) * s + nh / 2, (b - h / 2) * s + nh / 2) for t, b in spans]
        if row["key"].startswith(("synth_", "omr_clean", "text_")) or rng.random() < 0.3:
            ink = degrade(ink.astype(np.float32), rng)
    h, w = ink.shape
    y0 = rng.randrange(0, max(1, h - CROP_H + 1)) if train else 0
    x0 = rng.randrange(0, max(1, w - CROP_W + 1)) if train else 0
    ch, cw = (CROP_H, CROP_W) if train else (math.ceil(h / 32) * 32, math.ceil(w / 32) * 32)
    crop = np.zeros((ch, cw), np.float32)
    part = ink[y0:y0 + ch, x0:x0 + cw]
    crop[:part.shape[0], :part.shape[1]] = part
    pts = pts - [x0, y0] if len(pts) else pts
    hm = heatmap(pts, ch, cw)
    weight = np.ones_like(hm)
    for t, b in spans:
        t, b = t - y0, b - y0
        if (t < 0 < b) or (t < ch < b):
            weight[max(int(t // STRIDE) - 2, 0):int(b // STRIDE) + 3] = 0
    return crop, hm, weight


# ---------------------------------------------------------------------------------------------
# The network


def network():
    import torch
    from torch import nn

    def conv(cin, cout, stride=1, k=(3, 3)):
        pad = (k[0] // 2, k[1] // 2)
        return nn.Sequential(nn.Conv2d(cin, cout, k, stride, pad, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            c = [16, 24, 48, 64, 96, 128]
            self.stem = conv(1, c[0])
            self.d1 = nn.Sequential(conv(c[0], c[1], 2), conv(c[1], c[1]))  # 1/2
            self.d2 = nn.Sequential(conv(c[1], c[2], 2), conv(c[2], c[2]))  # 1/4
            self.d3 = nn.Sequential(conv(c[2], c[3], 2), conv(c[3], c[3]))  # 1/8
            self.d4 = nn.Sequential(conv(c[3], c[4], 2), conv(c[4], c[4]))  # 1/16
            self.d5 = nn.Sequential(conv(c[4], c[5], 2), conv(c[5], c[5]))  # 1/32
            # Tall kernels: a system of four staves is some 300 pixels high, and a barline is known
            # by where it starts and stops
            self.tall = nn.Sequential(conv(c[5], c[5], k=(9, 1)), conv(c[5], c[5], k=(9, 1)), conv(c[5], c[5]))
            self.u4 = conv(c[5] + c[4], c[4])
            self.u3 = conv(c[4] + c[3], c[3])
            self.u2 = nn.Sequential(conv(c[3] + c[2], c[2]), conv(c[2], c[2], k=(7, 1)))
            self.head = nn.Conv2d(c[2], 1, 1)
            nn.init.constant_(self.head.bias, -4.0)

        def forward(self, x):
            # Pages are padded to a multiple of 32, so each level is exactly twice the one below: a
            # fixed factor, not a size read off the tensor, which Core ML's converter cannot follow
            up = lambda a, b: torch.cat([nn.functional.interpolate(a, scale_factor=2.0, mode="nearest"), b], 1)
            s0 = self.stem(x)
            s1 = self.d1(s0)
            s2 = self.d2(s1)
            s3 = self.d3(s2)
            s4 = self.d4(s3)
            s5 = self.tall(self.d5(s4))
            y = self.u4(up(s5, s4))
            y = self.u3(up(y, s3))
            y = self.u2(up(y, s2))
            return self.head(y)

    return Net()


def device():
    """An NVIDIA GPU, a Mac's GPU, or the CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def focal(logits, target, weight):
    """CenterNet's loss: peaks pulled up, the rest pushed down less near a peak."""
    import torch
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = (target >= 0.999).float() * weight
    neg = (1 - (target >= 0.999).float()) * weight
    loss = -(pos * (1 - p) ** 2 * torch.log(p) + neg * (1 - target) ** 4 * p ** 2 * torch.log(1 - p))
    return loss.sum() / pos.sum().clamp(min=1)


class Pages:
    def __init__(self, rows, seed):
        self.rows, self.seed = rows, seed

    def __len__(self):
        return 10 ** 7

    def __getitem__(self, i):
        import torch
        rng = random.Random(self.seed * 1_000_003 + i)
        row = self.rows[rng.randrange(len(self.rows))]
        crop, hm, weight = sample(row, rng)
        return torch.from_numpy(crop)[None], torch.from_numpy(hm)[None], torch.from_numpy(weight)[None]


def train(args) -> None:
    import torch
    rows = [r for r in json.loads((WORK / "index.json").read_text()) if r["set"] == "train"]
    if args.no_real:
        rows = [r for r in rows if not r["key"].startswith("al_")]
    real = [r for r in rows if r["key"].startswith("al_")]
    synth = [r for r in rows if not r["key"].startswith("al_")]
    # Real scans are few; drawn a third of the time
    mix = synth + real * max(1, len(synth) // (2 * max(len(real), 1))) if real else synth
    print(f"train: {len(synth)} synthetic pages, {len(real)} real", flush=True)
    dev = device()
    net = network().to(dev)
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M parameters", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    loader = torch.utils.data.DataLoader(Pages(mix, args.seed), batch_size=args.batch, num_workers=args.workers,
                                         persistent_workers=args.workers > 0,
                                         prefetch_factor=4 if args.workers > 0 else None)
    net.train()
    t0, running = time.time(), 0.0
    for step, (x, y, w) in enumerate(loader, 1):
        x, y, w = x.to(dev), y.to(dev), w.to(dev)
        loss = focal(net(x), y, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 100 == 0:
            print(f"step {step} loss {running:.3f} ({(time.time() - t0) / step:.2f}s a step)", flush=True)
        if step % 2000 == 0 or step == args.steps:
            torch.save(net.state_dict(), args.model)
        if step >= args.steps:
            break


# ---------------------------------------------------------------------------------------------
# Counting


def peaks(prob: np.ndarray, threshold: float, strength: bool = False) -> np.ndarray:
    """Local maxima above threshold, one per bar: a maximum over three cells across and nine down,
    as one barline's peak can spread down a tall system. With strength, each peak's height too."""
    from scipy import ndimage
    top = ndimage.maximum_filter(prob, size=(9, 3))
    ys, xs = np.nonzero((prob == top) & (prob >= threshold))
    if strength:
        return prob[ys, xs]
    return np.stack([xs, ys], 1) * STRIDE + STRIDE / 2 if len(xs) else np.zeros((0, 2))


def reconcile(heights: list[np.ndarray], threshold: float, total: int) -> list[int]:
    """Each page's count, made to add up to the bars the score is known to have: the peaks nearest
    the threshold, on whichever side, are the ones most likely wrong, so the strongest peaks under
    it are taken when bars are missing, and the weakest over it dropped when there are too many."""
    counts = [int((h >= threshold).sum()) for h in heights]
    short = total - sum(counts)
    if short > 0:
        under = sorted(((v, k) for k, h in enumerate(heights) for v in h if v < threshold), reverse=True)
        for _, k in under[:short]:
            counts[k] += 1
    elif short < 0:
        over = sorted((v, k) for k, h in enumerate(heights) for v in h if v >= threshold)
        for _, k in over[:-short]:
            counts[k] -= 1
    return counts


def predict(net, dev, row: dict) -> np.ndarray:
    import torch
    crop, _, _ = sample(row, random.Random(0), train=False)
    with torch.no_grad():
        logits = net(torch.from_numpy(crop)[None, None].to(dev))
    return torch.sigmoid(logits)[0, 0].cpu().numpy()


def evaluate(args) -> None:
    import torch
    rows = [r for r in json.loads((WORK / "index.json").read_text()) if r["set"].startswith("test")]
    if args.only:
        rows = [r for r in rows if args.only in r["key"]]
    dev = device()
    net = network().to(dev)
    net.load_state_dict(torch.load(args.model, map_location=dev))
    net.eval()
    heights = {r["key"]: peaks(predict(net, dev, r), 0.05, strength=True) for r in rows}
    out = {}
    for threshold in args.thresholds:
        out[threshold] = {r["key"]: int((heights[r["key"]] >= threshold).sum()) for r in rows}
    import collections
    for threshold, res in out.items():
        print(f"threshold {threshold}")
        by = collections.defaultdict(list)
        for r in rows:
            group = r["set"] + (" beethoven" if "Beethoven" in r["work"] else " schubert" if "Schubert" in r["work"] else "")
            by[group].append((res[r["key"]], len(r["boxes"]), r))
        for g, v in sorted(by.items()):
            exact = sum(a == b for a, b, _ in v)
            err = sum(abs(a - b) for a, b, _ in v) / max(1, sum(b for _, b, _ in v))
            line = f"  {g:24} pages {len(v):4}  exact {exact / len(v):6.1%}  bar error rate {err:.2%}"
            if g.startswith(("test_clean", "test_scan")):
                works = collections.defaultdict(list)
                for a, b, r in v:
                    works[r["work"]].append((int(r["key"].rsplit("_", 1)[1]), a, b, r["key"]))
                right = total = whole = flagged = silent = fixed = fixed_whole = 0
                for pages in works.values():
                    pages.sort()
                    true = np.cumsum([0] + [b for _, _, b, _ in pages[:-1]])
                    pred = np.cumsum([0] + [a for _, a, _, _ in pages[:-1]])
                    right += int((pred == true).sum())
                    total += len(pages)
                    ok = bool((pred == true).all())
                    whole += ok
                    adds_up = sum(a for _, a, _, _ in pages) == sum(b for _, _, b, _ in pages)
                    flagged += (not ok) and (not adds_up)
                    silent += (not ok) and adds_up
                    # Made to add up to the score's known total
                    counts = reconcile([heights[k] for *_, k in pages], threshold, sum(b for _, _, b, _ in pages))
                    pred = np.cumsum([0] + counts[:-1])
                    fixed += int((pred == true).sum())
                    fixed_whole += bool((pred == true).all())
                line += (f"  page starts {right / total:6.1%}  whole works {whole}/{len(works)}"
                         f" (wrong: {flagged} flagged by the total, {silent} not)"
                         f"  reconciled: starts {fixed / total:6.1%}, whole {fixed_whole}/{len(works)}")
            print(line)
    (WORK / "evaluation.json").write_text(json.dumps({str(k): v for k, v in out.items()}))


# ---------------------------------------------------------------------------------------------
# Counting in bars


def drawn_back(row: dict, f: float) -> np.ndarray:
    """A prepared page as ink, at the size it was rendered (150 dpi) times f."""
    im = Image.open(WORK / row["png"]).convert("L")
    s = f / row["scale"]
    return 1 - np.asarray(im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR), np.float32) / 255


def turned(scan: dict, clean: dict) -> float:
    """How far a page made to look scanned was turned, in degrees as scipy.ndimage.rotate turns a
    page (training/omr_testset.py): the angle at which its clean page, so turned, lies best over it.
    Each was scaled on its own, so both are drawn back at the size they were rendered, at half of it,
    and blurred a little for the fit to change smoothly with the angle."""
    from scipy import ndimage
    a, b = (ndimage.gaussian_filter(drawn_back(r, 0.5), 1.0) for r in (clean, scan))
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    fit = lambda t: float((ndimage.rotate(a, t, reshape=False, order=1) * b).sum())
    best = max(np.arange(-1.4, 1.401, 0.05), key=fit)
    return float(max(np.arange(best - 0.04, best + 0.041, 0.01), key=fit))


def turns(rows: list[dict]) -> dict[str, float]:
    """How far each page made to look scanned was turned (`turned`), kept beside the pages in
    angles.json, as it takes a second a page to find."""
    path = WORK / "angles.json"
    known = json.loads(path.read_text()) if path.exists() else {}
    by_key = {r["key"]: r for r in rows}
    todo = [r for r in rows if r["key"].startswith("omr_scan_") and r["key"] not in known]
    for k, r in enumerate(todo):
        known[r["key"]] = turned(r, by_key[r["key"].replace("omr_scan_", "omr_clean_", 1)])
        if k % 50 == 49 or k == len(todo) - 1:
            path.write_text(json.dumps(known, indent=0))
            print(f"how far the scanned pages were turned: {k + 1} of {len(todo)}", flush=True)
    return known


def bar_ends(row: dict, rows: dict[str, dict], angles: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Where each bar on a test page ends: on its barline, halfway down its system; and half its
    system's height. MuseScore's box for a system's last bar takes in the courtesy key and time
    signature after it, and a change to seven sharps in a new time is wider than the five spaces
    training looks back (`snap`): here the barline is looked for twelve spaces back, within the box.
    A page made to look scanned was turned as it was scanned, and its barlines no longer stand
    straight enough to snap to: its bars are found on the clean page and turned as the page was
    (`turns`)."""
    if row["key"].startswith("omr_scan_"):
        clean = rows[row["key"].replace("omr_scan_", "omr_clean_", 1)]
        pts, half = bar_ends(clean, rows, angles)
        pts, half = pts / clean["scale"], half / clean["scale"]  # on the page as drawn, at 150 dpi
        h, w = (v / clean["scale"] for v in clean["size"])
        angle = angles[row["key"]]
        # As scipy.ndimage.rotate turns a page, about its middle
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        dx, dy = pts[:, 0] - w / 2, pts[:, 1] - h / 2
        pts = np.stack([w / 2 + c * dx + s * dy, h / 2 - s * dx + c * dy], 1)
        return pts * row["scale"], half * row["scale"]
    ink = np.asarray(Image.open(WORK / row["png"]).convert("L")) < 128
    by_hand = row["key"].startswith("al_")
    boxes = snap(ink, row["boxes"], reach=1 if by_hand else 12, need=0.45 if by_hand else 0.8)
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    return ends(boxes), (b[:, 3] - b[:, 1]) / 2


def match(found: np.ndarray, truth: np.ndarray, half: np.ndarray, reach: float) -> int:
    """How many of the peaks found are bars that are there: each peak paired with one bar's end at
    most, and only within `reach` of it across and within its system down."""
    if not len(found) or not len(truth):
        return 0
    from scipy.optimize import linear_sum_assignment
    dx = np.abs(found[:, None, 0] - truth[None, :, 0])
    dy = np.abs(found[:, None, 1] - truth[None, :, 1])
    tall = np.maximum(half, 2 * SPACE)[None, :]
    near = (dx <= reach) & (dy <= tall)
    rows, cols = linear_sum_assignment(np.where(near, dx / reach + dy / tall, 1e6))
    return int(near[rows, cols].sum())


def bars(args) -> None:
    """Networks compared bar by bar on the held-out pages. A bar is found when a peak stands on its
    barline (within --reach spaces across, and within its system down), and missed when none does;
    a peak with no bar of its own is a bar made up. Bars wrong are those missed and made up, over
    the bars there are: a page can be counted right with one bar missed and another made up, and
    counted bar by bar that is two bars wrong. Bars miscounted is the looser measure, each page's
    count against its bars. Networks whose names differ only by a last _s<seed> are one network
    trained from several seeds, their bars summed and the spread of their seeds given."""
    import collections
    import re
    import torch
    rows = [r for r in json.loads((WORK / "index.json").read_text()) if r["set"].startswith("test")]
    by_key = {r["key"]: r for r in rows}
    group = lambda r: {"test_real": "AudioLabs held out, " + ("Beethoven" if "Beethoven" in r["work"] else "Schubert"),
                       "test_clean": "MuseScore, clean", "test_scan": "MuseScore, made to look scanned"}[r["set"]]
    # Every page: its group, how to give it to the network, where its bars end and half their systems' height
    angles = turns(rows)
    pages = {r["key"]: (group(r), lambda r=r: sample(r, random.Random(0), train=False)[0], *bar_ends(r, by_key, angles))
             for r in rows}
    # Pages engraved by Verovio, which no network here was trained on (PDF2MusicXML's tools/engrave.py):
    # each bar's barline, and each system's staff lines, on the page as drawn; scaled as an app scales a page
    for js in sorted(Path(args.engraved).glob("*/*.json")) if args.engraved else []:
        variant = "clean" if js.parent.name.endswith("-clean") else "made to look scanned"
        for pg in json.loads(js.read_text())["pages"]:
            png = js.parent / pg["png"]
            scale = page_scale(gray(png))
            ends_ = [(x, (s["top"] + s["bottom"]) / 2, (s["bottom"] - s["top"]) / 2) for s in pg["systems"] for x in s["bars"]]
            e = np.asarray(ends_, np.float32).reshape(-1, 3) * scale
            pages[f"verovio/{js.parent.name}/{pg['png']}"] = (f"Verovio, {variant}", lambda png=png: scaled_page(png)[0],
                                                              e[:, :2], e[:, 2])
    # The IMSLP scans counted by eye, where they have been drawn (results/real): their counts alone
    real_dir = ROOT / "results" / "real"
    real_truth = {w: t for w, t in json.loads((real_dir / "truth.json").read_text()).items()
                  if not w.startswith("_") and len(list((real_dir / w).glob("p*.png"))) == len(t)}
    dev = device()
    found: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    real_heights: dict[str, dict[str, list[np.ndarray]]] = {}
    for path in args.models:
        net = network().to(dev)
        net.load_state_dict(torch.load(path, map_location=dev))
        net.eval()
        found[Path(path).stem] = {}
        for key, (_, given, _, _) in pages.items():
            with torch.no_grad():
                prob = torch.sigmoid(net(torch.from_numpy(given())[None, None].to(dev)))[0, 0].cpu().numpy()
            found[Path(path).stem][key] = (peaks(prob, 0.05), peaks(prob, 0.05, strength=True))
        real_heights[Path(path).stem] = {w: [page_heights(net, dev, p) for p in sorted((real_dir / w).glob("p*.png"))]
                                         for w in real_truth}
        print(f"{path}: {len(pages)} pages, and {sum(map(len, real_truth.values()))} IMSLP pages", flush=True)

    def tally(name: str, threshold: float) -> dict[str, collections.Counter]:
        out = collections.defaultdict(collections.Counter)
        for key, (own, _, pts, half) in pages.items():
            where, heights = found[name][key]
            where = where[heights >= threshold] if len(where) else where
            hit = match(where, pts, half, args.reach * SPACE)
            for g in (own, "All held-out pages"):
                c = out[g]
                c["pages"] += 1
                c["bars"] += len(pts)
                c["missed"] += len(pts) - hit
                c["made up"] += len(where) - hit
                c["miscounted"] += abs(len(where) - len(pts))
                c["pages exact"] += len(where) == len(pts)
        return out

    configs = collections.defaultdict(list)
    for name in found:
        configs[re.sub(r"_s\d+$", "", name)].append(name)
    tallies = {name: tally(name, args.threshold) for name in found}
    groups = sorted({g for t in tallies.values() for g in t}, key=lambda g: (g.startswith("All"), g))
    rate = lambda c: (c["missed"] + c["made up"]) / max(c["bars"], 1)
    print(f"\nthreshold {args.threshold}, a bar found within {args.reach:g} spaces of its barline")
    print(f"{'':30} {'bars':>7} {'missed':>8} {'made up':>8} {'bars wrong':>11} {'seeds':>15} {'miscounted':>11} "
          f"{'pages exact':>14}")
    report = {}
    for g in groups:
        print(g)
        for config, names in configs.items():
            c = sum((tallies[n][g] for n in names), collections.Counter())
            per_seed = [rate(tallies[n][g]) for n in names]
            spread = f"{min(per_seed):.2%}-{max(per_seed):.2%}" if len(names) > 1 else ""
            label = config + (f" ({len(names)} seeds)" if len(names) > 1 else "")
            print(f"  {label:28} {c['bars']:>7} {c['missed']:>8} {c['made up']:>8} {rate(c):>11.2%} {spread:>15} "
                  f"{c['miscounted'] / max(c['bars'], 1):>11.2%} {c['pages exact']:>6}/{c['pages']:<7}")
            report.setdefault(g, {})[config] = {**c, "bars wrong": rate(c), "seeds": per_seed}
    if real_truth:
        print(f"IMSLP scans counted by eye ({', '.join(real_truth)}): counts only, so bars miscounted")
        for config, names in configs.items():
            per_seed, c = [], collections.Counter()
            for n in names:
                one = collections.Counter()
                for w, true in real_truth.items():
                    for h, t in zip(real_heights[n][w], true):
                        one.update({"pages": 1, "bars": t, "miscounted": abs(int((h >= args.threshold).sum()) - t),
                                    "pages exact": int((h >= args.threshold).sum()) == t})
                per_seed.append(one["miscounted"] / max(one["bars"], 1))
                c += one
            spread = f"{min(per_seed):.2%}-{max(per_seed):.2%}" if len(names) > 1 else ""
            label = config + (f" ({len(names)} seeds)" if len(names) > 1 else "")
            print(f"  {label:28} {c['bars']:>7} {'':>8} {'':>8} {'':>11} {spread:>15} "
                  f"{c['miscounted'] / max(c['bars'], 1):>11.2%} {c['pages exact']:>6}/{c['pages']:<7}")
            report.setdefault("IMSLP scans", {})[config] = {**c, "seeds": per_seed}
    print(f"\nbars wrong, all held-out pages, at other thresholds")
    for config, names in configs.items():
        line = []
        for threshold in args.thresholds:
            c = sum((tally(n, threshold)["All held-out pages"] for n in names), collections.Counter())
            line.append(f"{threshold:g}: {rate(c):.2%}")
        print(f"  {config:28} " + "   ".join(line))
    (WORK / "bars.json").write_text(json.dumps({"threshold": args.threshold, "reach": args.reach, "groups": report,
                                                "pages": {n: {k: [len(pages[k][2]), int((h >= args.threshold).sum())]
                                                              for k, (_, h) in f.items()} for n, f in found.items()}},
                                               indent=1))


def count(args) -> None:
    """Count the bars on each page image in a folder, and draw what was counted beside them."""
    import torch
    from PIL import ImageDraw
    dev = device()
    net = network().to(dev)
    net.load_state_dict(torch.load(args.model, map_location=dev))
    net.eval()
    folder = Path(args.pages)
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    counts = []
    for png in sorted(folder.glob("*.png")):
        page = gray(png)
        space = staff_space(page)
        scale = float(np.clip(SPACE / space if space else 1240 / page.shape[1] * SPACE / 10.3, 0.4, 2.5))
        small = np.asarray(Image.fromarray(page).resize((round(page.shape[1] * scale), round(page.shape[0] * scale)),
                                                        Image.BILINEAR if scale > 1 else Image.LANCZOS), np.float32)
        h, w = small.shape
        x = np.zeros((math.ceil(h / 32) * 32, math.ceil(w / 32) * 32), np.float32)
        x[:h, :w] = 1 - small / 255
        with torch.no_grad():
            prob = torch.sigmoid(net(torch.from_numpy(x)[None, None].to(dev)))[0, 0].cpu().numpy()
        found = peaks(prob, args.threshold)
        counts.append(len(found))
        print(f"{png.name}: {len(found)} (space {space and round(space, 1)})", flush=True)
        if out:
            im = Image.fromarray(small.astype(np.uint8)).convert("RGB")
            d = ImageDraw.Draw(im)
            for k, (px, py) in enumerate(sorted(found.tolist(), key=lambda p: (round(p[1] / 60), p[0]))):
                d.ellipse([px - 6, py - 6, px + 6, py + 6], outline="red", width=3)
                d.text((px + 7, py - 18), str(k + 1), fill="red")
            im.save(out / png.name)
    print(json.dumps({"pages": counts, "total": int(sum(counts)),
                      "starts": [int(v) for v in np.cumsum([0] + counts[:-1])]}))


def page_scale(page: np.ndarray) -> float:
    """How much a page is scaled for a staff space to be SPACE pixels, measured from the page."""
    space = staff_space(page)
    return float(np.clip(SPACE / space if space else 1240 / page.shape[1] * SPACE / 10.3, 0.4, 2.5))


def scaled_page(png: Path) -> tuple[np.ndarray, float]:
    """A page as the network takes it, ink 1 and paper 0, scaled (`page_scale`) and padded with paper
    to a multiple of 32 each way; and the scale."""
    page = gray(png)
    scale = page_scale(page)
    small = np.asarray(Image.fromarray(page).resize((round(page.shape[1] * scale), round(page.shape[0] * scale)),
                                                    Image.BILINEAR if scale > 1 else Image.LANCZOS), np.float32)
    h, w = small.shape
    x = np.zeros((math.ceil(h / 32) * 32, math.ceil(w / 32) * 32), np.float32)
    x[:h, :w] = 1 - small / 255
    return x, scale


def page_heights(net, dev, png: Path) -> np.ndarray:
    """Every peak on a page with its height, however faint, for thresholds to be tried after."""
    import torch
    x, _ = scaled_page(png)
    with torch.no_grad():
        prob = torch.sigmoid(net(torch.from_numpy(x)[None, None].to(dev)))[0, 0].cpu().numpy()
    return peaks(prob, 0.05, strength=True)


def real(args) -> None:
    """The real IMSLP scans whose pages were counted by eye (results/omr_real/truth.json): pages
    counted exactly; page starts, as counted and made to add up to the score's total; and how well
    doubt finds the wrong pages (a page is in doubt when a peak falls between the two bounds)."""
    import torch
    dev = device()
    net = network().to(dev)
    net.load_state_dict(torch.load(args.model, map_location=dev))
    net.eval()
    folder = ROOT / "results" / "real"
    truth = {k: v for k, v in json.loads((folder / "truth.json").read_text()).items() if not k.startswith("_")}
    heights = {w: [page_heights(net, dev, p) for p in sorted((folder / w).glob("p*.png"))] for w in truth}
    lo, hi = args.doubt
    for threshold in args.thresholds:
        exact = starts = fixed = n = wrong = flagged = caught = 0
        lines = []
        for w, true in truth.items():
            counts = [int((h >= threshold).sum()) for h in heights[w]]
            ok = [c == t for c, t in zip(counts, true)]
            doubt = [bool(((h >= lo) & (h < hi)).any()) for h in heights[w]]
            ts = np.cumsum([0] + true[:-1])
            starts += int((np.cumsum([0] + counts[:-1]) == ts).sum())
            rec = reconcile(heights[w], threshold, sum(true))
            fixed += int((np.cumsum([0] + rec[:-1]) == ts).sum())
            exact += sum(ok)
            n += len(true)
            wrong += ok.count(False)
            flagged += sum(doubt)
            caught += sum(d and not o for d, o in zip(doubt, ok))
            bad = [f"p{k + 1} {c}/{t}" for k, (c, t) in enumerate(zip(counts, true)) if c != t]
            lines.append(f"    {w:15} exact {sum(ok)}/{len(true)}  total {sum(counts)}/{sum(true)}  wrong: {', '.join(bad)}")
        print(f"threshold {threshold}: pages exact {exact}/{n} ({exact / n:.1%}); page starts {starts / n:.1%}, "
              f"made to add up {fixed / n:.1%}; in doubt {flagged} pages, catching {caught} of the {wrong} wrong")
        for line in lines:
            print(line)


def export(args) -> None:
    """The network as an app runs it: Core ML, half precision, pages of any size that is a
    multiple of 32, and the probability rather than the logit out, so the app only finds peaks.
    What an app needs to know to use it (the cut-off, the staff space) goes in its metadata."""
    import coremltools as ct
    import torch
    net = network()
    net.load_state_dict(torch.load(args.model, map_location="cpu"))
    net.eval()

    class Probability(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return torch.sigmoid(self.net(x))

    wrapped = Probability(net).eval()
    example = torch.zeros(1, 1, 1536, 1088)
    traced = torch.jit.trace(wrapped, example)
    size = lambda default: ct.RangeDim(lower_bound=64, upper_bound=4096, default=default)
    ml = ct.convert(traced, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=ct.target.iOS17,
                    inputs=[ct.TensorType(name="page", shape=(1, 1, size(1536), size(1088)))],
                    outputs=[ct.TensorType(name="bars")])
    ml.short_description = ("Where each bar on a page of music ends: one peak per bar, at the barline halfway "
                            "down its system, on a grid a quarter of the page's size.")
    ml.input_description["page"] = ("The page in grey, ink 1 and paper 0, scaled so a staff space is "
                                    f"{SPACE:g} pixels and padded with paper to a multiple of 32 each way.")
    ml.output_description["bars"] = "The probability that a bar ends at each cell of the grid."
    ml.user_defined_metadata.update({"space": f"{SPACE:g}", "stride": str(STRIDE), "threshold": str(args.threshold),
                                     "trained": Path(args.model).name})
    out = Path(args.out) if args.out else ROOT / "model" / "BarCounter.mlpackage"
    out.parent.mkdir(parents=True, exist_ok=True)
    ml.save(str(out))
    # The same page through both, at a size other than the traced one: a real scan if one is here
    found = sorted((ROOT / "results" / "real").glob("*/p*.png"))
    if not found:
        print(f"{out}: written (no page in results/real to check it against PyTorch)")
        return
    page = gray(found[len(found) // 2])
    scale = SPACE / staff_space(page)
    small = np.asarray(Image.fromarray(page).resize((round(page.shape[1] * scale), round(page.shape[0] * scale)),
                                                    Image.LANCZOS), np.float32)
    h, w = small.shape
    x = np.zeros((math.ceil(h / 32) * 32, math.ceil(w / 32) * 32), np.float32)
    x[:h, :w] = 1 - small / 255
    with torch.no_grad():
        want = wrapped(torch.from_numpy(x)[None, None]).numpy()[0, 0]
    got = ct.models.MLModel(str(out)).predict({"page": x[None, None]})["bars"][0, 0]
    print(f"{out}: {x.shape[1]}x{x.shape[0]} page, largest difference {np.abs(got - want).max():.4f}, "
          f"bars {int((peaks(want, args.threshold, True)).size)} in PyTorch, "
          f"{int((peaks(got.astype(np.float32), args.threshold, True)).size)} in Core ML")


def main() -> int:
    global WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "train", "evaluate", "bars", "count", "real", "export"])
    ap.add_argument("--models", nargs="+", default=[], help="bars: the networks to compare")
    ap.add_argument("--reach", type=float, default=2.0, help="bars: how near its barline a bar is found, in spaces")
    ap.add_argument("--engraved", default="", help="bars: pages engraved by Verovio too (PDF2MusicXML's tools/engrave.py)")
    ap.add_argument("--doubt", type=float, nargs=2, default=[0.2, 0.7])
    ap.add_argument("--pages", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default=str(WORK / "model.pt"))
    ap.add_argument("--no-real", action="store_true", help="train on synthetic pages only")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5])
    ap.add_argument("--only", default="")
    ap.add_argument("--data", default="", help="the folder holding the prepared index.json and pages/ (default data/omr_ml)")
    args = ap.parse_args()
    if args.data:
        WORK = Path(args.data).resolve()
        os.environ["OMR_ML_WORK"] = str(WORK)  # the workers read it again
        if args.model == str(DATA / "omr_ml" / "model.pt"):
            args.model = str(WORK / "model.pt")
    if args.command == "prepare":
        prepare(args.workers)
    elif args.command == "train":
        train(args)
    elif args.command == "bars":
        bars(args)
    elif args.command == "count":
        count(args)
    elif args.command == "real":
        real(args)
    elif args.command == "export":
        export(args)
    else:
        evaluate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
