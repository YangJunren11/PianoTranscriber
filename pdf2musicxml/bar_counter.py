"""The bar counter's network (github.com/YangJunren11/Bar-counter-for-PDF-Piano-Scores): where each bar
on a page ends, one peak per bar at its barline halfway down its system. The network and the code that
finds its peaks are the bar counter's (`pdf2musicxml/barcounter/net.py`, its weights `models/barcounter.pt`),
so a page is run as in its evaluation; the page is the one `pdf2musicxml.page` makes, scaled as
the bar counter scales it. Needs PyTorch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

THRESHOLD = 0.4  # the bar counter's: a peak at least this high is a bar
FLOOR = 0.05  # fainter peaks are kept too, to judge a page's count by


@dataclass
class Peak:
    x: float  # where a bar ends, on the page: its barline
    y: float  # halfway down its system
    height: float  # how sure the network is


class BarCounter:
    def __init__(self, weights: Path | str, threads: int | None = None):
        """The network with its weights (`models/barcounter.pt`), run on the CPU."""
        import torch

        from .barcounter import net

        if threads:
            torch.set_num_threads(threads)
        self.torch, self.omr_ml = torch, net
        self.net = net.network()
        self.net.load_state_dict(torch.load(Path(weights), map_location="cpu"))
        self.net.eval()

    def peaks(self, pixels: np.ndarray) -> list[Peak]:
        """Every peak on a page at least FLOOR high: grey, 0 black and 255 white, a staff space
        9 pixels (`page.Page.pixels`)."""
        h, w = pixels.shape
        x = np.zeros((math.ceil(h / 32) * 32, math.ceil(w / 32) * 32), np.float32)
        x[:h, :w] = 1 - pixels.astype(np.float32) / 255
        with self.torch.no_grad():
            prob = self.torch.sigmoid(self.net(self.torch.from_numpy(x)[None, None]))[0, 0].numpy()
        where = self.omr_ml.peaks(prob, FLOOR)
        heights = self.omr_ml.peaks(prob, FLOOR, strength=True)
        return [Peak(float(px), float(py), float(v)) for (px, py), v in zip(where, heights)]
