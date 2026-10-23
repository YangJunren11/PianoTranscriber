"""The bar counter's network and the code that finds its peaks, copied unchanged from
`training/omr_ml.py` of github.com/YangJunren11/Bar-counter-for-PDF-Piano-Scores (Apache License 2.0,
see LICENSE and NOTICE here). The weights, `models/barcounter.pt`, are under their own licence
(`models/LICENSE.md`).

A small fully convolutional network that marks where every bar on a page ends, one peak per bar,
at the barline halfway down its system. Needs PyTorch and SciPy.
"""

from __future__ import annotations

import numpy as np

STRIDE = 4  # the heatmap's cell, in pixels


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


def peaks(prob: np.ndarray, threshold: float, strength: bool = False) -> np.ndarray:
    """Local maxima above threshold, one per bar: a maximum over three cells across and nine down,
    as one barline's peak can spread down a tall system. With strength, each peak's height too."""
    from scipy import ndimage
    top = ndimage.maximum_filter(prob, size=(9, 3))
    ys, xs = np.nonzero((prob == top) & (prob >= threshold))
    if strength:
        return prob[ys, xs]
    return np.stack([xs, ys], 1) * STRIDE + STRIDE / 2 if len(xs) else np.zeros((0, 2))

