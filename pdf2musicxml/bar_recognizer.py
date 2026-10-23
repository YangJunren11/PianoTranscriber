"""The bar recognizer: a bar's crop read as Linearized MusicXML (LMX), bar after bar through a score.

  the crop and its mask (the bar itself, as against its margins), as two channels
   -> ResNet-18 cut after its second stage: features at an eighth of the crop's size
   -> flattened into a sequence, each feature with its row and column added (sines, as a
      transformer's positions)
   -> a transformer decoder that attends to the features and writes the bar's LMX token by token,
      after the state the bar starts from: its clefs, key and time, and the notes tied over to it
      (`lmx_bars.State`), which the bar before it leaves

A crop keeps the pipeline's scale, a staff space 9 pixels, so every bar is read at the size it is
printed; one bigger than MAX_HEIGHT by MAX_WIDTH is shrunk to fit. Needs PyTorch and torchvision.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .lmx_bars import State, reset_state

SPECIALS = ["<pad>", "<bos>", "<eos>", "<state>", "<unk>"]
PAD, BOS, EOS, STATE, UNK = range(len(SPECIALS))
MAX_HEIGHT, MAX_WIDTH = 384, 1536  # pixels: a crop bigger than this is shrunk to fit
MAX_TOKENS = 768  # the state and the bar's tokens together
CHECK = 8  # cached reading looks whether every bar is done this often (steps)
STRIDE = 8  # the encoder's features, in pixels of the crop


class Vocabulary:
    """Every LMX token, after the few the recognizer needs of its own."""

    def __init__(self, tokens: list[str] | None = None):
        if tokens is None:
            from lmx.linearization.vocabulary import ALL_TOKENS

            tokens = SPECIALS + list(ALL_TOKENS)
        self.tokens = tokens
        self.index = {t: k for k, t in enumerate(tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.index.get(t, UNK) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        out = []
        for i in ids:
            if i == EOS:
                break
            if i >= len(SPECIALS):
                out.append(self.tokens[i])
        return out

    def prefix(self, state: State) -> list[int]:
        """What the decoder is given before the bar: the state it starts from."""
        return [STATE, *self.encode(state.tokens()), BOS]


def prepare(image: np.ndarray, mask: np.ndarray | list[int]) -> torch.Tensor:
    """A crop (grey, 0 black and 255 white) and its mask (an array, 1 over the bar, or the bar's
    rectangle: left, top, right, bottom) as the encoder takes them: two channels, ink and mask."""
    h, w = image.shape
    if not isinstance(mask, np.ndarray):
        left, top, right, bottom = mask
        mask = np.zeros((h, w), np.float32)
        mask[top:bottom, left:right] = 1
    x = torch.from_numpy(np.stack([1 - image.astype(np.float32) / 255, mask.astype(np.float32)]))
    scale = min(1.0, MAX_HEIGHT / h, MAX_WIDTH / w)
    if scale < 1:
        size = (max(1, round(h * scale)), max(1, round(w * scale)))
        x = nn.functional.interpolate(x[None], size=size, mode="area")[0]
    return x


def batch(crops: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Crops of any size as one batch, padded with paper to a multiple of STRIDE, and which of the
    encoder's features fall in the padding."""
    h = max(c.shape[1] for c in crops)
    w = max(c.shape[2] for c in crops)
    h, w = math.ceil(h / STRIDE) * STRIDE, math.ceil(w / STRIDE) * STRIDE
    images = torch.zeros(len(crops), 2, h, w)
    padding = torch.ones(len(crops), h // STRIDE, w // STRIDE, dtype=torch.bool)
    for k, c in enumerate(crops):
        images[k, :, :c.shape[1], :c.shape[2]] = c
        padding[k, :math.ceil(c.shape[1] / STRIDE), :math.ceil(c.shape[2] / STRIDE)] = False
    return images, padding


def places(d: int, h: int, w: int, device) -> torch.Tensor:
    """Each feature's row and column as sines, half the channels each: 1 x d x h x w."""
    def sines(n: int, k: int) -> torch.Tensor:
        pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
        freq = torch.exp(torch.arange(0, k, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / k))
        return torch.cat([torch.sin(pos * freq), torch.cos(pos * freq)], 1)

    half = d // 2
    rows = sines(h, half)[:, None, :].expand(h, w, half)
    cols = sines(w, d - half)[None, :, :].expand(h, w, d - half)
    return torch.cat([rows, cols], 2).permute(2, 0, 1)[None]


class BarRecognizer(nn.Module):
    def __init__(self, vocabulary: int, d_model: int = 256, layers: int = 4, heads: int = 8, feedforward: int = 1024,
                 dropout: float = 0.1, pretrained: bool = False):
        super().__init__()
        import torchvision

        resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        # Two channels in, ink and mask: ImageNet's colour filters averaged for the ink, none for the mask
        conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                conv1.weight.zero_()
                conv1.weight[:, 0] = resnet.conv1.weight.mean(1)
        # Stem (a quarter), first stage, second stage (an eighth): 128 channels
        self.encoder = nn.Sequential(conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2)
        self.project = nn.Conv2d(128, d_model, 1)
        self.embed = nn.Embedding(vocabulary, d_model, padding_idx=PAD)
        self.position = nn.Embedding(MAX_TOKENS + 2, d_model)
        layer = nn.TransformerDecoderLayer(d_model, heads, feedforward, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(d_model))
        self.out = nn.Linear(d_model, vocabulary)
        self.d_model = d_model

    def encode(self, images: torch.Tensor, padding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The features as a sequence, and which of them are padding."""
        f = self.project(self.encoder(images))
        f = f + places(self.d_model, f.shape[2], f.shape[3], f.device)
        return f.flatten(2).transpose(1, 2), padding.flatten(1)

    def decode(self, memory: torch.Tensor, memory_padding: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        n = tokens.shape[1]
        x = self.embed(tokens) + self.position(torch.arange(n, device=tokens.device))[None]
        causal = torch.ones(n, n, dtype=torch.bool, device=tokens.device).triu(1)  # True: not to be seen
        y = self.decoder(x, memory, tgt_mask=causal, tgt_is_causal=True, tgt_key_padding_mask=tokens == PAD,
                         memory_key_padding_mask=memory_padding)
        return self.out(y)

    def forward(self, images: torch.Tensor, padding: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        memory, memory_padding = self.encode(images, padding)
        return self.decode(memory, memory_padding, tokens)

    @torch.no_grad()
    def read_uncached(self, images: torch.Tensor, padding: torch.Tensor, prefixes: list[list[int]],
                      max_tokens: int = MAX_TOKENS) -> list[list[int]]:
        """Each bar's tokens, the likeliest one at a time after its prefix, until it ends: the whole
        decoder run over everything written so far at every step (the reference for `read`)."""
        memory, memory_padding = self.encode(images, padding)
        b, device = len(prefixes), images.device
        lengths = torch.tensor([len(p) for p in prefixes], device=device)
        start = lengths.clone()
        tokens = torch.full((b, max_tokens + 1), PAD, dtype=torch.long, device=device)
        for k, p in enumerate(prefixes):
            tokens[k, :len(p)] = torch.tensor(p, device=device)
        done = lengths >= max_tokens
        while not done.all():
            logits = self.decode(memory, memory_padding, tokens[:, :int(lengths.max())])
            next_ = logits[torch.arange(b, device=device), lengths - 1].argmax(-1)
            ending = ~done & (next_ == EOS)
            rows = torch.nonzero(~done & (next_ != EOS)).flatten()
            tokens[rows, lengths[rows]] = next_[rows]
            lengths[rows] += 1
            done |= ending | (lengths >= max_tokens)
        return [tokens[k, start[k]:lengths[k]].tolist() for k in range(b)]

    # -- cached decoding: the decoder layers re-run with the layers' own weights, keys and values kept --

    def _heads(self, attn: nn.MultiheadAttention, x: torch.Tensor, rows: slice) -> torch.Tensor:
        """Part of the attention's input projection (0 query, 1 key, 2 value) of x, as (batch, heads, n, head_dim)."""
        d = attn.embed_dim
        w = attn.in_proj_weight[rows]
        b = attn.in_proj_bias[rows] if attn.in_proj_bias is not None else None
        y = nn.functional.linear(x, w, b)
        return y.view(x.shape[0], x.shape[1], attn.num_heads, d // attn.num_heads).transpose(1, 2)

    def _attend(self, attn: nn.MultiheadAttention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                allowed: torch.Tensor) -> torch.Tensor:
        """Attention over k, v (batch, heads, keys, head_dim) where `allowed` (True), then the output projection."""
        y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        y = y.transpose(1, 2).reshape(q.shape[0], q.shape[2], attn.embed_dim)
        return attn.out_proj(y)

    def _layer(self, layer: nn.TransformerDecoderLayer, x: torch.Tensor, cache: list,
               at: torch.Tensor | None, cross: tuple[torch.Tensor, torch.Tensor], self_allowed: torch.Tensor,
               memory_allowed: torch.Tensor, width: int = 0) -> torch.Tensor:
        """One pre-norm decoder layer over x (batch, n, d). Its keys and values are written into `cache`
        at positions `at` (batch,) for one new token per sample, or (at None) at 0..n-1, the cache made `width` long."""
        sa = layer.self_attn
        h = layer.norm1(x)
        qkv = nn.functional.linear(h, sa.in_proj_weight, sa.in_proj_bias)  # query, key and value in one go
        q, k, v = qkv.view(x.shape[0], x.shape[1], 3, sa.num_heads, sa.embed_dim // sa.num_heads).permute(2, 0, 3, 1, 4)
        if at is None:  # the cache takes the dtype the layer computes in (autocast may make it half)
            for i, new in enumerate((k, v)):
                cache[i] = new.new_zeros(new.shape[0], new.shape[1], width, new.shape[3])
                cache[i][:, :, :new.shape[2]] = new
        else:
            rows = torch.arange(x.shape[0], device=x.device)
            cache[0][rows, :, at] = k[:, :, 0]
            cache[1][rows, :, at] = v[:, :, 0]
        if at is not None:
            k, v = cache
        x = x + self._attend(sa, q, k, v, self_allowed)
        ca = layer.multihead_attn
        q = self._heads(ca, layer.norm2(x), slice(0, ca.embed_dim))
        x = x + self._attend(ca, q, cross[0], cross[1], memory_allowed)
        return x + layer.linear2(layer.activation(layer.linear1(layer.norm3(x))))

    def _finish(self, x: torch.Tensor) -> torch.Tensor:
        if self.decoder.norm is not None:
            x = self.decoder.norm(x)
        return self.out(x)

    @torch.no_grad()
    def read(self, images: torch.Tensor, padding: torch.Tensor, prefixes: list[list[int]],
             max_tokens: int = MAX_TOKENS) -> list[list[int]]:
        """Each bar's tokens, the likeliest one at a time after its prefix, until it ends. Same result
        as `read_uncached`, with the decoder's keys and values kept instead of recomputed: the prefixes
        are run through in one pass, then one new token per bar at a time, at its own position."""
        memory, memory_padding = self.encode(images, padding)
        return self.read_memory(memory, memory_padding, prefixes, max_tokens)

    @torch.no_grad()
    def read_memory(self, memory: torch.Tensor, memory_padding: torch.Tensor, prefixes: list[list[int]],
                    max_tokens: int = MAX_TOKENS) -> list[list[int]]:
        """`read` from the encoder's features."""
        b, device = len(prefixes), memory.device
        layers = list(self.decoder.layers)
        lengths = torch.tensor([len(p) for p in prefixes], device=device)
        start = lengths.clone()
        width = max(max_tokens, int(lengths.max())) + 1
        tokens = torch.full((b, width), PAD, dtype=torch.long, device=device)
        for k, p in enumerate(prefixes):
            tokens[k, :len(p)] = torch.tensor(p, device=device)
        done = lengths >= max_tokens
        if done.all():
            return [[] for _ in range(b)]
        for layer in layers:
            assert layer.self_attn._qkv_same_embed_dim and layer.multihead_attn._qkv_same_embed_dim
        # Cross-attention keys and values, once
        memory_allowed = ~memory_padding[:, None, None, :]
        cross = []
        for layer in layers:
            ca = layer.multihead_attn
            cross.append((self._heads(ca, memory, slice(ca.embed_dim, 2 * ca.embed_dim)),
                          self._heads(ca, memory, slice(2 * ca.embed_dim, 3 * ca.embed_dim))))
        caches = [[None, None] for _ in layers]
        # Prefill: every prefix at once, causal, each bar's own length
        n = int(lengths.max())
        x = self.embed(tokens[:, :n]) + self.position(torch.arange(n, device=device))[None]
        idx = torch.arange(n, device=device)
        allowed = (idx[None, :] <= idx[:, None])[None, None] & (idx[None, :] < lengths[:, None])[:, None, None, :]
        for layer, cache, kv in zip(layers, caches, cross):
            x = self._layer(layer, x, cache, None, kv, allowed, memory_allowed, width)
        x = x[torch.arange(b, device=device), lengths - 1][:, None].contiguous()
        keys, rows = torch.arange(width, device=device), torch.arange(b, device=device)

        def advance() -> None:
            """The next token of every bar written and the one after it worked out: the whole step on tensors
            that stay where they are, so that it can be recorded once and replayed. A bar that is done
            writes nothing more and rewrites its last entry in the cache unchanged."""
            next_ = self._finish(x)[:, 0].argmax(-1)
            ending = next_ == EOS
            write = ~done & ~ending
            tokens[rows, lengths] = torch.where(write, next_, tokens[rows, lengths])
            lengths.add_(write)
            done.logical_or_(ending | (lengths >= max_tokens))
            at = lengths - 1
            h = self.embed(tokens[rows, at])[:, None] + self.position(at)[:, None]
            allowed = (keys[None, :] <= at[:, None])[:, None, None, :]
            for layer, cache, kv in zip(layers, caches, cross):
                h = self._layer(layer, h, cache, at, kv, allowed, memory_allowed)
            x.copy_(h)

        # On a GPU the step is recorded as a CUDA graph after two steps made as they are: what slows a
        # step down there is launching its hundred small kernels one by one, not the work in them
        graph, steps = None, 0
        while True:
            if graph is not None:
                graph.replay()
            else:
                advance()
                if device.type == "cuda" and steps == 1 and not torch.is_autocast_enabled():
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        advance()
                    steps += 1
                    graph.replay()
            steps += 1
            if steps % CHECK == 0 and bool(done.all()):
                break
        return [tokens[k, start[k]:lengths[k]].tolist() for k in range(b)]


class Reader:
    """A trained bar recognizer, reading bars."""

    def __init__(self, checkpoint: Path | str, device: str | None = None):
        saved = torch.load(checkpoint, map_location="cpu")
        self.vocabulary = Vocabulary(saved["vocabulary"])
        self.model = BarRecognizer(len(self.vocabulary), **saved["config"])
        self.model.load_state_dict(saved["model"])
        self.variant = saved.get("variant", {})  # how it was trained, where that changes how it reads: "system_reset"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()

    def read(self, crops: list[tuple[np.ndarray, np.ndarray | list[int]]], states: list[State]) -> list[list[str]]:
        """Bars each read from the state given for it: bars of different scores, or with their true
        states, read side by side."""
        images, padding = batch([prepare(image, mask) for image, mask in crops])
        ids = self.model.read(images.to(self.device), padding.to(self.device),
                              [self.vocabulary.prefix(s) for s in states])
        return [self.vocabulary.decode(i) for i in ids]

    def read_score(self, crops: list[tuple[np.ndarray, np.ndarray | list[int]]], state: State | None = None,
                   firsts: list[bool] | None = None) -> list[list[str]]:
        """A score's bars in order, each from the state the bar before it left. A recognizer trained with
        a system reset reads a bar that opens a system (`firsts[k]`) from the notes tied over to it
        alone: the key, time and clefs it prints are read, and carry on from there."""
        state = state or State()
        bars = []
        for k, crop in enumerate(crops):
            bar = self.read([crop], [self.reset(state, firsts and firsts[k])])[0]
            bars.append(bar)
            state = self.reset(state, firsts and firsts[k]).after(bar)
        return bars

    def read_apart(self, crops: list[tuple[np.ndarray, np.ndarray | list[int]]], states: list[State]) -> list[list[str]]:
        """As `read`, but each crop goes through the encoder on its own, as a batch of one: the encoder's
        features near a crop's edge depend on the paper padded around it in a batch, so the bars read
        are exactly those `read` gives for one crop at a time. The decoder still reads them together."""
        memories = []
        for image, mask in crops:
            images, padding = batch([prepare(image, mask)])
            with torch.no_grad():
                memories.append(self.model.encode(images.to(self.device), padding.to(self.device)))
        n = max(m.shape[1] for m, _ in memories)
        memory = torch.zeros(len(crops), n, memories[0][0].shape[2], device=self.device, dtype=memories[0][0].dtype)
        padding = torch.ones(len(crops), n, dtype=torch.bool, device=self.device)
        for k, (m, p) in enumerate(memories):
            memory[k, :m.shape[1]], padding[k, :m.shape[1]] = m[0], p[0]
        ids = self.model.read_memory(memory, padding, [self.vocabulary.prefix(s) for s in states])
        return [self.vocabulary.decode(i) for i in ids]

    def resets(self, first: bool | None) -> bool | str:
        """How a bar is read without its state's signatures, or False: it opens a system, and this
        recognizer was trained to read them from the page. True: key, time and clefs; "key-clefs":
        the key and clefs only (the time is not printed at a system start, so it carries on)."""
        return (self.variant.get("system_reset") or False) if first else False

    def reset(self, state: State, first: bool | None) -> State:
        """The state a bar is read from: the state left by the bar before, or for a bar that opens a
        system (by a recognizer trained for it) only the notes tied over to it."""
        return reset_state(state, self.resets(first))
