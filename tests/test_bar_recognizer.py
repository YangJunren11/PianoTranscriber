"""The bar recognizer's parts, on a tiny network: crops made ready and batched, a bar learned and read
back, a checkpoint read back as a reader."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from transcriber.bar_recognizer import (EOS, MAX_HEIGHT, MAX_WIDTH, PAD, BarRecognizer, Reader,  # noqa: E402
                                         Vocabulary, batch, prepare)
from transcriber.lmx_bars import State  # noqa: E402
from test_lmx_accuracy import BAR, CONTEXT  # noqa: E402

TINY = {"d_model": 64, "layers": 2, "heads": 4, "feedforward": 128, "dropout": 0.0}


def crop() -> tuple[np.ndarray, list[int]]:
    image = np.full((120, 240), 255, np.uint8)
    for y in range(30, 75, 9):  # a staff
        image[y, 10:230] = 0
    image[30:67, 200] = 0  # a barline
    return image, [18, 9, 200, 111]


def test_a_crop_and_its_mask_are_two_channels_ink_and_bar() -> None:
    image, mask = crop()
    x = prepare(image, mask)
    assert x.shape == (2, 120, 240)
    assert x[0, 30, 50] == 1 and x[0, 31, 50] == 0  # ink is 1, paper 0
    assert x[1, 50, 100] == 1 and x[1, 5, 5] == 0 and x[1, 50, 210] == 0


def test_a_crop_bigger_than_the_largest_is_shrunk_to_fit() -> None:
    x = prepare(np.full((800, 4000), 255, np.uint8), [0, 0, 4000, 800])
    assert x.shape[1] <= MAX_HEIGHT and x.shape[2] <= MAX_WIDTH and x.shape[2] / x.shape[1] == pytest.approx(5, rel=0.01)


def test_crops_of_any_size_batch_together_with_their_padding_marked() -> None:
    images, padding = batch([torch.ones(2, 100, 200), torch.ones(2, 60, 330)])
    assert images.shape == (2, 2, 104, 336) and padding.shape == (2, 13, 42)
    assert not padding[0, :, :25].any() and padding[0, :, 25:].all()
    assert not padding[1, :8].any() and padding[1, 8:].all()
    assert images[0, :, 100:].abs().sum() == 0  # padded with paper


def test_a_bar_learned_by_heart_is_read_back_token_for_token() -> None:
    """Training and reading must agree on everything (positions, masks, where the bar starts): a
    network taught one bar must read it back exactly, from the state it was taught with."""
    torch.manual_seed(0)
    vocabulary = Vocabulary()
    model = BarRecognizer(len(vocabulary), **TINY)
    images, padding = batch([prepare(*crop())])
    prefix = vocabulary.prefix(State.parse(CONTEXT))
    ids = torch.tensor([prefix + vocabulary.encode(BAR.split()) + [EOS]])
    labels = ids[:, 1:].clone()
    labels[:, :len(prefix) - 1] = -100
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(300):
        loss = torch.nn.functional.cross_entropy(model(images, padding, ids[:, :-1]).flatten(0, 1), labels.flatten(),
                                                 ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    read = model.read(images, padding, [prefix])[0]
    assert vocabulary.decode(read) == BAR.split()


def test_reading_stops_at_the_most_tokens_allowed() -> None:
    model = BarRecognizer(len(Vocabulary()), **TINY).eval()
    images, padding = batch([prepare(*crop()), prepare(*crop())])
    prefix = Vocabulary().prefix(State())
    read = model.read(images, padding, [prefix, prefix], max_tokens=len(prefix) + 4)
    assert all(len(r) <= 4 and PAD not in r for r in read)


def test_a_checkpoint_reads_back_as_a_reader(tmp_path) -> None:
    vocabulary = Vocabulary()
    model = BarRecognizer(len(vocabulary), **TINY)
    torch.save({"config": TINY, "model": model.state_dict(), "vocabulary": vocabulary.tokens}, tmp_path / "r.pt")
    reader = Reader(tmp_path / "r.pt", device="cpu")
    bars = reader.read([crop(), crop()], [State.parse(CONTEXT), State()])
    assert len(bars) == 2 and all(isinstance(t, str) for bar in bars for t in bar)
    assert len(reader.read_score([crop(), crop()], State.parse(CONTEXT))) == 2
