# PianoTranscriber

A small model that splits a music score pdf by bar and outputs LMX, by [Olimpic](https://github.com/ufal/olimpic-icdar24), which is then converted to MusicXML for general use.

It is designed to be lightweight while having comparable accuracy with other open-sourced models, but omits details such as dynamics and tempo. 
It is suitable for use when the full resolution is not necessary, for example in music tracking via chroma and online time warping methods, (e.g. [PageTurner](https://github.com/YangJunren11/Tracker-for-the-solo-piano) by me, in Tracker-for-the-solo-piano) and the bar recognition can be used in some app features as well, for example bar highlighting.

## Install

Python 3.10 or newer.

```
python -m venv .venv
.venv\Scripts\activate        # Linux and macOS: source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch with CUDA is optional: the models run on the GPU if there is one and on the CPU if not. For a
CPU-only install, run `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
before the last command.

## Run

From the repository root:

```
python -m pdf2musicxml.transcribe score.pdf --out score.musicxml --json score.json
```

`--out` is the MusicXML to write; `--json` (optional) writes data about every bar read. To read only some
pages, add `--pages 1-3,7` (1-based; the default is all of them).

From Python, from the repository root:

```python
from pdf2musicxml import pdf_to_musicxml

pdf_to_musicxml("score.pdf", "score.musicxml", pages="1-3", json_path="score.json")
```
