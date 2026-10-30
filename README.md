Hi.

# PianoTranscriber

A small model, ~11.8 million parameters and ~47MiB, that splits a music score pdf by bar and outputs LMX, by [Olimpic](https://github.com/ufal/olimpic-icdar24), which is then converted to MusicXML for general use.

It is designed to be lightweight while having comparable accuracy with other open-sourced models, but omits details such as dynamics and tempo. 
It is suitable for use when the full resolution is not necessary, for example in music tracking via chroma and online time warping methods, (e.g. [PageTurner](https://github.com/YangJunren11/Tracker-for-the-solo-piano) by me) and the bar recognition can be used in some app features as well, for example bar highlighting.

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
The models were trained on 26 October 2026, and the time signature is checked against how the bars fill it and corrected if they do not; `--no-resolve-time` turns this check off.

From Python, from the repository root:

```python
from pdf2musicxml import pdf_to_musicxml

pdf_to_musicxml("score.pdf", "score.musicxml", pages="1-3", json_path="score.json")
```

## Results

Note F1 of the bundled models on bars never seen in training, read as the pipeline runs (main model with the specialist). SER is the symbol error rate over whole systems, as the OLiMPiC paper computes it: the bars of a system joined, one edit distance over LMX tokens per system, pooled over the set:

| Test set | Bars | Note F1 | SER | Timed F1 |
|---|---:|---:|---:|---:|
| OLiMPiC scanned test set | 4,490 | 81.2% | 17.1% | 73.3% |
| Real scans of printed editions, Baroque to 20th century (our test set) | 4,951 | 81.8% | 24.0% | 73.7% |
| Engraved scores (our test set) | 4,065 | 95.0% | 8.1% | 93.1% |

On the complete OLiMPiC scanned test set under OLiMPiC's own protocol, all 1,493 systems with every system cut into bars by our bar counter and read, and a system the cutting fails on counted as entirely wrong, the SER is 23.4%. The table's row holds the 4,490 bars whose systems were cut into exactly the right number of bars.

## Examples
All scores below are in the public domain.

Bar splitting before going into the recognizer:

<img width="698" height="284" alt="image" src="https://github.com/user-attachments/assets/b29b6eba-257e-4368-9995-64148311a86f" />

Recognizer and its output LMX:

<img width="547" height="331" alt="image" src="https://github.com/user-attachments/assets/8e670a57-7998-41cf-a8bd-3cf6d6d57fda" />

## License

CC BY-NC-SA 4.0 (non-commercial, share alike): see [LICENSE](LICENSE). The code, the models and their weights are for non-commercial use only; the bar counter weights carry the same licence (`models/LICENSE.md`).
