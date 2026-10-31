# Evaluation scripts

Scripts that score the transcriber on test sets and compare it with other OMR systems. Run them from the repository root
with the packages of `requirements.txt` installed (`python -I -X utf8 ...`; `-I` keeps stray folders off the import path).
They import only `pdf2musicxml`, `third_party/lmx`, the requirements and the sibling files here. The unit tests are in `tests/`
(`python -m pytest tests -q`).

## Bar-level evaluation of the recognizer

    python -I -X utf8 evaluation/evaluate_lmx.py DATA --split val --models models/main.pt --starts-model models/specialist.pt \
        --chained --by style --batch 16 --out preds.jsonl > evaluate_lmx.txt

Reads every bar of the split of a bar dataset (`DATA/bars.jsonl` and the crops beside it) with the recognizer, each bar
from the state the reading of the bar before left (`--chained`), the bars that open a system (and bars whose reading changes
the key, time or clef) by the specialist (`--starts-model`). Prints, per style or piece (`--by`), the token error rate, the
share of bars read exactly, note F1, pitch F1, timed F1, the share of bars whose state is carried on correctly, and the share
of readings that are valid LMX. `--out` writes every bar's scores as JSON lines; `--limit N` takes a random slice of N bars
(whole pieces with `--chained`) for a quick try.

    python -I -X utf8 evaluation/system_ser.py DATA --split val --preds preds.jsonl [--out result.json]

Symbol error rate per printed system, as the OLiMPiC paper scores it, from the dump written by `evaluate_lmx.py --out`: the
bars of a system joined (the first with its key and clefs in front), one Levenshtein distance over LMX tokens per system,
the edits summed and divided by the gold tokens. Run it on a dump of the whole split; on a `--limit` slice most bars are
missing from the dump and the figure is meaningless.

## Which test sets can be rebuilt

    python -I -X utf8 evaluation/systems_dataset.py olimpic OLIMPIC_DIR OUT --bar-counter models/barcounter.pt [--subset scanned] [--workers 8]
    python -I -X utf8 evaluation/systems_dataset.py grandstaff GRANDSTAFF_DIR OUT --bar-counter models/barcounter.pt

Builds a bar dataset (the format `evaluate_lmx.py` reads) from the systems of OLiMPiC or GrandStaff: each system image is cut
into bars by the bar counter and the bar cropper, and each bar is paired with the LMX of the measure in the same place. This
is the only public test set that can be rebuilt here: the OLiMPiC systems (download them yourself). The real-scan and the
engraved test sets are built from scores, scans and engravings that are not published, so their builders are not included;
`evaluate_lmx.py` and `system_ser.py` score any dataset in that format. Note that the bars are cut with the pipeline in this
repository, which can differ in detail from the one that cut the sets of the results table.

## Comparison with other OMR systems (`evaluation/compare/`)

The runners of the other systems (homr, oemer, Audiveris, SMT, Zeus) are not included; these scripts only read the MusicXML
(or LMX) files those systems wrote, listed in a manifest.

    python -I -X utf8 evaluation/compare/system_ser_models.py selftest
    python -I -X utf8 evaluation/compare/system_ser_models.py run MANIFEST.json --out results.json [--systems A B] [--pieces X Y] [--dump-systems DIR]

`system_ser_models.py` scores the MusicXML of any system per printed system: the file's bars as LMX (`omr_convert.py`
repairs what the Linearizer refuses), aligned to the gold measures (`align.py`), grouped by one common partition of the piece
into systems, and scored as `system_ser.py` does (SER, and SER without stem and beam tokens), plus bar-level SER and the note
F1 of `score_direct.py` (music21, no LMX). The manifest format is in the script's docstring. `selftest` checks it on made-up bars. MuseScore pads incomplete voices with invisible rests, so by default the scorer removes the hidden-rest groups (`print-object:no rest ...`, with the backup that only served them, and then any dangling backup/forward pair) from both the gold and the prediction before the Levenshtein distance. `--keep-hidden-rests` turns this off.

    python -I -X utf8 evaluation/compare/normalise_formal.py [--only SYSTEM ...]

Passes every MusicXML of the benchmark, the gold included, through MuseScore 4 (path in `MS`), so every system is written by
the same engraver before it is scored, then drops a restated, unchanged time signature; writes `formal/`, `formal_raw/`,
`formal/convert_log.json`, `manifest_formal.json` (normalised files only) and `manifest_formal2.json` in the current directory.
The second manifest adds `fallback_files` to each system entry and `fallback_truth` to each piece (the raw files): the scorer
then takes a bar from the normalised file only if the linearizer accepts it and its pitched-note count equals the raw bar's,
otherwise the raw bar, otherwise an empty bar; a gold measure is the normalised one if LMX can write it, else the raw one.
So no bar is scored empty because of MuseScore or the converter, and the result records how many bars fell back.

    python -I -X utf8 evaluation/compare/build_manifest.py
    python -I -X utf8 evaluation/compare/make_results.py results.json [manifest.json] > tables.md
    python -I -X utf8 evaluation/compare/verify_consistency.py results.json
    python -I -X utf8 evaluation/compare/verify_formal.py report|convdet
    python -I -X utf8 evaluation/compare/spotcheck.py DUMPDIR/NAME__PIECE.json INDEX [--width 40] [--max-rows 60]

`build_manifest.py` writes the manifest from the stored outputs of the earlier comparison (folder named by the environment
variable `COMPARISON_DIR`, the ASAP scores by `ASAP_DIR`; neither is distributed). `make_results.py` turns `results.json` into
markdown tables. `verify_consistency.py` checks the bar-level figures against those stored by the earlier comparison.
`verify_formal.py` checks the MuseScore-normalised benchmark against the raw one (counts conserved, gold, headline, ranking;
`convdet` converts three files twice and compares hashes). `spotcheck.py` prints one system's gold and predicted tokens side by
side with a plain dynamic-programming edit count as an independent check of the library's. The scripts that read or write a
benchmark's files (`build_manifest`, `normalise_formal`, `make_results`, `verify_*`) work in the current directory.

Helper modules: `bar_dataset.py` (read MusicXML, find the piano part, linearize it), `compare/omr_convert.py`,
`compare/align.py`, `compare/score_direct.py` (reduced to the functions used) and `compare/kern_to_musicxml.py` (Humdrum
**kern to MusicXML, for SMT's output), all taken from the research code without changing what they compute.
