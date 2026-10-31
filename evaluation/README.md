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

    python -I -X utf8 evaluation/compare/build_manifest.py

`build_manifest.py` writes the manifest `f1_models.py` reads, from the stored outputs of the earlier comparison (folder named by
the environment variable `COMPARISON_DIR`, the ASAP scores by `ASAP_DIR`; neither is distributed). It works in the current
directory (`pieces_new.json`, `runs/`, `manifest.json`).

    python -I -X utf8 evaluation/compare/f1_models.py selftest
    python -I -X utf8 evaluation/compare/f1_models.py run MANIFEST.json --out f1.json [--systems A B] [--pieces X Y] [--no-rests] [--apply-cut A] [--rename "OLD=NEW"]
    python -I -X utf8 evaluation/compare/f1_models.py merge BASE.json EXTRA.json [EXTRA.json ...] --out MERGED.json

`f1_models.py` is the music-level comparison: note, onset and timed F1 of each system's MusicXML as the system wrote it, read by
music21 for every system and for the gold alike (`score_direct.py`), with no LMX, MuseScore or normaliser in between. The whole
piece is the denominator: the events of a measure no bar was aligned to count as misses, those of a bar aligned to no measure as
false alarms (bars are aligned to measures by Needleman-Wunsch for every system, whatever its bar count). It also prints the
measures found (bars read against measures) and the measures read exactly, and the pooled figures of the sets "ten" (the
manifest's list `ten`), "four" (its list `engraved`) and "fourteen" (both), each list of piece ids possibly empty. Rests count as events unless
`--no-rests` is given. `--apply-cut A` scores system A, which read only the first pages, against the first `cut` true measures of
the manifest (its unread pages are then not misses); `--rename "OLD=NEW"` gives a row another name, so that the same system can
be run whole and cut, and `merge` adds the rows of such runs to one result. The manifest is the one `build_manifest.py` writes
(`truth`, and per system and piece `note_files` or `files`, `cut`) plus the two lists. This is the comparison used for the figures
against other OMR systems; `selftest` checks the scorer on made-up music and files.

Helper modules: `bar_dataset.py` (read MusicXML, find the piano part, linearize it) and `compare/score_direct.py` (reduced to the
functions used), both taken from the research code without changing what they compute.
