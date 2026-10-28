# Training commands

Run everything from the repository root. The prepared training sets come from the separate zip of training
sets; unpack it so the recognizer's data roots sit under `data/` and the bar counter's pages under
`data/omr_ml/` (or point `--data` at them). The dataset builders (`bar_dataset.py`, `scan_dataset.py`, and the
other tools that made the sets) are not included.

## Dataset layout (recognizer)

Each data root is a folder with `bars.jsonl` and the crops it names. A line is one bar: `image` (the crop,
a grey PNG, path relative to the root), `mask` (the rectangle of the crop that is the bar itself: left, top,
right, bottom), `lmx` (the bar's Linearized MusicXML), `context` (the clefs, key and time in force where it
starts, as LMX tokens), `piece`, `style`, `page`, `bar`, `system`, `first` (the bar opens its system),
`split` (train or val) and `licence`.

## Main recognizer (`models/main.pt`)

Sanity check of the real-scan set (not included here; it needs the dataset builders):
`tools/check_realscan_set.py data/bars_realscans4b`.

```
python training/train_recognizer.py data/bars_realscans2 data/bars_realscans3 data/bars_realscans4b data/bars_lmx data/bars_asap2 data/bars_pdmx data/bars_olimpic data/bars_grandstaff_noval --repeat data/bars_realscans2:3 --repeat data/bars_realscans3:3 --repeat data/bars_realscans4b:3 --val-data data/bars_olimpic_scanned --guard data/bars_realscans3 --pretrained --system-reset --reset-keeps-time --lr 3e-4 --warmup 2000 --steps 100000 --check 5000 --sample 512 --loss-bars 800 --patience 6 --stop-on "note F1" --workers 4 --gpu-memory 0.65 --out runs/FULL_scratch_mix3.pt
```

## Specialist for the first bar of each system (`models/specialist.pt`)

The same trainer, continuing from an earlier specialist and training only the state (clefs, key, time) tokens:

```
python training/train_recognizer.py data/bars_realscans2 data/bars_realscans3 data/bars_realscans4b data/bars_lmx data/bars_asap2 data/bars_pdmx data/bars_olimpic data/bars_grandstaff_noval --repeat data/bars_realscans2:3 --repeat data/bars_realscans3:3 --repeat data/bars_realscans4b:3 --val-data data/bars_olimpic_scanned data/bars_lmx --guard data/bars_realscans3 --init runs/S2_mix3.pt --only state --system-reset --reset-keeps-time --oversample first_time:30 --oversample first_key:10 --lr 5e-5 --warmup 100 --steps 10000 --check 1000 --sample 512 --loss-bars 800 --patience 4 --stop-on "state right" --workers 4 --gpu-memory 0.65 --out runs/S_R234_mix3.pt
```

Lineage of `--init runs/S2_mix3.pt`, each step `--only state` from the one before:

1. `finetune_v2_alldata.pt` (a general recognizer) to `starts.pt`:
   `python training/train_recognizer.py data/bars_lmx data/bars_asap2 data/bars_pdmx data/bars_olimpic data/bars_grandstaff_noval --out starts.pt --init finetune_v2_alldata.pt --only state --system-reset --reset-keeps-time --lr 1e-4 --warmup 200 --check 1000 --sample 512 --loss-bars 800 --patience 4 --stop-on "state right" --steps 15000 --workers 6`
2. `starts.pt` to `starts_v2.pt`:
   `python training/train_recognizer.py data/bars_lmx data/bars_asap2 data/bars_pdmx data/bars_olimpic data/bars_grandstaff_noval --out starts_v2.pt --init starts.pt --only state --oversample first_time:30 --oversample first_key:10 --system-reset --reset-keeps-time --lr 5e-5 --warmup 100 --check 1000 --sample 512 --loss-bars 800 --patience 3 --stop-on "state right" --steps 8000 --workers 6`
3. `starts_v2.pt` to `S2_mix3.pt`:
   `python training/train_recognizer.py data/bars_realscans2 data/bars_lmx data/bars_asap2 data/bars_pdmx data/bars_olimpic data/bars_grandstaff_noval --repeat data/bars_realscans2:3 --val-data data/bars_olimpic_scanned --guard data/bars_realscans2 --init starts_v2.pt --only state --system-reset --reset-keeps-time --oversample first_time:30 --oversample first_key:10 --lr 5e-5 --warmup 100 --steps 10000 --check 1000 --sample 512 --loss-bars 800 --patience 4 --stop-on "state right" --workers 6 --out runs/S2_mix3.pt`
4. `S2_mix3.pt` to `S_R234_mix3.pt`: the specialist command above.

## Bar counter (`models/barcounter.pt`)

Trained on pages prepared by `omr_ml.py prepare` (MuseScore engravings of PDMX and ASAP scores, AudioLabs v2
scans, generated text pages); the prepared pages (`index.json` and `pages/`) come from the zip. The code is in
`training/barcounter/` (Apache 2.0, see its LICENSE and NOTICE).

```
python training/barcounter/omr_ml.py train --data data/omr_ml --model barcounter_new.pt
```

`--steps`, `--batch`, `--lr` and `--workers` set the schedule (defaults 12000, 8, 2e-3, 8). Install
`training/barcounter/requirements.txt` first.
