"""System-level symbol error rate of a chained per-bar reading, from the bar-level dump of evaluation/evaluate_lmx.py.

  python evaluation/system_ser.py data/bars_realscans3 --split val --preds preds.jsonl [--out result.json]

preds.jsonl comes from `evaluation/evaluate_lmx.py <data> --split val --models M --starts-model S --chained --out preds.jsonl`
(a JSON line a bar: image, read, tokens, edits, ...). The bars of the split are grouped by (piece, style, page,
system) in bar order. A system's true sequence is its bars' true LMX in the form the evaluator scores them in (a bar
that opens a system, field "first", with its signature from the context in front: `lmx_bars.reset_tokens`, system
reset on), joined; its predicted sequence is the bars' `read` joined. Per system the Levenshtein distance over
tokens (rapidfuzz, as lmx_accuracy.py) and the true token count. Reported: pooled system-level SER, mean per-system
SER, and, as a check, the bar-level pooled SER of the same bars computed here (per-bar edits over per-bar true tokens)
next to the evaluator's own figure from the dump's `edits` and `tokens`.

Caveat: a bar dataset holds only the bars the pairing kept, so a "system" here is a system's kept bars, not
necessarily all of it.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdf2musicxml.lmx_bars import reset_tokens, true_states  # noqa: E402


def edits(a: list[str], b: list[str]) -> int:
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.distance(a, b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--split", default="val")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-reset", action="store_true", help="truth of a system's first bar without its signature in front")
    a = ap.parse_args()
    root = Path(a.data)
    with (root / "bars.jsonl").open(encoding="utf-8") as f:
        rows = [r for r in map(json.loads, f) if r["split"] == a.split]
    with open(a.preds, encoding="utf-8") as f:
        dump = {d["image"]: d for d in map(json.loads, f)}
    states = true_states(rows)
    truth = [" ".join(reset_tokens(r["lmx"].split(), s, False if a.no_reset else ("key-clefs" if r.get("first") else False)))
             for r, s in zip(rows, states)]
    missing = [r["image"] for r in rows if r["image"] not in dump]
    mismatch = sum(len(t.split()) != dump[r["image"]]["tokens"] for r, t in zip(rows, truth) if r["image"] in dump)
    groups = collections.defaultdict(list)
    for k, r in enumerate(rows):
        groups[(r["piece"], r["style"], r["page"], r.get("system", 0))].append(k)
    per, bar_e, bar_t, ev_e, ev_t = [], 0, 0, 0, 0
    for key, ks in groups.items():
        ks.sort(key=lambda k: rows[k]["bar"])
        t = [x for k in ks for x in truth[k].split()]
        p = [x for k in ks for x in dump.get(rows[k]["image"], {}).get("read", "").split()]
        per.append({"system": list(key), "bars": len(ks), "gold_tokens": len(t), "edits": edits(p, t)})
        for k in ks:
            d = dump.get(rows[k]["image"], {"read": "", "edits": 0, "tokens": 0})
            bar_e += edits(d["read"].split(), truth[k].split())
            bar_t += len(truth[k].split())
            ev_e += d["edits"]
            ev_t += d["tokens"]
    tot_e, tot_t = sum(x["edits"] for x in per), sum(x["gold_tokens"] for x in per)
    ser = [x["edits"] / x["gold_tokens"] for x in per if x["gold_tokens"]]
    result = {"data": a.data, "split": a.split, "preds": a.preds, "bars": len(rows), "systems": len(per),
              "bars_missing_from_dump": len(missing), "truth_length_differs_from_dump": mismatch,
              "gold_tokens": tot_t, "system_pooled_ser_pct": 100 * tot_e / tot_t,
              "system_mean_ser_pct": 100 * statistics.mean(ser), "system_median_ser_pct": 100 * statistics.median(ser),
              "bar_pooled_ser_pct_recomputed": 100 * bar_e / bar_t, "bar_pooled_ser_pct_evaluator_dump": 100 * ev_e / ev_t,
              "mean_bars_per_system": len(rows) / len(per)}
    print(json.dumps(result, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps({**result, "per_system": per}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
