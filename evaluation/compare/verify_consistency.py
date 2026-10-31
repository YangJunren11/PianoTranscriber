"""Verification (1): this tool's bar-level figures on the ten pieces against what the earlier comparison folder stored.

  python -I -X utf8 evaluation/compare/verify_consistency.py results.json

For every system that experiments/13 scored: bar_ser_matched (edits over tokens of the aligned pairs, from the manifest's
MusicXML through this tool) against `accuracy.token_error_rate` of the stored result.json (the same quantity, made by
compare.py from its stored bars), and the direct note F1 this tool computed (score_direct) against direct_scores.json.
"""
import json
import sys
from pathlib import Path

import os

HERE = Path.cwd()  # the benchmark's working directory
C13 = Path(os.environ.get("COMPARISON_DIR", "comparison"))  # the earlier comparison's folder: runs/, direct_scores.json
res = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
direct = json.loads((C13 / "direct_scores.json").read_text(encoding="utf-8"))
STORED = {"ours 26 Oct, scale floor 0.4": ("ours_final/SCRATCH_RESOLVE", "ours SCRATCH resolve"),
          "ours 26 Oct, via its MusicXML": ("ours_final/SCRATCH_RESOLVE", "ours SCRATCH resolve"), "homr": ("homr", "homr"),
          "Audiveris": ("audiveris", "audiveris"), "oemer": ("oemer", "oemer"), "SMT": ("smt-camera-gs-v2", "smt-camera-gs-v2"),
          "Zeus": ("zeus-gs-pieces", "zeus-gs-pieces"), "img2Mxml (first page, key and time given)": ("img2mxml", "img2mxml")}
dmap = {(r["system"], r["edition"]): r for r in direct if isinstance(r, dict) and "system" in r}
print("| system | piece | bar SER matched, here | token error rate, stored | diff (points) | note F1 here | direct note F1 stored | diff |")
print("|---|---|---|---|---|---|---|---|")
worst = 0
for name, (folder, dname) in STORED.items():
    for ed, r in res["systems"].get(name, {}).items():
        f = C13 / "runs" / folder / ed / "result.json"
        if ed.startswith("eng") or not f.exists():
            continue
        stored = json.loads(f.read_text(encoding="utf-8")).get("accuracy") or {}
        ter = None if stored.get("token_error_rate") is None else 100 * stored["token_error_rate"]
        here = r.get("bar_ser_matched_cut", r["bar_ser_matched"])
        d = dmap.get((dname, ed))
        nf = None if d is None else 100 * d["note_F1"]
        dn = None if (nf is None or r["note_F1"] is None) else r["note_F1"] - nf
        diff = None if (ter is None or here is None) else here - ter
        worst = max(worst, abs(diff or 0))
        fmt = lambda x, k=1: "-" if x is None else f"{x:.{k}f}"
        print(f"| {name} | {ed} | {fmt(here)} | {fmt(ter)} | {fmt(diff, 2)} | {fmt(r['note_F1'])} | {fmt(nf)} | {fmt(dn, 2)} |")
print(f"\nlargest absolute difference in bar-level SER: {worst:.2f} points")
