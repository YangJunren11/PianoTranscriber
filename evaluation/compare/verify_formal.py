"""Checks of the normalised benchmark; prints markdown tables (ASCII).

  python -I -X utf8 evaluation/compare/verify_formal.py report     (conservation, gold, measures, headline, ranking)
  python -I -X utf8 evaluation/compare/verify_formal.py convdet    (3 files converted twice, hashes)
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path.cwd()  # the benchmark's working directory (manifest.json, results.json, formal/ ...), not this folder
MS = r"C:/Program Files/MuseScore 4/bin/MuseScore4.exe"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pct(a, b):
    return "-" if not b else f"{100 * (a - b) / b:+.2f}%"


def pooled(res, s, ids, e="edits", g="gold_tokens"):
    rows = [res["systems"][s][i] for i in ids if i in res["systems"][s]]
    if len(rows) < len(ids):
        return None
    return 100 * sum(r[e] for r in rows) / sum(r[g] for r in rows)


def report():
    log = json.loads((HERE / "formal" / "convert_log.json").read_text(encoding="utf-8"))
    man = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    raw = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    new = json.loads((HERE / "results_formal.json").read_text(encoding="utf-8")) if (HERE / "results_formal.json").exists() else None
    agg = {}
    for k, v in log.items():
        s = k.split("|")[0]
        a = agg.setdefault(s, dict(files=0, fail=0, sec=0.0, dropped=0, **{f"{w}_{c}": 0 for w in ("b", "a") for c in ("pitched", "grace", "rests", "measures")}))
        a["files"] += 1
        a["sec"] += v["seconds"] if isinstance(v["seconds"], (int, float)) else 0
        if not v["ok"]:
            a["fail"] += 1
            continue
        a["dropped"] += v.get("dropped_times", 0)
        for c in ("pitched", "grace", "rests", "measures"):
            a[f"b_{c}"] += v["before"][c]
            a[f"a_{c}"] += v["after"][c]
    print("## MuseScore conversions, note conservation, measures, dropped time signatures (per system, converted files only)\n")
    print("| system | files | failed | seconds | pitched before | after | change | grace before | after | rests before | after | measures before | after | times dropped |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s, a in agg.items():
        flag = " **>1%**" if a["b_pitched"] and abs(a["a_pitched"] - a["b_pitched"]) > 0.01 * a["b_pitched"] else ""
        print(f"| {s} | {a['files']} | {a['fail']} | {a['sec']:.0f} | {a['b_pitched']} | {a['a_pitched']} | {pct(a['a_pitched'], a['b_pitched'])}{flag} | {a['b_grace']} | {a['a_grace']} | "
              f"{a['b_rests']} | {a['a_rests']} | {a['b_measures']} | {a['a_measures']} | {a['dropped']} |")
    print("\n## Failures\n")
    for k, v in log.items():
        if not v["ok"]:
            print(f"- {k}: exit {v['exit']}, {v['seconds']} s")
    print("\n## Per file with a change in pitched notes of more than 1% or in measures\n")
    for k, v in log.items():
        if v["ok"]:
            b, a = v["before"], v["after"]
            if (b["pitched"] and abs(a["pitched"] - b["pitched"]) > 0.01 * b["pitched"]) or a["measures"] != b["measures"]:
                print(f"- {k}: pitched {b['pitched']} -> {a['pitched']}, grace {b['grace']} -> {a['grace']}, measures {b['measures']} -> {a['measures']}")
    if new is None:
        return
    print("\n## Gold before / after (tokens, measures LMX cannot write)\n")
    print("| piece | measures before | unwritable before | gold tokens before | measures after | unwritable after | gold tokens after |")
    print("|---|---|---|---|---|---|---|")
    ref = "ours (via its MusicXML)"
    for pid in man["pieces"]:
        b, a = raw["systems"][ref][pid], new["systems"][ref][pid]
        print(f"| {pid} | {b['measures']} | {b['measures_unwritable']} | {b['gold_tokens']} | {a['measures']} | {a['measures_unwritable']} | {a['gold_tokens']} |")
    ten, eng = man["ten"], man["engraved"]
    sets = (("ten", ten), ("four", eng), ("fourteen", ten + eng))
    print("\n## Headline: SER (%) pooled, raw vs normalised (gold tokens differ between the two; note F1 = mean of pieces, raw -> normalised)\n")
    print("| system | set | raw | normalised | difference | note F1 raw | note F1 normalised |")
    print("|---|---|---|---|---|---|---|")
    rank = {}
    for s in new["systems"]:
        for label, ids in sets:
            r, n = pooled(raw, s, ids) if s in raw["systems"] else None, pooled(new, s, ids)
            if n is None:
                continue
            def f1(res):
                v = [res["systems"][s][i]["note_F1"] for i in ids if s in res["systems"] and i in res["systems"][s] and res["systems"][s][i].get("note_F1") is not None]
                return f"{sum(v) / len(v):.1f}" if len(v) == len(ids) else "-"
            d = "-" if r is None else f"{n - r:+.1f}"
            print(f"| {s} | {label} | {'-' if r is None else f'{r:.1f}'} | {n:.1f} | {d} | {f1(raw)} | {f1(new)} |")
            rank.setdefault(label, {})[s] = (r, n)
    print("\n## Order of systems (best first)\n")
    for label, d in rank.items():
        print(f"- {label} raw: " + " < ".join(s for s, (r, n) in sorted(d.items(), key=lambda x: (x[1][0] is None, x[1][0] or 0)) if r is not None))
        print(f"- {label} normalised: " + " < ".join(s for s, (r, n) in sorted(d.items(), key=lambda x: x[1][1])))
    print("\n## Bars refused by the converter / bars read, normalised vs raw\n")
    for s in new["systems"]:
        def tot(res):
            rows = [res["systems"][s][i] for i in ten + eng if s in res["systems"] and i in res["systems"][s]]
            return sum(r["bars_refused"] for r in rows), sum(r["bars_read"] for r in rows)
        print(f"- {s}: raw {tot(raw) if s in raw['systems'] else '-'}, normalised {tot(new)}")


def convdet():
    log = json.loads((HERE / "formal" / "convert_log.json").read_text(encoding="utf-8"))
    pick = [k for k in log if k.startswith("gold|col0141")] + [k for k in log if k.startswith("homr|col0141|p001")] + [k for k in log if k.startswith("Audiveris|col0141")]
    with tempfile.TemporaryDirectory() as td:
        for k in pick:
            src = log[k]["src"]
            hs = []
            for n in (1, 2):
                out = Path(td) / f"{n}_{Path(src).stem}.musicxml"
                subprocess.run([MS, "-f", "-o", str(out), src], capture_output=True, timeout=300)
                hs.append(sha(out))
            print(k, "twice identical:", hs[0] == hs[1], hs[0][:12], hs[1][:12])


{"report": report, "convdet": convdet}[sys.argv[1]]()
