"""Tables (markdown) from results.json, written by evaluation/compare/system_ser_models.py.

  python -I -X utf8 evaluation/compare/make_results.py results.json > tables.md
"""
import json
import sys
from pathlib import Path

HERE = Path.cwd()  # the benchmark's working directory (manifest.json, results.json, formal/ ...), not this folder
res = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
man = json.loads((Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "manifest.json").read_text(encoding="utf-8"))
TEN, NEW = man["ten"], man["engraved"]
ORDER = ["ours", "ours (via its MusicXML)", "ours 26 Oct, scale floor 0.4", "ours 26 Oct, via its MusicXML", "homr", "Audiveris", "oemer", "SMT", "Zeus",
         "img2Mxml (first page, key and time given)"]
systems = [s for s in ORDER if s in res["systems"]] + [s for s in res["systems"] if s not in ORDER]


def f(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def pooled(s, ids, e="edits", g="gold_tokens"):
    rows = [res["systems"][s][i] for i in ids if i in res["systems"][s]]
    if len(rows) < len(ids):
        return None
    gold = sum(r[g] for r in rows)
    return 100 * sum(r[e] for r in rows) / gold if gold else None


def mean(s, ids, key):
    v = [res["systems"][s][i][key] for i in ids if i in res["systems"][s] and res["systems"][s][i].get(key) is not None]
    return sum(v) / len(v) if len(v) == len(ids) else None


def title(i):
    return res["pieces"][i]["title"]


POOL = {"ser": ("edits", "gold_tokens"), "ser_nostem": ("edits_nostem", "gold_tokens_nostem"),
        "ser_without_refused": ("edits_without_refused", "gold_tokens_without_refused"),
        "bar_ser_all": ("bar_edits_all", "bar_gold_tokens"), "bar_ser_matched": ("bar_edits_matched", "bar_tokens_matched")}


def table(ids, key, label, d=1):
    out = [f"| piece | gold tokens | " + " | ".join(systems) + " |", "|---|---|" + "---|" * len(systems)]
    for i in ids:
        gold = next((res["systems"][s][i]["gold_tokens"] for s in systems if i in res["systems"][s]), 0)
        out.append(f"| {title(i)} ({i}) | {gold} | " + " | ".join(f(res['systems'][s].get(i, {}).get(key), d) for s in systems) + " |")
    gold = sum(next((res["systems"][s][i]["gold_tokens"] for s in systems if i in res["systems"][s]), 0) for i in ids)
    if key in POOL:
        out.append(f"| **pooled ({label})** | {gold} | " + " | ".join(f(pooled(s, ids, *POOL[key])) for s in systems) + " |")
    else:
        out.append(f"| **mean of the pieces ({label})** | | " + " | ".join(f(mean(s, ids, key)) for s in systems) + " |")
    return chr(10).join(out)


print("## System-level SER (%), per piece\n")
print("### Ten real-scan pieces\n")
print(table(TEN, "ser", "ten"))
print("\n### Four engraved pieces\n")
print(table(NEW, "ser", "four"))
ALL = TEN + NEW
print("\n### Pooled\n")
print("| system | ten pieces | four engraved | all fourteen | mean per piece, ten | mean per piece, four |")
print("|---|---|---|---|---|---|")
for s in systems:
    print(f"| {s} | {f(pooled(s, TEN))} | {f(pooled(s, NEW))} | {f(pooled(s, ALL))} | {f(mean(s, TEN, 'ser'))} | {f(mean(s, NEW, 'ser'))} |")
for key, name in (("bar_ser_all", "Bar-level SER, pooled, bars unmatched counted (insertions and deletions), %"),
                  ("bar_ser_matched", "Bar-level SER on aligned pairs only (experiments/13's token error rate), %"),
                  ("ser_nostem", "System-level SER with stem and beam tokens left out of both sides, %"), ("note_F1", "Note F1 (direct scorer, music21), %"), ("ser_without_refused", "System-level SER leaving out bars the converter refused, %")):
    print(f"\n## {name}\n")
    print("### Ten\n")
    print(table(TEN, key, "ten"))
    print("\n### Four engraved\n")
    print(table(NEW, key, "four"))
print("\n## Bars the converter refused / bars read\n")
print("| system | " + " | ".join(ALL) + " | total refused / read |")
print("|---|" + "---|" * (len(ALL) + 1))
for s in systems:
    row = [f"{res['systems'][s][i]['bars_refused']}/{res['systems'][s][i]['bars_read']}" if i in res["systems"][s] else "-" for i in ALL]
    r = [res["systems"][s][i] for i in ALL if i in res["systems"][s]]
    print(f"| {s} | " + " | ".join(row) + f" | {sum(x['bars_refused'] for x in r)}/{sum(x['bars_read'] for x in r)} |")
print("\n## Alignment: bars read, aligned, unaligned bars, measures missed\n")
print("| system | piece | bars read | aligned | unaligned bars | measures missed | measures (unwritable gold) |")
print("|---|---|---|---|---|---|---|")
for s in systems:
    for i in ALL:
        r = res["systems"][s].get(i)
        if r:
            print(f"| {s} | {i} | {r['bars_read']} | {r['bars_aligned']} | {r['bars_unaligned']} | {r['measures_missed']} | {r['measures']} ({r['measures_unwritable']}) |")
print("\n## Partition (printed systems from our cut)\n")
print("| piece | measures | systems | measures with no bar of ours |")
print("|---|---|---|---|")
for i in ALL:
    p = res["pieces"][i]["partition"]
    print(f"| {i} | {p['measures']} | {p['systems']} | {p['measures_with_no_bar_of_ours']} |")

print("\n## Summary: every system, every metric, by set\n")
print("| system | set | SER (system-level) | SER without stem/beam | SER leaving out refused bars | bar-level SER (unmatched bars counted) | bar-level SER (aligned pairs only) | note F1, mean of pieces | pieces |")
print("|---|---|---|---|---|---|---|---|---|")
for s in systems:
    for label, ids in (("ten", TEN), ("four", NEW), ("fourteen", ALL)):
        have = [i for i in ids if i in res["systems"][s]]
        if len(have) < len(ids):
            continue
        print(f"| {s} | {label} | {f(pooled(s, ids, *POOL['ser']))} | {f(pooled(s, ids, *POOL['ser_nostem']))} | {f(pooled(s, ids, *POOL['ser_without_refused']))} | "
              f"{f(pooled(s, ids, *POOL['bar_ser_all']))} | {f(pooled(s, ids, *POOL['bar_ser_matched']))} | {f(mean(s, ids, 'note_F1'))} | {len(ids)} |")
