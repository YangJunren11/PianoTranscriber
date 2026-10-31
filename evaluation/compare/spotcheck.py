"""Print one printed system's gold and predicted tokens side by side (an alignment by Levenshtein edit operations), for the
hand check of evaluation/compare/system_ser_models.py.

  python -I -X utf8 evaluation/compare/spotcheck.py DUMPDIR/NAME__PIECE.json INDEX [--width 40] [--max-rows 60]

DUMPDIR: evaluation/compare/system_ser_models.py --dump-systems. INDEX: the position in the file's list of systems.
Columns: gold token | predicted token | op (= equal, S substituted, D missing in the prediction, I extra in the prediction).
Ends with the edit count recomputed here by a plain dynamic-programming Levenshtein (no library) next to the tool's.
"""
import json
import sys


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main():
    path, idx = sys.argv[1], int(sys.argv[2])
    width = int(sys.argv[sys.argv.index("--width") + 1]) if "--width" in sys.argv else 38
    maxrows = int(sys.argv[sys.argv.index("--max-rows") + 1]) if "--max-rows" in sys.argv else 70
    s = json.loads(open(path, encoding="utf-8").read())[idx]
    g, p = s["gold"], s["pred"]
    from rapidfuzz.distance import Levenshtein

    ops = Levenshtein.opcodes(g, p)
    rows = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            rows += [(g[i1 + k], p[j1 + k], "=") for k in range(i2 - i1)]
        elif tag == "replace":
            n = max(i2 - i1, j2 - j1)
            for k in range(n):
                a = g[i1 + k] if i1 + k < i2 else ""
                b = p[j1 + k] if j1 + k < j2 else ""
                rows.append((a, b, "S" if a and b else ("D" if a else "I")))
        elif tag == "delete":
            rows += [(g[i1 + k], "", "D") for k in range(i2 - i1)]
        else:
            rows += [("", p[j1 + k], "I") for k in range(j2 - j1)]
    print(f"{path} system #{idx} {s['system']}: gold {len(g)} tokens, predicted {len(p)} tokens, tool edits {s['edits']}")
    shown = 0
    for a, b, o in rows:
        if shown >= maxrows:
            print(f"... ({len(rows) - shown} more rows)")
            break
        print(f"{a[:width]:<{width}} | {b[:width]:<{width}} | {o}")
        shown += 1
    ops_count = sum(o != "=" for _, _, o in rows)
    print(f"edits by the opcodes: {ops_count}; plain DP Levenshtein: {lev(g, p)}; tool: {s['edits']}")


main()
