"""Normalise every MusicXML of the benchmark (gold included) through MuseScore 4 and write manifest_formal.json.

  python -I -X utf8 evaluation/compare/normalise_formal.py [--only SYSTEM ...]

Per file: MuseScore4.exe -f -o OUT IN (timeout 300 s, one file at a time; raw MuseScore output kept in formal_raw/), then one
post-step on a copy: a <time> element equal to the time already in force (per part) is dropped. Nothing else is changed.
Writes formal/<system>/<piece>/<name>.musicxml, formal/gold/<piece>.musicxml, formal/convert_log.json (exit code, seconds,
counts before and after per file) and manifest_formal.json. A file MuseScore fails on stays un-normalised in the manifest.
Resumes: a file already in the log is not converted again.
"""
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path.cwd()  # the benchmark's working directory (manifest.json, results.json, formal/ ...), not this folder
MS = r"C:/Program Files/MuseScore 4/bin/MuseScore4.exe"
FORMAL = HERE / "formal"
RAW = HERE / "formal_raw"
LOG = FORMAL / "convert_log.json"
REFERENCE = ("ours", "ours 26 Oct, scale floor 0.4")  # native-LMX rows, left as they are


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def read_root(path):
    p = Path(path)
    if p.suffix.lower() == ".mxl":
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".xml", ".musicxml")) and not n.startswith("META-INF")]
            return ET.fromstring(z.read(names[0]))
    return ET.parse(p).getroot()


def counts(path):
    """pitched notes, grace notes, rests, measures (of the longest part), time elements."""
    root = read_root(path)
    pitched = grace = rests = 0
    for n in root.iter("note"):
        if n.find("rest") is not None:
            rests += 1
        elif n.find("pitch") is not None:
            if n.find("grace") is not None:
                grace += 1
            else:
                pitched += 1
    measures = max((len(p.findall("measure")) for p in root.findall("part")), default=0)
    return {"pitched": pitched, "grace": grace, "rests": rests, "measures": measures, "times": len(list(root.iter("time")))}


def drop_restated_times(src, dst):
    """Copy src to dst without any <time> equal to the time in force in its part. Returns how many were dropped."""
    tree = ET.parse(src)
    dropped = 0
    for part in tree.getroot().findall("part"):
        cur = None
        for meas in part.findall("measure"):
            for attrs in meas.findall("attributes"):
                for t in attrs.findall("time"):
                    key = tuple((c.tag, (c.text or "").strip()) for c in t if c.tag in ("beats", "beat-type", "senza-misura"))
                    if key == cur:
                        attrs.remove(t)
                        dropped += 1
                    else:
                        cur = key
                if len(attrs) == 0:
                    meas.remove(attrs)
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return dropped


def convert(src, raw_out, log, key):
    if key in log:
        return log[key]
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    if raw_out.exists():
        raw_out.unlink()
    t0 = time.time()
    try:
        r = subprocess.run([MS, "-f", "-o", str(raw_out), str(src)], capture_output=True, timeout=300)
        code = r.returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    sec = round(time.time() - t0, 1)
    ok = code == 0 and raw_out.exists() and raw_out.stat().st_size > 0
    rec = {"src": str(src), "exit": code, "seconds": sec, "ok": ok}
    log[key] = rec
    LOG.write_text(json.dumps(log, indent=1), encoding="utf-8")
    return rec


def main():
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else None
    man = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    FORMAL.mkdir(exist_ok=True)
    log = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else {}
    jobs = []  # (system, piece, src, out, raw)
    for pid, p in man["pieces"].items():
        jobs.append(("gold", pid, p["truth"], FORMAL / "gold" / f"{pid}.musicxml", RAW / "gold" / f"{pid}.musicxml"))
    for name, per in man["systems"].items():
        if name in REFERENCE:
            continue
        for pid, spec in per.items():
            files = spec.get("note_files") if name == "Zeus" else spec.get("files")
            for f in files or []:
                stem = Path(f).stem
                jobs.append((name, pid, f, FORMAL / slug(name) / pid / f"{stem}.musicxml", RAW / slug(name) / pid / f"{stem}.musicxml"))
    todo = [j for j in jobs if not only or j[0] in only]
    print(f"{len(todo)} files", flush=True)
    for k, (name, pid, src, out, raw) in enumerate(todo, 1):
        key = f"{name}|{pid}|{Path(src).name}"
        rec = convert(src, raw, log, key)
        if rec["ok"] and "after" not in rec:
            out.parent.mkdir(parents=True, exist_ok=True)
            rec["dropped_times"] = drop_restated_times(raw, out)
            rec["before"] = counts(src)
            rec["after"] = counts(out)
            rec["raw_after"] = counts(raw)
            LOG.write_text(json.dumps(log, indent=1), encoding="utf-8")
        elif not rec["ok"] and "before" not in rec:
            try:
                rec["before"] = counts(src)
            except Exception as e:  # noqa: BLE001
                rec["before"] = {"error": str(e)[:100]}
            LOG.write_text(json.dumps(log, indent=1), encoding="utf-8")
        print(f"[{k}/{len(todo)}] {key} exit {rec['exit']} {rec['seconds']}s", flush=True)

    # manifest
    def mapped(name, pid, f):
        key = f"{name}|{pid}|{Path(f).name}"
        rec = log.get(key)
        if rec and rec["ok"]:
            stem = Path(f).stem
            base = FORMAL / "gold" if name == "gold" else FORMAL / slug(name) / pid
            return str(base / (f"{pid}.musicxml" if name == "gold" else f"{stem}.musicxml"))
        return f  # failed or not converted: the un-normalised file

    out = {"pieces": {}, "systems": {}, "ten": man["ten"], "engraved": man["engraved"]}
    for pid, p in man["pieces"].items():
        out["pieces"][pid] = {**p, "truth": mapped("gold", pid, p["truth"])}
    for name, per in man["systems"].items():
        out["systems"][name] = {}
        for pid, spec in per.items():
            if name in REFERENCE:
                out["systems"][name][pid] = spec
                continue
            new = {k: v for k, v in spec.items() if k not in ("native_json", "files", "note_files")}
            if name == "Zeus" and spec.get("note_files"):
                new["files"] = [mapped(name, pid, f) for f in spec["note_files"]]
                new["note_files"] = new["files"]
            else:
                new["files"] = [mapped(name, pid, f) for f in spec.get("files", [])]
            out["systems"][name][pid] = new
    (HERE / "manifest_formal.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    # round 2: normalised files with the raw files as fallback (evaluation/compare/system_ser_models.py: lossless-or-raw rule)
    out2 = {"pieces": {}, "systems": {}, "ten": man["ten"], "engraved": man["engraved"]}
    for pid, p in man["pieces"].items():
        out2["pieces"][pid] = {**out["pieces"][pid], "fallback_truth": p["truth"]}
    for name, per in man["systems"].items():
        out2["systems"][name] = {}
        for pid, spec in per.items():
            if name in REFERENCE:
                out2["systems"][name][pid] = spec
                continue
            new = dict(out["systems"][name][pid])
            raw_files = spec.get("files", [])
            if name == "Zeus":  # raw bars are LMX; a file MuseScore crashed on is raw throughout
                new["fallback_files"] = list(raw_files)
                new["files"] = [raw_files[i] if new["files"][i] == spec["note_files"][i] else new["files"][i] for i in range(len(raw_files))]
            else:
                new["fallback_files"] = list(raw_files)
            out2["systems"][name][pid] = new
    (HERE / "manifest_formal2.json").write_text(json.dumps(out2, indent=1), encoding="utf-8")
    fails = [k for k, v in log.items() if not v["ok"]]
    print("failures:", len(fails), fails)


main()
