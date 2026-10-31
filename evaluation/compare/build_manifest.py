"""Write manifest.json for evaluation/compare/f1_models.py: the ten pieces of the earlier comparison folder with the outputs stored
there, and the four engraved pieces with the outputs of run_systems.py (runs/ here).

  python -I -X utf8 evaluation/compare/build_manifest.py
"""
import glob
import json
import os
from pathlib import Path

HERE = Path.cwd()  # the benchmark's working directory (pieces_new.json, runs/, manifest.json)
C13 = Path(os.environ.get("COMPARISON_DIR", "comparison"))  # the earlier comparison's folder: pieces10.json, pieces.json, runs/
RUNS = HERE / "runs"
TEN = json.loads((C13 / "pieces10.json").read_text(encoding="utf-8"))["pieces"]
PIECES13 = json.loads((C13 / "pieces.json").read_text(encoding="utf-8"))
NEW = json.loads((HERE / "pieces_new.json").read_text(encoding="utf-8"))["pieces"]
ASAP = Path(os.environ.get("ASAP_DIR", "data/asap-dataset"))  # the ASAP dataset (not distributed here)


def truth_of_ten(p):
    ed = p["edition"]
    if ed in PIECES13:
        return PIECES13[ed]["truth"]
    return str((ASAP / p["asap"] / "xml_score.musicxml").resolve())


def files(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern))


def cut_of(folder: Path):
    r = folder / "result.json"
    if r.exists():
        return json.loads(r.read_text(encoding="utf-8")).get("truth_measures_used")
    return None


def main() -> None:
    pieces, ids_ten = {}, []
    for p in TEN:
        ed = p["edition"]
        ids_ten.append(ed)
        pieces[ed] = {"title": p["title"], "period": p["period"], "split": p["split"], "truth": truth_of_ten(p),
                      "partition": str(RUNS / "ours" / ed / f"{ed}.json"), "set": "ten"}
    for p in NEW:
        pieces[p["id"]] = {"title": p["title"], "period": p["period"], "split": "unseen", "truth": str(HERE / "pieces" / p["id"] / "score.musicxml"),
                           "partition": str(RUNS / "ours" / p["id"] / f"{p['id']}.json"), "set": "engraved"}
    systems: dict = {}

    def add(name, ed, fs, ran=None, **extra):
        # a system that was run but wrote nothing is kept (no output: every gold token an edit); one never run on a piece is absent
        if fs or ran:
            systems.setdefault(name, {})[ed] = {"files": fs, **extra}

    for ed in pieces:
        new = pieces[ed]["set"] == "engraved"
        base = RUNS if new else C13 / "runs"
        # ours, the installed pipeline (this experiment's own runs): the recognizer's own tokens (the system's output is LMX, as
        # Zeus's is), and, as a second row, what its MusicXML file gives back through the converter
        d = RUNS / "ours" / ed
        if (d / f"{ed}.musicxml").exists():
            systems.setdefault("ours", {})[ed] = {"native_json": str(d / f"{ed}.json"), "note_files": [str(d / f"{ed}.musicxml")]}
            add("ours (via its MusicXML)", ed, [str(d / f"{ed}.musicxml")])
        if not new:
            d = C13 / "runs" / "ours_final" / "SCRATCH_RESOLVE" / ed
            if (d / f"{ed}.musicxml").exists():
                systems.setdefault("ours 26 Oct, scale floor 0.4", {})[ed] = {"native_json": str(d / f"{ed}.json"), "note_files": [str(d / f"{ed}.musicxml")]}
                add("ours 26 Oct, via its MusicXML", ed, [str(d / f"{ed}.musicxml")])
        for name, folder, pat, in [("homr", "homr", "p[0-9][0-9][0-9].musicxml"), ("oemer", "oemer", "p[0-9][0-9][0-9].musicxml"),
                                   ("Audiveris", "audiveris", f"{ed}*.mxl")]:
            d = base / folder / ed
            add(name, ed, files(str(d / pat)), ran=(d / "result.json").exists() or (d / "timing.json").exists(), cut=None if new else cut_of(d))
        if not new:
            for name, folder, pat in [("SMT", "smt-camera-gs-v2", "smt_score.musicxml"), ("Zeus", "zeus-gs-pieces", "bars.lmx"),
                                      ("img2Mxml (first page, key and time given)", "img2mxml", "p[0-9][0-9][0-9].musicxml")]:
                d = C13 / "runs" / folder / ed
                add(name, ed, files(str(d / pat)), ran=(d / "result.json").exists(), cut=cut_of(d), **({"note_files": files(str(d / "score.musicxml"))} if name == "Zeus" else {}))
        else:
            for name, folder, pat in [("SMT", "smt", "smt_score.musicxml"), ("Zeus", "zeus", "bars.lmx")]:
                d = RUNS / folder / ed
                add(name, ed, files(str(d / pat)), ran=(d / "result.json").exists(), **({"note_files": files(str(d / "score.musicxml"))} if name == "Zeus" else {}))
    (HERE / "manifest.json").write_text(json.dumps({"pieces": pieces, "systems": systems, "ten": ids_ten,
                                                    "engraved": [p["id"] for p in NEW]}, indent=1), encoding="utf-8")
    for s, v in systems.items():
        print(f"{s:45} {len(v)} pieces")


main()
