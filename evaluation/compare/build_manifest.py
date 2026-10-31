"""Write manifest.json for evaluation/compare/f1_models.py: the ten pieces of the earlier comparison folder with the outputs stored
there, and the four engraved pieces with the outputs of run_systems.py (runs/ here).

  python -I -X utf8 evaluation/compare/build_manifest.py

Every system has an entry for every piece it was run on, whether or not it wrote anything, and the pipeline here ("ours") is
no exception: a run that produced no output is kept with an empty file list and "status": "failed", which f1_models.py scores
as 0 on that piece, so a failure cannot drop a piece out of a system's pooled figures (a system scored on the pieces it got
right alone would look better than one scored on every piece). A system was run on a piece when its output folder for the
piece exists: the runners make one for every piece they attempt and write result.json or timing.json into it, and a system
that crashed before writing anything still has the folder. A system with no folder was never run on the piece: it has no
entry, and f1_models.py lists the piece as not run. The summary printed names, for every system, the pieces it ran and wrote
nothing on and the pieces it was not run on.
"""
import glob
import json
import os
from pathlib import Path

OURS_26_OCT = "ours 26 Oct, scale floor 0.4"


def files(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern))


def ran(folder: Path) -> bool:
    """Whether a system was run on a piece: its output folder exists (made for every piece attempted, whether or not the
    system then wrote anything; result.json or timing.json, where the runner writes one, are in it)."""
    return folder.is_dir()


def cut_of(folder: Path):
    r = folder / "result.json"
    if r.exists():
        return json.loads(r.read_text(encoding="utf-8")).get("truth_measures_used")
    return None


def entry(folder: Path, fs: list[str], **extra) -> dict | None:
    """The manifest's entry of a system on a piece: its output files `fs` (f1_models.py scores "note_files" where there are
    some, else "files") and "status": "ok" when it wrote something, "failed" when it ran and wrote nothing. None when the
    system was not run on the piece (no output, no folder)."""
    scored = extra.get("note_files") or fs
    if not scored and not ran(folder):
        return None
    return {"files": fs, "status": "ok" if scored else "failed", **extra}


def ours_entry(folder: Path, ed: str) -> dict | None:
    """The pipeline's entry: its MusicXML as "note_files" (none when the run failed), and the recognizer's own tokens
    ("native_json") where it wrote them."""
    native = folder / f"{ed}.json"
    return entry(folder, [], note_files=files(str(folder / f"{ed}.musicxml")), **({"native_json": str(native)} if native.exists() else {}))


def build(here: Path, c13: Path, asap: Path) -> dict:
    """The manifest: pieces, systems (system -> piece -> entry), and the piece lists "ten" and "engraved"."""
    ten = json.loads((c13 / "pieces10.json").read_text(encoding="utf-8"))["pieces"]
    pieces13 = json.loads((c13 / "pieces.json").read_text(encoding="utf-8"))
    new = json.loads((here / "pieces_new.json").read_text(encoding="utf-8"))["pieces"]
    runs = here / "runs"

    def truth_of_ten(p):
        ed = p["edition"]
        if ed in pieces13:
            return pieces13[ed]["truth"]
        return str((asap / p["asap"] / "xml_score.musicxml").resolve())

    pieces, ids_ten = {}, []
    for p in ten:
        ed = p["edition"]
        ids_ten.append(ed)
        pieces[ed] = {"title": p["title"], "period": p["period"], "split": p["split"], "truth": truth_of_ten(p),
                      "partition": str(runs / "ours" / ed / f"{ed}.json"), "set": "ten"}
    for p in new:
        pieces[p["id"]] = {"title": p["title"], "period": p["period"], "split": "unseen", "truth": str(here / "pieces" / p["id"] / "score.musicxml"),
                           "partition": str(runs / "ours" / p["id"] / f"{p['id']}.json"), "set": "engraved"}
    systems: dict = {}

    def add(name, ed, e):
        if e is not None:
            systems.setdefault(name, {})[ed] = e

    for ed in pieces:
        engraved = pieces[ed]["set"] == "engraved"
        base = runs if engraved else c13 / "runs"
        # ours, the installed pipeline (this experiment's own runs): the recognizer's own tokens (the system's output is LMX, as
        # Zeus's is), and, as a second row, what its MusicXML file gives back through the converter. Both rows have an entry
        # for every piece the pipeline was run on, a failed run included, like every other system
        d = runs / "ours" / ed
        add("ours", ed, ours_entry(d, ed))
        add("ours (via its MusicXML)", ed, entry(d, files(str(d / f"{ed}.musicxml"))))
        if not engraved:
            d = c13 / "runs" / "ours_final" / "SCRATCH_RESOLVE" / ed
            add(OURS_26_OCT, ed, ours_entry(d, ed))
            add("ours 26 Oct, via its MusicXML", ed, entry(d, files(str(d / f"{ed}.musicxml"))))
        for name, folder, pat, in [("homr", "homr", "p[0-9][0-9][0-9].musicxml"), ("oemer", "oemer", "p[0-9][0-9][0-9].musicxml"),
                                   ("Audiveris", "audiveris", f"{ed}*.mxl")]:
            d = base / folder / ed
            add(name, ed, entry(d, files(str(d / pat)), cut=None if engraved else cut_of(d)))
        if not engraved:
            for name, folder, pat in [("SMT", "smt-camera-gs-v2", "smt_score.musicxml"), ("Zeus", "zeus-gs-pieces", "bars.lmx"),
                                      ("img2Mxml (first page, key and time given)", "img2mxml", "p[0-9][0-9][0-9].musicxml")]:
                d = c13 / "runs" / folder / ed
                add(name, ed, entry(d, files(str(d / pat)), cut=cut_of(d), **({"note_files": files(str(d / "score.musicxml"))} if name == "Zeus" else {})))
        else:
            for name, folder, pat in [("SMT", "smt", "smt_score.musicxml"), ("Zeus", "zeus", "bars.lmx")]:
                d = runs / folder / ed
                add(name, ed, entry(d, files(str(d / pat)), **({"note_files": files(str(d / "score.musicxml"))} if name == "Zeus" else {})))
    return {"pieces": pieces, "systems": systems, "ten": ids_ten, "engraved": [p["id"] for p in new]}


def summary(manifest: dict) -> list[str]:
    """A line per system: the pieces it has an entry for, those it ran and wrote nothing on, and those it was not run on."""
    out = []
    for s, v in manifest["systems"].items():
        failed = [ed for ed, e in v.items() if e.get("status") == "failed"]
        absent = [ed for ed in manifest["pieces"] if ed not in v]
        line = f"{s:45} {len(v)} pieces"
        if failed:
            line += f"; ran and wrote nothing on {len(failed)} (scored 0): {', '.join(failed)}"
        if absent:
            line += f"; not run on {len(absent)}: {', '.join(absent)}"
        out.append(line)
    return out


def main() -> None:
    here = Path.cwd()  # the benchmark's working directory (pieces_new.json, runs/, manifest.json)
    c13 = Path(os.environ.get("COMPARISON_DIR", "comparison"))  # the earlier comparison's folder: pieces10.json, pieces.json, runs/
    asap = Path(os.environ.get("ASAP_DIR", "data/asap-dataset"))  # the ASAP dataset (not distributed here)
    manifest = build(here, c13, asap)
    (here / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("\n".join(summary(manifest)))


if __name__ == "__main__":
    main()
