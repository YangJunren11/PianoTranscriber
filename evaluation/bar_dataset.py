"""The pieces of the bar dataset reader that the evaluation uses: reading a MusicXML or .mxl, finding the piano part, and
linearizing it measure by measure with LMX (github.com/OMR-Research/lmx). Reduced from the research repository's
tools/bar_dataset.py; the functions are unchanged."""

from __future__ import annotations

import copy
import io
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transcriber  # noqa: E402,F401  (puts third_party/lmx on the path)

NOT_PIANO = re.compile(r"harp|guitar|organ|accordion|marimba|vibraphone|xylophone|celest|voice|violin|flute", re.I)


def read(path: Path) -> ET.Element:
    if path.suffix == ".mxl":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if not n.startswith("META-INF") and n.endswith((".xml", ".musicxml")))
            return ET.fromstring(z.read(name))
    return ET.parse(path).getroot()


def piano_part(score: ET.Element) -> ET.Element | None:
    """The score's part, if it is for piano: its only part, on two staves, named for no other instrument."""
    parts = score.findall("part")
    if len(parts) != 1 or {s.text.strip() for s in parts[0].iter("staves")} != {"2"}:
        return None
    names = " ".join(e.text or "" for e in score.iter() if e.tag in ("part-name", "instrument-name"))
    return None if NOT_PIANO.search(names) else parts[0]


def whole_bar_rests(part: ET.Element) -> int:
    """Marks every rest with no <type> that lasts a whole bar as a whole-bar rest; how many."""
    divisions = beats = beat_type = None
    marked = 0
    for measure in part.findall("measure"):
        for d in measure.iter("divisions"):
            divisions = int(d.text)
        for t in measure.iter("time"):
            if t.find("beats") is not None and t.find("beat-type") is not None:
                beats, beat_type = int(t.find("beats").text), int(t.find("beat-type").text)
        whole = divisions * 4 * beats // beat_type if divisions and beats and beat_type else None
        for note in measure.findall("note"):
            rest, duration = note.find("rest"), note.find("duration")
            if (rest is not None and note.find("type") is None and rest.get("measure") != "yes"
                    and duration is not None and whole is not None and int(duration.text) == whole):
                rest.set("measure", "yes")
                marked += 1
    return marked


def linearize(part: ET.Element) -> list[tuple[str | None, str]]:
    """Each measure's LMX, None where LMX cannot write it, and the context it starts in."""
    from lmx.linearization.Linearizer import Linearizer

    class Bars(Linearizer):
        def process_measure(self, measure):
            start, said = len(self.output_tokens), self._errout.tell()
            context = []
            if self._key_signature_fifths is not None:
                context.append(f"key:fifths:{self._key_signature_fifths}")
            if self._beats_per_measure:
                context += ["time", f"beats:{self._beats_per_measure}", f"beat-type:{self._beat_type}"]
            for staff, clef in sorted(self._clefs.items()):
                context += [clef, f"staff:{staff}"]
            try:
                super().process_measure(measure)
                ok = self._errout.tell() == said
            except Exception:  # an unknown token, or an assumption of LMX's broken
                ok = False
            self.bars.append((" ".join(self.output_tokens[start:]) if ok else None, " ".join(context)))
            del self.output_tokens[start:]

    part = copy.deepcopy(part)
    whole_bar_rests(part)
    lin = Bars(errout=io.StringIO())
    lin.bars = []
    lin.process_part(part)
    return lin.bars
