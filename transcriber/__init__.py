"""PDF to MusicXML, bar by bar: a page's bars counted and its systems found, each bar cropped and
read by a network, and the bars stitched back into MusicXML."""

import sys
from pathlib import Path

# The LMX library (linearized MusicXML) is vendored in third_party/lmx
_LMX = str(Path(__file__).resolve().parents[1] / "third_party" / "lmx" / "src")
if _LMX not in sys.path:
    sys.path.append(_LMX)

from .api import Transcriber, pdf_to_musicxml  # noqa: E402,F401  (light: torch is imported when a model loads)
