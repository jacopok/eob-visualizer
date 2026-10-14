"""EOB dynamics of gravitational-wave posterior samples, rendered in Blender.

Layout:
- `models`: one `EOBRun` subclass per waveform model (SEOBNRv5PHM, SEOBNRv4PHM,
  TEOBResumS-Dali), each reproducing the PE call and exposing the dynamics in a common form;
- `events`: one `Event` subclass per event (posterior, PE settings, data for the panel);
- `export`, `camera`, `blender_scene`, `panel`, `ppd`: the event-agnostic pipeline;
- `python -m eobviz ...`: command line (see `__main__.py`).

Kept free of heavy imports: `blender_scene.py` imports `eobviz.events` inside Blender.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
