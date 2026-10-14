"""Events: `get_event("GW150914")` returns an instance of its `Event` subclass.

Each event lives in a directory of the repository named after it (lower case), holding
its products/ (scene data, camera paths, PPDs), renders/, validation scripts and README,
and runs in the Python environment listed here (the one with its waveform model).
Importing this module is cheap (it is also used inside Blender).
"""

from importlib import import_module

from .. import REPO

# name -> (module, class, environment relative to the repository)
EVENTS = {
    "GW250114": ("gw250114", "GW250114", "gw250114/.venv"),
    "GW190521": ("gw190521", "GW190521", "gw250114/.venv"),
    "GW150914": ("gw150914", "GW150914", "gw150914/.venv"),
}


def get_event(name):
    module, cls, _ = EVENTS[name.upper()]
    return getattr(import_module(f"{__name__}.{module}"), cls)()


def event_directory(name):
    return REPO / name.lower()


def event_environment(name):
    return REPO / EVENTS[name.upper()][2]
