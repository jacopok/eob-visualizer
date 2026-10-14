"""Command line of the pipeline, from the repository root:

    python -m eobviz ppd EVENT [--ifo H1] [--n-samples 500] [--approximant NRSur7dq4]
    python -m eobviz export EVENT [--n-samples 200 | --index I]    # default: max likelihood
    python -m eobviz camera EVENT --name cloud200 [--shot orbit]
    python -m eobviz render EVENT --name cloud200 [blender_scene.py options, e.g. --style solid_highlight]
    python -m eobviz panel EVENT --name cloud200 [--no-video]
    python -m eobviz scene EVENT --name cloud200 [blender_scene.py options]   # interactive Blender
        (fast, plain look for iterating: --draft [--max-samples 50], with scene or render)
    python -m eobviz events

The command re-runs itself in the event's Python environment (`events.EVENTS`) if it is
not already running there, so any Python 3 can start it.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import REPO
from .events import EVENTS, event_environment

BLENDER_SCRIPT = Path(__file__).resolve().parent / "blender_scene.py"


def parse_args(argv):
    p = argparse.ArgumentParser(prog="python -m eobviz", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("events", help="list the events and their environments")

    def command(name, help, needs_name=False):
        c = sub.add_parser(name, help=help)
        c.add_argument("event", type=str.upper, choices=sorted(EVENTS))
        if needs_name:
            c.add_argument("--name", required=True, help="scene set, e.g. cloud200 or sample123")
        return c

    c = command("ppd", "whitened posterior predictive distribution of the data")
    c.add_argument("--ifo", default=None)
    c.add_argument("--n-samples", type=int, default=500)
    c.add_argument("--approximant", default=None, help="another PE run of the event (GW250114: NRSur7dq4)")
    c.add_argument("--processes", type=int, default=8)

    c = command("export", "per-frame scene data of posterior samples")
    group = c.add_mutually_exclusive_group()
    group.add_argument("--index", type=int, default=None, help="one posterior sample (default: max likelihood)")
    group.add_argument("--n-samples", type=int, default=None,
                       help="the first N of the random posterior samples of the PPD")
    c.add_argument("--processes", type=int, default=8)

    c = command("camera", "camera path", needs_name=True)
    c.add_argument("--shot", default="close-up", choices=["close-up", "orbit"],
                   help="middle shot: co-rotating close-up of a cluster, or a slow orbit")

    c = command("render", "render the 3D frames with Blender (background)", needs_name=True)
    c.add_argument("--blender", default="blender")
    c = command("scene", "open the scene in Blender to explore it interactively", needs_name=True)
    c.add_argument("--blender", default="blender")

    c = command("panel", "panel frames and the final video", needs_name=True)
    c.add_argument("--no-video", action="store_true")
    # unknown options of render/scene go to blender_scene.py
    return p.parse_known_args(argv)


def ensure_environment(event_name):
    """Re-execute in the event's environment, unless already there."""
    env = event_environment(event_name)
    if Path(sys.prefix).resolve() == env.resolve() or os.environ.get("EOBVIZ_NO_REEXEC"):
        return
    python = env / "bin" / "python"
    if not python.exists():
        sys.exit(f"{event_name} needs the environment {env} (see its README)")
    environ = dict(os.environ, EOBVIZ_NO_REEXEC="1",
                   PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))
    os.execve(python, [str(python), "-m", "eobviz", *sys.argv[1:]], environ)


def blender(args, extra, interactive):
    cmd = [args.blender]
    if not interactive:
        cmd += ["-b", "--factory-startup"]
    cmd += ["-P", str(BLENDER_SCRIPT), "--", "--event", args.event, "--name", args.name, *extra]
    if interactive:
        cmd.append("--interactive")
    print(" ".join(cmd))
    return subprocess.run(cmd).returncode


def main(argv=None):
    args, extra = parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "events":
        for name, (_, _, env) in EVENTS.items():
            print(f"{name}: directory {name.lower()}/, environment {env}")
        return 0
    if extra and args.command not in ("render", "scene"):
        sys.exit(f"unrecognized arguments: {' '.join(extra)}")
    if args.command in ("render", "scene"):  # Blender has its own Python
        return blender(args, extra, interactive=args.command == "scene")

    ensure_environment(args.event)
    from .events import get_event

    event = get_event(args.event)
    if args.command == "ppd":
        from .ppd import make_ppd

        print(make_ppd(event, args.ifo, args.n_samples, args.approximant, args.processes))
    elif args.command == "export":
        from .export import export_set, ppd_indices

        if args.n_samples:
            export_set(event, ppd_indices(event, args.n_samples), f"cloud{args.n_samples}", args.processes)
        else:
            index = event.max_likelihood_index() if args.index is None else args.index
            export_set(event, [index], f"sample{index}", 1)
    elif args.command == "camera":
        from .camera import camera_path

        camera_path(args.name, event.products, args.shot)
    elif args.command == "panel":
        from .panel import compose, render_panels

        frames, fps = render_panels(event, args.name)
        if not args.no_video:
            print(compose(frames, fps, event.renders / f"{args.name}.mp4"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
