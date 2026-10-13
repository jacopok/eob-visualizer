"""Render the whitened-strain panel frames (PPD band, data, shown samples, time cursor)
and stack them under the Blender frames into a video.

Usage: python panel.py --name cloud40     (products/<name>.json from export_scene.py)
"""

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ppd import OUTDIR

RENDERS = Path(__file__).parent / "renders"
WIDTH_PX, HEIGHT_PX, DPI = 1920, 270, 100
WINDOW_BEFORE, WINDOW_AFTER = 13.0, 6.0  # scrolling window around the cursor [s of video]


def render_panels(name, products=OUTDIR):
    manifest = json.load(open(products / f"{name}.json"))
    labels = manifest["labels"]
    scenes = products / "scenes"
    scene = np.load(scenes / f"{labels[0]}.npz")
    infos = [json.load(open(scenes / f"{label}.json")) for label in labels]
    event = manifest.get("event", "GW250114")
    approximant = manifest.get("approximant", "SEOBNRv5PHM")
    ppd = np.load(products / manifest.get("ppd_file", f"ppd_SEOBNRv5PHM_{scene['ifo']}.npz"))
    before = WINDOW_BEFORE / float(scene["slowdown"])
    after = WINDOW_AFTER / float(scene["slowdown"])
    shown = []
    for index in manifest["indices"]:
        where = np.flatnonzero(ppd["sample_indices"] == index)
        if len(where) == 0:
            raise ValueError(f"whitened waveform of sample {index} not in the PPD file")
        shown.append(ppd["waveforms"][where[-1]])
    n = len(shown)

    t_frames = scene["t_det_rel"]
    t = ppd["times"]
    sel = (t >= t_frames[0] - before - 0.01) & (t <= t_frames[-1] + after + 0.01)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    ax = fig.add_axes([0.045, 0.2, 0.94, 0.75])
    ax.plot(t[sel], ppd["data"][sel], c="0.45", lw=0.8, label=f"{scene['ifo']} data")
    ax.fill_between(t[sel], ppd["lower"][sel], ppd["upper"][sel], color="#56B4E9", alpha=0.7,
                    lw=0, label=f"{approximant} 90% PPD")
    if n == 1:  # (for a cloud, the coincident sample curves would hide the band)
        ax.plot(t[sel], shown[0][sel], c="w", lw=1.0, label="shown posterior sample")
    t_attach = [info["t_attach_det_rel"] for info in infos]
    ax.axvspan(min(t_attach), max(t_attach) + 1e-4, color="#E69F00", alpha=0.25, lw=0,
               label="attachment (merger) times")
    cursor = ax.axvline(t_frames[0], c="#E69F00", lw=1.5)
    peak = max(np.abs(ppd["upper"][sel]).max(), np.abs(ppd["lower"][sel]).max())
    ax.set_ylim(-1.3 * max(peak, 4), 1.3 * max(peak, 4))
    ax.set_xlabel(f"{scene['ifo']} time $-$ {float(scene['reference_time']):.1f} s [s]", fontsize=9)
    ax.set_ylabel(r"Whitened strain [$\sigma$]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper left", fontsize=7, frameon=False)

    outdir = products.parent / "renders" / name
    outdir.mkdir(parents=True, exist_ok=True)
    for f, tf in enumerate(t_frames):
        cursor.set_xdata([tf, tf])
        ax.set_xlim(tf - before, tf + after)
        fig.savefig(outdir / f"panel_{f:04d}.png", dpi=DPI)
    plt.close(fig)
    return outdir, int(scene["fps"])


def compose_for(name, products, fps):
    global RENDERS
    RENDERS = products.parent / "renders"
    return compose(RENDERS / name, fps, name)


def compose(outdir, fps, label):
    video = RENDERS / f"{label}.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps), "-i", str(outdir / "3d_%04d.png"),
            "-framerate", str(fps), "-i", str(outdir / "panel_%04d.png"),
            "-filter_complex", "[0:v][1:v]vstack=inputs=2,format=yuv420p",
            "-c:v", "libx264", "-crf", "18", str(video),
        ],
        check=True,
    )
    return video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="manifest name, e.g. cloud40 or sample19987")
    parser.add_argument("--products", type=Path, default=OUTDIR, help="event products directory")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    outdir, fps = render_panels(args.name, args.products)
    if not args.no_video:
        print(compose_for(args.name, args.products, fps))
