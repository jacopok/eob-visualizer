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

from export_scene import REMNANT_DEFORMATION, SCENES
from ppd import OUTDIR, TIME_RANGE

RENDERS = Path(__file__).parent / "renders"
WIDTH_PX, HEIGHT_PX, DPI = 1280, 320, 100


def render_panels(name):
    manifest = json.load(open(OUTDIR / f"{name}.json"))
    labels = manifest["labels"]
    scene = np.load(SCENES / f"{labels[0]}.npz")
    infos = [json.load(open(SCENES / f"{label}.json")) for label in labels]
    ppd = np.load(OUTDIR / f"ppd_SEOBNRv5PHM_{scene['ifo']}.npz")
    shown = []
    for index in manifest["indices"]:
        where = np.flatnonzero(ppd["sample_indices"] == index)
        if len(where) == 0:
            raise ValueError(f"whitened waveform of sample {index} not in the PPD file")
        shown.append(ppd["waveforms"][where[-1]])
    n = len(shown)

    t_frames = scene["t_det_rel"]
    window = (min(TIME_RANGE[0], t_frames[0]), max(TIME_RANGE[1], t_frames[-1]))
    t = ppd["times"]
    sel = (t >= window[0] - 0.01) & (t <= window[1] + 0.01)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    ax = fig.add_axes([0.06, 0.2, 0.58, 0.72])
    ax.plot(t[sel], ppd["data"][sel], c="0.45", lw=0.8, label=f"{scene['ifo']} data")
    ax.fill_between(t[sel], ppd["lower"][sel], ppd["upper"][sel], color="#56B4E9", alpha=0.7,
                    lw=0, label="SEOBNRv5PHM 90% PPD")
    if n == 1:  # (for a cloud, the coincident sample curves would hide the band)
        ax.plot(t[sel], shown[0][sel], c="w", lw=1.0, label="shown posterior sample")
    t_attach = [info["t_attach_det_rel"] for info in infos]
    ax.axvspan(min(t_attach), max(t_attach) + 1e-4, color="#E69F00", alpha=0.25, lw=0,
               label="attachment (merger) times")
    cursor = ax.axvline(t_frames[0], c="#E69F00", lw=1.5)
    ax.set_xlim(*window)
    ax.set_ylim(-15, 15)
    ax.set_xlabel(f"{scene['ifo']} time $-$ {float(scene['reference_time']):.1f} s [s]", fontsize=9)
    ax.set_ylabel(r"Whitened strain [$\sigma$]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper left", fontsize=7, frameon=False)

    if n == 1:
        s = infos[0]["sample"]
        header = (
            "GW250114, one SEOBNRv5PHM posterior sample\n"
            f"$m_1, m_2$ = {s['mass_1_source']:.1f}, {s['mass_2_source']:.1f} $M_\\odot$ (source)   "
            f"$\\chi_1, \\chi_2$ = {s['a_1']:.2f}, {s['a_2']:.2f}\n"
            f"$\\theta_{{JN}}$ = {np.degrees(s['theta_jn']):.0f}°,  "
            f"remnant $M_f$ = {infos[0]['final_mass_source']:.1f} $M_\\odot$, $\\chi_f$ = {infos[0]['final_spin']:.2f}\n"
        )
        grid = "Grid: plane $\\perp \\hat J(t\\to-\\infty)$, 100 km spacing\n"
    else:
        n_away = manifest["n_J_inf_away_from_observer"]
        header = (
            f"GW250114, {n} SEOBNRv5PHM posterior samples, each in its own\n"
            f"3D orientation. $\\theta_{{JN}}$ is bimodal: {n - n_away} with $J$ towards us,\n"
            f"{n_away} away (mirror images: same sky ellipse, opposite sense)\n"
        )
        grid = ("Grid: plane $\\perp$ mean $\\hat J(t\\to-\\infty)$ (mirrored towards us),\n"
                "100 km spacing\n")
    text = (
        header
        + "View from Earth along the line of sight (orthographic;\nnorth up, east left)\n"
        + grid
        + "Spheres: horizon areal radius; arrows: spins, protruding\n"
        "by $\\chi \\times 200$ km; trails fade over ~1/3 orbit\n"
        f"Remnant: $\\ell=m=2$ ringdown, 220 + 221 QNMs fitted to each\nsample's waveform "
        f"(shape amplitude illustrative, {REMNANT_DEFORMATION:.0%} at start)\n"
        f"Playback {int(scene['slowdown'])}× slower than real time"
    )
    fig.text(0.665, 0.95, text, fontsize=7, va="top", ha="left", color="0.85", linespacing=1.25)

    outdir = RENDERS / name
    outdir.mkdir(parents=True, exist_ok=True)
    for f, tf in enumerate(t_frames):
        cursor.set_xdata([tf, tf])
        fig.savefig(outdir / f"panel_{f:04d}.png", dpi=DPI)
    plt.close(fig)
    return outdir, int(scene["fps"])


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
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    outdir, fps = render_panels(args.name)
    if not args.no_video:
        print(compose(outdir, fps, args.name))
