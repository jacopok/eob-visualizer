"""Render the whitened-strain panel frames (PPD band, data, shown samples, time cursor)
and stack them under the Blender frames into a video.

Usage: python -m eobviz panel EVENT --name cloud200   (products/<name>.json from export.py)
"""

import json
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WIDTH_PX, HEIGHT_PX, DPI = 1920, 270, 100
WINDOW_BEFORE, WINDOW_AFTER = 13.0, 6.0  # scrolling window around the cursor [s of video]


def load_ppd(path, indices):
    """The PPD arrays, and the whitened waveforms of the posterior samples `indices`.

    Each access to an array of an open NpzFile reads it from disk again, and a row of
    it keeps the whole array alive: indexing `waveforms` once per shown sample used to
    hold one full copy (up to 130 MB) per sample, ~26 GB for 200 GW150914 samples.
    """
    with np.load(path) as f:
        ppd = {k: f[k] for k in f.files if k != "waveforms"}
        rows = []
        for index in indices:
            where = np.flatnonzero(ppd["sample_indices"] == index)
            if len(where) == 0:
                raise ValueError(f"whitened waveform of sample {index} not in the PPD file")
            rows.append(where[-1])
        shown = f["waveforms"][rows]
    return ppd, shown


def render_panels(event, name):
    products = event.products
    manifest = json.load(open(products / f"{name}.json"))
    labels = manifest["labels"]
    scenes = products / "scenes"
    with np.load(scenes / f"{labels[0]}.npz") as f:
        scene = {k: f[k] for k in ("t_det_rel", "slowdown", "ifo", "reference_time", "fps")}
    infos = [json.load(open(scenes / f"{label}.json")) for label in labels]
    approximant = manifest.get("approximant", event.approximant)
    ppd, shown = load_ppd(products / manifest.get("ppd_file", event.ppd_file().name), manifest["indices"])
    before = WINDOW_BEFORE / float(scene["slowdown"])
    after = WINDOW_AFTER / float(scene["slowdown"])
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

    outdir = event.renders / name
    outdir.mkdir(parents=True, exist_ok=True)
    for f, tf in enumerate(t_frames):
        cursor.set_xdata([tf, tf])
        ax.set_xlim(tf - before, tf + after)
        fig.savefig(outdir / f"panel_{f:04d}.png", dpi=DPI)
    plt.close(fig)
    return outdir, int(scene["fps"])


def compose(frames, fps, video):
    """Stack the 3D frames (3d_####.png) over the panel frames (panel_####.png) of the
    directory `frames` into `video`."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps), "-i", str(frames / "3d_%04d.png"),
            "-framerate", str(fps), "-i", str(frames / "panel_%04d.png"),
            "-filter_complex", "[0:v][1:v]vstack=inputs=2,format=yuv420p",
            "-c:v", "libx264", "-crf", "18", str(video),
        ],
        check=True,
    )
    return video
