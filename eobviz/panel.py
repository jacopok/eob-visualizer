"""Render the whitened-strain panel frames (PPD band, data, shown samples, time cursor)
and stack them under the Blender frames into a video.

Frames are those of the camera path (`<name>_camera.npz`): with a storyboard the cursor
stops while the evolution pauses, and the captions (LaTeX) are written over the 3D view
from left to right, manim-style, and fade out.

Usage: python -m eobviz panel EVENT --name cloud200   (products/<name>.json from export.py)
"""

import json
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WIDTH_PX, HEIGHT_PX, DPI = 1920, 270, 100
WINDOW_BEFORE, WINDOW_AFTER = 13.0, 6.0  # scrolling window around the cursor [s of video]
VIEW_HEIGHT_PX = 810  # of the 3D view
CAPTION_X_PX, CAPTION_Y_PX = 60, (30, 115, 180, 700, 180)  # top left corner of the caption rows
CAPTION_SIZE = (26, 22, 24, 26, 24)  # font size [pt] of the rows
# row 3: centred on x, under the sets; row 4: as row 2, in the colour of the remnant
CAPTION_COLOR = ("white", "0.8", "#FFCC33", "#FFCC33", "#33DD77")
WRITE_S, FADE_S, WIPE_EDGE_PX = 0.8, 0.4, 60  # write-in and fade-out times, soft edge of the wipe


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


def render_panels(event, name, draft=False):
    products = event.products
    manifest = json.load(open(products / f"{name}.json"))
    labels = manifest["labels"]
    scenes = products / "scenes"
    with np.load(scenes / f"{labels[0]}.npz") as f:
        scene = {k: f[k] for k in ("t_det_rel", "slowdown", "ifo", "reference_time", "fps")}
    camera = products / f"{name}_camera.npz"
    if camera.exists():
        with np.load(camera) as f:
            t_frames = f["t_det_rel"]
            fps = float(f["fps"]) if "fps" in f.files else float(scene["fps"])
    else:
        t_frames, fps = scene["t_det_rel"], float(scene["fps"])
    infos = [json.load(open(scenes / f"{label}.json")) for label in labels]
    approximant = manifest.get("approximant", event.approximant)
    ppd, shown = load_ppd(products / manifest.get("ppd_file", event.ppd_file().name), manifest["indices"])
    before = WINDOW_BEFORE / float(scene["slowdown"])
    after = WINDOW_AFTER / float(scene["slowdown"])
    n = len(shown)

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

    outdir = event.renders / (name + ("_draft" if draft else ""))
    outdir.mkdir(parents=True, exist_ok=True)
    for f, tf in enumerate(t_frames):
        if f > 0 and tf == t_frames[f - 1]:  # paused
            shutil.copyfile(outdir / f"panel_{f - 1:04d}.png", outdir / f"panel_{f:04d}.png")
            continue
        cursor.set_xdata([tf, tf])
        ax.set_xlim(tf - before, tf + after)
        fig.savefig(outdir / f"panel_{f:04d}.png", dpi=DPI)
    plt.close(fig)
    return outdir, fps


def caption_image(text, size, color):
    """RGBA image (float, 0..1) of a LaTeX caption, cropped to the text."""
    from io import BytesIO

    from PIL import Image

    with plt.rc_context({"text.usetex": True, "font.family": "serif",
                         "text.latex.preamble": r"\usepackage{amsmath}"}):
        fig = plt.figure(figsize=(19, 1.2), dpi=DPI)
        fig.text(0.0, 0.5, text, fontsize=size, color=color, va="center")
        buf = BytesIO()
        fig.savefig(buf, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGBA"), float) / 255


def overlay(frame_rgb, captions, images, v):
    """The 3D frame with the captions shown at video time v."""
    out = frame_rgb
    for c, img in zip(captions, images):
        if not c["t_in"] <= v <= c["t_out"]:
            continue
        h, w = img.shape[:2]
        x0, y0 = CAPTION_X_PX, CAPTION_Y_PX[c["row"]]
        if c.get("x", -1) >= 0:
            x0 = int(c["x"] * out.shape[1] - w / 2)
        h, w = min(h, out.shape[0] - y0), min(w, out.shape[1] - x0)
        reveal = np.clip((v - c["t_in"]) / WRITE_S, 0, 1) * (w + WIPE_EDGE_PX)
        wipe = np.clip((reveal - np.arange(w)) / WIPE_EDGE_PX, 0, 1)
        fade = np.clip((c["t_out"] - v) / FADE_S, 0, 1)
        alpha = img[:h, :w, 3] * wipe[None, :] * fade
        region = out[y0:y0 + h, x0:x0 + w]
        out[y0:y0 + h, x0:x0 + w] = region * (1 - alpha[..., None]) + img[:h, :w, :3] * alpha[..., None]
    return out


def compose_with_captions(frames, fps, captions):
    """Frames (frame_####.png) of the 3D view with the captions, over the panel."""
    from PIL import Image

    images = [caption_image(c["text"], CAPTION_SIZE[c["row"]], CAPTION_COLOR[c["row"]]) for c in captions]
    for path in sorted(frames.glob("3d_*.png")):
        f = int(path.stem.removeprefix("3d_"))
        view = np.asarray(Image.open(path).convert("RGB"), float) / 255
        view = overlay(view, captions, images, f / fps)
        panel = np.asarray(Image.open(frames / f"panel_{f:04d}.png").convert("RGB"), float) / 255
        Image.fromarray((np.vstack([view, panel]) * 255).round().astype(np.uint8)).save(frames / f"frame_{f:04d}.png")


def load_captions(products, name):
    camera = products / f"{name}_camera.npz"
    if not camera.exists():
        return []
    with np.load(camera) as f:
        if "caption_text" not in f.files:
            return []
        return [dict(text=str(text), row=int(row), t_in=float(t_in), t_out=float(t_out), x=float(x))
                for text, row, t_in, t_out, x in zip(f["caption_text"], f["caption_row"], f["caption_in"],
                                                     f["caption_out"], f["caption_x"])]


def compose(frames, fps, video, captions=()):
    """Stack the 3D frames (3d_####.png) over the panel frames (panel_####.png) of the
    directory `frames` into `video`, with the captions over the 3D view."""
    if captions:
        compose_with_captions(frames, fps, captions)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", str(frames / "frame_%04d.png"), "-vf", "format=yuv420p",
                        "-c:v", "libx264", "-crf", "18", str(video)], check=True)
        return video
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
