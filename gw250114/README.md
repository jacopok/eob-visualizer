# GW250114 in Blender: SEOBNRv5PHM dynamics of posterior samples

A cloud of posterior samples of the SEOBNRv5PHM PE run (or a single one), orbiting,
merging and ringing down, viewed from Earth.

## Pipeline

```bash
uv sync --python 3.10            # see "Environment" below for the pyseobnr rebuild
.venv/bin/python validate_dynamics.py                  # consistency checks -> validation/
cd ..                                                  # the pipeline runs from the repository root
python -m eobviz ppd GW250114                          # whitened L1 PPD (500 samples + max L) -> products/
python -m eobviz export GW250114 --n-samples 200       # per-frame scene data -> products/scenes/, products/cloud200.json
                                                       # (no argument: max-likelihood sample only -> products/sample<i>.json)
python -m eobviz camera GW250114 --name cloud200       # camera path -> products/cloud200_camera.npz
python -m eobviz render GW250114 --name cloud200 --style solid_highlight --save-blend   # -> renders/cloud200/
python -m eobviz panel GW250114 --name cloud200        # panel frames + final video renders/cloud200.mp4
python -m eobviz scene GW250114 --name cloud200        # explore the scene in Blender, back and forth in time (--draft: fast, plain look)
```

The explanatory video (`eobviz/story.py`): 400 samples, pausing to explain and fold the
ψ and θ_JN copies and to relabel the BHs by orbit clustering. A quick low-frame-rate draft:

```
python -m eobviz export GW250114 --n-samples 400
python -m eobviz camera GW250114 --name cloud400 --story degeneracies --fps 10
python -m eobviz render GW250114 --name cloud400 --draft   # -> renders/cloud400_draft/
python -m eobviz panel GW250114 --name cloud400 --draft    # -> renders/cloud400_draft.mp4
```
(full quality: `camera` without `--fps`, `render` and `panel` without `--draft`).

The code is in `../eobviz` (see the repository README); `python -m eobviz` switches to this
directory's `.venv` by itself. The SEOBNRv5PHM call is in `eobviz/models/seobnrv5.py`, the
event's settings and data in `eobviz/events/gw250114.py`. The same pipeline serves GW190521
(`../gw190521/README.md`) and GW150914 (TEOBResumS-Dali, eccentric and precessing:
`../gw150914/README.md`).

## What is shown, and how it is computed

- **Dynamics**: SEOBNRv5PHM (pyseobnr 0.3.5, the version used in PE) is called with exactly
  the arguments bilby's `gwsignal_binary_black_hole` used in the likelihood (f_ref = 20 Hz,
  f_min = 10 Hz lowered by the conditioning, `lmax_nyquist=1`). The orbit is the EOB
  relative orbit, `n(t) = R(t) (cos phi, sin phi, 0)` with `R` the co-precessing → inertial
  rotation used to twist up the modes; the bodies sit at `x1 = (m2/M) r n`, `x2 = -(m1/M) r n`.
  Spins are pyseobnr's PN-evolved spins (as functions of the EOB orbital frequency).
- **Samples**: the cloud is the first N of the 500 random posterior draws used for the PPD.
  Every sample keeps its own 3D orientation. With `--style solid_highlight`, horizons are
  opaque black with coloured rims, spin arrows protrude by 400 km × |χ|, and the trails are
  thin and alpha-blended (they saturate to the body colour, independent of the drawing
  order); the max-likelihood sample is drawn in lighter colours with longer, opaque trails. With
  q ≈ 1, which body is the heavier (orange) one varies across samples.
- **Trails** fade exponentially with the orbital phase elapsed since the body was there
  (e-folding 0.3 orbits) and are cut off smoothly between 0.4 and 0.6 orbits; after the
  attachment the phase keeps running at the attachment orbital frequency.
- **Time window**: L1 time −0.29 to 0.37 s (merger at ≈0.217), i.e. from 0.5 s before merger,
  played back 100 times slower; the panel scrolls with a fixed 0.19 s window.
- **Camera** (`eobviz/camera.py`): starts on the line of sight from Earth, far away (nearly
  orthographic), framing the orbits and slowly zooming in as they shrink; then flies to a
  close-up that follows one cluster of bodies (those of the majority θ_JN family near the
  same body of a reference sample) from above its orbital plane, co-rotating with it; then
  back to the line of sight for the merger and ringdown. Whenever the camera is off the line
  of sight, a camera icon on it, joined to the centre of mass and labelled "to Earth", marks
  the observer's direction, and the panel says how far off-axis the camera is.
- **Storyboard** (`--story degeneracies`, `eobviz/story.py`): early in the inspiral the
  evolution slows to a stop and the camera turns to a side view in which the four
  sky-orientation copies (ψ or ψ + π, θ_JN or π − θ_JN) are apart. Captions (LaTeX, written
  in manim-style) give the reason for each copy; the two sides of each fold move apart to
  the left and right of the view, labelled, then the discarded side fades out and the kept
  one comes back: the cloud is restricted to 0 < ψ < π and 0 < θ_JN < π/2. Last, the BHs of
  the canonical samples are relabelled by orbit clustering (`eobviz/labels.py`, as in
  `relabel.py` but within the canonical domain): the samples labelled against the majority
  by m_1 ≥ m_2 move to the right, swap colours and come back. The camera goes back to the
  line of sight, the evolution resumes with the canonical set only, and the close-up follows
  one of its clusters; the other copies fade back in after the merger.
- **After the attachment time** the remnant is shown with pyseobnr's final mass and spin,
  the spin along J at attachment (as modelled by pyseobnr). It recoils with the
  NRSur7dq4Remnant kick (spins and orbit at -100 M from the peak) from the attachment on.
- **Ringdown**: the remnant's surface is `r = R_f [1 + Re(C(t) e^{-2i phi}) sin^2 theta]`
  around its spin axis, `C(t) = 0.3 conj(h(t)/h(t_attach))`, with `h` the 220 + 221 Kerr-QNM
  fit (frequencies from `qnm` for each sample's M_f, chi_f) to that sample's own (2,2) mode in
  the final-J frame, from t_attach. The l = m = 2 bar starts along the separation of the
  two BHs at attachment and then rotates with the pattern speed omega_R/2 and decays with
  the two damping times, with the fitted 221/220 amplitude ratio (≈1.25) and phase. This
  assumes the horizon's l = m = 2 shape rings with the same relative mode content as the
  waveform; the overall amplitude (30 %) is illustrative, and spheroidal vs spherical
  harmonics and the retrograde/mirror modes are neglected.
- **Sizes**: spheres have the Kerr horizon areal radius `sqrt(A/4pi)`; lengths in source-frame km.
  EOB coordinates are gauge dependent, so positions/sizes near merger are indicative.
- **Frame**: sky frame (X west, Y north, Z towards Earth). The LAL I-frame is mapped
  onto it so that the model's polarization basis goes onto bilby's `(m, n)` basis for the
  sample's `psi`, and the direction of the observer onto +Z.
- **Grid**: plane through the centre of mass perpendicular to J at t → −∞, obtained by
  integrating pyseobnr's PN precession equations backwards from f_ref (it differs from
  J(f_ref) by ~0.1° for the max-L sample). For a cloud, the mean of the samples' J(−∞),
  after reflecting through the sky plane those pointing away from us: θ_JN is bimodal
  (≈45° or ≈135°), and the two families are mirror images through the sky plane, which
  look the same in an orthographic view along the line of sight (same projected
  ellipses, opposite senses of rotation). The grid only adds light, so it never hides
  bodies on "the wrong side" of it.
- **ψ vs ψ + π**: the two give identical detector responses, i.e. the whole configuration
  rotated by 180° about the line of sight is equally allowed. bilby samples ψ ∈ [0, π),
  an arbitrary cut; each sample is shown at ψ or ψ + π with probability ½ (seeded by its
  posterior index), so the cloud shows the full uncertainty. The two halves' J(−∞) cancel
  on the sky, so the grid ends up nearly face-on.
- **Time**: frames are uniform in L1 detector time; the binary state shown at `t_det`
  is the one at waveform time `t_det − geocent_time − Δt_L1`.
- **Panel**: whitened L1 strain as in `waveform_reconstruction.py` of the GW250114 paper
  (same data dump, whitening and `sqrt(beta)` factor, 90% band from 500 samples); waveforms
  from SEOBNRv5PHM samples with PE settings and per-sample calibration. The data dump of the
  NRSur7dq4 run is used: its PSDs are identical to the SEOBNRv5PHM run's.

## Checks (`validate_dynamics.py`)

1. `R(t) z` coincides with L_N(t) (to 0.002°); at t_ref, `n = x` and `L_N = z` (LAL convention).
2. h+ and hx computed from the point-mass orbit with the Cartesian multipole formula
   (mass quadrupole + octupole + current quadrupole) match the model's (≈20% residual, from PN
   corrections not in the leading-order formula); other polarization bases give >130%;
   swapping the bodies is disfavoured in an unequal-mass test (q=3).
3. bilby's frequency-domain L1/H1 response matches the time-domain polarizations placed at
   `geocent_time + delay` with zero lag.
4. `python -m eobviz ppd GW250114 --approximant NRSur7dq4` reproduces the paper's reconstruction.
5. Ringdown (max-L and random samples): a 220 + 221 fit of h22 in the final-J frame from
   t_attach has a relative residual of 0.18–0.21 (220 alone: 0.54), almost all within the
   first few M, where the model's amplitude has a flat peak that no QNM sum reproduces;
   |A221/A220| ≈ 1.25 is stable against the fit length. In the inspiral,
   2 φ_n + arg h22^J drifts by ~1 rad over the last 20 M, so the bar is anchored to the
   separation exactly at attachment. Running pyseobnr with mode output
   (needed for the J-frame modes) leaves the dynamics bit-identical.

## Environment

`pyproject.toml` pins the PE-run versions. pyseobnr is built from an sdist and must be
compiled against the runtime scipy; if `import pyseobnr` fails with a `cython_special` error:

```bash
uv pip install --no-cache "cython>=3.0.11" setuptools-scm "setuptools>=78,<80" wheel
uv pip install --no-cache --no-build-isolation --reinstall --no-deps pyseobnr==0.3.5
```

Blender ≥ 5.2 (EEVEE). ffmpeg for the final video.
