# GW150914 in Blender: TEOBResumS-Dali (eccentric, precessing) dynamics of posterior samples

The GW250114 pipeline (`../gw250114/`) applied to the eccentric-precessing TEOBResumS-Dali
bilby samples of GW150914 from Gamba et al., CQG 10.1088/1361-6382/adfe50
(`gw150914_teob_prc_ecc_allmodes_15_..._result.hdf5`: f_min = f_ref = 15 Hz,
modes (2,2), (2,1), (3,3), (4,4), eccentricity and mean anomaly at 15 Hz).

```bash
uv sync --python 3.10            # TEOBResumS from bitbucket tag 2505.21612; see "Environment"
.venv/bin/python validate_teob.py                      # consistency checks -> validation/
cd ..                                                  # the pipeline runs from the repository root
python -m eobviz ppd GW150914                          # whitened H1 PPD (500 samples) -> products/
python -m eobviz export GW150914 --n-samples 200       # scene data -> products/scenes/, products/cloud200.json
python -m eobviz camera GW150914 --name cloud200 --shot orbit
python -m eobviz render GW150914 --name cloud200 --style solid_highlight
python -m eobviz panel GW150914 --name cloud200        # -> renders/cloud200.mp4
python -m eobviz scene GW150914 --name cloud200        # explore the scene in Blender
```

The TEOBResumS call is in `eobviz/models/teob_dali.py`, the event's settings and data in
`eobviz/events/gw150914.py`.

## How PE called the model, and how we reproduce it

The run used bilby's `gwsignal_eccentric_binary_black_hole` (bilby fork 6028debe) and
gwsignal's `TEOBResumSDALI` generator (lalsuite `teobresums-dali`, 37da46cd, built from a
modified tree), with gwsignal's conditioning:

- TEOBResumS is called at f22_start = 15 Hz: for models with spins and eccentricity
  defined at the start, gwsignal does not lower f_min (its "fallback" routine), so the
  sample's spins (`bilby_to_lalsimulation_spins` at 15 Hz), eccentricity and anomaly
  (mean → true, as gwsignal) are the initial data of the EOB evolution.
- Δt = 1/2048 s (896 Hz rounded up to a power-of-two multiple of 1/8 Hz); h₊, h× are
  tapered at the start (XLALSimInspiralREAL8WaveTaper, as ported in gwsignal), zero-padded
  to 8 s and FFT'd.
- **Time origin**: gwsignal's `resize_gwpy_timeseries` relabels the time axis so that the
  maximum of h₊ (signed, not |h|) is at t = 0, and bilby puts that sample at
  `geocent_time`. So `geocent_time` is *not* the peak of the waveform amplitude; for the
  max-likelihood sample the co-precessing A₂₂ peak (TEOB's own t = 0) is 2.8 ms later.
- **Polarizations**: TEOBResumS's own h₊, h× = −(νM/D) Σ h_lm ₋₂Y_lm(ι, φ), with all the
  inertial modes it returns (m > 0, m < 0, m = 0) and the observer at azimuth φ = `phase`
  (LAL convention with φ_ref = π/2 − phase; the −1 is LAL's tetrad sign). The 37da46cd
  wrapper as committed cannot parse the m ≤ 0 keys TEOB returns for precessing runs, so
  we checked this against the PE output (below), rather than reading it off the code.

## Dynamics (`eobviz/models/teob_dali.py`)

- TEOBResumS-Dali at the paper's version (bitbucket tag `2505.21612`, commit 6558331).
  Its Python interface returns the co-precessing EOB dynamics (r, φ, p_φ, Ω, …), the
  inertial modes and h₊, h×; we also ask it to write its spin dynamics (S₁, S₂, L̂ and the
  Euler angles α, β, γ on the EOB time axis) and its co-precessing modes to files. This
  leaves the waveform bit-identical.
- I-frame: TEOB's, z along L̂ at 15 Hz (LAL's L0 frame). The orbit is
  n(t) = R_z(α) R_y(β) R_z(−γ) (cos φ, sin φ, 0): R z = L̂ to 2e-6°, and γ̇ = +α̇ cos β, so
  this is the minimal-rotation frame. Bodies at x₁ = (m₂/M) r n, x₂ = −(m₁/M) r n, with the
  EOB r (it oscillates: the orbit is eccentric). Spins: TEOB's evolved S₁, S₂.
- Dali's twist interpolates the Euler angles in time up to the co-precessing A₂₂ peak and
  continues them with a QNM prescription about J_mrg = S₁ + S₂ + L_2PN L̂. We take that
  peak as the merger ("attachment"; the EOB dynamics continue ~9 M to r = 1.5), and
  J_mrg as the remnant spin direction; M_f and χ_f are TEOB's.
- **Ringdown**: 220 + 221 fit of the co-precessing (2,2) mode from the peak (where TEOB
  attaches its QNM template); then as for GW250114. Residual 0.29 for the max-L sample
  (220 alone 0.69), |A₂₂₁/A₂₂₀| = 1.39.
- **Kick**: NRSur7dq4Remnant from the spins and orbit at −100 M from the peak (a
  quasi-circular fit; the residual eccentricity at merger is negligible).
- **J(t → −∞)**: the spins at 15 Hz evolved backwards with pyseobnr's PN precession
  equations (quasi-circular: eccentricity neglected there). 0.3° from J(15 Hz).

## Checks (`validate_teob.py`, log in `validation/`)

1. File output leaves h₊, h× bit-identical.
2. PE path, 8 samples: with TEOB's own polarizations the stored optimal SNRs (H1, L1) are
   reproduced to ≤ 0.5%, and the complex matched-filter SNRs to ≤ 0.5% after a time shift
   of −0.1 ms or +0.4 ms (≤ one 2048 Hz sample: which sample is the maximum of h₊ is
   not quite the same as in PE, whose TEOB build is recorded only as "teobresums 0.0.2").
   Building the m < 0 modes from the m > 0 ones instead misses the optimal SNRs by up to
   19%. (The PE bilby version folded the Tukey window factor into the PSD; we divide by
   its square root.)
3. Multipole formula (mass quadrupole + octupole + current quadrupole) on the orbit vs
   h₊, h×: relative residual 0.26 for the adopted conventions, ≥ 1.4 for the other γ sign,
   swapped/mirrored polarization basis or flipped sign; 0.29 vs 0.35 with the bodies
   swapped in a q = 3, e = 0.15 precessing test. The residual is the 0.24 rad PN/tail
   phase between the co-precessing h₂₂ and −2φ (arg h₂₂ᴾ + 2φ = π − 0.24 ± 0.05 rad).

## Other differences from GW250114

- Panel: whitened **H1** strain (the louder detector), GWOSC 4 kHz R1 data, 8 s segment
  ending 2 s after the trigger, GWTC-1 PSDs, 20–896 Hz, Tukey roll-off 0.4 s with its
  sqrt(β); per-sample spline calibration as in PE.
- Window: H1 time −0.48 to 0.175 s around 1126259462.4 (merger at ≈ +0.02 s), 100× slower.

## Environment

Its own uv project (same pins as `../gw250114`, plus TEOBResumS, built from the bitbucket
tag; needs GSL headers). pyseobnr (used for the PN backward evolution) needs the same
rebuild as in `../gw250114/README.md`:

```bash
uv pip install --no-cache "cython>=3.0.11" setuptools-scm "setuptools>=78,<80" wheel
uv pip install --no-cache --no-build-isolation --reinstall --no-deps pyseobnr==0.3.5
```

`python -m eobviz` switches to this environment by itself for this event.
