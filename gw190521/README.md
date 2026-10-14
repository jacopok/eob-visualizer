# GW190521 in Blender: SEOBNRv4PHM dynamics of posterior samples

The GW250114 pipeline (`../gw250114/`) applied to the GWTC-2.1 SEOBNRv4PHM samples of
GW190521 (`C01:SEOBNRv4PHM` in `../IGWN-GWTC2p1-v2-GW190521_030229_PEDataRelease_mixed_nocosmo.h5`).
Uses the GW250114 environment (`../gw250114/.venv`, see its README).

```bash
PY=../gw250114/.venv/bin/python
$PY validate_v4phm.py                    # consistency checks -> validation/
# data/: GWOSC 4 kHz strain, 64 s around the event (gwpy TimeSeries.fetch_open_data)
$PY reconstruct_phase.py                 # coalescence phases, see below -> products/phase_reconstruction.npz
cd ..                                    # the pipeline runs from the repository root
python -m eobviz ppd GW190521            # whitened L1 PPD (500 samples) -> products/
python -m eobviz export GW190521 --n-samples 200   # scene data -> products/scenes/, products/cloud200.json
python -m eobviz camera GW190521 --name cloud200
python -m eobviz render GW190521 --name cloud200
python -m eobviz panel GW190521 --name cloud200    # -> renders/cloud200.mp4
python -m eobviz scene GW190521 --name cloud200    # explore the scene in Blender (--draft: fast, plain look)
```

The event's settings, data and phase handling are in `eobviz/events/gw190521.py`.

## Dynamics (`eobviz/models/seobnrv4.py`)

- SEOBNRv4PHM from LALSimulation, called as in PE (ChooseTDWaveform, f_min = 5 Hz as the
  run's fmin-template, 512 Hz): v4PHM ignores f_ref, so the samples' Cartesian spins are
  those at 5 Hz. ChooseTDWaveform calls `SimIMRSpinPrecEOBWaveformAll`, which also returns
  the dynamics (separation, momenta, spins in the LAL L0 frame of f_min), the modes in the
  I- and final-J frames and the merger quantities (t_attach, final J, M_f, chi_f). We call it
  directly with the same flags; this reproduces the ChooseTDWaveform polarizations to
  1e-16 (up to the sign from its polarization rotation by -pi/2).
- Checks (`validate_v4phm.py`): n = x, L_N = z at the start; waveform t = 0 at the model's
  peak time; the multipole formula on the orbit reproduces h_+, h_x for the same observer
  direction and polarization basis as GW250114 (residual 0.33, alternatives > 1.27, mainly a
  ~0.1 rad PN phase lag) and identifies the bodies in an unequal-mass test (0.33 vs 0.41);
  J(t_attach) vs final J: 0.4°; J(-inf) from backward PN evolution vs J(f_min): 0.6°;
  220 + 221 fit of h22^J from t_attach: residual 0.22 (220 alone 0.63), |A221/A220| = 1.24.
- J(t -> -inf) uses pyseobnr's PN precession equations (v5 family) starting from the
  v4PHM spins at 5 Hz.

## Coalescence phase and time: not usable in the release

In this (RIFT) run every sample has the same `geocent_time`, and the stored `phase` is
uniformly distributed: time and phase were marginalized over. Whitened waveforms built with
the stored phases are out of phase with the data by up to ~2 rad, differently for each
sample, and no global convention change (phase -> ±phase + k pi/2, psi -> ±psi + {0, pi/2})
fixes it; with the phase re-optimized at the stored time, every sample matches the data
within 0.6 ms.

So, as an approximation, `reconstruct_phase.py` draws each sample's phase from its
conditional posterior given the H1 + L1 data at the stored time (release PSDs, 20–224 Hz,
no calibration, no Virgo). The phase only sets the observer's azimuth in v4PHM
(polarizations = -(M/D) sum_lm h^I_lm _{-2}Y_lm(iota, pi/2 - phase), checked to machine
precision), so it also sets the orientation of each sample in the sky frame. The
conditional posteriors are narrow (median rms width 5°). The time is left at the stored value.
A proper treatment would draw time and phase jointly (and check RIFT's conventions).

## Other differences from GW250114

- 4 s segment at 512 Hz around the trigger, release PSDs, band 20–224 Hz, Tukey roll-off
  0.4 s (with the corresponding sqrt(beta)); no calibration model (none in the release).
- Window: L1 time −0.95 to 0.15 s around 1242442967.4 (median merger at +0.026 s),
  played back 50 times slower (the system is ~4 times heavier than GW250114).
- The posterior is much broader: merger times spread over ~43 ms in L1, J(−∞) directions
  over tens of degrees, and early in the window the bodies are spread around the whole
  orbit (the data constrain only the last few cycles), so the "cloud" only condenses
  towards the merger, and the close-up is wide.
