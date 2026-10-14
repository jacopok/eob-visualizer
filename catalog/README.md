# Orbit-based BH relabeling across the GWTC catalog

Runs the relabeling of `gw250114/relabel.py` on every event with a supported EOB run:
each posterior sample's "BH 1" becomes the body whose orbit, in the sky frame and on a
common geocentric time axis, agrees with BH 1 in the other samples (Z2 sign
synchronization of u(t) = n̂(t); see the docstrings of `relabel.py` and `relabel_catalog.py`).

Uses the `gw250114` uv environment:

```sh
R="uv run --project ../gw250114 python"
$R catalog.py manifest                      # 282 events on Zenodo, 105 GB in full
$R catalog.py fetch --release GWTC-4.1      # download, keep only EOB runs (~30 MB/event)
$R catalog.py inventory                     # data/inventory.csv: pipeline, settings, support
$R relabel_catalog.py run --n 1000          # all supported events
$R relabel_catalog.py summary
# full posterior on a wide grid, then compare windows on the cached u(t):
$R relabel_catalog.py run --events GW231231_154016 --n 0 --grid -4000 50 --step 1
$R relabel_catalog.py windows --events GW231231_154016 --grid -4000 50 --step 1 --windows -350:0 -1500:0
```

All steps resume: downloaded slim files and cached u(t) are skipped.

## What is supported

| Release  | Runs                    | Pipeline              | Status |
|----------|-------------------------|-----------------------|--------|
| GWTC-4.1 | SEOBNRv5PHM (~88 events) | bilby + gwsignal      | supported: same waveform call as GW250114, settings read from each run's config |
| GWTC-5.0 | SEOBNRv5PHM (~104)       | RIFT                  | needs RIFT's waveform call and phase/time/ψ conventions reproduced and validated |
| GWTC-3, GWTC-2.1 | SEOBNRv4PHM (~90) | RIFT or LALInference | needs the v4PHM dynamics generalized from `eobviz/models/seobnrv4.py`, and the same validation |

Only GWTC-4.1 can run now. A new pipeline goes in `Run.compute_u`, and `catalog.supported`
gets switched on for it, but only after its waveform reproduces the PE one, as checked in
`gw250114/validate_dynamics.py` and `gw190521/validate_v4phm.py`. Using the SEOBNRv5PHM
dynamics with another pipeline's phase and time conventions would put the bodies in the
wrong place on the sky. The clustering wouldn't catch this; it would just produce wrong labels.

## Cost

About 0.8 s of wall time per sample on 8 cores for a ~50 M☉ binary. Lighter binaries
have longer dynamics and take more. 1000 samples per event × 88 events ≈ 20 h.

## Reading the summary

`products/summary.csv`, one row per event:
- `relabeled`: fraction of samples swapped relative to m₁ ≥ m₂;
- `score_50`, `score_5`, `ambiguous` (score < 0.3): how cleanly the orbits fall into
  one cluster. Low scores mean the orbital phase is poorly measured. The labeling is
  still valid, but it's close to arbitrary for those samples;
- `score_50_null`: the median score after rotating each sample's orbit by a random angle
  about the line of sight. When `score_50` is close to it, the orbit's orientation isn't
  measured and the labels are arbitrary. Examples: GW250114 scores 0.87 against a null of
  0.52; GW231231_154016 (H1 only, SNR 13) scores 0.46 against 0.37, and its labels change
  with the window;
- `psi_flipped`: samples moved to the ψ + π representative before clustering.

The analysis window is the last 350 M before the median peak. Starting it much earlier
would split sub-populations that agree in phase at merger but dephase going back in time
(as seen for GW250114).
