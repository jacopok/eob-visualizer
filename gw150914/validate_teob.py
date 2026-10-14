"""Consistency checks of the TEOBResumS-Dali dynamics extracted in `eobviz/models/teob_dali.py`.

1. Writing the spin dynamics and co-precessing modes to files leaves the waveform
   bit-identical.
2. PE path: the stored complex matched-filter and optimal SNRs (H1, L1) are reproduced
   with TEOB's own polarizations (all inertial modes) and the h_+-maximum time origin;
   the alternative with the m < 0 modes mirrored from the m > 0 ones is worse.
3. Frames: L_hat(0) = z; R(t) z = L_hat(t) for the Euler convention; gamma is the
   minimal-rotation angle; n at the start.
4. The point-particle orbit fed to the multipole formula reproduces h_+, h_x for the
   observer direction and polarization basis adopted (alternatives do worse), for the
   max-likelihood sample and an unequal-mass precessing eccentric test.
5. J: J(t -> -inf) vs J at 15 Hz; J at the peak (TEOB's) vs L + S1 + S2.
6. Ringdown: 220 + 221 fit of the co-precessing h22 from the peak.
"""

import sys
import tempfile
from pathlib import Path

import EOBRun_module
import h5py
import lal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eobviz import physics  # noqa: E402
from eobviz.events import get_event  # noqa: E402
from eobviz.events.gw150914 import frequency_domain_polarizations, polarizations_mirrored  # noqa: E402
from eobviz.models import teob_dali as td  # noqa: E402
from eobviz.multipole import multipole_strain, project  # noqa: E402

EVENT = get_event("GW150914")

OUTDIR = Path(__file__).parent / "validation"
OUTDIR.mkdir(exist_ok=True)


def check_file_output(sample):
    p = td.teob_parameters(sample)
    _, hp, hc, _, _ = EOBRun_module.EOBRunPy(dict(p))
    with tempfile.TemporaryDirectory() as tmp:
        _, hp2, hc2, _, _ = EOBRun_module.EOBRunPy(dict(
            p, output_dynamics="yes", output_multipoles="yes", output_lm=p["use_mode_lm"], output_dir=tmp))
    print(f"[output] waveform with file output bit-identical: {np.array_equal(hp, hp2) and np.array_equal(hc, hc2)}")


def check_snrs(posterior, indices):
    with h5py.File(EVENT.posterior_file, "r") as f:
        stored = {d: f["posterior"][f"{d}_matched_filter_snr"][()] for d in ("H1", "L1")}
    ifos = {d: EVENT.load_interferometer(d) for d in ("H1", "L1")}
    errors = {"native": [], "mirrored": []}
    rho_errors = {"native": [], "mirrored": []}
    shifts = np.arange(-1.0e-3, 1.0e-3, 1e-5)
    for i in indices:
        s = EVENT.sample(posterior, i)
        model = td.run_model(s)
        for name, hpc in (("native", None), ("mirrored", polarizations_mirrored(model, s))):
            line, mismatch = [], np.zeros(len(shifts))
            for d, ifo in ifos.items():
                signal = ifo.get_detector_response(
                    frequency_domain_polarizations(model, ifo.frequency_array, hpc), s)
                # the PE bilby version includes the Tukey window factor in the PSD
                w = np.sqrt(ifo.strain_data.window_factor)
                z = ifo.matched_filter_snr(signal) / w
                rho = np.sqrt(ifo.optimal_snr_squared(signal).real) / w
                mismatch += [abs(ifo.matched_filter_snr(signal * np.exp(-2j * np.pi * ifo.frequency_array * dt)) / w
                                 - stored[d][i]) / abs(stored[d][i]) for dt in shifts]
                rho_errors[name].append(abs(rho / s[d + "_optimal_snr"] - 1))
                line.append(f"{d}: z {z:.2f} (PE {stored[d][i]:.2f}), rho {rho:.2f} (PE {s[d + '_optimal_snr']:.2f})")
            k = np.argmin(mismatch)
            errors[name].append(mismatch[k] / 2)
            print(f"[SNR] sample {i} {name:8s} " + "; ".join(line)
                  + f"; best time shift {shifts[k] * 1e3:+.2f} ms -> |dz| / |z| {mismatch[k] / 2:.4f}")
    for name in errors:
        print(f"[SNR] {name}: |rho / rho_PE - 1| max {np.max(rho_errors[name]):.4f}; "
              f"|z - z_PE| / |z_PE| after the time shift: max {np.max(errors[name]):.4f}")


def check_frames(model, label):
    d = td.inertial_dynamics(model, model.dyn["t"][model.dyn["t"] < model.t_peak])
    for conv in ("zyz-", "zyz"):
        dc = td.inertial_dynamics(model, d["t"], convention=conv)
        cos = np.einsum("ti,ti->t", dc["ez_coprecessing"], dc["LN_hat"])
        print(f"[{label}] Euler {conv}: max angle R z vs L_hat {np.degrees(np.arccos(np.clip(cos.min(), -1, 1))):.1e} deg")
    s = model.spin
    sel = (s["t"] > 100) & (s["t"] < model.t_peak - 50)
    ratio = np.gradient(s["gamma"], s["t"])[sel] / (np.gradient(s["alpha"], s["t"])[sel] * np.cos(s["beta"][sel]))
    print(f"[{label}] gamma_dot / (alpha_dot cos beta): median {np.median(ratio):.6f} (minimal rotation for R = Rz(a) Ry(b) Rz(-g))")
    print(f"[{label}] start: L_hat = {np.round(d['LN_hat'][0], 6)}, n = {np.round(d['n'][0], 4)}, "
          f"r = {d['r'][0]:.3f} M; peak at {model.t_peak:.2f} M, r there {np.interp(model.t_peak, model.dyn['t'], model.dyn['r']):.2f} M, "
          f"PE time origin {model.t_seconds(model.t_peak) * 1e3:+.2f} ms from it")
    ph = np.unwrap(np.angle(model.hP[(2, 2)]))
    orb = np.interp(model.t_modes, model.dyn["t"], model.dyn["phi"])
    k = (model.t_modes > 200) & (model.t_modes < model.t_peak - 200)
    off = ph[k] + 2 * orb[k]
    print(f"[{label}] arg h22^P + 2 phi: {np.mean(off) % (2 * np.pi):.3f} +- {np.std(off):.3f} rad")


def check_orbit_vs_waveform(sample, label):
    run = td.TEOBResumSDali(sample)
    model = run.model
    check_frames(model, label)
    dt = 0.25
    t_geom = np.arange(model.dyn["t"][0] + 200, model.t_peak - 300, dt)
    h_model = CubicSpline(model.t_modes, model.hpc)(t_geom)
    hp_model, hc_model = h_model.real, -h_model.imag
    R_M = sample["luminosity_distance"] * 1e6 * lal.PC_SI / (model.M * lal.MRSUN_SI)
    # TEOB's observer is at (iota, phase): LAL's phi_ref = pi/2 - phase
    N, P, Q = physics.observer_direction_I(model.parameters["inclination"], td.observer_phi_ref(sample))
    candidates = {
        "adopted": ("zyz-", P, Q, 1, 1),
        "Euler zyz (gamma sign)": ("zyz", P, Q, 1, 1),
        "bodies swapped": ("zyz-", P, Q, -1, 1),
        "basis P <-> Q": ("zyz-", Q, P, 1, 1),
        "mirror P -> -P": ("zyz-", -P, Q, 1, 1),
        "sign flipped": ("zyz-", P, Q, 1, -1),
    }
    trim = slice(20, -20)
    results = {}
    for name, (conv, Pc, Qc, swap, sign) in candidates.items():
        dyn = td.inertial_dynamics(model, t_geom, convention=conv)
        h = multipole_strain(swap * dyn["x1"], swap * dyn["x2"], model.m_1, model.m_2, dt, N)
        hp, hc = project(h, Pc, Qc)
        hp, hc = sign * hp / R_M, sign * hc / R_M
        res = np.sqrt(np.sum((hp - hp_model)[trim] ** 2 + (hc - hc_model)[trim] ** 2)
                      / np.sum(hp_model[trim] ** 2 + hc_model[trim] ** 2))
        results[name] = (res, hp, hc)
        print(f"[{label}] {name:24s} relative residual {res:.3f}")

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    tt = model.t_seconds(t_geom)
    sel = tt > tt[-1] - 0.3
    for ax, hm, k in zip(axs, (hp_model, hc_model), (0, 1)):
        ax.plot(tt[sel], hm[sel], "k", lw=2, label="TEOBResumS-Dali")
        for name, (res, *hq) in results.items():
            ax.plot(tt[sel], hq[k][sel], lw=0.8, label=f"{name} ({res:.2f})")
    axs[0].legend(fontsize=7)
    axs[1].set_xlabel("time from geocent_time [s]")
    fig.savefig(OUTDIR / f"orbit_vs_waveform_{label}.png", dpi=120)
    plt.close(fig)
    return run


def check_angular_momentum(run):
    model = run.model
    J0 = td.J_start(model)
    J_inf, _, J_pn = run.J_hat_at_past_infinity()
    J0_hat = J0 / np.linalg.norm(J0)
    print(f"[J] PN J_hat(15 Hz) vs TEOB J_hat(start): {np.degrees(np.arccos(np.clip(J_pn[-1] @ J0_hat, -1, 1))):.2f} deg; "
          f"J(-inf) vs J(15 Hz): {np.degrees(np.arccos(np.clip(J_inf @ J0_hat, -1, 1))):.2f} deg")
    print(f"[J] J at the peak (TEOB's, for the post-merger angles) vs J at the start: "
          f"{np.degrees(np.arccos(np.clip(model.Jfhat_attach @ J0_hat, -1, 1))):.2f} deg")


def check_ringdown(run, label):
    model = run.model
    fit = run.fit_ringdown()
    A220, A221 = fit["amplitudes"]
    t, h, ta = model.t_modes, model.hP[(2, 2)], model.t_attach
    sel = (t >= ta) & (t <= ta + 50)
    b = np.exp(-1j * fit["omegas"][0] * (t[sel] - ta))
    a0 = (b.conj() @ h[sel]) / (b.conj() @ b)
    print(f"[RD] M_f = {model.final_mass:.4f} M, chi_f = {model.final_spin:.4f}; fit with {fit['n_points']} points: "
          f"residual {fit['residual']:.3f} (220 only {np.linalg.norm(a0 * b - h[sel]) / np.linalg.norm(h[sel]):.3f}), "
          f"|A221/A220| = {abs(A221 / A220):.2f}, arg = {np.angle(A221 / A220):.2f}")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    w = (t > ta - 150) & (t < ta + 60)
    tr = np.linspace(ta, ta + 60, 300)
    ax.semilogy(t[w] - ta, np.abs(h[w]), ".-", label=r"$|h_{22}^P|$ TEOBResumS-Dali")
    ax.semilogy(tr - ta, np.abs(np.exp(-1j * np.outer(tr - ta, fit["omegas"])) @ fit["amplitudes"]), "--",
                label="220 + 221 fit")
    ax.set_xlabel(r"$t - t_{\rm peak}$ [M]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"ringdown_{label}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    posterior = EVENT.load_posterior()
    i_maxl = EVENT.max_likelihood_index(posterior)
    sample = EVENT.sample(posterior, i_maxl)
    check_file_output(sample)
    rng = np.random.default_rng(1)
    check_snrs(posterior, [i_maxl] + [int(i) for i in rng.choice(len(posterior), 7, replace=False)])
    run = check_orbit_vs_waveform(sample, "GW150914_maxL")
    check_angular_momentum(run)
    check_ringdown(run, "GW150914_maxL")

    test = dict(sample)
    test.update(mass_1=54.0, mass_2=18.0, a_1=0.7, a_2=0.4, tilt_1=1.2, tilt_2=2.0, theta_jn=1.1,
                phase=0.7, eccentricity=0.15, mean_per_ano=1.0)
    check_orbit_vs_waveform(test, "q3_precessing_eccentric")
