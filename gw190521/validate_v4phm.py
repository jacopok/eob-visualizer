"""Consistency checks of the SEOBNRv4PHM dynamics extracted in `eob_v4phm.py`.

1. SimIMRSpinPrecEOBWaveformAll with our flags reproduces ChooseTDWaveform (PE path).
2. At the start of the dynamics n = x and L_N = z (LAL L0 frame of f_min); the
   waveform's t = 0 is the model's peak time.
3. The point-particle orbit fed to the multipole formula reproduces h_+, h_x for the
   observer direction and basis of `observer_direction_I` (alternatives do worse),
   for the max-likelihood sample and an unequal-mass precessing test.
4. J(t) from the dynamics vs the final J; J(t -> -inf) vs J at f_min.
5. Ringdown: 220 + 221 fit of h22 in the final-J frame, and the phase relation between
   h22^J and the orbit in the late inspiral.
"""

import sys
from pathlib import Path

import lal
import lalsimulation as ls
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

import eob_v4phm as ev

sys.path.insert(0, str(Path(__file__).parent.parent / "gw250114"))
import eob_dynamics as ed  # noqa: E402
from validate_dynamics import multipole_strain, project  # noqa: E402

OUTDIR = Path(__file__).parent / "validation"
OUTDIR.mkdir(exist_ok=True)


def check_choose_td(sample, model):
    S1 = [sample[f"spin_1{c}"] for c in "xyz"]
    S2 = [sample[f"spin_2{c}"] for c in "xyz"]
    hp, hc = ls.SimInspiralChooseTDWaveform(
        sample["mass_1"] * lal.MSUN_SI, sample["mass_2"] * lal.MSUN_SI, *S1, *S2,
        sample["luminosity_distance"] * 1e6 * lal.PC_SI, sample["iota"], sample["phase"],
        0, 0, 0, 1 / ev.SAMPLING_RATE, ev.WAVEFORM_MINIMUM_FREQUENCY, 10.0, None, ls.SEOBNRv4PHM)
    ref = np.array(hp.data.data) - 1j * np.array(hc.data.data)
    print(f"[PE path] ChooseTDWaveform vs ours: relative difference "
          f"{np.linalg.norm(ref - model.hpc) / np.linalg.norm(ref):.1e}, "
          f"epochs {float(hp.epoch):.6f} vs {model.epoch:.6f} s")


def check_orbit_vs_waveform(sample, label):
    model = ev.run_model(sample)
    start = ev.inertial_dynamics(model, model.dynamics[:1, 0])
    print(f"[{label}] start: n = {np.round(start['n'][0], 6)}, L_N = {np.round(start['LN_hat'][0], 4)}, "
          f"f22 = {model.dynamics[0, 22] / (np.pi * model.M_sec):.3f} Hz")
    print(f"[{label}] t_peak = {model.t_peak:.2f} M -> waveform time "
          f"{model.t_seconds(model.t_peak) * 1e3:+.2f} ms")

    dt = 0.25
    t_geom = np.arange(model.dynamics[0, 0] + 200, model.t_attach - 300, dt)
    dyn = ev.inertial_dynamics(model, t_geom)
    h_model = CubicSpline(model.t_modes, model.hpc)(t_geom)
    hp_model, hc_model = h_model.real, -h_model.imag

    R_M = sample["luminosity_distance"] * 1e6 * lal.PC_SI / (model.M * lal.MRSUN_SI)
    N, P, Q = ed.observer_direction_I(sample["iota"], sample["phase"])
    candidates = {
        "P=e_phi, Q=-e_theta (adopted)": (P, Q, +1),
        "adopted basis, bodies swapped": (P, Q, -1),
        "P=-e_theta, Q=e_phi": (Q, P, +1),
        "P=-e_phi, Q=-e_theta (mirror)": (-P, Q, +1),
        "P=e_phi, Q=-e_theta, sign flipped": (P, Q, 0),
    }
    trim = slice(20, -20)
    results = {}
    for name, (Pc, Qc, sign) in candidates.items():
        h = multipole_strain((sign or 1) * dyn["x1"], (sign or 1) * dyn["x2"],
                             model.m_1, model.m_2, dt, N)
        hp, hc = project(h, Pc, Qc)
        hp, hc = hp / R_M, hc / R_M
        if sign == 0:
            hp, hc = -hp, -hc
        res = np.sqrt(np.sum((hp - hp_model)[trim] ** 2 + (hc - hc_model)[trim] ** 2)
                      / np.sum(hp_model[trim] ** 2 + hc_model[trim] ** 2))
        results[name] = (res, hp, hc)
        print(f"[{label}] {name:37s} relative residual {res:.3f}")

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    tt = model.t_seconds(t_geom)
    sel = tt > tt[-1] - 0.4
    for ax, hm, k in zip(axs, (hp_model, hc_model), (0, 1)):
        ax.plot(tt[sel], hm[sel], "k", lw=2, label="SEOBNRv4PHM")
        for name, (res, *hq) in results.items():
            ax.plot(tt[sel], hq[k][sel], lw=0.8, label=f"{name} ({res:.2f})")
    axs[0].legend(fontsize=7)
    axs[1].set_xlabel("time from peak [s]")
    fig.savefig(OUTDIR / f"orbit_vs_waveform_{label}.png", dpi=120)
    plt.close(fig)
    return model


def check_angular_momentum(model):
    d = ev.inertial_dynamics(model)
    J_hat = d["J"] / np.linalg.norm(d["J"], axis=1)[:, None]
    Jf = model.Jfhat_attach
    print(f"[J] angle J(t_attach) vs J_final: {np.degrees(np.arccos(np.clip(J_hat[-1] @ Jf, -1, 1))):.2f} deg; "
          f"|J(t_attach)| = {np.linalg.norm(d['J'][-1]):.4f}, |J_final| = {np.linalg.norm(model.J_final):.4f} M^2")
    J_inf, f22, J_pn = ev.J_hat_at_past_infinity(model)
    print(f"[J] PN J_hat(f_min) vs dynamics J_hat(start): "
          f"{np.degrees(np.arccos(np.clip(J_pn[-1] @ J_hat[0], -1, 1))):.2f} deg; "
          f"J(-inf) vs J(f_min): {np.degrees(np.arccos(np.clip(J_inf @ J_hat[0], -1, 1))):.2f} deg")


def check_ringdown(model, label):
    from pyseobnr.eob.utils.utils_precession_opt import SEOBBuildJframeVectors

    fit = ev.fit_ringdown(model)
    A220, A221 = fit["amplitudes"]
    print(f"[RD] M_f = {model.final_mass:.4f} M, chi_f = {model.final_spin:.4f}; fit with "
          f"{fit['n_points']} points: residual {fit['residual']:.3f}, |A221/A220| = {abs(A221 / A220):.2f}, "
          f"arg = {np.angle(A221 / A220):.2f}")
    t, h, ta = model.t_modes, model.hJ[(2, 2)], model.t_attach
    sel = (t >= ta) & (t <= ta + 50)
    b = np.exp(-1j * fit["omegas"][0] * (t[sel] - ta))
    a0 = (b.conj() @ h[sel]) / (b.conj() @ b)
    print(f"[RD] 220-only residual {np.linalg.norm(a0 * b - h[sel]) / np.linalg.norm(h[sel]):.3f}")

    e1, e2, e3 = SEOBBuildJframeVectors(model.Jfhat_attach)
    tt = np.linspace(ta - 300, ta, 600)
    d = ev.inertial_dynamics(model, tt)
    az = np.unwrap(np.arctan2(d["n"] @ e2, d["n"] @ e1))
    h_tt = CubicSpline(t, h)(tt)
    K = np.unwrap(2 * az + np.unwrap(np.angle(h_tt)))
    K -= 2 * np.pi * np.round(K[-1] / (2 * np.pi))
    print(f"[RD] 2 az(n) + arg h22^J at t_a - 300, -100, -20, 0 M: "
          + ", ".join(f"{K[np.searchsorted(tt, ta - x) if x else -1]:+.2f}" for x in (300, 100, 20, 0)))

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.5))
    w = (t > ta - 150) & (t < ta + 60)
    tr = np.linspace(ta, ta + 60, 300)
    hf = np.exp(-1j * np.outer(tr - ta, fit["omegas"])) @ fit["amplitudes"]
    axs[0].semilogy(t[w] - ta, np.abs(h[w]), ".-", label=r"$|h_{22}^J|$ SEOBNRv4PHM")
    axs[0].semilogy(tr - ta, np.abs(hf), "--", label="220 + 221 fit")
    axs[0].set_xlabel(r"$t - t_{\rm attach}$ [M]")
    axs[0].legend(fontsize=8)
    axs[1].plot(tt - ta, K)
    axs[1].set_xlabel(r"$t - t_{\rm attach}$ [M]")
    axs[1].set_ylabel(r"$2\varphi_n + \arg h^J_{22}$ [rad]")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"ringdown_{label}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    posterior = ev.load_posterior()
    i_maxl = int(np.argmax(posterior["log_likelihood"]))
    sample = ev.sample_as_dict(posterior, i_maxl)
    model = check_orbit_vs_waveform(sample, "GW190521_maxL")
    check_choose_td(sample, model)
    check_angular_momentum(model)
    check_ringdown(model, "GW190521_maxL")

    test = dict(sample)
    test.update(mass_1=150.0, mass_2=50.0, iota=1.2, phase=0.7,
                spin_1x=0.6, spin_1y=0.2, spin_1z=0.3, spin_2x=-0.3, spin_2y=0.3, spin_2z=-0.2)
    check_orbit_vs_waveform(test, "q3_precessing")
