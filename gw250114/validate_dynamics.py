"""Consistency checks of the SEOBNRv5PHM dynamics extracted in `eobviz/models/seobnrv5.py`.

1. The co-precessing z axis R(t) z coincides with L_N(t) from the PN splines.
2. The point-particle orbit, fed to the Cartesian multipole formula
   (mass quadrupole + mass octupole + current quadrupole, Thorne 1980 eq. 4.12'),
   reproduces the model's h_+ and h_x for the observer direction and polarization
   basis given by `observer_direction_I`. Alternative bases and the swapped
   labelling of the two bodies are shown for comparison: they must do worse.
   Run on the max-likelihood GW250114 sample and on an unequal-mass,
   strongly precessing system, for which the odd-parity multipoles
   (sensitive to which body is which) are large.
3. The time axis: the H1 strain built from the time-domain polarizations
   placed at geocent_time + (H1 delay) matches bilby's frequency-domain
   detector response, which is what enters the likelihood.
4. The ringdown: the (2,2) mode in the final-J frame after attachment is fitted
   with the 220 + 221 Kerr QNMs; in the late inspiral its phase follows twice the
   azimuth of the separation vector about J_f, so that the l = m = 2 shape of the
   remnant can be continued from the orientation of the binary at attachment.
5. Running pyseobnr with `with_modes=True` leaves the dynamics untouched.
"""

import sys
from pathlib import Path

import bilby
import lal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eobviz import physics  # noqa: E402
from eobviz.events import get_event  # noqa: E402
from eobviz.models import seobnrv5 as ed  # noqa: E402
from eobviz.multipole import multipole_strain, project  # noqa: E402

OUTDIR = Path(__file__).parent / "validation"
OUTDIR.mkdir(exist_ok=True)


def check_orbit_vs_waveform(sample, label):
    run = ed.SEOBNRv5PHM(sample, with_modes=False)
    model, gen = run.model, run.generator
    params = gen.parameters
    M = model.M
    to_seconds = ed.geometric_to_seconds(model)

    # 1. co-precessing z axis vs L_N
    dyn = ed.inertial_dynamics(model)
    cosang = np.einsum("ti,ti->t", dyn["ez_coprecessing"], dyn["LN_hat"])
    insp = dyn["t"] < model.t_attach
    print(
        f"[{label}] max angle between R(t)z and L_N(t) in the inspiral: "
        f"{np.degrees(np.arccos(np.clip(cosang[insp].min(), -1, 1))):.2e} deg"
    )
    at_ref = ed.inertial_dynamics(model, np.atleast_1d(model.t_ref))
    print(f"[{label}] n(t_ref) = {at_ref['n'][0]}, L_N(t_ref) = {at_ref['LN_hat'][0]}")
    print(f"[{label}] dynamics grid spacing: median {np.median(np.diff(dyn['t'])):.1f} M, max {np.diff(dyn['t']).max():.1f} M")

    # 2. multipole strain vs model on a uniform grid in the early inspiral
    dt = 0.25
    t_lo = model.dynamics[0, 0] + 1000
    t_hi = model.t_ref
    t_geom = np.arange(t_lo, t_hi, dt)
    dyn = ed.inertial_dynamics(model, t_geom)

    hpc = model.hpc  # physical h_+ - i h_x, on model.t in seconds
    h_model = CubicSpline(model.t, hpc)(to_seconds(t_geom))
    hp_model, hc_model = h_model.real, -h_model.imag

    R_M = params["distance"] * 1e6 * lal.PC_SI / (M * lal.MRSUN_SI)
    N, P, Q = physics.observer_direction_I(params["inclination"], params["phi_ref"])
    e_theta = -Q
    candidates = {
        "P=e_phi, Q=-e_theta (adopted)": (P, Q, +1),
        "adopted basis, bodies swapped": (P, Q, -1),
        "P=e_theta, Q=e_phi": (e_theta, P, +1),
        "P=-e_phi, Q=-e_theta (mirror)": (-P, Q, +1),
    }
    trim = slice(20, -20)  # drop finite-difference edge effects
    results = {}
    for name, (Pc, Qc, sign) in candidates.items():
        h = multipole_strain(
            sign * dyn["x1"], sign * dyn["x2"], model.m_1, model.m_2, dt, N
        )
        hp, hc = project(h, Pc, Qc)
        hp, hc = hp / R_M, hc / R_M
        res = np.sqrt(
            np.sum((hp - hp_model)[trim] ** 2 + (hc - hc_model)[trim] ** 2)
            / np.sum(hp_model[trim] ** 2 + hc_model[trim] ** 2)
        )
        results[name] = (res, hp, hc)
        print(f"[{label}] {name:35s} relative residual {res:.3f}")

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    tt = to_seconds(t_geom)
    sel = tt > tt[-1] - 0.15
    for ax, hm, k in zip(axs, (hp_model, hc_model), (1, 2)):
        ax.plot(tt[sel], hm[sel], "k", lw=2, label="SEOBNRv5PHM")
        for name, (res, *hq) in results.items():
            ax.plot(tt[sel], hq[k - 1][sel], lw=0.8, label=f"{name} ({res:.2f})")
        ax.set_ylabel(["$h_+$", r"$h_\times$"][k - 1])
    axs[0].legend(fontsize=7)
    axs[1].set_xlabel("time from peak [s]")
    fig.suptitle(f"{label}: multipole formula on EOB orbit vs model")
    fig.savefig(OUTDIR / f"orbit_vs_waveform_{label}.png", dpi=120)
    plt.close(fig)
    return run


def check_time_alignment(sample, model, gen, ifo_name="H1"):
    """bilby FD detector response vs TD polarizations shifted by hand."""
    trigger_time = 1420878141.222656
    start_time = trigger_time + 2 - ed.DURATION
    ifo = bilby.gw.detector.get_empty_interferometer(ifo_name)
    ifo.set_strain_data_from_zero_noise(
        sampling_frequency=2048, duration=ed.DURATION, start_time=start_time
    )
    wfg = bilby.gw.waveform_generator.WaveformGenerator(
        duration=ed.DURATION,
        sampling_frequency=2048,
        start_time=start_time,
        frequency_domain_source_model=bilby.gw.source.gwsignal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant="SEOBNRv5PHM",
            reference_frequency=ed.REFERENCE_FREQUENCY,
            minimum_frequency=ed.WAVEFORM_MINIMUM_FREQUENCY,
            maximum_frequency=ed.MAXIMUM_FREQUENCY,
            lmax_nyquist=ed.LMAX_NYQUIST,
        ),
    )
    pols = wfg.frequency_domain_strain(sample)
    fd = ifo.get_detector_response(pols, sample)
    td_bilby = np.fft.irfft(fd) * ifo.sampling_frequency
    t_bilby = ifo.time_array

    # by hand: F+ h+ + Fx hx at t - geocent_time - delay
    delay = ifo.time_delay_from_geocenter(sample["ra"], sample["dec"], sample["geocent_time"])
    Fp = ifo.antenna_response(sample["ra"], sample["dec"], sample["geocent_time"], sample["psi"], "plus")
    Fc = ifo.antenna_response(sample["ra"], sample["dec"], sample["geocent_time"], sample["psi"], "cross")
    hpc = model.hpc
    t_wave = model.t + sample["geocent_time"] + delay
    td_hand = Fp * hpc.real + Fc * (-hpc.imag)

    # band-limit the hand-made one like the FD one (20 Hz - 896 Hz) for a fair comparison
    td_hand_on_grid = np.interp(t_bilby, t_wave, td_hand, left=0, right=0)
    f = np.fft.rfftfreq(len(t_bilby), 1 / 2048)
    band = (f >= ed.WAVEFORM_MINIMUM_FREQUENCY) & (f <= ed.MAXIMUM_FREQUENCY)
    td_hand_bl = np.fft.irfft(np.fft.rfft(td_hand_on_grid) * band)
    td_bilby_bl = np.fft.irfft(np.fft.rfft(td_bilby) * band)

    win = (t_bilby > sample["geocent_time"] - 0.3) & (t_bilby < sample["geocent_time"] + 0.05)
    # residual time shift by cross-correlation (sub-sample resolution via upsampling)
    a, b = td_bilby_bl[win], td_hand_bl[win]
    up = 16
    from scipy.signal import resample

    au, bu = resample(a, len(a) * up), resample(b, len(b) * up)
    cc = np.correlate(au, bu, "full")
    lag = (np.argmax(cc) - (len(bu) - 1)) / (2048 * up)
    res = np.sqrt(np.sum((a - b) ** 2) / np.sum(a**2))
    print(
        f"[time] {ifo_name}: relative residual bilby-FD vs hand-made TD = {res:.3f}, "
        f"best lag = {lag * 1e3:.3f} ms"
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_bilby[win] - sample["geocent_time"], a, "k", lw=2, label="bilby FD response")
    ax.plot(t_bilby[win] - sample["geocent_time"], b, "C1", lw=1, label="TD polarizations by hand")
    ax.set_xlabel("t - geocent_time [s]")
    ax.legend()
    fig.savefig(OUTDIR / f"time_alignment_{ifo_name}.png", dpi=120)
    plt.close(fig)


def check_J_infinity(run):
    J_inf, f22, J_hat = run.J_hat_at_past_infinity()
    iref = np.argmin(np.abs(f22 - ed.REFERENCE_FREQUENCY))
    J_ref = J_hat[iref]
    print(f"[J] J_hat(f_ref={f22[iref]:.2f} Hz) = {J_ref}")
    for f in [10, 5, 1, 0.1, 0.01, f22[0]]:
        i = np.argmin(np.abs(f22 - f))
        ang = np.degrees(np.arccos(np.clip(J_hat[i] @ J_ref, -1, 1)))
        print(f"[J]   f22 = {f22[i]:9.4f} Hz: angle(J_hat, J_hat(f_ref)) = {ang:.3f} deg")
    return J_inf


def check_ringdown(sample, label):
    model_pe, _ = ed.run_model(sample)
    run = ed.SEOBNRv5PHM(sample, with_modes=True)
    model = run.model
    print(f"[RD] max |dynamics difference| PE run vs with_modes: "
          f"{np.abs(model.dynamics - model_pe.dynamics).max():.1e}, "
          f"t_attach difference {model.t_attach - model_pe.t_attach:.1e}")

    to_geom = ed.seconds_to_geometric(model)
    t = to_geom(model.t)
    h = model.wavefom_modesJ["2,2"]
    ta = model.t_attach
    fit = run.fit_ringdown()
    A220, A221 = fit["amplitudes"]
    print(f"[RD] M_f = {model.final_mass:.4f} M, chi_f = {model.final_spin:.4f}; "
          f"M omega_220 = {fit['omegas'][0]:.4f}, M omega_221 = {fit['omegas'][1]:.4f}")
    print(f"[RD] 220+221 fit from t_attach: residual {fit['residual']:.3f}, "
          f"|A221/A220| = {abs(A221 / A220):.2f}, arg(A221/A220) = {np.angle(A221 / A220):.2f}")
    for length in (30, 80):
        f2 = run.fit_ringdown(fit_length=length)
        print(f"[RD]   fit length {length} M: residual {f2['residual']:.3f}, "
              f"|A221/A220| = {abs(f2['amplitudes'][1] / f2['amplitudes'][0]):.2f}")
    # 220 only, for comparison
    sel = (t >= ta) & (t <= ta + 50)
    b = np.exp(-1j * fit["omegas"][0] * (t[sel] - ta))
    a0 = (b.conj() @ h[sel]) / (b.conj() @ b)
    print(f"[RD] 220-only fit residual {np.linalg.norm(a0 * b - h[sel]) / np.linalg.norm(h[sel]):.3f}")

    # phase of h22^J against twice the azimuth of n about J_f, before attachment
    e1, e2, e3 = ed.J_frame_basis(model)
    tt = np.linspace(ta - 300, ta, 600)
    d = ed.inertial_dynamics(model, tt)
    az = np.unwrap(np.arctan2(d["n"] @ e2, d["n"] @ e1))
    h_tt = CubicSpline(t, h.real)(tt) + 1j * CubicSpline(t, h.imag)(tt)
    K = np.unwrap(2 * az + np.unwrap(np.angle(h_tt)))
    K -= 2 * np.pi * np.round(K[-1] / (2 * np.pi))
    print(f"[RD] 2 az(n) + arg h22^J: at t_a - 300 M {K[0]:+.2f}, t_a - 100 M "
          f"{K[np.searchsorted(tt, ta - 100)]:+.2f}, t_a - 20 M {K[np.searchsorted(tt, ta - 20)]:+.2f}, "
          f"t_a {K[-1]:+.2f} rad (phase of h22 decreasing: {np.angle(h_tt[-1] / h_tt[-2]) < 0})")

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.5))
    w = (t > ta - 150) & (t < ta + 60)
    tr = np.linspace(ta, ta + 60, 300)
    hf = physics.ringdown_h22(fit, tr, ta)
    axs[0].semilogy(t[w] - ta, np.abs(h[w]), label=r"$|h_{22}^J|$ SEOBNRv5PHM")
    axs[0].semilogy(tr - ta, np.abs(hf), "--", label="220 + 221 fit")
    axs[0].semilogy(tr - ta, np.abs(fit["amplitudes"][0] * np.exp(-1j * fit["omegas"][0] * (tr - ta))),
                    ":", label="220 part")
    axs[0].set_xlabel(r"$t - t_{\rm attach}$ [M]")
    axs[0].legend(fontsize=8)
    axs[1].plot(tt - ta, K)
    axs[1].set_xlabel(r"$t - t_{\rm attach}$ [M]")
    axs[1].set_ylabel(r"$2\varphi_n + \arg h^J_{22}$ [rad]")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"ringdown_{label}.png", dpi=120)
    plt.close(fig)
    return fit


def check_kick(sample, label):
    """Recoil from NRSur7dq4Remnant two ways: (a) as in eobviz/export.py, from the EOB spins and
    orbit at -100 M from the peak; (b) surfinBH alone, from the PE spins at f_ref in the
    LAL frame of f_ref (= our I-frame), evolved with its own PN + NRSur7dq4 spin dynamics.
    Both are in the I-frame; they differ by the spin evolution (EOB vs PN/NRSur)."""
    import surfinBH

    model, gen = ed.run_model(sample)
    p = gen.parameters
    d = ed.inertial_dynamics(model, np.atleast_1d(model.t_attach + physics.KICK_EPOCH))
    v_a, err = physics.nrsur_kick(model.m_1, model.m_2, d["chi1"][0], d["chi2"][0], d["n"][0], d["LN_hat"][0])
    fit = surfinBH.LoadFits("NRSur7dq4Remnant")
    omega0 = np.pi * p["f_ref"] * model.M * lal.MTSUN_SI
    chi1 = np.array([p["spin1x"], p["spin1y"], p["spin1z"]])
    chi2 = np.array([p["spin2x"], p["spin2y"], p["spin2z"]])
    _, _, v_b, *_ = fit.all(p["mass1"] / p["mass2"], chi1, chi2, omega0=omega0, allow_extrap=True)
    c = lal.C_SI / 1e3
    angle = np.degrees(np.arccos(np.clip(v_a @ v_b / np.linalg.norm(v_a) / np.linalg.norm(v_b), -1, 1)))
    print(f"[kick {label}] EOB spins at -100 M: {np.linalg.norm(v_a) * c:.0f} +- {err * c:.0f} km/s; "
          f"surfinBH from f_ref: {np.linalg.norm(v_b) * c:.0f} km/s; angle {angle:.0f} deg")
    return v_a, v_b


if __name__ == "__main__":
    event = get_event("GW250114")
    posterior = event.load_posterior()
    sample = event.sample(posterior, event.max_likelihood_index(posterior))

    run = check_orbit_vs_waveform(sample, "GW250114_maxL")
    check_time_alignment(sample, run.model, run.generator, "H1")
    check_time_alignment(sample, run.model, run.generator, "L1")
    check_J_infinity(run)
    check_ringdown(sample, "GW250114_maxL")
    check_kick(sample, "GW250114_maxL")

    # unequal masses + large in-plane spins + edge-on-ish: odd multipoles matter
    test = dict(sample)
    test.update(
        mass_1=60.0, mass_2=20.0, a_1=0.8, a_2=0.5, tilt_1=1.2, tilt_2=2.0,
        theta_jn=1.2, phase=0.7,
    )
    check_orbit_vs_waveform(test, "q3_precessing")
    check_kick(test, "q3_precessing")
    for i in np.random.default_rng(2).choice(len(posterior), 5, replace=False):
        check_kick(event.sample(posterior, i), f"sample{i}")
