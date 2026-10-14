"""Model-independent physics: observer frame, horizon sizes, ringdown, recoil, J(t -> -inf).

Frames
------
I-frame (LAL "L0" frame): z along the Newtonian orbital angular momentum at the
reference point of the spins, x along the separation vector n = (x_1 - x_2)/r there.

Sky frame: X = west, Y = north, Z = towards the observer (right-handed).
A camera at +Z looking towards -Z with Y up sees the sky as seen from Earth.
bilby's polarization basis is m = (cos psi, sin psi, 0), n = (-sin psi, cos psi, 0)
in this frame, and the propagation direction m x n = +Z.
"""

import lal
import numpy as np

# km per solar mass, G M_sun / c^2
KM_PER_MSUN = lal.MRSUN_SI / 1e3


def observer_direction_I(inclination, phi_ref):
    """Unit vector towards the observer in the I-frame (LAL convention),
    together with the polarization basis vectors P, Q such that
    h_+ = (P P - Q Q) : h / 2 and h_x = (P Q + Q P) : h / 2.
    The basis is verified numerically against each model in the event validation scripts.
    """
    theta, varphi = inclination, np.pi / 2 - phi_ref
    N = np.array(
        [np.sin(theta) * np.cos(varphi), np.sin(theta) * np.sin(varphi), np.cos(theta)]
    )
    e_theta = np.array(
        [np.cos(theta) * np.cos(varphi), np.cos(theta) * np.sin(varphi), -np.sin(theta)]
    )
    e_varphi = np.array([-np.sin(varphi), np.cos(varphi), 0.0])
    return N, e_varphi, -e_theta


def rotation_I_to_sky(inclination, phi_ref, psi):
    """Rotation matrix taking I-frame vectors to the sky frame (west, north, observer).

    It maps the polarization basis (P, Q) onto bilby's (m, n), and the
    direction of the observer onto +Z.
    """
    N, P, Q = observer_direction_I(inclination, phi_ref)
    m = np.array([np.cos(psi), np.sin(psi), 0.0])
    n = np.array([-np.sin(psi), np.cos(psi), 0.0])
    z = np.array([0.0, 0.0, 1.0])
    return np.column_stack([m, n, z]) @ np.vstack([P, Q, N])


def areal_radius(m, chi):
    """sqrt(A / 4 pi) = sqrt(2 m r_+) for a Kerr BH, in units of m."""
    r_plus = m * (1 + np.sqrt(1 - chi**2))
    return np.sqrt(2 * m * r_plus)


def kerr_qnm_frequency(ell, m, n, final_spin, final_mass):
    """Complex Kerr QNM frequency in units of 1/M (Re > 0, Im < 0, h ~ exp(-i omega t))."""
    import qnm

    if final_spin <= 0:
        raise NotImplementedError("remnant spin anti-aligned with J")
    omega, _, _ = qnm.modes_cache(s=-2, l=ell, m=m, n=n)(a=final_spin)
    return omega / final_mass


RINGDOWN_MODES = ((2, 2, 0), (2, 2, 1))


def fit_ringdown(t, h22, t_attach, final_mass, final_spin, fit_length=50.0):
    """Least-squares fit of a (2,2) mode h22(t) with the 220 and 221 Kerr QNMs of the
    remnant, from t_attach to `fit_length` M after it (times in M).

    Returns the complex QNM frequencies [1/M], the complex amplitudes at t_attach,
    the relative residual and the number of points fitted.
    """
    sel = (t >= t_attach) & (t <= t_attach + fit_length)
    omegas = np.array([
        kerr_qnm_frequency(*lmn, final_spin, final_mass) for lmn in RINGDOWN_MODES
    ])
    basis = np.exp(-1j * np.outer(t[sel] - t_attach, omegas))
    amplitudes, *_ = np.linalg.lstsq(basis, h22[sel], rcond=None)
    residual = np.linalg.norm(basis @ amplitudes - h22[sel]) / np.linalg.norm(h22[sel])
    return dict(omegas=omegas, amplitudes=amplitudes, residual=residual, n_points=int(sel.sum()))


def ringdown_h22(fit, t_geom, t_attach):
    """The fitted 220 + 221 (2,2) mode at times t_geom >= t_attach."""
    return np.exp(-1j * np.outer(t_geom - t_attach, fit["omegas"])) @ fit["amplitudes"]


KICK_EPOCH = -100.0  # [M] from the peak: NRSur7dq4Remnant's reference epoch
_nrsur_remnant = {}


def nrsur_kick(m_1, m_2, chi1, chi2, n, LN_hat):
    """Recoil velocity [c] from NRSur7dq4Remnant (surfinBH), given the spins and the orbit
    at KICK_EPOCH from the peak, in the frame of the input vectors. The fit takes the spins
    in the coorbital frame of that epoch (z along L, x along the separation from the lighter
    to the heavier body) and returns the kick in the same frame. Also returns the fit's
    1-sigma error estimate on the speed."""
    if "fit" not in _nrsur_remnant:
        import surfinBH

        _nrsur_remnant["fit"] = surfinBH.LoadFits("NRSur7dq4Remnant")
    if m_1 >= m_2:
        q, chiA, chiB, x = m_1 / m_2, chi1, chi2, n
    else:  # n points from body 2 to body 1
        q, chiA, chiB, x = m_2 / m_1, chi2, chi1, -n
    e3 = LN_hat / np.linalg.norm(LN_hat)
    e1 = x - (x @ e3) * e3
    e1 /= np.linalg.norm(e1)
    basis = np.column_stack([e1, np.cross(e3, e1), e3])
    _, _, vf, _, _, vf_err = _nrsur_remnant["fit"].all(
        q, basis.T @ chiA, basis.T @ chiB, allow_extrap=True)
    return basis @ vf, float(np.linalg.norm(vf_err))


def J_hat_at_past_infinity(omega_ref, omega_low, m_1, m_2, chi1, chi2):
    """Direction of J = L + S1 + S2 at orbital frequency omega_low [1/M], integrating
    pyseobnr's (quasi-circular) PN spin-precession equations backwards from the spins
    chi1, chi2 at omega_ref, in the frame of the input spins.

    Returns J_hat at omega_low, and the orbital frequencies and J_hat along the way,
    so that convergence can be checked.
    """
    from pyseobnr.eob.dynamics.pn_evolution_opt import (
        build_splines_PN,
        compute_quasiprecessing_PNdynamics_opt,
    )

    t, y = compute_quasiprecessing_PNdynamics_opt(omega_ref, omega_low, m_1, m_2, chi1, chi2)
    # keep only the backward part (orbital frequency <= omega_ref)
    keep = y[:, -1] <= omega_ref * (1 + 1e-12)
    y = y[keep]
    splines = build_splines_PN(t[keep], y, m_1, m_2, omega_low)
    pn = splines["everything"](y[:, -1])
    J = pn[:, 13:16] + m_1**2 * pn[:, 4:7] + m_2**2 * pn[:, 7:10]
    J_hat = J / np.linalg.norm(J, axis=1)[:, None]
    return J_hat[0], y[:, -1], J_hat
