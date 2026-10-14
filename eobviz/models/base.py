"""Common interface of the EOB models: what the scene export needs from a run."""

from abc import ABC, abstractmethod

import numpy as np

from .. import physics


class EOBRun(ABC):
    """One run of an EOB model for one posterior sample, called as in PE.

    Times are in units of the total detector-frame mass M, on the model's own time axis;
    "waveform time" [s] is relative to the waveform's t = 0, which bilby places at
    geocent_time. Vectors are in the model's I-frame (`physics`).

    Subclasses run the model in `__init__`, keep the model's own output in `self.model`,
    and set:
        M: total detector-frame mass [M_sun]; m_1, m_2: mass fractions; M_sec: M in s
        final_mass [M], final_spin: remnant
        t_attach: merger; the remnant is shown from here
        t_peak: the peak, origin of the kick epoch (`physics.KICK_EPOCH`)
        t_start: first time of the dynamics; t_eval_end: last time at which the
            dynamics are evaluated (later times are clipped to it)
        omega_ref, chi1_ref, chi2_ref: orbital frequency [1/M] and dimensionless spins
            where the sample's spins are defined (where J(t -> -inf) is evolved from)
        inclination, phi_ref: the observer's direction in the I-frame, LAL convention
    """

    approximant = None

    @abstractmethod
    def inertial_dynamics(self, t_geom):
        """Dict of arrays at times t_geom: r, phi (orbital phase [rad]), omega [1/M],
        n (unit separation, body 2 -> body 1), x1, x2 [M] (centre-of-mass positions),
        chi1, chi2, LN_hat."""

    @abstractmethod
    def t_geometric(self, t_sec):
        """Waveform time [s] -> model time [M]."""

    @abstractmethod
    def t_seconds(self, t_geom):
        """Model time [M] -> waveform time [s]."""

    @property
    @abstractmethod
    def Jfhat_attach(self):
        """Direction of the remnant spin (I-frame)."""

    @abstractmethod
    def J_reference(self):
        """J = L + S1 + S2 [M^2] where the sample's spins are defined (I-frame)."""

    @abstractmethod
    def ringdown_mode(self):
        """(t [M], h22): the (2,2) mode in the remnant's frame, to fit the ringdown on."""

    @property
    def omega_attach(self):
        """Orbital frequency [1/M] at which the orbital phase keeps running after t_attach
        (it sets how the trails keep fading)."""
        return float(self.inertial_dynamics(np.atleast_1d(self.t_attach))["omega"][0])

    def rotation_I_to_sky(self, psi):
        return physics.rotation_I_to_sky(self.inclination, self.phi_ref, psi)

    def J_hat_at_past_infinity(self, f_low=1e-3):
        """J direction at GW frequency f_low [Hz] (by default, t -> -inf), from the spins at
        the reference evolved backwards with pyseobnr's PN precession equations.
        Returns J_hat there, and the GW frequencies [Hz] and J_hat along the way."""
        J_hat, omega, J_path = physics.J_hat_at_past_infinity(
            self.omega_ref, np.pi * f_low * self.M_sec, self.m_1, self.m_2,
            self.chi1_ref, self.chi2_ref)
        return J_hat, omega / (np.pi * self.M_sec), J_path

    def fit_ringdown(self, fit_length=50.0):
        """220 + 221 fit of `ringdown_mode` from t_attach (`physics.fit_ringdown`)."""
        t, h22 = self.ringdown_mode()
        return physics.fit_ringdown(t, h22, self.t_attach, self.final_mass, self.final_spin,
                                    fit_length)

    def info(self):
        """Model-specific entries for the scene's info file."""
        return {}
