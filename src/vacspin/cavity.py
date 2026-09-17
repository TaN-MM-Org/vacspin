"""The cavity-coupled interface: transition-resolved Purcell budget.

Everything between the spin and the detector, in the bad-cavity
(Purcell) regime, following the transition-resolved conventions of
Lee et al., arXiv:2511.05740 (Eqs. 1-10) as used by Mahim et al.
(Optics Express, 2026) for the SnV-on-TFLN interface:

    F_max = (3 / 4 pi^2) (lam/n)^3 Q_L / V         ideal Purcell factor
    F_C   = 1 + (F_max - 1) xi_pol xi_pos          the coupled transition
    F_C2  = same, suppressed by the cavity Lorentzian at the qubit
            detuning (the spin-flipping partner)
    zeta  = 1 + eta_q eta_DW eta_BR (F_C - 1)      lifetime reduction
    beta  = eta_q eta_DW eta_BR F_C / zeta         photons into the mode
    eta   = beta eta_wg T_link eta_chip eta_det    detected per decay
    Lambda_cav = Lambda_0 zeta / zeta_flip         cavity-boosted cyclicity
    g     = (1/2) sqrt(F_C kappa gamma_rad,C)      coupling rate, so that
            F_C = 4 g^2 / (kappa gamma_rad,C) holds as an identity

with V in units of (lam/n)^3 and kappa = f_cav / Q_L. `xi_pol_overlap`
gives the polarisation overlap sin^2(theta) (1 + sin 2 psi)/2 of a
<111> dipole with a linearly polarised cavity mode (2/3 at psi = 45 deg
for <100>-oriented diamond).

Validity is physics, not paperwork, so it is computed rather than
assumed: the `validity` dict carries the bad-cavity, weak-coupling and
spin-selectivity conditions, and `require_valid()` refuses -- with the
failing conditions named -- instead of returning numbers whose
derivation no longer applies.
"""
from __future__ import annotations

import numpy as np

from .params import EmissionBudget

__all__ = ["purcell_max", "lorentzian_suppression", "xi_pol_overlap",
           "CavityInterface"]


def purcell_max(q_loaded, v_rel):
    """Ideal Purcell factor F = (3/4 pi^2) Q/V, V in (lam/n)^3."""
    q, v = float(q_loaded), float(v_rel)
    if q <= 0 or v <= 0:
        raise ValueError("Q and V must be positive")
    return 3.0 / (4.0 * np.pi ** 2) * q / v


def lorentzian_suppression(delta_hz, f_cav_hz, q_loaded):
    """Purcell reduction of a transition detuned by delta from the
    cavity: L = 1 / (1 + (2 Q delta / f_cav)^2). Exactly 1 on
    resonance and exactly 1/2 at delta = f_cav / (2 Q) = kappa/2."""
    if f_cav_hz <= 0 or q_loaded <= 0:
        raise ValueError("f_cav_hz and Q must be positive")
    x = 2.0 * float(q_loaded) * float(delta_hz) / float(f_cav_hz)
    return 1.0 / (1.0 + x * x)


def xi_pol_overlap(theta_deg=54.7356, psi_deg=45.0):
    """Polarisation overlap |mu_hat . e_cav|^2 of a <111> dipole with a
    linearly polarised in-plane cavity mode:
    sin^2(theta) (1 + sin 2 psi) / 2, equal to 2/3 at psi = 45 deg for
    theta = arccos(1/sqrt 3) (a <100>-oriented sample)."""
    th = np.deg2rad(float(theta_deg))
    ps = np.deg2rad(float(psi_deg))
    return float(np.sin(th) ** 2 * (1.0 + np.sin(2.0 * ps)) / 2.0)


class CavityInterface:
    """All derived quantities of one cavity-coupled colour centre.

    Parameters
    ----------
    q_loaded : loaded quality factor of the cavity.
    v_rel : mode volume in units of (lam/n)^3.
    eta_wg : fraction of cavity decay leaving through the collection
        port (kappa_wg / kappa_total), in [0, 1].
    budget : `EmissionBudget` of the emitter (cited).
    lambda0 : bare (no-cavity) cyclicity of the operating point --
        compute it with `vacspin.cyclicity` from your spin parameters
        and field, or measure it.
    omega_q_hz : qubit frequency (Hz), the detuning of the
        spin-flipping partner transition.
    xi_pol, xi_pos : polarisation and position overlap of the emitter
        with the cavity field, each in [0, 1].
    t_link, eta_chip, eta_det : downstream transfer, routing and
        detector efficiencies, each in [0, 1] (defaults 1: the
        cavity-only budget).
    delta_partner_hz : optional detuning of a competing SAME-spin
        transition (e.g. the D line); when given, its Purcell factor
        F_partner and the selectivity condition kappa < delta are
        reported.

    Everything is computed in __init__; `validity` holds the regime
    checks and `require_valid()` refuses when any fails.
    """

    def __init__(self, q_loaded, v_rel, eta_wg, budget: EmissionBudget,
                 lambda0, omega_q_hz, xi_pol, xi_pos=1.0,
                 t_link=1.0, eta_chip=1.0, eta_det=1.0,
                 delta_partner_hz=None):
        for name, v in (("eta_wg", eta_wg), ("xi_pol", xi_pol),
                        ("xi_pos", xi_pos), ("t_link", t_link),
                        ("eta_chip", eta_chip), ("eta_det", eta_det)):
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        if lambda0 <= 0 or not np.isfinite(lambda0):
            raise ValueError("lambda0 must be a finite positive "
                             "cyclicity (compute or measure it)")
        if omega_q_hz <= 0:
            raise ValueError("omega_q_hz must be positive")
        self.q = float(q_loaded)
        self.v = float(v_rel)
        self.budget = budget
        self.f_cav_hz = 299792458.0 / (budget.zpl_nm * 1e-9)
        self.kappa_hz = self.f_cav_hz / self.q
        self.omega_q_hz = float(omega_q_hz)

        a = budget.radiative_fraction
        self.f_max = purcell_max(q_loaded, v_rel)
        self.f_c = 1.0 + (self.f_max - 1.0) * xi_pol * xi_pos
        lc2 = lorentzian_suppression(omega_q_hz, self.f_cav_hz, self.q)
        self.f_c2 = 1.0 + (self.f_max - 1.0) * xi_pol * xi_pos * lc2
        self.zeta = 1.0 + a * (self.f_c - 1.0)
        zeta_flip = 1.0 + a * (self.f_c2 - 1.0)
        self.gamma_cav = budget.gamma0 * self.zeta        # 1/s
        self.tau_cav_s = 1.0 / self.gamma_cav
        self.lambda_cav = float(lambda0) * self.zeta / zeta_flip
        self.beta = a * self.f_c / self.zeta
        self.eta_wg = float(eta_wg)
        self.eta = (self.beta * self.eta_wg * float(t_link)
                    * float(eta_chip) * float(eta_det))
        # coupling rate defined so F_C = 4 g^2 / (kappa gamma_rad,C)
        gamma_rad_c = budget.gamma0 * a                   # 1/s
        self.g_hz = 0.5 * np.sqrt(self.f_c * self.kappa_hz * gamma_rad_c)
        self.cooperativity = (4.0 * self.g_hz ** 2
                              / (self.kappa_hz * budget.gamma0))
        gamma_cav_hz = self.gamma_cav / (2.0 * np.pi)
        self.validity = dict(
            spin_selectivity=bool(gamma_cav_hz < omega_q_hz / 5.0),
            bad_cavity=bool(gamma_cav_hz < self.kappa_hz / 10.0),
            weak_coupling=bool(self.g_hz < self.kappa_hz / 10.0),
        )
        self.f_partner = None
        if delta_partner_hz is not None:
            self.f_partner = 1.0 + (self.f_max - 1.0) * xi_pol * xi_pos \
                * lorentzian_suppression(delta_partner_hz,
                                         self.f_cav_hz, self.q)
            self.validity["partner_selectivity"] = \
                bool(self.kappa_hz < float(delta_partner_hz))

    def require_valid(self):
        """Refuse -- naming the failed conditions -- unless every
        regime assumption behind the formulas above holds."""
        bad = [k for k, ok in self.validity.items() if not ok]
        if bad:
            raise ValueError(
                "interface outside its regime of validity: "
                f"{', '.join(bad)} failed. The Purcell/rate formulas "
                "assume the bad-cavity, weak-coupling, spin-selective "
                "regime; change Q_L, V or the operating point.")
        return self

    def summary(self):
        out = dict(q_loaded=self.q, v_rel=self.v, f_max=self.f_max,
                   f_c=self.f_c, f_c2=self.f_c2, zeta=self.zeta,
                   tau_cav_ns=self.tau_cav_s * 1e9,
                   lambda_cav=self.lambda_cav, beta=self.beta,
                   eta=self.eta, g_mhz=self.g_hz / 1e6,
                   kappa_ghz=self.kappa_hz / 1e9,
                   cooperativity=self.cooperativity,
                   validity=dict(self.validity))
        if self.f_partner is not None:
            out["f_partner"] = self.f_partner
        return out
