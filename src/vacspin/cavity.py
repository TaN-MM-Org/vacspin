"""The cavity-coupled interface: transition-resolved Purcell budget.

Everything between the spin and the detector, in the bad-cavity
(Purcell) regime. The decay-rate budget follows Lee et al.,
arXiv:2511.05740: the coupled transition C decays F_C times faster
than in bulk (their Eq. 2), with the Debye-Waller factor eta_DW and the
branching ratio eta_BR (their Eq. 3), and the polarization overlap of
the dipole with the cavity field (their Eqs. 6-10). The quantum
efficiency eta_q, the ideal Purcell factor F_max and the position
overlap are added as in the SnV-on-TFLN study of Mahim et al.
(submitted to Optics Express, 2026):

    F_max = (3 / 4 pi^2) (lam/n)^3 Q_L / V         ideal Purcell factor
    F_C   = 1 + (F_max - 1) xi_pol xi_pos          the coupled transition
    F_C2  = same, suppressed by the cavity Lorentzian at the qubit
            detuning (the spin-flipping partner)
    a     = eta_q eta_DW eta_BR                    radiative fraction of C
    C     = a (F_C - 1)                            cooperativity
    zeta  = 1 + C                                  lifetime reduction
    beta  = C / (1 + C) = a (F_C - 1) / zeta       photons into the mode
    eta   = beta eta_wg T_link eta_chip eta_det    detected per decay
    Lambda_cav = Lambda_0 zeta / zeta_flip         cavity-boosted cyclicity
    g     = (1/2) sqrt((F_C - 1) kappa gamma_rad,C)  coupling rate, so that
            F_C - 1 = 4 g^2 / (kappa gamma_rad,C) holds as an identity

F_C counts the whole emission of transition C with the cavity present.
Only the added part, (F_C - 1) times the bulk rate, goes into the
cavity mode; the bulk part still goes where it went without the
cavity. So with no overlap (xi = 0) beta, g and C are exactly 0.
Before 0.6.0 beta used F_C instead of F_C - 1 (and g likewise), which
counted the bulk emission of C as cavity photons: beta was 0.342, not
0, with no overlap, and the design point of the study above gave
eta = 0.649 instead of 0.615. That form is still available as
``beta_form="transition"``.

with V in units of (lam/n)^3 and kappa = f_cav / Q_L. In the g formula
all three rates are in the same units: g_hz and kappa_hz are in Hz, and
the radiative rate gamma_rad,C (1/s) enters as the linewidth
gamma_rad,C / (2 pi) in Hz. `xi_pol_overlap`
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
    if not (q > 0 and v > 0 and np.isfinite(q) and np.isfinite(v)):
        raise ValueError("Q and V must be positive and finite")
    return 3.0 / (4.0 * np.pi ** 2) * q / v


def lorentzian_suppression(delta_hz, f_cav_hz, q_loaded):
    """Purcell reduction of a transition detuned by delta from the
    cavity: L = 1 / (1 + (2 Q delta / f_cav)^2). Exactly 1 on
    resonance and exactly 1/2 at delta = f_cav / (2 Q) = kappa/2."""
    if not (f_cav_hz > 0 and q_loaded > 0 and np.isfinite(f_cav_hz)
            and np.isfinite(q_loaded)):
        raise ValueError("f_cav_hz and Q must be positive and finite")
    if not np.isfinite(delta_hz):
        raise ValueError("delta_hz must be finite")
    x = 2.0 * float(q_loaded) * float(delta_hz) / float(f_cav_hz)
    return 1.0 / (1.0 + x * x)


def xi_pol_overlap(theta_deg: float = 54.7356,
                   psi_deg: float = 45.0) -> float:
    """Polarization overlap |mu_hat . e_cav|^2 of a <111> dipole with a
    linearly polarized in-plane cavity mode:
    sin^2(theta) (1 + sin 2 psi) / 2, equal to 2/3 at psi = 45 deg for
    theta = arccos(1/sqrt 3) (a <100>-oriented sample). Both angles
    are in degrees and must be finite (since 0.5.0 a NaN or infinite
    angle is refused instead of returning NaN)."""
    th, ps = float(theta_deg), float(psi_deg)
    if not (np.isfinite(th) and np.isfinite(ps)):
        raise ValueError("theta_deg and psi_deg must be finite angles in "
                         "degrees")
    th = np.deg2rad(th)
    ps = np.deg2rad(ps)
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
    beta_form : "purcell" (default since 0.6.0) counts only the
        cavity-added emission, (F_C - 1), as photons into the mode.
        "transition" reproduces vacspin 0.5.0 and earlier, which used
        F_C for beta, g and the cooperativity.

    Everything is computed in __init__; `validity` holds the regime
    checks and `require_valid()` refuses when any fails.

    Since 0.7.0 the default form refuses a cavity whose ideal Purcell
    factor F_max is below 1 when the emitter overlaps the mode at all
    (Q_L / V below 4 pi^2 / 3, about 13.2, for example a large-volume
    resonator with Q_L = 1000 and V = 100): then F_C < 1, and the
    cavity-mode share a (F_C - 1) / zeta would be negative. 0.6.0
    returned a negative `beta` and `eta` and a NaN `g_hz` there.
    """

    def __init__(self, q_loaded, v_rel, eta_wg, budget: EmissionBudget,
                 lambda0, omega_q_hz, xi_pol, xi_pos=1.0,
                 t_link=1.0, eta_chip=1.0, eta_det=1.0,
                 delta_partner_hz=None, beta_form: str = "purcell"):
        for name, v in (("eta_wg", eta_wg), ("xi_pol", xi_pol),
                        ("xi_pos", xi_pos), ("t_link", t_link),
                        ("eta_chip", eta_chip), ("eta_det", eta_det)):
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        if lambda0 <= 0 or not np.isfinite(lambda0):
            raise ValueError("lambda0 must be a finite positive "
                             "cyclicity (compute or measure it)")
        if not (omega_q_hz > 0 and np.isfinite(omega_q_hz)):
            raise ValueError("omega_q_hz must be positive and finite")
        for name, v in (("q_loaded", q_loaded), ("v_rel", v_rel)):
            if not (float(v) > 0 and np.isfinite(float(v))):
                raise ValueError(f"{name} must be positive and finite")
        if delta_partner_hz is not None and not (
                float(delta_partner_hz) > 0
                and np.isfinite(float(delta_partner_hz))):
            raise ValueError("delta_partner_hz must be positive and "
                             "finite (the partner line's detuning "
                             "from the cavity, Hz)")
        if beta_form not in ("purcell", "transition"):
            raise ValueError('beta_form must be "purcell" (the default) '
                             'or "transition" (the form used before '
                             f'0.6.0), not {beta_form!r}')
        self.beta_form = beta_form
        self.q = float(q_loaded)
        self.v = float(v_rel)
        self.budget = budget
        self.f_cav_hz = 299792458.0 / (budget.zpl_nm * 1e-9)
        self.kappa_hz = self.f_cav_hz / self.q
        self.omega_q_hz = float(omega_q_hz)

        a = budget.radiative_fraction
        self.f_max = purcell_max(q_loaded, v_rel)
        self.f_c = 1.0 + (self.f_max - 1.0) * xi_pol * xi_pos
        if beta_form == "purcell" and self.f_c < 1.0 - 1e-12:
            # F_max < 1 with some overlap: the cavity-added emission
            # F_C - 1 would be negative, and with it beta, eta, g and C
            # (0.6.0 returned a negative beta and eta and a NaN g)
            raise ValueError(
                f"the ideal Purcell factor F_max = {self.f_max:.4g} "
                f"(Q_L = {self.q:.4g}, V = {self.v:.4g} (lam/n)^3) is "
                "below 1, so F_C - 1 = "
                f"{self.f_c - 1.0:.3g} < 0 and the photon share of the "
                "cavity mode would be negative. This budget needs "
                "F_max >= 1 (Q_L / V >= 4 pi^2 / 3, about 13.2); raise "
                "Q_L or lower V")
        lc2 = lorentzian_suppression(omega_q_hz, self.f_cav_hz, self.q)
        self.f_c2 = 1.0 + (self.f_max - 1.0) * xi_pol * xi_pos * lc2
        self.zeta = 1.0 + a * (self.f_c - 1.0)
        zeta_flip = 1.0 + a * (self.f_c2 - 1.0)
        self.gamma_cav = budget.gamma0 * self.zeta        # 1/s
        self.tau_cav_s = 1.0 / self.gamma_cav
        self.lambda_cav = float(lambda0) * self.zeta / zeta_flip
        # the cavity-mode share of transition C's enhanced emission
        # (max: F_max equal to 1 up to round-off must not give a share
        # of -1e-16)
        f_mode = (max(self.f_c - 1.0, 0.0) if beta_form == "purcell"
                  else self.f_c)
        self.beta = a * f_mode / self.zeta
        self.eta_wg = float(eta_wg)
        self.eta = (self.beta * self.eta_wg * float(t_link)
                    * float(eta_chip) * float(eta_det))
        # coupling rate defined so F_C - 1 = 4 g^2 / (kappa gamma_rad,C)
        # (F_C for beta_form="transition"), every rate in Hz: the decay
        # rates (1/s) become linewidths gamma / (2 pi) in Hz, as in the
        # bad-cavity check below
        gamma_rad_c_hz = budget.gamma0 * a / (2.0 * np.pi)
        self.g_hz = 0.5 * np.sqrt(f_mode * self.kappa_hz
                                  * gamma_rad_c_hz)
        self.cooperativity = (4.0 * self.g_hz ** 2
                              / (self.kappa_hz * budget.gamma0
                                 / (2.0 * np.pi)))
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
