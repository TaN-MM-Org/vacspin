"""Cavity-interface anchors: the Purcell identity F_C - 1 = 4 g^2 /
(kappa gamma_rad) as an exact round trip, exact Lorentzian limits,
the photon-budget bounds beta <= 1 and eta <= 1 with monotonicity,
cavity-boosted cyclicity limits, and the validity refusal."""
import numpy as np
import pytest

from vacspin import (CavityInterface, lorentzian_suppression,
                     purcell_max, snv_emission, xi_pol_overlap)

BUD = snv_emission()


def _ci(q=500.0, **kw):
    args = dict(q_loaded=q, v_rel=0.68, eta_wg=0.999, budget=BUD,
                lambda0=2244.0, omega_q_hz=3.677e9,
                xi_pol=xi_pol_overlap(), xi_pos=0.5)
    args.update(kw)
    return CavityInterface(**args)


def test_purcell_formula_and_identity():
    """F_max = (3/4pi^2) Q/V exactly, and the coupling rate satisfies
    F_C - 1 = 4 g^2 / (kappa gamma_rad,C) as an identity (F_C with
    beta_form="transition", the form used before 0.6.0)."""
    assert abs(purcell_max(4.0 * np.pi ** 2 / 3.0, 1.0) - 1.0) < 1e-12
    ci = _ci()
    # every rate in Hz: the radiative decay rate as a linewidth
    gamma_rad_c_hz = BUD.gamma0 * BUD.radiative_fraction / (2.0 * np.pi)
    f_from_g = 4.0 * ci.g_hz ** 2 / (ci.kappa_hz * gamma_rad_c_hz)
    assert abs(f_from_g - (ci.f_c - 1.0)) < 1e-9 * ci.f_c
    old = _ci(beta_form="transition")
    f_from_g = 4.0 * old.g_hz ** 2 / (old.kappa_hz * gamma_rad_c_hz)
    assert abs(f_from_g - old.f_c) < 1e-9 * old.f_c


def test_lorentzian_exact_limits():
    assert lorentzian_suppression(0.0, 4.8e14, 500) == 1.0
    # exactly 1/2 at delta = kappa/2 = f/(2Q)
    f0, q = 4.8e14, 500.0
    assert abs(lorentzian_suppression(f0 / (2 * q), f0, q) - 0.5) < 1e-12
    assert lorentzian_suppression(1e18, f0, q) < 1e-9


def test_xi_pol_overlap_closed_form():
    """2/3 exactly at theta = arccos(1/sqrt 3), psi = 45 deg; zero for
    a dipole along the mode normal."""
    assert abs(xi_pol_overlap() - 2.0 / 3.0) < 1e-6
    assert xi_pol_overlap(theta_deg=0.0) == 0.0


def test_budget_bounds_and_monotonicity():
    """beta and eta stay in [0, 1] for any Q; beta increases with Q
    toward the radiative-fraction-independent limit 1; zeta is exactly
    linear in F_C."""
    a = BUD.radiative_fraction
    betas = []
    for q in (10.0, 100.0, 1000.0, 1e5, 1e7):
        ci = _ci(q=q)
        assert 0.0 < ci.beta <= 1.0
        assert 0.0 < ci.eta <= 1.0
        assert abs(ci.zeta - (1.0 + a * (ci.f_c - 1.0))) < 1e-12
        betas.append(ci.beta)
    assert all(b2 > b1 for b1, b2 in zip(betas, betas[1:]))
    assert betas[-1] > 0.99                      # -> 1 at large Purcell


def test_cavity_boosted_cyclicity():
    """The cavity enhances only the spin-conserving line (the partner
    is detuned by omega_q), so Lambda_cav >= Lambda_0, approaching
    Lambda_0 * zeta when the partner suppression is complete -- and
    collapsing back to Lambda_0 when omega_q >> kappa fails."""
    ci = _ci(q=1.5e3)
    assert ci.lambda_cav >= 2244.0
    zeta_flip = 1.0 + BUD.radiative_fraction * (ci.f_c2 - 1.0)
    assert abs(ci.lambda_cav - 2244.0 * ci.zeta / zeta_flip) < 1e-9 * \
        ci.lambda_cav
    # tiny Q -> broad cavity -> partner equally enhanced -> no boost
    ci_broad = _ci(q=10.0)
    assert ci_broad.lambda_cav / 2244.0 < 1.05


def test_validity_refusal_fires_and_passes():
    ok = _ci(q=500.0).require_valid()            # the design regime
    assert all(ok.validity.values())
    with pytest.raises(ValueError, match="regime"):
        _ci(q=2e7).require_valid()               # huge Q: conditions fail


def test_input_refusals():
    with pytest.raises(ValueError):
        _ci(eta_wg=1.5)
    with pytest.raises(ValueError):
        _ci(xi_pos=-0.1)
    with pytest.raises(ValueError):
        _ci(lambda0=0.0)
    with pytest.raises(ValueError):
        purcell_max(-1.0, 1.0)
    with pytest.raises(ValueError):
        lorentzian_suppression(1.0, -5.0, 100.0)
    for bad in (np.nan, np.inf):
        with pytest.raises(ValueError, match="finite"):
            xi_pol_overlap(theta_deg=bad)
        with pytest.raises(ValueError, match="finite"):
            xi_pol_overlap(psi_deg=bad)


def test_partner_selectivity_reporting():
    """With the D-line detuning supplied, the partner Purcell factor
    is suppressed by the exact Lorentzian and the selectivity flag
    tracks kappa < delta."""
    ci = _ci(q=500.0, delta_partner_hz=2.097e12)
    assert ci.f_partner < ci.f_c
    assert ci.validity["partner_selectivity"] == (ci.kappa_hz < 2.097e12)


def test_coupling_rate_in_consistent_hz_units():
    """g_hz is a frequency in Hz like kappa_hz: computed in angular
    units (kappa = 2 pi f_cav / Q_L, gamma = 1/tau in 1/s) and divided
    by 2 pi. Before 0.3.1 the code multiplied kappa in Hz by gamma in
    1/s, which made g_hz too large by sqrt(2 pi) and made the
    weak-coupling check stricter than stated. The cooperativity is
    unit-free and must equal F_C - 1 times the radiative fraction
    (F_C times it with beta_form="transition")."""
    a = BUD.radiative_fraction
    for form, f_mode in (("purcell", lambda c: c.f_c - 1.0),
                         ("transition", lambda c: c.f_c)):
        ci = _ci(beta_form=form)
        kappa_ang = 2.0 * np.pi * ci.kappa_hz            # rad/s
        gamma_rad_ang = BUD.gamma0 * a                   # 1/s
        g_ang = 0.5 * np.sqrt(f_mode(ci) * kappa_ang * gamma_rad_ang)
        assert abs(ci.g_hz - g_ang / (2.0 * np.pi)) < 1e-12 * ci.g_hz
        assert abs(ci.cooperativity - f_mode(ci) * a) < \
            1e-12 * ci.cooperativity
    assert ci.validity["weak_coupling"] == (ci.g_hz < ci.kappa_hz / 10.0)


def test_only_the_cavity_added_emission_goes_into_the_mode():
    """Since 0.6.0 (beta_form="purcell"): transition C decays F_C times
    faster than in bulk (Lee et al., arXiv:2511.05740, Eq. 2), and only
    the added part (F_C - 1) goes into the cavity mode. With C = a (F_C
    - 1) the budget obeys the bad-cavity relations zeta = 1 + C and
    beta = C / (1 + C), and with no overlap beta, g and C are exactly
    0. In 0.5.0 beta used F_C: 0.342 (the radiative fraction a) with
    no overlap, and 0.9093 instead of 0.8622 at Q_L = 500."""
    a = BUD.radiative_fraction
    for q in (10.0, 500.0, 1e5):
        ci = _ci(q=q)
        c = a * (ci.f_c - 1.0)
        assert abs(ci.cooperativity - c) < 1e-12 * c
        assert abs(ci.zeta - (1.0 + c)) < 1e-12
        assert abs(ci.beta - c / (1.0 + c)) < 1e-12
    none = _ci(xi_pos=0.0)
    assert none.f_c == 1.0
    assert none.beta == 0.0 and none.g_hz == 0.0
    assert none.cooperativity == 0.0 and none.eta == 0.0
    old = _ci(xi_pos=0.0, beta_form="transition")
    assert abs(old.beta - a) < 1e-15
    new, old = _ci(), _ci(beta_form="transition")
    assert abs(old.beta - a * old.f_c / old.zeta) < 1e-15
    assert abs(new.beta - 0.8622) < 5e-5 and abs(old.beta - 0.9093) < 5e-5
    # everything that does not involve the mode share is the same
    for name in ("f_max", "f_c", "f_c2", "zeta", "gamma_cav",
                 "lambda_cav", "kappa_hz"):
        assert getattr(new, name) == getattr(old, name)
    with pytest.raises(ValueError, match="beta_form"):
        _ci(beta_form="total")


def test_cavity_with_f_max_below_one_is_refused():
    """Since 0.7.0. A large-volume resonator (Q_L = 1000, V = 100
    (lam/n)^3) has F_max = (3 / 4 pi^2) Q / V = 0.760 < 1, so
    F_C = 1 + (F_max - 1) xi = 0.920 and the cavity-mode share
    a (F_C - 1) / zeta is negative. 0.6.0 returned beta = eta = -0.0281
    and g_hz = NaN (and require_valid() then blamed weak coupling); the
    interface now refuses and says why. With no overlap F_C = 1 and
    nothing changes (beta = 0); the "transition" form keeps its 0.5.0
    numbers (beta = a F_C / zeta); and F_max = 1 up to round-off
    (Q / V = 4 pi^2 / 3) gives beta = 0, not a negative value."""
    assert abs(purcell_max(1000.0, 100.0) - 0.7599) < 1e-4
    with pytest.raises(ValueError, match="F_max = 0.7599.*below 1"):
        _ci(q=1000.0, v_rel=100.0, xi_pos=0.5)
    none = _ci(q=1000.0, v_rel=100.0, xi_pos=0.0)
    assert none.f_c == 1.0 and none.beta == 0.0 and none.g_hz == 0.0
    old = _ci(q=1000.0, v_rel=100.0, beta_form="transition")
    a = BUD.radiative_fraction
    assert abs(old.beta - a * old.f_c / old.zeta) < 1e-15
    assert 0.0 < old.beta < 1.0 and np.isfinite(old.g_hz)
    edge = _ci(q=4.0 * np.pi ** 2 / 3.0, v_rel=1.0)
    assert 0.0 <= edge.beta < 1e-12 and np.isfinite(edge.g_hz)
