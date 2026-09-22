"""Cavity-interface anchors: the Purcell identity F_C = 4 g^2 /
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
    F_C = 4 g^2 / (kappa gamma_rad,C) as an identity."""
    assert abs(purcell_max(4.0 * np.pi ** 2 / 3.0, 1.0) - 1.0) < 1e-12
    ci = _ci()
    # every rate in Hz: the radiative decay rate as a linewidth
    gamma_rad_c_hz = BUD.gamma0 * BUD.radiative_fraction / (2.0 * np.pi)
    f_from_g = 4.0 * ci.g_hz ** 2 / (ci.kappa_hz * gamma_rad_c_hz)
    assert abs(f_from_g - ci.f_c) < 1e-9 * ci.f_c


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
    unit-free and must equal F_C times the radiative fraction."""
    ci = _ci()
    a = BUD.radiative_fraction
    kappa_ang = 2.0 * np.pi * ci.kappa_hz            # rad/s
    gamma_rad_ang = BUD.gamma0 * a                   # 1/s
    g_ang = 0.5 * np.sqrt(ci.f_c * kappa_ang * gamma_rad_ang)
    assert abs(ci.g_hz - g_ang / (2.0 * np.pi)) < 1e-12 * ci.g_hz
    assert abs(ci.cooperativity - ci.f_c * a) < 1e-12 * ci.cooperativity
    assert ci.validity["weak_coupling"] == (ci.g_hz < ci.kappa_hz / 10.0)
