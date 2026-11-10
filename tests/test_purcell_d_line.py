"""0.10.0: the measured cavity budget with the D line.

`purcell_from_lifetimes(..., include_d_line=True)` inverts the budget of
`CavityInterface(..., include_d_line=True)` (the D-line term of the
revised source study's Eq. 1). Checked against: the forward budget
itself at three cavity linewidths (a separate code path, which builds
F_C from Q and V and the D-line Purcell factor from the cavity
Lorentzian), central finite differences of the inverse for its slopes,
and 400000 sampled inputs for its error bars and correlations.
"""
import numpy as np
import pytest

import vacspin as vs

BUD = vs.snv_emission()
DELTA_D = 9.0349e11          # the C-D spacing used in test_fourlevel (Hz)
NAMES = ("zeta", "cooperativity", "beta", "f_c", "g_hz", "f_partner")


def _design(q):
    """A Q_L = q cavity with F_max = 82 and the D line counted."""
    return vs.CavityInterface(
        q_loaded=q, v_rel=vs.purcell_max(q, 1.0) / 82.0, eta_wg=0.92,
        budget=BUD, lambda0=2180.0, omega_q_hz=3.677e9, xi_pol=2.0 / 3.0,
        xi_pos=0.5, delta_partner_hz=DELTA_D, include_d_line=True)


@pytest.mark.parametrize("q", [300.0, 870.0, 3000.0])
def test_round_trip_with_the_d_line_budget(q):
    """Given the two lifetimes of a CavityInterface design with the D line
    (the bulk lifetime and tau_cav_s), the inverse returns that design's
    zeta, C-line cooperativity, beta, F_C, g_hz and D-line Purcell
    factor (to 1e-12). The cavity linewidths (1.6, 0.56 and 0.16 THz)
    put the D line at different points of the Lorentzian."""
    ci = _design(q)
    r = vs.purcell_from_lifetimes(BUD.tau0_s, ci.tau_cav_s, budget=BUD,
                                  kappa_hz=ci.kappa_hz,
                                  delta_partner_hz=DELTA_D,
                                  include_d_line=True)
    want = dict(zeta=ci.zeta, cooperativity=ci.cooperativity,
                beta=ci.beta, f_c=ci.f_c, g_hz=ci.g_hz,
                f_partner=ci.f_partner)
    assert r["names"] == NAMES
    for k, v in want.items():
        assert r["values"][k] == pytest.approx(v, rel=1e-12)
        assert r["sigma"][k] == 0.0


def test_the_revised_design_point():
    """At the revised study's design point (Q_L = 870, F_C = 28) the
    default inverse reads the D-line emission as C-line enhancement and
    gives F_C = 28.78; the D-line inverse gives 28.00. Both agree on
    zeta (10.50) and beta."""
    ci = _design(870.0)
    assert ci.f_c == pytest.approx(28.0, rel=1e-12)
    plain = vs.purcell_from_lifetimes(BUD.tau0_s, ci.tau_cav_s, budget=BUD)
    with_d = vs.purcell_from_lifetimes(BUD.tau0_s, ci.tau_cav_s,
                                       budget=BUD, kappa_hz=ci.kappa_hz,
                                       delta_partner_hz=DELTA_D,
                                       include_d_line=True)
    assert plain["values"]["f_c"] == pytest.approx(28.78, abs=0.005)
    assert with_d["values"]["f_c"] == pytest.approx(28.0, rel=1e-12)
    for k in ("zeta", "beta"):
        assert with_d["values"][k] == pytest.approx(plain["values"][k],
                                                    rel=1e-14)
    assert with_d["values"]["zeta"] == pytest.approx(10.50, abs=0.005)


def _inverse(x):
    r = vs.purcell_from_lifetimes(x[0], x[1], budget=BUD, kappa_hz=x[2],
                                  delta_partner_hz=DELTA_D,
                                  include_d_line=True)
    return np.array([r["values"][k] for k in NAMES])


def test_slopes_equal_finite_differences():
    """The covariance J C J^T uses exact derivatives; built instead from
    central finite differences of the inverse itself, it is the same to
    1e-6 (relative, every entry)."""
    x0 = np.array([4.5e-9, 0.43e-9, 5.6e11])
    sig = np.array([0.03e-9, 0.005e-9, 1e10])
    r = vs.purcell_from_lifetimes(*x0[:2], *sig[:2], budget=BUD,
                                  kappa_hz=x0[2], sigma_kappa_hz=sig[2],
                                  delta_partner_hz=DELTA_D,
                                  include_d_line=True)
    jac = np.empty((len(NAMES), 3))
    for j in range(3):
        h = 1e-6 * x0[j]
        xp, xm = x0.copy(), x0.copy()
        xp[j] += h
        xm[j] -= h
        jac[:, j] = (_inverse(xp) - _inverse(xm)) / (2.0 * h)
    cov = jac @ np.diag(sig ** 2) @ jac.T
    assert np.max(np.abs(cov / r["cov"] - 1.0)) < 1e-6


def test_error_bars_match_sampling():
    """Against 400000 sampled inputs (small, independent Gaussian errors
    on both lifetimes and the cavity linewidth): every error bar within
    2 %, and the F_C-beta correlation within 0.01."""
    t_r, t_c, kap = 4.5e-9, 0.43e-9, 5.6e11
    s_r, s_c, s_k = 0.02e-9, 0.003e-9, 1e10
    r = vs.purcell_from_lifetimes(t_r, t_c, s_r, s_c, budget=BUD,
                                  kappa_hz=kap, sigma_kappa_hz=s_k,
                                  delta_partner_hz=DELTA_D,
                                  include_d_line=True)
    rng = np.random.default_rng(9)
    n = 400000
    tr = t_r + s_r * rng.standard_normal(n)
    tc = t_c + s_c * rng.standard_normal(n)
    kk = kap + s_k * rng.standard_normal(n)
    a = BUD.radiative_fraction
    b = BUD.eta_q * BUD.eta_dw * (1.0 - BUD.eta_br)
    lor = 1.0 / (1.0 + (2.0 * DELTA_D / kk) ** 2)
    fm1 = (tr / tc - 1.0) / (a + b * lor)
    samples = dict(zeta=tr / tc, cooperativity=a * fm1, beta=1 - tc / tr,
                   f_c=1 + fm1,
                   g_hz=0.5 * np.sqrt(a * fm1 * kk / (2 * np.pi * tr)),
                   f_partner=1 + fm1 * lor)
    for k, v in samples.items():
        assert r["sigma"][k] == pytest.approx(np.std(v), rel=0.02)
    corr = np.corrcoef(samples["f_c"], samples["beta"])[0, 1]
    i, j = NAMES.index("f_c"), NAMES.index("beta")
    assert r["cov"][i, j] / (r["sigma"]["f_c"] * r["sigma"]["beta"]) \
        == pytest.approx(corr, abs=0.01)


def test_default_is_unchanged():
    """include_d_line=False (the default) is the 0.8.0 inverse: passing
    it, or a delta_partner_hz without it, changes nothing."""
    args = (4.5e-9, 0.62e-9, 0.02e-9, 0.004e-9)
    kw = dict(budget=BUD, kappa_hz=9.7e11, sigma_kappa_hz=1e10)
    a = vs.purcell_from_lifetimes(*args, **kw)
    for extra in (dict(include_d_line=False),
                  dict(delta_partner_hz=DELTA_D)):
        b = vs.purcell_from_lifetimes(*args, **kw, **extra)
        assert b["names"] == a["names"] == NAMES[:5]
        for k in a["names"]:
            assert b["values"][k] == a["values"][k]
            assert b["sigma"][k] == a["sigma"][k]


def test_refusals():
    t_r, t_c = 4.5e-9, 0.43e-9
    with pytest.raises(ValueError, match="True or False"):
        vs.purcell_from_lifetimes(t_r, t_c, include_d_line=1)
    for missing in ("budget", "kappa_hz", "delta_partner_hz"):
        kw = dict(budget=BUD, kappa_hz=5.6e11, delta_partner_hz=DELTA_D)
        kw.pop(missing)
        with pytest.raises(ValueError, match="needs budget, kappa_hz"):
            vs.purcell_from_lifetimes(t_r, t_c, include_d_line=True, **kw)
    for bad in (0.0, -1e11, float("nan"), float("inf"), "x"):
        with pytest.raises(ValueError, match="delta_partner_hz"):
            vs.purcell_from_lifetimes(t_r, t_c, budget=BUD, kappa_hz=5.6e11,
                                      delta_partner_hz=bad,
                                      include_d_line=True)
    with pytest.raises(ValueError, match="quarter"):
        vs.purcell_from_lifetimes(4.5e-9, 4.0e-9, sigma_cav_s=0.2e-9,
                                  budget=BUD, kappa_hz=5.6e11,
                                  delta_partner_hz=DELTA_D,
                                  include_d_line=True)
