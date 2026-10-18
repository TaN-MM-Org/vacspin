"""Lifetime from a time-resolved photon histogram, and the cavity budget
from two lifetimes (both new in 0.8.0).

Checks:

* Closed form (derived here): without background, with bins of width w
  from the decay's start and a window of 40 lifetimes (so the cut-off
  tail, e^-40 = 4e-18, is negligible), the bin index of a count is
  geometric, P(k) = (1 - p) p^k with p = e^(-w / tau). Its maximum
  likelihood estimate is p = k_mean / (1 + k_mean), so
  tau = w / ln(1 + 1 / k_mean), and the Fisher information of N counts
  gives sigma_tau = tau^2 (1 - p) / (w sqrt(N p)). The free amplitude
  of the Poisson fit does not change either (its information is
  uncorrelated with tau when the window holds the whole decay).
* The exact expected histogram (computed here bin by bin with
  math.exp), scaled to 1e12 counts so that rounding to whole counts
  changes it by less than 1e-12, returns the true lifetime, amplitude
  and background.
* At the fit the Poisson score equations sum_i (n_i / mu_i - 1)
  d mu_i / d theta = 0 hold, computed here from scratch.
* 200 simulated experiments: (fit - truth) / error bar has a mean
  within 0.25 of 0 and a spread within 15 % of 1; a two-lifetime decay
  fitted with one lifetime gives a tiny p-value.
* `purcell_from_lifetimes` returns the `CavityInterface` values when
  given that budget's lifetimes (a forward and an inverse calculation),
  its error bars are exact where the result is linear in one lifetime,
  and agree with direct sampling of the inputs elsewhere."""
import math

import numpy as np
import pytest

import vacspin as vs

TAU = 0.62e-9


def _sim(rng, tau=TAU, n_sig=4000, n_bg=400, t_max=10e-9, width=16e-12):
    edges = np.arange(0.0, t_max + 0.5 * width, width)
    t = rng.exponential(tau, rng.poisson(n_sig))
    b = rng.uniform(0.0, edges[-1], rng.poisson(n_bg))
    return np.histogram(np.concatenate([t, b]), edges)[0], edges


def test_closed_form_without_background():
    rng = np.random.default_rng(42)
    w = TAU / 8
    edges = np.arange(0, 321) * w                    # 40 lifetimes
    counts = np.histogram(rng.exponential(TAU, 5000), edges)[0]
    n = counts.sum()
    k_mean = np.dot(np.arange(counts.size), counts) / n
    tau_mle = w / math.log1p(1.0 / k_mean)
    p = math.exp(-w / tau_mle)
    sigma = tau_mle ** 2 * (1 - p) / (w * math.sqrt(n * p))
    fit = vs.fit_lifetime_histogram(counts, edges, background=False)
    assert fit.tau_s == pytest.approx(tau_mle, rel=1e-9)
    assert fit.sigma_tau_s == pytest.approx(sigma, rel=1e-6)
    assert fit.values["amplitude"] == pytest.approx(n, rel=1e-9)
    assert fit.values["background_rate"] == 0.0
    assert fit.vary == ("tau_s", "amplitude")


def test_exact_expected_histogram_returns_the_truth():
    tau, amp, bg = 0.8e-9, 1e12, 3e19          # bg: counts per second
    edges = np.arange(-1e-9, 15e-9 + 1e-15, 25e-12)
    mu = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        dec = 0.0
        if hi > 0.0:
            a = max(lo, 0.0)
            dec = amp * (math.exp(-a / tau) - math.exp(-hi / tau))
        mu.append(dec + bg * (hi - lo))
    counts = np.round(np.array(mu))
    fit = vs.fit_lifetime_histogram(counts, edges, t_start=0.5e-9)
    t0 = fit.t_start_s
    assert t0 >= 0.5e-9 and t0 - 0.5e-9 < 25e-12
    assert fit.tau_s == pytest.approx(tau, rel=1e-9)
    assert fit.values["amplitude"] == pytest.approx(
        amp * math.exp(-t0 / tau), rel=1e-8)
    assert fit.values["background_rate"] == pytest.approx(bg, rel=1e-8)
    assert fit.n_bins == int(np.sum(edges[:-1] >= 0.5e-9 - 1e-21))
    assert fit.chi2 < 1e-6


def test_score_equations_hold_at_the_fit():
    rng = np.random.default_rng(7)
    counts, edges = _sim(rng)
    fit = vs.fit_lifetime_histogram(counts, edges)
    tau, amp, bg = (fit.values[k] for k in
                    ("tau_s", "amplitude", "background_rate"))
    e0 = np.exp(-(edges[:-1] - edges[0]) / tau)
    e1 = np.exp(-(edges[1:] - edges[0]) / tau)
    t_lo, t_hi = edges[:-1] - edges[0], edges[1:] - edges[0]
    u = e0 - e1
    du = (t_lo * e0 - t_hi * e1) / tau ** 2
    w = np.diff(edges)
    mu = amp * u + bg * w
    r = counts / mu - 1.0
    assert np.sum(mu) == pytest.approx(counts.sum(), rel=1e-9)
    for grad, val in ((amp * du, tau), (u, amp), (w, bg)):
        # score times the parameter: a dimensionless number of counts
        assert abs(np.dot(r, grad) * val) < 1e-6 * counts.sum()


def test_error_bars_match_simulated_experiments():
    rng = np.random.default_rng(2026)
    pulls, pvals = [], []
    for _ in range(200):
        counts, edges = _sim(rng)
        fit = vs.fit_lifetime_histogram(counts, edges)
        pulls.append((fit.tau_s - TAU) / fit.sigma_tau_s)
        pvals.append(fit.p_value)
    pulls = np.array(pulls)
    assert abs(pulls.mean()) < 0.25
    assert abs(pulls.std(ddof=1) - 1.0) < 0.15
    assert np.mean(np.array(pvals) > 0.01) >= 0.95


def test_two_lifetimes_are_flagged():
    rng = np.random.default_rng(3)
    edges = np.arange(0.0, 20e-9, 20e-12)
    t = np.concatenate([rng.exponential(0.6e-9, 10000),
                        rng.exponential(3.0e-9, 10000)])
    fit = vs.fit_lifetime_histogram(np.histogram(t, edges)[0], edges,
                                    background=False)
    assert fit.p_value < 1e-6
    one = np.histogram(rng.exponential(0.6e-9, 20000), edges)[0]
    assert vs.fit_lifetime_histogram(one, edges,
                                     background=False).p_value > 0.01


def test_lifetime_propagation():
    rng = np.random.default_rng(11)
    counts, edges = _sim(rng)
    fit = vs.fit_lifetime_histogram(counts, edges)
    q = vs.propagate_uncertainty(fit, lambda v: v["tau_s"])
    assert q["sigma"] == pytest.approx(fit.sigma_tau_s, rel=1e-9)
    rate = vs.propagate_uncertainty(fit, lambda v: 1.0 / v["tau_s"])
    assert rate["sigma"] == pytest.approx(
        fit.sigma_tau_s / fit.tau_s ** 2, rel=1e-6)
    a = np.array([[1e9, 2e-4, 1e-10], [0.0, 1.0, -3e-9]])
    lin = vs.propagate_uncertainty(
        fit, lambda v: a @ [v["tau_s"], v["amplitude"],
                            v["background_rate"]])
    assert np.allclose(lin["cov"], a @ fit.cov @ a.T, rtol=1e-8,
                       atol=0.0)


def _design(q, xi_pos, eta_q=0.8):
    budget = vs.EmissionBudget(tau0_s=4.5e-9, eta_q=eta_q, eta_dw=0.57,
                               eta_br=0.75, zpl_nm=619.0,
                               reference="test budget, values of "
                                         "snv_emission")
    ci = vs.CavityInterface(q_loaded=q, v_rel=0.68, eta_wg=0.9,
                            budget=budget, lambda0=2244.0,
                            omega_q_hz=3.6e9, xi_pol=vs.xi_pol_overlap(),
                            xi_pos=xi_pos)
    return budget, ci


@pytest.mark.parametrize("q, xi_pos, eta_q", [(50, 0.3, 0.8),
                                              (500, 0.5, 0.8),
                                              (5000, 1.0, 0.5)])
def test_purcell_round_trip_with_cavity_interface(q, xi_pos, eta_q):
    budget, ci = _design(q, xi_pos, eta_q)
    r = vs.purcell_from_lifetimes(budget.tau0_s, ci.tau_cav_s,
                                  budget=budget, kappa_hz=ci.kappa_hz)
    want = dict(zeta=ci.zeta, cooperativity=ci.cooperativity,
                beta=ci.beta, f_c=ci.f_c, g_hz=ci.g_hz)
    assert r["names"] == ("zeta", "cooperativity", "beta", "f_c", "g_hz")
    for k, v in want.items():
        assert r["values"][k] == pytest.approx(v, rel=1e-12)
        assert r["sigma"][k] == 0.0


def test_purcell_error_bars():
    t_r, t_c = 4.5e-9, 0.62e-9
    # linear in one lifetime: exact
    r = vs.purcell_from_lifetimes(t_r, t_c, sigma_cav_s=0.01e-9)
    assert r["sigma"]["beta"] == pytest.approx(0.01e-9 / t_r, rel=1e-12)
    r = vs.purcell_from_lifetimes(t_r, t_c, sigma_ref_s=0.05e-9)
    assert r["sigma"]["zeta"] == pytest.approx(0.05e-9 / t_c, rel=1e-12)
    # nonlinear: against sampling of small, independent input errors
    budget = vs.snv_emission()
    s_r, s_c, kap, s_k = 0.02e-9, 0.004e-9, 9.7e11, 1e10
    r = vs.purcell_from_lifetimes(t_r, t_c, s_r, s_c, budget=budget,
                                  kappa_hz=kap, sigma_kappa_hz=s_k)
    rng = np.random.default_rng(5)
    n = 400000
    tr = t_r + s_r * rng.standard_normal(n)
    tc = t_c + s_c * rng.standard_normal(n)
    kk = kap + s_k * rng.standard_normal(n)
    c = tr / tc - 1.0
    samples = dict(zeta=tr / tc, cooperativity=c, beta=1 - tc / tr,
                   f_c=1 + c / budget.radiative_fraction,
                   g_hz=0.5 * np.sqrt(c * kk / (2 * np.pi * tr)))
    for k, v in samples.items():
        assert r["sigma"][k] == pytest.approx(np.std(v), rel=0.02)
    # the covariance carries the correlations
    corr = np.corrcoef(samples["zeta"], samples["beta"])[0, 1]
    i, j = r["names"].index("zeta"), r["names"].index("beta")
    assert r["cov"][i, j] / (r["sigma"]["zeta"] * r["sigma"]["beta"]) \
        == pytest.approx(corr, abs=0.01)


def test_purcell_from_lifetime_fits():
    rng = np.random.default_rng(8)
    fits = []
    for tau in (4.5e-9, 0.62e-9):
        counts, edges = _sim(rng, tau=tau, n_sig=20000, t_max=50e-9,
                             width=25e-12)
        fits.append(vs.fit_lifetime_histogram(counts, edges))
    a = vs.purcell_from_lifetimes(*fits)
    b = vs.purcell_from_lifetimes(fits[0].tau_s, fits[1].tau_s,
                                  fits[0].sigma_tau_s, fits[1].sigma_tau_s)
    for k in a["names"]:
        assert a["values"][k] == b["values"][k]
        assert a["sigma"][k] == b["sigma"][k]
    with pytest.raises(ValueError, match="carries its own error bar"):
        vs.purcell_from_lifetimes(fits[0], fits[1], sigma_ref_s=1e-11)


def test_purcell_refusals():
    with pytest.raises(ValueError, match="not shorter"):
        vs.purcell_from_lifetimes(1e-9, 1e-9)
    with pytest.raises(ValueError, match="not shorter"):
        vs.purcell_from_lifetimes(1e-9, 2e-9)
    for bad in (0.0, -1e-9, float("nan"), float("inf"), "x"):
        with pytest.raises(ValueError):
            vs.purcell_from_lifetimes(bad, 1e-9)
        with pytest.raises(ValueError):
            vs.purcell_from_lifetimes(4e-9, bad)
    for bad in (-1e-12, float("nan")):
        with pytest.raises(ValueError, match="error bar"):
            vs.purcell_from_lifetimes(4e-9, 1e-9, sigma_ref_s=bad)
    with pytest.raises(ValueError, match="EmissionBudget"):
        vs.purcell_from_lifetimes(4e-9, 1e-9, budget="snv")
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="kappa_hz"):
            vs.purcell_from_lifetimes(4e-9, 1e-9, kappa_hz=bad)
    with pytest.raises(ValueError, match="sigma_kappa_hz"):
        vs.purcell_from_lifetimes(4e-9, 1e-9, kappa_hz=1e12,
                                  sigma_kappa_hz=-1.0)
    with pytest.raises(ValueError, match="quarter"):
        vs.purcell_from_lifetimes(4.5e-9, 4.0e-9, sigma_cav_s=0.2e-9)


def test_lifetime_fit_refusals():
    rng = np.random.default_rng(1)
    counts, edges = _sim(rng)
    for bad in (np.vstack([counts, counts]), -counts, counts + 0.5,
                np.where(counts > 5, np.nan, counts), []):
        with pytest.raises(ValueError, match="counts"):
            vs.fit_lifetime_histogram(bad, edges)
    with pytest.raises(ValueError, match="one more entry"):
        vs.fit_lifetime_histogram(counts, edges[:-1])
    with pytest.raises(ValueError, match="increasing"):
        vs.fit_lifetime_histogram(counts, edges[::-1])
    with pytest.raises(ValueError, match="at least 10"):
        vs.fit_lifetime_histogram(np.minimum(counts, 1) * (np.arange(
            counts.size) < 5), edges)
    with pytest.raises(ValueError, match="bins in the fitted range"):
        vs.fit_lifetime_histogram(counts, edges, t_start=edges[-3])
    for bad in (float("nan"), "x"):
        with pytest.raises(ValueError, match="t_start"):
            vs.fit_lifetime_histogram(counts, edges, t_start=bad)
    for bad in (0.0, -1e-9, float("inf")):
        with pytest.raises(ValueError, match="tau_start"):
            vs.fit_lifetime_histogram(counts, edges, tau_start=bad)
    with pytest.raises(ValueError, match="True or False"):
        vs.fit_lifetime_histogram(counts, edges, background="yes")
    # background-free data: the background ends at its limit 0
    clean = np.histogram(rng.exponential(TAU, 20000), edges)[0]
    with pytest.raises(ValueError, match="background=False"):
        vs.fit_lifetime_histogram(clean, edges)
    assert vs.fit_lifetime_histogram(clean, edges,
                                     background=False).p_value > 0.01
    # a lifetime far longer than the window, or far shorter than a bin
    slow = np.histogram(rng.exponential(1e-6, 5000), edges)[0]
    with pytest.raises(ValueError):
        vs.fit_lifetime_histogram(slow, edges, background=False)
    fast = np.histogram(rng.exponential(1e-13, 5000), edges)[0]
    with pytest.raises(ValueError):
        vs.fit_lifetime_histogram(fast, edges, background=False)
