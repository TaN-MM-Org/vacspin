"""0.5.0 lab-data anchors: fitting the readout model to measured
photon-count histograms, and goodness of fit for both fits.

Each check compares with something computed independently: the
closed-form maximum-likelihood estimates and error bars of the Poisson
and geometric limits, seeded simulations of the counting process
(written from the process, not from the package's distributions), the
chi-squared survival function's closed forms and direct numerical
integration of its density, and the spin model's own solver."""
import dataclasses
import math

import numpy as np
import pytest

from vacspin import (ReadoutFit, field_on_circle, fit_readout_histograms,
                     fit_spin_parameters, qubit_frequency, snv_emission,
                     snv_rosenthal2023)
from vacspin.lab import (_chi2_sf, _histogram_pmfs, _moment_start,
                         _observe)
from vacspin.readout import _count_distributions

GAMMA0 = snv_emission().gamma0
S = 2.0
R = 0.5 * GAMMA0 * S / (1.0 + S)


def _simulate(eta, lam, tau, noise, n, rng):
    """Shots of the counting process: Poisson signal at rate eta R until
    an Exp(Gp) spin flip or the end of the window, plus Poisson
    background; the dark state gives background only."""
    gp = R / (1.0 + lam)
    t_flip = rng.exponential(1.0 / gp, n)
    kb = rng.poisson(eta * R * np.minimum(t_flip, tau)) \
        + rng.poisson(noise * tau, n)
    kd = rng.poisson(noise * tau, n)
    return np.bincount(kb), np.bincount(kd)


def test_poisson_limit_matches_closed_form_mle():
    """With no spin flips in the window (lam fixed at 1e15) and no
    background, the bright counts are Poisson with mean eta R tau. The
    maximum-likelihood estimate is the sample mean divided by R tau,
    and the inverse Fisher information gives
    sigma = sqrt(mean / N) / (R tau) (both to 1e-6 relative)."""
    rng = np.random.default_rng(1)
    tau = 1e-7
    counts = rng.poisson(0.05 * R * tau, 5000)
    fit = fit_readout_histograms(np.bincount(counts), GAMMA0, tau, S,
                                 eta=0.1, lam=1e15, vary=("eta",))
    mean = counts.mean()
    assert isinstance(fit, ReadoutFit)
    assert abs(fit.values["eta"] - mean / (R * tau)) < 1e-6 * fit.values["eta"]
    want = math.sqrt(mean / counts.size) / (R * tau)
    assert abs(fit.sigma["eta"] - want) < 1e-6 * want
    assert fit.n_bright_shots == 5000 and fit.n_dark_shots == 0
    assert fit.values["lam"] == 1e15 and fit.values["noise_rate"] == 0.0


def test_joint_bright_dark_poisson_limit_closed_form():
    """Bright Poisson((eta R + N) tau) and dark Poisson(N tau): the
    joint estimates are N = mean_d / tau and
    eta = (mean_b - mean_d) / (R tau), and the covariance follows from
    the two Poisson means by the exact change of variables:
    var(N) = mean_d / (n_d tau^2),
    var(eta) = (mean_b / n_b + mean_d / n_d) / (R tau)^2,
    cov(eta, N) = -(mean_d / n_d) / (R tau^2)  (to 1e-6 relative)."""
    rng = np.random.default_rng(2)
    tau, eta, noise = 2e-7, 0.1, 3e6
    kb = rng.poisson((eta * R + noise) * tau, 4000)
    kd = rng.poisson(noise * tau, 3000)
    fit = fit_readout_histograms(np.bincount(kb), GAMMA0, tau, S,
                                 dark_hist=np.bincount(kd), eta=0.2,
                                 lam=1e15, noise_rate=1e6,
                                 vary=("eta", "noise_rate"))
    mb, md = kb.mean(), kd.mean()
    rt = R * tau
    assert abs(fit.values["noise_rate"] - md / tau) \
        < 1e-6 * fit.values["noise_rate"]
    assert abs(fit.values["eta"] - (mb - md) / rt) < 1e-6 * fit.values["eta"]
    want = np.array([
        [(mb / kb.size + md / kd.size) / rt ** 2,
         -(md / kd.size) / (rt * tau)],
        [-(md / kd.size) / (rt * tau), md / (kd.size * tau ** 2)]])
    assert np.allclose(fit.cov, want, rtol=1e-6, atol=0.0)


def test_geometric_limit_matches_closed_form_mle():
    """Deep in the flip-terminated regime (Gp tau = 60) the bright
    count is geometric with mean m = eta (lam + 1). The geometric
    maximum-likelihood mean is the sample mean, with
    var(m) = m (1 + m) / N; so eta = mean / (lam + 1) and
    sigma = sqrt(m (1 + m) / N) / (lam + 1) (to 1e-6 relative)."""
    rng = np.random.default_rng(3)
    lam = 30.0
    tau = 60.0 * (1.0 + lam) / R
    hb, _ = _simulate(0.4, lam, tau, 0.0, 3000, rng)
    fit = fit_readout_histograms(hb, GAMMA0, tau, S, eta=0.2, lam=lam,
                                 vary=("eta",))
    k = np.arange(hb.size)
    m = float(np.dot(k, hb) / hb.sum())
    assert abs(fit.values["eta"] - m / (lam + 1.0)) < 1e-6 * fit.values["eta"]
    want = math.sqrt(m * (1.0 + m) / hb.sum()) / (lam + 1.0)
    assert abs(fit.sigma["eta"] - want) < 1e-6 * want


def test_error_bars_match_simulated_experiments():
    """300 seeded simulated experiments of 2000 bright shots each, in
    the regime where both eta and lam matter (Gp tau about 1), each
    fitted from the automatic (method-of-moments) start: the fitted
    values scatter around the truth as the reported error bars say
    (spread of (fit - truth) / sigma within 15 % of 1, mean within
    0.25), and the model passes its own goodness-of-fit test about as
    often as it should (p > 0.01 in at least 95 % of experiments)."""
    rng = np.random.default_rng(4)
    eta, lam = 0.3, 200.0
    tau = 1.0 * (1.0 + lam) / R
    pulls, p_ok = [], 0
    for _ in range(300):
        hb, _ = _simulate(eta, lam, tau, 0.0, 2000, rng)
        fit = fit_readout_histograms(hb, GAMMA0, tau, S)
        pulls.append([(fit.values["eta"] - eta) / fit.sigma["eta"],
                      (fit.values["lam"] - lam) / fit.sigma["lam"]])
        p_ok += fit.p_value > 0.01
    pulls = np.array(pulls)
    assert np.all(np.abs(pulls.std(axis=0, ddof=1) - 1.0) < 0.15)
    assert np.all(np.abs(pulls.mean(axis=0)) < 0.25)
    assert p_ok >= 0.95 * 300


def test_goodness_of_fit_flags_a_wrong_model():
    """Bright shots where 20 % of the preparations failed (those shots
    show background only) are not described by the model: the fit's
    p-value is tiny. The same number of correctly prepared shots
    passes."""
    rng = np.random.default_rng(5)
    tau = 1e-6
    mean = 0.08 * R * tau                         # about 5.9 counts
    good = rng.poisson(mean, 5000)
    bad = np.where(rng.random(5000) < 0.2, rng.poisson(0.05, 5000),
                   rng.poisson(mean, 5000))
    kw = dict(gamma=GAMMA0, tau=tau, s=S, eta=0.1, lam=1e15,
              vary=("eta",))
    assert fit_readout_histograms(np.bincount(good), **kw).p_value > 0.01
    assert fit_readout_histograms(np.bincount(bad), **kw).p_value < 1e-6


def test_chi2_survival_function():
    """dof = 1: erfc(sqrt(x/2)); dof = 2: exp(-x/2) (closed forms, to
    1e-14); other dof: direct numerical integration of the density
    x^(k/2-1) e^(-x/2) / (2^(k/2) Gamma(k/2)) (to 1e-9). The 5 % point
    of one degree of freedom is 1.959963984540054^2."""
    for x in (0.01, 0.7, 3.0, 25.0):
        assert abs(_chi2_sf(x, 1) - math.erfc(math.sqrt(x / 2))) < 1e-14
        assert abs(_chi2_sf(x, 2) - math.exp(-x / 2)) < 1e-14
    assert abs(_chi2_sf(1.959963984540054 ** 2, 1) - 0.05) < 1e-12
    for k in (3, 4, 7, 10, 31):
        for x in (0.5, k - 1.0, 2.0 * k + 5.0):
            t = np.linspace(x, x + 400.0, 400001)
            logpdf = ((k / 2 - 1) * np.log(t) - t / 2
                      - (k / 2) * math.log(2) - math.lgamma(k / 2))
            y = np.exp(logpdf)
            h = t[1] - t[0]
            integral = h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum()
                                + 2 * y[2:-1:2].sum())
            assert abs(_chi2_sf(x, k) - integral) < 1e-9
    assert _chi2_sf(0.0, 5) == 1.0
    assert _chi2_sf(1e5, 3) == 0.0
    with pytest.raises(ValueError):
        _chi2_sf(1.0, 0)


def test_spin_fit_p_value_and_residuals():
    """`SpinFit.residuals_ghz` equals the measured values minus the
    fitted parameters' predictions from the full solver
    (`qubit_frequency`, to 1e-9 GHz); `p_value` with 2 degrees of
    freedom equals exp(-chi2/2) (to 1e-14); without measurement errors
    it is None."""
    p = snv_rosenthal2023()
    kinds = ["qubit"] * 4
    fields = np.array([field_on_circle(z, 0.15) for z in (20, 60, 100, 140)])
    rng = np.random.default_rng(6)
    values = np.array([qubit_frequency(p, b) for b in fields]) \
        + 0.05 * rng.standard_normal(4)
    fit = fit_spin_parameters(kinds, fields, values, p,
                              vary=("lam_g", "ups_g"), sigmas_ghz=0.05)
    want = values - np.array([qubit_frequency(fit.params, b)
                              for b in fields])
    assert np.max(np.abs(fit.residuals_ghz - want)) < 1e-9
    assert fit.chi2_dof == 2
    assert abs(fit.p_value - math.exp(-fit.chi2 / 2)) < 1e-14
    assert abs(fit.chi2 - np.sum((fit.residuals_ghz / 0.05) ** 2)) \
        < 1e-9 * fit.chi2
    fit_ns = fit_spin_parameters(kinds, fields, values, p,
                                 vary=("lam_g", "ups_g"))
    assert fit_ns.p_value is None
    # a SpinFit built by hand (as before 0.5.0) still works
    hand = dataclasses.replace(fit, residuals_ghz=None)
    assert hand.residuals_ghz is None and hand.p_value is not None
    assert _observe(fit.params, "qubit", fields[0], 1.0) == \
        pytest.approx(values[0] - fit.residuals_ghz[0], abs=1e-9)


def test_readout_fit_refusals():
    rng = np.random.default_rng(7)
    tau = 1e-7
    hb = np.bincount(rng.poisson(0.05 * R * tau, 500))
    kw = dict(gamma=GAMMA0, tau=tau, s=S, eta=0.1, lam=1e15)
    with pytest.raises(ValueError, match="whole numbers"):
        fit_readout_histograms([10, -1, 3], **kw)
    with pytest.raises(ValueError, match="whole numbers"):
        fit_readout_histograms([10.5, 3.0], **kw)
    with pytest.raises(ValueError, match="1-D"):
        fit_readout_histograms([[10, 3], [2, 1]], **kw)
    with pytest.raises(ValueError, match="at least 10"):
        fit_readout_histograms([3, 2, 1], **kw)
    with pytest.raises(ValueError, match="cannot vary"):
        fit_readout_histograms(hb, vary=("gamma",), **kw)
    with pytest.raises(ValueError, match="repeated"):
        fit_readout_histograms(hb, vary=("eta", "eta"), **kw)
    with pytest.raises(ValueError, match="noise_rate"):
        fit_readout_histograms(hb, vary=("eta", "noise_rate"), **kw)
    with pytest.raises(ValueError, match="eta"):
        fit_readout_histograms(hb, **dict(kw, eta=1.5), vary=("eta",))
    # without spin flips in the window the counts do not depend on lam
    with pytest.raises(ValueError, match="hold lam fixed"):
        fit_readout_histograms(hb, **kw)
    # few flips in the window (Gp tau = 0.01): lam is fitted, but its
    # error bar is larger than a quarter of the value, so it is refused
    lam = 200.0
    tau_few = 0.01 * (1.0 + lam) / R
    hb_few, _ = _simulate(0.3, lam, tau_few, 0.0, 5000,
                          np.random.default_rng(1))
    with pytest.raises(ValueError, match="poorly determined: lam"):
        fit_readout_histograms(hb_few, GAMMA0, tau_few, S, eta=0.2,
                               lam=400.0)
    # more counts than eta = 1 allows: the best fit is at the boundary
    many = np.bincount(rng.poisson(1.5 * R * tau, 500))
    with pytest.raises(ValueError, match="boundary"):
        fit_readout_histograms(many, vary=("eta",), **kw)


@pytest.mark.parametrize("eta,lam,tau,noise,leak", [
    (0.3, 200.0, 3e-6, 2e4, 0.0),
    (0.002, 2244.0, 20e-6, 4e3, 0.0),
    (0.65, 16000.0, 5e-8, 1e3, 3.7e-3),
])
def test_likelihood_pmfs_match_readout_and_quadrature(eta, lam, tau,
                                                      noise, leak):
    """The fit's pmfs are the readout model's: they equal the
    distributions `fidelity` uses (to 1e-12 absolute), and in the far
    tail of the bright signal, where an absolute error of 1e-16 would
    ruin a likelihood, they keep their relative accuracy: compared
    with log-space quadrature of int gp e^{-gp t} Po(k; c t) dt plus
    the no-flip term (to 1e-6 relative)."""
    kmax = 200
    pb, pd = _histogram_pmfs(eta, lam, noise, GAMMA0, tau, S, leak, kmax)
    qb, qd = _count_distributions(eta, lam, GAMMA0, tau, S, leak, noise,
                                  kmax)
    assert np.max(np.abs(pb - qb)) < 1e-12
    assert np.max(np.abs(pd - qd)) < 1e-12
    if noise == 0.0 or leak > 0.0:
        return
    sig, _ = _histogram_pmfs(eta, lam, 0.0, GAMMA0, tau, S, 0.0, kmax)
    gp, c = R / (1.0 + lam), eta * R
    t = np.linspace(0.0, tau, 200001)
    h = t[1] - t[0]
    for k in (150, 200):
        logf = (math.log(gp) - gp * t[1:] + k * np.log(c * t[1:])
                - c * t[1:] - math.lgamma(k + 1))
        y = np.concatenate(([0.0], np.exp(logf)))
        quad = h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum()
                        + 2 * y[2:-1:2].sum())
        quad += math.exp(-gp * tau + k * math.log(c * tau) - c * tau
                         - math.lgamma(k + 1))
        assert quad < 1e-17          # below the 1e-16 absolute round-off
        assert abs(sig[k] - quad) < 1e-6 * quad


def test_moment_start_recovers_exact_moments():
    """Fed the exact expected histogram (the pmf times 10^9 shots), the
    method-of-moments start must return the true eta, lam and
    background: its mean and variance formulas, E[D] and E[D^2] of the
    bright time, are an independent derivation from the process (to
    1e-6 relative)."""
    eta, lam, tau, noise = 0.3, 200.0, 3e-6, 2e4
    pb, pd = _histogram_pmfs(eta, lam, noise, GAMMA0, tau, S, 0.0, 400)
    got = _moment_start(1e9 * pb, 1e9 * pd, R, tau, None, None, None,
                        ("eta", "lam", "noise_rate"))
    assert abs(got["noise_rate"] - noise) < 1e-6 * noise
    assert abs(got["eta"] - eta) < 1e-6 * eta
    assert abs(got["lam"] - lam) < 1e-6 * lam


def test_start_value_refusals():
    rng = np.random.default_rng(8)
    tau = 1e-7
    hb = np.bincount(rng.poisson(0.05 * R * tau, 500))
    with pytest.raises(ValueError, match="lam is held fixed"):
        fit_readout_histograms(hb, GAMMA0, tau, S, vary=("eta",))
    with pytest.raises(ValueError, match="no counts"):
        fit_readout_histograms(hb, GAMMA0, tau, S, dark_hist=[500],
                               lam=1e15, vary=("eta", "noise_rate"))
    with pytest.raises(ValueError, match="dark histogram"):
        fit_readout_histograms(hb, GAMMA0, tau, S, lam=1e15,
                               vary=("eta", "noise_rate"))
    with pytest.raises(ValueError, match="not above the background"):
        fit_readout_histograms(hb, GAMMA0, tau, S, lam=1e15,
                               noise_rate=1e9, vary=("eta",))
