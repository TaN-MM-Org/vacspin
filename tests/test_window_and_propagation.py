"""0.7.0 anchors: the best readout window, and error bars on anything
predicted from a readout calibration.

Each check compares with something computed a different way: the
closed-form peak of the threshold-1 fidelity, a dense scan of the
fidelity, the closed-form Poisson error bar, the exact result of
first-order propagation for linear quantities, sampling of the fit's
Gaussian uncertainty, and simulated experiments of the counting
process (written from the process, not from the package's
distributions)."""
import dataclasses
import math

import numpy as np
import pytest

from vacspin import (ReadoutFit, fidelity, fit_readout_histograms,
                     optimal_window, propagate_uncertainty, snv_emission)

GAMMA0 = snv_emission().gamma0


def _p0(c, gp, t):
    """Probability of no signal photon from a flip-terminated bright
    spin: int_0^t gp e^{-gp u} e^{-c u} du + e^{-gp t} e^{-c t}."""
    a = gp + c
    return gp / a * (1.0 - math.exp(-a * t)) + math.exp(-a * t)


@pytest.mark.parametrize("eta, lam, s, noise", [
    (0.002, 2244.0, 10.0, 4000.0),       # confocal readout, example 3
    (5e-4, 2244.0, 10.0, 4000.0),
    (0.01, 300.0, 1.0, 2.0e4),
])
def test_optimal_window_matches_closed_form_peak(eta, lam, s, noise):
    """With a threshold of one photon and background rate N,
    F(tau) = 1/2 + e^{-N tau} (1 - P0(tau)) / 2 with
    P0(tau) = (gp / a)(1 - e^{-a tau}) + e^{-a tau}, a = gp + eta R.
    Since dP0/dtau = -eta R e^{-a tau}, dF/dtau = 0 gives
    e^{a tau} = 1 + a / N: tau* = ln(1 + a / N) / a. The returned window
    equals tau* (to 1e-5 relative) and its fidelity equals the closed
    form at tau* (to 1e-10), at threshold 1."""
    r = 0.5 * GAMMA0 * s / (1.0 + s)
    gp, c = r / (1.0 + lam), eta * r
    a = gp + c
    tau_star = math.log(1.0 + a / noise) / a
    f_star = 0.5 + 0.5 * math.exp(-noise * tau_star) \
        * (1.0 - _p0(c, gp, tau_star))
    out = optimal_window(eta, lam, GAMMA0, s=s, noise_rate=noise,
                         tau_max=1.0)
    assert out["threshold"] == 1
    assert abs(out["tau"] - tau_star) < 1e-5 * tau_star
    assert abs(out["fidelity"] - f_star) < 1e-10
    # and the result is the full fidelity dict at that window
    again = fidelity(eta, lam, GAMMA0, out["tau"], s=s, noise_rate=noise)
    for key, val in again.items():
        assert out[key] == val


def test_optimal_window_is_the_highest_point_of_a_dense_scan():
    """Two cases where the best threshold changes with the window (a
    leaking dark spin; the cavity design point of the test suite):
    no window of a 400-point scan, nor of a fine scan within 5 % of
    the result, beats the returned fidelity (beyond 1e-12)."""
    s = 2.0
    cases = [dict(eta=0.002, lam=2244.0, gamma=GAMMA0, s=10.0,
                  leak=0.01, noise_rate=4000.0, tau_max=1e-3),
             dict(eta=0.6150, lam=16000.0, gamma=GAMMA0 * 7.26, s=s,
                  leak=0.0037, noise_rate=1e3, tau_max=1e-6)]
    for kw in cases:
        out = optimal_window(**kw)
        tau_max = kw.pop("tau_max")

        def f(t):
            return fidelity(tau=t, **kw)["fidelity"]

        coarse = max(f(t) for t in np.geomspace(1e-9, tau_max, 400))
        fine = max(f(t) for t in np.linspace(0.95 * out["tau"],
                                             1.05 * out["tau"], 201))
        assert coarse <= out["fidelity"] + 1e-12
        assert fine <= out["fidelity"] + 1e-12
        assert out["fidelity"] > 0.5


def test_optimal_window_refusals():
    """No peak inside the range is refused, not reported as a peak:
    without background or leak the fidelity never falls; with the
    peak (16.8 us) below tau_min the best value sits at tau_min."""
    with pytest.raises(ValueError, match="no peak below tau_max"):
        optimal_window(0.002, 2244.0, GAMMA0, s=10.0, tau_max=1.0)
    with pytest.raises(ValueError, match="peak lies below"):
        optimal_window(0.002, 2244.0, GAMMA0, s=10.0, noise_rate=4000.0,
                       tau_min=1e-4, tau_max=1e-2)
    with pytest.raises(ValueError, match="eta = 0"):
        optimal_window(0.0, 2244.0, GAMMA0, noise_rate=4000.0)
    for lo, hi in ((1e-6, 1e-6), (-1.0, 1e-3), (1e-9, np.inf),
                   (np.nan, 1e-3), ("x", 1e-3)):
        with pytest.raises(ValueError, match="tau_min and tau_max"):
            optimal_window(0.002, 2244.0, GAMMA0, noise_rate=4000.0,
                           tau_min=lo, tau_max=hi)
    with pytest.raises(ValueError, match="f0"):
        optimal_window(0.002, 2244.0, GAMMA0, noise_rate=4000.0, f0=0.0)
    with pytest.raises(ValueError):
        optimal_window(0.002, 2244.0, GAMMA0, noise_rate=-1.0)


# ---------------------------------------------------------------------
# error bars on predictions from a readout calibration
# ---------------------------------------------------------------------

S = 10.0
R = 0.5 * GAMMA0 * S / (1.0 + S)
TAU = 20e-6


def _simulate(eta, lam, noise, n, rng, tau=TAU):
    """Shots of the counting process: Poisson signal at rate eta R until
    an Exp(Gp) spin flip or the end of the window, plus Poisson
    background; the dark state gives background only."""
    t_flip = rng.exponential((1.0 + lam) / R, n)
    kb = rng.poisson(eta * R * np.minimum(t_flip, tau)) \
        + rng.poisson(noise * tau, n)
    kd = rng.poisson(noise * tau, n)
    return np.bincount(kb), np.bincount(kd)


def _three_parameter_fit(seed=7, n=5000):
    hb, hd = _simulate(0.002, 2244.0, 4000.0, n,
                       np.random.default_rng(seed))
    return fit_readout_histograms(hb, GAMMA0, TAU, S, dark_hist=hd,
                                  vary=("eta", "lam", "noise_rate"))


def test_readout_propagation_is_exact_for_linear_quantities():
    """A fitted parameter returns its own error bar (1e-9 relative),
    all three return the fit's covariance (1e-8 relative), and a fixed
    linear combination A x returns A C A^T (1e-8 relative)."""
    fit = _three_parameter_fit()
    for k in fit.vary:
        q = propagate_uncertainty(fit, lambda v, k=k: v[k])
        assert q["value"] == fit.values[k]
        assert abs(q["sigma"] - fit.sigma[k]) < 1e-9 * fit.sigma[k]
    q = propagate_uncertainty(
        fit, lambda v: [v["eta"], v["lam"], v["noise_rate"]])
    assert np.allclose(q["cov"], fit.cov, rtol=1e-8, atol=0.0)
    a = np.array([[2.0e5, 1.0e-3, -0.5], [0.0, 3.0, 1.0e-2]])

    def lin(v):
        return a @ np.array([v["eta"], v["lam"], v["noise_rate"]])

    q = propagate_uncertainty(fit, lin)
    want = a @ fit.cov @ a.T
    assert np.allclose(q["cov"], want, rtol=1e-8, atol=0.0)
    assert q["sigma"].shape == (2,)


def test_readout_propagation_poisson_closed_form():
    """Poisson limit (no flips: lam held at 1e12; no background): the
    fitted eta is m / (R tau) for the mean count m of N shots, with
    error bar sqrt(m / N) / (R tau). The predicted mean count
    eta R tau of a window twice as long then has the error bar
    2 sqrt(m / N) (to 1e-6 relative)."""
    rng = np.random.default_rng(3)
    counts = rng.poisson(0.05 * R * TAU, 4000)
    m, n = counts.mean(), counts.size
    fit = fit_readout_histograms(np.bincount(counts), GAMMA0, TAU, S,
                                 lam=1e12, noise_rate=0.0,
                                 vary=("eta",))
    q = propagate_uncertainty(fit, lambda v: v["eta"] * R * 2.0 * TAU)
    assert abs(q["value"] - 2.0 * m) < 1e-6 * 2.0 * m
    assert abs(q["sigma"] - 2.0 * math.sqrt(m / n)) \
        < 1e-6 * 2.0 * math.sqrt(m / n)


def test_readout_propagation_one_sided_at_eta_one():
    """At eta = 1 the upper step leaves the model, so the slope is
    one-sided and still exact for a linear quantity (1e-9 relative)."""
    fit = ReadoutFit(values={"eta": 1.0, "lam": 50.0,
                             "noise_rate": 0.0},
                     sigma={"eta": 0.01}, cov=np.array([[1e-4]]),
                     vary=("eta",), log_likelihood=0.0, chi2=None,
                     chi2_dof=None, p_value=None, n_bright_shots=100,
                     n_dark_shots=0, condition_number=1.0, n_iter=1)
    q = propagate_uncertainty(fit, lambda v: 3.0 * v["eta"])
    assert abs(q["sigma"] - 0.03) < 1e-9 * 0.03


def test_predicted_fidelity_error_bar_matches_sampling():
    """For the fidelity of an unmeasured window (nonlinear in the
    parameters), the first-order error bar agrees within 5 % with the
    spread of the fidelity over 2000 parameter sets drawn from the
    fit's Gaussian uncertainty."""
    fit = _three_parameter_fit()
    tau = 16.8e-6

    def f(v):
        return fidelity(v["eta"], v["lam"], GAMMA0, tau, S,
                        noise_rate=v["noise_rate"])["fidelity"]

    q = propagate_uncertainty(fit, f)
    rng = np.random.default_rng(11)
    x0 = np.array([fit.values[k] for k in fit.vary])
    draws = rng.multivariate_normal(x0, fit.cov, 2000)
    spread = np.std([f(dict(zip(fit.vary, d))) for d in draws], ddof=1)
    assert abs(q["sigma"] - spread) < 0.05 * spread


def test_predicted_fidelity_error_bar_matches_simulated_experiments():
    """150 simulated experiments of 3000 bright and 3000 dark shots,
    each fitted and its fidelity at 16.8 us predicted with an error
    bar: (prediction - true fidelity) / error bar has a spread within
    20 % of 1 and a mean within 0.25 of 0."""
    tau = 16.8e-6
    rng = np.random.default_rng(11)

    def f(v):
        return fidelity(v["eta"], v["lam"], GAMMA0, tau, S,
                        noise_rate=v["noise_rate"])["fidelity"]

    truth = f({"eta": 0.002, "lam": 2244.0, "noise_rate": 4000.0})
    pulls = []
    for _ in range(150):
        hb, hd = _simulate(0.002, 2244.0, 4000.0, 3000, rng)
        fit = fit_readout_histograms(hb, GAMMA0, TAU, S, dark_hist=hd,
                                     vary=("eta", "lam", "noise_rate"))
        q = propagate_uncertainty(fit, f)
        pulls.append((q["value"] - truth) / q["sigma"])
    assert abs(np.std(pulls, ddof=1) - 1.0) < 0.2
    assert abs(np.mean(pulls)) < 0.25


def test_readout_propagation_refusals():
    fit = _three_parameter_fit()
    with pytest.raises(ValueError, match="non-finite"):
        propagate_uncertainty(fit, lambda v: float("nan"))
    with pytest.raises(ValueError, match="rel_step"):
        propagate_uncertainty(fit, lambda v: v["eta"], rel_step=0.5)
    with pytest.raises(TypeError, match="ReadoutFit"):
        propagate_uncertainty({"eta": 1.0}, lambda v: v["eta"])
    bad = dataclasses.replace(fit, cov=np.eye(2))
    with pytest.raises(ValueError, match="cov"):
        propagate_uncertainty(bad, lambda v: v["eta"])
