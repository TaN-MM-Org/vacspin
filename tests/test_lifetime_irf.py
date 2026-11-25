"""0.11.0: lifetime fits with a measured instrument response
(`fit_lifetime_histogram(..., irf=...)`), and decay left over from
earlier excitation pulses.

Checks:

* The response kernel (a decay that starts at a time spread evenly over
  one bin) against brute-force averaging over 200000 start times, and
  its slope against central differences.
* A response that sits in a single bin: after that bin the kernel is a
  plain exponential with the same lifetime, so the reconvolution fit
  from the next bin on must equal the plain fit from there.
* The exact expected histogram, built here with the brute-force kernel
  and scaled to 1e12 counts, returns the true lifetime, amplitude and
  background.
* A continuous Gaussian response. A decay that starts at a Gaussian
  time t0 + sigma Z has the distribution function (derived by
  completing the square in the Gaussian integral)
      P(T <= t) = Phi(a) - exp(-(t - t0) / tau + sigma^2 / (2 tau^2))
                  Phi(a - sigma / tau),   a = (t - t0) / sigma,
  which gives the exact expected counts. The fit takes the response as
  constant within each bin, so a small bias remains; it is measured
  here.
* 200 simulated experiments with that Gaussian response: (fit - truth)
  / error bar has a mean within 0.25 of 0 and a spread within 15 % of
  1, and at least 95 % of the p-values exceed 0.01.
* Leftover decay: the histogram of a pulse train (60 earlier pulses
  summed here term by term) gives the plain fit the true lifetime, and
  an amplitude larger by 1 / (1 - e^(-T / tau)).
"""
import math

import numpy as np
import pytest

import vacspin as vs
from vacspin.lab import _irf_kernel

W = 16e-12          # bin width
TAU = 0.6e-9


def _brute_kernel(w, tau, m, n=200000):
    """Share of the decay in bins 0..m-1, averaged over n start times
    spread evenly over bin 0 (midpoints)."""
    s = (np.arange(n) + 0.5) / n * w
    out = np.empty(m)
    for j in range(m):
        lo = np.maximum(j * w - s, 0.0)
        hi = np.maximum((j + 1) * w - s, 0.0)
        out[j] = np.mean(np.exp(-lo / tau) - np.exp(-hi / tau))
    return out


def test_kernel_against_brute_force():
    """Closed form against brute-force averaging (to 1e-10; measured
    8e-16, 1e-13 and 5e-12, the last limited by the averaging itself for
    bins five lifetimes wide), the slope against central differences of
    the brute-force kernel (to 1e-6 of its largest value; measured
    1e-10), and the shares add up to 1 (to 1e-12; measured 1e-16)."""
    for tau in (TAU, 3 * W, 0.2 * W):
        k, dk = _irf_kernel(W, tau, 60)
        assert np.max(np.abs(k - _brute_kernel(W, tau, 60))) < 1e-10
        h = 1e-5 * tau
        fd = (_brute_kernel(W, tau + h, 60)
              - _brute_kernel(W, tau - h, 60)) / (2 * h)
        assert np.max(np.abs(dk - fd)) < 1e-6 * np.max(np.abs(dk))
        m = int(60 * tau / W) + 200
        assert abs(_irf_kernel(W, tau, m)[0].sum() - 1.0) < 1e-12


def _edges(n_bins):
    return np.arange(n_bins + 1) * W


def test_single_bin_response_equals_the_plain_fit():
    """All of the response in bin 20: from bin 21 on, the reconvolution
    model is A phi(y) (1 - e^-y) e^(-(i - 21) y) plus background, the
    plain model with amplitude A phi(y), phi(y) = (1 - e^-y) / y,
    y = w / tau. Both fits from bin 21 on give the same lifetime, error
    bar and background (to 1e-9 relative; measured 1e-13) and that
    amplitude ratio (to 1e-9; measured 1e-14)."""
    rng = np.random.default_rng(3)
    edges = _edges(700)
    t = 20 * W + rng.uniform(0, W, 6000) + rng.exponential(TAU, 6000)
    b = rng.uniform(0, edges[-1], 800)
    counts = np.histogram(np.concatenate([t, b]), edges)[0]
    irf = np.zeros(700)
    irf[20] = 1.0
    plain = vs.fit_lifetime_histogram(counts, edges, t_start=edges[21])
    rec = vs.fit_lifetime_histogram(counts, edges, t_start=edges[21],
                                    irf=irf)
    assert rec.tau_s == pytest.approx(plain.tau_s, rel=1e-9)
    assert rec.sigma_tau_s == pytest.approx(plain.sigma_tau_s, rel=1e-9)
    assert rec.values["background_rate"] == pytest.approx(
        plain.values["background_rate"], rel=1e-9)
    y = W / rec.tau_s
    assert plain.values["amplitude"] == pytest.approx(
        rec.values["amplitude"] * (-math.expm1(-y)) / y, rel=1e-9)


def _gauss_bins(edges, t0, sig):
    cdf = np.array([0.5 * math.erfc(-(e - t0) / (sig * math.sqrt(2.0)))
                    for e in edges])
    return np.diff(cdf)


def test_exact_expected_histogram_returns_the_truth():
    """A Gaussian-shaped response, constant within each bin, and the
    expected counts built with the brute-force kernel, scaled to 1e12
    counts: the fit returns the lifetime, amplitude and background (to
    1e-7 relative; measured 2e-11, 9e-12 and 4e-10)."""
    n_bins = 600
    edges = _edges(n_bins)
    irf = _gauss_bins(edges, 1.0e-9, 40e-12)
    irf /= irf.sum()
    kernel = _brute_kernel(W, TAU, n_bins)
    a, rate = 1e12, 1e12 * 0.05 / (n_bins * W)
    mu = a * np.convolve(irf, kernel)[:n_bins] + rate * W
    fit = vs.fit_lifetime_histogram(np.round(mu), edges, irf=irf)
    assert fit.tau_s == pytest.approx(TAU, rel=1e-7)
    assert fit.values["amplitude"] == pytest.approx(a, rel=1e-7)
    assert fit.values["background_rate"] == pytest.approx(rate, rel=1e-7)


def _emg_counts(edges, t0, sig, tau, a, rate):
    """Exact expected counts for a Gaussian start time and an
    exponential decay (distribution function in the module docstring),
    plus a flat background."""
    def cdf(t):
        x = (t - t0) / sig
        phi1 = 0.5 * math.erfc(-x / math.sqrt(2.0))
        phi2 = 0.5 * math.erfc(-(x - sig / tau) / math.sqrt(2.0))
        return phi1 - math.exp(-(t - t0) / tau
                               + sig * sig / (2.0 * tau * tau)) * phi2
    c = np.array([cdf(e) for e in edges])
    return a * np.diff(c) + rate * np.diff(edges)


def test_continuous_gaussian_response():
    """A continuous Gaussian response (sigma = 40 ps, 16 ps bins) and its
    exact expected counts, scaled to 1e12: the fit with the response's
    bin probabilities as `irf` returns the lifetime within 1e-3 (the
    cost of taking the response as constant within each bin; measured
    3e-5). A plain fit of the same counts from 0 without background,
    which ignores the response, is off by more than 10 % (measured
    200 %); a plain fit of the tail from t0 + 8 sigma on agrees with the
    truth within 1e-6 (measured 6e-13), where the response has died
    away."""
    n_bins = 600
    edges = _edges(n_bins)
    t0, sig = 1.0e-9, 40e-12
    mu = _emg_counts(edges, t0, sig, TAU, 1e12, 1e12 * 0.05 / (n_bins * W))
    irf = _gauss_bins(edges, t0, sig)
    fit = vs.fit_lifetime_histogram(np.round(mu), edges, irf=irf)
    assert abs(fit.tau_s / TAU - 1.0) < 1e-3
    whole = vs.fit_lifetime_histogram(np.round(mu), edges, background=False)
    assert abs(whole.tau_s / TAU - 1.0) > 0.1
    tail = vs.fit_lifetime_histogram(np.round(mu), edges,
                                     t_start=t0 + 8 * sig)
    assert abs(tail.tau_s / TAU - 1.0) < 1e-6


def test_error_bars_match_simulated_experiments():
    """200 experiments: 4000 decays started at Gaussian times
    (sigma = 40 ps) and 400 background counts in 16 ps bins over 10 ns,
    fitted with the Gaussian response. Pull mean within 0.25 (measured
    -0.08), spread within 15 % of 1 (measured 0.96), p-values above 0.01
    in at least 95 % of runs (measured 98 %)."""
    rng = np.random.default_rng(2026)
    n_bins = 625
    edges = _edges(n_bins)
    t0, sig = 1.0e-9, 40e-12
    irf = _gauss_bins(edges, t0, sig)
    pulls, pvals = [], []
    for _ in range(200):
        n_sig = rng.poisson(4000)
        t = rng.normal(t0, sig, n_sig) + rng.exponential(TAU, n_sig)
        b = rng.uniform(0.0, edges[-1], rng.poisson(400))
        counts = np.histogram(np.concatenate([t, b]), edges)[0]
        fit = vs.fit_lifetime_histogram(counts, edges, irf=irf)
        pulls.append((fit.tau_s - TAU) / fit.sigma_tau_s)
        pvals.append(fit.p_value)
    pulls = np.array(pulls)
    assert abs(pulls.mean()) < 0.25
    assert abs(pulls.std() - 1.0) < 0.15
    assert np.mean(np.array(pvals) > 0.01) >= 0.95


def test_leftover_decay_from_earlier_pulses():
    """Pulses every T = 5 ns (about 8 lifetimes of 0.62 ns), each giving a
    decay of amplitude A; the histogram covers one period, with the
    leftover of 60 earlier pulses added term by term, scaled to 1e12
    counts. The plain fit returns the lifetime (to 1e-8; measured 9e-12)
    and the amplitude A / (1 - e^(-T / tau)) (to 1e-8; measured
    2e-14)."""
    tau, period, a = 0.62e-9, 5e-9, 1e12
    edges = np.linspace(0.0, period, 313)
    mu = np.zeros(edges.size - 1)
    for m in range(61):
        shift = m * period
        mu += a * (np.exp(-(edges[:-1] + shift) / tau)
                   - np.exp(-(edges[1:] + shift) / tau))
    fit = vs.fit_lifetime_histogram(np.round(mu), edges, background=False)
    assert fit.tau_s == pytest.approx(tau, rel=1e-8)
    assert fit.values["amplitude"] == pytest.approx(
        a / (-math.expm1(-period / tau)), rel=1e-8)


def test_irf_refusals():
    edges = _edges(100)
    counts = np.full(100, 20)
    good = _gauss_bins(edges, 0.4e-9, 40e-12)
    for bad in (good[:-1], -good, np.zeros(100), np.full(100, np.nan),
                "abc"):
        with pytest.raises(ValueError, match="irf"):
            vs.fit_lifetime_histogram(counts, edges, irf=bad)
    uneven = edges.copy()
    uneven[50] += 0.3 * W
    with pytest.raises(ValueError, match="same width"):
        vs.fit_lifetime_histogram(counts, uneven, irf=good)
    early = np.zeros(100)
    early[0] = 5
    early[30:] = 50
    irf = np.zeros(100)
    irf[25:30] = 1.0
    with pytest.raises(ValueError, match="before the instrument response"):
        vs.fit_lifetime_histogram(early, edges, irf=irf, background=False)
