"""The fidelity at the threshold an experiment uses, the two error
probabilities (both since 0.8.0), and the faster bright-count loop.

Checks against closed forms derived here: with no spin flips both
states are Poisson, so the two errors are a Poisson distribution
function and its complement (summed from math.lgamma, independently of
the package); when every bright record ends in a flip the bright count
is geometric, P(K < N) = 1 - p^N with p = m / (m + 1) and
m = eta (Lambda + 1); at threshold 1 the bright error is the
probability of no count at all, e^(-N tau) P0 with
P0 = gp / a (1 - e^(-a tau)) + e^(-a tau), a = gp + eta R (from
integrating over the exponential flip time)."""
import math

import numpy as np
import pytest

import vacspin as vs
from vacspin.readout import _bright_pmf, poisson_pmf

GAMMA = vs.snv_emission().gamma0


def _bright_pmf_0_7_0(eta, r, gp, tau, kmax):
    """The 0.7.0 loop (NumPy scalars), kept as the reference."""
    a = gp + eta * r
    q = eta * r / a
    b = poisson_pmf(eta * r * tau, kmax) * np.exp(-gp * tau)
    pmf = np.empty(kmax + 1)
    p_prev = (gp / a) * (1.0 - np.exp(-a * tau))
    pmf[0] = p_prev + b[0]
    for k in range(1, kmax + 1):
        p_prev = q * p_prev - (gp / a) * b[k]
        if p_prev < 0.0:
            p_prev = 0.0
        pmf[k] = p_prev + b[k]
    return pmf


def test_bright_pmf_equals_the_0_7_0_loop():
    """Same operations in the same order on Python floats; compared
    with a tolerance of 1e-12 of the largest value all the same."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        eta = 10 ** rng.uniform(-4, 0)
        r = 10 ** rng.uniform(5, 9)
        gp = r / (1 + 10 ** rng.uniform(0, 6))
        tau = 10 ** rng.uniform(-9, -4)
        m = eta * r * tau
        kmax = int(min(m + 10 * math.sqrt(m + 1) + 60, 6000))
        new = _bright_pmf(eta, r, gp, tau, kmax)
        old = _bright_pmf_0_7_0(eta, r, gp, tau, kmax)
        assert new.shape == old.shape
        assert np.max(np.abs(new - old)) <= 1e-12 * np.max(old)


def _poisson_cdf(m, n):
    """P(K <= n) for K ~ Poisson(m), from lgamma, summed exactly."""
    return math.fsum(math.exp(k * math.log(m) - m - math.lgamma(k + 1.0))
                     for k in range(n + 1))


@pytest.mark.parametrize("eta, tau, noise", [(0.05, 2e-6, 2e5),
                                             (0.3, 1e-6, 1e6)])
def test_fixed_threshold_poisson_closed_form(eta, tau, noise):
    s = 2.0
    r = 0.5 * GAMMA * s / (1 + s)
    m_b = eta * r * tau + noise * tau
    m_d = noise * tau
    for n in (1, 2, 3, 5, 8, 13):
        out = vs.fidelity(eta, 1e15, GAMMA, tau, s, noise_rate=noise,
                          threshold=n)
        assert out["threshold"] == n
        assert out["truncation_bound"] == 0.0
        assert out["error_bright"] == pytest.approx(
            _poisson_cdf(m_b, n - 1), abs=1e-9)
        assert out["error_dark"] == pytest.approx(
            1.0 - _poisson_cdf(m_d, n - 1), abs=1e-9)
        for f0 in (1.0, 0.9):
            o2 = vs.fidelity(eta, 1e15, GAMMA, tau, s, noise_rate=noise,
                             f0=f0, threshold=n)
            want = f0 * (1 - 0.5 * (o2["error_bright"] + o2["error_dark"])) \
                + 0.5 * (1 - f0)
            assert o2["fidelity"] == pytest.approx(want, abs=1e-15)


def test_fixed_threshold_geometric_closed_form():
    eta, lam, s = 0.4, 30.0, 10.0
    r = 0.5 * GAMMA * s / (1 + s)
    gp = r / (1 + lam)
    tau = 60.0 / gp                       # e^(-gp tau) = 1e-26
    m = eta * (lam + 1)
    p = m / (m + 1)
    for n in (1, 2, 4, 10, 25, 60):
        out = vs.fidelity(eta, lam, GAMMA, tau, s, threshold=n)
        assert out["error_bright"] == pytest.approx(1 - p ** n, abs=1e-10)
        assert out["error_dark"] == 0.0


def test_threshold_one_closed_form_with_flips_and_background():
    eta, lam, s, noise = 0.01, 500.0, 5.0, 3e4
    r = 0.5 * GAMMA * s / (1 + s)
    gp = r / (1 + lam)
    a = gp + eta * r
    for tau in (1e-6, 1e-5, 4e-5):
        p0 = gp / a * (1 - math.exp(-a * tau)) + math.exp(-a * tau)
        out = vs.fidelity(eta, lam, GAMMA, tau, s, noise_rate=noise,
                          threshold=1)
        assert out["error_bright"] == pytest.approx(
            math.exp(-noise * tau) * p0, rel=1e-12)
        assert out["error_dark"] == pytest.approx(
            -math.expm1(-noise * tau), rel=1e-12)


def test_best_threshold_errors_and_scan():
    """The default result is the best of all thresholds (checked by a
    scan of fixed thresholds) and carries its two errors."""
    cases = [dict(eta=0.002, lam=2244.0, gamma=GAMMA, tau=50e-6, s=10.0,
                  noise_rate=4000.0),
             dict(eta=0.002, lam=2244.0, gamma=GAMMA, tau=50e-6, s=10.0,
                  leak=0.01, noise_rate=4000.0),
             dict(eta=0.6, lam=1800.0, gamma=1.6e9, tau=25e-9, s=2.0,
                  leak=0.004, noise_rate=1e3)]
    for kw in cases:
        best = vs.fidelity(**kw)
        assert best["fidelity"] == pytest.approx(
            1 - 0.5 * (best["error_bright"] + best["error_dark"]),
            abs=1e-15)
        scan = [vs.fidelity(threshold=n, **kw)["fidelity"]
                for n in range(1, 40)]
        assert max(scan) == pytest.approx(best["fidelity"], abs=1e-12)
        assert int(np.argmax(scan)) + 1 == best["threshold"]
        fixed = vs.fidelity(threshold=best["threshold"], **kw)
        for key in ("error_bright", "error_dark"):
            assert fixed[key] == pytest.approx(best[key], abs=1e-12)
        # results of 0.7.0 are unchanged by the new keys
        for key in ("n_bright", "n_dark", "n_bright_total", "n_dark_total"):
            assert fixed[key] == best[key]


def test_threshold_refusals():
    for bad in (0, -1, 1.5, True, "2", (1 << 18) + 1, 2.0):
        with pytest.raises(ValueError, match="threshold"):
            vs.fidelity(0.1, 100.0, GAMMA, 1e-6, threshold=bad)
    # numpy integers are whole numbers too
    out = vs.fidelity(0.1, 100.0, GAMMA, 1e-6, threshold=np.int64(2))
    assert out["threshold"] == 2
