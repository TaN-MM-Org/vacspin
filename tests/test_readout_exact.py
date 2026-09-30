"""0.4.0 readout anchors: background counts in the bright state, the
exact dark switch-on distribution, and the window search.

Each check compares with something computed a different way: direct
numerical integration (Simpson's rule) of the switch-on mixture, the
closed-form mean, a Poisson closed form for the optimal threshold,
seeded Monte Carlo simulation of the photon-counting process, and the
closed-form window of a flip-free emitter."""
import math
import warnings

import numpy as np
import pytest

from vacspin import fidelity, poisson_pmf, required_window, snv_emission
from vacspin.readout import _switch_on_pmf

GAMMA0 = snv_emission().gamma0


def _simpson(fvals, h):
    return h / 3.0 * (fvals[0] + fvals[-1] + 4.0 * fvals[1:-1:2].sum()
                      + 2.0 * fvals[2:-1:2].sum())


def _switch_on_by_quadrature(c, gp, r_sw, tau, kmax, n=40001):
    """P(K = k) = P(no switch) [k = 0] + E[Po(k; c D); switched], with
    the switch time integrated numerically: switch at t (density
    r_sw e^{-r_sw t}), then a bright time min(T_flip, tau - t) whose
    two parts (flip before the end, or not) are integrated as a double
    integral in (t, u). Written from the process, not from the
    density formula the package uses."""
    t = np.linspace(0.0, tau, n)
    h = t[1] - t[0]
    out = np.zeros(kmax + 1)
    out[0] += math.exp(-r_sw * tau)
    lgf = [math.lgamma(k + 1) for k in range(kmax + 1)]

    def po(k, m):
        m = np.maximum(m, 1e-300)
        return np.exp(k * np.log(m) - m - lgf[k])

    for k in range(kmax + 1):
        # no flip before the window ends: bright for tau - t
        inner_end = math.exp(0.0) * np.exp(-gp * (tau - t)) \
            * po(k, c * (tau - t))
        # flip at u after the switch, u < tau - t
        inner_flip = np.empty(n)
        for i, ti in enumerate(t):
            rem = tau - ti
            if rem <= 0.0:
                inner_flip[i] = 0.0
                continue
            m = 2 * max(8, int(200 * rem / tau)) + 1
            u = np.linspace(0.0, rem, m)
            inner_flip[i] = _simpson(gp * np.exp(-gp * u) * po(k, c * u),
                                     u[1] - u[0])
        dens = r_sw * np.exp(-r_sw * t)
        out[k] += _simpson(dens * (inner_end + inner_flip), h)
    return out


@pytest.mark.parametrize("c,gp,r_sw,tau", [
    (2e6, 1e5, 3e5, 5e-6),    # moderate switch-on, a = gp - r_sw + c > 0
    (1e6, 1e5, 4e6, 2e-6),    # fast switch-on, a < 0 (other branch)
])
def test_switch_on_pmf_matches_direct_integration(c, gp, r_sw, tau):
    kmax = 25
    got = _switch_on_pmf(c, gp, r_sw, tau, kmax)
    want = _switch_on_by_quadrature(c, gp, r_sw, tau, kmax, n=2001)
    assert np.max(np.abs(got - want)) < 1e-6


def test_switch_on_pmf_normalised_with_closed_form_mean():
    """Sums to 1 (to 1e-12) and its mean is c E[D], with E[D] the
    integral of the survival function (1 - e^{-r_sw (tau - d)})
    e^{-gp d}, here integrated by Simpson's rule (to 1e-9 relative)."""
    for c, gp, r_sw, tau in ((2e6, 1e5, 3e5, 5e-6), (1e6, 1e5, 4e7, 2e-6),
                             (1e7, 1e6, 1e3, 1e-6), (1e5, 1e4, 1e6, 1e-4)):
        pmf = _switch_on_pmf(c, gp, r_sw, tau, 5000)
        assert abs(pmf.sum() - 1.0) < 1e-12
        d = np.linspace(0.0, tau, 200001)
        surv = (1.0 - np.exp(-r_sw * (tau - d))) * np.exp(-gp * d)
        mean_d = _simpson(surv, d[1] - d[0])
        mean_k = float(np.dot(np.arange(pmf.size), pmf))
        assert abs(mean_k - c * mean_d) < 1e-9 * max(c * mean_d, 1e-12)


def _optimal_poisson(n_b, n_d, kmax=2000):
    pb, pd = poisson_pmf(n_b, kmax), poisson_pmf(n_d, kmax)
    cb, cd = np.cumsum(pb), np.cumsum(pd)
    frs = 1.0 - 0.5 * ((1.0 - cd[:-1]) + cb[:-1])     # thresholds 1..kmax
    i = int(np.argmax(frs))
    return float(frs[i]), i + 1


@pytest.mark.parametrize("eta,tau,noise", [
    (0.05, 2e-6, 2e5),          # threshold 3, mild background
    (0.5, 4.05e-6, 4.94e7),     # 150 signal on 200 background counts
])
def test_background_reaches_the_bright_state(eta, tau, noise):
    """Without spin flips and without leak, both states are Poisson:
    the bright one with mean signal + background (the background does
    not depend on the spin), the dark one with the background alone. The optimal
    threshold and fidelity must equal that closed form (to 1e-9).
    Before 0.4.0 the bright state carried no background: the second
    case returned 0.5 instead of 0.999998."""
    s = 2.0
    out = fidelity(eta, 1e15, GAMMA0, tau, s=s, noise_rate=noise)
    r = 0.5 * GAMMA0 * s / (1.0 + s)
    f_want, thr_want = _optimal_poisson(eta * r * tau + noise * tau,
                                        noise * tau)
    assert out["threshold"] == thr_want
    assert abs(out["fidelity"] - f_want) < 1e-9
    assert out["truncation_bound"] <= 1e-9
    assert abs(out["n_bright_total"] - (eta * r * tau + noise * tau)) \
        < 1e-9 * out["n_bright_total"]


def _simulate(eta, lam, gamma, tau, s, leak, noise, n, seed):
    """Monte Carlo of the counting process the model describes."""
    rng = np.random.default_rng(seed)
    r = 0.5 * gamma * s / (1.0 + s)
    gp = r / (1.0 + lam)
    c = eta * r
    r_sw = leak * r * lam / (lam + 1.0)
    t_flip = rng.exponential(1.0 / gp, n)
    k_bright = rng.poisson(c * np.minimum(t_flip, tau)) \
        + rng.poisson(noise * tau, n)
    t_sw = rng.exponential(1.0 / r_sw, n) if r_sw > 0 else np.full(n, 2 * tau)
    t_flip2 = rng.exponential(1.0 / gp, n)
    d = np.where(t_sw < tau, np.minimum(t_flip2, tau - t_sw), 0.0)
    k_dark = rng.poisson(c * d) \
        + rng.poisson((noise + eta * leak * r) * tau, n)
    return k_bright, k_dark


@pytest.mark.parametrize("eta,lam,tau,s,leak,noise", [
    (0.3, 500.0, 2e-6, 1.0, 1e-3, 5e4),       # switch-on ~ 5 % of shots
    (0.002, 2244.0, 50e-6, 10.0, 0.01, 4e3),  # switch-on almost certain
    (0.65, 16000.0, 2e-8, 2.0, 3.7e-3, 1e3),  # the cavity design point
])
def test_fidelity_matches_monte_carlo(eta, lam, tau, s, leak, noise):
    """400 000 simulated shots per state: the empirical fidelity at the
    returned threshold, and the empirical mean dark count, agree with
    the model within 5 standard errors, and so does the empirical mean
    bright count with `n_bright_total`. The second case is where the
    old 24-point switch-on sum failed (0.3.1 gave 0.7181 instead of
    0.50: the switch happens within the first grid step)."""
    gamma = GAMMA0 * (7.26 if lam == 16000.0 else 1.0)
    out = fidelity(eta, lam, gamma, tau, s=s, leak=leak, noise_rate=noise)
    n = 400_000
    kb, kd = _simulate(eta, lam, gamma, tau, s, leak, noise, n, seed=3)
    thr = out["threshold"]
    f_mc = 1.0 - 0.5 * (np.mean(kd >= thr) + np.mean(kb < thr))
    se = math.sqrt(0.25 / n)
    assert abs(f_mc - out["fidelity"]) < 5.0 * se
    se_mean = kd.std() / math.sqrt(n)
    assert abs(kd.mean() - out["n_dark_total"]) < 5.0 * se_mean + 1e-12
    se_b = kb.std() / math.sqrt(n)
    assert abs(kb.mean() - out["n_bright_total"]) < 5.0 * se_b + 1e-12


def test_nt_switch_is_deprecated_and_ignored():
    kw = dict(eta=0.3, lam=500.0, gamma=GAMMA0, tau=2e-6, leak=1e-3)
    with pytest.warns(DeprecationWarning, match="nt_switch"):
        a = fidelity(nt_switch=9, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b = fidelity(**kw)
    assert a == b


def test_too_wide_distributions_refused():
    with pytest.raises(ValueError, match="too wide"):
        fidelity(0.5, 100.0, GAMMA0, 1e-3, noise_rate=1e9)


def test_required_window_below_first_grid_point():
    """A flip-free emitter with no background reaches fidelity
    F = 1 - exp(-c tau)/2 at threshold 1, so the shortest window has
    the closed form tau* = ln(1 / (2 (1 - F))) / c. Here tau* is below
    the 1 ns start of the search grid; before 0.4.0 the function
    returned 1 ns."""
    eta, s, gamma = 0.9, 10.0, 1.6e9
    c = eta * 0.5 * gamma * s / (1.0 + s)
    target = 0.6
    tau = required_window(target, eta=eta, lam=1e15, gamma=gamma, s=s)
    want = math.log(1.0 / (2.0 * (1.0 - target))) / c
    assert tau < 1e-9
    assert abs(tau - want) < 1e-5 * want


def test_required_window_is_the_shortest():
    """On the rising flank the returned window reaches the target and
    a window 0.1 % shorter does not."""
    kw = dict(eta=0.3, lam=2244.0, gamma=GAMMA0 * 20, s=1.0, leak=1e-7,
              noise_rate=1e3)
    tau = required_window(0.99, **kw)
    f = lambda t: fidelity(kw["eta"], kw["lam"], kw["gamma"], t, kw["s"],
                           kw["leak"], kw["noise_rate"])["fidelity"]
    assert f(tau) >= 0.99
    assert f(0.999 * tau) < 0.99
    with pytest.raises(ValueError, match="tau_max"):
        required_window(0.9, eta=0.3, lam=100.0, gamma=GAMMA0,
                        tau_max=np.inf)


def test_mean_bright_count_accurate_for_rare_flips():
    """For Gp tau << 1 the mean signal eta (Lambda + 1)(1 - e^{-Gp tau})
    tends to eta R tau (1 - Gp tau / 2). Written as 1 - exp(...) it lost
    digits (relative error 6e-5 at Lambda = 1e15 before 0.4.0); with
    expm1 it matches the series to 1e-12."""
    from vacspin import readout_counts
    eta, s, tau = 0.5, 2.0, 4.05e-6
    r = 0.5 * GAMMA0 * s / (1.0 + s)
    for lam in (1e9, 1e12, 1e15):
        x = r / (1.0 + lam) * tau
        want = eta * r * tau * (1.0 - x / 2.0)
        nb, nd = readout_counts(eta, lam, GAMMA0, tau, s=s)
        assert nd == 0.0
        assert abs(nb - want) < 1e-12 * want
        out = fidelity(eta, lam, GAMMA0, tau, s=s)
        assert abs(out["n_bright"] - want) < 1e-12 * want


def test_non_finite_background_refused():
    for kw in (dict(leak=np.nan), dict(noise_rate=np.inf)):
        with pytest.raises(ValueError, match="finite"):
            fidelity(0.1, 100.0, GAMMA0, 1e-6, **kw)


def test_leak_counts_only_in_the_dark_state():
    """`fidelity` gives the bright spin background but no leak counts;
    `readout_counts` adds the whole dark mean (leak included) to the
    bright signal. The documented difference is exactly
    eta leak R tau (to 1e-12 relative), zero without leak; the bright
    mean itself is checked against Monte Carlo above."""
    from vacspin import readout_counts
    eta, lam, tau, s, noise = 0.002, 2244.0, 50e-6, 10.0, 4e3
    r = 0.5 * GAMMA0 * s / (1.0 + s)
    for leak in (0.0, 1e-3, 0.01):
        nb, nd = readout_counts(eta, lam, GAMMA0, tau, s=s, leak=leak,
                                noise_rate=noise)
        out = fidelity(eta, lam, GAMMA0, tau, s=s, leak=leak,
                       noise_rate=noise)
        assert abs(nd - out["n_dark"]) < 1e-12 * nd
        diff = nb - out["n_bright_total"]
        assert abs(diff - eta * leak * r * tau) < 1e-12 * nb
