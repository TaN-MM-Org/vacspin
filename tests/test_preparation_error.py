"""0.10.0: failed spin preparations in the readout calibration.

`fit_readout_histograms(..., vary=(..., "prep_error_bright",
"prep_error_dark"))` fits the share of bright shots in which the spin
started dark and of dark shots in which it started bright. Each check
compares with something computed independently of the fit:

* the closed-form maximum-likelihood estimates and error bars of the
  well-separated limit, where the preparation errors are binomial
  shares (derived in the test's docstring);
* the exact expected histograms, built from the count distributions of
  `vacspin.readout` (the code path `fidelity` uses, not the fit's own
  log-space distributions);
* seeded simulations of the counting process, written from the process;
* the fidelity of the mixed count distributions, computed in the test
  threshold by threshold, against `fidelity(..., f0=fit.f0)`.
"""
import math

import numpy as np
import pytest

from vacspin import (ReadoutFit, fidelity, fit_readout_histograms,
                     propagate_uncertainty, snv_emission)
from vacspin.readout import _count_distributions

GAMMA0 = snv_emission().gamma0
S = 2.0
R = 0.5 * GAMMA0 * S / (1.0 + S)
ALL = ("eta", "lam", "noise_rate", "prep_error_bright", "prep_error_dark")


def _simulate(eta, lam, tau, noise, e_b, e_d, n_b, n_d, rng):
    """Shots of the counting process. A bright spin gives Poisson signal
    at rate eta R until an Exp(Gp) spin flip or the end of the window; a
    dark spin gives none. Both get Poisson background. A share e_b of the
    bright-prepared shots holds a dark spin, and a share e_d of the
    dark-prepared shots a bright one."""
    gp = R / (1.0 + lam)

    def bright(n):
        t_flip = rng.exponential(1.0 / gp, n)
        return (rng.poisson(eta * R * np.minimum(t_flip, tau))
                + rng.poisson(noise * tau, n))

    def dark(n):
        return rng.poisson(noise * tau, n)

    kb = np.where(rng.random(n_b) < e_b, dark(n_b), bright(n_b))
    kd = np.where(rng.random(n_d) < e_d, bright(n_d), dark(n_d))
    return np.bincount(kb), np.bincount(kd)


def test_separated_limit_matches_closed_form_mle():
    """Bright spin: Poisson counts of mean m = 30 (no flips, no
    background), so it gives 0 counts with probability e^-30 = 1e-13;
    dark spin: always 0 counts. Then a bright-prepared shot with 0
    counts is a failed preparation, and a dark-prepared shot with counts
    is one. With z_b zero-count bright shots and c_d dark shots with
    counts, out of N each, the likelihood factorizes (up to e^-30) into
    two binomials and one Poisson mean, so

        e_b = z_b / N,  e_d = c_d / N,
        m = (all counts) / (N - z_b + c_d),  eta = m / (R tau),

    with error bars sqrt(e (1 - e) / N) for each share and
    sqrt(m / (N - z_b + c_d)) / (R tau) for eta, and no correlation
    between them (the cross terms of the information are of order
    e^-30). The fit agrees to 1e-7 (values) and 1e-6 (error bars), and
    the correlations are below 1e-6."""
    rng = np.random.default_rng(12)
    tau, n, e_b, e_d = 2e-6, 4000, 0.1, 0.05
    m_true = 30.0
    kb = np.where(rng.random(n) < e_b, 0, rng.poisson(m_true, n))
    kd = np.where(rng.random(n) < e_d, rng.poisson(m_true, n), 0)
    fit = fit_readout_histograms(
        np.bincount(kb), GAMMA0, tau, S, dark_hist=np.bincount(kd),
        lam=1e15, vary=("eta", "prep_error_bright", "prep_error_dark"))
    z_b, c_d = int(np.sum(kb == 0)), int(np.sum(kd > 0))
    n_bright = n - z_b + c_d
    m = (kb.sum() + kd.sum()) / n_bright
    want = {"eta": m / (R * tau), "prep_error_bright": z_b / n,
            "prep_error_dark": c_d / n}
    want_sig = {
        "eta": math.sqrt(m / n_bright) / (R * tau),
        "prep_error_bright": math.sqrt(z_b / n * (1 - z_b / n) / n),
        "prep_error_dark": math.sqrt(c_d / n * (1 - c_d / n) / n)}
    for k in want:
        assert fit.values[k] == pytest.approx(want[k], rel=1e-7)
        assert fit.sigma[k] == pytest.approx(want_sig[k], rel=1e-6)
    corr = fit.cov / np.sqrt(np.outer(np.diag(fit.cov), np.diag(fit.cov)))
    assert np.max(np.abs(corr - np.eye(3))) < 1e-6
    # f0 = 1 - e_b - e_d
    assert fit.f0 == pytest.approx(1.0 - (z_b + c_d) / n, rel=1e-7)


@pytest.mark.parametrize("eta,lam,noise,leak,e_b,e_d", [
    (0.05, 200.0, 2e5, 0.0, 0.10, 0.05),     # overlapping histograms
    (0.20, 1000.0, 5e4, 1e-4, 0.03, 0.08),   # with a leaking dark spin
])
def test_exact_expected_histograms_return_the_truth(eta, lam, noise, leak,
                                                    e_b, e_d):
    """The exact expected histograms of 1e12 shots per state, mixed by
    the preparation errors and rounded, built from the distributions of
    `vacspin.readout` (the path `fidelity` uses), fitted for all five
    parameters from the automatic start: every parameter comes back to
    1e-6."""
    tau = (1.0 + lam) / R                       # Gp tau = 1
    pb, pd = _count_distributions(eta, lam, GAMMA0, tau, S, leak, noise,
                                  400)
    hb = np.round(1e12 * ((1.0 - e_b) * pb + e_b * pd))
    hd = np.round(1e12 * ((1.0 - e_d) * pd + e_d * pb))
    hb = hb[:np.max(np.nonzero(hb)) + 1]
    hd = hd[:np.max(np.nonzero(hd)) + 1]
    fit = fit_readout_histograms(hb, GAMMA0, tau, S, dark_hist=hd,
                                 leak=leak, vary=ALL)
    truth = dict(eta=eta, lam=lam, noise_rate=noise,
                 prep_error_bright=e_b, prep_error_dark=e_d)
    for k in ALL:
        assert fit.values[k] == pytest.approx(truth[k], rel=1e-6)


def test_error_bars_match_simulated_experiments():
    """150 simulated experiments of 3000 bright and 3000 dark shots, with
    overlapping histograms (mean bright count about 6, background 0.5,
    10 % and 5 % failed preparations), all five parameters fitted from
    the automatic start: (fit - truth) / sigma has a mean within 0.25 of
    0 and a spread within 15 % of 1 for every parameter, and p > 0.01
    in at least 95 % of the experiments."""
    rng = np.random.default_rng(2)
    eta, lam, noise, e_b, e_d = 0.05, 200.0, 2e5, 0.10, 0.05
    tau = (1.0 + lam) / R
    truth = dict(eta=eta, lam=lam, noise_rate=noise,
                 prep_error_bright=e_b, prep_error_dark=e_d)
    pulls, p_ok = [], 0
    for _ in range(150):
        hb, hd = _simulate(eta, lam, tau, noise, e_b, e_d, 3000, 3000, rng)
        fit = fit_readout_histograms(hb, GAMMA0, tau, S, dark_hist=hd,
                                     vary=ALL)
        pulls.append([(fit.values[k] - truth[k]) / fit.sigma[k]
                      for k in ALL])
        p_ok += fit.p_value > 0.01
    pulls = np.array(pulls)
    assert np.all(np.abs(pulls.mean(axis=0)) < 0.25)
    assert np.all(np.abs(pulls.std(axis=0, ddof=1) - 1.0) < 0.15)
    assert p_ok >= 0.95 * 150


def test_failed_preparations_are_recovered():
    """5000 bright shots of which 20 % hold a dark spin (background
    counts only), and 5000 dark shots: without a preparation error the
    model fails its goodness-of-fit test (p < 1e-6, as in
    test_goodness_of_fit_flags_a_wrong_model); with it, the fit passes
    (p > 0.01) and finds the 20 % within 3 error bars."""
    rng = np.random.default_rng(5)
    tau, noise = 1e-6, 5e4
    eta = 5.9 / (R * tau)
    hb, hd = _simulate(eta, 1e15, tau, noise, 0.2, 0.0, 5000, 5000, rng)
    kw = dict(gamma=GAMMA0, tau=tau, s=S, dark_hist=hd, lam=1e15)
    plain = fit_readout_histograms(hb, vary=("eta", "noise_rate"), **kw)
    assert plain.p_value < 1e-6
    fit = fit_readout_histograms(
        hb, vary=("eta", "noise_rate", "prep_error_bright"), **kw)
    assert fit.p_value > 0.01
    assert abs(fit.values["prep_error_bright"] - 0.2) \
        < 3.0 * fit.sigma["prep_error_bright"]
    assert abs(fit.values["eta"] - eta) < 3.0 * fit.sigma["eta"]


def test_f0_turns_the_readout_fidelity_into_the_measured_one():
    """With a share e_b of bright shots that started dark and e_d of dark
    shots that started bright, the fidelity measured at threshold N is
    1 - (P_bright_shots(K < N) + P_dark_shots(K >= N)) / 2, with the
    mixed distributions (1 - e_b) P_b + e_b P_d and (1 - e_d) P_d +
    e_d P_b. Computed here threshold by threshold and maximized, it
    equals `fidelity(..., f0=1 - e_b - e_d)` (value to 1e-12, and the
    same best threshold), in three cases with flips, background and
    leak; `ReadoutFit.f0` is that 1 - e_b - e_d."""
    cases = [(0.05, 200.0, 2.7e-6, 2e5, 0.0, 0.10, 0.05),
             (0.3, 500.0, 2e-6, 5e4, 1e-3, 0.02, 0.07),
             (0.002, 2244.0, 50e-6, 4e3, 0.0, 0.0, 0.12)]
    for eta, lam, tau, noise, leak, e_b, e_d in cases:
        out = fidelity(eta, lam, GAMMA0, tau, s=S, leak=leak,
                       noise_rate=noise, f0=1.0 - e_b - e_d)
        pb, pd = _count_distributions(eta, lam, GAMMA0, tau, S, leak,
                                      noise, 2000)
        mb = (1.0 - e_b) * pb + e_b * pd
        md = (1.0 - e_d) * pd + e_d * pb
        f_n = [1.0 - 0.5 * (mb[:n].sum() + (1.0 - md[:n].sum()))
               for n in range(1, 1500)]
        assert out["fidelity"] == pytest.approx(max(f_n), abs=1e-12)
        assert out["threshold"] == 1 + int(np.argmax(f_n))
    fit = ReadoutFit(values={"eta": 0.1, "lam": 50.0, "noise_rate": 0.0,
                             "prep_error_bright": 0.03,
                             "prep_error_dark": 0.01},
                     sigma={}, cov=np.zeros((0, 0)), vary=(),
                     log_likelihood=0.0, chi2=None, chi2_dof=None,
                     p_value=None, n_bright_shots=10, n_dark_shots=10,
                     condition_number=1.0, n_iter=0)
    assert fit.f0 == pytest.approx(0.96, abs=1e-15)
    fit.values = {"eta": 0.1, "lam": 50.0, "noise_rate": 0.0}
    assert fit.f0 == 1.0


def test_error_bar_of_f0_is_exact():
    """f0 = 1 - e_b - e_d is linear in the fitted shares, so its
    propagated error bar is exactly sqrt(s_b^2 + s_d^2 + 2 cov_bd) from
    the fit's covariance (to 1e-8 relative)."""
    rng = np.random.default_rng(3)
    eta, lam, noise = 0.05, 200.0, 2e5
    tau = (1.0 + lam) / R
    hb, hd = _simulate(eta, lam, tau, noise, 0.1, 0.05, 3000, 3000, rng)
    fit = fit_readout_histograms(hb, GAMMA0, tau, S, dark_hist=hd, vary=ALL)
    q = propagate_uncertainty(
        fit, lambda v: 1.0 - v["prep_error_bright"] - v["prep_error_dark"])
    i, j = ALL.index("prep_error_bright"), ALL.index("prep_error_dark")
    c = fit.cov
    want = math.sqrt(c[i, i] + c[j, j] + 2.0 * c[i, j])
    assert q["value"] == pytest.approx(fit.f0, rel=1e-14)
    assert q["sigma"] == pytest.approx(want, rel=1e-8)


def test_without_preparation_errors_nothing_changes():
    """Left out, the preparation errors are not modeled and `values` has
    the same three keys as before 0.10.0; held at 0 explicitly they give
    the same fit (the same computation), and the keys appear."""
    rng = np.random.default_rng(4)
    lam = 200.0
    tau = (1.0 + lam) / R
    hb, hd = _simulate(0.05, lam, tau, 2e5, 0.0, 0.0, 2000, 2000, rng)
    kw = dict(dark_hist=hd, vary=("eta", "lam", "noise_rate"))
    old = fit_readout_histograms(hb, GAMMA0, tau, S, **kw)
    zero = fit_readout_histograms(hb, GAMMA0, tau, S, prep_error_bright=0.0,
                                  prep_error_dark=0.0, **kw)
    assert set(old.values) == {"eta", "lam", "noise_rate"}
    assert old.f0 == 1.0 and zero.f0 == 1.0
    for k in ("eta", "lam", "noise_rate"):
        assert zero.values[k] == old.values[k]
        assert zero.sigma[k] == old.sigma[k]
    assert zero.values["prep_error_bright"] == 0.0
    assert zero.values["prep_error_dark"] == 0.0


def test_preparation_error_refusals():
    rng = np.random.default_rng(6)
    lam = 200.0
    tau = (1.0 + lam) / R
    hb, hd = _simulate(0.05, lam, tau, 2e5, 0.1, 0.05, 2000, 2000, rng)
    kw = dict(gamma=GAMMA0, tau=tau, s=S)
    with pytest.raises(ValueError, match="needs dark_hist"):
        fit_readout_histograms(hb, vary=("eta", "prep_error_dark"),
                               lam=lam, **kw)
    for bad in (-0.1, 0.5, 0.7, float("nan"), "0.1", True):
        with pytest.raises(ValueError, match=r"\[0, 0.5\)"):
            fit_readout_histograms(hb, dark_hist=hd, lam=lam,
                                   prep_error_bright=bad, **kw)
    with pytest.raises(ValueError, match="start above 0"):
        fit_readout_histograms(hb, dark_hist=hd, lam=lam,
                               vary=("eta", "prep_error_bright"),
                               prep_error_bright=0.0, **kw)
    # a sample without failed preparations: the share is not determined
    # away from its boundary at 0, and the fit says so
    hb0, hd0 = _simulate(0.05, lam, tau, 2e5, 0.0, 0.0, 2000, 2000, rng)
    with pytest.raises(ValueError, match="poorly determined: prep_error"):
        fit_readout_histograms(hb0, dark_hist=hd0, vary=ALL, **kw)
