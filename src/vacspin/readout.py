"""Single-shot spin readout: exact counting statistics, honest limits.

Implements the two-level readout model of Rosenthal et al.,
arXiv:2403.13110, Appendix A (Eqs. A6-A20), generalized to any
cavity-coupled centre this package can describe.

The physical picture: drive the spin-conserving (bright) transition at
saturation parameter s; while the spin is bright, detected photons
arrive as a Poisson process of rate eta R with
R = (gamma/2) s / (1 + s); each emission flips the spin with
probability 1/(Lambda + 1), so the bright record terminates at the
polarisation rate Gp = R / (Lambda + 1). The dark spin contributes
background counts, direct off-resonant scattering (`leak` = the
off-resonant to resonant scattering ratio), and -- the subtle term --
SWITCH-ON: an off-resonant excitation can decay spin-flippingly and
turn the dark spin bright for the rest of the window.

Two independent statistical treatments are implemented and the tests
require them to agree in their shared regimes:

* `fidelity_threshold1`: the closed form at threshold N_r = 1,
  F = 1/2 + (f0/2) (exp(-n_d) - exp(-n_b))                (Eq. A20).
* `fidelity`: the exact bright distribution (a Poisson process
  terminated by an exponential flip time, integrated in closed
  quadrature), the full dark model, and an optimal integer threshold.

Exact limits asserted in the tests rather than stated: with no spin
flips (Lambda -> inf) the bright count is exactly Poisson; deep in the
flip-terminated regime (Gp tau >> 1) it is exactly geometric with mean
eta Lambda; the general optimizer restricted to N_r = 1 reproduces the
closed form; fidelity is bounded by [1/2, 1] (at f0 = 1), increases
with eta, and the shipped SnV- numbers reproduce the measured confocal
operating point of arXiv:2403.13110 (n_b ~ 4 detected photons at
eta = 0.2%, Lambda = 2244, tau = 50 us).

Inverse tools answer the questions an experiment actually asks --
`required_efficiency` and `required_window` -- and refuse when no
value can reach the target, instead of returning the boundary.
"""
from __future__ import annotations

import numpy as np

__all__ = ["polarization_rate", "readout_counts", "fidelity_threshold1",
           "geometric_pmf", "poisson_pmf", "fidelity",
           "required_efficiency", "required_window"]


def _check_common(eta, lam, gamma, tau, s):
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must be in [0, 1]")
    if lam <= 0 or not np.isfinite(lam):
        raise ValueError("cyclicity Lambda must be finite and positive")
    if gamma <= 0 or tau <= 0 or s <= 0:
        raise ValueError("gamma, tau and s must be positive")


def polarization_rate(gamma, lam, s=1.0, delta_over_gamma=0.0):
    """Spin polarisation (bright-record termination) rate
    Gp = R / (1 + Lambda), R = (gamma/2) s / (1 + s + (2 delta/gamma)^2)
    (Rosenthal Eq. A8)."""
    r = 0.5 * gamma * s / (1.0 + s + (2.0 * delta_over_gamma) ** 2)
    return r / (1.0 + lam)


def readout_counts(eta, lam, gamma, tau, s=1.0, leak=0.0,
                   noise_rate=0.0):
    """Mean detected counts (n_bright, n_dark) in a window tau:
    n_b = n_d + eta (Lambda + 1)(1 - exp(-Gp tau)) (Eqs. A13-A15),
    n_d = (eta leak R + noise_rate) tau."""
    _check_common(eta, lam, gamma, tau, s)
    r = 0.5 * gamma * s / (1.0 + s)
    gp = r / (1.0 + lam)
    n_d = (eta * leak * r + noise_rate) * tau
    n_b = n_d + eta * (lam + 1.0) * (1.0 - np.exp(-gp * tau))
    return float(n_b), float(n_d)


def fidelity_threshold1(n_bright, n_dark, f0=1.0):
    """Single-shot fidelity at threshold N_r = 1 (Eq. A20):
    F = 1/2 + (f0/2)(exp(-n_d) - exp(-n_b))."""
    if n_bright < 0 or n_dark < 0:
        raise ValueError("mean counts must be >= 0")
    if not (0.0 < f0 <= 1.0):
        raise ValueError("f0 must be in (0, 1]")
    return float(0.5 + 0.5 * f0 * (np.exp(-n_dark) - np.exp(-n_bright)))


def geometric_pmf(mean, kmax):
    """Flip-terminated detected-count distribution: each emission is
    detected (probability p = mean/(mean+1)) or the spin flips, so
    P(K = k) = (1 - p) p^k with mean eta Lambda (the per-emission
    bookkeeping). The continuous Poisson-process model's exact
    infinite-window limit is the same law with mean eta (Lambda + 1);
    the two agree at order 1/Lambda, and the tests pin both
    statements."""
    if mean < 0:
        raise ValueError("mean must be >= 0")
    p = mean / (mean + 1.0)
    ks = np.arange(int(kmax) + 1)
    return (1.0 - p) * p ** ks


def poisson_pmf(mean, kmax):
    """Poisson pmf up to kmax, computed in log space."""
    if mean < 0:
        raise ValueError("mean must be >= 0")
    ks = np.arange(int(kmax) + 1)
    lgf = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, kmax + 1)))))
    if mean == 0.0:
        pmf = np.zeros(kmax + 1)
        pmf[0] = 1.0
        return pmf
    return np.exp(ks * np.log(mean) - mean - lgf)


def _bright_pmf(eta, r, gp, tau, kmax):
    """Exact bright-count pmf, in closed form: photons arrive as
    Poisson(eta r t) while the spin is bright; the record ends at the
    flip time T ~ Exp(gp) truncated at tau, so

    P(K = k) = int_0^tau gp e^{-gp t} Po(k; eta r t) dt
               + e^{-gp tau} Po(k; eta r tau).

    The integral is an incomplete-gamma expression evaluated by the
    exact recurrence (a = gp + eta r, q = eta r / a,
    B_k = e^{-gp tau} Po(k; eta r tau)):

        p_0 = (gp / a)(1 - e^{-a tau}),
        p_k = q p_{k-1} - (gp / a) B_k,
        pmf_k = p_k + B_k,

    which follows from int_0^tau t^k e^{-a t} dt =
    (k I_{k-1} - tau^k e^{-a tau}) / a. No numerical quadrature: the
    Poisson and geometric limits hold to machine precision, and the
    tests assert both."""
    a = gp + eta * r
    q = eta * r / a
    b = poisson_pmf(eta * r * tau, kmax) * np.exp(-gp * tau)
    pmf = np.empty(kmax + 1)
    p_prev = (gp / a) * (1.0 - np.exp(-a * tau))
    pmf[0] = p_prev + b[0]
    for k in range(1, kmax + 1):
        p_prev = q * p_prev - (gp / a) * b[k]
        if p_prev < 0.0:                          # roundoff floor
            p_prev = 0.0
        pmf[k] = p_prev + b[k]
    return pmf


def fidelity(eta, lam, gamma, tau, s=1.0, leak=0.0, noise_rate=0.0,
             f0=1.0, nt_switch=25):
    """Single-shot readout fidelity with the optimal integer threshold.

    eta : total detection efficiency per emission event (from
        `CavityInterface.eta`, or measured).
    lam : cyclicity at the operating point (cavity-boosted:
        `CavityInterface.lambda_cav`).
    gamma : total optical decay rate 1/s (`CavityInterface.gamma_cav`).
    tau : readout window (s). s : saturation parameter.
    leak : off-resonant/resonant scattering ratio of the dark spin
        (power broadening included by the caller); 0 disables the
        dark-scattering and switch-on channels.
    noise_rate : background count rate (1/s). f0 : preparation/other
        infidelity prefactor.

    Returns dict(fidelity, threshold, n_bright, n_dark).
    """
    _check_common(eta, lam, gamma, tau, s)
    if leak < 0 or noise_rate < 0:
        raise ValueError("leak and noise_rate must be >= 0")
    if not (0.0 < f0 <= 1.0):
        raise ValueError("f0 must be in (0, 1]")
    r = 0.5 * gamma * s / (1.0 + s)
    gp = r / (1.0 + lam)
    m_b = eta * (lam + 1.0) * (1.0 - np.exp(-gp * tau))
    m_win = eta * r * tau
    m = min(m_b, m_win)
    kmax = int(min(m + 8.0 * np.sqrt(m + 1.0) + 60, 8000))
    pmf_b = _bright_pmf(eta, r, gp, tau, kmax)
    p_ge_b = np.empty(kmax + 1)
    p_ge_b[0] = 1.0
    p_ge_b[1:] = 1.0 - np.cumsum(pmf_b)[:-1]
    n_d = (eta * leak * r + noise_rate) * tau
    cdf_d = np.cumsum(poisson_pmf(n_d, kmax))
    # dark switch-on: off-resonant excitation flips the dark spin into
    # the bright manifold at rate R_sw = leak * R * Lambda/(Lambda+1);
    # a switch at time t contributes a bright record of length tau - t.
    p_sw_ge = np.zeros(kmax + 1)
    r_sw = leak * r * lam / (lam + 1.0)
    if r_sw > 0.0:
        ts = np.linspace(0.0, tau, int(nt_switch))[1:]
        wts = np.gradient(ts) * r_sw * np.exp(-r_sw * ts)
        for t, w in zip(ts, wts):
            pmf_t = _bright_pmf(eta, r, gp, max(tau - t, 1e-15), kmax)
            ge = np.empty(kmax + 1)
            ge[0] = 1.0
            ge[1:] = 1.0 - np.cumsum(pmf_t)[:-1]
            p_sw_ge += w * ge
    nrs = np.arange(1, kmax + 1)
    err_dark = np.minimum(1.0, (1.0 - cdf_d[nrs - 1]) + p_sw_ge[nrs])
    err_bright = 1.0 - p_ge_b[nrs]
    frs = 1.0 - 0.5 * (err_dark + err_bright)
    i = int(np.argmax(frs))
    return dict(fidelity=float(f0 * frs[i] + (1.0 - f0) * 0.5),
                threshold=int(nrs[i]), n_bright=float(m_b),
                n_dark=float(n_d))


def _bisect_increasing(fun, lo, hi, target, tol=1e-6, itmax=80):
    flo, fhi = fun(lo), fun(hi)
    if fhi < target:
        return None
    if flo >= target:
        return lo
    for _ in range(itmax):
        mid = 0.5 * (lo + hi)
        if fun(mid) >= target:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol * max(hi, 1e-12):
            break
    return hi


def required_efficiency(target_fidelity, lam, gamma, tau, s=1.0,
                        leak=0.0, noise_rate=0.0, f0=1.0):
    """The smallest detection efficiency eta reaching the target
    single-shot fidelity, by bisection of the exact model. Refuses --
    with the achievable maximum named -- when even eta = 1 falls
    short."""
    if not (0.5 < target_fidelity < 1.0):
        raise ValueError("target_fidelity must be in (0.5, 1)")

    def f(eta):
        return fidelity(eta, lam, gamma, tau, s, leak, noise_rate,
                        f0)["fidelity"]

    best = f(1.0)
    if best < target_fidelity:
        raise ValueError(
            f"target fidelity {target_fidelity} is unreachable at this "
            f"operating point: even eta = 1 gives {best:.4f}. Increase "
            "the cyclicity, the window, or reduce the dark counts.")
    return float(_bisect_increasing(f, 1e-6, 1.0, target_fidelity))


def required_window(target_fidelity, eta, lam, gamma, s=1.0, leak=0.0,
                    noise_rate=0.0, f0=1.0, tau_max=1e-2):
    """The shortest readout window reaching the target fidelity.
    Refuses when no window up to tau_max does (with dark counts the
    fidelity is not monotone in tau, so the search is a scan followed
    by bisection on the rising flank)."""
    if not (0.5 < target_fidelity < 1.0):
        raise ValueError("target_fidelity must be in (0.5, 1)")
    taus = np.geomspace(1e-9, tau_max, 60)
    vals = [fidelity(eta, lam, gamma, t, s, leak, noise_rate,
                     f0)["fidelity"] for t in taus]
    above = [i for i, v in enumerate(vals) if v >= target_fidelity]
    if not above:
        raise ValueError(
            f"target fidelity {target_fidelity} is unreachable for any "
            f"window up to {tau_max} s (best {max(vals):.4f}); improve "
            "eta or the cyclicity.")
    i = above[0]
    if i == 0:
        return float(taus[0])
    lo, hi = taus[i - 1], taus[i]

    def f(t):
        return fidelity(eta, lam, gamma, t, s, leak, noise_rate,
                        f0)["fidelity"]

    return float(_bisect_increasing(f, lo, hi, target_fidelity))
