"""Single-shot spin readout: exact counting statistics, honest limits.

Implements the two-level readout model of Rosenthal et al.,
arXiv:2403.13110 (Phys. Rev. X 14, 041008 (2024)), Appendix A
(Eqs. A6-A20), generalized to any
cavity-coupled center this package can describe.

The physical picture: drive the spin-conserving (bright) transition at
saturation parameter s; while the spin is bright, detected photons
arrive as a Poisson process of rate eta R with
R = (gamma/2) s / (1 + s); each emission flips the spin with
probability 1/(Lambda + 1), so the bright record terminates at the
polarisation rate Gp = R / (Lambda + 1). Detector background
(`noise_rate`: detector dark counts and stray light) does not depend
on the spin, so it adds Poisson counts to BOTH spin states. The dark
spin also contributes direct off-resonant scattering (`leak` = the
off-resonant to resonant scattering ratio) and -- the subtle term --
SWITCH-ON: an off-resonant excitation can decay spin-flippingly and
turn the dark spin bright for the rest of the window. The leak is a
property of the DARK spin, so `fidelity` gives the bright spin no
leak counts (see `readout_counts` for the one place the mean-count
bookkeeping differs).

Two statistical treatments are implemented and the tests require them
to agree in their shared regimes:

* `fidelity_threshold1`: the closed form at threshold N_r = 1,
  F = 1/2 + (f0/2) (exp(-n_d) - exp(-n_b))                (Eq. A20).
* `fidelity`: the full count distributions of both spin states and an
  optimal integer threshold. The bright distribution (a Poisson
  process terminated by an exponential flip time) and, since 0.4.0,
  the dark switch-on distribution are evaluated in closed form, and
  the background is combined with each by exact convolution.

Exact limits asserted in the tests rather than stated: with no spin
flips (Lambda -> inf) the bright count is Poisson, and with background
the optimal-threshold fidelity then equals the Poisson closed form;
deep in the flip-terminated regime (Gp tau >> 1) the bright count is
geometric with mean eta (Lambda + 1); the switch-on distribution sums
to one, has the closed-form mean, and matches direct numerical
integration; fidelity is bounded by [1/2, 1] (at f0 = 1) and increases
with eta. The confocal operating point of arXiv:2403.13110 (about 4
bright counts at Lambda = 2244, tau = 50 us) is reproduced with
eta = 0.2 %; note that the source's own fitted efficiency is about
0.1 %, so this eta is chosen to match the reported counts, not taken
from the paper.

Inverse tools answer the questions an experiment actually asks --
`required_efficiency` and `required_window` -- and refuse when no
value can reach the target, instead of returning the boundary.

Since 0.11.0 `fidelity(..., repeated_flips=True)` drops the two
shortcuts of the model above (a bright spin that has turned dark stays
dark; a dark spin turns bright at most once and keeps its leak counts):
the spin is a two-state jump process with any number of changes in the
window, and `t1_s` adds spin relaxation. Its count distributions come
from the generating function of the jump process, inverted by a
discrete Fourier transform; the tests check them against exact limits,
time integration and simulation (tests/test_jump_model.py).
"""
from __future__ import annotations

import math
import numbers
import warnings
from typing import Optional

import numpy as np

__all__ = ["polarization_rate", "readout_counts", "fidelity_threshold1",
           "geometric_pmf", "poisson_pmf", "fidelity",
           "required_efficiency", "required_window", "optimal_window"]


def _check_common(eta, lam, gamma, tau, s):
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must be in [0, 1]")
    if lam <= 0 or not np.isfinite(lam):
        raise ValueError("cyclicity Lambda must be finite and positive")
    if gamma <= 0 or tau <= 0 or s <= 0:
        raise ValueError("gamma, tau and s must be positive")
    if not (np.isfinite(gamma) and np.isfinite(tau) and np.isfinite(s)):
        raise ValueError("gamma, tau and s must be finite")


def _check_background(leak, noise_rate):
    if not (np.isfinite(leak) and np.isfinite(noise_rate)):
        raise ValueError("leak and noise_rate must be finite")
    if leak < 0 or noise_rate < 0:
        raise ValueError("leak and noise_rate must be >= 0")


def polarization_rate(gamma, lam, s=1.0, delta_over_gamma=0.0):
    """Spin polarisation (bright-record termination) rate
    Gp = R / (1 + Lambda), R = (gamma/2) s / (1 + s + (2 delta/gamma)^2)
    (Rosenthal Eq. A8).

    gamma is the decay rate in 1/s, so delta_over_gamma is the angular
    detuning 2 pi Delta_Hz over gamma. Equivalently it is the detuning
    in Hz over the linewidth gamma / (2 pi) in Hz. Do not divide a
    detuning in Hz by gamma in 1/s: that is 2 pi too small."""
    # element-wise, so array inputs keep working
    for name, v in (("gamma", gamma), ("s", s)):
        v = np.asarray(v, dtype=float)
        if not (np.all(np.isfinite(v)) and np.all(v > 0)):
            raise ValueError(f"{name} must be positive and finite")
    lam_a = np.asarray(lam, dtype=float)
    if not (np.all(np.isfinite(lam_a)) and np.all(lam_a >= 0)):
        raise ValueError("lam must be finite and >= 0")
    if not np.all(np.isfinite(np.asarray(delta_over_gamma, dtype=float))):
        raise ValueError("delta_over_gamma must be finite")
    r = 0.5 * gamma * s / (1.0 + s + (2.0 * delta_over_gamma) ** 2)
    return r / (1.0 + lam)


def readout_counts(eta, lam, gamma, tau, s=1.0, leak=0.0,
                   noise_rate=0.0):
    """Mean detected counts (n_bright, n_dark) in a window tau:
    n_b = n_d + eta (Lambda + 1)(1 - exp(-Gp tau)),
    n_d = (eta leak R + noise_rate) tau
    (the mean-count bookkeeping of Rosenthal et al., arXiv:2403.13110,
    Appendix A, used with `fidelity_threshold1`).

    Note: n_b here adds the WHOLE dark mean n_d, leak included, to the
    bright signal. `fidelity` gives the bright spin only the
    background noise_rate tau, because the leak is off-resonant
    scattering of the dark spin; so with leak > 0,
    n_b = fidelity(...)["n_bright_total"] + eta leak R tau
    (tested to 1e-12). With leak = 0 the two agree."""
    _check_common(eta, lam, gamma, tau, s)
    _check_background(leak, noise_rate)
    r = 0.5 * gamma * s / (1.0 + s)
    gp = r / (1.0 + lam)
    n_d = (eta * leak * r + noise_rate) * tau
    n_b = n_d + eta * (lam + 1.0) * (-np.expm1(-gp * tau))
    return float(n_b), float(n_d)


def fidelity_threshold1(n_bright, n_dark, f0=1.0):
    """Single-shot fidelity at threshold N_r = 1 (Eq. A20):
    F = 1/2 + (f0/2)(exp(-n_d) - exp(-n_b))."""
    if not (np.isfinite(n_bright) and np.isfinite(n_dark)):
        raise ValueError("mean counts must be finite")
    if n_bright < 0 or n_dark < 0:
        raise ValueError("mean counts must be >= 0")
    if not (0.0 < f0 <= 1.0):
        raise ValueError("f0 must be in (0, 1]")
    return float(0.5 + 0.5 * f0 * (np.exp(-n_dark) - np.exp(-n_bright)))


def _check_kmax(kmax) -> int:
    """kmax as an int: a whole number >= 0 (an int, or a float with no
    fractional part). Since 0.10.0; before, a negative kmax gave an empty
    array and a fractional one was cut down without notice."""
    ok = isinstance(kmax, numbers.Real) and not isinstance(kmax, bool)
    if ok:
        k = float(kmax)
        ok = math.isfinite(k) and k >= 0.0 and k == math.floor(k)
    if not ok:
        raise ValueError("kmax must be a whole number of counts >= 0 "
                         f"(got {kmax!r})")
    return int(kmax)


def geometric_pmf(mean, kmax):
    """Flip-terminated detected-count distribution: each emission is
    detected (probability p = mean/(mean+1)) or the spin flips, so
    P(K = k) = (1 - p) p^k with mean eta Lambda (the per-emission
    bookkeeping). The continuous Poisson-process model's exact
    infinite-window limit is the same law with mean eta (Lambda + 1);
    the two agree at order 1/Lambda, and the tests pin both
    statements. Returns P(K = k) for k = 0..kmax; kmax must be a whole
    number >= 0 (since 0.10.0)."""
    if not np.isfinite(mean) or mean < 0:
        raise ValueError("mean must be finite and >= 0")
    kmax = _check_kmax(kmax)
    p = mean / (mean + 1.0)
    ks = np.arange(kmax + 1)
    return (1.0 - p) * p ** ks


def poisson_pmf(mean, kmax):
    """Poisson pmf for k = 0..kmax, computed in log space; kmax must be
    a whole number >= 0 (since 0.10.0)."""
    if not np.isfinite(mean) or mean < 0:
        raise ValueError("mean must be finite and >= 0")
    kmax = _check_kmax(kmax)
    ks = np.arange(kmax + 1)
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
    tests assert both.

    Since 0.8.0 the loop runs on Python floats instead of NumPy scalars:
    the same IEEE double operations in the same order, 2.3 to 3.2 times
    faster at 100 to 10000 counts (the tests compare it with the 0.7.0
    loop)."""
    a = gp + eta * r
    q = float(eta * r / a)
    g_a = float(gp / a)
    b = (poisson_pmf(eta * r * tau, kmax) * np.exp(-gp * tau)).tolist()
    pmf = [0.0] * (kmax + 1)
    p_prev = float((gp / a) * (1.0 - np.exp(-a * tau)))
    pmf[0] = p_prev + b[0]
    for k in range(1, kmax + 1):
        p_prev = q * p_prev - g_a * b[k]
        if p_prev < 0.0:                          # roundoff floor
            p_prev = 0.0
        pmf[k] = p_prev + b[k]
    return np.array(pmf)


_KMAX_CAP = 1 << 18          # largest count the distributions are built to
_TRUNC_TOL = 1e-9            # allowed truncation error of the fidelity


def _phi(y):
    """(1 - exp(-y)) / y for real y >= 0, accurate near 0."""
    y = float(y)
    if y < 0.5:
        term, total = 1.0, 1.0
        for n in range(1, 25):              # series sum (-y)^n / (n+1)!
            term *= -y / (n + 1)
            total += term
        return total
    return -math.expm1(-y) / y


def _log_factorials(kmax):
    return np.concatenate(
        ([0.0], np.cumsum(np.log(np.arange(1, kmax + 1, dtype=float)))))


def _exp_poisson_integrals(beta, c, tau, kmax, log_pref=0.0):
    """T_k = exp(log_pref) * int_0^tau exp(-beta d) Po(k; c d) dd for
    k = 0..kmax, with Po(k; m) the Poisson pmf and beta of either sign.

    With a = beta + c two exact representations are used, each
    evaluated in log space without cancellation:

    * a > 0: T_k = exp(log_pref) (1/a) (c/a)^k P[Po(a tau) >= k + 1]
      (the regularized lower incomplete gamma function, summed as a
      Poisson tail);
    * a <= 0: with x = -a tau >= 0, T_k = exp(log_pref) tau
      exp(-beta tau) Po(k; c tau) m_k, where
      m_k = int_0^1 exp(-x (1 - v)) v^k dv is computed by its forward
      recurrence m_k = (1 - k m_{k-1}) / x for k <= x and its backward
      recurrence m_{k-1} = (1 - x m_k) / k above (each direction is
      the stable one where it is used).

    The tests check both branches against direct numerical
    integration."""
    kmax = int(kmax)
    out = np.zeros(kmax + 1)
    if c <= 0.0:                             # no photons: only k = 0
        y = beta * tau
        if y >= 0:
            out[0] = math.exp(log_pref) * tau * _phi(y)
        else:
            out[0] = tau * math.exp(log_pref - y) * _phi(-y)
        return out
    ks = np.arange(kmax + 1, dtype=float)
    lgf = _log_factorials(kmax)
    a = beta + c
    if a > 0.0:
        x = a * tau
        if kmax < x:
            # every k is below the mean of Po(x): 1 - cdf has no
            # cancellation problem
            lp = ks * math.log(x) - x - lgf
            log_cdf = np.logaddexp.accumulate(lp)
            log_q = np.log1p(-np.minimum(np.exp(log_cdf), 1.0))
        else:
            jmax = int(kmax + 1 + 12.0 * math.sqrt(x + 1.0) + 80)
            js = np.arange(jmax + 1, dtype=float)
            lp = js * math.log(x) - x - _log_factorials(jmax)
            tail = np.logaddexp.accumulate(lp[::-1])[::-1]
            log_q = tail[1:kmax + 2]
        with np.errstate(divide="ignore"):
            logt = (log_pref - math.log(a) + ks * (math.log(c)
                                                   - math.log(a))
                    + log_q)
        return np.exp(logt)
    x = -a * tau
    m = np.empty(kmax + 1)
    if x == 0.0:
        m[:] = 1.0 / (ks + 1.0)
    else:
        n0 = min(int(math.floor(x)), kmax)
        mk = _phi(x)
        m[0] = mk
        for k in range(1, n0 + 1):
            mk = (1.0 - k * mk) / x
            m[k] = mk
        if kmax > n0:
            kk = int(max(kmax, math.ceil(x)) + math.ceil(9.0 * math.sqrt(x))
                     + 60)
            mk = 1.0 / (kk + 1.0 + x)
            for k in range(kk, n0 + 1, -1):
                mk = (1.0 - x * mk) / k        # this is m_{k-1}
                if k - 1 <= kmax:
                    m[k - 1] = mk
    ct = c * tau
    with np.errstate(divide="ignore"):
        logt = (log_pref + math.log(tau) - beta * tau
                + ks * math.log(ct) - ct - lgf
                + np.log(np.maximum(m, 0.0)))
    return np.exp(logt)


def _switch_on_pmf(c, gp, r_sw, tau, kmax):
    """Exact distribution of the counts a dark spin produces after
    switching on (k = 0..kmax).

    The switch happens at rate r_sw (time t_s ~ Exp(r_sw)); from then
    on the spin is bright, emitting detected photons at rate c until
    it flips back (rate gp) or the window ends. The bright time
    D = min(T_flip, tau - t_s) (D = 0 without a switch) has an atom
    exp(-r_sw tau) at zero and, on (0, tau), the density

        f(d) = gp e^{-gp d} + (r_sw - gp) e^{-r_sw tau} e^{(r_sw - gp) d},

    from its survival function (1 - e^{-r_sw (tau - d)}) e^{-gp d}.
    The counts are Poisson(c D), so

        P(K = k) = e^{-r_sw tau} [k = 0]
                   + gp int_0^tau e^{-gp d} Po(k; c d) dd
                   + (r_sw - gp) e^{-r_sw tau}
                     int_0^tau e^{(r_sw - gp) d} Po(k; c d) dd.
    """
    i1 = _exp_poisson_integrals(gp, c, tau, kmax)
    j2 = _exp_poisson_integrals(gp - r_sw, c, tau, kmax,
                                log_pref=-r_sw * tau)
    pmf = gp * i1 + (r_sw - gp) * j2
    pmf[0] += math.exp(-r_sw * tau)
    return np.maximum(pmf, 0.0)


def _switch_on_mean(c, gp, r_sw, tau):
    """Closed-form mean of `_switch_on_pmf`: c E[D] with
    E[D] = int_0^tau (1 - e^{-r_sw (tau - d)}) e^{-gp d} dd."""
    if r_sw <= 0.0:
        return 0.0
    i0 = _exp_poisson_integrals(gp, 0.0, tau, 0)[0]
    j0 = _exp_poisson_integrals(gp - r_sw, 0.0, tau, 0,
                                log_pref=-r_sw * tau)[0]
    return c * (i0 - j0)


def _convolve_trunc(p, q, n):
    """First n entries of the convolution of two pmfs."""
    if n <= 4096:
        return np.convolve(p[:n], q[:n])[:n]
    size = 1 << int(math.ceil(math.log2(2 * n)))
    out = np.fft.irfft(np.fft.rfft(p[:n], size)
                       * np.fft.rfft(q[:n], size), size)[:n]
    return np.maximum(out, 0.0)


def _count_distributions(eta, lam, gamma, tau, s, leak, noise_rate,
                         kmax):
    """Bright and dark total-count pmfs (k = 0..kmax) of the model."""
    r = 0.5 * gamma * s / (1.0 + s)
    gp = r / (1.0 + lam)
    c = eta * r
    n_noise = noise_rate * tau
    n_leak = eta * leak * r * tau
    noise = poisson_pmf(n_noise, kmax)
    pmf_b = _convolve_trunc(_bright_pmf(eta, r, gp, tau, kmax), noise,
                            kmax + 1)
    r_sw = leak * r * lam / (lam + 1.0)
    if r_sw > 0.0:
        pmf_d = _convolve_trunc(
            poisson_pmf(n_noise + n_leak, kmax),
            _switch_on_pmf(c, gp, r_sw, tau, kmax), kmax + 1)
    else:
        pmf_d = poisson_pmf(n_noise + n_leak, kmax)
    return pmf_b, pmf_d


# --- the spin as a two-state jump process (since 0.11.0) -------------------

_JUMP_TAIL = 1e-10           # bound on the probability beyond the count range


def _phi_c(x):
    """(1 - exp(-x)) / x for complex x with Re(x) >= 0, element-wise,
    accurate near 0 (1 at x = 0)."""
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = -np.expm1(-x[nz]) / x[nz]
    return out


def _jump_rates(eta, lam, gamma, s, leak, t1_s):
    """Detected count rates (bright, dark) and spin-change rates
    (bright to dark, dark to bright), all in 1/s, of the jump model."""
    r = 0.5 * gamma * s / (1.0 + s)
    relax = 0.0 if t1_s is None else 0.5 / t1_s
    # relaxation acts in the ground state, and the driven bright spin
    # spends the share 1 - s / (2 (1 + s)) of its time there
    to_dark = r / (1.0 + lam) + relax * (2.0 + s) / (2.0 * (1.0 + s))
    to_bright = leak * r * lam / (lam + 1.0) + relax
    return eta * r, eta * leak * r, to_dark, to_bright


def _jump_generating_functions(c_b, c_d, k_bd, k_db, tau, z):
    """G(z) = E[z^K] of the photons counted in a window tau, for a spin
    that starts bright and one that starts dark.

    The spin state (bright, dark) is a two-state Markov chain with rates
    k_bd (bright to dark) and k_db (dark to bright); while bright,
    photons are counted at rate c_b, while dark at rate c_d. With
    p = (p_bright, p_dark) and u = (z - 1) tau,

        A = [[-k_bd tau + c_b u, k_db tau], [k_bd tau, -k_db tau + c_d u]],
        G_start = (1, 1) exp(A) e_start.

    For a 2 x 2 matrix, with q = trace / 2 and d^2 = q^2 - det
    (Re d >= 0),

        exp(A) = e^q cosh(d) I + e^q (sinh(d) / d) (A - q I)
               = e^l1 [ (1 + e^-2d) / 2 I + phi(2d) (A - q I) ],

    with l1 = q + d the eigenvalue of larger real part and
    phi(x) = (1 - e^-x) / x. The column sums of the rate part of A
    vanish, so (1, 1)(A - q I) e_start = c_start u - q. det is expanded
    so that the large rate products cancel exactly, and l1 is computed
    as det / (q - d), which does not cancel either: on the unit circle
    Re(u) <= 0 and k_bd > 0, so Re(q) < 0 <= Re(d) and q - d stays away
    from 0. (For real z > 1 this is not so; the count range uses
    `_jump_log_g_real` there.)"""
    u = (z - 1.0) * tau
    ub, ud = c_b * u, c_d * u
    xb, xd = k_bd * tau, k_db * tau
    q = 0.5 * (ub - xb + ud - xd)
    half = 0.5 * ((ub - xb) - (ud - xd))
    d = np.sqrt(half * half + xb * xd + 0j)
    det = ub * ud - xb * ud - xd * ub          # = a11 a22 - a12 a21
    den = q - d
    safe = den != 0
    l1 = np.where(safe, det / np.where(safe, den, 1.0), q + d)
    e1 = np.exp(l1)
    e2d = np.exp(-2.0 * d)
    ph = _phi_c(2.0 * d)
    g_b = e1 * (0.5 * (1.0 + e2d) + ph * (ub - q))
    g_d = e1 * (0.5 * (1.0 + e2d) + ph * (ud - q))
    return g_b, g_d


def _jump_log_g_real(c_b, c_d, k_bd, k_db, tau, x):
    """log G(x) for real x > 1 (an array), for a spin that starts bright
    and one that starts dark: the same 2 x 2 exponential as in
    `_jump_generating_functions`, written so that every term is >= 0
    (A has non-negative off-diagonal entries for real x >= 1) and kept
    in log form, so that nothing overflows. With l1 = q + d and
    e = exp(-2d),

        exp(A) / e^l1 = [[(d + h) + (d - h) e, x_d (1 - e)],
                         [x_b (1 - e), (d - h) + (d + h) e]] / (2 d),

    h = (a11 - a22) / 2, where d - h (or d + h) is computed as
    x_b x_d / (d + h) (or / (d - h)) to avoid cancellation."""
    u = (x - 1.0) * tau
    a11 = c_b * u - k_bd * tau
    a22 = c_d * u - k_db * tau
    xb, xd = k_bd * tau, k_db * tau
    q = 0.5 * (a11 + a22)
    h = 0.5 * (a11 - a22)
    d = np.sqrt(h * h + xb * xd)
    two_d = 2.0 * d
    big = d + np.abs(h)
    with np.errstate(divide="ignore", invalid="ignore"):
        small = np.where(big > 0.0, xb * xd / big, 0.0)   # d - |h|
        dph = np.where(h >= 0.0, big, small)              # d + h
        dmh = np.where(h >= 0.0, small, big)              # d - h
        ph = np.where(two_d > 0.0, -np.expm1(-two_d) / two_d, 1.0)
        # the terms of each column sum, in logs (e = exp(-2d) can
        # underflow while its term still matters)
        l2d = np.log(two_d)
        flip_b = np.log(xb * ph)
        flip_d = np.log(xd * ph)
        bb = np.logaddexp(np.log(dph) - l2d, np.log(dmh) - two_d - l2d)
        dd = np.logaddexp(np.log(dmh) - l2d, np.log(dph) - two_d - l2d)
        e = np.exp(-two_d)
        near = two_d <= 1e-3          # no cancellation: the plain form
        bb = np.where(near, np.log(np.maximum(0.5 * (1.0 + e) + ph * h,
                                              1e-300)), bb)
        dd = np.where(near, np.log(np.maximum(0.5 * (1.0 + e) - ph * h,
                                              1e-300)), dd)
        log_b = q + d + np.logaddexp(bb, flip_b)
        log_d = q + d + np.logaddexp(dd, flip_d)
    return log_b, log_d


def _jump_count_range(c_b, c_d, k_bd, k_db, tau, noise_rate):
    """Smallest power of two N >= 64 with P(K >= N) below _JUMP_TAIL for
    both starts, or None above _KMAX_CAP.

    Chernoff bound on the model's own generating function: for any real
    x > 1, P(K >= N) <= G(x) x^-N, with the background's
    exp((x - 1) noise_rate tau) in G. The bound is minimized over a grid
    of x from 1 + 1e-6 to 1 + 1e6 (refined around its best point), so
    the range follows the counts the model can actually give (for
    example a bright spin that soon turns dark for good), not only the
    largest count rate. For a bright record that ends in a spin flip
    (geometric counts with mean m) it needs about 27 m counts, so a mean
    count near 10^4 reaches the limit of 2^18."""
    def worst(x):
        log_b, log_d = _jump_log_g_real(c_b, c_d, k_bd, k_db, tau, x)
        return np.maximum(log_b, log_d) + (x - 1.0) * noise_rate * tau

    t = np.geomspace(1e-6, 1e6, 241)
    w = worst(1.0 + t)
    n = 64
    while n <= _KMAX_CAP:
        b = w - n * np.log1p(t)
        j = int(np.argmin(b))
        best = float(b[j])
        if best > math.log(_JUMP_TAIL):
            # refine between the neighbors of the best grid point
            tf = np.geomspace(t[max(j - 1, 0)], t[min(j + 1, t.size - 1)],
                              101)
            best = float(np.min(worst(1.0 + tf) - n * np.log1p(tf)))
        if best <= math.log(_JUMP_TAIL):
            return n
        n *= 2
    return None


def _jump_count_distributions(eta, lam, gamma, tau, s, leak, noise_rate,
                              t1_s, n=None):
    """Bright and dark total-count pmfs of the jump model, on 0..N-1.

    The generating function is evaluated at the N-th roots of unity and
    inverted by a discrete Fourier transform. N is chosen by a Chernoff
    bound on the model's own generating function (`_jump_count_range`)
    to leave less than 1e-10 beyond it, which is also the most the
    wrap-around of the transform can add. (`n` forces N, for tests.)"""
    c_b, c_d, k_bd, k_db = _jump_rates(eta, lam, gamma, s, leak, t1_s)
    if n is None:
        n = _jump_count_range(c_b, c_d, k_bd, k_db, tau, noise_rate)
    if n is None:
        m_b, m_d = _jump_means(eta, lam, gamma, tau, s, leak, noise_rate,
                               t1_s)
        raise ValueError(
            "count distributions too wide to evaluate: more than "
            f"{_KMAX_CAP} counts would be needed (mean counts: bright "
            f"{m_b:.4g}, dark {m_d:.4g}); shorten the window or reduce "
            "the background")
    z = np.exp(2j * np.pi * np.arange(n) / n)
    g_b, g_d = _jump_generating_functions(c_b, c_d, k_bd, k_db, tau, z)
    if noise_rate > 0.0:
        bg = np.exp((z - 1.0) * noise_rate * tau)
        g_b, g_d = g_b * bg, g_d * bg
    p_b = np.clip(np.real(np.fft.fft(g_b)) / n, 0.0, None)
    p_d = np.clip(np.real(np.fft.fft(g_d)) / n, 0.0, None)
    return p_b, p_d


def _psi(x):
    """(x - 1 + exp(-x)) / x^2 for real x >= 0, accurate near 0
    (1/2 at x = 0)."""
    x = float(x)
    if x < 1e-3:
        return 0.5 - x / 6.0 + x * x / 24.0
    return (x + math.expm1(-x)) / (x * x)


def _jump_means(eta, lam, gamma, tau, s, leak, noise_rate, t1_s):
    """Exact mean counts (bright start, dark start) of the jump model.

    With S = k_bd + k_db, the chance of being bright at time t is
    p(t) = p(0) e^{-S t} + (k_db / S)(1 - e^{-S t}), whose integral over
    the window is p(0) tau phi(S tau) + k_db tau^2 psi(S tau), with
    phi(x) = (1 - e^-x) / x and psi(x) = (x - 1 + e^-x) / x^2. The mean
    count is c_d tau + (c_b - c_d) times that integral, plus the
    background."""
    c_b, c_d, k_bd, k_db = _jump_rates(eta, lam, gamma, s, leak, t1_s)
    x = (k_bd + k_db) * tau
    stay = tau * _phi(x)
    arrive = k_db * tau * tau * _psi(x)
    out = []
    for p0 in (1.0, 0.0):
        t_bright = p0 * stay + arrive
        out.append(c_d * tau + (c_b - c_d) * t_bright + noise_rate * tau)
    return out[0], out[1]


def _check_jump_inputs(repeated_flips, t1_s):
    """repeated_flips must be True or False; t1_s None, or a positive
    finite time in seconds that needs repeated_flips=True."""
    if not isinstance(repeated_flips, (bool, np.bool_)):
        raise ValueError("repeated_flips must be True or False")
    if t1_s is None:
        return None
    ok = isinstance(t1_s, numbers.Real) and not isinstance(t1_s, bool)
    if ok:
        t1 = float(t1_s)
        ok = math.isfinite(t1) and t1 > 0.0
    if not ok:
        raise ValueError("t1_s must be None or a positive, finite spin "
                         f"relaxation time in seconds (got {t1_s!r})")
    if not repeated_flips:
        raise ValueError("t1_s needs repeated_flips=True: spin relaxation "
                         "can turn the spin back and forth, which only the "
                         "full jump model describes")
    return t1


def _check_threshold(threshold):
    """None, or a whole number of counts from 1 to 2^18."""
    if threshold is None:
        return None
    ok = (isinstance(threshold, numbers.Integral)
          and not isinstance(threshold, bool)
          and 1 <= int(threshold) <= _KMAX_CAP)
    if not ok:
        raise ValueError("threshold must be None (the best threshold) or a "
                         f"whole number of counts from 1 to {_KMAX_CAP} "
                         f"(got {threshold!r})")
    return int(threshold)


def fidelity(eta: float, lam: float, gamma: float, tau: float,
             s: float = 1.0, leak: float = 0.0, noise_rate: float = 0.0,
             f0: float = 1.0, nt_switch=None,
             threshold: Optional[int] = None,
             repeated_flips: bool = False,
             t1_s: Optional[float] = None) -> dict:
    """Single-shot readout fidelity with the optimal integer threshold.

    eta : total detection efficiency per emission event (from
        `CavityInterface.eta`, or measured).
    lam : cyclicity at the operating point (cavity-boosted:
        `CavityInterface.lambda_cav`).
    gamma : total optical decay rate 1/s (`CavityInterface.gamma_cav`).
    tau : readout window (s). s : saturation parameter.
    leak : off-resonant/resonant scattering ratio of the dark spin
        (power broadening included by the caller); 0 disables the
        dark-scattering and switch-on channels. Each leak excitation
        switches the dark spin on with probability Lambda / (Lambda + 1),
        as for excitation into the bright spin's excited state (the
        spin-flipping line A2). Light the dark spin scatters through
        its own spin-conserving line (B2) leaves it dark and is not
        this channel; the revised source study finds that route
        dominant once the optical line is Purcell-broadened, and this
        model has no input for it: use `vacspin.fidelity_four_level`
        (new in 0.9.0) there.
    noise_rate : background count rate (1/s), present for both spin
        states. f0 : preparation/other infidelity prefactor; with
        failed preparations fitted by `fit_readout_histograms`,
        `ReadoutFit.f0` (since 0.10.0) is the value that matches them.
    nt_switch : ignored since 0.4.0 (the switch-on channel is now
        evaluated exactly); passing it gives a DeprecationWarning.
    threshold : None (default) for the best threshold, or (since 0.8.0)
        the whole number of counts your experiment uses: a shot with at
        least this many counts is called bright. The result is then the
        fidelity at that threshold, which is never above the best one.
    repeated_flips : (new in 0.11.0) False (default) keeps the model
        described below, in which the bright spin turns dark at most
        once and the dark spin turns bright at most once. True treats
        the spin as a two-state jump process for the whole window, with
        any number of changes: while bright it gives counts at eta R and
        turns dark at Gp = R / (Lambda + 1); while dark it gives leak
        counts at eta leak R and turns bright at
        leak R Lambda / (Lambda + 1). This removes the two
        simplifications of the default: a bright spin that has turned
        dark gives leak counts and can turn bright again, and a dark
        spin that has turned bright stops giving leak counts. For
        leak = 0 both give the same distributions. With leak the
        difference is not always small: for the confocal readout of
        README example 3 (50 us window, about two spin flips per bright
        record) with leak = 1e-4, the default gives 0.722 and this model
        0.752, which a simulation of the jump process confirms. The
        distributions come from the generating function of the jump
        process, inverted by a discrete Fourier transform (see
        `_jump_generating_functions`).
    t1_s : (new in 0.11.0, needs repeated_flips=True) the spin
        relaxation time T1 of the ground state, in seconds; None (the
        default) means no relaxation. Relaxation turns the ground-state
        spin over at the rate 1 / (2 T1) in each direction, so a spin
        left alone relaxes towards equal populations at the rate 1 / T1.
        It acts in the ground state only, as in `fidelity_four_level`:
        the dark spin is almost always there, while the driven bright
        spin spends the share s / (2 (1 + s)) of its time excited, so
        it turns dark by relaxation at the rate
        (1 / (2 T1)) (2 + s) / (2 (1 + s)). Equal rates in both
        directions hold when the temperature is well above
        h f_q / k_B (0.18 K for a 3.7 GHz qubit, with the exact SI
        values h = 6.62607015e-34 J s and k_B = 1.380649e-23 J/K, as
        listed by NIST, "Meet the Constants",
        https://www.nist.gov/si-redefinition/meet-constants). Far below
        it, relaxation mostly goes one way and this input does not
        describe it.

    The model: the bright spin gives flip-terminated Poisson counts
    plus Poisson background (noise_rate tau); the dark spin gives
    Poisson background and leak counts (eta leak R tau) plus the
    counts after an off-resonant switch-on at rate
    leak R Lambda / (Lambda + 1). Each distribution is exact within
    this model; the fidelity is maximized over the integer threshold.
    Counts are tracked up to a limit chosen so that the fidelity is
    within 1e-9 of the optimum over all thresholds (`truncation_bound`
    reports the guaranteed bound); a case that would need more than
    2^18 counts is refused.

    Returns dict(fidelity, threshold, n_bright, n_dark,
    n_bright_total, n_dark_total, truncation_bound, error_bright,
    error_dark): n_bright is the
    mean SIGNAL count of the bright spin, eta (Lambda + 1)
    (1 - exp(-Gp tau)); n_dark is (eta leak R + noise_rate) tau (both
    as before 0.4.0); n_bright_total and n_dark_total are the means of
    the two full distributions used (bright: signal + background;
    dark: n_dark + switch-on counts). Since 0.8.0 error_bright is the
    probability that a bright shot gives fewer than `threshold` counts
    and error_dark the probability that a dark shot gives at least
    `threshold`, both from the count model alone (f0 not applied), so
    fidelity = f0 (1 - (error_bright + error_dark) / 2) + (1 - f0) / 2.
    With a given threshold no search over thresholds is made, and
    truncation_bound is 0.

    With repeated_flips=True the same keys are returned: n_bright_total
    and n_dark_total are the exact means of the jump model (closed
    form), n_bright is n_bright_total without the background
    (noise_rate tau), and n_dark is n_dark_total. The counts are tracked
    far enough that less than 1e-10 of either distribution lies beyond,
    so that the fidelity is within 1e-9 as in the default model
    (truncation_bound then reports at most about 1e-10); a case
    that could need more than 2^18 counts is refused. The range is set
    by a Chernoff bound, which needs room for about 27 times the mean
    count of a bright record that ends in a spin flip: mean counts near
    10^4 and above are refused, where the default model, which only
    needs the fidelity within 1e-9, still works.
    """
    _check_common(eta, lam, gamma, tau, s)
    _check_background(leak, noise_rate)
    if not (0.0 < f0 <= 1.0):
        raise ValueError("f0 must be in (0, 1]")
    threshold = _check_threshold(threshold)
    t1 = _check_jump_inputs(repeated_flips, t1_s)
    if nt_switch is not None:
        warnings.warn("nt_switch is ignored since vacspin 0.4.0: the "
                      "dark switch-on channel is evaluated exactly",
                      DeprecationWarning, stacklevel=2)
    if repeated_flips:
        return _fidelity_jumps(eta, lam, gamma, tau, s, leak, noise_rate,
                               f0, threshold, t1)
    r = 0.5 * gamma * s / (1.0 + s)
    gp = r / (1.0 + lam)
    m_b = eta * (lam + 1.0) * (-np.expm1(-gp * tau))
    n_d = (eta * leak * r + noise_rate) * tau
    m0 = m_b + n_d + noise_rate * tau
    kmax = int(min(m0 + 10.0 * np.sqrt(m0 + 1.0) + 60, _KMAX_CAP))
    if threshold is not None:
        # the errors at a given threshold need the counts below it only
        kmax = max(kmax, threshold)
    while True:
        pmf_b, pmf_d = _count_distributions(eta, lam, gamma, tau, s,
                                            leak, noise_rate, kmax)
        cdf_b = np.cumsum(pmf_b)
        cdf_d = np.cumsum(pmf_d)
        if threshold is not None:
            e_b = min(max(float(cdf_b[threshold - 1]), 0.0), 1.0)
            e_d = min(max(1.0 - float(cdf_d[threshold - 1]), 0.0), 1.0)
            f_n = 1.0 - 0.5 * (e_d + e_b)
            n_r, bound = threshold, 0.0
            break
        nrs = np.arange(1, kmax + 1)
        err_dark = np.clip(1.0 - cdf_d[nrs - 1], 0.0, 1.0)
        err_bright = np.clip(cdf_b[nrs - 1], 0.0, 1.0)
        frs = 1.0 - 0.5 * (err_dark + err_bright)
        i = int(np.argmax(frs))
        # any threshold above kmax scores at most
        # min(F(kmax) + P_dark(K >= kmax)/2, 1 - P_bright(K <= kmax)/2)
        bound = max(0.0, min(0.5 * (1.0 - cdf_d[-2]),
                             1.0 - 0.5 * cdf_b[-1] - frs[i]))
        if bound <= _TRUNC_TOL:
            f_n, n_r = frs[i], int(nrs[i])
            e_b, e_d = float(err_bright[i]), float(err_dark[i])
            break
        if kmax >= _KMAX_CAP:
            raise ValueError(
                "count distributions too wide to evaluate: more than "
                f"{_KMAX_CAP} counts would be needed (mean bright signal "
                f"{m_b:.4g}, mean dark {n_d:.4g}); shorten the window "
                "or reduce the background")
        kmax = min(2 * kmax, _KMAX_CAP)
    return dict(fidelity=float(f0 * f_n + (1.0 - f0) * 0.5),
                threshold=int(n_r), n_bright=float(m_b),
                n_dark=float(n_d),
                n_bright_total=float(m_b + noise_rate * tau),
                n_dark_total=float(n_d + _switch_on_mean(
                    eta * r, gp, leak * r * lam / (lam + 1.0), tau)),
                truncation_bound=float(bound),
                error_bright=e_b, error_dark=e_d)


def _fidelity_jumps(eta, lam, gamma, tau, s, leak, noise_rate, f0,
                    threshold, t1):
    """`fidelity(..., repeated_flips=True)` (since 0.11.0)."""
    pmf_b, pmf_d = _jump_count_distributions(eta, lam, gamma, tau, s, leak,
                                             noise_rate, t1)
    cdf_b = np.cumsum(pmf_b)
    cdf_d = np.cumsum(pmf_d)
    kmax = pmf_b.size - 1
    if threshold is not None:
        if threshold <= kmax:
            e_b = min(max(float(cdf_b[threshold - 1]), 0.0), 1.0)
            e_d = min(max(1.0 - float(cdf_d[threshold - 1]), 0.0), 1.0)
        else:
            # beyond the count range, where less than 1e-10 is left
            e_b = min(max(float(cdf_b[-1]), 0.0), 1.0)
            e_d = 0.0
        n_r, bound = threshold, 0.0
    else:
        nrs = np.arange(1, kmax + 1)
        err_dark = np.clip(1.0 - cdf_d[nrs - 1], 0.0, 1.0)
        err_bright = np.clip(cdf_b[nrs - 1], 0.0, 1.0)
        frs = 1.0 - 0.5 * (err_dark + err_bright)
        i = int(np.argmax(frs))
        bound = max(0.0, min(0.5 * (1.0 - cdf_d[-2]),
                             1.0 - 0.5 * cdf_b[-1] - frs[i]))
        n_r = int(nrs[i])
        e_b, e_d = float(err_bright[i]), float(err_dark[i])
    f_n = 1.0 - 0.5 * (e_b + e_d)
    m_b, m_d = _jump_means(eta, lam, gamma, tau, s, leak, noise_rate, t1)
    return dict(fidelity=float(f0 * f_n + (1.0 - f0) * 0.5),
                threshold=int(n_r), n_bright=float(m_b - noise_rate * tau),
                n_dark=float(m_d), n_bright_total=float(m_b),
                n_dark_total=float(m_d), truncation_bound=float(bound),
                error_bright=e_b, error_dark=e_d)


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


def _check_target(target_fidelity):
    ok = (isinstance(target_fidelity, numbers.Real)
          and not isinstance(target_fidelity, bool)
          and 0.5 < float(target_fidelity) < 1.0)
    if not ok:
        raise ValueError("target_fidelity must be a number in (0.5, 1) "
                         f"(got {target_fidelity!r})")


def required_efficiency(target_fidelity: float, lam: float, gamma: float,
                        tau: float, s: float = 1.0, leak: float = 0.0,
                        noise_rate: float = 0.0,
                        f0: float = 1.0, repeated_flips: bool = False,
                        t1_s: Optional[float] = None) -> float:
    """The smallest detection efficiency eta reaching the target
    single-shot fidelity, by bisection of the exact model. Refuses --
    with the achievable maximum named -- when even eta = 1 falls
    short.

    The bisection starts at eta = 1e-6. When that already reaches the
    target (a target close to 1/2 with a long window or a very high
    cyclicity), the lower end is moved down by factors of 1000 until
    it falls short, so the result is the smallest efficiency, not the
    start of the search (before 0.5.0 it returned 1e-6 there).

    repeated_flips and t1_s (since 0.11.0) select the jump model of
    `fidelity`."""
    _check_target(target_fidelity)
    _check_jump_inputs(repeated_flips, t1_s)

    def f(eta):
        return fidelity(eta, lam, gamma, tau, s, leak, noise_rate, f0,
                        repeated_flips=repeated_flips,
                        t1_s=t1_s)["fidelity"]

    best = f(1.0)
    if best < target_fidelity:
        raise ValueError(
            f"target fidelity {target_fidelity} is unreachable at this "
            f"operating point: even eta = 1 gives {best:.4f}. Increase "
            "the cyclicity, the window, or reduce the dark counts.")
    hi, lo = 1.0, 1e-6
    for _ in range(100):
        if f(lo) < target_fidelity:
            break
        hi, lo = lo, lo / 1e3
    else:                                   # pragma: no cover
        raise ValueError("no efficiency above 1e-300 falls short of the "
                         "target; the model cannot resolve it")
    return float(_bisect_increasing(f, lo, hi, target_fidelity))


def _peak_near(f, lo, hi, n_scan=64, n_golden=60):
    """Largest value of f on [lo, hi] (0 < lo < hi): a scan on a
    logarithmic grid, then a golden-section search in log(tau) between
    the neighbors of the best scan point. The fidelity is the maximum
    over thresholds of smooth curves, so it can have more than one
    local maximum; the scan picks the highest before the golden
    section refines it. Returns (t_best, f_best)."""
    ts = np.geomspace(lo, hi, n_scan)
    vs = [f(t) for t in ts]
    j = int(np.argmax(vs))
    best_t, best_f = float(ts[j]), float(vs[j])
    a = math.log(ts[max(j - 1, 0)])
    b = math.log(ts[min(j + 1, n_scan - 1)])
    g = 0.5 * (math.sqrt(5.0) - 1.0)
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(math.exp(c)), f(math.exp(d))
    for _ in range(n_golden):
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(math.exp(d))
        for t_, f_ in ((c, fc), (d, fd)):
            if f_ > best_f:
                best_t, best_f = math.exp(t_), f_
    return best_t, best_f


def required_window(target_fidelity: float, eta: float, lam: float,
                    gamma: float, s: float = 1.0, leak: float = 0.0,
                    noise_rate: float = 0.0, f0: float = 1.0,
                    tau_max: float = 1e-2, repeated_flips: bool = False,
                    t1_s: Optional[float] = None) -> float:
    """The shortest readout window reaching the target fidelity.
    Refuses when no window up to tau_max does. With dark counts the
    fidelity is not monotone in tau, so the search scans a logarithmic
    grid from 1 ns upward, stops at the first window that reaches the
    target, and bisects on the rising flank just below it (below 1 ns
    when the first grid point already reaches the target).

    When no grid point reaches the target, the best part of the grid
    is searched finely for the peak fidelity before refusing (since
    0.5.0): a target between the best grid value and the true peak is
    reachable, and 0.4.0 refused it. The refusal names the peak
    found.

    repeated_flips and t1_s (since 0.11.0) select the jump model of
    `fidelity`."""
    _check_target(target_fidelity)
    _check_jump_inputs(repeated_flips, t1_s)
    try:
        tau_max = float(tau_max)
    except (TypeError, ValueError):
        tau_max = float("nan")
    if not (np.isfinite(tau_max) and tau_max > 1e-9):
        raise ValueError("tau_max must be finite and longer than 1 ns")

    def f(t):
        return fidelity(eta, lam, gamma, t, s, leak, noise_rate, f0,
                        repeated_flips=repeated_flips,
                        t1_s=t1_s)["fidelity"]

    taus = np.geomspace(1e-9, tau_max, 60)
    vals = []
    for i, t in enumerate(taus):
        vals.append(f(t))
        if vals[-1] >= target_fidelity:
            break
    else:
        # the peak can fall between two grid points: refine around the
        # best one before refusing
        j = int(np.argmax(vals))
        left = taus[j - 1] if j > 0 else taus[0] / 1e3
        right = taus[min(j + 1, taus.size - 1)]
        t_pk, f_pk = _peak_near(f, left, right)
        if f_pk < target_fidelity:
            raise ValueError(
                f"target fidelity {target_fidelity} is unreachable for "
                f"any window up to {tau_max} s (best {f_pk:.4f}, at "
                f"{t_pk:.4g} s); improve eta or the cyclicity.")
        lo = taus[j - 1] if j > 0 else taus[0] / 1e3
        while f(lo) >= target_fidelity:      # only possible for j = 0
            lo = lo / 1e3
        return float(_bisect_increasing(f, lo, t_pk, target_fidelity))
    if i == 0:
        # the fidelity tends to 1/2 as tau -> 0, so a shorter window
        # below the target exists; step down until it is found
        hi, lo = taus[0], taus[0] / 1e3
        while f(lo) >= target_fidelity:
            hi, lo = lo, lo / 1e3
    else:
        lo, hi = taus[i - 1], taus[i]
    return float(_bisect_increasing(f, lo, hi, target_fidelity))


def optimal_window(eta: float, lam: float, gamma: float, s: float = 1.0,
                   leak: float = 0.0, noise_rate: float = 0.0,
                   f0: float = 1.0, tau_min: float = 1e-9,
                   tau_max: float = 1e-2, repeated_flips: bool = False,
                   t1_s: Optional[float] = None) -> dict:
    """The readout window with the highest single-shot fidelity (new in
    0.7.0).

    With background counts or a leaking dark spin the fidelity rises,
    peaks and falls as the window grows: a longer window collects more
    bright photons but also more dark counts. This function finds the
    peak: a scan of 80 windows spaced evenly in log(tau) from `tau_min`
    to `tau_max`, then a golden-section search in log(tau) between the
    neighbors of the best one (as `required_window` does before it
    refuses). The arguments are those of `fidelity`.

    Returns the `fidelity` dict at the best window (fidelity,
    threshold, mean counts, truncation_bound) plus `tau`, the window
    in seconds.

    Refuses (ValueError) when there is no peak inside the range: when
    the fidelity at `tau_max` is within 1e-9 of the best value found
    (it is still rising, or has leveled off, as it does without
    background and leak), or when the best value is at `tau_min`. For
    a fidelity that keeps rising, ask for a target instead
    (`required_window`). The fidelity is the maximum over integer
    thresholds, so it can have more than one local peak; the scan
    picks the highest one it sees before refining it.

    Check (tests): with a threshold of one photon and background rate
    N, F(tau) = 1/2 + e^{-N tau} (1 - P0(tau)) / 2 with P0 the
    probability of no signal photon, and dF/dtau = 0 gives the closed
    form tau* = ln(1 + a / N) / a, a = R / (Lambda + 1) + eta R.

    repeated_flips and t1_s (since 0.11.0) select the jump model of
    `fidelity`. With spin relaxation the fidelity falls at long windows
    even without background or leak, so it then has a peak.
    """
    _check_common(eta, lam, gamma, 1.0, s)
    _check_background(leak, noise_rate)
    _check_jump_inputs(repeated_flips, t1_s)
    if not (0.0 < f0 <= 1.0):
        raise ValueError("f0 must be in (0, 1]")
    try:
        lo, hi = float(tau_min), float(tau_max)
    except (TypeError, ValueError):
        lo = hi = float("nan")
    if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo < hi):
        raise ValueError("tau_min and tau_max must be finite with "
                         "0 < tau_min < tau_max (seconds)")
    if eta == 0.0:
        raise ValueError("eta = 0: no signal photon is ever detected, "
                         "so every window gives fidelity 1/2")

    def f(t):
        return fidelity(eta, lam, gamma, t, s, leak, noise_rate, f0,
                        repeated_flips=repeated_flips,
                        t1_s=t1_s)["fidelity"]

    n_scan = 80
    taus = np.geomspace(lo, hi, n_scan)
    vals = np.array([f(t) for t in taus])
    j = int(np.argmax(vals))
    left = taus[max(j - 1, 0)]
    right = taus[min(j + 1, n_scan - 1)]
    t_pk, f_pk = _peak_near(f, left, right)
    if f_pk < vals[j]:                       # keep the scan's best
        t_pk, f_pk = float(taus[j]), float(vals[j])
    if vals[-1] >= f_pk - 1e-9:
        raise ValueError(
            f"no peak below tau_max = {hi:.4g} s: the fidelity there "
            f"({vals[-1]:.6f}) is within 1e-9 of the best found "
            f"({f_pk:.6f}), so it is still rising or has leveled off "
            "(without background and leak it never falls). Give a "
            "larger tau_max, or ask required_window for a target")
    if t_pk <= lo * (1.0 + 1e-9):
        raise ValueError(
            f"the best window is at tau_min = {lo:.4g} s (fidelity "
            f"{f_pk:.6f}); the peak lies below it. Give a smaller "
            "tau_min")
    out = fidelity(eta, lam, gamma, t_pk, s, leak, noise_rate, f0,
                   repeated_flips=repeated_flips, t1_s=t1_s)
    out["tau"] = float(t_pk)
    return out
