"""Four-level single-shot readout in a broad cavity (new in 0.9.0).

The two-state counting model of `vacspin.readout` follows Rosenthal et
al., arXiv:2403.13110, Appendix A: the dark spin's only light is
`leak`, an excitation that switches it on. In a cavity the optical line
is Purcell-broadened, and the dark spin is then excited mainly through
its own spin-conserving line B2, only omega_e - omega_q away from the
laser (0.73 GHz for the SnV- at the operating point of the source
study). That light is counted, but it leaves the spin dark; it switches
the spin on only through the weak spin-flipping decay B1. The revised
source study (Mahim, Rahman and Mohsin, "Fast single-shot readout of
tin-vacancy spins with an overcoupled diamond nanocavity on thin-film
lithium niobate", submitted to Optics Express, 2026; code
Tanvir-Mahmud-Mahim/snv-tfln-cavity-interface, sim/readout_fcs.py)
therefore computes the readout from the four-level master equation.
This module implements that model.

States: |dn> (bright ground spin state, the lower one), |up> (dark),
|A> and |B> (the two spin states of the lower excited orbital branch;
|A> is the one |dn> is cycled to). The C line has four transitions:
A1 = |dn>-|A> (driven by the laser), B2 = |up>-|B>, and the
spin-flipping A2 = |up>-|A> and B1 = |dn>-|B>. `c_line` takes their
frequencies and polarization-summed dipole strengths S_ge from the spin
Hamiltonian of this package at the actual field and strain.

The model, in the frame rotating at the laser frequency:

    H = 2 pi [w_q |up><up| - d_L |A><A| + (w_e - d_L) |B><B|]
        + sum_ge (Omega_ge / 2)(|e><g| + |g><e|),
    Omega_ge = Omega_A1 sqrt(S_ge / S_A1),  Omega_A1^2 = s gamma G2,
    G2 = gamma / 2 + gamma_phi,

with d_L the laser detuning from A1. Each excited state e decays at the
common total rate gamma (the broad cavity enhances all four lines
alike) into the ground state g with weight S_ge / (S_dn,e + S_up,e);
gamma_phi = pi * extra_width_hz is extra pure dephasing of the excited
states (homogeneous broadening, FWHM extra_width_hz). Every emitted
photon is detected with probability eta, whatever its line; detector
background `noise_rate` adds Poisson counts to both spin states.

Counting statistics are exact within this master equation (full
counting statistics): the generating function
G(z) = Tr[exp(L_z tau) rho_0], in which each jump term is weighted by
1 - eta + eta z, is evaluated at N points on the unit circle and
inverted by a discrete Fourier transform. exp(L_z tau) comes from the
eigen-decomposition of L_z, or, where its eigenvectors are
ill-conditioned (condition number above 1e5, for example near the
critically damped drive s = 1/8 of a two-level line), from scaling and
squaring of a Taylor series. The count range N is chosen so that wrap-around
cannot exceed 1e-12: the photon emission rate never exceeds gamma, so
the detected count is stochastically below a Poisson count of mean
(eta gamma + noise_rate) tau, whose tail beyond N is bounded by the
Chernoff bound exp(-mu) (e mu / N)^N. The result is refused if the
count distributions do not sum to 1 within 1e-8. Round-off leaves
about 1e-12 in the distributions for windows of a few hundred
nanoseconds, growing with the window to about 1e-8 at a millisecond.

Quasi-static spectral diffusion (a common Gaussian shift of all lines,
fixed during one window, FWHM `spectral_diffusion_hz`) is included by
averaging the two count distributions over the laser detuning BEFORE
the threshold is chosen. The average runs on an even grid over +-6
standard deviations with Gaussian weights; the grid is refined (every
step halved) until no probability changes by more than 1e-7, and the
result is refused if 1025 points are not enough. (A fixed set of 9 to
41 Gauss-Hermite points, as used in the source study's code, does not
converge once the diffusion is wider than the optical line: at 2 GHz
FWHM the bright mean count still moves from 8.25 to 5.96.)

What it assumes: the cavity linewidth is much larger than the C-line
splittings (all four lines see the same Purcell factor), and `gamma`
is the total decay rate of the excited states with the cavity
(`CavityInterface.gamma_cav`, with `include_d_line=True` to count the
D line as the source study does). The D-line photons themselves are
counted through `eta` like the others. Spin relaxation in the ground
state (T1) and charge-state jumps are not modeled.

Validated in tests/test_fourlevel.py against: the closed-form excited
population of a resonantly driven two-level emitter (Torrey), whose
time integral gives the mean count; a time-domain integration of the
photon-number-resolved master equation (a different algorithm); the
two-state rate model of `vacspin.readout` where both apply; and the
numbers printed in the revised source study, including its confocal
check against the measured readout of arXiv:2403.13110.
"""
from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .hamiltonian import solve
from .params import SpinParameters
from .transitions import _require_split, transition_strength

__all__ = ["CLine", "c_line", "four_level_counts", "fidelity_four_level"]

_N_MIN = 64
_N_CAP = 1 << 16          # largest count range (and transform length)
_TAIL = 1e-12             # bound on the probability beyond the range
_NORM_TOL = 1e-8          # allowed error of the total probability
_COND_MAX = 1e5           # eigenvector condition number accepted
_CHUNK = 1024             # points of the unit circle per batch
_SD_TOL = 1e-7            # convergence of the spectral-diffusion average
_SD_MAX_POINTS = 1025


@dataclass(frozen=True)
class CLine:
    """The four C-line transitions of one spin at one field and strain.

    strengths : 2 x 2 array S[g, e], g in (dn, up), e in (A, B), the
        polarization-summed dipole strengths (units of `transition_strength`).
    w_q_ghz : qubit frequency, the splitting of |dn> and |up> (GHz).
    w_e_ghz : splitting of |A> and |B> (GHz); B2 lies w_e - w_q above A1.
    lam_a, lam_b : cyclicities of |A> (S_A1 / S_A2) and |B> (S_B2 / S_B1).
    delta_g_ghz : ground-state orbital splitting (GHz), the C-D spacing.
    """
    strengths: np.ndarray
    w_q_ghz: float
    w_e_ghz: float
    lam_a: float
    lam_b: float
    delta_g_ghz: float

    @property
    def detuning_b2_ghz(self) -> float:
        """Frequency of B2 minus that of A1 (GHz): w_e - w_q."""
        return self.w_e_ghz - self.w_q_ghz


def c_line(params: SpinParameters, b_lab, strain_scale: float = 1.0) -> CLine:
    """The C-line structure from the spin Hamiltonian at a lab-frame
    field (tesla) and strain scale.

    The bright state |dn> is the lower state of the lowest ground
    doublet and |A> the state of the lower excited doublet it couples
    to more strongly; |B> is the other one. Refuses a field that does
    not split the two doublets, and a dark line with no strength at all
    (S_B2 + S_B1 = 0 or S_A1 + S_A2 = 0)."""
    eg, vg, ee, ve = solve(params, b_lab, strain_scale)
    _require_split(eg, 0, "qubit (lowest ground)", "C line")
    _require_split(ee, 0, "lower excited", "C line")
    s = np.array([[transition_strength(vg[:, g], ve[:, e])
                   for e in (0, 1)] for g in (0, 1)])
    a = int(np.argmax(s[0]))
    b = 1 - a
    s = s[:, [a, b]]
    if not (s[0, 0] > 0.0 and s[:, 1].sum() > 0.0):
        raise ValueError("the C line has an excited state with no optical "
                         "strength to the qubit doublet; the four-level "
                         "model does not apply")
    w_q = float(eg[1] - eg[0])
    w_e = float(ee[b] - ee[a])
    lam_a = float(s[0, 0] / s[1, 0]) if s[1, 0] > 0 else np.inf
    lam_b = float(s[1, 1] / s[0, 1]) if s[0, 1] > 0 else np.inf
    return CLine(strengths=s, w_q_ghz=w_q, w_e_ghz=w_e, lam_a=lam_a,
                 lam_b=lam_b, delta_g_ghz=float(eg[2] - eg[0]))


def _check_cline(cl):
    if not isinstance(cl, CLine):
        raise ValueError("pass the C line as a CLine (from vacspin.c_line)")
    s = np.asarray(cl.strengths, dtype=float)
    if s.shape != (2, 2) or not np.all(np.isfinite(s)) or np.any(s < 0):
        raise ValueError("CLine.strengths must be a finite 2 x 2 array "
                         "of non-negative dipole strengths")
    if not (s[0, 0] > 0.0 and s[:, 1].sum() > 0.0):
        raise ValueError("the four-level model needs S_A1 > 0 and some "
                         "strength into |B>")
    for name in ("w_q_ghz", "w_e_ghz"):
        v = float(getattr(cl, name))
        if not np.isfinite(v):
            raise ValueError(f"CLine.{name} must be finite")
    return s


def _check_inputs(gamma, eta, tau, s, noise_rate, extra_width_hz,
                  spectral_diffusion_hz, laser_detuning_hz):
    for name, v in (("gamma", gamma), ("tau", tau), ("s", s)):
        v = float(v)
        if not (np.isfinite(v) and v > 0.0):
            raise ValueError(f"{name} must be positive and finite")
    if not (0.0 <= float(eta) <= 1.0):
        raise ValueError("eta must be in [0, 1]")
    for name, v in (("noise_rate", noise_rate),
                    ("extra_width_hz", extra_width_hz),
                    ("spectral_diffusion_hz", spectral_diffusion_hz)):
        v = float(v)
        if not (np.isfinite(v) and v >= 0.0):
            raise ValueError(f"{name} must be finite and >= 0")
    if not np.isfinite(float(laser_detuning_hz)):
        raise ValueError("laser_detuning_hz must be finite")


def _superoperators(s_ge, w_q, w_e, gamma, sat, gphi, d_l):
    """L0 (everything but the jumps) and J (the sum of the jump terms),
    column-stacking convention vec(X rho Y) = (Y^T kron X) vec(rho).
    Units: frequencies in GHz, rates in 1/ns."""
    d = 4
    two_pi = 2.0 * np.pi
    g2 = gamma / 2.0 + gphi
    om0 = np.sqrt(sat * gamma * g2)
    h = np.zeros((d, d), dtype=complex)
    h[1, 1] = two_pi * w_q
    h[2, 2] = -two_pi * d_l
    h[3, 3] = two_pi * (w_e - d_l)
    for g in (0, 1):
        for e in (0, 1):
            om = om0 * np.sqrt(s_ge[g, e] / s_ge[0, 0])
            h[2 + e, g] = h[g, 2 + e] = om / 2.0
    eye = np.eye(d)
    l0 = -1j * (np.kron(eye, h) - np.kron(h.T, eye))
    jump = np.zeros((d * d, d * d), dtype=complex)
    for e in (0, 1):
        tot = s_ge[0, e] + s_ge[1, e]
        for g in (0, 1):
            if s_ge[g, e] == 0.0:
                continue
            c = np.zeros((d, d))
            c[g, 2 + e] = np.sqrt(gamma * s_ge[g, e] / tot)
            cdc = c.T @ c
            jump += np.kron(c.conj(), c)
            l0 += -0.5 * np.kron(eye, cdc) - 0.5 * np.kron(cdc.T, eye)
    if gphi > 0.0:
        p = np.zeros((d, d))
        p[2, 2] = p[3, 3] = 1.0
        c = np.sqrt(2.0 * gphi) * p
        cdc = c.T @ c
        l0 += (np.kron(c, c) - 0.5 * np.kron(eye, cdc)
               - 0.5 * np.kron(cdc.T, eye))
    return l0, jump


def _expm_batch(a):
    """exp(a) for a stack of square matrices: scaling and squaring of a
    Taylor series (terms until they fall below 1e-17 of the sum)."""
    norm = float(np.max(np.sum(np.abs(a), axis=-2)))
    m = max(0, int(np.ceil(np.log2(norm / 0.5)))) if norm > 0.5 else 0
    x = a / (2.0 ** m)
    eye = np.broadcast_to(np.eye(a.shape[-1], dtype=a.dtype), a.shape)
    out = eye.copy()
    term = eye.copy()
    for k in range(1, 40):
        term = term @ x / k
        out = out + term
        if float(np.max(np.abs(term))) < 1e-17:
            break
    for _ in range(m):
        out = out @ out
    return out


def _propagators(lz, tau):
    """exp(lz * tau) for a stack of generators: eigen-decomposition
    where the eigenvectors are well conditioned, otherwise Taylor
    scaling and squaring."""
    w, v = np.linalg.eig(lz)
    cond = np.linalg.cond(v)
    good = np.isfinite(cond) & (cond < _COND_MAX)
    prop = np.empty_like(lz)
    if np.any(good):
        vg = v[good]
        prop[good] = (vg * np.exp(w[good] * tau)[:, None, :]) \
            @ np.linalg.inv(vg)
    if not np.all(good):
        prop[~good] = _expm_batch(lz[~good] * tau)
    return prop


def _range_for(mu):
    """Smallest power of two N (at least _N_MIN) with
    P(Poisson(mu) >= N) <= _TAIL by the Chernoff bound."""
    n = _N_MIN
    while True:
        if n > mu and (-mu + n * (1.0 + np.log(max(mu, 1e-300))
                                  - np.log(n))) <= np.log(_TAIL):
            return n
        n *= 2
        if n > _N_CAP:
            return None


def _pmfs(s_ge, w_q, w_e, gamma, eta, sat, tau, noise, gphi, d_l, n):
    """Count distributions (bright, dark) on 0..n-1 by inverting the
    generating function at n points of the unit circle (in batches of
    _CHUNK points, to bound the memory)."""
    l0, jump = _superoperators(s_ge, w_q, w_e, gamma, sat, gphi, d_l)
    z = np.exp(2j * np.pi * np.arange(n) / n)
    vec_i = np.eye(4).reshape(-1, order="F")
    g_b = np.empty(n, dtype=complex)
    g_d = np.empty(n, dtype=complex)
    for i0 in range(0, n, _CHUNK):
        zc = z[i0:i0 + _CHUNK]
        lz = l0[None] + (1.0 - eta + eta * zc)[:, None, None] * jump[None]
        rows = np.einsum("i,zij->zj", vec_i, _propagators(lz, tau))
        g_b[i0:i0 + _CHUNK] = rows[:, 0]       # start in vec(|dn><dn|)
        g_d[i0:i0 + _CHUNK] = rows[:, 5]       # start in vec(|up><up|)
    if noise > 0.0:
        bg = np.exp((z - 1.0) * noise * tau)
        g_b = g_b * bg
        g_d = g_d * bg
    p_b = np.real(np.fft.fft(g_b)) / n
    p_d = np.real(np.fft.fft(g_d)) / n
    return p_b, p_d


def four_level_counts(cline: CLine, gamma: float, eta: float, tau: float,
                      s: float = 1.0, noise_rate: float = 0.0,
                      extra_width_hz: float = 0.0,
                      spectral_diffusion_hz: float = 0.0,
                      laser_detuning_hz: float = 0.0):
    """Exact detected-count distributions of the four-level model.

    cline : `CLine` from `c_line`.
    gamma : total decay rate of the excited states (1/s), for example
        `CavityInterface.gamma_cav`.
    eta : detection probability per emitted photon, any line, in [0, 1].
    tau : readout window (s). s : saturation parameter of A1
        (Omega_A1^2 = s gamma (gamma / 2 + gamma_phi)).
    noise_rate : Poisson background count rate (1/s), both spin states.
    extra_width_hz : extra homogeneous FWHM of the optical lines (Hz),
        as pure dephasing of the excited states.
    spectral_diffusion_hz : FWHM (Hz) of a quasi-static Gaussian shift
        of all lines, averaged on a grid refined until converged (see
        the module docstring); each grid point costs one full
        evaluation, so a diffusion much wider than the optical line is
        slow.
    laser_detuning_hz : fixed laser detuning from A1 (Hz).

    Returns (pmf_bright, pmf_dark): arrays over 0, 1, 2, ... counts for
    a spin prepared in |dn> and in |up>, each summing to 1 within 1e-8.
    """
    s_ge = _check_cline(cline)
    _check_inputs(gamma, eta, tau, s, noise_rate, extra_width_hz,
                  spectral_diffusion_hz, laser_detuning_hz)
    g_ns = float(gamma) * 1e-9
    t_ns = float(tau) * 1e9
    noise_ns = float(noise_rate) * 1e-9
    gphi = np.pi * float(extra_width_hz) * 1e-9
    d0 = float(laser_detuning_hz) * 1e-9
    # the emission rate never exceeds gamma, so the detected count is
    # stochastically below Poisson((eta gamma + noise) tau)
    mu = (float(eta) * g_ns + noise_ns) * t_ns
    n = _range_for(mu)
    if n is None:
        raise ValueError(
            f"the count distributions may need more than {_N_CAP} counts "
            "(window too long or rate too high for this model)")

    def at(dl):
        return _pmfs(s_ge, cline.w_q_ghz, cline.w_e_ghz, g_ns, float(eta),
                     float(s), t_ns, noise_ns, gphi, dl, n)

    if spectral_diffusion_hz > 0.0:
        sig = float(spectral_diffusion_hz) * 1e-9 / (
            2.0 * np.sqrt(2.0 * np.log(2.0)))
        m = 17
        x = np.linspace(-6.0, 6.0, m)
        cache = [at(d0 + sig * xi) for xi in x]
        prev = None
        while True:
            w = np.exp(-0.5 * x ** 2)
            w /= w.sum()
            p_b = sum(wi * c[0] for wi, c in zip(w, cache))
            p_d = sum(wi * c[1] for wi, c in zip(w, cache))
            if prev is not None and max(
                    np.max(np.abs(p_b - prev[0])),
                    np.max(np.abs(p_d - prev[1]))) <= _SD_TOL:
                break
            prev = (p_b, p_d)
            if 2 * m - 1 > _SD_MAX_POINTS:
                raise ValueError(
                    "the spectral-diffusion average did not converge on "
                    f"{m} points (to {_SD_TOL:g} in each probability); the "
                    "diffusion is too wide compared with the optical line")
            mid = 0.5 * (x[1:] + x[:-1])
            new = [at(d0 + sig * xi) for xi in mid]
            merged = [None] * (2 * m - 1)
            merged[0::2] = cache
            merged[1::2] = new
            cache = merged
            m = 2 * m - 1
            x = np.linspace(-6.0, 6.0, m)
    else:
        p_b, p_d = at(d0)
    p_b = np.clip(p_b, 0.0, None)
    p_d = np.clip(p_d, 0.0, None)
    for p, name in ((p_b, "bright"), (p_d, "dark")):
        if abs(p.sum() - 1.0) > _NORM_TOL:
            raise ValueError(
                f"the {name} count distribution did not converge (sum "
                f"{p.sum():.12f}); the inputs are outside the numerically "
                "safe range")
    return p_b, p_d


def fidelity_four_level(cline: CLine, gamma: float, eta: float,
                        tau: float, s: float = 1.0,
                        noise_rate: float = 0.0, f0: float = 1.0,
                        extra_width_hz: float = 0.0,
                        spectral_diffusion_hz: float = 0.0,
                        laser_detuning_hz: float = 0.0,
                        threshold: Optional[int] = None) -> dict:
    """Single-shot readout fidelity from the four-level count
    distributions of `four_level_counts` (same inputs).

    A shot with at least `threshold` counts is called bright;
    threshold=None picks the best whole number. f0 is a state-
    preparation factor: fidelity = f0 (1 - (e_b + e_d) / 2) + (1 - f0) / 2,
    as in `vacspin.fidelity` (the source study uses f0 = 0.99).
    The spin prepared in |dn> (the one the laser at A1 drives) is the
    bright one; with the laser tuned onto another line the labels do
    not follow, and the fidelity can come out near 1/2.

    Returns dict(fidelity, threshold, n_bright, n_dark, error_bright,
    error_dark): the mean counts of the two distributions, and the
    probabilities that a bright shot falls below the threshold and a
    dark shot reaches it (count model only, f0 not applied)."""
    if not (0.0 < float(f0) <= 1.0):
        raise ValueError("f0 must be in (0, 1]")
    if threshold is not None and not (
            isinstance(threshold, numbers.Integral)
            and not isinstance(threshold, bool) and int(threshold) >= 1):
        raise ValueError("threshold must be None (the best threshold) or a "
                         f"whole number of counts >= 1 (got {threshold!r})")
    p_b, p_d = four_level_counts(cline, gamma, eta, tau, s, noise_rate,
                                 extra_width_hz, spectral_diffusion_hz,
                                 laser_detuning_hz)
    k = np.arange(p_b.size)
    cdf_b = np.cumsum(p_b)
    cdf_d = np.cumsum(p_d)
    if threshold is None:
        nr = np.arange(1, p_b.size)
        err_b = np.clip(cdf_b[nr - 1], 0.0, 1.0)
        err_d = np.clip(1.0 - cdf_d[nr - 1], 0.0, 1.0)
        i = int(np.argmax(1.0 - 0.5 * (err_b + err_d)))
        n_r, e_b, e_d = int(nr[i]), float(err_b[i]), float(err_d[i])
    else:
        n_r = int(threshold)
        if n_r >= p_b.size:
            # beyond the count range: no shot reaches the threshold
            e_b = float(min(max(cdf_b[-1], 0.0), 1.0))
            e_d = 0.0
        else:
            e_b = float(min(max(cdf_b[n_r - 1], 0.0), 1.0))
            e_d = float(min(max(1.0 - cdf_d[n_r - 1], 0.0), 1.0))
    f_count = 1.0 - 0.5 * (e_b + e_d)
    return dict(fidelity=float(f0) * f_count + (1.0 - float(f0)) / 2.0,
                threshold=n_r, n_bright=float((k * p_b).sum()),
                n_dark=float((k * p_d).sum()), error_bright=e_b,
                error_dark=e_d)
