"""0.11.0: the two-state readout as a full jump process
(`fidelity(..., repeated_flips=True)`), spin relaxation (`t1_s`) in it
and in the four-level model.

Each check compares with something independent:

* exact limits: with leak = 0 the jump model must give the default
  model's distributions (a separate closed-form calculation), and when
  both spin states give counts at the same rate the count is Poisson
  whatever the spin does;
* the mean counts against a time integration of the two-state rate
  equations written here;
* the count distributions against a time integration (RK4) of the
  photon-number-resolved rate equations written here (no generating
  function, no Fourier transform);
* the fidelity and the means against a simulation of the jump process,
  event by event;
* the four-level model with relaxation against its own RK4 integration
  written here, and against the two-state jump model where both apply.
"""
import math

import numpy as np
import pytest

import vacspin as vs
from vacspin.fourlevel import CLine
from vacspin.readout import _count_distributions, _jump_count_distributions

GAMMA0 = vs.snv_emission().gamma0


def _rates(eta, lam, gamma, s, leak, t1):
    """The jump model's rates, written from its docstring: counts at
    eta R (bright) and eta leak R (dark); bright to dark at
    R / (lam + 1) plus relaxation in the ground state, which the bright
    spin occupies the share 1 - s / (2 (1 + s)) of the time; dark to
    bright at leak R lam / (lam + 1) plus relaxation."""
    r = gamma / 2.0 * s / (1.0 + s)
    k = 0.0 if t1 is None else 1.0 / (2.0 * t1)
    ground_share = 1.0 - s / (2.0 * (1.0 + s))
    return (eta * r, eta * leak * r, r / (lam + 1.0) + k * ground_share,
            leak * r * lam / (lam + 1.0) + k)


CASES = [
    # eta, lam, gamma, tau, s, leak, noise, t1
    (0.002, 2244.0, GAMMA0, 50e-6, 10.0, 1e-3, 4e3, None),   # confocal
    (0.3, 500.0, GAMMA0, 2e-6, 1.0, 1e-3, 5e4, 3e-6),
    (0.6, 16000.0, 7.26 * GAMMA0, 2e-8, 2.0, 3.7e-3, 1e3, None),
    (0.002, 2244.0, GAMMA0, 50e-6, 10.0, 1e-4, 4e3, 20e-6),
]


@pytest.mark.parametrize("case", [
    (0.002, 2244.0, 50e-6, 10.0, 4e3), (0.6, 16000.0, 2e-8, 2.0, 1e3),
    (0.3, 40.0, 3e-6, 1.0, 0.0), (0.3, 500.0, 2e-6, 1.0, 5e4)])
def test_without_leak_the_two_models_agree(case):
    """With leak = 0 a bright spin that turns dark stays dark, so the
    jump model must give the default model's distributions, which come
    from a separate closed form (measured: 1.4e-15 at most; the
    fidelity 1.1e-16, the means 2.2e-16 relative)."""
    eta, lam, tau, s, noise = case
    pb, pd = _jump_count_distributions(eta, lam, GAMMA0, tau, s, 0.0, noise,
                                       None)
    qb, qd = _count_distributions(eta, lam, GAMMA0, tau, s, 0.0, noise,
                                  pb.size - 1)
    assert np.max(np.abs(pb - qb)) < 1e-13
    assert np.max(np.abs(pd - qd)) < 1e-13
    a = vs.fidelity(eta, lam, GAMMA0, tau, s=s, noise_rate=noise)
    b = vs.fidelity(eta, lam, GAMMA0, tau, s=s, noise_rate=noise,
                    repeated_flips=True)
    assert b["threshold"] == a["threshold"]
    assert b["fidelity"] == pytest.approx(a["fidelity"], abs=1e-12)
    for key in ("n_bright", "n_bright_total", "n_dark_total"):
        assert b[key] == pytest.approx(a[key], rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("t1", [None, 1e-6])
def test_equal_count_rates_give_poisson_counts(t1):
    """With leak = 1 both spin states give counts at eta R, so the count
    is Poisson with mean (eta R + noise) tau however often the spin
    changes (measured: 1.1e-15)."""
    eta, lam, tau, s, noise = 0.3, 50.0, 2e-6, 1.0, 1e5
    pb, pd = _jump_count_distributions(eta, lam, GAMMA0, tau, s, 1.0, noise,
                                       t1)
    want = vs.poisson_pmf((eta * GAMMA0 / 4.0 + noise) * tau, pb.size - 1)
    assert np.max(np.abs(pb - want)) < 1e-13
    assert np.max(np.abs(pd - want)) < 1e-13


def _rk4_means(case, steps=20000):
    """Mean counts of both starts from the rate equations
    dp/dt = Q p of the bright share, integrated with RK4, the counts
    accumulated alongside."""
    eta, lam, gamma, tau, s, leak, noise, t1 = case
    c_b, c_d, k_bd, k_db = _rates(eta, lam, gamma, s, leak, t1)

    def rhs(y):          # y = (p_bright, p_dark, counts)
        pb, pd = y[0], y[1]
        return np.array([-k_bd * pb + k_db * pd, k_bd * pb - k_db * pd,
                         c_b * pb + c_d * pd])

    out = []
    for start in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        y = np.array(start)
        dt = tau / steps
        for _ in range(steps):
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * dt * k1)
            k3 = rhs(y + 0.5 * dt * k2)
            k4 = rhs(y + dt * k3)
            y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(y[2] + noise * tau)
    return out


@pytest.mark.parametrize("case", CASES)
def test_mean_counts(case):
    """The closed-form means equal the RK4 integration of the rate
    equations (to 1e-9 relative; measured 1.2e-14) and the means of the
    returned distributions (to 1e-9 relative; measured 1.3e-13)."""
    want_b, want_d = _rk4_means(case)
    eta, lam, gamma, tau, s, leak, noise, t1 = case
    out = vs.fidelity(eta, lam, gamma, tau, s, leak, noise,
                      repeated_flips=True, t1_s=t1)
    assert out["n_bright_total"] == pytest.approx(want_b, rel=1e-9)
    assert out["n_dark_total"] == pytest.approx(want_d, rel=1e-9)
    assert out["n_bright"] == pytest.approx(want_b - noise * tau, rel=1e-9)
    pb, pd = _jump_count_distributions(eta, lam, gamma, tau, s, leak, noise,
                                       t1)
    k = np.arange(pb.size)
    assert (k * pb).sum() == pytest.approx(want_b, rel=1e-9)
    assert (k * pd).sum() == pytest.approx(want_d, rel=1e-9)


def _rk4_distributions(case, n_max, steps):
    """P(k) of both starts from the photon-number-resolved rate
    equations
        dP_n^B/dt = -(k_bd + c_b) P_n^B + k_db P_n^D + c_b P_(n-1)^B,
        dP_n^D/dt = -(k_db + c_d) P_n^D + k_bd P_n^B + c_d P_(n-1)^D,
    integrated with RK4, then convolved with the Poisson background."""
    eta, lam, gamma, tau, s, leak, noise, t1 = case
    c_b, c_d, k_bd, k_db = _rates(eta, lam, gamma, s, leak, t1)

    def rhs(p):          # p: (start, state, n)
        b, d = p[:, 0], p[:, 1]
        db = -(k_bd + c_b) * b + k_db * d
        dd = -(k_db + c_d) * d + k_bd * b
        db[:, 1:] += c_b * b[:, :-1]
        dd[:, 1:] += c_d * d[:, :-1]
        return np.stack([db, dd], axis=1)

    p = np.zeros((2, 2, n_max + 1))
    p[0, 0, 0] = 1.0
    p[1, 1, 0] = 1.0
    dt = tau / steps
    for _ in range(steps):
        k1 = rhs(p)
        k2 = rhs(p + 0.5 * dt * k1)
        k3 = rhs(p + 0.5 * dt * k2)
        k4 = rhs(p + dt * k3)
        p = p + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    sig = p.sum(axis=1)                       # (start, n)
    bg = vs.poisson_pmf(noise * tau, n_max)
    return [np.convolve(sig[i], bg)[:n_max + 1] for i in range(2)]


@pytest.mark.parametrize("case", [CASES[0], CASES[1], CASES[3]])
def test_distributions_against_time_integration(case):
    """The count distributions equal an RK4 integration of the
    photon-number-resolved rate equations (to 1e-9; measured 6e-14 at
    most) over the first 40 counts."""
    eta, lam, gamma, tau, s, leak, noise, t1 = case
    pb, pd = _jump_count_distributions(eta, lam, gamma, tau, s, leak, noise,
                                       t1)
    qb, qd = _rk4_distributions(case, n_max=60, steps=4000)
    assert np.max(np.abs(pb[:40] - qb[:40])) < 1e-9
    assert np.max(np.abs(pd[:40] - qd[:40])) < 1e-9


def _simulate(case, n, start_bright, rng):
    """The jump process event by event: the time to the next spin change
    is exponential; counts in each stretch are Poisson."""
    eta, lam, gamma, tau, s, leak, noise, t1 = case
    c_b, c_d, k_bd, k_db = _rates(eta, lam, gamma, s, leak, t1)
    t = np.zeros(n)
    bright = np.full(n, start_bright)
    counts = rng.poisson(noise * tau, n)
    active = np.ones(n, dtype=bool)
    while active.any():
        idx = np.nonzero(active)[0]
        rate = np.where(bright[idx], k_bd, k_db)
        with np.errstate(divide="ignore"):
            dt = rng.exponential(1.0, idx.size) / rate
        end = np.minimum(t[idx] + dt, tau)
        counts[idx] += rng.poisson(np.where(bright[idx], c_b, c_d)
                                   * (end - t[idx]))
        t[idx] = end
        done = end >= tau
        bright[idx[~done]] = ~bright[idx[~done]]
        active[idx[done]] = False
    return counts


@pytest.mark.parametrize("case", [CASES[0], CASES[1], CASES[3]])
def test_fidelity_matches_simulation(case):
    """400 000 simulated shots per spin state: the fidelity at the
    returned threshold and both mean counts agree with the jump model
    within 5 standard errors (measured: within 1.8)."""
    eta, lam, gamma, tau, s, leak, noise, t1 = case
    out = vs.fidelity(eta, lam, gamma, tau, s, leak, noise,
                      repeated_flips=True, t1_s=t1)
    rng = np.random.default_rng(11)
    n = 400_000
    kb = _simulate(case, n, True, rng)
    kd = _simulate(case, n, False, rng)
    thr = out["threshold"]
    f_sim = 1.0 - 0.5 * (np.mean(kd >= thr) + np.mean(kb < thr))
    assert abs(f_sim - out["fidelity"]) < 5.0 * math.sqrt(0.25 / n)
    assert abs(kb.mean() - out["n_bright_total"]) < 5.0 * kb.std() / n ** 0.5
    assert abs(kd.mean() - out["n_dark_total"]) < 5.0 * kd.std() / n ** 0.5


def test_where_the_default_model_is_off():
    """The documented case: confocal readout (README example 3) with
    leak = 1e-4. The default model gives 0.722 and the jump model
    0.752; a simulation of the jump process (400 000 shots per state)
    lands within 5 standard errors of the jump model and more than 20
    from the default (measured: 1.1 and 40)."""
    case = (0.002, 2244.0, GAMMA0, 50e-6, 10.0, 1e-4, 4e3, None)
    eta, lam, gamma, tau, s, leak, noise, _ = case
    old = vs.fidelity(eta, lam, gamma, tau, s, leak, noise)
    new = vs.fidelity(eta, lam, gamma, tau, s, leak, noise,
                      repeated_flips=True)
    assert round(old["fidelity"], 3) == 0.722
    assert round(new["fidelity"], 3) == 0.752
    rng = np.random.default_rng(5)
    n = 400_000
    kb = _simulate(case, n, True, rng)
    kd = _simulate(case, n, False, rng)
    se = math.sqrt(0.25 / n)

    def f_sim(thr):
        return 1.0 - 0.5 * (np.mean(kd >= thr) + np.mean(kb < thr))

    assert abs(f_sim(new["threshold"]) - new["fidelity"]) < 5.0 * se
    assert abs(f_sim(old["threshold"]) - old["fidelity"]) > 20.0 * se


def test_relaxation_makes_a_peak():
    """Without background or leak the fidelity never falls and
    `optimal_window` refuses. With T1 = 50 us the dark spin relaxes into
    the bright state, so the fidelity peaks; no window of a 400-point
    scan does better (beyond 1e-12)."""
    kw = dict(eta=0.002, lam=2244.0, gamma=GAMMA0, s=10.0)
    with pytest.raises(ValueError, match="no peak"):
        vs.optimal_window(**kw, tau_max=1e-3)
    out = vs.optimal_window(**kw, tau_max=1e-3, repeated_flips=True,
                            t1_s=50e-6)
    assert 1e-6 < out["tau"] < 1e-3
    scan = [vs.fidelity(kw["eta"], kw["lam"], kw["gamma"], t, kw["s"],
                        repeated_flips=True, t1_s=50e-6)["fidelity"]
            for t in np.geomspace(1e-7, 1e-3, 400)]
    assert max(scan) <= out["fidelity"] + 1e-12


def test_search_functions_use_the_jump_model():
    """`required_window` and `required_efficiency` with
    repeated_flips=True return values that reach the target in the jump
    model, while 0.1 % less does not."""
    kw = dict(lam=2244.0, gamma=GAMMA0, s=10.0, leak=1e-4, noise_rate=4e3,
              repeated_flips=True, t1_s=1e-3)

    def f(eta, tau):
        return vs.fidelity(eta, kw["lam"], kw["gamma"], tau, kw["s"],
                           kw["leak"], kw["noise_rate"], repeated_flips=True,
                           t1_s=1e-3)["fidelity"]

    tau = vs.required_window(0.7, eta=0.002, **kw)
    assert f(0.002, tau) >= 0.7 > f(0.002, 0.999 * tau)
    eta = vs.required_efficiency(0.9, tau=20e-6, **kw)
    assert f(eta, 20e-6) >= 0.9 > f(0.999 * eta, 20e-6)


def test_jump_model_refusals():
    args = (0.3, 500.0, GAMMA0, 2e-6)
    with pytest.raises(ValueError, match="needs repeated_flips"):
        vs.fidelity(*args, t1_s=1e-3)
    for bad in (0.0, -1.0, np.nan, np.inf, "1e-3", True):
        with pytest.raises(ValueError, match="t1_s"):
            vs.fidelity(*args, repeated_flips=True, t1_s=bad)
    with pytest.raises(ValueError, match="repeated_flips"):
        vs.fidelity(*args, repeated_flips=1)
    with pytest.raises(ValueError, match="too wide"):
        vs.fidelity(0.5, 100.0, GAMMA0, 1e-3, noise_rate=1e9,
                    repeated_flips=True)
    with pytest.raises(ValueError, match="needs repeated_flips"):
        vs.optimal_window(0.3, 500.0, GAMMA0, noise_rate=1e3, t1_s=1e-3)
    with pytest.raises(ValueError, match="needs repeated_flips"):
        vs.required_window(0.9, 0.3, 500.0, GAMMA0, t1_s=1e-3)
    with pytest.raises(ValueError, match="needs repeated_flips"):
        vs.required_efficiency(0.9, 500.0, GAMMA0, 1e-6, t1_s=1e-3)


def test_given_threshold_in_the_jump_model():
    """A given threshold returns the errors of the same distributions,
    and the best threshold is the best of a scan (to 1e-12)."""
    eta, lam, gamma, tau, s, leak, noise, t1 = CASES[1]
    best = vs.fidelity(eta, lam, gamma, tau, s, leak, noise,
                       repeated_flips=True, t1_s=t1)
    pb, pd = _jump_count_distributions(eta, lam, gamma, tau, s, leak, noise,
                                       t1)
    scan = []
    for thr in range(1, 40):
        out = vs.fidelity(eta, lam, gamma, tau, s, leak, noise, threshold=thr,
                          repeated_flips=True, t1_s=t1)
        assert out["error_bright"] == pytest.approx(pb[:thr].sum(), abs=1e-12)
        assert out["error_dark"] == pytest.approx(1.0 - pd[:thr].sum(),
                                                  abs=1e-12)
        scan.append(out["fidelity"])
    assert max(scan) == pytest.approx(best["fidelity"], abs=1e-12)
    assert int(np.argmax(scan)) + 1 == best["threshold"]
    far = vs.fidelity(eta, lam, gamma, tau, s, leak, noise, threshold=10 ** 5,
                      repeated_flips=True, t1_s=t1)
    assert far["error_dark"] == 0.0
    assert far["error_bright"] == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------- the four-level model with T1

P = vs.snv_rosenthal2023()


def _rk4_four_level(cl, gamma, eta, s, tau, n_max, dt, relax):
    """Photon-number-resolved master equation of the four-level model
    with ground-state relaxation, written here from the module
    docstring: counted jumps eta J, uncounted (1 - eta) J, and the
    relaxation jumps |dn> <-> |up> at rate `relax` each way (never
    counted). RK4 on 4 x 4 matrices."""
    S = cl.strengths
    two_pi = 2.0 * np.pi
    om0 = np.sqrt(s * gamma * gamma / 2.0)
    H = np.zeros((4, 4), dtype=complex)
    H[1, 1] = two_pi * cl.w_q_ghz
    H[3, 3] = two_pi * cl.w_e_ghz
    for g in range(2):
        for e in range(2):
            H[2 + e, g] = H[g, 2 + e] = 0.5 * om0 * np.sqrt(S[g, e] / S[0, 0])
    ops = []
    for e in range(2):
        for g in range(2):
            c = np.zeros((4, 4))
            c[g, 2 + e] = np.sqrt(gamma * S[g, e] / S[:, e].sum())
            ops.append(c)
    rel = []
    for a, b in ((0, 1), (1, 0)):
        c = np.zeros((4, 4))
        c[b, a] = np.sqrt(relax)
        rel.append(c)
    cdc = sum(c.T @ c for c in ops + rel)

    def rhs(r):
        jumped = sum(c @ r @ c.T for c in ops)
        out = (-1j * (H @ r - r @ H) - 0.5 * (cdc @ r + r @ cdc)
               + (1.0 - eta) * jumped + sum(c @ r @ c.T for c in rel))
        out[:, 1:] += eta * jumped[:, :-1]
        return out

    r = np.zeros((2, n_max + 1, 4, 4), dtype=complex)
    r[0, 0, 0, 0] = 1.0
    r[1, 0, 1, 1] = 1.0
    for _ in range(int(round(tau / dt))):
        k1 = rhs(r)
        k2 = rhs(r + 0.5 * dt * k1)
        k3 = rhs(r + 0.5 * dt * k2)
        k4 = rhs(r + dt * k3)
        r = r + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    p = np.real(np.trace(r, axis1=2, axis2=3))
    return p[0], p[1]


def test_four_level_relaxation_against_time_integration():
    """At the study's C line, a 20 ns window and an exaggerated
    T1 = 40 ns (so that relaxation matters in the window): the
    generating-function result equals the RK4 integration written here
    (to 1e-7; measured 8e-15 with steps of 5 ps)."""
    b = vs.field_from_axis(P, 0.1118, 9.0, azimuth_deg=270.0)
    cl = vs.c_line(P, b)
    gamma, eta, s, tau, t1 = 10.5 * GAMMA0, 0.32, 1.0, 20e-9, 40e-9
    p_b, p_d = vs.four_level_counts(cl, gamma, eta, tau, s=s, t1_s=t1)
    q_b, q_d = _rk4_four_level(cl, gamma * 1e-9, eta, s, tau * 1e9, 24,
                               0.005, 0.5 / (t1 * 1e9))
    assert np.max(np.abs(p_b[:20] - q_b[:20])) < 1e-7
    assert np.max(np.abs(p_d[:20] - q_d[:20])) < 1e-7
    # relaxation visibly changes the dark spin's counts here (by 0.098)
    p_d0 = vs.four_level_counts(cl, gamma, eta, tau, s=s)[1]
    assert np.max(np.abs(p_d - p_d0)) > 1e-2


def test_four_level_relaxation_matches_the_two_state_jump_model():
    """A bright line with cyclicity 40 and a dark spin that scatters
    almost nothing (its lines weakened to 1e-9 and 30 GHz away), with
    T1 = 2 us: the mean counts of both spin states agree with the
    two-state jump model (bright within 1e-3, dark within 1e-2 at 1 us
    and 2e-3 at 3 and 20 us; measured 5e-4, 6e-3, 1.1e-3 and 2.6e-4).
    The rest is the start-up of the optical cycle, which the rate model
    leaves out."""
    lam = 40.0
    cl = CLine(strengths=np.array([[1.0, 1e-9], [1.0 / lam, 1e-9]]),
               w_q_ghz=30.0, w_e_ghz=31.0, lam_a=lam, lam_b=1.0,
               delta_g_ghz=900.0)
    gamma, eta, s, t1 = 1.0 / 4.5e-9, 0.3, 1.0, 2e-6
    for tau, tol_d in ((1e-6, 1e-2), (3e-6, 2e-3), (20e-6, 2e-3)):
        p_b, p_d = vs.four_level_counts(cl, gamma, eta, tau, s=s, t1_s=t1)
        k = np.arange(p_b.size)
        two = vs.fidelity(eta, lam, gamma, tau, s=s, repeated_flips=True,
                          t1_s=t1)
        assert (k * p_b).sum() == pytest.approx(two["n_bright_total"],
                                                rel=1e-3)
        assert (k * p_d).sum() == pytest.approx(two["n_dark_total"],
                                                rel=tol_d)


def test_four_level_without_relaxation_is_unchanged():
    b = vs.field_from_axis(P, 0.1118, 9.0, azimuth_deg=270.0)
    cl = vs.c_line(P, b)
    a = vs.four_level_counts(cl, 10.5 * GAMMA0, 0.32, 50e-9)
    c = vs.four_level_counts(cl, 10.5 * GAMMA0, 0.32, 50e-9, t1_s=None)
    assert np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1])
    for bad in (0.0, -1e-6, np.nan, np.inf, "1e-3", True):
        with pytest.raises(ValueError, match="t1_s"):
            vs.four_level_counts(cl, 10.5 * GAMMA0, 0.32, 50e-9, t1_s=bad)
    with pytest.raises(ValueError, match="t1_s"):
        vs.fidelity_four_level(cl, 10.5 * GAMMA0, 0.32, 50e-9, t1_s=-1.0)


# ------------------------------------------------ the count range

def test_real_generating_function():
    """log G(x) for real x > 1, used for the count range, against the
    matrix exponential of the 2 x 2 generator by Taylor scaling and
    squaring (vacspin.fourlevel._expm_batch, a separate routine; to
    1e-12 relative; measured 5e-15), against the Poisson closed form
    (x - 1) c tau when both states count at the same rate (to 1e-12;
    measured 1.1e-14), and log G = 0 for a dark spin that can never turn
    bright (to 1e-12; measured 3e-14)."""
    from vacspin.fourlevel import _expm_batch
    from vacspin.readout import _jump_log_g_real, _jump_rates
    for case in CASES[:2] + CASES[3:]:
        eta, lam, gamma, tau, s, leak, noise, t1 = case
        c_b, c_d, k_bd, k_db = _jump_rates(eta, lam, gamma, s, leak, t1)
        x = np.array([1.0, 1.0001, 1.01, 1.3, 2.0])
        log_b, log_d = _jump_log_g_real(c_b, c_d, k_bd, k_db, tau, x)
        for i, xi in enumerate(x):
            u = (xi - 1.0) * tau
            a = np.array([[c_b * u - k_bd * tau, k_db * tau],
                          [k_bd * tau, c_d * u - k_db * tau]], dtype=complex)
            col = _expm_batch(a[None])[0].real.sum(axis=0)
            assert log_b[i] == pytest.approx(math.log(col[0]), rel=1e-12,
                                             abs=1e-12)
            assert log_d[i] == pytest.approx(math.log(col[1]), rel=1e-12,
                                             abs=1e-12)
    x = 1.0 + np.geomspace(1e-6, 1e6, 241)
    c_b, c_d, k_bd, k_db = _jump_rates(0.3, 500.0, GAMMA0, 1.0, 1.0, 1e-6)
    log_b, log_d = _jump_log_g_real(c_b, c_d, k_bd, k_db, 2e-6, x)
    want = (x - 1.0) * c_b * 2e-6
    assert np.max(np.abs(log_b - want) / np.maximum(want, 1.0)) < 1e-12
    assert np.max(np.abs(log_d - want) / np.maximum(want, 1.0)) < 1e-12
    c_b, c_d, k_bd, k_db = _jump_rates(0.3, 500.0, GAMMA0, 1.0, 0.0, None)
    _, log_d = _jump_log_g_real(c_b, c_d, k_bd, k_db, 2e-6, x)
    assert np.max(np.abs(log_d)) < 1e-12


@pytest.mark.parametrize("case", CASES + [
    (0.6, 16000.0, 7.26 * GAMMA0, 1e-5, 2.0, 3.7e-3, 1e3, None),
    (0.002, 2244.0, GAMMA0, 50e-6, 10.0, 1e-2, 4e3, None)])
def test_count_range_holds_the_distributions(case):
    """With four times the chosen count range, less than 1e-10 of either
    distribution lies beyond it (measured 8e-13 at most), and the counts
    inside do not change by more than 1e-12 (measured 1.2e-15)."""
    pb, pd = _jump_count_distributions(*case)
    n = pb.size
    qb, qd = _jump_count_distributions(*case, n=4 * n)
    assert qb[n:].sum() < 1e-10 and qd[n:].sum() < 1e-10
    assert np.max(np.abs(pb - qb[:n])) < 1e-12
    assert np.max(np.abs(pd - qd[:n])) < 1e-12


def test_long_window_with_many_counts():
    """The cavity design point with a 1 ms window: the bright spin emits
    about 9600 counts before it turns dark for good. Without leak the
    jump model must equal the default model; it does with 2^18 counts
    (to 1e-12; measured 1.3e-16)."""
    args = (0.6, 16000.0, 7.26 * GAMMA0, 1e-3, 2.0, 0.0, 1e3)
    pb, pd = _jump_count_distributions(*args, None)
    qb, qd = _count_distributions(*args, pb.size - 1)
    assert pb.size == 1 << 18
    assert np.max(np.abs(pb - qb)) < 1e-12
    assert np.max(np.abs(pd - qd)) < 1e-12
