"""0.9.0: the four-level readout model and the D-line term of the cavity
budget, checked against closed forms, a second algorithm, the two-state
model, and the revised source study.

Source study: T. M. Mahim, M. M. Rahman and A. S. M. Mohsin, "Fast
single-shot readout of tin-vacancy spins with an overcoupled diamond
nanocavity on thin-film lithium niobate", revised manuscript (submitted
to Optics Express, 2026); code Tanvir-Mahmud-Mahim/snv-tfln-cavity-
interface (commit c83302e), data doi:10.5281/zenodo.23507013.

The "study code" values below were computed by running that commit's
sim/readout_fcs.py (with its own qutip-based Hamiltonian, sim/snv.py)
at the same field, decay rate and efficiency; they are recorded here as
numbers, so the study code is not needed to run the tests.
"""
import numpy as np
import pytest

import vacspin as vs
from vacspin.fourlevel import CLine

P = vs.snv_rosenthal2023()
BUD = vs.snv_emission()


def _axis_dir(misalign_deg):
    th, ph = P.theta_rad, P.phi_rad
    mu = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                   np.cos(th)])
    e1 = np.cross(mu, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    m = np.deg2rad(misalign_deg)
    return np.cos(m) * mu + np.sin(m) * e1


def _reference_field():
    """The study's reference field: 9 deg from the spin axis, magnitude
    where the model qubit frequency is the measured 3.677 GHz."""
    d = _axis_dir(9.0)
    lo, hi = 0.05, 0.2
    for _ in range(100):
        m = 0.5 * (lo + hi)
        if vs.qubit_frequency(P, m * d) < 3.677:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi) * d


B_REF = _reference_field()
CL = vs.c_line(P, B_REF)
# the study's decay rate and efficiency at its design point
# (cavity_budget of the study code, Q_L = 870, F_max = 82)
G_STUDY = 10.500891084981417 * BUD.gamma0
ETA_STUDY = 0.31950812062340844


# ------------------------------------------------------- the C line

def test_c_line_at_the_reference_field():
    """Study code: B = 0.111795 T, w_q = 3.677, w_e = 4.40413 GHz,
    A1-B2 detuning 0.72713 GHz, Lambda_A = 2179.68, Lambda_B = 2132.06,
    ground splitting 903.489 GHz. vacspin's Hamiltonian is a separate
    implementation of the same model (Rosenthal et al., PRX 13, 031022,
    Appendix B)."""
    assert np.linalg.norm(B_REF) == pytest.approx(0.111795000923, rel=1e-9)
    assert CL.w_q_ghz == pytest.approx(3.677, abs=1e-9)
    assert CL.w_e_ghz == pytest.approx(4.404128843496, rel=1e-9)
    assert CL.detuning_b2_ghz == pytest.approx(0.727128843496, rel=1e-9)
    assert CL.lam_a == pytest.approx(2179.682961686, rel=1e-9)
    assert CL.lam_b == pytest.approx(2132.061429607, rel=1e-9)
    assert CL.delta_g_ghz == pytest.approx(903.488802776, rel=1e-9)
    # Lambda_A is the cyclicity vacspin.cyclicity reports
    assert CL.lam_a == pytest.approx(vs.cyclicity(P, B_REF), rel=1e-12)


# ------------------------------------------------ closed-form limits

def _two_level_line(w_e=5.0):
    """A1 only from |dn>; |up> couples only to |B> (no spin flips)."""
    return CLine(strengths=np.array([[1.0, 0.0], [0.0, 1.0]]),
                 w_q_ghz=2.0, w_e_ghz=w_e, lam_a=np.inf, lam_b=np.inf,
                 delta_g_ghz=900.0)


@pytest.mark.parametrize("s", [0.3, 1.0, 4.0])
def test_bright_mean_is_the_torrey_integral(s):
    """A resonantly driven two-level emitter starting in the ground
    state (Torrey's solution, no dephasing): with Omega^2 = s gamma^2/2,
    rho_ee(t) = s / (2 (1 + s)) [1 - exp(-3 gamma t / 4)
    (cos mu t + (3 gamma / (4 mu)) sin mu t)], mu^2 = Omega^2 - gamma^2/16.
    The mean count is eta gamma times its integral, done here by the
    trapezoid rule on 400001 points."""
    gamma, eta, tau = 1.0 / 4.5e-9, 0.37, 30e-9
    om2 = s * gamma ** 2 / 2.0
    mu = np.sqrt(om2 - gamma ** 2 / 16.0)
    t = np.linspace(0.0, tau, 400001)
    rho = s / (2.0 * (1.0 + s)) * (1.0 - np.exp(-0.75 * gamma * t) * (
        np.cos(mu * t) + 0.75 * gamma / mu * np.sin(mu * t)))
    mean = eta * gamma * np.sum((rho[1:] + rho[:-1]) * 0.5 * np.diff(t))
    p_b, p_d = vs.four_level_counts(_two_level_line(), gamma, eta, tau, s=s)
    k = np.arange(p_b.size)
    assert (k * p_b).sum() == pytest.approx(mean, rel=1e-8)


def test_critically_damped_drive():
    """At s = 1/8 (Omega = gamma / 4) the two-level solution is
    critically damped: rho_ee(t) = s / (2 (1 + s)) [1 - exp(-3 gamma t
    / 4) (1 + 3 gamma t / 4)]. There the eigenvectors of the generator
    are nearly parallel, so this also checks the fallback to the Taylor
    propagator (mean count to 1e-8)."""
    gamma, eta, tau, s = 1.0 / 4.5e-9, 0.37, 30e-9, 0.125
    t = np.linspace(0.0, tau, 400001)
    x = 0.75 * gamma * t
    rho = s / (2.0 * (1.0 + s)) * (1.0 - np.exp(-x) * (1.0 + x))
    mean = eta * gamma * np.sum((rho[1:] + rho[:-1]) * 0.5 * np.diff(t))
    p_b, _ = vs.four_level_counts(_two_level_line(), gamma, eta, tau, s=s)
    assert (np.arange(p_b.size) * p_b).sum() == pytest.approx(mean, rel=1e-8)


def test_dark_rate_is_the_detuned_lorentzian():
    """The dark spin here is a two-level emitter driven off resonance,
    detuned by w_e - w_q: its steady emission rate is
    R = (gamma/2) s / (1 + s + (2 Delta / gamma)^2) (Delta in rad/s), so
    the mean count over a long window is eta R tau up to a start-up
    transient of order 1 / gamma (less than 1e-3 here)."""
    gamma, eta, tau, s = 2.3e9, 0.5, 2e-6, 1.0
    cl = _two_level_line(w_e=2.73)                 # Delta = 0.73 GHz
    delta = 2.0 * np.pi * 0.73e9
    r = 0.5 * gamma * s / (1.0 + s + (2.0 * delta / gamma) ** 2)
    _, p_d = vs.four_level_counts(cl, gamma, eta, tau, s=s)
    k = np.arange(p_d.size)
    assert (k * p_d).sum() == pytest.approx(eta * r * tau, rel=1e-3)


def test_noise_only_dark_spin_is_poisson():
    """With no light at all from the dark spin (no strength from |up>)
    the dark count is the Poisson background alone, to 1e-12."""
    cl = CLine(strengths=np.array([[1.0, 0.5], [0.0, 0.0]]),
               w_q_ghz=3.0, w_e_ghz=4.0, lam_a=np.inf, lam_b=0.0,
               delta_g_ghz=900.0)
    noise, tau = 3.0e7, 100e-9
    _, p_d = vs.four_level_counts(cl, 2e9, 0.3, tau, noise_rate=noise)
    m = noise * tau
    k = np.arange(30)
    poisson = np.exp(-m) * m ** k / np.cumprod(np.r_[1.0, np.arange(1, 30)])
    assert np.max(np.abs(p_d[:30] - poisson)) < 1e-12


# ------------------------------------------- an independent algorithm

def _rk4_count_distribution(cl, gamma, eta, s, tau, n_max, dt):
    """P(k) for the bright and dark spin from the photon-number-resolved
    master equation, d rho_n / dt = L0' rho_n + eta J rho_(n-1), where L0'
    holds the Hamiltonian, the anticommutators and the UNDETECTED jumps
    (1 - eta) J. Integrated by fixed-step RK4 on 4 x 4 matrices, written
    here independently of vacspin.fourlevel (no superoperators, no
    Fourier transform)."""
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
    cdc = sum(c.T @ c for c in ops)

    def rhs(r):                   # r: (start, n_max + 1, 4, 4)
        jumped = sum(c @ r @ c.T for c in ops)
        out = (-1j * (H @ r - r @ H) - 0.5 * (cdc @ r + r @ cdc)
               + (1.0 - eta) * jumped)
        out[:, 1:] += eta * jumped[:, :-1]
        return out

    r = np.zeros((2, n_max + 1, 4, 4), dtype=complex)
    r[0, 0, 0, 0] = 1.0                           # start in |dn>
    r[1, 0, 1, 1] = 1.0                           # start in |up>
    for _ in range(int(round(tau / dt))):
        k1 = rhs(r)
        k2 = rhs(r + 0.5 * dt * k1)
        k3 = rhs(r + 0.5 * dt * k2)
        k4 = rhs(r + dt * k3)
        r = r + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    p = np.real(np.trace(r, axis1=2, axis2=3))
    return p[0], p[1]


def test_full_counting_statistics_against_time_integration():
    """At the reference C line (all four lines, real strengths), a
    20 ns window, gamma / 2 pi = 0.37 GHz, eta = 0.32, s = 1: the two
    algorithms give the same count distributions to 1e-7."""
    gamma, eta, s, tau = G_STUDY, ETA_STUDY, 1.0, 20e-9
    p_b, p_d = vs.four_level_counts(CL, gamma, eta, tau, s=s)
    q_b, q_d = _rk4_count_distribution(CL, gamma * 1e-9, eta, s,
                                       tau * 1e9, n_max=24, dt=0.002)
    assert np.max(np.abs(p_b[:20] - q_b[:20])) < 1e-7
    assert np.max(np.abs(p_d[:20] - q_d[:20])) < 1e-7
    assert abs(q_b.sum() - 1.0) < 1e-6 and abs(q_d.sum() - 1.0) < 1e-6


# ---------------------------------------- the two-state model's regime

def test_bright_mean_matches_the_two_state_model():
    """With a dark spin that scatters almost nothing (its lines weakened
    to 1e-9 and 30 GHz away) and a small cyclicity (Lambda = 40), the
    bright record is the flip-terminated count of the two-state model,
    eta (Lambda + 1) (1 - exp(-Gp tau)), Gp = R / (Lambda + 1). The
    two-state model treats the emissions as a Poisson process; a driven
    emitter emits more regularly (antibunching), which shifts the mean
    count before the record ends by 0.11 % at 1 us and 0.05 % at 3 us.
    Once the spin has surely flipped (20 us, Gp tau = 27) both give
    eta (Lambda + 1) = 12.3, to 2e-5."""
    lam = 40.0
    cl = CLine(strengths=np.array([[1.0, 1e-9], [1.0 / lam, 1e-9]]),
               w_q_ghz=30.0, w_e_ghz=31.0, lam_a=lam, lam_b=1.0,
               delta_g_ghz=900.0)
    gamma, eta, s = 1.0 / 4.5e-9, 0.3, 1.0
    for tau, rel in ((1e-6, 2e-3), (3e-6, 1e-3), (20e-6, 1e-4)):
        p_b, _ = vs.four_level_counts(cl, gamma, eta, tau, s=s)
        k = np.arange(p_b.size)
        n_b, n_d = vs.readout_counts(eta, lam, gamma, tau, s=s)
        assert (k * p_b).sum() == pytest.approx(n_b - n_d, rel=rel)
    assert (k * p_b).sum() == pytest.approx(eta * (lam + 1.0), rel=1e-4)


# ------------------------------------- the revised source study

# the study code's default detector dark-count rate, 1e-7 per ns
DARK_COUNTS = 100.0


def test_reference_readout_matches_the_study_code():
    """Study code at the reference point (gamma = 10.5009 gamma_0,
    eta = 0.31951, s = 1, 100 ns, f0 = 0.99, dark counts 100 per s):
    F = 0.9882343103653719, threshold 8, n_b = 18.306682249296514,
    n_d = 2.184252342789895. Paper: 98.8 %, n_b about 18.5 and n_d
    about 2.22 within 100 ns (its loaded-cavity numbers come from the
    FDTD mode volume, which this test replaces by F_max = 82)."""
    out = vs.fidelity_four_level(CL, G_STUDY, ETA_STUDY, 100e-9, s=1.0,
                                 f0=0.99, noise_rate=DARK_COUNTS)
    assert out["fidelity"] == pytest.approx(0.9882343103653719, abs=1e-10)
    assert out["threshold"] == 8
    assert out["n_bright"] == pytest.approx(18.306682249296514, rel=1e-8)
    assert out["n_dark"] == pytest.approx(2.184252342789895, rel=1e-8)
    assert round(out["fidelity"], 3) == 0.988                  # paper


def test_optimal_window_matches_the_study_code():
    """Study code: the best window of its grid is 508.3 ns at s = 0.1,
    F = 0.9901299490857048 (paper: 99.0 % at the optimal window of
    508 ns)."""
    out = vs.fidelity_four_level(CL, G_STUDY, ETA_STUDY,
                                 508.2967532431553e-9, s=0.1, f0=0.99,
                                 noise_rate=DARK_COUNTS)
    assert out["fidelity"] == pytest.approx(0.9901299490857048, abs=1e-10)
    assert out["threshold"] == 6
    assert round(out["fidelity"], 3) == 0.990                  # paper


@pytest.mark.parametrize("xlw, sd, bg, f_study, nb, nd, tol", [
    (0.1e9, 0.0, 0.0, 0.9845002935212213, 18.336646691658, 3.278055191953408,
     1e-9),
    (0.0, 0.0, 5e7, 0.9775800750535012, 23.30667224904181,
     7.184242342748542, 1e-9),
    (0.0, 0.2e9, 0.0, 0.9830109561579788, 16.832142484510705,
     2.250875030809462, 1e-5),
    (0.1e9, 0.1e9, 0.0, 0.9835406208992096, 18.05838674319274,
     3.2986010422879164, 1e-5),
])
def test_broadening_and_background_match_the_study_code(xlw, sd, bg,
                                                         f_study, nb, nd,
                                                         tol):
    """Extra homogeneous width, Poisson background and quasi-static
    spectral diffusion, each at 100 ns and s = 1, against the study code
    (run with its dark-count rate set to the background given here).
    Without spectral diffusion the two agree to 1e-9. With it they agree
    to 1e-5: the study code averages over 9 Gauss-Hermite points, this
    package on a grid refined until converged (test below)."""
    out = vs.fidelity_four_level(CL, G_STUDY, ETA_STUDY, 100e-9, s=1.0,
                                 f0=0.99, extra_width_hz=xlw,
                                 spectral_diffusion_hz=sd, noise_rate=bg)
    assert out["fidelity"] == pytest.approx(f_study, abs=tol)
    assert out["n_bright"] == pytest.approx(nb, rel=tol)
    assert out["n_dark"] == pytest.approx(nd, rel=tol)


def test_spectral_diffusion_average_is_converged():
    """The spectral-diffusion average against a dense average done here
    by hand: 241 fixed laser detunings over +-6 standard deviations
    (steps of 21 MHz, against a 0.37 GHz line) with Gaussian weights, in
    a 30 ns window to keep it quick. Agreement to 2e-7 at 1 GHz FWHM,
    where a fixed 9-point Gauss-Hermite rule is off by more than 1e-4."""
    tau, s = 30e-9, 1.0
    for fwhm in (1.0e9,):
        sig = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        x = np.linspace(-6.0, 6.0, 241)
        w = np.exp(-0.5 * x ** 2)
        w /= w.sum()
        ref_b = ref_d = 0.0
        for xi, wi in zip(x, w):
            b, d = vs.four_level_counts(CL, G_STUDY, ETA_STUDY, tau, s=s,
                                        laser_detuning_hz=sig * xi)
            ref_b = ref_b + wi * b
            ref_d = ref_d + wi * d
        p_b, p_d = vs.four_level_counts(CL, G_STUDY, ETA_STUDY, tau, s=s,
                                        spectral_diffusion_hz=fwhm)
        assert np.max(np.abs(p_b - ref_b)) < 2e-7
        assert np.max(np.abs(p_d - ref_d)) < 2e-7
        gh_x, gh_w = np.polynomial.hermite_e.hermegauss(9)
        gh_w = gh_w / gh_w.sum()
        gh_b = sum(wi * vs.four_level_counts(
            CL, G_STUDY, ETA_STUDY, tau, s=s,
            laser_detuning_hz=sig * xi)[0] for xi, wi in zip(gh_x, gh_w))
        assert np.max(np.abs(gh_b - ref_b)) > 1e-4


def test_confocal_readout_of_the_source_experiment():
    """Paper, Sec. 2.5: with the confocal parameters of Rosenthal et al.
    (no cavity, eta = 0.2 %, background 0.2 counts per 50 us, same field)
    the model gives n_b = 3.8 and n_d = 0.22 in 50 us and F = 83.3 % at
    the best drive; for eta = 0.4 %, 89.8 %."""
    noise = 0.2 / 50e-6
    for eta, f_paper in ((0.002, 0.833), (0.004, 0.898)):
        outs = [vs.fidelity_four_level(CL, BUD.gamma0, eta, 50e-6, s=s,
                                       noise_rate=noise, f0=0.99)
                for s in (0.5, 1.0, 2.0, 5.0)]
        best = max(outs, key=lambda o: o["fidelity"])
        assert round(best["fidelity"], 3) == f_paper
        if eta == 0.002:
            assert round(best["n_bright"], 1) == 3.8
            assert round(best["n_dark"], 2) == 0.22


# ----------------------------------------------- the D-line budget

def _design(include_d_line):
    return vs.CavityInterface(
        q_loaded=870.0, v_rel=vs.purcell_max(870.0, 1.0) / 82.0,
        eta_wg=0.92, budget=BUD, lambda0=CL.lam_a,
        omega_q_hz=CL.w_q_ghz * 1e9, xi_pol=vs.xi_pol_overlap(),
        xi_pos=0.5, t_link=0.997, eta_chip=0.5, eta_det=0.77,
        delta_partner_hz=CL.delta_g_ghz * 1e9,
        include_d_line=include_d_line)


def test_d_line_term_is_eq1_of_the_study():
    """zeta = 1 + eta_q eta_DW [eta_BR (F_C - 1) + (1 - eta_BR)(F_D - 1)],
    beta = (zeta - 1) / zeta (the study's Eq. 1), with F_D the partner
    Purcell factor at the ground-state splitting. Study code (its cavity
    frequency is that of 618.9 nm, vacspin's of 619.0 nm):
    zeta = 10.500891, beta = 0.904770, eta = 0.319508; paper 10.5, 91 %,
    32 %, lifetime 4.5 -> 0.43 ns."""
    ci = _design(True)
    eq, edw, ebr = BUD.eta_q, BUD.eta_dw, BUD.eta_br
    zeta = 1.0 + eq * edw * (ebr * (ci.f_c - 1.0)
                             + (1.0 - ebr) * (ci.f_partner - 1.0))
    assert ci.zeta == pytest.approx(zeta, rel=1e-14)
    assert ci.beta == pytest.approx((zeta - 1.0) / zeta, rel=1e-14)
    assert ci.eta == pytest.approx(ci.beta * 0.92 * 0.997 * 0.5 * 0.77,
                                   rel=1e-14)
    assert ci.gamma_cav == pytest.approx(BUD.gamma0 * zeta, rel=1e-14)
    assert ci.zeta == pytest.approx(10.500891, rel=2e-5)
    assert ci.beta == pytest.approx(0.904770, rel=2e-5)
    assert ci.eta == pytest.approx(0.319508, rel=2e-5)
    assert round(ci.tau_cav_s * 1e9, 2) == 0.43
    # the broad cavity keeps the bare cyclicity (the study's assumption)
    assert ci.lambda_cav == pytest.approx(CL.lam_a, rel=2e-4)
    # g and the cooperativity stay those of the C transition
    off = _design(False)
    assert ci.g_hz == off.g_hz and ci.cooperativity == off.cooperativity
    assert ci.zeta - off.zeta == pytest.approx(ci.d_term, rel=1e-12)


def test_whole_chain_reproduces_the_paper():
    """Spin model -> cavity budget with the D line -> four-level readout,
    no number passed by hand: 98.8 % in 100 ns (paper 98.8 %)."""
    ci = _design(True)
    out = vs.fidelity_four_level(CL, ci.gamma_cav, ci.eta, 100e-9, s=1.0,
                                 f0=0.99)
    assert out["fidelity"] == pytest.approx(0.98823, abs=2e-5)


# (zeta, beta, eta, gamma_cav, lambda_cav, g_hz, cooperativity) printed
# by the released vacspin 0.8.1 for four designs and both beta forms
_V081 = [
    ((7.255824412808734, 0.8621796858486961, 0.6149611607958815,
      1612405425.0686078, 2244.112917300151, 7319747724.314404,
      6.255824412808734),
     (7.255824412808734, 0.909314233288442, 0.6485805054439777,
      1612405425.0686078, 2244.112917300151, 7517167281.459447,
      6.597824412808736)),
    ((10.306970306544116, 0.9029782787513146, 0.3188754016147145,
      2290437845.8986926, 2180.343520051614, 6768358876.128327,
      9.306970306544114),
     (10.306970306544116, 0.9361597074183651, 0.3305930050624822,
      2290437845.8986926, 2180.343520051614, 6891594232.514526,
      9.648970306544117)),
    ((720.5690413487284, 0.9986122079320419, 0.49930610396602093,
      160126453633.05078, 529.06802253107, 5551049182.887724,
      719.5690413487283),
     (720.5690413487284, 0.9990868328192835, 0.4995434164096417,
      160126453633.05078, 529.06802253107, 5552368189.951714,
      719.9110413487285)),
    ((11.053553441703855, 0.9095313552086238, 0.9004360416565376,
      2456345209.2675233, 500.0000007755102, 46396330968.49535,
      10.053553441703853),
     (11.053553441703855, 0.9404716317272743, 0.9310669154100015,
      2456345209.2675233, 500.0000007755102, 47178882533.227974,
      10.395553441703852)),
]
_DESIGNS = [
    dict(q_loaded=500.0, v_rel=0.68, eta_wg=1 - 500 / 1.39e6,
         lambda0=2244.0, omega_q_hz=3.7e9, xi_pol=2 / 3, xi_pos=0.5,
         t_link=0.991, eta_chip=0.8, eta_det=0.9),
    dict(q_loaded=870.0, v_rel=0.8, eta_wg=0.92, lambda0=2180.0,
         omega_q_hz=3.677e9, xi_pol=2 / 3, xi_pos=0.5, t_link=0.997,
         eta_chip=0.5, eta_det=0.77, delta_partner_hz=9.0349e11),
    dict(q_loaded=1.0e5, v_rel=1.3, eta_wg=0.5, lambda0=30.0,
         omega_q_hz=1.0e10, xi_pol=0.4, xi_pos=0.9,
         delta_partner_hz=2.0e12),
    dict(q_loaded=20.0, v_rel=0.05, eta_wg=0.99, lambda0=500.0,
         omega_q_hz=5e8, xi_pol=1.0, xi_pos=1.0),
]


def test_default_budget_equals_the_released_0_8_1():
    """With include_d_line left at False every budget number equals what
    the released 0.8.1 printed (stored above), exactly."""
    for d, (v_p, v_t) in zip(_DESIGNS, _V081):
        for bf, ref in (("purcell", v_p), ("transition", v_t)):
            ci = vs.CavityInterface(budget=BUD, beta_form=bf, **d)
            got = (ci.zeta, ci.beta, ci.eta, ci.gamma_cav, ci.lambda_cav,
                   float(ci.g_hz), float(ci.cooperativity))
            assert got == ref


def test_default_budget_is_unchanged():
    """include_d_line defaults to False and then gives exactly the
    budget without a D line (the D term is 0 and adds nothing)."""
    a = _design(False)
    b = vs.CavityInterface(
        q_loaded=870.0, v_rel=vs.purcell_max(870.0, 1.0) / 82.0,
        eta_wg=0.92, budget=BUD, lambda0=CL.lam_a,
        omega_q_hz=CL.w_q_ghz * 1e9, xi_pol=vs.xi_pol_overlap(),
        xi_pos=0.5, t_link=0.997, eta_chip=0.5, eta_det=0.77)
    for k in ("zeta", "beta", "eta", "gamma_cav", "lambda_cav", "g_hz"):
        assert getattr(a, k) == getattr(b, k)
    assert a.d_term == 0.0


# ------------------------------------------------------- refusals

def test_refusal_when_the_average_does_not_converge(monkeypatch):
    """With the grid capped at 17 points, a 1 GHz diffusion (which needs
    more) is refused rather than returned unconverged."""
    import vacspin.fourlevel as fl
    monkeypatch.setattr(fl, "_SD_MAX_POINTS", 17)
    with pytest.raises(ValueError, match="did not converge"):
        vs.four_level_counts(CL, G_STUDY, ETA_STUDY, 30e-9,
                             spectral_diffusion_hz=1e9)


def test_refusals():
    with pytest.raises(ValueError, match="needs delta_partner_hz"):
        vs.CavityInterface(q_loaded=870.0, v_rel=0.8, eta_wg=0.9,
                           budget=BUD, lambda0=2000.0, omega_q_hz=3.7e9,
                           xi_pol=0.6, include_d_line=True)
    with pytest.raises(ValueError, match="beta_form"):
        vs.CavityInterface(q_loaded=870.0, v_rel=0.8, eta_wg=0.9,
                           budget=BUD, lambda0=2000.0, omega_q_hz=3.7e9,
                           xi_pol=0.6, delta_partner_hz=9e11,
                           beta_form="transition", include_d_line=True)
    with pytest.raises(ValueError, match="True or False"):
        vs.CavityInterface(q_loaded=870.0, v_rel=0.8, eta_wg=0.9,
                           budget=BUD, lambda0=2000.0, omega_q_hz=3.7e9,
                           xi_pol=0.6, delta_partner_hz=9e11,
                           include_d_line=1)
    bad = [dict(gamma=0.0), dict(gamma=np.inf), dict(eta=1.2),
           dict(tau=-1e-9), dict(s=0.0), dict(noise_rate=-1.0),
           dict(extra_width_hz=np.nan), dict(spectral_diffusion_hz=-1.0),
           dict(laser_detuning_hz=np.inf)]
    base = dict(gamma=2e9, eta=0.3, tau=1e-7)
    for b in bad:
        with pytest.raises(ValueError):
            vs.four_level_counts(CL, **{**base, **b})
    with pytest.raises(ValueError, match="CLine"):
        vs.four_level_counts("not a line", 2e9, 0.3, 1e-7)
    with pytest.raises(ValueError, match="f0"):
        vs.fidelity_four_level(CL, 2e9, 0.3, 1e-7, f0=0.0)
    with pytest.raises(ValueError, match="threshold"):
        vs.fidelity_four_level(CL, 2e9, 0.3, 1e-7, threshold=0)
    with pytest.raises(ValueError, match="more than"):
        vs.four_level_counts(CL, 2e9, 1.0, 1e-3)    # ~2e6 counts
    with pytest.raises(ValueError, match="does not resolve"):
        vs.c_line(P, [0.0, 0.0, 0.0])



def test_given_threshold_and_errors():
    """A given threshold gives at most the best fidelity, and the two
    error probabilities rebuild the fidelity exactly."""
    best = vs.fidelity_four_level(CL, G_STUDY, ETA_STUDY, 100e-9, f0=0.99)
    for nr in (1, 5, best["threshold"], 12, 200):
        out = vs.fidelity_four_level(CL, G_STUDY, ETA_STUDY, 100e-9,
                                     f0=0.99, threshold=nr)
        assert out["fidelity"] <= best["fidelity"] + 1e-15
        f = 0.99 * (1 - 0.5 * (out["error_bright"] + out["error_dark"]))
        assert out["fidelity"] == pytest.approx(f + 0.005, abs=1e-15)
    same = vs.fidelity_four_level(CL, G_STUDY, ETA_STUDY, 100e-9, f0=0.99,
                                  threshold=best["threshold"])
    assert same["fidelity"] == best["fidelity"]


def test_field_from_axis():
    """The field helper: size and angle from the axis as asked, any
    azimuth; azimuth 270 is the study's misalignment plane (towards
    axis x lab z, the direction this file builds by hand); refusals."""
    th, ph = P.theta_rad, P.phi_rad
    mu = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                   np.cos(th)])
    for m, a in ((0.0, 0.0), (9.0, 270.0), (35.0, 45.0), (90.0, 180.0)):
        b = vs.field_from_axis(P, 0.15, m, a)
        assert np.linalg.norm(b) == pytest.approx(0.15, rel=1e-14)
        cosang = float(b @ mu) / 0.15
        assert np.degrees(np.arccos(np.clip(cosang, -1, 1))) == \
            pytest.approx(m, abs=1e-6)
    assert np.max(np.abs(vs.field_from_axis(P, 0.15, 9.0, 270.0)
                         - 0.15 * _axis_dir(9.0))) < 1e-15
    for bad in ((-0.1, 9.0, 0.0), (np.nan, 9.0, 0.0), (0.1, np.inf, 0.0)):
        with pytest.raises(ValueError):
            vs.field_from_axis(P, *bad)
