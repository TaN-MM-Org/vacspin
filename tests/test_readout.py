"""Readout-statistics anchors: exact Poisson and geometric limits of
the exact bright distribution (two independent code paths each), the
threshold-1 closed form recovered by the general optimizer, bounds
and monotonicity, the measured confocal operating point of
arXiv:2403.13110 reproduced by the shipped SnV- numbers, and the
inverse tools' round trips and refusals."""
import numpy as np
import pytest

from vacspin import (fidelity, fidelity_threshold1, geometric_pmf,
                     poisson_pmf, polarization_rate, readout_counts,
                     required_efficiency, required_window, snv_emission)
from vacspin.readout import _bright_pmf

GAMMA0 = snv_emission().gamma0                   # 1/(4.5 ns)


def test_pmfs_normalize_and_have_exact_means():
    for mean in (0.3, 4.0, 40.0):
        g = geometric_pmf(mean, 5000)
        assert abs(g.sum() - 1.0) < 1e-9
        assert abs((np.arange(g.size) * g).sum() - mean) < 1e-6 * (1 + mean)
        p = poisson_pmf(mean, 5000)
        assert abs(p.sum() - 1.0) < 1e-12
        assert abs((np.arange(p.size) * p).sum() - mean) < 1e-9 * (1 + mean)


def test_bright_pmf_poisson_limit():
    """With no spin flips (Lambda -> inf, Gp -> 0) the bright count is
    exactly Poisson with mean eta R tau."""
    eta, r, tau = 0.3, 2e7, 2e-6
    pmf = _bright_pmf(eta, r, gp=1e-9, tau=tau, kmax=100)
    want = poisson_pmf(eta * r * tau, 100)
    assert np.max(np.abs(pmf - want)) < 1e-10


def test_bright_pmf_geometric_limit():
    """Deep in the flip-terminated regime (Gp tau >> 1) the bright
    count is exactly geometric with mean eta R / Gp = eta (Lambda+1)
    -- the continuous model's exact limit, machine precision. The
    per-emission bookkeeping (mean eta Lambda) differs at order
    1/Lambda, and the same comparison shows exactly that gap."""
    eta, lam = 0.4, 30.0
    r = 5e7
    gp = r / (1.0 + lam)
    tau = 60.0 / gp                              # Gp tau = 60
    pmf = _bright_pmf(eta, r, gp, tau, kmax=400)
    want = geometric_pmf(eta * (lam + 1.0), 400)
    assert np.max(np.abs(pmf - want)) < 1e-10
    # the discrete-emission convention deviates at O(1/Lambda), no more
    off = np.max(np.abs(pmf - geometric_pmf(eta * lam, 400)))
    assert 1e-4 < off < 10.0 / lam ** 2


def test_threshold1_closed_form_and_optimizer_agree():
    """Where the optimal threshold IS 1, the general optimizer must
    land on the closed form; and the closed form itself obeys its
    exact zero-signal and perfect-separation limits."""
    assert fidelity_threshold1(0.0, 0.0) == 0.5
    assert abs(fidelity_threshold1(50.0, 0.0) - 1.0) < 1e-12
    out = fidelity(eta=0.05, lam=2244.0, gamma=GAMMA0, tau=50e-6,
                   s=2.0, leak=0.0, noise_rate=0.0)
    assert out["threshold"] == 1
    nb, nd = readout_counts(0.05, 2244.0, GAMMA0, 50e-6, s=2.0)
    assert abs(out["n_bright"] - nb) < 1e-9
    # with no dark channel the optimal-threshold fidelity is exactly
    # 1 - P(K_bright = 0)/2, with the exact zero-count probability
    r = 0.5 * GAMMA0 * 2.0 / 3.0
    gp = r / 2245.0
    p0 = _bright_pmf(0.05, r, gp, 50e-6, 50)[0]
    assert abs(out["fidelity"] - (1.0 - 0.5 * p0)) < 1e-9
    # and the mean-count closed form (Eq. A20) agrees to model accuracy
    assert abs(out["fidelity"] - fidelity_threshold1(nb, nd)) < 0.02


def test_fidelity_bounds_and_monotonicity_in_eta():
    vals = [fidelity(e, 500.0, GAMMA0, 10e-6, leak=1e-6,
                     noise_rate=1e3)["fidelity"]
            for e in (0.01, 0.05, 0.2, 0.6)]
    assert all(0.5 <= v <= 1.0 for v in vals)
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_confocal_operating_point_reproduced():
    """arXiv:2403.13110 confocal readout: eta ~ 0.2%, Lambda = 2244,
    tau = 50 us gives n_b ~ 4 detected photons (measured), n_d <= 0.2
    with the reported background. The mean-count model must land on
    the measured photon number."""
    nb, nd = readout_counts(0.002, 2244.0, GAMMA0, 50e-6, s=10.0,
                            noise_rate=0.2 / 50e-6)
    assert 3.0 < nb - nd < 5.0                   # measured ~ 4
    assert abs(nd - 0.2) < 1e-9
    fr = fidelity_threshold1(nb, nd, f0=0.935)
    assert 0.82 < fr < 0.92                      # measured 0.874


def test_polarization_rate_forms():
    """Gp = R/(1+Lambda) with the saturation form of R: exact at its
    limits (s -> inf gives gamma/2; detuning quenches it)."""
    g = 1e8
    assert abs(polarization_rate(g, 0.0, s=1e12) - 0.5 * g) < 1e-3 * g
    assert polarization_rate(g, 0.0, s=1.0, delta_over_gamma=100.0) \
        < 1e-4 * polarization_rate(g, 0.0, s=1.0)
    assert abs(polarization_rate(g, 9.0, s=1.0)
               - polarization_rate(g, 0.0, s=1.0) / 10.0) < 1e-12 * g


def test_required_efficiency_round_trip_and_refusal():
    kw = dict(lam=2244.0, gamma=GAMMA0 * 20, tau=1e-6, s=1.0,
              leak=1e-7, noise_rate=1e3)
    eta_min = required_efficiency(0.99, **kw)
    f = fidelity(eta_min, **kw)["fidelity"]
    assert f >= 0.99 and f < 0.9999
    assert fidelity(0.5 * eta_min, **kw)["fidelity"] < 0.99
    with pytest.raises(ValueError, match="unreachable"):
        required_efficiency(0.999, lam=3.0, gamma=GAMMA0, tau=1e-6)


def test_required_window_round_trip_and_refusal():
    kw = dict(eta=0.3, lam=2244.0, gamma=GAMMA0 * 20, s=1.0,
              leak=1e-7, noise_rate=1e3)
    tau_min = required_window(0.99, **kw)
    assert fidelity(kw["eta"], kw["lam"], kw["gamma"], tau_min,
                    kw["s"], kw["leak"], kw["noise_rate"])["fidelity"] \
        >= 0.99
    with pytest.raises(ValueError, match="unreachable"):
        required_window(0.9999, eta=0.001, lam=3.0, gamma=GAMMA0)


def test_input_refusals():
    with pytest.raises(ValueError):
        readout_counts(1.5, 100.0, GAMMA0, 1e-6)
    with pytest.raises(ValueError):
        readout_counts(0.5, -1.0, GAMMA0, 1e-6)
    with pytest.raises(ValueError):
        fidelity(0.5, 100.0, GAMMA0, 1e-6, f0=0.0)
    with pytest.raises(ValueError):
        fidelity(0.5, 100.0, GAMMA0, 1e-6, leak=-0.1)
    with pytest.raises(ValueError):
        fidelity_threshold1(-1.0, 0.0)
    with pytest.raises(ValueError):
        required_efficiency(0.4, 100.0, GAMMA0, 1e-6)


def test_readout_counts_refuses_negative_background():
    """readout_counts refuses a negative leak or noise rate, as
    fidelity does; before 0.3.1 it returned a negative dark count."""
    with pytest.raises(ValueError, match="noise_rate"):
        readout_counts(0.1, 100.0, GAMMA0, 1e-6, noise_rate=-1e6)
    with pytest.raises(ValueError, match="leak"):
        readout_counts(0.1, 100.0, GAMMA0, 1e-6, leak=-0.5)


def test_non_finite_rates_and_windows_refused():
    """An infinite window or a NaN rate is refused with a clear
    message; before 0.3.1 fidelity returned NaN for tau = inf and an
    unrelated integer-conversion error for gamma = NaN."""
    for kw in (dict(tau=np.inf), dict(gamma=np.nan), dict(s=np.inf)):
        args = dict(eta=0.1, lam=100.0, gamma=GAMMA0, tau=1e-6, s=1.0)
        args.update(kw)
        with pytest.raises(ValueError, match="finite"):
            fidelity(**args)
        with pytest.raises(ValueError, match="finite"):
            readout_counts(**args)
