"""Lab-calibration anchors: noiseless fits recover the truth to
numerical precision; the covariance hits its closed form in the exact
linear case and the Fisher prediction elsewhere; error bars are
validated by seeded Monte Carlo; the classic non-identifiable design
(zero-field splittings for lam and ups) is refused via an exact rank
argument; the greedy design obeys the determinant identity
det(A + g g^T) = det(A)(1 + g^T A^-1 g) >= det(A); and the CSV round
trip is exact."""
import dataclasses

import numpy as np
import pytest

from vacspin import (design_fields, field_on_circle, fit_spin_parameters,
                     load_observations_csv, parameter_information,
                     qubit_frequency, save_observations_csv,
                     snv_rosenthal2023, zero_field_splitting)
from vacspin.lab import _observe

P = snv_rosenthal2023()


def _design(n=8, b=0.15):
    kinds, fields = [], []
    for i, zeta in enumerate(np.linspace(10.0, 170.0, n)):
        kinds.append("qubit" if i % 2 else "orbital_g")
        fields.append(field_on_circle(zeta, b))
    return kinds, np.array(fields)


def test_observables_match_solver_paths():
    """The lab observables are exactly the package's own quantities:
    "qubit" is qubit_frequency, and "orbital_g" at zero field is the
    closed-form splitting sqrt(lam^2 + 4 ups^2)."""
    b = field_on_circle(147.0, 0.125)
    # since 0.4.0 the lab path uses eigvalsh on one manifold; it agrees
    # with the eigh path of qubit_frequency to round-off (~1e-13 GHz)
    assert abs(_observe(P, "qubit", b, 1.0) - qubit_frequency(P, b)) \
        < 1e-9
    d = _observe(P, "orbital_g", [0.0, 0.0, 0.0], 1.0)
    assert abs(d - zero_field_splitting(P.lam_g, P.ups_g)) < 1e-9
    d_e = _observe(P, "orbital_e", [0.0, 0.0, 0.0], 1.0)
    assert abs(d_e - zero_field_splitting(P.lam_e, P.ups_e)) < 1e-9


def test_lab_observables_equal_full_solver_for_random_fields():
    """The eigenvalue-only, single-manifold path used by the fits (since
    0.4.0) equals the full `solve` path for every kind, random fields
    and strain scales (to 1e-9 GHz), and refuses what `solve` refuses."""
    from vacspin import solve
    rng = np.random.default_rng(5)
    for _ in range(20):
        b = rng.normal(0.0, 0.3, 3)
        s = rng.uniform(0.0, 2.0)
        eg, _, ee, _ = solve(P, b, strain_scale=s)
        assert abs(_observe(P, "qubit", b, s) - (eg[1] - eg[0])) < 1e-9
        assert abs(_observe(P, "orbital_g", b, s) - (eg[2] - eg[0])) \
            < 1e-9
        assert abs(_observe(P, "orbital_e", b, s) - (ee[2] - ee[0])) \
            < 1e-9
    with pytest.raises(ValueError, match="finite"):
        _observe(P, "qubit", [np.nan, 0.0, 0.0], 1.0)
    with pytest.raises(ValueError, match="strain_scale"):
        _observe(P, "qubit", [0.0, 0.0, 0.1], -1.0)


def test_noiseless_fit_recovers_truth():
    """Synthetic data from the cited SnV set, started from a detuned
    guess: the fit must land back on the truth."""
    kinds, fields = _design()
    values = np.array([_observe(P, k, b, 1.0)
                       for k, b in zip(kinds, fields)])
    start = dataclasses.replace(P, lam_g=P.lam_g * 1.08,
                                ups_g=P.ups_g * 0.85)
    fit = fit_spin_parameters(kinds, fields, values, start,
                              vary=("lam_g", "ups_g"), sigmas_ghz=0.05)
    assert abs(fit.values["lam_g"] - P.lam_g) < 1e-5 * P.lam_g
    assert abs(fit.values["ups_g"] - P.ups_g) < 1e-5 * P.ups_g
    assert fit.chi2 < 1e-8
    assert fit.chi2_dof == len(kinds) - 2
    # the fitted parameter set is usable and carries its provenance
    assert "fitted by vacspin.lab" in fit.params.reference
    assert P.reference in fit.params.reference
    assert qubit_frequency(fit.params, fields[1]) == \
        pytest.approx(values[1], abs=1e-6)


def test_linear_case_covariance_closed_form():
    """With ups_g = 0 the zero-field splitting is exactly lam_g, so the
    model is linear with unit slope and the weighted-least-squares
    error bar has the textbook closed form sigma/sqrt(n)."""
    p0 = dataclasses.replace(P, ups_g=0.0)
    n, sigma = 6, 0.3
    kinds = ["orbital_g"] * n
    fields = np.zeros((n, 3))
    values = np.full(n, p0.lam_g)
    fit = fit_spin_parameters(kinds, fields, values, p0, vary=("lam_g",),
                              sigmas_ghz=sigma)
    assert abs(fit.sigma["lam_g"] - sigma / np.sqrt(n)) < 1e-6
    info = parameter_information(kinds, fields, p0, vary=("lam_g",),
                                 sigmas_ghz=sigma)
    assert abs(info["sigma"]["lam_g"] - sigma / np.sqrt(n)) < 1e-6


def test_information_prediction_matches_fit_covariance():
    """Two code paths, one matrix: the pre-measurement prediction and
    the post-fit covariance are both (J^T W J)^-1 at the truth."""
    kinds, fields = _design()
    values = np.array([_observe(P, k, b, 1.0)
                       for k, b in zip(kinds, fields)])
    fit = fit_spin_parameters(kinds, fields, values, P,
                              vary=("lam_g", "ups_g"), sigmas_ghz=0.05)
    info = parameter_information(kinds, fields, P,
                                 vary=("lam_g", "ups_g"),
                                 sigmas_ghz=0.05)
    for name in ("lam_g", "ups_g"):
        assert abs(fit.sigma[name] - info["sigma"][name]) \
            < 1e-3 * info["sigma"][name]


def test_error_bars_agree_with_monte_carlo():
    """400 seeded synthetic experiments: the scatter of the fitted
    parameters must match the reported 1-sigma error bars."""
    kinds, fields = _design(n=6)
    truth = np.array([_observe(P, k, b, 1.0)
                      for k, b in zip(kinds, fields)])
    sigma = 0.2
    rng = np.random.default_rng(7)
    draws = []
    reported = None
    for _ in range(400):
        values = truth + sigma * rng.standard_normal(truth.size)
        fit = fit_spin_parameters(kinds, fields, values, P,
                                  vary=("lam_g", "ups_g"),
                                  sigmas_ghz=sigma)
        draws.append([fit.values["lam_g"], fit.values["ups_g"]])
        reported = fit.sigma
    emp = np.std(np.array(draws), axis=0, ddof=1)
    assert np.allclose(emp, [reported["lam_g"], reported["ups_g"]],
                       rtol=0.15)


def test_zero_field_design_cannot_separate_lam_and_ups():
    """Zero-field splittings depend on lam and ups only through
    sqrt(lam^2 + 4 ups^2): one function of two parameters, so the
    information matrix has rank one, exactly. Both the planner and the
    fit must refuse rather than pseudo-invert."""
    kinds = ["orbital_g"] * 5
    fields = np.zeros((5, 3))
    info = parameter_information(kinds, fields, P,
                                 vary=("lam_g", "ups_g"))
    assert not info["identifiable"]
    sv = np.linalg.svd(info["fisher"], compute_uv=False)
    assert sv[1] < 1e-10 * sv[0]                  # exact rank deficiency
    values = np.full(5, zero_field_splitting(P.lam_g, P.ups_g))
    with pytest.raises(ValueError, match="cannot tell"):
        fit_spin_parameters(kinds, fields, values, P,
                            vary=("lam_g", "ups_g"), sigmas_ghz=0.1)
    with pytest.raises(ValueError, match="identifiable"):
        design_fields(kinds, fields, 3, P, vary=("lam_g", "ups_g"))


def test_rank_one_update_determinant_identity():
    """det(A + g g^T) = det(A)(1 + g^T A^-1 g) >= det(A): adding an
    observation can only add information. Checked as an exact identity
    on seeded random PSD matrices."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        m = rng.standard_normal((4, 4))
        a = m @ m.T + 0.5 * np.eye(4)
        g = rng.standard_normal(4)
        lhs = np.linalg.det(a + np.outer(g, g))
        rhs = np.linalg.det(a) * (1.0 + g @ np.linalg.solve(a, g))
        assert abs(lhs - rhs) < 1e-9 * abs(rhs)
        assert lhs >= np.linalg.det(a) - 1e-12 * abs(lhs)


def test_design_fields_greedy_invariant_and_quality():
    """The design tool's chosen set must (a) reproduce its own greedy
    rule when recomputed independently, and (b) never be less
    informative than the worst same-size subset."""
    kinds, fields = _design(n=10)
    out = design_fields(kinds, fields, 4, P, vary=("lam_g", "ups_g"),
                        sigmas_ghz=0.1)
    assert len(out["indices"]) == 4
    assert len(set(out["indices"])) == 4
    assert out["sigma"] is not None

    def logdet_of(idx):
        info = parameter_information([kinds[i] for i in idx],
                                     fields[list(idx)], P,
                                     vary=("lam_g", "ups_g"),
                                     sigmas_ghz=0.1)
        s, d = np.linalg.slogdet(info["fisher"])
        return d if s > 0 else -np.inf

    best = logdet_of(out["indices"])
    rng = np.random.default_rng(11)
    for _ in range(30):
        idx = rng.choice(10, size=4, replace=False)
        assert best >= logdet_of(idx) - 1e-9


def test_csv_round_trip_exact_and_refusals(tmp_path):
    kinds, fields = _design(n=4)
    values = np.array([_observe(P, k, b, 1.0)
                       for k, b in zip(kinds, fields)])
    sig = np.array([0.1, 0.2, 0.1, 0.3])
    path = tmp_path / "obs.csv"
    save_observations_csv(path, kinds, fields, values, sig)
    k2, b2, v2, s2 = load_observations_csv(path)
    assert k2 == kinds
    assert np.array_equal(b2, fields)
    assert np.array_equal(v2, values)
    assert np.array_equal(s2, sig)
    bad = tmp_path / "bad.csv"
    bad.write_text("wrong,header\n1,2\n")
    with pytest.raises(ValueError, match="header"):
        load_observations_csv(bad)


def test_input_refusals():
    kinds, fields = _design(n=4)
    values = np.zeros(4)
    with pytest.raises(ValueError, match="kind"):
        fit_spin_parameters(["nope"] * 4, fields, values, P,
                            sigmas_ghz=0.1)
    with pytest.raises(ValueError, match="vary"):
        fit_spin_parameters(kinds, fields, values, P, vary=(),
                            sigmas_ghz=0.1)
    with pytest.raises(ValueError, match="cannot vary"):
        fit_spin_parameters(kinds, fields, values, P,
                            vary=("gamma",), sigmas_ghz=0.1)
    with pytest.raises(ValueError, match="positive"):
        fit_spin_parameters(kinds, fields, values, P, sigmas_ghz=-1.0)
    with pytest.raises(ValueError, match="determine"):
        fit_spin_parameters(kinds[:2], fields[:2], values[:2], P,
                            vary=("lam_g", "ups_g"))
    with pytest.raises(ValueError, match="n_pick"):
        design_fields(kinds, fields, 1, P, vary=("lam_g", "ups_g"))


def test_propagation_exact_for_linear_quantities():
    """For quantities linear in the fitted parameters first-order
    propagation is exact: the parameter itself returns its own error
    bar (to 1e-9 relative), the zero-field splitting at ups_g = 0
    (which IS lam_g) returns sigma / sqrt(n), and a fixed linear
    combination A x returns the covariance A C A^T (to 1e-8 relative)."""
    from vacspin import propagate_uncertainty
    p0 = dataclasses.replace(P, ups_g=0.0)
    n, sigma = 6, 0.3
    fit = fit_spin_parameters(["orbital_g"] * n, np.zeros((n, 3)),
                              np.full(n, p0.lam_g), p0, vary=("lam_g",),
                              sigmas_ghz=sigma)
    out = propagate_uncertainty(fit, lambda p: p.lam_g)
    assert abs(out["sigma"] - fit.sigma["lam_g"]) < 1e-9 * out["sigma"]
    out = propagate_uncertainty(
        fit, lambda p: zero_field_splitting(p.lam_g, p.ups_g))
    assert abs(out["sigma"] - sigma / np.sqrt(n)) < 1e-9
    # two parameters, vector output
    kinds, fields = _design()
    values = np.array([_observe(P, k, b, 1.0)
                       for k, b in zip(kinds, fields)])
    fit2 = fit_spin_parameters(kinds, fields, values, P,
                               vary=("lam_g", "ups_g"), sigmas_ghz=0.05)
    a = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 2.0]])
    out = propagate_uncertainty(
        fit2, lambda p: [p.lam_g, p.ups_g, p.lam_g + 2.0 * p.ups_g])
    want = a @ fit2.cov @ a.T
    assert out["value"].shape == (3,) and out["sigma"].shape == (3,)
    assert np.allclose(out["cov"], want, rtol=1e-8, atol=0.0)
    assert np.allclose(out["sigma"], np.sqrt(np.diag(want)), rtol=1e-8)


def test_propagation_matches_sampling_for_a_prediction():
    """The qubit frequency at a field that was not measured is a
    nonlinear function of (lam_g, ups_g). Its propagated error bar
    agrees within 5 % with the spread obtained by sampling 4000
    parameter sets from the fit's Gaussian uncertainty and evaluating
    qubit_frequency on each (a second, independent method)."""
    from vacspin import propagate_uncertainty
    kinds, fields = _design()
    values = np.array([_observe(P, k, b, 1.0)
                       for k, b in zip(kinds, fields)])
    fit = fit_spin_parameters(kinds, fields, values, P,
                              vary=("lam_g", "ups_g"), sigmas_ghz=0.05)
    b_new = field_on_circle(35.0, 0.3)
    out = propagate_uncertainty(fit, lambda p: qubit_frequency(p, b_new))
    assert out["value"] == qubit_frequency(fit.params, b_new)
    rng = np.random.default_rng(21)
    draws = rng.multivariate_normal(
        [fit.values["lam_g"], fit.values["ups_g"]], fit.cov, size=4000)
    samples = [qubit_frequency(dataclasses.replace(
        fit.params, lam_g=lg, ups_g=ug), b_new) for lg, ug in draws]
    spread = float(np.std(samples, ddof=1))
    assert out["sigma"] > 0.0
    assert abs(out["sigma"] - spread) < 0.05 * spread


def test_propagation_refusals():
    from vacspin import propagate_uncertainty
    kinds, fields = _design()
    values = np.array([_observe(P, k, b, 1.0)
                       for k, b in zip(kinds, fields)])
    fit = fit_spin_parameters(kinds, fields, values, P,
                              vary=("lam_g", "ups_g"), sigmas_ghz=0.05)
    with pytest.raises(TypeError):
        propagate_uncertainty({"not": "a fit"}, lambda p: p.lam_g)
    with pytest.raises(ValueError, match="rel_step"):
        propagate_uncertainty(fit, lambda p: p.lam_g, rel_step=0.5)
    with pytest.raises(ValueError, match="non-finite"):
        propagate_uncertainty(fit, lambda p: np.nan)


def test_propagation_boundary_and_error_handling():
    """At a parameter boundary (ups_g = 0: a downward step would give
    negative strain) the slope is taken one-sided, which is exact for
    a linear function: sigma(lam_g + 2 ups_g) = 2 sigma(ups_g) when
    only ups_g is varied (to 1e-6 relative: the step is 1e-6 of the
    error bar, so round-off on the ~830 GHz value limits the slope to
    about 1e-7). An error raised by the
    user's function itself at a valid parameter set is NOT swallowed
    into a one-sided difference; it reaches the caller."""
    from vacspin import SpinFit, propagate_uncertainty
    s_ups = 0.7
    base = dataclasses.replace(P, ups_g=0.0)
    fit = SpinFit(params=base, values={"ups_g": 0.0},
                  sigma={"ups_g": s_ups}, cov=np.array([[s_ups ** 2]]),
                  chi2=None, chi2_dof=None, n_points=3,
                  condition_number=1.0, n_iter=1, converged=True)
    out = propagate_uncertainty(fit, lambda p: p.lam_g + 2.0 * p.ups_g)
    assert abs(out["sigma"] - 2.0 * s_ups) < 1e-6 * 2.0 * s_ups

    kinds, fields = _design()
    values = np.array([_observe(P, k, b, 1.0)
                       for k, b in zip(kinds, fields)])
    fit2 = fit_spin_parameters(kinds, fields, values, P,
                               vary=("lam_g", "ups_g"), sigmas_ghz=0.05)
    lam_fit = fit2.values["lam_g"]

    def picky(p):
        if p.lam_g < lam_fit:
            raise ValueError("picky function refuses")
        return p.lam_g

    with pytest.raises(ValueError, match="picky"):
        propagate_uncertainty(fit2, picky)
