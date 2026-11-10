"""0.10.0 fixes. Each test fails on 0.9.0 and passes now."""
import math

import numpy as np
import pytest

import vacspin as vs


def test_xi_pol_overlap_default_is_two_thirds():
    """The default angle is arccos(1/sqrt 3) to full precision, so the
    default overlap is the documented 2/3 (to 1e-15). 0.9.0 used the
    rounded 54.7356 degrees and returned 0.66666650, 2.5e-7 low."""
    assert vs.xi_pol_overlap() == pytest.approx(2.0 / 3.0, abs=1e-15)
    exact = math.degrees(math.acos(1.0 / math.sqrt(3.0)))
    assert vs.xi_pol_overlap() == \
        pytest.approx(vs.xi_pol_overlap(exact), abs=1e-15)
    # the rounded angle still gives what 0.9.0 gave by default
    assert vs.xi_pol_overlap(54.7356) == \
        pytest.approx(0.666666496895125, abs=1e-14)


def test_pmf_kmax_must_be_a_whole_number():
    """0.9.0 returned an empty array for kmax = -1, cut kmax = 2.9 down to
    2, and raised OverflowError for an infinite kmax. Now each is a
    ValueError, while a whole-number float or a NumPy integer works as
    the int does (the same computation, so the arrays are identical)."""
    for fn in (vs.poisson_pmf, vs.geometric_pmf):
        for bad in (-1, 2.9, float("inf"), float("nan"), True, "3"):
            with pytest.raises(ValueError, match="kmax"):
                fn(3.0, bad)
        ref = fn(3.0, 10)
        assert ref.size == 11
        assert np.array_equal(fn(3.0, 10.0), ref)
        assert np.array_equal(fn(3.0, np.int64(10)), ref)
        assert fn(3.0, 0).size == 1


def test_design_fields_refuses_a_fractional_n_pick():
    """n_pick = 2.9 picked 2 points in 0.9.0; now refused. n_pick = 2.0
    picks the same points as 2."""
    p = vs.snv_rosenthal2023()
    kinds = ["qubit", "orbital_g"] * 3
    fields = [vs.field_on_circle(z, 0.15)
              for z in (20, 60, 100, 140, 160, 80)]
    with pytest.raises(ValueError, match="whole number"):
        vs.design_fields(kinds, fields, 2.9, p)
    with pytest.raises(ValueError, match="whole number"):
        vs.design_fields(kinds, fields, float("nan"), p)
    assert vs.design_fields(kinds, fields, 2.0, p)["indices"] == \
        vs.design_fields(kinds, fields, 2, p)["indices"]


def test_dark_counts_with_the_background_held_at_zero_are_named():
    """A dark histogram with counts, fitted with the background held at
    0 (noise_rate left at None and not varied), has zero probability.
    0.9.0 refused with a message that blamed the bright starting values
    (and printed a model mean equal to the measured one); the message
    now names the dark histogram and the background."""
    gamma = vs.snv_emission().gamma0
    rng = np.random.default_rng(1)
    hb = np.bincount(rng.poisson(5.0, 2000))
    hd = np.bincount(rng.poisson(0.3, 2000))
    with pytest.raises(ValueError, match="dark histogram has shots with "
                                         "counts.*background is held at 0"):
        vs.fit_readout_histograms(hb, gamma, 1e-6, 2.0, dark_hist=hd,
                                  lam=1e9, vary=("eta",))
    # with the background fitted, the same data are fine
    fit = vs.fit_readout_histograms(hb, gamma, 1e-6, 2.0, dark_hist=hd,
                                    lam=1e9, vary=("eta", "noise_rate"))
    assert fit.values["noise_rate"] * 1e-6 == pytest.approx(0.3, rel=0.1)


def test_spin_fit_reference_records_the_strain_scale():
    """A fit at strain_scale = 1.3 returns the unscaled strain, so its
    parameter set reproduces the data only at that scale; the reference
    now says so. At strain_scale = 1 the reference reads as before."""
    p = vs.snv_rosenthal2023()
    kinds = ["qubit", "orbital_g"] * 3
    fields = np.array([vs.field_on_circle(z, 0.15)
                       for z in (20, 60, 100, 140, 160, 80)])
    for scale in (1.0, 1.3):
        values = [vs.qubit_frequency(p, b, scale) if k == "qubit"
                  else (lambda e: e[2] - e[0])(vs.solve(p, b, scale)[0])
                  for k, b in zip(kinds, fields)]
        fit = vs.fit_spin_parameters(kinds, fields, values, p,
                                     sigmas_ghz=0.05, strain_scale=scale)
        ref = fit.params.reference
        if scale == 1.0:
            assert "strain_scale" not in ref
            assert ref.startswith("fitted by vacspin.lab.fit_spin_parameters"
                                  " to 6 measured points (vary=lam_g,ups_g)"
                                  "; base set: ")
        else:
            assert "strain_scale=1.3" in ref
            # the set reproduces the data at that scale
            assert vs.qubit_frequency(fit.params, fields[0], scale) == \
                pytest.approx(values[0], abs=1e-6)
