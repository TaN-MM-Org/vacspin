"""Transition-table anchors: the exact dipole sum rule (each ground
state carries total strength exactly 3 over the sixteen transitions,
because each dipole component squares to the identity), table
structure, field-sweep geometry, and refusals."""
import numpy as np
import pytest

from vacspin import (cyclicity, field_on_circle, rabi_rate,
                     siv_hepp2014, snv_rosenthal2023, transition_table)

P = snv_rosenthal2023()


def test_dipole_sum_rule_exact():
    """sum_j sum_i |<e_j| p_i |g>|^2 = <g| (px^2 + py^2 + pz^2) |g> = 3
    exactly, for every ground state, any field, any centre -- the
    completeness of the excited manifold makes it an identity."""
    for params in (P, siv_hepp2014()):
        for b in ([0, 0, 0], [0.05, -0.02, 0.11], [0.2, 0.0, 0.0]):
            t = transition_table(params, b)
            row_sums = t["strengths"].sum(axis=1)
            assert np.allclose(row_sums, 3.0, rtol=0, atol=1e-10)


def test_table_structure():
    t = transition_table(P, [0.0, 0.0, 0.18])
    assert t["strengths"].shape == (4, 4)
    assert np.all(t["strengths"] >= 0.0)
    # energies[i, j] = E_e[j] - E_g[i], exactly
    want = t["evals_e"][None, :] - t["evals_g"][:, None]
    assert np.allclose(t["energies_ghz"], want, atol=1e-12)


def test_field_on_circle_geometry_and_refusal():
    for plane in ("xz", "xy", "yz"):
        for z in (0.0, 30.0, 147.0, 290.0):
            b = field_on_circle(z, 0.18, plane)
            assert abs(np.linalg.norm(b) - 0.18) < 1e-12
    assert np.allclose(field_on_circle(0.0, 1.0, "xz"), [1, 0, 0])
    assert np.allclose(field_on_circle(90.0, 1.0, "xz"), [0, 0, 1],
                       atol=1e-12)
    with pytest.raises(ValueError):
        field_on_circle(10.0, 0.1, plane="zz")


def test_cyclicity_manifold_choice_and_refusal():
    b = field_on_circle(147.0, 0.18)
    lam_lower = cyclicity(P, b, manifold="lower")
    lam_upper = cyclicity(P, b, manifold="upper")
    assert lam_lower > 0 and lam_upper > 0
    with pytest.raises(ValueError):
        cyclicity(P, b, manifold="middle")


def test_rabi_vanishes_without_drive():
    b = field_on_circle(147.0, 0.125)
    assert rabi_rate(P, b, [0.0, 0.0, 0.0]) < 1e-15


def test_siv_unstrained_cyclicity_diverges_along_axis():
    """The SiV set at zero strain: a field along the <111> axis keeps
    spin a good quantum number, so the flip channel is exactly dark --
    the same protection the SnV shows, from a different parameter
    set."""
    p = siv_hepp2014()
    mu = np.array([np.sin(p.theta_rad) * np.cos(p.phi_rad),
                   np.sin(p.theta_rad) * np.sin(p.phi_rad),
                   np.cos(p.theta_rad)])
    assert cyclicity(p, 0.1 * mu) > 1e12
