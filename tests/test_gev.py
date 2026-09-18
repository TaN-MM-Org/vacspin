"""GeV- anchors: the shipped values are locked to their source
(Bhaskar et al., PRL 118, 223603 (2017): 152 GHz ground, 981 GHz
excited, measured); at zero strain the closed-form zero-field
splitting IS the spin-orbit constant, exactly; the full solver
reproduces both splittings; the dipole sum rule holds for this centre
as an identity; an aligned field keeps the flip channel exactly dark
(the same protection SnV and SiV show); and the provenance string
carries its source and its deliberate omissions."""
import numpy as np

from vacspin import (cyclicity, gev_bhaskar2017, solve,
                     transition_table, zero_field_splitting)


def test_values_locked_to_source():
    p = gev_bhaskar2017()
    assert (p.lam_g, p.lam_e) == (152.0, 981.0)
    assert (p.ups_g, p.ups_e) == (0.0, 0.0)
    assert (p.f_g, p.f_e, p.delta_g, p.delta_e) == (0.0,) * 4
    assert "Bhaskar" in p.reference and "223603" in p.reference
    assert "not" in p.reference        # the omissions are stated too


def test_zero_field_splittings_exact():
    p = gev_bhaskar2017()
    assert zero_field_splitting(p.lam_g, p.ups_g) == 152.0
    assert zero_field_splitting(p.lam_e, p.ups_e) == 981.0
    eg, _, ee, _ = solve(p, [0.0, 0.0, 0.0])
    assert abs((eg[2] - eg[0]) - 152.0) < 1e-9
    assert abs((ee[2] - ee[0]) - 981.0) < 1e-9


def test_sum_rule_and_aligned_protection():
    p = gev_bhaskar2017()
    for b in ([0, 0, 0], [0.03, -0.05, 0.09]):
        t = transition_table(p, b)
        assert np.allclose(t["strengths"].sum(axis=1), 3.0, rtol=0,
                           atol=1e-10)
    mu = np.array([np.sin(p.theta_rad) * np.cos(p.phi_rad),
                   np.sin(p.theta_rad) * np.sin(p.phi_rad),
                   np.cos(p.theta_rad)])
    assert cyclicity(p, 0.1 * mu) > 1e12
