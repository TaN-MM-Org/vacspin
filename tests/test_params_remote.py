"""Parameter provenance and remote-entanglement anchors: mandatory
references, derived-quantity identities, the SiV set carrying exactly
what its source states, and the Barrett-Kok budget's exact form,
symmetry and refusals."""
import numpy as np
import pytest

from vacspin import (EmissionBudget, SpinParameters, barrett_kok_success,
                     entanglement_rate, siv_hepp2014, snv_emission,
                     snv_rosenthal2023, zero_field_splitting)


def test_references_are_mandatory():
    with pytest.raises(ValueError, match="reference"):
        SpinParameters(lam_g=100, ups_g=0, lam_e=200, ups_e=0, f_g=0,
                       f_e=0, delta_g=0, delta_e=0, theta_rad=0.0,
                       phi_rad=0.0, reference="")
    with pytest.raises(ValueError, match="reference"):
        EmissionBudget(tau0_s=1e-9, eta_q=0.5, eta_dw=0.5, eta_br=0.5,
                       zpl_nm=700.0, reference="  ")


def test_budget_identities_and_refusals():
    b = snv_emission()
    assert abs(b.gamma0 * b.tau0_s - 1.0) < 1e-15
    assert abs(b.radiative_fraction - b.eta_q * b.eta_dw * b.eta_br) \
        < 1e-15
    with pytest.raises(ValueError):
        EmissionBudget(tau0_s=-1.0, eta_q=0.5, eta_dw=0.5, eta_br=0.5,
                       zpl_nm=700.0, reference="a real citation here")
    with pytest.raises(ValueError):
        EmissionBudget(tau0_s=1e-9, eta_q=1.5, eta_dw=0.5, eta_br=0.5,
                       zpl_nm=700.0, reference="a real citation here")


def test_snv_derived_quenching_values():
    """f = gL p and delta = gL delta_p from the cited reduction
    factors: the derivation is part of the provenance."""
    p = snv_rosenthal2023()
    assert abs(p.f_g - 0.363 * 0.471) < 1e-12
    assert abs(p.f_e - 0.581 * 0.125) < 1e-12
    assert abs(p.delta_g - 0.363 * 0.042) < 1e-12
    assert abs(p.delta_e - 0.581 * 0.303) < 1e-12


def test_siv_set_ships_only_what_its_source_states():
    """Hepp 2014: 50 GHz ground, 260 GHz excited, zero strain and zero
    quenching (sample-specific, deliberately not shipped) -- so the
    zero-field splittings ARE the spin-orbit constants, exactly, and
    the default orientation is the exact <111> geometry."""
    p = siv_hepp2014()
    assert p.ups_g == 0.0 and p.f_g == 0.0 and p.delta_g == 0.0
    assert zero_field_splitting(p.lam_g, p.ups_g) == 50.0
    assert zero_field_splitting(p.lam_e, p.ups_e) == 260.0
    assert abs(np.cos(p.theta_rad) - 1.0 / np.sqrt(3.0)) < 1e-15
    assert "not shipped" in p.reference


def test_barrett_kok_exact_form():
    assert barrett_kok_success(1.0, 1.0) == 0.5   # the intrinsic ceiling
    assert barrett_kok_success(0.0, 1.0) == 0.0
    assert abs(barrett_kok_success(0.3, 0.5) - 0.5 * 0.15) < 1e-15
    # symmetric default and exact symmetry
    assert barrett_kok_success(0.42) == barrett_kok_success(0.42, 0.42)
    assert barrett_kok_success(0.3, 0.7) == barrett_kok_success(0.7, 0.3)
    assert abs(entanglement_rate(1e5, 0.2) - 1e5 * 0.5 * 0.04) < 1e-9
    with pytest.raises(ValueError):
        barrett_kok_success(1.2)
    with pytest.raises(ValueError):
        entanglement_rate(0.0, 0.5)
