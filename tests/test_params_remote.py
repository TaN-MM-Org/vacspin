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


def test_non_finite_inputs_refused():
    """NaN and infinity are refused with a message instead of flowing
    into NaN results (before 0.4.0: a NaN spin-orbit constant was
    accepted, purcell_max(nan, 1) returned nan, CavityInterface with
    Q = 0 raised ZeroDivisionError and with omega_q = nan returned a
    validity verdict, entanglement_rate(inf, 0.5) returned inf)."""
    import dataclasses
    from vacspin import (CavityInterface, fidelity_threshold1,
                         lorentzian_suppression, polarization_rate,
                         purcell_max, qubit_frequency_perpendicular,
                         xi_pol_overlap)
    p = snv_rosenthal2023()
    for name in ("lam_g", "ups_e", "f_g", "delta_e", "theta_rad"):
        for bad in (np.nan, np.inf):
            with pytest.raises(ValueError, match="finite"):
                dataclasses.replace(p, **{name: bad})
    with pytest.raises(ValueError, match="finite"):
        qubit_frequency_perpendicular(p, np.nan)
    with pytest.raises(ValueError):
        purcell_max(np.nan, 1.0)
    with pytest.raises(ValueError):
        purcell_max(np.inf, 1.0)
    with pytest.raises(ValueError):
        lorentzian_suppression(np.nan, 4.8e14, 500.0)
    base = dict(q_loaded=500.0, v_rel=0.68, eta_wg=0.99,
                budget=snv_emission(), lambda0=2244.0, omega_q_hz=3.7e9,
                xi_pol=xi_pol_overlap(), xi_pos=0.5)
    for key, bad in (("q_loaded", 0.0), ("q_loaded", np.inf),
                     ("v_rel", np.nan), ("omega_q_hz", np.nan),
                     ("delta_partner_hz", np.nan),
                     ("delta_partner_hz", -1e12)):
        args = dict(base)
        args[key] = bad
        with pytest.raises(ValueError):
            CavityInterface(**args)
    CavityInterface(**base)                       # the valid case builds
    with pytest.raises(ValueError, match="finite"):
        entanglement_rate(np.inf, 0.5)
    with pytest.raises(ValueError, match="finite"):
        entanglement_rate(np.nan, 0.5)
    with pytest.raises(ValueError, match="finite"):
        fidelity_threshold1(np.nan, 0.0)
    with pytest.raises(ValueError, match="finite"):
        polarization_rate(np.nan, 10.0)
    # array inputs keep working
    rates = polarization_rate(1e8, np.array([0.0, 9.0]))
    assert abs(rates[1] - rates[0] / 10.0) < 1e-6
