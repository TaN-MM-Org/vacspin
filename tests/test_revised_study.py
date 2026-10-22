"""0.8.1: the revised source study, checked number by number.

T. M. Mahim, M. M. Rahman and A. S. M. Mohsin, "Fast single-shot
readout of tin-vacancy spins with an overcoupled diamond nanocavity on
thin-film lithium niobate", revised manuscript (revision 1, submitted
to Optics Express, 2026); code: Tanvir-Mahmud-Mahim/snv-tfln-cavity-
interface; data: doi:10.5281/zenodo.23507013.

The revision moved the design point to an overcoupled cavity with
Q_L = 870 and C-transition Purcell factor F_C = 28, and replaced the
two-level readout model of the original submission by a four-level
model. This file checks what vacspin computes against the numbers the
revised paper prints, and pins down where the two differ:

* the spin model at the paper's reference operating point (112 mT,
  9 degrees from the spin axis): qubit frequency 3.68 GHz, excited
  splitting 4.40 GHz (vacspin: 4.41), A1-B2 detuning 0.73 GHz,
  cyclicity 2180, and the 77 mT field at which the A1-B2 detuning is
  500 MHz;
* the cavity budget: F_C = 28, D-line Purcell factor 3.4, coupling
  6.8 GHz (vacspin: 6.74) with g / kappa = 0.012, the bare cyclicity
  kept by the broad cavity, and a detected-photon efficiency of 32 %;
* the one term vacspin leaves out: the paper's decay rate (its Eq. 1)
  also counts the cavity-enhanced emission of the D line. With it the
  rate rises 10.5 times (lifetime 0.43 ns, beta = 91 %); vacspin's
  `zeta` is smaller by exactly eta_q eta_DW (1 - eta_BR) (F_D - 1).

Every paper value is quoted from the revised manuscript as printed;
each assertion states its tolerance.
"""
import numpy as np
import pytest

import vacspin as vs

P = vs.snv_rosenthal2023()
BUD = vs.snv_emission()


def _field(b_mag_t, misalign_deg, azimuth_deg=0.0):
    """A field of b_mag_t tesla, misalign_deg from the spin (<111>)
    axis, turned by azimuth_deg about that axis."""
    th, ph = P.theta_rad, P.phi_rad
    mu = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                   np.cos(th)])
    e1 = np.cross(mu, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(mu, e1)
    a, m = np.deg2rad(azimuth_deg), np.deg2rad(misalign_deg)
    d = np.cos(a) * e1 + np.sin(a) * e2
    return b_mag_t * (np.cos(m) * mu + np.sin(m) * d)


def _splittings(b):
    eg, _, ee, _ = vs.solve(P, b)
    return eg[1] - eg[0], ee[1] - ee[0]          # omega_q, omega_e (GHz)


def test_reference_operating_point():
    """Paper: at |B| = 112 mT, 9 deg from the spin axis, the
    Hamiltonian gives omega_q = 3.68 GHz, omega_e = 4.40 GHz, an A1-B2
    detuning of 0.73 GHz and Lambda = 2180. vacspin gives 3.684, 4.412,
    0.7285 and 2179.7, and the same values at four azimuths of the
    field about the spin axis (a vacspin result; the paper states the
    azimuth independence for the readout fidelity)."""
    for az in (0.0, 70.0, 155.0, 290.0):
        b = _field(0.112, 9.0, az)
        wq, we = _splittings(b)
        assert abs(wq - 3.68) < 0.005
        assert abs(we - 4.40) < 0.015
        assert abs((we - wq) - 0.73) < 0.005
        assert abs(vs.cyclicity(P, b) - 2180.0) < 5.0
    # A1 is |dn> -> |A>, B2 is |up> -> |B>: their energy difference in
    # the transition table is the same detuning, and the two
    # spin-conserving lines have equal strength to 1e-3
    t = vs.transition_table(P, _field(0.112, 9.0))
    e, s = t["energies_ghz"], t["strengths"]
    wq, we = _splittings(_field(0.112, 9.0))
    assert abs((e[1, 1] - e[0, 0]) - (we - wq)) < 1e-9
    assert abs(s[1, 1] / s[0, 0] - 1.0) < 1e-3


def test_field_that_gives_a_500_mhz_detuning():
    """Paper: in the model, a 500 MHz A1-B2 detuning (the value
    reported experimentally) corresponds to |B| = 77 mT at 9 deg."""
    wq, we = _splittings(_field(0.077, 9.0))
    assert abs((we - wq) - 0.500) < 0.005


def _design_point():
    """The revised design point: Q_L = 870 and F_C = 28.

    The paper takes F_C = 1 + F_cav xi_pol xi_pos, with F_cav = 81
    from its simulated mode volume and Q_L (xi_pol = 2/3,
    xi_pos = 0.5). vacspin writes F_C = 1 + (F_max - 1) xi_pol xi_pos
    with F_max from the same kind of formula, so the two conventions
    differ by 1 in the Purcell factor. To compare at the paper's
    F_C = 28 the test sets F_max = 82: v_rel is then the mode volume
    that gives it at Q_L = 870, not the paper's mode volume. (With
    F_max = 81, vacspin's form gives F_C = 27.67 and eta = 0.318.) Downstream: waveguide fraction 0.92 (FDTD),
    taper 0.997 (EME), chip-to-fiber 0.5 and detector 0.77 (measured
    values the paper uses)."""
    v_rel = vs.purcell_max(870.0, 1.0) / 82.0
    b = _field(0.112, 9.0)
    return vs.CavityInterface(
        q_loaded=870.0, v_rel=v_rel, eta_wg=0.92, budget=BUD,
        lambda0=vs.cyclicity(P, b),
        omega_q_hz=vs.qubit_frequency(P, b) * 1e9,
        xi_pol=vs.xi_pol_overlap(), xi_pos=0.5,
        t_link=0.997, eta_chip=0.5, eta_det=0.77,
        delta_partner_hz=0.90e12)                # paper: D line 0.90 THz away


def test_cavity_budget_of_the_revised_design():
    ci = _design_point().require_valid()
    assert ci.f_max == pytest.approx(82.0, rel=1e-12)
    assert ci.f_c == pytest.approx(28.0, rel=1e-6)      # paper: F_C = 28
    assert abs(ci.f_partner - 3.4) < 0.05               # paper: F_D = 3.4
    # paper: g / 2 pi = 6.8 GHz, g / kappa = 0.012
    assert abs(ci.g_hz / 1e9 - 6.8) < 0.07
    assert abs(ci.g_hz / ci.kappa_hz - 0.012) < 0.0005
    # paper: the cavity (kappa / 2 pi about 0.6 THz) enhances all four
    # C-line transitions alike, so the cyclicity is the bare one
    b = _field(0.112, 9.0)
    assert abs(ci.lambda_cav / vs.cyclicity(P, b) - 1.0) < 1e-3
    assert abs(ci.kappa_hz / 1e12 - 0.6) < 0.05


def test_detected_photon_efficiency_and_the_d_line_term():
    """Paper Eq. (1): gamma_cav / gamma_0 = 1 + eta_q eta_DW
    [eta_BR (F_C - 1) + (1 - eta_BR) (F_D - 1)], beta = 1 - gamma_0 /
    gamma_cav, eta = beta eta_wg T eta_chip eta_det. Evaluated here by
    hand from vacspin's F_C and F_D, it gives the paper's 10.5, 0.43 ns,
    91 % and 32 %. vacspin's own `zeta` leaves out the D-line term, so
    it is smaller by exactly that term (10.23 instead of 10.50); as a
    consequence its beta and eta are 0.3 % lower (0.902 and 0.319)."""
    ci = _design_point()
    eq, edw, ebr = BUD.eta_q, BUD.eta_dw, BUD.eta_br
    d_term = eq * edw * (1.0 - ebr) * (ci.f_partner - 1.0)
    zeta_eq1 = 1.0 + eq * edw * (ebr * (ci.f_c - 1.0)) + d_term
    beta_eq1 = 1.0 - 1.0 / zeta_eq1
    eta_eq1 = beta_eq1 * 0.92 * 0.997 * 0.5 * 0.77
    # Eq. (1) by hand reproduces the paper's printed values
    assert abs(zeta_eq1 - 10.5) < 0.05                  # paper: 10.5
    assert abs(BUD.tau0_s / zeta_eq1 * 1e9 - 0.43) < 0.005   # 4.5 -> 0.43 ns
    assert abs(beta_eq1 - 0.91) < 0.006                 # paper: 91 %
    assert abs(eta_eq1 - 0.32) < 0.005                  # paper: 32 %
    # vacspin differs by the D-line term alone
    assert ci.zeta == pytest.approx(zeta_eq1 - d_term, rel=1e-12)
    assert ci.beta == pytest.approx(1.0 - 1.0 / ci.zeta, rel=1e-12)
    assert ci.eta == pytest.approx(ci.beta * 0.92 * 0.997 * 0.5 * 0.77,
                                   rel=1e-12)
    assert abs(ci.zeta - 10.23) < 0.01 and abs(ci.beta - 0.902) < 0.001
    assert abs(ci.eta / eta_eq1 - 1.0) < 0.004
    assert abs(ci.eta - 0.32) < 0.005                   # still 32 %


def test_efficiency_gain_over_the_confocal_chain():
    """Paper: the detected fraction (32 %) is about two orders of
    magnitude above the measured confocal 0.1 to 0.4 %."""
    ci = _design_point()
    assert 50.0 < ci.eta / 0.004 and ci.eta / 0.001 < 500.0
