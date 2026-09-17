"""The shipped SnV- set against published measurements -- no free
parameters. Zero-field splittings (PRX 13, 031022 Table I), the qubit
frequency at the readout operating point, the cyclicity landscape
(2244 near alignment, 8.6 at 53 deg; arXiv:2403.13110), the exact
divergence for an aligned field, and the MHz-scale Rabi rate."""
import numpy as np

from vacspin import (cyclicity, field_on_circle, qubit_frequency,
                     rabi_rate, snv_rosenthal2023, solve,
                     transition_strength, zero_field_splitting)

P = snv_rosenthal2023()


def _mu_lab():
    th, ph = P.theta_rad, P.phi_rad
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                     np.cos(th)])


def test_zero_field_splittings():
    """Delta_g = 902.98 GHz (PL measurement), Delta_e = 3000 GHz:
    closed form AND full diagonalisation, two code paths."""
    dg = zero_field_splitting(P.lam_g, P.ups_g)
    de = zero_field_splitting(P.lam_e, P.ups_e)
    assert abs(dg - 902.98) < 0.002 * 902.98
    assert abs(de - 3000.0) < 0.002 * 3000.0
    eg, _, ee, _ = solve(P, [0.0, 0.0, 0.0])
    assert abs((eg[2] - eg[0]) - dg) < 1e-9 * dg
    assert abs((ee[2] - ee[0]) - de) < 1e-9 * de


def test_qubit_frequency_at_operating_point():
    """3.677 GHz measured at zeta = 147 deg, 125 mT."""
    fq = qubit_frequency(P, field_on_circle(147.0, 0.125))
    assert abs(fq - 3.677) < 0.02 * 3.677


def test_cyclicity_landscape():
    """Measured: 8.6 +/- 0.4 at zeta = 53 deg, 180 mT; 2244 +/- 108 at
    ~9 deg misalignment from the axis."""
    lam_53 = cyclicity(P, field_on_circle(53.0, 0.180))
    assert abs(lam_53 - 8.6) < 0.10 * 8.6
    mu = _mu_lab()
    e1 = np.cross(mu, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    b9 = 0.180 * (np.cos(np.deg2rad(9)) * mu + np.sin(np.deg2rad(9)) * e1)
    lam_9 = cyclicity(P, b9)
    assert abs(lam_9 - 2244.0) < 0.10 * 2244.0


def test_aligned_field_divergence():
    """For B exactly along the <111> axis the spin-flip strength is
    numerically zero and the cyclicity diverges -- the analytic limit,
    asserted."""
    mu = _mu_lab()
    eg, vg, ee, ve = solve(P, 0.180 * mu)
    s0 = transition_strength(vg[:, 0], ve[:, 0])
    s1 = transition_strength(vg[:, 0], ve[:, 1])
    j = 0 if s0 >= s1 else 1
    assert transition_strength(vg[:, 1], ve[:, j]) < 1e-20
    assert cyclicity(P, 0.180 * mu) > 1e12


def test_rabi_rate_scale_and_linearity():
    """~ 6.25 MHz measured at |b| = 0.6 mT drive (arXiv:2403.13110);
    the model lands on the MHz scale at the right field, and the rate
    is exactly linear in the drive amplitude."""
    b0 = field_on_circle(147.0, 0.125)
    drives = []
    for ang in (0.0, 45.0, 90.0):
        a = np.deg2rad(ang)
        drives.append(rabi_rate(P, b0,
                                6e-4 * np.array([np.sin(a), 0, np.cos(a)]))
                      * 1e3)                      # MHz
    assert 1.0 < max(drives) < 20.0
    om1 = rabi_rate(P, b0, [6e-4, 0.0, 0.0])
    om2 = rabi_rate(P, b0, [1.2e-3, 0.0, 0.0])
    assert abs(om2 - 2 * om1) < 1e-12 * max(om2, 1e-30)


def test_strain_engineering_moves_the_splitting():
    """Doubling the strain moves Delta_g exactly along the closed
    form -- the strain-tuning axis works and stays anchored."""
    for s in (0.0, 1.0, 2.5):
        eg, _, _, _ = solve(P, [0.0, 0.0, 0.0], strain_scale=s)
        want = zero_field_splitting(P.lam_g, s * P.ups_g)
        assert abs((eg[2] - eg[0]) - want) < 1e-9 * max(want, 1.0)
