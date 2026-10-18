"""Weak fields: the cyclicity and the Rabi rate need a resolved qubit
doublet (since 0.8.0).

At zero field each Kramers doublet is exactly degenerate, so its two
states are any orthonormal pair, and the cyclicity and Rabi rate built
from them depend on how the field goes to zero. Up to 0.7.0 the
functions returned whatever the eigensolver's arbitrary pair gave: for
the SnV- set at 53 degrees, 1.1e34 at 1e-15 T and 9.06 at 1e-12 T,
where the weak-field limit is 7.85; a Rabi rate of 2.5 MHz at 1e-15 T,
where the limit is 5.9 MHz. They now refuse a doublet split by less
than 1e-8 of the manifold's energy scale.

The independent check is degenerate perturbation theory: for a weak
field the doublet states are the eigenvectors of the Zeeman operator
projected onto the zero-field doublet (a 2 x 2 problem built from the
zero-field eigenvectors, which a gap of hundreds of GHz fixes well).
Its corrections are of relative order (Zeeman energy) / (zero-field
splitting), below 1e-6 at 30 uT, and the round-off of the full
diagonalization at an accepted field is below about
eps E_max / d = 2.2e-16 / 1e-8 = 2.2e-8 relative (d the doublet
splitting, E_max the energy scale); the tolerance 1e-5 leaves a wide
margin for both."""
import numpy as np
import pytest

import vacspin as vs
from vacspin.hamiltonian import GAMMA_GHZ_PER_T, h_manifold, lab_to_spin
from vacspin.transitions import transition_strength


def _doublet_states(lam, ups, f, delta, bs, lower=True):
    """First-order states of one doublet in a weak spin-frame field bs:
    eigenvectors of the Zeeman operator inside the zero-field doublet."""
    h0 = h_manifold(lam, ups, 0.0, f, delta, np.zeros(3))
    zee = h_manifold(lam, ups, 0.0, f, delta, bs) - h0   # linear in B
    _, v0 = np.linalg.eigh(h0)
    sub = v0[:, :2] if lower else v0[:, 2:]
    m = sub.conj().T @ zee @ sub
    _, c = np.linalg.eigh(m)
    return sub @ c[:, 0], sub @ c[:, 1]


def _weak_field_limits(p, b_lab, drive):
    bs = lab_to_spin(b_lab, p.theta_rad, p.phi_rad)
    g0, g1 = _doublet_states(p.lam_g, p.ups_g, p.f_g, p.delta_g, bs)
    ea, eb = _doublet_states(p.lam_e, p.ups_e, p.f_e, p.delta_e, bs)
    e = ea if transition_strength(g0, ea) >= transition_strength(g0, eb) \
        else eb
    lam = transition_strength(g0, e) / transition_strength(g1, e)
    ds = lab_to_spin(drive, p.theta_rad, p.phi_rad)
    zee = np.array([[ds[2], ds[0] - 1j * ds[1]],
                    [ds[0] + 1j * ds[1], -ds[2]]])
    h_mw = 0.5 * GAMMA_GHZ_PER_T * np.kron(np.eye(2), zee)
    return lam, abs(np.vdot(g1, h_mw @ g0))


def _directions(n, seed):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((n, 3))
    return d / np.linalg.norm(d, axis=1)[:, None]


def test_weak_field_values_equal_degenerate_perturbation_theory():
    p = vs.snv_rosenthal2023()
    drive = [6e-4, 0.0, 0.0]
    dirs = [vs.field_on_circle(a, 1.0) for a in (0.0, 53.0, 90.0, 147.0)]
    dirs += list(_directions(8, 11))
    for u in dirs:
        b = 3e-5 * np.asarray(u)                       # 30 uT
        lam_pt, rabi_pt = _weak_field_limits(p, b, drive)
        assert vs.cyclicity(p, b) == pytest.approx(lam_pt, rel=1e-5)
        assert vs.rabi_rate(p, b, drive) == pytest.approx(rabi_pt,
                                                          rel=1e-5)
    # the 53-degree limit quoted in the docstrings
    lam53, rabi53 = _weak_field_limits(p, vs.field_on_circle(53.0, 3e-5),
                                       drive)
    assert lam53 == pytest.approx(7.85, abs=0.005)
    assert rabi53 == pytest.approx(5.93e-3, abs=0.005e-3)


def test_unresolved_doublet_is_refused():
    """These calls returned 1.1e34, 9.06, inf and a 2.5 MHz Rabi rate
    in 0.7.0 (weak-field limits 7.85 and 5.9 MHz)."""
    p = vs.snv_rosenthal2023()
    for bm in (1e-15, 1e-12, 0.0):
        b = vs.field_on_circle(53.0, bm)
        with pytest.raises(ValueError, match="does not resolve"):
            vs.cyclicity(p, b)
        with pytest.raises(ValueError, match="does not resolve"):
            vs.rabi_rate(p, b, [6e-4, 0.0, 0.0])
    with pytest.raises(ValueError, match="does not resolve"):
        vs.cyclicity(p, np.zeros(3), manifold="upper")
    # SiV- without strain or orbital terms: a field perpendicular to the
    # axis leaves the qubit exactly degenerate at any strength
    siv = vs.siv_hepp2014()
    th, ph = siv.theta_rad, siv.phi_rad
    perp = 0.2 * np.array([-np.sin(ph), np.cos(ph), 0.0])
    assert abs(np.dot(perp, [np.sin(th) * np.cos(ph),
                             np.sin(th) * np.sin(ph), np.cos(th)])) < 1e-15
    with pytest.raises(ValueError, match="does not resolve"):
        vs.cyclicity(siv, perp)
    with pytest.raises(ValueError, match="does not resolve"):
        vs.rabi_rate(siv, perp, [6e-4, 0.0, 0.0])


def test_threshold_matches_its_definition():
    """The refusal sits where the doublet splitting crosses 1e-8 of the
    largest energy of its manifold, computed here from `solve`."""
    p = vs.snv_rosenthal2023()
    u = vs.field_on_circle(147.0, 1.0)
    eg = vs.solve(p, 1e-3 * u)[0]
    # the qubit splitting is linear in a weak field
    b_min = 1e-8 * np.max(np.abs(eg)) / (eg[1] - eg[0]) * 1e-3
    vs.rabi_rate(p, 1.01 * b_min * u, [6e-4, 0.0, 0.0])
    with pytest.raises(ValueError, match="does not resolve"):
        vs.rabi_rate(p, 0.99 * b_min * u, [6e-4, 0.0, 0.0])
