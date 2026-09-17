"""Hamiltonian anchors: Hermiticity for random inputs, exact Kramers
doublets at zero field, the closed-form zero-field splitting against
full diagonalisation, the perturbative transverse-field qubit
frequency (Rosenthal Eq. B9) against the exact eigenvalues, and the
exact orthonormality of the frame rotation."""
import numpy as np
import pytest

from vacspin import (h_manifold, lab_to_spin, qubit_frequency,
                     qubit_frequency_perpendicular, snv_rosenthal2023,
                     solve, zero_field_splitting)


def test_hermitian_for_random_inputs():
    rng = np.random.default_rng(1)
    for _ in range(20):
        lam, ux, uy = rng.uniform(1, 3000, 3)
        f, d = rng.uniform(0, 0.5, 2)
        b = rng.normal(0, 0.2, 3)
        h = h_manifold(lam, ux, uy, f, d, b)
        assert np.allclose(h, h.conj().T, rtol=0, atol=1e-12)


def test_kramers_doublets_and_closed_form_splitting():
    """At B = 0 each manifold is two exactly degenerate doublets split
    by Delta = sqrt(lam^2 + 4(Ux^2 + Uy^2)) -- machine precision, both
    manifolds, with and without strain."""
    rng = np.random.default_rng(2)
    for _ in range(10):
        lam = rng.uniform(10, 3000)
        ux, uy = rng.uniform(0, 500, 2)
        e = np.linalg.eigvalsh(h_manifold(lam, ux, uy, 0.3, 0.1,
                                          (0, 0, 0)))
        assert abs(e[1] - e[0]) < 1e-9          # Kramers degeneracy
        assert abs(e[3] - e[2]) < 1e-9
        delta = zero_field_splitting(lam, ux, uy)
        assert abs((e[2] - e[0]) - delta) < 1e-9 * max(delta, 1.0)


def test_rotation_is_exactly_orthonormal():
    p = snv_rosenthal2023()
    rng = np.random.default_rng(3)
    for _ in range(10):
        v = rng.normal(0, 1, 3)
        w = lab_to_spin(v, p.theta_rad, p.phi_rad)
        assert abs(np.linalg.norm(w) - np.linalg.norm(v)) < 1e-12
    # basis vectors map to an orthonormal triad
    m = np.column_stack([lab_to_spin(e, p.theta_rad, p.phi_rad)
                         for e in np.eye(3)])
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)


def test_perturbative_transverse_qubit_frequency():
    """Eq. B9, f = 2 gamma B_perp Ups / Delta, against full
    diagonalisation with the field applied purely transversally in the
    spin frame -- two independent code paths, 2% at 50 mT (the same
    yardstick the companion pipeline uses)."""
    p = snv_rosenthal2023()
    b_perp = 0.050
    e = np.linalg.eigvalsh(h_manifold(p.lam_g, p.ups_g, 0.0, p.f_g,
                                      p.delta_g, (b_perp, 0.0, 0.0)))
    f_diag = e[1] - e[0]
    f_pert = qubit_frequency_perpendicular(p, b_perp)
    assert abs(f_pert - f_diag) < 0.02 * f_diag
    # the perturbative form is exactly linear in B
    assert abs(qubit_frequency_perpendicular(p, 2 * b_perp)
               - 2 * f_pert) < 1e-12


def test_solve_orders_and_refuses():
    p = snv_rosenthal2023()
    eg, vg, ee, ve = solve(p, [0.05, 0.02, 0.1])
    assert np.all(np.diff(eg) >= -1e-12)
    assert np.all(np.diff(ee) >= -1e-12)
    # eigenvectors are orthonormal columns
    assert np.allclose(vg.conj().T @ vg, np.eye(4), atol=1e-10)
    with pytest.raises(ValueError):
        solve(p, [np.nan, 0, 0])
    with pytest.raises(ValueError):
        solve(p, [0, 0, 0.1], strain_scale=-1.0)
    with pytest.raises(ValueError):
        qubit_frequency_perpendicular(p, 0.05, manifold="middle")


def test_qubit_frequency_zeeman_scale():
    """With zero strain and zero quenching, a transverse field splits
    the lowest doublet by ~ 0 (spin-orbit protection), while a field
    along z splits it by exactly gamma (1 + 2 delta) B to first order
    -- both statements checked against the exact eigenvalues."""
    lam = 800.0
    e_z = np.linalg.eigvalsh(h_manifold(lam, 0.0, 0.0, 0.0, 0.0,
                                        (0.0, 0.0, 0.1)))
    from vacspin import GAMMA_GHZ_PER_T
    assert abs((e_z[1] - e_z[0]) - GAMMA_GHZ_PER_T * 0.1) < 1e-9
    e_x = np.linalg.eigvalsh(h_manifold(lam, 0.0, 0.0, 0.0, 0.0,
                                        (0.1, 0.0, 0.0)))
    # transverse splitting is second order: << the Zeeman scale
    assert (e_x[1] - e_x[0]) < 0.01 * GAMMA_GHZ_PER_T * 0.1
