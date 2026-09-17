"""The group-IV effective Hamiltonian: 4x4 per manifold, pure NumPy.

Implements the effective ground/excited-manifold Hamiltonian shared by
all group-IV vacancy centres in diamond, exactly as written for the
SnV- in Rosenthal et al., Phys. Rev. X 13, 031022 (2023), Appendix B
(Eqs. B1-B5), in the basis {|ex up>, |ex dn>, |ey up>, |ey dn>}
(orbital doublet x spin 1/2):

    H = H_SO + H_JT + H_Z + H_L

    H_SO = -(lam/2) [[0, i], [-i, 0]] (x) sigma_z          (spin-orbit)
    H_JT = [[Ux, Uy], [Uy, -Ux]] (x) 1                     (strain/JT)
    H_Z  = (gamma/2) 1 (x) [[(1+2 delta) Bz, Bx - i By],
                            [Bx + i By, -(1+2 delta) Bz]]  (spin Zeeman)
    H_L  = (gamma f/2) Bz [[0, i], [-i, 0]] (x) 1     (orbital Zeeman)

with all energies in GHz (angular frequency over 2 pi), fields in
tesla, and gamma/2pi = 28.0 GHz/T the electron gyromagnetic ratio. The
field enters in the SPIN FRAME whose z axis lies along the centre's
<111> high-symmetry axis; `lab_to_spin` performs the exact rotation
from your lab frame using the (theta, phi) of the parameter set.

Exact structure this module's tests lean on rather than stored
numbers: the Hamiltonian is Hermitian for every input; at zero field
each manifold consists of two exact Kramers doublets split by the
closed form Delta = sqrt(lam^2 + 4 (Ux^2 + Uy^2)); the perturbative
qubit frequency for a purely transverse field (Rosenthal Eq. B9),
f = 2 gamma B_perp Ups / Delta, agrees with full diagonalisation; and
the frame rotation is exactly orthonormal, so |B| is preserved to
machine precision.
"""
from __future__ import annotations

import numpy as np

from .params import SpinParameters

__all__ = ["GAMMA_GHZ_PER_T", "h_manifold", "lab_to_spin", "solve",
           "zero_field_splitting", "qubit_frequency",
           "qubit_frequency_perpendicular"]

# electron gyromagnetic ratio gamma/2pi (GHz/T), gamma = g_s mu_B / hbar
# with g_s ~ 2: the standard value used by Rosenthal et al. (2023).
GAMMA_GHZ_PER_T = 28.0

_I2 = np.eye(2)
_SZ = np.array([[1.0, 0.0], [0.0, -1.0]])
_ORB_Y = np.array([[0.0, 1j], [-1j, 0.0]])   # orbital angular momentum


def h_manifold(lam, ups_x, ups_y, f, delta, b_spin):
    """4x4 manifold Hamiltonian (GHz) for a field b_spin = (Bx, By, Bz)
    in tesla, given in the spin frame."""
    bx, by, bz = (float(b) for b in b_spin)
    h_so = -0.5 * lam * np.kron(_ORB_Y, _SZ)
    h_jt = np.kron(np.array([[ups_x, ups_y], [ups_y, -ups_x]]), _I2)
    zee = np.array([[(1 + 2 * delta) * bz, bx - 1j * by],
                    [bx + 1j * by, -(1 + 2 * delta) * bz]])
    h_z = 0.5 * GAMMA_GHZ_PER_T * np.kron(_I2, zee)
    h_l = 0.5 * GAMMA_GHZ_PER_T * f * bz * np.kron(_ORB_Y, _I2)
    return h_so + h_jt + h_z + h_l


def lab_to_spin(v_lab, theta_rad, phi_rad):
    """Rotate a lab-frame vector into the spin frame whose z axis
    points along the centre's <111> axis at polar angle theta, azimuth
    phi. Rows of the rotation are the spin-frame basis vectors in the
    lab frame; the rotation is exactly orthonormal."""
    ct, st = np.cos(theta_rad), np.sin(theta_rad)
    cp, sp = np.cos(phi_rad), np.sin(phi_rad)
    ex = np.array([ct * cp, ct * sp, -st])
    ey = np.array([-sp, cp, 0.0])
    ez = np.array([st * cp, st * sp, ct])
    return np.vstack([ex, ey, ez]) @ np.asarray(v_lab, dtype=float)


def zero_field_splitting(lam, ups_x, ups_y=0.0):
    """Exact zero-field splitting of one manifold:
    Delta = sqrt(lam^2 + 4 (Ux^2 + Uy^2))."""
    return float(np.sqrt(lam ** 2 + 4.0 * (ups_x ** 2 + ups_y ** 2)))


def solve(params: SpinParameters, b_lab, strain_scale=1.0):
    """Eigenstructure of both manifolds at a lab-frame field (tesla).

    strain_scale multiplies the strain of both manifolds (the strain-
    engineering axis); the strain axis is taken along x, which fixes
    the azimuthal origin of the spin frame.

    Returns (evals_g, evecs_g, evals_e, evecs_e): energies in GHz
    sorted ascending, eigenvectors as columns.
    """
    if not np.all(np.isfinite(b_lab)):
        raise ValueError("magnetic field must be finite")
    s = float(strain_scale)
    if s < 0:
        raise ValueError("strain_scale must be >= 0")
    bs = lab_to_spin(b_lab, params.theta_rad, params.phi_rad)
    hg = h_manifold(params.lam_g, s * params.ups_g, 0.0,
                    params.f_g, params.delta_g, bs)
    he = h_manifold(params.lam_e, s * params.ups_e, 0.0,
                    params.f_e, params.delta_e, bs)
    eg, vg = np.linalg.eigh(hg)
    ee, ve = np.linalg.eigh(he)
    return eg, vg, ee, ve


def qubit_frequency(params: SpinParameters, b_lab, strain_scale=1.0):
    """Ground-state qubit frequency (GHz): the splitting of the lowest
    Kramers doublet at the given lab-frame field."""
    eg, _, _, _ = solve(params, b_lab, strain_scale)
    return float(eg[1] - eg[0])


def qubit_frequency_perpendicular(params: SpinParameters, b_perp_t,
                                  manifold="ground"):
    """Perturbative qubit frequency for a purely TRANSVERSE field
    (Rosenthal et al. Eq. B9): f = 2 gamma B_perp Ups / Delta, valid
    for gamma B_perp << Delta. The tests hold this against full
    diagonalisation; use `qubit_frequency` for anything quantitative.
    """
    b = float(b_perp_t)
    if b < 0:
        raise ValueError("b_perp_t must be >= 0")
    if manifold == "ground":
        lam, ups = params.lam_g, params.ups_g
    elif manifold == "excited":
        lam, ups = params.lam_e, params.ups_e
    else:
        raise ValueError('manifold must be "ground" or "excited"')
    delta = zero_field_splitting(lam, ups)
    return 2.0 * GAMMA_GHZ_PER_T * b * ups / delta
