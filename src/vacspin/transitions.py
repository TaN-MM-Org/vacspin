"""Optical transitions, cyclicity and microwave control.

Dipole operators between the two manifolds in the orbital basis
(Rosenthal et al., PRX 13, 031022 (2023), Eq. B12; identity in spin):

    p_x = sigma_z (orbital),  p_y = -sigma_x (orbital),  p_z = 1.

`transition_strength` sums |<e| p_i |g>|^2 over the three components
(unpolarised detection, Eqs. B13-B14). The cyclicity Lambda is the
ratio of spin-conserving to spin-flipping strength from the lowest
ground doublet to its spin-conserving partner in the chosen excited
doublet (Eq. B15) -- the number that decides how many photons a spin
can emit before it flips, and therefore everything about readout.

Exact structure the tests lean on: for a field exactly along the
centre's <111> axis the spin-flip strength vanishes and the cyclicity
diverges (spin is a good quantum number); tilting the field switches
the flip channel on. The shipped SnV- set reproduces the measured
cyclicity landscape of Rosenthal et al., arXiv:2403.13110 (2244 near
alignment at 180 mT, 8.6 at 53 deg) within the measurement scatter,
with no free parameters.
"""
from __future__ import annotations

import numpy as np

from .hamiltonian import GAMMA_GHZ_PER_T, lab_to_spin, solve
from .params import SpinParameters

__all__ = ["transition_strength", "transition_table", "cyclicity",
           "rabi_rate", "field_on_circle"]

_I2 = np.eye(2)
_PX = np.kron(np.array([[1.0, 0.0], [0.0, -1.0]]), _I2)
_PY = np.kron(np.array([[0.0, -1.0], [-1.0, 0.0]]), _I2)
_PZ = np.kron(_I2, _I2)


def transition_strength(ket_g, ket_e):
    """|<e|p|g>|^2 summed over the three dipole components."""
    return float(sum(abs(np.vdot(ket_e, p @ ket_g)) ** 2
                     for p in (_PX, _PY, _PZ)))


def field_on_circle(zeta_deg, b_mag_t, plane="xz"):
    """Lab-frame field of magnitude b_mag_t swept on a great circle:
    plane='xz' gives B = |B| (cos zeta, 0, sin zeta) (the magnet sweep
    of Rosenthal et al.); 'xy' and 'yz' are the other two planes."""
    z = np.deg2rad(float(zeta_deg))
    c, s = np.cos(z), np.sin(z)
    if plane == "xz":
        return float(b_mag_t) * np.array([c, 0.0, s])
    if plane == "xy":
        return float(b_mag_t) * np.array([c, s, 0.0])
    if plane == "yz":
        return float(b_mag_t) * np.array([0.0, c, s])
    raise ValueError('plane must be "xz", "xy" or "yz"')


def transition_table(params: SpinParameters, b_lab, strain_scale=1.0):
    """Energies and strengths of all 16 ground-to-excited transitions.

    Returns dict(energies_ghz (4, 4), strengths (4, 4), evals_g,
    evals_e): entry [i, j] is the |g_i> -> |e_j> transition. Energies
    are the manifold differences (add the ZPL frequency for absolute
    optical frequencies)."""
    eg, vg, ee, ve = solve(params, b_lab, strain_scale)
    energies = ee[None, :] - eg[:, None]
    strengths = np.empty((4, 4))
    for i in range(4):
        for j in range(4):
            strengths[i, j] = transition_strength(vg[:, i], ve[:, j])
    return dict(energies_ghz=energies, strengths=strengths,
                evals_g=eg, evals_e=ee)


def cyclicity(params: SpinParameters, b_lab, strain_scale=1.0,
              manifold="lower"):
    """Cyclicity Lambda of the lowest ground doublet.

    The spin-conserving partner of |g0> is whichever state of the
    chosen excited doublet ('lower': the C/D doublet, states 0-1;
    'upper': the A/B doublet, states 2-3) has the larger dipole
    strength; Lambda is its strength from |g0> divided by its strength
    from |g1>. Diverges for a field exactly along the <111> axis
    (asserted, not stated, in the tests)."""
    eg, vg, ee, ve = solve(params, b_lab, strain_scale)
    if manifold == "lower":
        i0 = 0
    elif manifold == "upper":
        i0 = 2
    else:
        raise ValueError('manifold must be "lower" or "upper"')
    s0 = transition_strength(vg[:, 0], ve[:, i0])
    s1 = transition_strength(vg[:, 0], ve[:, i0 + 1])
    j = i0 if s0 >= s1 else i0 + 1
    p_conserve = transition_strength(vg[:, 0], ve[:, j])
    p_flip = transition_strength(vg[:, 1], ve[:, j])
    if p_flip == 0.0:
        return np.inf
    return p_conserve / p_flip


def rabi_rate(params: SpinParameters, b_lab, b_drive_lab,
              strain_scale=1.0):
    """Microwave Rabi rate Omega/2pi (GHz) between the two lowest
    ground states for a drive field b_drive_lab (tesla, lab frame):
    |<g1| H_drive |g0>| with the spin-Zeeman drive Hamiltonian
    (Rosenthal Eqs. B10-B11)."""
    eg, vg, ee, ve = solve(params, b_lab, strain_scale)
    bs = lab_to_spin(b_drive_lab, params.theta_rad, params.phi_rad)
    bx, by, bz = bs
    zee = np.array([[bz, bx - 1j * by], [bx + 1j * by, -bz]])
    h_mw = 0.5 * GAMMA_GHZ_PER_T * np.kron(_I2, zee)
    return float(abs(np.vdot(vg[:, 1], h_mw @ vg[:, 0])))
