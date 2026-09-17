"""Remote entanglement budgets: what two interfaces can do together.

The standard two-photon heralding scheme (S. D. Barrett and P. Kok,
Phys. Rev. A 71, 060310(R) (2005)) succeeds when BOTH nodes deliver
their photon through their full detection chains and the Bell
measurement projects onto the useful half of the two-photon space:

    P_success = (1/2) eta_A eta_B

per attempt, with eta the end-to-end detected-photon efficiency of
each node (the `eta` of its `CavityInterface`). The factor 1/2 is the
intrinsic ceiling of linear-optics Bell analysis on two photons; it is
not an engineering loss and no detector upgrade removes it -- stated
here so nobody budgets it away.

`entanglement_rate` multiplies by the attempt rate; an attempt takes
at least one optical lifetime plus reset, so the attempt rate is an
input you own, not a number this package guesses.
"""
from __future__ import annotations

__all__ = ["barrett_kok_success", "entanglement_rate"]


def barrett_kok_success(eta_a, eta_b=None):
    """Success probability per attempt of two-photon heralding:
    (1/2) eta_A eta_B (Barrett-Kok). eta_b defaults to eta_a
    (symmetric nodes)."""
    if eta_b is None:
        eta_b = eta_a
    for name, v in (("eta_a", eta_a), ("eta_b", eta_b)):
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError(f"{name} must be in [0, 1]")
    return 0.5 * float(eta_a) * float(eta_b)


def entanglement_rate(attempt_rate_hz, eta_a, eta_b=None):
    """Heralded entanglement rate: attempt_rate x (1/2) eta_A eta_B.
    The attempt rate is bounded by your reset and communication times;
    supply it, this package does not guess it."""
    if attempt_rate_hz <= 0:
        raise ValueError("attempt_rate_hz must be positive")
    return float(attempt_rate_hz) * barrett_kok_success(eta_a, eta_b)
