"""Cited parameter sets for group-IV colour-centre spin-photon interfaces.

No physical number in this package is made up, and none is accepted
without a source: both dataclasses refuse to exist without a
`reference`. Two sets ship, and each ships only what its source states:

* `snv_rosenthal2023` / `snv_emission`: the negatively charged
  tin-vacancy (SnV-) centre in diamond, the device-fit values of
  Table I of J. A. Rosenthal et al., Phys. Rev. X 13, 031022 (2023)
  (arXiv:2306.13199), with the ab initio orbital-quenching reduction
  factors of G. Thiering and A. Gali, Phys. Rev. X 8, 021063 (2018),
  and the emission budget of Rosenthal et al., arXiv:2403.13110
  (lifetime, quantum efficiency) with the Debye-Waller factor of
  Goerlitz et al. and the C/D branching ratio of Lee et al.,
  arXiv:2511.05740. This set is validated against measured numbers in
  the test suite (zero-field splittings, qubit frequency, cyclicity,
  Rabi rate) with no free parameters.
* `siv_hepp2014`: the silicon-vacancy (SiV-) centre, unstrained
  spin-orbit splittings of C. Hepp et al., Phys. Rev. Lett. 112,
  036405 (2014) (arXiv:1310.3106): 50 GHz ground, 260 GHz excited,
  737 nm zero-phonon line. Strain and orbital quenching are
  sample-specific and deliberately NOT shipped for SiV: the set
  carries zero strain and zero quenching, stated in its reference
  string, and you supply your measured values for field-dependent
  work.

For any other centre (GeV-, PbV-, or your own SnV sample), populate the
dataclasses from your measurements or the literature; the mandatory
`reference` field keeps the provenance attached to every prediction.
"""
from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["SpinParameters", "EmissionBudget", "snv_rosenthal2023",
           "snv_emission", "siv_hepp2014"]


def _check_ref(reference):
    if not isinstance(reference, str) or len(reference.strip()) < 8:
        raise ValueError("a real `reference` string is required: every "
                         "physical number in this package carries its "
                         "source")


@dataclasses.dataclass(frozen=True)
class SpinParameters:
    """Effective-Hamiltonian parameters of one group-IV centre.

    All group-IV vacancy centres (SiV-, GeV-, SnV-, PbV-) share the
    same D3d level structure: an orbital doublet times spin 1/2 in
    each manifold, split by spin-orbit coupling and strain. Units:
    every energy in GHz (angular frequency over 2 pi), fields in
    tesla.

    lam_g, lam_e : spin-orbit constants of ground/excited manifold.
    ups_g, ups_e : Jahn-Teller/strain energies (the x component; the
        strain axis fixes the azimuthal origin of the spin frame).
    f_g, f_e : orbital Zeeman quenching factors (gL * p).
    delta_g, delta_e : spin g-factor anisotropies (gL * delta_p).
    theta_rad, phi_rad : polar and azimuthal angle of the centre's
        high-symmetry <111> axis in your lab frame.
    reference : where every number comes from. Required, on purpose.
    """

    lam_g: float
    ups_g: float
    lam_e: float
    ups_e: float
    f_g: float
    f_e: float
    delta_g: float
    delta_e: float
    theta_rad: float
    phi_rad: float
    reference: str

    def __post_init__(self):
        _check_ref(self.reference)
        for name in ("lam_g", "lam_e"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive (GHz)")
        for name in ("ups_g", "ups_e"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0 (GHz); the strain "
                                 "axis convention absorbs the sign")


@dataclasses.dataclass(frozen=True)
class EmissionBudget:
    """Optical emission budget of one centre (bulk values).

    tau0_s : total excited-state lifetime (s).
    eta_q : radiative quantum efficiency in [0, 1].
    eta_dw : Debye-Waller (zero-phonon) fraction in [0, 1].
    eta_br : branching ratio into the cavity-coupled transition, [0, 1].
    zpl_nm : zero-phonon-line wavelength (nm).
    reference : where the numbers come from. Required.
    """

    tau0_s: float
    eta_q: float
    eta_dw: float
    eta_br: float
    zpl_nm: float
    reference: str

    def __post_init__(self):
        _check_ref(self.reference)
        if not (self.tau0_s > 0 and np.isfinite(self.tau0_s)):
            raise ValueError("tau0_s must be a positive lifetime in s")
        if not (self.zpl_nm > 0 and np.isfinite(self.zpl_nm)):
            raise ValueError("zpl_nm must be a positive wavelength in nm")
        for name in ("eta_q", "eta_dw", "eta_br"):
            v = getattr(self, name)
            if not (0.0 < v <= 1.0):
                raise ValueError(f"{name} must be in (0, 1]")

    @property
    def gamma0(self):
        """Total decay rate 1/tau0 (1/s)."""
        return 1.0 / self.tau0_s

    @property
    def radiative_fraction(self):
        """eta_q * eta_dw * eta_br: the fraction of decays that emit a
        zero-phonon photon on the cavity-coupled transition."""
        return self.eta_q * self.eta_dw * self.eta_br


def snv_rosenthal2023() -> SpinParameters:
    """SnV- device values, Table I of Rosenthal et al., Phys. Rev. X
    13, 031022 (2023), with quenching reduction factors from Thiering
    and Gali, Phys. Rev. X 8, 021063 (2018):
    f = gL p, delta = gL delta_p. The dipole angles are the same
    paper's device fit (theta fixed at 125.3 deg, phi fitted 37.33
    deg). Validated against measured splittings, qubit frequency,
    cyclicity and Rabi rate in this package's test suite."""
    gL_g, gL_e = 0.363, 0.581
    p_g, p_e = 0.471, 0.125
    dp_g, dp_e = 0.042, 0.303
    return SpinParameters(
        lam_g=830.15, ups_g=177.67, lam_e=2988.0, ups_e=134.00,
        f_g=gL_g * p_g, f_e=gL_e * p_e,
        delta_g=gL_g * dp_g, delta_e=gL_e * dp_e,
        theta_rad=np.deg2rad(125.3), phi_rad=np.deg2rad(37.33),
        reference="Rosenthal et al., PRX 13, 031022 (2023), Table I "
                  "(device values); quenching: Thiering & Gali, PRX 8, "
                  "021063 (2018)")


def snv_emission() -> EmissionBudget:
    """SnV- emission budget: lifetime 4.5 ns and quantum efficiency
    0.8 from Rosenthal et al., arXiv:2403.13110 (Table II);
    Debye-Waller factor 0.57 (Goerlitz et al. 2020, as used by Lee et
    al.); C/D branching ratio 0.75 (Lee et al., arXiv:2511.05740);
    zero-phonon line 619 nm."""
    return EmissionBudget(
        tau0_s=4.5e-9, eta_q=0.80, eta_dw=0.57, eta_br=0.75,
        zpl_nm=619.0,
        reference="Rosenthal et al., arXiv:2403.13110 (tau, eta_q); "
                  "Goerlitz et al. 2020 (eta_DW); Lee et al., "
                  "arXiv:2511.05740 (eta_BR)")


def siv_hepp2014(theta_rad=None, phi_rad=None) -> SpinParameters:
    """SiV- unstrained model values of Hepp et al., PRL 112, 036405
    (2014): spin-orbit splittings 50 GHz (ground) and 260 GHz
    (excited) at zero strain. Strain and orbital quenching are
    sample-specific and NOT shipped: this set carries ups = 0 and
    f = delta = 0, stated here on purpose -- supply your measured
    values for quantitative field-dependent work. Default orientation:
    the <111> axis of a (001)-oriented sample (theta = arccos(1/sqrt 3),
    phi = 45 deg), exact crystallography, overridable."""
    if theta_rad is None:
        theta_rad = float(np.arccos(1.0 / np.sqrt(3.0)))
    if phi_rad is None:
        phi_rad = float(np.pi / 4)
    return SpinParameters(
        lam_g=50.0, ups_g=0.0, lam_e=260.0, ups_e=0.0,
        f_g=0.0, f_e=0.0, delta_g=0.0, delta_e=0.0,
        theta_rad=theta_rad, phi_rad=phi_rad,
        reference="Hepp et al., PRL 112, 036405 (2014): unstrained "
                  "splittings 50/260 GHz, ZPL 737 nm; strain and "
                  "orbital quenching deliberately not shipped (sample-"
                  "specific) -- supply measured values")
