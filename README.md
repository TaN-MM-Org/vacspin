# vacspin

[![PyPI](https://img.shields.io/pypi/v/vacspin)](https://pypi.org/project/vacspin/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22819698-blue)](https://doi.org/10.5281/zenodo.22819698) [![tests](https://github.com/TaN-MM-Org/vacspin/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/vacspin/actions)

How well can a single trapped spin in diamond talk to light? A
vacancy colour centre (SiV-, GeV-, SnV-, PbV-) holds one electron
spin whose optical transitions depend on strain, magnetic field and
the cavity around it -- and every quantum-network number you care
about (how many photons before the spin flips, how fast you can read
it out, how often two nodes entangle) follows from that chain.
`vacspin` computes the whole chain: the spin Hamiltonian, the optical
transitions and their cyclicity, the cavity Purcell budget, the exact
single-shot readout statistics, and the remote-entanglement rate --
with every physics claim pinned in the test suite to a closed form, a
published measurement, or two independent code paths.

## Install

```
pip install vacspin        # NumPy only
```

For development: clone the repository and `pip install -e .[test]`.

## The chain in one example

```python
import vacspin as vs

spin = vs.snv_rosenthal2023()                  # cited SnV- parameters
b = vs.field_on_circle(147.0, 0.125)           # 125 mT at 147 degrees
f_q = vs.qubit_frequency(spin, b)              # -> 3.64 GHz (measured 3.677)

ci = vs.CavityInterface(
    q_loaded=500, v_rel=0.68, eta_wg=0.9996,   # your cavity
    budget=vs.snv_emission(),                  # cited emission budget
    lambda0=2244.0,                            # measured cyclicity
    omega_q_hz=f_q * 1e9,
    xi_pol=vs.xi_pol_overlap(), xi_pos=0.5,
    t_link=0.99, eta_chip=0.8, eta_det=0.9).require_valid()

out = vs.fidelity(ci.eta, ci.lambda_cav, ci.gamma_cav, tau=50e-9, s=2.0)
print(ci.f_c, ci.eta, out["fidelity"])         # Purcell ~19, eta ~0.65
```

`require_valid()` is not decoration: the Purcell formulas assume the
bad-cavity, weak-coupling, spin-selective regime, and the interface
refuses -- naming the failed condition -- rather than extrapolating
outside it.

## What is inside

- **The spin Hamiltonian** (`h_manifold`, `solve`,
  `qubit_frequency`): the effective Hamiltonian all group-IV centres
  share (spin-orbit + strain + Zeeman; Rosenthal et al., PRX 13,
  031022 (2023)), in pure NumPy, with the exact frame rotation from
  your lab coordinates, the closed-form zero-field splitting, a
  strain-tuning axis, and the perturbative transverse-field formula
  held against full diagonalisation.
- **Transitions and cyclicity** (`transition_table`, `cyclicity`,
  `rabi_rate`): all sixteen optical transitions with their strengths
  (which obey an exact sum rule the tests assert), the cyclicity that
  decides how many photons a spin emits before it flips -- including
  its exact divergence when the field is aligned with the symmetry
  axis -- and microwave Rabi rates.
- **The cavity budget** (`CavityInterface`, `purcell_max`): the
  transition-resolved Purcell factor, the lifetime-reduction factor,
  the fraction of decays that reach your detector through the full
  chain (cavity mode, out-coupling port, link, chip, detector), the
  cavity-boosted cyclicity (the cavity enhances only the
  spin-conserving line; its partner is detuned by the qubit
  frequency), and the coupling rate g with
  F = 4 g^2 / (kappa gamma) holding as an exact identity.
- **Single-shot readout** (`fidelity`, `required_efficiency`,
  `required_window`): exact counting statistics -- the bright record
  is a Poisson photon stream terminated by the spin flip, evaluated
  in closed form, no numerical quadrature -- with background counts,
  off-resonant scattering of the dark spin, and the dark-state
  switch-on channel, at the optimal integer threshold. The inverse
  tools answer the questions an experiment actually asks (what
  efficiency, how long a window) and refuse when the target is
  unreachable, naming the best achievable value.
- **Remote entanglement** (`barrett_kok_success`,
  `entanglement_rate`): the two-photon heralding budget, with its
  intrinsic factor 1/2 stated as a ceiling no detector removes.

## Cited parameters

No physical number is made up, and none is accepted without a source
-- the `reference` field of both parameter dataclasses is mandatory.

- `snv_rosenthal2023()` + `snv_emission()`: the SnV- device values of
  Rosenthal et al., PRX 13, 031022 (2023) with the Thiering-Gali
  quenching factors, and the emission budget of arXiv:2403.13110 /
  Goerlitz et al. / Lee et al. (arXiv:2511.05740). Validated against
  measurement in the tests with no free parameters: the 902.98 GHz
  splitting, the 3.677 GHz qubit frequency, the cyclicity landscape
  (2244 near alignment, 8.6 at 53 degrees), the MHz Rabi scale.
- `siv_hepp2014()`: the unstrained SiV- splittings (50 and 260 GHz)
  of Hepp et al., PRL 112, 036405 (2014). Strain and orbital
  quenching are sample-specific and deliberately not shipped; the
  reference string says so.

For GeV-, PbV-, or your own sample, populate `SpinParameters` and
`EmissionBudget` from your measurements or the literature; the
provenance travels with every prediction.

## How it is checked

46 tests (Python 3.9-3.13, run in CI on every push), every claim
anchored to an exact result, a published measurement, or two
independent code paths -- never a stored number. Highlights: exact
Kramers doublets and the closed-form splitting against full
diagonalisation; the measured SnV- landscape reproduced with no free
parameters; the exact aligned-field cyclicity divergence; the dipole
sum rule (total strength exactly 3 per ground state) as an identity;
the Purcell relation F = 4 g^2/(kappa gamma) as an exact round trip;
exact Lorentzian and polarisation-overlap limits; the photon budget
bounded and monotone; the bright-count distribution hitting its
Poisson and geometric limits at machine precision (the closed-form
recurrence makes them exact, not approximate); the measured confocal
operating point (about 4 detected photons at 0.2% efficiency)
reproduced; the source study's design point (Purcell factor ~19,
end-to-end efficiency ~0.65, >98% fidelity at the optimal window,
a >100x speedup over confocal readout) recovered end to end from its
stated inputs; and inverse-tool round trips and refusals throughout.

## Honest limits

Deliberate scope, designed out with reasons: no photonic device
design (cavity Q, mode volume and taper transfer are inputs you
simulate or measure -- the companion study's pipeline shows how);
the readout model is the two-level bad-cavity treatment, and
`require_valid()` refuses the strong-coupling regime rather than
mis-describing it; spin coherence times (T1, T2) are
sample-dependent measurements, not shipped constants; and the
switch-on dark channel is treated by a union bound, so the reported
fidelity is the conservative side of the exact answer.

## Associated study

> T. M. Mahim, M. M. Rahman and A. S. M. Mohsin, "Fast single-shot
> readout of tin-vacancy spins with an overcoupled diamond nanocavity
> on thin-film lithium niobate" (submitted to Optics Express, 2026);
> pipeline: https://github.com/Tanvir-Mahmud-Mahim/snv-tfln-cavity-interface

This package is the general-purpose engine; the paper repository
holds the device design (FEM/EME/GME photonics) and reproduces the
specific study. Formalism: Rosenthal et al., PRX 13, 031022 (2023)
and arXiv:2403.13110; transition-resolved cavity conventions: Lee et
al., arXiv:2511.05740; heralding: Barrett and Kok, PRA 71, 060310(R)
(2005).

## Support and governance

Written and maintained by Tanvir Mahmud Mahim (Department of
Electrical and Electronic Engineering, BRAC University), who reviews
every change and takes the final decision on scope and releases.
Design questions are discussed in the open in issues and pull
requests, and the standing rule of
[CONTRIBUTING.md](CONTRIBUTING.md) binds the maintainer exactly as it
binds contributors: a change that touches physics arrives with a
test, and a constant arrives with its source.

Support runs through the
[issue tracker](https://github.com/TaN-MM-Org/vacspin/issues). Usage
questions are welcome alongside bug reports; a docstring that left a
unit or a sign convention unclear is treated as a documentation bug,
not user error. While the version is below 1.0 the API may still move
between minor versions; such changes are called out in the release
notes.

## License

Apache-2.0. Every release is archived on Zenodo under the concept DOI
[10.5281/zenodo.22819698](https://doi.org/10.5281/zenodo.22819698),
which always resolves to the latest version.
