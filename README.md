# vacspin

[![PyPI](https://img.shields.io/pypi/v/vacspin)](https://pypi.org/project/vacspin/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22819698-blue)](https://doi.org/10.5281/zenodo.22819698) [![tests](https://github.com/TaN-MM-Org/vacspin/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/vacspin/actions)

`vacspin` is a Python package for working out how well a single
electron spin trapped in diamond can be read out with light and linked
to another spin. The spin sits in a **group-IV vacancy colour centre**:
a tin, silicon, germanium or lead atom next to a missing carbon atom
(written SnV-, SiV-, GeV-, PbV-). Such a centre can store one quantum
bit and send it out as a photon, which is why it is studied as a node
of a quantum network.

Starting from the numbers that describe one centre, and the cavity and
optics around it, the package answers questions such as:

- At a given magnetic field, what is the qubit frequency, and how many
  photons can the spin emit before it flips by accident?
- How much does an optical cavity speed up the emission, and what
  fraction of the photons reach the detector?
- With that detector, how reliably can one measurement tell the two
  spin states apart, and in how short a time window?
- What detection efficiency, or what window, does a target reliability
  need?
- How often would two such nodes become entangled?
- From your own measured frequencies, what are the parameters of *your*
  centre, with error bars -- and will a planned set of measurements be
  able to determine them at all?

The whole chain runs on NumPy alone. Results are checked by automated
tests against exact formulas, published measurements or a second
calculation done a different way (see
[How the results are checked](#how-the-results-are-checked), which
also gives each tolerance). When the physics behind a formula no longer
applies, or a question cannot be answered from the data given, the
package stops with an error that says why, rather than returning a
number that looks fine but is not.

## Contents

- [A short guide to the words used here](#a-short-guide-to-the-words-used-here)
- [Install, units and conventions](#install-units-and-conventions)
- [Examples](#examples) (each with the output it prints)
- [Cited parameter sets](#cited-parameter-sets)
- [What is in the package](#what-is-in-the-package)
- [When it refuses, and why](#when-it-refuses-and-why)
- [How the results are checked](#how-the-results-are-checked)
- [Corrections in earlier versions](#corrections-in-earlier-versions)
- [Limits](#limits)
- [Where it comes from](#where-it-comes-from)
- [Citing, support and license](#citing-support-and-license)

## A short guide to the words used here

- **Ground and excited manifold** -- the centre has a group of four
  low-energy states (ground) and four states one optical photon higher
  (excited). In each group the four states form two pairs.
- **Kramers doublet** -- one such pair. At zero magnetic field the two
  states of a pair have exactly the same energy; a field splits them.
  The lowest ground pair is the **qubit**, and its splitting is the
  **qubit frequency**.
- **Zero-field splitting** -- the energy gap between the two pairs of
  a manifold at zero field. It comes from **spin-orbit coupling**
  (`lam`, a property of the atom) and **strain** (`ups`, how the
  crystal around the centre is squeezed; it varies from sample to
  sample). The two combine as `sqrt(lam^2 + 4 ups^2)`.
- **Spin Hamiltonian** -- the 4 x 4 matrix whose eigenvalues are the
  energies of one manifold. `vacspin` builds it from spin-orbit
  coupling, strain and the magnetic field (the Zeeman terms).
- **Symmetry axis** -- each centre points along one of diamond's
  `<111>` crystal directions. Its angles in your lab frame are
  `theta_rad` and `phi_rad`.
- **Transition strength** -- how strongly light drives one
  ground-to-excited transition. There are 4 x 4 = 16 transitions.
- **Cyclicity** (`Lambda`) -- the strength of the transition that keeps
  the spin unchanged divided by the strength of the one that flips it:
  roughly, how many photons the spin emits before it flips. High
  cyclicity makes readout easy. It depends strongly on the angle
  between the field and the symmetry axis.
- **Orbital quenching** -- the crystal weakens how strongly a magnetic
  field acts on the electron's orbital motion. The published reduction
  factors enter the Zeeman factors `f_g`, `f_e`, `delta_g`, `delta_e`.
- **Emission budget** -- where the light from a decay goes. The
  **lifetime** is how long the excited state lasts; the **quantum
  efficiency** is the share of decays that give light at all; the
  **zero-phonon line** (ZPL) is the sharp colour emitted without
  shaking the crystal, and the **Debye-Waller factor** is the share of
  the light in it; the **branching ratio** is the share of that light on
  the one transition the cavity is tuned to. Their product is the
  `radiative_fraction`.
- **Purcell factor** (`F`) -- how much a small optical cavity speeds up
  the emission into its mode. It grows with the cavity's **quality
  factor** `Q` (how long light stays in it) and shrinks with its
  **mode volume** `V`.
- **Cavity linewidth** (`kappa`) and **coupling rate** (`g`) -- `kappa`
  is how fast light leaks out of the cavity, given as a frequency width
  `f_cav / Q`; `g` is how fast the centre and the cavity trade energy.
  In the **bad-cavity, weak-coupling** regime this package covers, both
  `g` and the centre's own decay are much slower than `kappa`, so the
  cavity simply speeds up the emission. The **cooperativity**
  `4 g^2 / (kappa gamma)` measures the same speed-up; here it equals the
  Purcell factor times the radiative fraction.
- **Detection efficiency** (`eta`) -- the fraction of emission events
  that end as a click on the detector, after every loss on the way.
- **Single-shot readout fidelity** -- the probability that one
  measurement names the spin state correctly. You count photons in a
  time **window** `tau`; the spin is called "bright" if the count
  reaches a **threshold**. 0.5 is a coin toss, 1 is perfect.
- **Dark counts / leak / switch-on** -- ways the dark spin state can
  still produce clicks: detector background (`noise_rate`), weak
  off-resonant scattering (`leak`), and an off-resonant excitation
  that flips the dark spin into the bright state during the window.
- **Heralded entanglement** (Barrett-Kok scheme) -- two nodes each
  send a photon; a joint detection announces that the two spins are
  now entangled.

## Install, units and conventions

```
pip install vacspin
```

It needs Python 3.9 or newer and NumPy 1.22 or newer, and nothing else.
For development: clone the repository and `pip install -e .[test]`,
then run `pytest`.

Units and conventions (each function's docstring states its own):

- **Energies and spin frequencies in GHz** (ordinary frequency, that is
  angular frequency divided by 2 pi): spin-orbit and strain constants,
  energies, qubit frequency, splittings, Rabi rates and fitted
  observables.
- **Magnetic fields in tesla**, as `(Bx, By, Bz)` vectors in your
  **lab frame**. The package rotates them into the centre's own frame
  using `theta_rad` and `phi_rad`. `field_on_circle(angle_deg, B)`
  gives a field of size `B` at an angle in degrees in the lab x-z
  plane (or the x-y or y-z plane).
- **Cavity quantities in Hz**: `omega_q_hz` (the qubit frequency in Hz,
  despite the name not an angular frequency), `kappa_hz` (the cavity
  linewidth, `f_cav / Q_L`), `g_hz` (the coupling rate, likewise given
  as an ordinary frequency: `g / (2 pi)` in Hz).
- **Decay and count rates in 1/s**: the optical decay rate `gamma`
  (`1/lifetime`), `gamma_cav`, `noise_rate`. Times in seconds.
- **Mode volume** `v_rel` in units of `(wavelength / refractive
  index)^3`.
- The **gyromagnetic ratio** is `GAMMA_GHZ_PER_T = 28.0` GHz/T.

## Examples

Each example below runs as written, and the output shown is what it
printed with vacspin 0.3.1. The cavity, detector, attempt-rate and
noise values are illustrative, not a recommended design; the centre's
parameters come from the cited sets described in
[Cited parameter sets](#cited-parameter-sets).

### 1. The spin model: splittings, qubit frequency, cyclicity

```python
import vacspin as vs

spin = vs.snv_rosenthal2023()          # cited SnV- parameter set
print(spin.reference)

# Zero-field splittings of the ground and excited manifolds (GHz)
print(f"ground splitting:  {vs.zero_field_splitting(spin.lam_g, spin.ups_g):.2f} GHz")
print(f"excited splitting: {vs.zero_field_splitting(spin.lam_e, spin.ups_e):.2f} GHz")

# Qubit frequency at 125 mT, 147 degrees in the lab x-z plane
b = vs.field_on_circle(147.0, 0.125)   # lab-frame field vector in tesla
print(f"qubit frequency:   {vs.qubit_frequency(spin, b):.3f} GHz")

# Cyclicity (photons per spin flip, roughly) at two field angles, 180 mT
for angle in (53.0, 147.0):
    lam = vs.cyclicity(spin, vs.field_on_circle(angle, 0.180))
    print(f"cyclicity at {angle:.0f} deg: {lam:.1f}")
```

```
Rosenthal et al., PRX 13, 031022 (2023), Table I (device values); quenching: Thiering & Gali, PRX 8, 021063 (2018)
ground splitting:  903.00 GHz
excited splitting: 2999.99 GHz
qubit frequency:   3.640 GHz
cyclicity at 53 deg: 7.9
cyclicity at 147 deg: 158.5
```

The measured values for this device are a 902.98 GHz ground splitting,
a 3.677 GHz qubit frequency at this field, and a cyclicity of 8.6 at
53 degrees. The model gets these with no fitted numbers; the tests
hold it to 0.2 %, 2 % and 10 % of them respectively. The measured
cyclicity near alignment is 2244. At the nominal 147-degree angle the
model gives only 158.5; the test suite reproduces 2244 (within 10 %)
for a field tilted 9 degrees from the symmetry axis, so the
difference is consistent with the field in the experiment being about
10 degrees off the axis rather than exactly at the nominal angle. That is why example 2 uses the measured
2244 as its input.

`transition_table(spin, b)` gives the energies and strengths of all 16
transitions, and `rabi_rate(spin, b, b_drive)` the microwave Rabi rate
in GHz.

### 2. The cavity budget and a fast readout

```python
import vacspin as vs

spin = vs.snv_rosenthal2023()
f_q = vs.qubit_frequency(spin, vs.field_on_circle(147.0, 0.125))   # GHz

ci = vs.CavityInterface(
    q_loaded=500, v_rel=0.68,          # cavity: loaded Q, mode volume in (lam/n)^3
    eta_wg=0.9996,                     # share of cavity light leaving by the useful port
    budget=vs.snv_emission(),          # cited SnV- emission budget
    lambda0=2244.0,                    # bare cyclicity (the measured value)
    omega_q_hz=f_q * 1e9,              # qubit frequency in Hz
    xi_pol=vs.xi_pol_overlap(), xi_pos=0.5,
    t_link=0.99, eta_chip=0.8, eta_det=0.9).require_valid()

print(f"Purcell factor F_C:      {ci.f_c:.2f}")
print(f"lifetime shortened by:   {ci.zeta:.2f}x  ({ci.tau_cav_s * 1e9:.3f} ns)")
print(f"photons into the mode:   {ci.beta:.3f}")
print(f"detected per decay, eta: {ci.eta:.3f}")
print(f"cavity linewidth kappa:  {ci.kappa_hz / 1e9:.1f} GHz")
print(f"coupling rate g:         {ci.g_hz / 1e9:.2f} GHz")
print("regime checks:", ci.validity)

out = vs.fidelity(ci.eta, ci.lambda_cav, ci.gamma_cav, tau=50e-9, s=2.0)
print(f"fidelity in 50 ns: {out['fidelity']:.5f} "
      f"(threshold {out['threshold']} photon, mean bright count {out['n_bright']:.2f})")
```

```
Purcell factor F_C:      19.29
lifetime shortened by:   7.26x  (0.620 ns)
photons into the mode:   0.909
detected per decay, eta: 0.648
cavity linewidth kappa:  968.6 GHz
coupling rate g:         7.52 GHz
regime checks: {'spin_selectivity': True, 'bad_cavity': True, 'weak_coupling': True}
fidelity in 50 ns: 0.99966 (threshold 1 photon, mean bright count 17.31)
```

`xi_pol` and `xi_pos` say how well the centre's dipole lines up with
the cavity field in direction and position (each from 0 to 1);
`xi_pol_overlap()` gives 2/3 for a centre in a `<100>`-oriented
diamond film. `eta` multiplies every loss from the emitter to the
detector: the share of emission into the cavity mode (`beta`), the
out-coupling port, the link, the chip and the detector.

This fidelity counts only the bright spin's own statistics: `leak` and
`noise_rate` are left at zero here, so the dark state never produces a
click. The test suite runs nearly the same design point (link 0.991,
out-coupling `1 - 500/1.39e6`) with a computed
off-resonant leak and a 1000 counts/s background and asserts a
fidelity above 98 % at the best of four windows (10 to 50 ns).

`require_valid()` is not decoration: the Purcell formulas assume that
the cavity loses light much faster than the emitter decays (bad
cavity), that the coupling is weak, and that the cavity picks out one
spin state (spin selectivity). If any of these fails the interface
refuses and names the failed condition:

```python
import vacspin as vs

ci = vs.CavityInterface(q_loaded=2e7, v_rel=0.68, eta_wg=0.999,
                        budget=vs.snv_emission(), lambda0=2244.0,
                        omega_q_hz=3.677e9, xi_pol=vs.xi_pol_overlap(),
                        xi_pos=0.5)
print(ci.validity)
try:
    ci.require_valid()
except ValueError as err:
    print("refused:", err)
```

```
{'spin_selectivity': False, 'bad_cavity': False, 'weak_coupling': False}
refused: interface outside its regime of validity: spin_selectivity, bad_cavity, weak_coupling failed. The Purcell/rate formulas assume the bad-cavity, weak-coupling, spin-selective regime; change Q_L, V or the operating point.
```

### 3. Readout without a cavity, and the inverse questions

```python
import vacspin as vs

gamma = vs.snv_emission().gamma0       # optical decay rate, 1/s (1 / 4.5 ns)

# Slow confocal readout: eta = 0.2 %, cyclicity 2244, 50 us window,
# background 0.2 counts per window.
n_b, n_d = vs.readout_counts(0.002, 2244.0, gamma, 50e-6, s=10.0,
                             noise_rate=0.2 / 50e-6)
print(f"mean counts: bright {n_b:.2f}, dark {n_d:.2f}")
print(f"threshold-1 fidelity: {vs.fidelity_threshold1(n_b, n_d):.4f}")

# How long a window reaches 85 %?
tau = vs.required_window(0.85, eta=0.002, lam=2244.0, gamma=gamma, s=10.0,
                         noise_rate=0.2 / 50e-6, tau_max=1.0)
print(f"window for 85 %: {tau * 1e6:.1f} us")

# What detection efficiency reaches 99 % in a 10 us window?
eta = vs.required_efficiency(0.99, lam=2244.0, gamma=gamma, tau=10e-6, s=10.0)
print(f"efficiency for 99 % in 10 us: {eta:.4f}")

# A target the model cannot reach is refused, with the best value named.
try:
    vs.required_efficiency(0.999, lam=3.0, gamma=gamma, tau=1e-6)
except ValueError as err:
    print("refused:", err)
```

```
mean counts: bright 4.22, dark 0.20
threshold-1 fidelity: 0.9020
window for 85 %: 9.4 us
efficiency for 99 % in 10 us: 0.0218
refused: target fidelity 0.999 is unreachable at this operating point: even eta = 1 gives 0.9000. Increase the cyclicity, the window, or reduce the dark counts.
```

These inputs are the confocal case the test suite uses for the
measured readout of Rosenthal et al., arXiv:2403.13110 (about 4
detected photons at eta = 0.2 %); the test asserts a bright-minus-dark
mean between 3 and 5. `s` is the **saturation parameter** of the optical drive (how
hard it is driven); `fidelity_threshold1` is the closed form for a
threshold of one photon. `required_window` and `required_efficiency`
search the full model (`fidelity`) and refuse when no value reaches the
target. Compared with example 2, the cavity turns a window of
microseconds into one of tens of nanoseconds; the test suite checks
that the required window shrinks by more than 100 times.

### 4. Remote entanglement

```python
import vacspin as vs

p = vs.barrett_kok_success(0.65)            # both nodes: eta = 0.65
print(f"success per attempt: {p:.4f}")
print(f"ceiling with perfect nodes: {vs.barrett_kok_success(1.0)}")
rate = vs.entanglement_rate(1e6, 0.65)      # 1 MHz attempt rate (your number)
print(f"entangled pairs per second: {rate:.0f}")
```

```
success per attempt: 0.2113
ceiling with perfect nodes: 0.5
entangled pairs per second: 211250
```

The success probability per attempt is `(1/2) eta_A eta_B`. The factor
1/2 is a ceiling of the two-photon scheme itself, not a loss any
detector upgrade removes. The attempt rate depends on your reset and
communication times, so you supply it; the package does not guess it.

### 5. Your sample, your numbers: plan, measure, fit

Every sample is strained differently, so the numbers that describe
your centre should come from your own measurements. The `lab` tools
work in the order a lab does: plan, measure, then calibrate.

```python
import dataclasses
import numpy as np
import vacspin as vs

p0 = vs.snv_rosenthal2023()                      # starting point

# Plan: 6 field points, alternating the two observable kinds
kinds = ["qubit", "orbital_g"] * 3
fields = [vs.field_on_circle(z, 0.15) for z in (20, 60, 100, 140, 160, 80)]

# 1. Before measuring: can these points pin down lam_g and ups_g,
#    and how large would the error bars be at 0.05 GHz per point?
info = vs.parameter_information(kinds, fields, p0,
                                vary=("lam_g", "ups_g"), sigmas_ghz=0.05)
print("identifiable:", info["identifiable"])
print({k: round(v, 2) for k, v in info["sigma"].items()})

# Zero-field splittings alone cannot separate the two:
zf = vs.parameter_information(["orbital_g"] * 4, np.zeros((4, 3)), p0,
                              vary=("lam_g", "ups_g"))
print("zero-field only identifiable:", zf["identifiable"])

# 2. Pick the 4 most informative of the 6 planned points
pick = vs.design_fields(kinds, fields, 4, p0, vary=("lam_g", "ups_g"))
print("chosen points:", pick["indices"])

# 3. "Measure": illustrative synthetic data from a sample whose
#    ups_g is 10 % higher than the cited value, plus 0.05 GHz noise.
truth = dataclasses.replace(p0, ups_g=1.1 * p0.ups_g)
rng = np.random.default_rng(1)
measured = [vs.qubit_frequency(truth, b) if k == "qubit"
            else vs.solve(truth, b)[0][2] - vs.solve(truth, b)[0][0]
            for k, b in zip(kinds, fields)]
measured = np.array(measured) + 0.05 * rng.standard_normal(len(kinds))

fit = vs.fit_spin_parameters(kinds, fields, measured, p0,
                             vary=("lam_g", "ups_g"), sigmas_ghz=0.05)
for name in ("lam_g", "ups_g"):
    print(f"{name}: {fit.values[name]:.2f} +/- {fit.sigma[name]:.2f} GHz "
          f"(true {getattr(truth, name):.2f})")
print(f"chi2 = {fit.chi2:.2f} for {fit.chi2_dof} degrees of freedom")
my_sample = fit.params                           # usable everywhere
print(my_sample.reference[:60] + "...")
```

```
identifiable: True
{'lam_g': 8.24, 'ups_g': 9.63}
zero-field only identifiable: False
chosen points: [1, 0, 5, 3]
lam_g: 824.35 +/- 8.88 GHz (true 830.15)
ups_g: 201.49 +/- 9.09 GHz (true 195.44)
chi2 = 3.19 for 4 degrees of freedom
fitted by vacspin.lab.fit_spin_parameters to 6 measured poin...
```

The observable kinds are `"qubit"` (the qubit frequency), `"orbital_g"`
(the gap between the two ground pairs) and `"orbital_e"` (the same gap
in the excited manifold), all in GHz. Zero-field splittings alone can
never separate spin-orbit coupling from strain, because they depend on
both only through `sqrt(lam^2 + 4 ups^2)`; the planner says so, and
`fit_spin_parameters` and `design_fields` refuse such a design with an
explanation. The error bars come from the standard weighted
least-squares covariance. A `chi2` close to the number of degrees of
freedom (measurements minus fitted parameters) means the fit matches
the data within the stated measurement errors. `design_fields` adds points one at a time,
each time taking the one that most increases the determinant of the
information matrix (greedy D-optimal design); this is a good
heuristic, not a proof of the best possible subset. Observation
records can be saved and reloaded with `save_observations_csv` and
`load_observations_csv`; the round trip returns identical values.

## Cited parameter sets

No physical number is made up, and none is accepted without a source:
the `reference` field of both parameter classes (`SpinParameters`,
`EmissionBudget`) is mandatory, and the package refuses an empty or
too-short one.

- `snv_rosenthal2023()` + `snv_emission()`: the SnV- device values of
  Rosenthal et al., PRX 13, 031022 (2023) (Table I) with the
  Thiering-Gali quenching factors (Thiering & Gali, PRX 8, 021063
  (2018)), and the emission budget of arXiv:2403.13110 (lifetime
  4.5 ns, quantum efficiency 0.8) / Goerlitz et al. 2020 (Debye-Waller
  factor 0.57) / Lee et al., arXiv:2511.05740 (branching ratio 0.75),
  zero-phonon line 619 nm. Checked against measurement in the tests
  with no fitted numbers: the 902.98 GHz splitting, the 3.677 GHz qubit
  frequency, the cyclicity landscape (8.6 at 53 degrees, 2244 near
  alignment), and a Rabi rate on the MHz scale (tolerances in
  [How the results are checked](#how-the-results-are-checked)).
- `siv_hepp2014()`: the unstrained SiV- splittings (50 and 260 GHz,
  zero-phonon line 737 nm) of Hepp et al., PRL 112, 036405 (2014).
  Strain and orbital quenching are sample-specific and deliberately
  not shipped (set to zero); the reference string says so.
- `gev_bhaskar2017()`: the measured GeV- splittings (152 and
  981 GHz, zero-phonon line 602 nm) of Bhaskar et al., PRL 118,
  223603 (2017), with the same deliberate omissions as SiV- (Siyushev
  et al., PRB 96, 081201(R) (2017) measured 170 GHz on a strained
  emitter): strain and quenching are your sample's numbers, and
  `vacspin.lab` exists to fit them.

For SiV- and GeV- the default orientation is the `<111>` axis of a
`(001)`-oriented sample; pass `theta_rad` and `phi_rad` to change it.
No emission budget ships for SiV- or GeV-; build an `EmissionBudget`
from your own cited numbers.

The lead-vacancy (PbV-) centre is documented but not shipped, on
purpose: its ground splitting is measured (about 3900 GHz; Wang et
al., ACS Photonics 8, 2947 (2021), confirmed by Chen et al.,
arXiv:2605.27841 (2026)), but no verified measured excited-state
splitting was available, and this package does not ship half a
parameter set. For PbV-, or your own sample of any centre, fill in
`SpinParameters` and `EmissionBudget` from your measurements or the
literature; the provenance travels with every prediction.

## What is in the package

Each function's docstring (`help(vacspin.cyclicity)`, for example)
gives its inputs, units and conventions.

**Parameter sets** (`vacspin.params`)

- `SpinParameters` -- spin-orbit constants `lam_g`, `lam_e`, strain
  `ups_g`, `ups_e`, orbital and spin Zeeman factors `f_g`, `f_e`,
  `delta_g`, `delta_e`, the axis angles `theta_rad`, `phi_rad`, and a
  required `reference`.
- `EmissionBudget` -- lifetime `tau0_s`, quantum efficiency `eta_q`,
  zero-phonon fraction `eta_dw`, branching ratio `eta_br`, zero-phonon
  wavelength `zpl_nm`, and a required `reference`; `gamma0` and
  `radiative_fraction` are derived from them.
- `snv_rosenthal2023`, `snv_emission`, `siv_hepp2014`,
  `gev_bhaskar2017` -- the cited sets above.

**Spin Hamiltonian** (`vacspin.hamiltonian`; Rosenthal et al., PRX 13,
031022 (2023), Appendix B)

- `h_manifold` -- the 4 x 4 Hamiltonian of one manifold (spin-orbit +
  strain + spin and orbital Zeeman), with the field in the centre's
  frame.
- `lab_to_spin` -- rotates a lab-frame vector into the centre's frame.
- `solve` -- energies and eigenvectors of both manifolds at a
  lab-frame field; `strain_scale` multiplies the strain (a
  strain-tuning knob).
- `qubit_frequency` -- the splitting of the lowest ground pair.
- `zero_field_splitting` -- the closed form `sqrt(lam^2 + 4 (Ux^2 +
  Uy^2))`.
- `qubit_frequency_perpendicular` -- the approximate formula
  `2 gamma B_perp ups / Delta` for a purely sideways field (Eq. B9),
  valid for small fields; use `qubit_frequency` for anything
  quantitative.
- `GAMMA_GHZ_PER_T` -- the gyromagnetic ratio, 28.0 GHz/T.

**Transitions** (`vacspin.transitions`)

- `transition_strength`, `transition_table` -- one or all 16
  transition strengths, and their energies.
- `cyclicity` -- the cyclicity of the lowest ground state on the lower
  (`manifold="lower"`) or upper excited pair; infinite when the
  spin-flip line is exactly dark.
- `rabi_rate` -- the microwave Rabi rate (GHz) for a drive field.
- `field_on_circle` -- a field of given size at an angle in one lab
  plane.

**Cavity** (`vacspin.cavity`; conventions of Lee et al.,
arXiv:2511.05740)

- `purcell_max` -- the ideal Purcell factor `(3 / 4 pi^2) Q / V`.
- `lorentzian_suppression` -- how much a transition detuned from the
  cavity is enhanced less: `1 / (1 + (2 Q delta / f_cav)^2)`.
- `xi_pol_overlap` -- the polarisation overlap of a `<111>` dipole
  with an in-plane cavity field.
- `CavityInterface` -- the full budget: `f_max`, `f_c` (Purcell factor
  of the coupled line), `f_c2` (of the detuned spin-flipping partner),
  `zeta` (lifetime shortening), `gamma_cav`, `tau_cav_s`, `beta`,
  `eta`, `lambda_cav` (cavity-boosted cyclicity), `kappa_hz`, `g_hz`,
  `cooperativity`, the `validity` checks, `require_valid()` and
  `summary()`. An optional `delta_partner_hz` adds the Purcell factor
  of a competing same-spin line and a check that the cavity is
  narrower than their spacing.

**Readout** (`vacspin.readout`; Rosenthal et al., arXiv:2403.13110,
Appendix A)

- `polarization_rate` -- how fast the bright record ends through spin
  flips.
- `readout_counts` -- mean bright and dark counts in a window.
- `fidelity_threshold1` -- the closed-form fidelity at a threshold of
  one photon (Eq. A20).
- `fidelity` -- the fidelity at the best whole-number threshold, from
  the full bright-count distribution and the dark-state model
  (background, `leak`, switch-on).
- `required_efficiency`, `required_window` -- the smallest efficiency
  or shortest window that reaches a target fidelity.
- `poisson_pmf`, `geometric_pmf` -- the two limiting count
  distributions (no spin flips; flips always end the record).

**Remote entanglement** (`vacspin.remote`; Barrett and Kok, PRA 71,
060310(R) (2005))

- `barrett_kok_success` -- success probability per attempt,
  `(1/2) eta_A eta_B`.
- `entanglement_rate` -- attempt rate times that probability.

**Your own sample** (`vacspin.lab`)

- `parameter_information` -- before measuring: is the plan
  identifiable, and what error bars would it give?
- `design_fields` -- the most informative subset of candidate points.
- `fit_spin_parameters` -- fit any of the ten spin parameters to your
  measured frequencies; returns a `SpinFit` with `params` (a ready
  `SpinParameters` whose `reference` records the fit and the base set),
  `values`, `sigma`, `cov`, `chi2`, `chi2_dof` and more.
- `save_observations_csv`, `load_observations_csv` -- a plain CSV
  format for observation records (columns `kind, bx_t, by_t, bz_t,
  value_ghz` and optionally `sigma_ghz`).

`__version__` gives the installed version.

## When it refuses, and why

`vacspin` raises an error instead of guessing when:

- a parameter set has no real `reference` (fewer than 8 characters), a
  spin-orbit constant that is not positive, a negative strain, a
  lifetime or wavelength that is not positive and finite, or an
  efficiency outside (0, 1];
- the magnetic field is not finite, `strain_scale` is negative, or the
  sideways field given to `qubit_frequency_perpendicular` is negative;
- a manifold or field-plane name is unknown;
- a cavity's `Q` is negative or `V` is not positive (in
  `CavityInterface`, `Q = 0` currently stops with a
  `ZeroDivisionError`), an efficiency or overlap is
  outside [0, 1], the bare cyclicity is not finite and positive, or the
  qubit frequency is not positive;
- `require_valid()` finds the interface outside the bad-cavity,
  weak-coupling or spin-selective regime (and, when
  `delta_partner_hz` is given, the partner-selectivity condition); it
  names every failed condition;
- a readout input is out of range: `eta` outside [0, 1], a cyclicity
  that is not finite and positive, a decay rate, window or saturation
  parameter that is not positive and finite, a negative `leak` or
  `noise_rate`, `f0` outside (0, 1], or negative mean counts;
- a target fidelity is outside (0.5, 1), or cannot be reached (by any
  efficiency up to 1, or any window up to `tau_max`); the message gives
  the best value found;
- a detection efficiency is outside [0, 1] or the attempt rate is not
  positive (remote entanglement);
- in `vacspin.lab`: an unknown observable kind or parameter name, a
  repeated or empty `vary`, fields of the wrong shape or not finite,
  measurement errors that are not positive, too few observations for
  the parameters (and, without `sigmas_ghz`, for an error scale), a
  design that cannot tell the parameters apart, an `n_pick` outside
  its range, a fit that does not converge (`RuntimeError`), or a CSV
  file with a wrong header, a short row, a non-numeric value or no
  data.

## How the results are checked

62 automated tests run on every push and pull request, on Python 3.9,
3.10, 3.11, 3.12, 3.13 and 3.14, and once more on Python 3.10 with the
oldest versions the package allows (NumPy 1.22.0, pytest 7.0.0). The
numerical checks compare the package with an exact formula, a
published measurement, or a second calculation done a different way.
The rest check that the refusals fire, plus one check that the version
numbers agree and one that every exported name exists. The main
checks, with the tolerances the tests actually use:

**Spin Hamiltonian and transitions**

- The Hamiltonian is Hermitian (equal to its own conjugate transpose,
  as any energy matrix must be) for 20 random inputs (to 1e-12).
- At zero field each manifold is two degenerate pairs (to 1e-9 GHz),
  and their gap equals the closed-form zero-field splitting (to 1 part
  in 10^9), for 10 random parameter sets with strain.
- The frame rotation keeps vector lengths (to 1e-12) and is
  orthonormal (a pure rotation, to 1e-12).
- The approximate transverse-field qubit frequency agrees with full
  diagonalisation within 2 % at 50 mT.
- Without strain, a field along the axis splits the qubit by exactly
  `gamma B` (to 1e-9 GHz), and a sideways field splits it by less than
  1 % of that.
- Scaling the strain moves the ground splitting along the closed form
  (to 1 part in 10^9).
- The 16 transition strengths from each ground state add up to 3 (to
  1e-10) for SnV- and SiV- at three fields, and for GeV- at two.
- For a field exactly along the symmetry axis the cyclicity is above
  10^12 for SnV-, SiV- and GeV- (for SnV- the spin-flip strength is
  also checked to be below 1e-20).
- The Rabi rate is zero without a drive, exactly doubles when the
  drive doubles (to 1 part in 10^12), and lies between 1 and 20 MHz for
  a 0.6 mT drive (measured: about 6.25 MHz).

**Against published measurements (SnV-, no fitted numbers)**

- Ground splitting within 0.2 % of 902.98 GHz; excited within 0.2 % of
  3000 GHz.
- Qubit frequency within 2 % of the measured 3.677 GHz at 125 mT,
  147 degrees.
- Cyclicity within 10 % of 8.6 at 53 degrees, 180 mT, and within 10 %
  of 2244 for a 180 mT field tilted 9 degrees from the axis.
- The confocal readout of arXiv:2403.13110: a bright-minus-dark mean
  count between 3 and 5 (measured about 4), the dark count equal to
  the stated 0.2 (to 1e-9), and a threshold-1 fidelity between 0.82
  and 0.92 with a preparation factor `f0 = 0.935` (measured 0.874).
- The GeV- and SiV- values equal their sources exactly, and so do
  their closed-form zero-field splittings; for GeV- the full solver
  also gives 152 and 981 GHz (to 1e-9 GHz).

**Cavity**

- `purcell_max` gives exactly 1 for `Q/V = 4 pi^2 / 3` (to 1e-12),
  and the coupling
  rate satisfies `F_C = 4 g^2 / (kappa gamma_rad)` with every rate in Hz
  (to 1 part in 10^9); `g_hz` also equals the angular-unit value
  divided by 2 pi and the cooperativity equals `F_C` times the
  radiative fraction (to 1 part in 10^12).
- The Lorentzian (the cavity's response versus detuning) is 1 on
  resonance, 1/2 at `delta = kappa/2` (to
  1e-12), and below 1e-9 far away; the polarisation overlap is 2/3 (to
  1e-6) at the default angles and 0 for a dipole along the normal.
- `beta` and `eta` stay in (0, 1] for Q from 10 to 10^7, `beta` rises
  with Q and passes 0.99, and `zeta` follows its formula (to 1e-12).
- The cavity-boosted cyclicity is at least the bare one, follows its
  formula (to 1 part in 10^9), and grows by less than 5 % for a very
  broad cavity (Q = 10).
- `require_valid()` passes at Q = 500 and refuses at Q = 2 x 10^7.

**Readout statistics**

- The Poisson and geometric distributions add up to 1 (to 1e-12 and
  1e-9) and have the right means (to 1e-9 and 1e-6 of `1 + mean`).
- With almost no spin flips, the bright-count distribution equals the
  Poisson distribution (to 1e-10 per value).
- When flips always end the record, it equals the geometric
  distribution with mean `eta (Lambda + 1)` (to 1e-10); the
  per-emission version with mean `eta Lambda` differs by more than
  1e-4 but less than `10 / Lambda^2`.
- Where the best threshold is 1, `fidelity` equals `1 - P(no bright
  photon)/2` (to 1e-9) and agrees with the closed form of Eq. A20
  within 0.02.
- The fidelity stays in [0.5, 1] and rises with `eta`.
- `polarization_rate` reaches `gamma/2` at strong drive (within
  `1e-3 gamma`), falls below 1e-4 of its value at a large detuning, and
  scales as `1/(1 + Lambda)` (to `1e-12 gamma`).
- `required_efficiency` returns an efficiency that reaches the target
  (and half of it does not); `required_window` returns a window that
  reaches it; both refuse unreachable targets.

**The source study's design point** (Q = 500 cavity, from the study's
stated inputs, with the qubit frequency computed by the package)

- Purcell factor within 5 % of 19 and detection efficiency within 5 %
  of 0.65, inside the valid regime.
- With a computed off-resonant leak and a 1000 counts/s background:
  fidelity above 98 % at the best of four windows (10, 20, 30,
  50 ns); above 94 % at 100 ns with `f0 = 0.99`; and lower when `eta`
  is halved.
- The window needed for 85 % fidelity is more than 100 times shorter
  with the cavity than confocally.

**Your own sample** (`vacspin.lab`)

- The observables equal `qubit_frequency` and the closed-form
  splittings (to 1e-9 GHz).
- A fit to noise-free synthetic data, started 8 % and 15 % off,
  recovers `lam_g` and `ups_g` to 1 part in 10^5.
- In a linear case the error bar equals `sigma / sqrt(n)` (to 1e-6
  GHz), from both the fit and the planner, and the two agree to 0.1 %
  in a nonlinear case.
- Over 400 seeded synthetic experiments, the scatter of the fitted
  values matches the reported error bars within 15 %.
- For zero-field data only, the information matrix has rank one (it
  carries information about only one combination of the two
  parameters: its second singular value is below 1e-10 of the first);
  the planner reports the design as not identifiable, and the fit and
  the design tool refuse it.
- The greedy design's chosen set is at least as informative as each
  of 30 random subsets of the same size. (The rank-one determinant
  identity behind the greedy rule is checked on random matrices.)
- A CSV round trip returns identical values.

## Corrections in earlier versions

**0.3.1 fixed three problems found in a review of 0.3.0.**

- `CavityInterface.g_hz` mixed units: it multiplied the cavity
  linewidth in Hz by the decay rate in 1/s, which made the reported
  coupling rate (`g_hz`, and `g_mhz` in `summary()`) too large by
  `sqrt(2 pi)`, about 2.5 times, and made the weak-coupling check
  refuse some designs it should have accepted. It now uses the decay
  linewidth in Hz, `gamma / (2 pi)`, as the bad-cavity check already
  did. The Purcell factor, efficiencies, lifetime, cyclicity and
  cooperativity are unchanged. If you used `g_hz` or the
  `weak_coupling` verdict from 0.3.0 or earlier, please re-check them.
- `readout_counts` accepted a negative `leak` or `noise_rate` and
  returned a negative dark count; it now refuses them, as `fidelity`
  already did.
- The readout functions accepted an infinite window, decay rate or
  saturation parameter (for example `fidelity` returned NaN for
  `tau = inf`); they now refuse non-finite values with a clear message.

Also in 0.3.1: the CI matrix now includes Python 3.10, and a CI job
tests the oldest allowed dependencies.

The full history is in [CHANGELOG.md](CHANGELOG.md).

## Limits

- **No photonic device design.** The cavity's Q, mode volume and
  out-coupling are inputs you simulate or measure; the companion
  study's pipeline shows how.
- **Weak-coupling, bad-cavity regime only.** The readout model is the
  two-level bad-cavity treatment. `require_valid()` refuses the
  strong-coupling regime rather than describing it wrongly. Its
  thresholds are fixed factors: decay linewidth below a tenth of the
  cavity linewidth, `g` below a tenth of the cavity linewidth, and
  decay linewidth below a fifth of the qubit frequency.
- **Dark-state switch-on is approximate.** The bright-count
  distribution is evaluated in closed form, but the switch-on channel
  is averaged over 24 switch times (a numerical sum, `nt_switch`), and
  its error probability is added to the background error and capped
  at 1 rather than combined exactly.
- **The measured cyclicity near alignment depends on alignment.** At
  the nominal angles the model gives a much lower cyclicity than the
  measured 2244; see example 1.
- **Spin coherence times** (T1, T2) are sample-dependent measurements
  and are not part of the model.
- **SiV- and GeV- ship without strain, quenching or emission budget**;
  supply or fit your own.
- **The `lab` fits** use a local Levenberg-Marquardt search (a standard
  step-by-step least-squares method) from your starting set; the error
  bars assume independent Gaussian measurement errors, and the greedy design is a heuristic.

## Where it comes from

`vacspin` is the general-purpose engine distilled from the
SnV-on-TFLN (tin-vacancy centre on thin-film lithium niobate) interface
study below, and generalised to any group-IV
colour centre and any cavity. The study's repository holds the device
design (FEM/EME/GME electromagnetic simulations) and reproduces the specific study.

> T. M. Mahim, M. M. Rahman and A. S. M. Mohsin, "Fast single-shot
> readout of tin-vacancy spins with an overcoupled diamond nanocavity
> on thin-film lithium niobate" (submitted to Optics Express, 2026);
> pipeline: https://github.com/Tanvir-Mahmud-Mahim/snv-tfln-cavity-interface

Formalism: Rosenthal et al., PRX 13, 031022 (2023) and
arXiv:2403.13110; transition-resolved cavity conventions: Lee et al.,
arXiv:2511.05740; heralding: Barrett and Kok, PRA 71, 060310(R)
(2005); optimal design: F. Pukelsheim, Optimal Design of Experiments,
SIAM (2006).

## Citing, support and license

If `vacspin` helps your work, please cite it with the concept DOI
[10.5281/zenodo.22819698](https://doi.org/10.5281/zenodo.22819698),
which always resolves to the latest version; every release is
archived on Zenodo. [CITATION.cff](CITATION.cff) has the details.

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

Licensed under Apache-2.0.
