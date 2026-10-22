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
- With those error bars, how uncertain is any number predicted from
  the fitted centre (a qubit frequency at a new field, a cyclicity)?
- From your own single-shot photon-count histograms, what are the
  detection efficiency, cyclicity and background of your setup, with
  error bars, and does the model describe the data at all?
- Which readout window gives the highest fidelity, and how well is
  that fidelity known from your calibration?
- From two lifetime measurements, with the cavity tuned away and on
  resonance, what are the Purcell factor, cooperativity and coupling
  rate of your cavity, with error bars? And how often does each spin
  state give the wrong answer at the threshold your counter uses?

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
- [Tutorials](#tutorials) (three end-to-end scripts in `examples/`)
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
  `C = 4 g^2 / (kappa gamma)` measures the same speed-up; here it equals
  the Purcell factor minus one, times the radiative fraction. The
  lifetime shortens by `1 + C`, and the share of photons that go into
  the cavity mode is `C / (1 + C)`.
- **Detection efficiency** (`eta`) -- the fraction of emission events
  that end as a click on the detector, after every loss on the way.
- **Single-shot readout fidelity** -- the probability that one
  measurement names the spin state correctly. You count photons in a
  time **window** `tau`; the spin is called "bright" if the count
  reaches a **threshold**. 0.5 is a coin toss, 1 is perfect.
- **Dark counts / leak / switch-on** -- ways the dark spin state can
  still produce clicks: detector background (`noise_rate`, which adds
  clicks to both spin states), weak off-resonant scattering (`leak`),
  and an off-resonant excitation that flips the dark spin into the
  bright state during the window (switch-on).
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
printed with vacspin 0.8.1 (the same as with 0.6.0 to 0.8.0). The cavity, detector, attempt-rate and
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
in GHz. By default the drive uses the spin-Zeeman term only, as in
Rosenthal et al. (their Eq. B10 sets the orbital factor and the g-factor
anisotropy to zero). `rabi_rate(..., full_zeeman=True)` keeps both, as
the static Hamiltonian does. For the SnV- set at this field and a
0.6 mT drive, that changes the rate by -3.6 % (drive along lab x),
+4.1 % (lab y), +1.7 % (lab z) and +18.8 % (along the symmetry axis).

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
photons into the mode:   0.862
detected per decay, eta: 0.614
cavity linewidth kappa:  968.6 GHz
coupling rate g:         7.32 GHz
regime checks: {'spin_selectivity': True, 'bad_cavity': True, 'weak_coupling': True}
fidelity in 50 ns: 0.99964 (threshold 1 photon, mean bright count 16.41)
```

Only the part of the emission that the cavity adds goes into the
mode. The coupled line decays `F_C` times faster than without the
cavity; its normal emission (the "1" in `F_C`) still goes where it
went before, so `beta` uses `F_C - 1`. With no overlap
(`xi_pos=0`) `beta`, `g` and the cooperativity are exactly 0. Before
0.6.0 `beta` and `g` used `F_C`, which gave `beta = 0.342` with no
overlap and `eta = 0.648` here. The original submission of the
companion study reported that number (about 65 %);
`beta_form="transition"` gives it back. The revised study uses another
design point (`Q_L = 870`, about 32 %), which
`tests/test_revised_study.py` checks.

`xi_pol` and `xi_pos` say how well the centre's dipole lines up with
the cavity field in direction and position (each from 0 to 1);
`xi_pol_overlap()` gives 2/3 for a centre in a `<100>`-oriented
diamond film. `eta` multiplies every loss from the emitter to the
detector: the share of emission into the cavity mode (`beta`), the
out-coupling port, the link, the chip and the detector.

This fidelity counts only the bright spin's own statistics: `leak` and
`noise_rate` are left at zero here, so the dark state never produces a
click. The test suite runs nearly the same design point (the original
submission's: link 0.991, out-coupling `1 - 500/1.39e6`) with a computed
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

# The same readout, but the dark spin also scatters light off-resonantly
# (leak = 1 % of the resonant rate), which also switches it on.
out = vs.fidelity(0.002, 2244.0, gamma, 50e-6, s=10.0, leak=0.01,
                  noise_rate=0.2 / 50e-6)
print(f"with leak: fidelity {out['fidelity']:.3f}, mean counts "
      f"bright {out['n_bright_total']:.2f}, dark {out['n_dark_total']:.2f}")
```

```
mean counts: bright 4.22, dark 0.20
threshold-1 fidelity: 0.9020
window for 85 %: 8.8 us
efficiency for 99 % in 10 us: 0.0218
refused: target fidelity 0.999 is unreachable at this operating point: even eta = 1 gives 0.9000. Increase the cyclicity, the window, or reduce the dark counts.
with leak: fidelity 0.500, mean counts bright 4.22, dark 4.30
```

These inputs are the confocal case the test suite uses for the
measured readout of Rosenthal et al., arXiv:2403.13110, which reports
about 4 bright counts and at most 0.2 dark counts in a 50 us window at
cyclicity 2244; the test asserts a bright-minus-dark mean between 3
and 5. The efficiency 0.2 % is chosen so that this package's
mean-count formula gives those 4 counts; the paper's own fitted
efficiency is about 0.1 %, so treat 0.2 % as a matched input, not a
quoted one. `s` is the **saturation parameter** of the optical drive
(how hard it is driven); `fidelity_threshold1` is the closed form for
a threshold of one photon. `required_window` and `required_efficiency`
search the full model (`fidelity`) and refuse when no value reaches the
target. Compared with example 2, the cavity turns a window of
microseconds into one of tens of nanoseconds; the test suite checks
that the required window shrinks by more than 100 times.

In the last case the leak switches the dark spin on within about a
microsecond, so for most of the window both spin states look bright
and one measurement cannot tell them apart (fidelity 0.5). `fidelity`
evaluates this switch-on channel exactly (since 0.4.0; 0.3.1 gave 0.72
here, see [Corrections](#corrections-in-earlier-versions)).

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
print(f"p-value {fit.p_value:.2f}; largest residual "
      f"{max(abs(fit.residuals_ghz)):.3f} GHz")
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
p-value 0.53; largest residual 0.063 GHz
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
the data within the stated measurement errors. `fit.p_value` (since
0.5.0) makes this a number: the probability of a `chi2` at least this
large if the model and the stated errors are right. A value below
about 0.001 means they are not: the errors are too small, or a
parameter you held fixed is wrong. `fit.residuals_ghz` lists measured
minus fitted values, point by point, so you can see which point is
off. `design_fields` adds points one at a time,
each time taking the one that most increases the determinant of the
information matrix (greedy D-optimal design); this is a good
heuristic, not a proof of the best possible subset. Observation
records can be saved and reloaded with `save_observations_csv` and
`load_observations_csv`; the round trip returns identical values.

### 6. Error bars on anything you predict from the fit

Continuing example 5 (run it first; `fit` and `truth` come from
there):

```python
# The qubit frequency at a field you did not measure, with its error bar
b_new = vs.field_on_circle(35.0, 0.30)
q = vs.propagate_uncertainty(fit, lambda p: vs.qubit_frequency(p, b_new))
print(f"qubit frequency at 300 mT, 35 deg: {q['value']:.3f} +/- {q['sigma']:.3f} GHz"
      f" (true {vs.qubit_frequency(truth, b_new):.3f})")

# Any function of the parameters works, including several outputs at once
z = vs.propagate_uncertainty(
    fit, lambda p: [vs.zero_field_splitting(p.lam_g, p.ups_g),
                    vs.cyclicity(p, b_new)])
print("ground splitting: {:.2f} +/- {:.2f} GHz".format(z["value"][0], z["sigma"][0]))
print("cyclicity:        {:.1f} +/- {:.1f}".format(z["value"][1], z["sigma"][1]))
```

```
qubit frequency at 300 mT, 35 deg: 4.126 +/- 0.141 GHz (true 4.032)
ground splitting: 917.57 +/- 0.03 GHz
cyclicity:        5.8 +/- 0.4
```

`propagate_uncertainty` takes the slopes of your function with
respect to the fitted parameters and combines them with the fit's
covariance matrix (first-order error propagation). It is exact for a
quantity that depends linearly on the fitted parameters, and a good
approximation when the error bars are small compared with how quickly
the quantity curves; the tests compare it with the spread obtained by
drawing many parameter sets from the fit's uncertainty (within 5 %).
The ground splitting is known far better than `lam_g` and `ups_g`
separately, because the fit pins down the combination
`sqrt(lam^2 + 4 ups^2)` directly; `cov` in the result carries such
correlations.

### 7. Calibrate the readout from your own count histograms

In a readout experiment you prepare the spin many times, count the
photons in each window, and histogram the counts.
`fit_readout_histograms` (new in 0.5.0) finds the detection efficiency
`eta`, the cyclicity `lam` and the background `noise_rate` that make
your histograms most probable under the same count distributions that
`fidelity` uses. It returns error bars and a goodness-of-fit test.

```python
import numpy as np
import vacspin as vs

gamma = vs.snv_emission().gamma0       # optical decay rate, 1/s
tau, s = 20e-6, 10.0                   # counting window (s), drive

# Stand-in for your data: photon counts of 5000 shots prepared bright
# and 5000 prepared dark. Here they are simulated (eta = 0.2 %,
# cyclicity 2244, background 4000 counts/s); use your own instead.
rng = np.random.default_rng(7)
r = 0.5 * gamma * s / (1 + s)                       # scattering rate
t_flip = rng.exponential((1 + 2244.0) / r, 5000)    # when the spin flips
bright = (rng.poisson(0.002 * r * np.minimum(t_flip, tau))
          + rng.poisson(4000 * tau, 5000))
dark = rng.poisson(4000 * tau, 5000)

fit = vs.fit_readout_histograms(
    np.bincount(bright), gamma, tau, s, dark_hist=np.bincount(dark),
    vary=("eta", "lam", "noise_rate"))
v, e = fit.values, fit.sigma
print(f"eta:        {100 * v['eta']:.3f} +/- {100 * e['eta']:.3f} %")
print(f"lam:        {v['lam']:.0f} +/- {e['lam']:.0f}")
print(f"noise_rate: {v['noise_rate']:.0f} +/- {e['noise_rate']:.0f} counts/s")
print(f"chi2 = {fit.chi2:.1f} for {fit.chi2_dof} degrees of freedom, "
      f"p-value {fit.p_value:.2f}")

# The fidelity these numbers predict for a 50 us window
out = vs.fidelity(v["eta"], v["lam"], gamma, 50e-6, s=s,
                  noise_rate=v["noise_rate"])
print(f"predicted fidelity at 50 us: {out['fidelity']:.3f}")
```

```
eta:        0.200 +/- 0.003 %
lam:        2272 +/- 104
noise_rate: 3988 +/- 199 counts/s
chi2 = 8.8 for 10 degrees of freedom, p-value 0.55
predicted fidelity at 50 us: 0.842
```

The fit recovers the simulated values within its error bars. Each
histogram is an array whose entry `k` is the number of shots with
exactly `k` counts; `numpy.bincount` turns per-shot counts into one.
The fit starts from the means and variances of the histograms (you
can pass your own starting values as `eta`, `lam` and `noise_rate`;
for a parameter you hold fixed, they give its value).
The optical decay rate `gamma`, the saturation parameter `s` and the
window `tau` are inputs you know, not fitted. The efficiency and the
drive enter the counts only through the detected rate `eta R` (with
`R = (gamma/2) s / (1 + s)`), so an error in `gamma` or `s` shows up
in the fitted `eta`, not in the fit quality.

What the fit can and cannot tell:

- The cyclicity shows in the counts only through spin flips during
  the window. If flips are rare (window much shorter than
  `(lam + 1) / R`), the bright counts hardly depend on `lam`. The fit
  then refuses to vary it: it says that the parameters cannot be told
  apart, or that `lam` is poorly determined (an error bar above a
  quarter of the value, where large-sample error bars are not
  reliable). Hold `lam` fixed, or use a longer window.
- The dark histogram is what pins down the background. Without it,
  the background can only be fitted from the bright histogram, which
  determines it far less well; you then have to give a starting
  `noise_rate`, or hold it fixed at a value you measured separately.
- The error bars are large-sample (Cramer-Rao) error bars. The tests
  check them against the exact answers of the Poisson and geometric
  limits and against 300 simulated experiments.
- A small `p_value` (below about 0.001) means the model does not
  describe your histograms. A common cause is imperfect spin
  preparation, which gives extra shots with background counts only.
  The test suite shows that 20 % failed preparations among 5000 shots
  give a p-value below 10^-6.

## Tutorials

Three scripts in the `examples/` folder of the repository (and, since
0.8.0, of the source distribution) go from data to a final result with
error bars. Each makes its own "measured" data with a fixed random
seed, so you can run it as is and then swap in your own numbers. Each
runs in about a second. The test suite runs all three and checks the
numbers below.

### The best readout window, from your count histograms

`examples/tutorial_readout_window.py` fits the detection efficiency,
cyclicity and background to 5000 bright and 5000 dark shots
(`fit_readout_histograms`). It then finds the window with the highest
fidelity (`optimal_window`, new in 0.7.0) and gives error bars on that
window and on its fidelity (`propagate_uncertainty` with the readout
fit, new in 0.7.0). It printed:

```
calibration from 5000 bright and 5000 dark shots:
  eta        = 0.1968 +/- 0.0033 %   (simulated 0.2000 %)
  lam        = 2398 +/- 112   (simulated 2244)
  noise_rate = 3752 +/- 193 counts/s   (simulated 4000)
  goodness of fit: chi2 = 8.3 for 10 degrees of freedom, p = 0.60
best window: 17.34 +/- 0.35 us (closed form for threshold 1: 17.34 us)
fidelity there: 0.8807 +/- 0.0025 (threshold 1 photon)
with the simulated values: 16.76 us, fidelity 0.8763
```

The fitted values lie within two error bars of the simulated ones, and
so do the window and the fidelity. With a threshold of one photon the
best window has a closed form, `ln(1 + a / N) / a` with
`a = R / (lam + 1) + eta R` and `N` the background rate; the search
lands on it.

### From your spin frequencies to a cavity readout

`examples/tutorial_spin_to_readout.py` fits the spin-orbit and strain
constants of a center whose strain is 10 % above the cited value, from
its qubit frequency and ground splitting at eight field angles. It then
predicts the cyclicity and qubit frequency at a readout field 9 degrees
from the center's axis, builds the Q_L = 500 cavity budget of example 2
on them, finds the best window and carries the fit's error bars through
the whole chain to the fidelity. It printed:

```
fitted ground state of this center:
  lam_g = 831.5 +/- 5.6 GHz   (simulated 830.1)
  ups_g = 194.0 +/- 6.0 GHz   (simulated 195.4)
  chi2 = 3.87 for 6 degrees of freedom, p = 0.69
at the readout field: cyclicity 1827 +/- 143, qubit frequency 5.910 +/- 0.005 GHz
cavity: F_C = 19.29, eta = 0.614, cavity-boosted cyclicity 1827
best window 24.8 ns (threshold 2 photons): fidelity 0.99075 +/- 0.00008
with the simulated center: fidelity 0.99073 at the same window
```

Here the calibration barely matters for the fidelity (an error bar of
1e-4). In the 25 ns window the bright spin emits only about 13 photons
(8 of them detected), so a cyclicity of 1800 +/- 140 hardly changes the
chance that it flips during the window. The cavity does not raise the cyclicity, because
its linewidth (about 1 THz) is much larger than the qubit frequency,
so it speeds up the spin-flipping line as much as the bright one. The
dark spin's leak is the ratio of the off-resonant to the resonant
scattering rate, from Eq. A8 of Rosenthal et al., PRX 14, 041008
(2024), at a detuning of one qubit frequency.

### From two lifetime histograms to the cavity and the readout

`examples/tutorial_lifetime_to_readout.py` (new in 0.8.0) fits the
excited-state lifetime of a center twice (`fit_lifetime_histogram`):
with the cavity tuned away and on resonance. Each histogram holds about
30000 decays and 3000 background counts in 16 ps bins. From the two
lifetimes, `purcell_from_lifetimes` gives the lifetime shortening, the
cooperativity, the share of photons into the cavity mode (`beta`), the
Purcell factor and the coupling rate, and compares them with the
Q_L = 500 design of example 2. Then it finds the best readout window,
splits the fidelity into the error of each spin state, shows what
happens at a threshold of 1 photon instead, and carries the lifetime
error bars through to the fidelity. It printed:

```
lifetime, cavity detuned:      4.5466 +/- 0.0357 ns (simulated 4.5000), p = 0.94
lifetime, cavity on resonance: 0.6107 +/- 0.0058 ns (simulated 0.6202), p = 0.41
lifetime shortened 7.44 +/- 0.09 times: cooperativity 6.44 +/- 0.09, beta 0.8657 +/- 0.0017
Purcell factor F_C = 19.84 +/- 0.27 (design 19.29), g = 7.39 +/- 0.08 GHz (design 7.32)
readout: eta = 0.6168, best window 20.9 ns, threshold 2 photons
  fidelity 0.98014 +/- 0.00027: bright error 0.00838, dark error 0.03134
  at threshold 1 instead: fidelity 0.96795, bright error 0.00158, dark error 0.06251
with the simulated lifetimes: fidelity 0.98051
```

The two lifetimes lie 1.3 and 1.6 error bars from the simulated ones.
The first is high and the second low, so their ratio moves by both:
the shortening, the cooperativity, `beta` and the Purcell factor land
about 2.1 error bars from the design (the coupling rate 0.8, as it also
carries the 2 % linewidth error). The predicted fidelity is 1.4 error
bars from the fidelity of the simulated lifetimes. The lifetimes alone
give the shortening, the cooperativity and `beta`; the Purcell factor
also needs the emission budget, and the coupling rate the cavity
linewidth. Lowering the threshold to 1 photon makes the
bright spin fail less often, but the dark spin, which scatters a little
light off-resonantly, now fails twice as often.

## Cited parameter sets

No physical number is made up, and none is accepted without a source:
the `reference` field of both parameter classes (`SpinParameters`,
`EmissionBudget`) is mandatory, and the package refuses an empty or
too-short one.

- `snv_rosenthal2023()` + `snv_emission()`: the SnV- device values of
  Rosenthal et al., PRX 13, 031022 (2023) (Table I) with the
  Thiering-Gali quenching factors (Thiering & Gali, PRX 8, 021063
  (2018)), and the emission budget of arXiv:2403.13110 (lifetime
  4.5 ns, quantum efficiency 0.8) / Goerlitz et al., New J. Phys. 22,
  013048 (2020) (Debye-Waller factor 0.57) / Lee et al.,
  arXiv:2511.05740 (branching ratio 0.75, zero-phonon line 619 nm). Checked against measurement in the tests
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
  spin-flip line is exactly dark. Since 0.8.0 it refuses a field too
  weak to split the qubit pair or the excited pair (see
  [When it refuses](#when-it-refuses-and-why)).
- `rabi_rate` -- the microwave Rabi rate (GHz) for a drive field;
  `full_zeeman=True` adds the orbital and anisotropy Zeeman terms to
  the drive. Since 0.8.0 it refuses a static field too weak to split
  the qubit pair.
- `field_on_circle` -- a field of given size at an angle in one lab
  plane.

**Cavity** (`vacspin.cavity`; decay-rate budget of Lee et al.,
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
  narrower than their spacing. `beta_form="transition"` gives the
  `beta`, `g` and cooperativity of version 0.5.0 and earlier.
- `purcell_from_lifetimes` (new in 0.8.0): the same budget measured
  instead of designed: from the lifetime with the cavity tuned away and
  the lifetime on resonance (numbers with error bars, or two
  `LifetimeFit` results) it gives the lifetime shortening `zeta`, the
  cooperativity `C = zeta - 1` and `beta = C / (1 + C)`; with an
  `EmissionBudget` also the Purcell factor `F_C = 1 + C / a` (`a` the
  radiative fraction), and with the cavity linewidth `kappa_hz` also
  the coupling rate `g_hz`. Each comes with its error bar and the full
  covariance.

**Readout** (`vacspin.readout`; Rosenthal et al., arXiv:2403.13110,
Appendix A)

- `polarization_rate` -- how fast the bright record ends through spin
  flips.
- `readout_counts` -- mean bright and dark counts in a window.
- `fidelity_threshold1` -- the closed-form fidelity at a threshold of
  one photon (Eq. A20).
- `fidelity` -- the fidelity at the best whole-number threshold, from
  the full count distributions of both spin states: the bright one
  (spin flips end its record) plus background, and the dark one
  (background, `leak`, and the switch-on channel, evaluated exactly).
  Besides `fidelity` and `threshold` it returns `n_bright` (mean signal
  count), `n_dark` (mean background plus leak count),
  `n_bright_total` and `n_dark_total` (the means of the two full
  distributions) and `truncation_bound` (a guaranteed bound, at most
  1e-9, on the error from tracking a finite number of counts). Since
  0.8.0 it also returns `error_bright` (the chance that a bright shot
  gives fewer counts than the threshold) and `error_dark` (the chance
  that a dark shot reaches it), and `threshold=N` gives the fidelity at
  the threshold your experiment uses instead of the best one.
- `required_efficiency`, `required_window` -- the smallest efficiency
  or shortest window that reaches a target fidelity.
- `optimal_window` (new in 0.7.0) -- the window with the highest
  fidelity, when the fidelity rises and then falls again (background or
  leak). Returns the `fidelity` result at that window plus `tau`. It
  scans 80 windows from `tau_min` to `tau_max` (1 ns to 10 ms by
  default) and refines the best one; each window costs one `fidelity`
  call, so a long `tau_max` with a bright cavity readout (thousands of
  counts) takes a few seconds.
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
  `values`, `sigma`, `cov`, `chi2`, `chi2_dof` and more. Since 0.5.0 it
  also has `p_value` (the goodness of fit) and `residuals_ghz`
  (measured minus fitted values).
- `propagate_uncertainty` -- after fitting: the value and error bar
  (and covariance) of any quantity computed from the fitted
  parameters, by first-order error propagation. Since 0.7.0 it also
  takes a `ReadoutFit`; your function then receives a dict with the
  keys `eta`, `lam` and `noise_rate`, for example to get the error bar
  of the fidelity at a window you have not measured. Since 0.8.0 it
  also takes a `LifetimeFit` (keys `tau_s`, `amplitude`,
  `background_rate`).
- `save_observations_csv`, `load_observations_csv` -- a plain CSV
  format for observation records (columns `kind, bx_t, by_t, bz_t,
  value_ghz` and optionally `sigma_ghz`). Since 0.5.0 the loader also
  reads files that a spreadsheet saved as "CSV UTF-8" (with a
  byte-order mark) and skips empty lines.
- `fit_readout_histograms` (new in 0.5.0): fit `eta`, `lam` and
  `noise_rate` (any subset) to your bright and dark photon-count
  histograms by maximum likelihood; returns a `ReadoutFit` with
  `values`, `sigma`, `cov`, `log_likelihood`, and a Pearson goodness
  of fit (`chi2`, `chi2_dof`, `p_value`, after pooling count bins until
  each expects at least 5 shots).
- `fit_lifetime_histogram` (new in 0.8.0): fit the excited-state
  lifetime to a time-resolved photon histogram (time-correlated single
  photon counting), with a constant background, by Poisson maximum
  likelihood. Returns a `LifetimeFit` with `tau_s`, `sigma_tau_s`,
  `values` (`tau_s`, `amplitude`, `background_rate`), `sigma`, `cov`
  and a Pearson goodness of fit (`chi2`, `chi2_dof`, `p_value`, after
  pooling bins until each expects at least 5 counts). `t_start` fits
  only the bins after a given delay, to leave out the excitation pulse
  and the detector's timing response; `background=False` holds the
  background at zero.

`__version__` gives the installed version.

## When it refuses, and why

`vacspin` raises an error instead of guessing when:

- a parameter set has no real `reference` (fewer than 8 characters), a
  number that is not finite (NaN or infinity), a spin-orbit constant
  that is not positive, a negative strain, a lifetime or wavelength
  that is not positive and finite, or an efficiency outside (0, 1];
- the magnetic field is not finite, `strain_scale` is negative or (since
  0.7.0) not finite, a Rabi drive field is not finite (since 0.7.0), or
  the sideways field given to `qubit_frequency_perpendicular` is
  negative or not finite;
- a manifold or field-plane name is unknown;
- a cavity's `Q` or `V` is not positive and finite, an efficiency or
  overlap is outside [0, 1], the bare cyclicity is not finite and
  positive, or the qubit frequency or `delta_partner_hz` is not
  positive and finite, or `beta_form` is not `"purcell"` or
  `"transition"`; since 0.7.0 also a cavity whose ideal Purcell factor
  `F_max` is below 1 (`Q / V` below `4 pi^2 / 3`, about 13.2) when the
  emitter overlaps the mode, with the default `beta_form`;
- `require_valid()` finds the interface outside the bad-cavity,
  weak-coupling or spin-selective regime (and, when
  `delta_partner_hz` is given, the partner-selectivity condition); it
  names every failed condition;
- a readout input is out of range: `eta` outside [0, 1], a cyclicity
  that is not finite and positive, a decay rate, window or saturation
  parameter that is not positive and finite, a `leak` or `noise_rate`
  that is negative or not finite, `f0` outside (0, 1], or mean counts
  that are negative or not finite;
- `fidelity` would need count distributions wider than 2^18 = 262144
  counts to stay within its 1e-9 accuracy bound (a very long window
  with a very high background);
- a target fidelity is outside (0.5, 1), or cannot be reached (by any
  efficiency up to 1, or any window up to `tau_max`, which must be
  finite and longer than 1 ns); the message gives the best value found;
- a detection efficiency is outside [0, 1] or the attempt rate is not
  positive and finite (remote entanglement);
- in `vacspin.lab`: an unknown observable kind or parameter name, a
  repeated or empty `vary`, fields of the wrong shape or not finite,
  measurement errors that are not positive, too few observations for
  the parameters (and, without `sigmas_ghz`, for an error scale), a
  design that cannot tell the parameters apart, an `n_pick` outside
  its range, a fit that does not converge (`RuntimeError`), a CSV
  file with a wrong header, a short row, a non-numeric value or no
  data, or (in `propagate_uncertainty`) a function that returns a
  non-finite value or slope;
- in `fit_readout_histograms`: a histogram that is not a 1-D array of
  whole numbers of shots, or holds fewer than 10 shots; an unknown or
  repeated name in `vary`; a fixed `eta` or `lam` without a value; a
  bright mean count not above the background; a background to fit with
  no dark counts, or with neither a dark histogram nor a starting
  value; starting values that give zero probability to an observed
  count; parameters the histograms cannot tell apart; a fitted value
  whose error bar is more than a quarter of the value; a best fit at
  `eta = 1`; or a fit that does not converge (`RuntimeError`);
- a target fidelity that is not a real number (since 0.5.0, for
  example the string `"0.9"`), or a polarization angle in
  `xi_pol_overlap` that is not finite;
- in `optimal_window` (since 0.7.0): no peak inside the range (the
  fidelity at `tau_max` is within 1e-9 of the best found, as without
  background and leak, where it never falls; or the best is at
  `tau_min`), `eta = 0`, or `tau_min` and `tau_max` that are not
  finite with `0 < tau_min < tau_max`;
- in `propagate_uncertainty` with a `ReadoutFit`: a function that
  returns a non-finite value or slope, or a covariance that does not
  match the fitted parameters;
- in `cyclicity` and `rabi_rate` (since 0.8.0): a field too weak to
  split the qubit pair (and, for `cyclicity`, the excited pair it
  uses), that is, split by less than 1e-8 of the largest energy of its
  manifold. Then the two states of the pair are set by round-off, not
  by the field. At zero field this always happens, and for a center
  without strain or orbital terms (the SiV- and GeV- sets) also in a
  field perpendicular to its axis. For the SnV- set it happens below a
  few microtesla (at most 6 uT for fields in the lab x-z plane);
- in `fidelity` (since 0.8.0): a `threshold` that is not a whole
  number of counts from 1 to 2^18;
- in `fit_lifetime_histogram` (since 0.8.0): counts that are not a 1-D
  array of whole numbers; bin edges that are not finite and strictly
  increasing, or not one more than the counts; fewer than 10 counts, or
  too few bins for the fit to have a spare degree of freedom; a
  `t_start` or `tau_start` that is not a finite number (and a positive
  one, for `tau_start`); no counts above the background; a best fit
  whose background ends at zero (refit with `background=False`); a
  lifetime that runs below 1e-3 of the narrowest bin or above 1000
  times the fitted window, or an amplitude that runs to zero (no decay
  this fit can measure); a lifetime or amplitude whose error bar is
  more than a quarter of the value; parameters the histogram cannot
  tell apart; or a fit that does not converge (`RuntimeError`);
- in `purcell_from_lifetimes` (since 0.8.0): a lifetime that is not
  positive and finite, an error bar that is negative or not finite, an
  error bar given together with a `LifetimeFit` (which carries its
  own), a cavity lifetime that is not shorter than the reference
  lifetime, a `budget` that is not an `EmissionBudget`, a `kappa_hz`
  that is not positive and finite, or a cooperativity whose error bar
  is more than a quarter of its value (first-order error bars are then
  not reliable).

## How the results are checked

147 automated tests run on every push and pull request, on Python 3.9,
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
- With `full_zeeman=True` and no orbital or anisotropy terms, the Rabi
  rate equals the default one (to 1 part in 10^13). With them, it
  equals the matrix element of `H(b0 + b_drive) - H(b0)` built from
  the full ground Hamiltonian at two fields (to 1 part in 10^9).
- The Hamiltonian built from precomputed matrices (since 0.4.0, for
  speed) equals the term-by-term construction for 50 random inputs
  (to 1e-12 GHz).
- Weak fields (since 0.8.0): at 30 uT in 12 directions the cyclicity
  and the Rabi rate equal degenerate perturbation theory (the Zeeman
  operator diagonalized inside the zero-field pairs, a separate
  calculation) to 1 part in 10^5; at 53 degrees that limit is 7.85 and
  5.93 MHz. At 1e-15 T, 1e-12 T and zero field both functions refuse
  (0.7.0 returned 1.1e34, 9.06 and infinity for the cyclicity), as
  they do for SiV- in a field perpendicular to its axis, and the
  refusal sits where the qubit splitting crosses 1e-8 of the largest
  ground energy (1 % above it passes, 1 % below refuses).

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
  rate satisfies `F_C - 1 = 4 g^2 / (kappa gamma_rad)` with every rate
  in Hz (to 1 part in 10^9); `g_hz` also equals the angular-unit value
  divided by 2 pi and the cooperativity equals `F_C - 1` times the
  radiative fraction (to 1 part in 10^12). The same holds with `F_C`
  for `beta_form="transition"`.
- The budget obeys `zeta = 1 + C` and `beta = C / (1 + C)` (to 1e-12)
  at Q = 10, 500 and 10^5; with no overlap `beta`, `g`, `C` and `eta`
  are exactly 0, and the old form gives `beta` equal to the radiative
  fraction. The two forms differ only in `beta`, `g`, `C` and `eta`.
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
- Since 0.7.0: a cavity with `F_max = 0.760` (Q = 1000, V = 100) is
  refused, while with no overlap it still gives `beta = 0`, the
  `"transition"` form keeps its formula (to 1e-15), and
  `F_max = 1` up to round-off gives `beta` between 0 and 1e-12.

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
  reaches it while a window 0.1 % shorter does not; both refuse
  unreachable targets.
- `required_window` below its 1 ns search grid: for an emitter without
  spin flips or background the shortest window has the closed form
  `ln(1 / (2 (1 - F))) / (eta R)`; the result agrees to 1 part in 10^5.
- Background in the bright state: without spin flips and leak both
  states are Poisson, and `fidelity` (value and threshold) equals the
  best-threshold Poisson closed form (to 1e-9), in two cases including
  150 signal counts on 200 background counts.
- The mean bright count for very rare spin flips (cyclicity 10^9 to
  10^15) matches its series limit `eta R tau (1 - Gp tau / 2)` (to 1
  part in 10^12).
- The dark switch-on distribution matches a direct double numerical
  integration over the switch and flip times (to 1e-6 per value, two
  cases covering both evaluation branches), sums to 1 (to 1e-12), and
  its mean equals the numerically integrated survival function times
  the count rate (to 1 part in 10^9), in four cases.
- Monte Carlo: 400 000 simulated shots per spin state of the counting
  process; the fidelity at the returned threshold and the mean bright
  and dark counts agree with `fidelity` within 5 standard errors, in
  three cases
  (slow switch-on, switch-on within the first microsecond, and the
  cavity design point).
- `readout_counts` and `fidelity` differ, as documented, only by the
  leak counts `eta leak R tau` in the bright mean (to 1e-12 relative;
  zero without leak).
- `required_window` when the fidelity peaks between two points of its
  search grid: with few counts the best threshold is 1 and the
  fidelity has the closed form `1/2 + e^(-N tau) (1 - P0(tau)) / 2`
  (`N` the background rate, `P0` the probability of no signal
  photon). For a target between the best grid value and the true
  peak, the returned window equals the closed-form crossing (to 1
  part in 10^5); 0.4.0 refused the target.
- `optimal_window` (since 0.7.0): with a threshold of one photon and
  background rate `N` the fidelity peaks at the closed form
  `tau* = ln(1 + a / N) / a`, `a = R / (Lambda + 1) + eta R` (from
  setting the slope of `1/2 + e^(-N tau) (1 - P0(tau)) / 2` to zero).
  In three cases the returned window equals it (to 1 part in 10^5)
  and its fidelity equals the closed form (to 1e-10). With a leaking
  dark spin, and at the cavity design point, no window of a 400-point
  scan or of a fine scan within 5 % of the result does better (beyond
  1e-12). Without background or leak it refuses, as it should.
- `required_efficiency` below its search start of `eta = 1e-6`:
  without background the fidelity is `1 - P0(eta)/2` in closed form,
  and the returned efficiency equals the closed-form one (to 1 part in
  10^5); 0.4.0 returned 1e-6.
- A given threshold and the two errors (since 0.8.0): without spin
  flips, `error_bright` and `error_dark` equal the Poisson distribution
  function and its complement (summed independently from `lgamma`, to
  1e-9) at six thresholds in two cases, and the fidelity equals
  `f0 (1 - (error_bright + error_dark) / 2) + (1 - f0) / 2` (to
  1e-15); when every bright record ends in a flip, `error_bright`
  equals the geometric `1 - p^N` (to 1e-10); at threshold 1 with flips
  and background it equals `e^(-N tau) P0` (to 1 part in 10^12). In
  three cases a scan of 39 fixed thresholds peaks at the best
  threshold with the best fidelity (to 1e-12).
- The bright-count loop on Python floats (since 0.8.0, for speed)
  equals the 0.7.0 loop, kept in the test, for 200 random cases (to
  1e-12 of the largest value; it performs the same operations in the
  same order).

**The source study's original design point** (Q = 500 cavity, from
the stated inputs of the study's original submission, with the qubit
frequency computed by the package)

- Purcell factor within 5 % of 19, inside the valid regime. With
  `beta_form="transition"` (that version's form) the detection
  efficiency is within 5 % of that version's 0.65 (0.6486); the default
  gives 0.6150, smaller by exactly `(F_C - 1) / F_C`.
- With the off-resonant leak that version computed (the spin-flipping
  line A2 at the full strength of A1; it left out the dark spin's own
  line B2, which the revised study finds dominant) and a 1000 counts/s
  background:
  fidelity above 98 % at the best of four windows (10, 20, 30,
  50 ns); above 94 % at 100 ns with `f0 = 0.99`; and lower when `eta`
  is halved.
- The window needed for 85 % fidelity is more than 100 times shorter
  with the cavity than confocally.

**The revised source study** (since 0.8.1; `Q_L = 870`, against the
numbers printed in the revised manuscript, with each tolerance stated
in the test)

- At 112 mT, 9 degrees from the spin axis, for four directions around
  that axis: qubit frequency 3.684 GHz (paper: 3.68), excited-state
  splitting 4.412 GHz (paper: 4.40), A1-B2 detuning 0.7285 GHz (paper:
  0.73) and cyclicity 2180 (paper: 2180). At 77 mT the A1-B2 detuning
  is 0.501 GHz (paper: 500 MHz). The two spin-conserving lines A1 and
  B2 have the same strength (to 0.1 %).
- The cavity at the paper's `F_C = 28`. The paper writes
  `F_C = 1 + F_cav xi_pol xi_pos` with `F_cav = 81`; vacspin writes
  `F_C = 1 + (F_max - 1) xi_pol xi_pos`, so the test sets
  `F_max = 82` (with `F_max = 81`, vacspin gives `F_C = 27.67`). Then:
  D-line Purcell factor 3.36 (paper: 3.4) with the D line 0.90 THz
  away, coupling 6.74 GHz (paper: 6.8), `g / kappa = 0.0121` (paper:
  0.012), and the bare cyclicity kept by the broad cavity (to 0.1 %).
- Detected-photon efficiency 0.319, within 0.005 of the paper's 32 %,
  about 100 times the confocal 0.1 to 0.4 %. The paper's decay rate
  (its Eq. 1), evaluated in the test, also adds the cavity-enhanced
  emission of the D line: it gives the paper's 10.5 times faster decay
  (0.43 ns) and `beta = 91 %`. vacspin leaves that term out, so its
  `zeta` (10.23) is smaller by exactly that term, and its `beta`
  (0.902) is lower as a result (see [Limits](#limits)).
- The revised study's readout fidelity (98.8 % in 0.1 us) comes from
  its four-level model and is not reproduced here (see
  [Limits](#limits)).

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
- The batched evaluation of all observations at once (since 0.7.0,
  for speed) equals the 0.6.0 point-by-point path for 150 random
  designs of all three centers (to 1e-12 GHz), and a fit, the planner
  and the error propagation give the same numbers either way (to 1
  part in 10^12).
- The faster eigenvalue-only path used by the fits equals the full
  solver for all three observables, 20 random fields and strain
  scales (to 1e-9 GHz).
- `propagate_uncertainty` is exact where it should be: a fitted
  parameter returns its own error bar (to 1e-9 relative), the
  zero-field splitting at zero strain returns `sigma / sqrt(n)` (to
  1e-9 GHz), and a fixed linear combination `A x` returns
  `A C A^T` (to 1e-8 relative). For a nonlinear prediction (the qubit
  frequency at a new field) it agrees within 5 % with the spread of
  4000 parameter sets drawn from the fit's uncertainty. At a
  parameter boundary (zero strain) the slope is taken one-sided and
  is still right for a linear quantity (to 1e-6 relative), and an
  error raised by your own function reaches you instead of being
  hidden.

**Readout calibration from your histograms** (`fit_readout_histograms`,
since 0.5.0)

- Poisson limit (no spin flips, no background): the fitted `eta` is
  the mean count divided by `R tau`, and its error bar is
  `sqrt(mean / N) / (R tau)` for `N` shots (both to 1 part in 10^6).
- Bright and dark histograms in the Poisson limit: the fitted `eta`
  and background, and their full covariance, equal the closed forms
  from the two Poisson means (to 1 part in 10^6).
- Geometric limit (spin flips always end the record): the fitted
  `eta` is the mean count divided by `lam + 1`, with the geometric
  error bar `sqrt(m (1 + m) / N) / (lam + 1)` (to 1 part in 10^6).
- 300 simulated experiments of 2000 shots each, simulated from the
  counting process itself and fitted from the automatic start: the
  spread of (fit minus truth) / error bar is within 15 % of 1 for
  both `eta` and `lam`, its mean within 0.25, and the goodness-of-fit
  p-value is above 0.01 in at least 95 % of them.
- A wrong model is caught: with 20 % failed preparations among 5000
  bright shots the p-value is below 10^-6, while correctly prepared
  shots give a p-value above 0.01.
- The fit's count distributions equal those `fidelity` uses (to
  1e-12), and keep their relative accuracy far in the tail, where
  they are below 10^-17 (against log-space numerical integration, to
  1 part in 10^6).
- The automatic start returns the true `eta`, `lam` and background
  from the exact expected histogram (to 1 part in 10^6).
- The chi-squared tail probability equals `erfc(sqrt(x/2))` and
  `exp(-x/2)` for one and two degrees of freedom (to 1e-14) and the
  numerically integrated density for 3, 4, 7, 10 and 31 (to 1e-9).
- `SpinFit.residuals_ghz` equals the measured values minus
  `qubit_frequency` of the fitted parameters (to 1e-9 GHz), and
  `SpinFit.p_value` with two degrees of freedom equals
  `exp(-chi2/2)` (to 1e-14).
- A CSV file with a byte-order mark and trailing empty lines loads to
  exactly the values written.
- Error bars on predictions from a readout fit (since 0.7.0): a fitted
  parameter returns its own error bar (to 1e-9 relative), all three
  return the fit's covariance and a linear combination `A x` returns
  `A C A^T` (to 1e-8 relative), and at `eta = 1` the one-sided slope is
  still exact. In the Poisson limit the error bar of a predicted mean
  count equals the closed form `2 sqrt(m / N)` (to 1 part in 10^6).
  For the fidelity at an unmeasured window it agrees within 5 % with
  the spread over 2000 parameter sets drawn from the fit's
  uncertainty. Over 150 simulated experiments of the counting process,
  (prediction minus true fidelity) / error bar has a spread within
  20 % of 1 and a mean within 0.25 of 0.
- The tutorials run, print the numbers documented here, and land
  within 3 error bars of the simulated truth (for the lifetime
  tutorial: both lifetimes, the design's Purcell factor and coupling
  rate, and the fidelity of the simulated lifetimes).

**Lifetimes and the measured cavity budget** (since 0.8.0)

- Without background, with equal bins of width `w` and a window of 40
  lifetimes, the bin index of a count is geometric, so the maximum
  likelihood lifetime is `w / ln(1 + 1 / k_mean)` (`k_mean` the mean
  bin index) with the error bar `tau^2 (1 - p) / (w sqrt(N p))`,
  `p = e^(-w / tau)`. The fit equals both (to 1 part in 10^9 and
  10^6).
- The exact expected histogram (background, and a decay that starts
  at 0, fitted from 0.5 ns on), scaled to 10^12 counts and rounded,
  returns the true lifetime, amplitude and background (to
  1 part in 10^8).
- At the fit the Poisson score equations hold (computed from scratch
  in the test, to 1e-6 of the total count) and the fitted counts add up
  to the measured ones (to 1 part in 10^9).
- 200 simulated experiments (a 0.62 ns lifetime, 4000 decays, 400
  background counts, 16 ps bins): (fit minus truth) / error bar has a
  mean within 0.25 of 0 and a spread within 15 % of 1, and at least
  95 % of the p-values are above 0.01. Two lifetimes fitted as one give
  a p-value below 10^-6.
- `propagate_uncertainty` with a `LifetimeFit` returns the lifetime's
  own error bar (to 1e-9 relative), `sigma / tau^2` for the decay rate
  `1 / tau` (to 1e-6 relative) and `A C A^T` for a linear combination
  (to 1e-8 relative).
- `purcell_from_lifetimes` given the lifetimes of a `CavityInterface`
  design returns its `zeta`, cooperativity, `beta`, `F_C` and `g_hz`
  (to 1 part in 10^12) at Q = 50, 500 and 5000; its error bars are
  exact where a result is linear in one lifetime, and within 2 % of the
  spread of 400000 sampled inputs elsewhere, with the right
  correlation (to 0.01).

**Inputs**

- NaN and infinite values are refused across the parameter classes,
  the cavity, readout and remote-entanglement functions (with array
  input to `polarization_rate` still accepted), and since 0.5.0 by
  `xi_pol_overlap`; a target fidelity must be a real number.

## Corrections in earlier versions

**0.8.0 refuses a cyclicity or Rabi rate at a field too weak to split
the qubit.** At zero field each pair of states is exactly degenerate,
so its two states can be any pair, and the cyclicity and Rabi rate
depend on the direction from which the field goes to zero. At 1e-12 T
the pair is split by only about 100 times the round-off of the
calculation, and the states it returns are off by about a percent;
at 1e-15 T the split is below the round-off. Up to 0.7.0 both functions returned
whatever pair the eigensolver happened to pick: for the SnV- set at
53 degrees the cyclicity came out infinite at zero field, 1.1e34 at
1e-15 T and 9.06 at 1e-12 T, where the weak-field limit is 7.85, and
the Rabi rate came out 2.5 MHz at 1e-15 T instead of 5.93 MHz. Both
now refuse a pair split by less than 1e-8 of the largest energy of its
manifold (for SnV-, fields below a few microtesla). Every result at a
field that splits the pairs is unchanged.

**0.7.0 refuses a cavity whose Purcell factor is below 1.** For a
large-volume resonator, for example Q_L = 1000 and V = 100
`(lam/n)^3`, the ideal Purcell factor `F_max = (3 / 4 pi^2) Q / V` is
0.760. Then `F_C = 1 + (F_max - 1) xi` is below 1, and the share of
photons in the cavity mode, `a (F_C - 1) / zeta`, is negative. 0.6.0
returned `beta = eta = -0.0281` and `g_hz = NaN` there, and
`require_valid()` then blamed weak coupling. `CavityInterface` now
refuses with a message that names `F_max`. With no overlap
(`xi_pos = 0`) nothing changes, and `beta_form="transition"` is
unchanged. Every result with `F_max >= 1` is unchanged.

Also in 0.7.0: a NaN or infinite `strain_scale` gave NumPy's
"Eigenvalues did not converge", and a NaN Rabi drive gave a NaN rate.
Both are now refused with a message that names the input.

**0.6.0 changed what counts as a photon into the cavity mode.** The
coupled line decays `F_C` times faster with the cavity than without
it (Lee et al., arXiv:2511.05740, Eq. 2). Only the added part,
`F_C - 1`, goes into the cavity mode. Up to 0.5.0, `beta`, `g` and the
cooperativity used `F_C`, so they counted the line's normal emission
as cavity photons. With no overlap at all, `beta` was 0.342 instead
of 0. For the design point of example 2, `beta` drops from 0.909 to
0.862, `eta` from 0.648 to 0.614, `g` from 7.52 to 7.32 GHz and the
cooperativity from 6.60 to 6.26. The fidelity of the test suite's
design point drops from 0.9813 to 0.9802 (best of four windows). The
Purcell factor, lifetime, linewidth and cyclicity do not change.
`beta_form="transition"` gives the old numbers.

**0.5.0 fixed two problems in the inverse readout tools.** Both only
changed answers in edge cases; every other result is unchanged.

- `required_window` scans 60 windows, each about 31 % longer than the
  last, and refused when none of them reached the target. But the
  fidelity can peak between two of them. For the confocal readout of
  example 3 the best grid value is 0.8757 (at 18.7 us), while the
  true peak is 0.8763 (at 16.8 us), so a target of 0.876 was refused
  as unreachable. The function now searches finely around the best
  grid point before refusing, and returns 15.52 us for that target.
- `required_efficiency` started its search at `eta = 1e-6` and
  returned 1e-6 whenever that already reached the target, instead of
  the smallest efficiency. For cyclicity 10^6, a 0.1 s window and a
  target of 0.7 it returned 1e-6; the closed form, and now the
  function, give 6.668e-7.

**0.4.0 fixed four problems in the readout model and the inputs.**

- `fidelity` left the detector background (`noise_rate`) out of the
  bright spin's counts, although detector dark counts and stray light
  do not depend on the spin, and this package's own `readout_counts`
  includes them in the bright mean.
  Background counts can only help the bright state pass the
  threshold, so the old fidelity was too low, sometimes badly: with
  150 signal counts on 200 background counts it returned 0.500
  instead of 0.999998.
- The dark switch-on channel was a 24-point sum over switch times and
  was added to the background error and capped at 1. When the switch
  happens early in the window the sum missed most of it: for the
  confocal readout with `leak = 0.01` (example 3) 0.3.1 gave 0.718,
  while the model's exact value and a Monte Carlo simulation give
  0.500. It is now evaluated exactly and combined with the background
  exactly.
- The mean bright count `eta (Lambda + 1)(1 - exp(-Gp tau))` lost
  digits for very large cyclicity (relative error 6e-5 at 10^15); it
  now uses `expm1`.
- NaN or infinite inputs could pass silently and give NaN or infinite
  results (for example a NaN spin-orbit constant, `purcell_max(nan,
  1)`, `entanglement_rate(inf, 0.5)`), and `CavityInterface` with
  `Q = 0` stopped with a `ZeroDivisionError`. These are now refused
  with a clear message. `required_window` returned 1 ns whenever the
  target was already reached at 1 ns; it now finds the shorter window.

What changed in numbers: see the "Behaviour changes" note in
[CHANGELOG.md](CHANGELOG.md). With no background and no leak every
result is unchanged.

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
- **The readout model is a two-state counting model.** Both count
  distributions are evaluated exactly within it, but it leaves out
  some second-order effects: once a bright spin has flipped to dark
  it produces no leak counts and cannot switch back on, and a dark
  spin's leak counts are counted for the whole window even after it
  has switched on (the leak is small whenever the readout is good).
  The `nt_switch` argument of `fidelity` is ignored since 0.4.0.
- **Leak counts belong to the dark spin only.** In `fidelity` the
  bright spin gets the background (`noise_rate`) but no `leak`
  counts, because the leak is off-resonant scattering of the dark
  spin. `readout_counts` keeps the simpler bookkeeping of its source
  and adds the whole dark mean, leak included, to the bright mean, so
  with `leak > 0` its bright mean exceeds `fidelity`'s
  `n_bright_total` by exactly `eta leak R tau` (example 3 with
  `leak = 0.01`: 4.3176 against 4.2166). With `leak = 0` they agree.
- **The D line's cavity-enhanced emission is not in the budget.**
  `CavityInterface` reports the D line's Purcell factor (`f_partner`,
  when you give `delta_partner_hz`), but `zeta`, `beta` and `eta`
  count only the coupled line. In a very broad cavity the D line is
  enhanced too. At the revised study's design point (D line 0.90 THz
  from a 0.56 THz wide cavity, `F_D = 3.4`) this leaves the decay 10.23
  times faster instead of 10.50, and `beta` at 0.902 instead of 0.905.
- **No input for light the dark spin scatters through its own line
  (B2).** `leak` is excitation that switches the dark spin on, as
  excitation through the spin-flipping line A2 does. Once the optical
  line is Purcell-broadened, the revised study finds that most of the
  dark spin's light comes instead through B2, only 0.73 GHz from the
  laser, which leaves the spin dark. This model cannot describe that, so it does not reproduce
  the revised study's four-level readout (98.8 % in 0.1 us). Do not
  put the B2 excitation into `leak`: every such excitation would then
  switch the spin on, and the fidelity would come out far too low.
- **The measured cyclicity near alignment depends on alignment.** At
  the nominal angles the model gives a much lower cyclicity than the
  measured 2244; see example 1.
- **Spin coherence times** (T1, T2) are sample-dependent measurements
  and are not part of the model.
- **SiV- and GeV- ship without strain, quenching or emission budget**;
  supply or fit your own.
- **The readout calibration** (`fit_readout_histograms`) holds
  `gamma`, `s`, `tau` and `leak` fixed, and its error bars are
  large-sample error bars: they assume many shots, and the fit
  refuses a parameter whose error bar exceeds a quarter of its value.
  It describes the same two-state counting model as `fidelity`, so
  effects outside that model (failed preparation, detector dead time,
  drifts during the run) show up as a small `p_value`, not as a
  correction.
- **The `lab` fits** use a local Levenberg-Marquardt search (a standard
  step-by-step least-squares method) from your starting set; the error
  bars assume independent Gaussian measurement errors, and the greedy
  design is a heuristic. `propagate_uncertainty` is first-order: when
  an error bar is large compared with the scale on which the predicted
  quantity curves (or reaches a boundary such as zero strain), sample
  the fit's uncertainty instead.
- **Lifetime fits** (`fit_lifetime_histogram`, since 0.8.0) fit one
  exponential decay plus a constant background. They do not
  deconvolve the instrument response (start the fit after it with
  `t_start`), and they assume that each bin count is Poisson (no
  pile-up: keep the detection probability per pulse small) and that
  each decay ends within the pulse period. The error bars are
  large-sample error bars. Two lifetimes, or a background that is not
  flat, show up as a small `p_value`.
- **`purcell_from_lifetimes`** (since 0.8.0) reads the whole
  shortening of the lifetime as Purcell enhancement on the coupled
  line. The reference lifetime must be that of the same emitter with
  the cavity tuned away; anything else that changes between the two
  measurements is counted as cavity effect. The two error bars are
  taken as independent, and the propagation is first-order.
- **`optimal_window`** finds the highest peak it sees on an 80-point
  scan and refines it. The fidelity is the larger of smooth curves (one
  per threshold), so it can have more than one local peak; a narrow
  peak between two scan points could be missed. Its error bar from
  `propagate_uncertainty` is first-order, like every other one here.

## Where it comes from

`vacspin` is the general-purpose engine distilled from the
SnV-on-TFLN (tin-vacancy centre on thin-film lithium niobate) interface
study below, and generalised to any group-IV
colour centre and any cavity. The study's repository holds the device
design (FEM/EME/GME electromagnetic simulations) and reproduces the specific study.

> T. M. Mahim, M. M. Rahman and A. S. M. Mohsin, "Fast single-shot
> readout of tin-vacancy spins with an overcoupled diamond nanocavity
> on thin-film lithium niobate" (submitted to Optics Express, 2026;
> revised manuscript); pipeline:
> https://github.com/Tanvir-Mahmud-Mahim/snv-tfln-cavity-interface;
> data: https://doi.org/10.5281/zenodo.23507013

The revised manuscript moved the design point to an overcoupled cavity
with `Q_L = 870` (`F_C = 28`, 32 % of photons detected) and reads the
spin with a four-level model. `tests/test_revised_study.py` checks
this package against its printed numbers. `tests/test_integration.py`
keeps the `Q_L = 500` design point of the original submission as a
regression test of the whole chain.

Formalism: Rosenthal et al., PRX 13, 031022 (2023) and PRX 14,
041008 (2024) (arXiv:2403.13110); decay-rate budget of the coupled
transition: Lee et al., arXiv:2511.05740; heralding: Barrett and Kok, PRA 71, 060310(R)
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
