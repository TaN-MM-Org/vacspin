# Changelog

Every physical claim added in any release is pinned by a test against
an exact result, a published measurement, or two independent code
paths; the release notes on GitHub carry the full anchor lists.

## v0.3.1 - 2026-09-22

A review release: three input and unit fixes, a wider CI, and a
rewritten README.

### Fixed

- `CavityInterface.g_hz` multiplied the cavity linewidth `kappa_hz`
  (Hz) by the radiative decay rate (1/s) without the factor 1/(2 pi),
  so `g_hz` (and `g_mhz` in `summary()`) was too large by sqrt(2 pi),
  about 2.5 times, and the `weak_coupling` check could refuse designs
  that meet its stated condition g < kappa/10 (for example Q_L = 3000,
  V = 0.68 with the SnV- budget, xi_pol = 2/3 and xi_pos = 0.5). The decay rate now enters as the
  linewidth gamma/(2 pi) in Hz, as the bad-cavity check already did.
  The Purcell factors, efficiencies, lifetime, cyclicity and
  cooperativity are unchanged.
- `readout_counts` accepted a negative `leak` or `noise_rate` and
  returned a negative dark count; it now refuses them, as `fidelity`
  already did.
- The readout functions accepted non-finite `gamma`, `tau` or `s`:
  `fidelity` returned NaN for `tau = inf` and failed with an unrelated
  integer-conversion error for `gamma = NaN`. They now refuse
  non-finite values with a clear message.

### Tests

- New: `test_coupling_rate_in_consistent_hz_units` (test_cavity.py),
  `test_readout_counts_refuses_negative_background` and
  `test_non_finite_rates_and_windows_refused` (test_readout.py). Each
  fails on 0.3.0.
- `test_purcell_formula_and_identity` now checks
  F_C = 4 g^2 / (kappa gamma_rad) with every rate in Hz.
- 62 tests (59 before).
- CI now also runs Python 3.10 (the matrix skipped it), and a new
  `oldest-dependencies` job runs the suite on Python 3.10 with
  NumPy 1.22.0 and pytest 7.0.0, the lowest versions allowed.

### Changed

- README rewritten for readers outside the field: a guide to the
  terms, units and conventions, six runnable examples with their exact
  output, every public name, the refusals, and each test's real
  tolerance.
- Docstrings: `vacspin.params` said "Two sets ship" (three centres
  ship); the `vacspin.readout` module docstring gave the geometric
  limit's mean as eta Lambda, while the code and test use
  eta (Lambda + 1).

### Corrections to earlier notes

- v0.2.0 said the greedy design "obeys the exact rank-one determinant
  identity and its recomputed greedy rule". The identity is tested on
  random matrices, not on `design_fields`, and no test recomputes the
  greedy rule; the test checks that the chosen set is at least as
  informative as 30 random subsets of the same size.
- Several earlier notes and the old README called checks "exact" or
  "machine precision" that the tests hold to a tolerance (for example
  the Kramers degeneracy to 1e-9 GHz, the dipole sum rule to 1e-10,
  the Poisson and geometric limits to 1e-10, the Purcell identity to
  1 part in 10^9), and "400 seeded Monte Carlo experiments match the
  reported error bars" means within 15 %. The measured SnV- values are
  reproduced within 0.2 % (splitting), 2 % (qubit frequency) and 10 %
  (cyclicity); the Rabi check is a 1-20 MHz window. The README now
  gives each tolerance.
- The old README said the readout statistics use "no numerical
  quadrature". That holds for the bright-count distribution; the
  dark-state switch-on channel is a numerical sum over 24 switch times.
- The old README said CI runs Python 3.9-3.14; 3.10 was missing until
  this release.

## v0.3.0 - 2026-09-18

The third measured centre, and a future-proofing pass.

- `gev_bhaskar2017()`: the germanium-vacancy centre with its
  measured unstrained orbital splittings -- 152 GHz ground, 981 GHz
  excited, zero-phonon line 602 nm (Bhaskar et al., PRL 118, 223603
  (2017)) -- shipping, as for SiV-, only what its source states:
  strain and orbital quenching are sample-specific, deliberately
  zero, and fittable from your own frequencies with `vacspin.lab`
  (context: Siyushev et al., PRB 96, 081201(R) (2017) measured
  170 GHz on a strained emitter; Senkalla et al., PRL 132, 026901
  (2024) measured 24.1 ms coherence below 300 mK).
- PbV- documented but not shipped, on purpose: the ground splitting
  is measured (~3900 GHz; Wang et al., ACS Photonics 8, 2947 (2021),
  confirmed by Chen et al., arXiv:2605.27841 (2026)), the excited
  splitting is not -- and this package does not ship half a
  parameter set.
- CI now also runs on Python 3.14.
- Anchors: GeV values locked to their source; the zero-strain
  closed form Delta = lam exact for both manifolds through the full
  solver; the dipole sum rule and the exact aligned-field
  protection hold for the new centre.

## v0.2.0 - 2026-09-17

Lab adaptability: plan, measure, calibrate -- fit the spin parameters
of YOUR sample from YOUR measurements, with honest error bars, and
know before spending beam time whether a planned experiment can
determine them at all.

- `lab.fit_spin_parameters`: fit any subset of the ten spin parameters
  to measured qubit frequencies and orbital splittings versus field
  (pure NumPy Levenberg-Marquardt), returning a ready-to-use
  `SpinParameters` whose mandatory `reference` records the fit and the
  base set, plus per-parameter error bars, the full covariance and a
  chi-squared when measurement errors are supplied.
- `lab.parameter_information`: the same (J^T W J) matrix before any
  data exists -- predicted error bars for a planned design, and an
  identifiability verdict instead of a surprise after the beam time.
- `lab.design_fields`: greedy D-optimal selection of the most
  informative subset of candidate observations (Pukelsheim, Optimal
  Design of Experiments, SIAM (2006)), refusing candidate lists that
  cannot identify the parameters.
- `lab.save_observations_csv` / `load_observations_csv`: a plain,
  checked CSV contract for observation records; round trips are exact.
- Anchors: noiseless fits recover the cited SnV- truth to numerical
  precision; the exact linear case (ups = 0 zero-field splitting) hits
  the textbook sigma/sqrt(n) closed form; planner and fit covariance
  agree as two code paths of one matrix; 400 seeded Monte Carlo
  experiments match the reported error bars; the zero-field
  lam-versus-ups degeneracy is refused via an exact rank argument; the
  greedy design obeys the exact rank-one determinant identity and its
  recomputed greedy rule.

## v0.1.0 - 2026-09-17

First release: the general-purpose engine distilled from the
SnV-on-TFLN interface study (Mahim, Rahman and Mohsin, submitted to
Optics Express, 2026), generalized to any group-IV colour centre and
any cavity.

- `hamiltonian`: the shared group-IV effective Hamiltonian (spin-orbit
  + strain/Jahn-Teller + spin and orbital Zeeman; Rosenthal et al.,
  PRX 13, 031022 (2023), Eqs. B1-B5) in pure NumPy, with the exact
  frame rotation, the closed-form zero-field splitting and the
  perturbative transverse-field qubit frequency (Eq. B9).
- `transitions`: dipole operators (Eq. B12), the full 16-transition
  table, cyclicity (Eq. B15), microwave Rabi rates, field sweeps.
- `params`: cited sets with mandatory provenance -- the SnV- device
  values of Rosenthal et al. (2023) with Thiering-Gali quenching, the
  SnV- emission budget (arXiv:2403.13110; Goerlitz et al.; Lee et
  al., arXiv:2511.05740), and the unstrained SiV- splittings of Hepp
  et al., PRL 112, 036405 (2014) -- each shipping only what its
  source states.
- `cavity`: the transition-resolved Purcell budget (F_max, F_C, the
  detuned spin-flipping partner, lifetime factor zeta, beta, the full
  detection chain eta, cavity-boosted cyclicity, g and cooperativity
  with F_C = 4 g^2 / (kappa gamma_rad) as an identity), and
  `require_valid()`: the bad-cavity, weak-coupling and
  spin-selectivity conditions refuse instead of extrapolating.
- `readout`: exact single-shot counting statistics -- the bright
  record as a Poisson process terminated by the spin flip, in CLOSED
  FORM (an incomplete-gamma recurrence, no numerical quadrature),
  dark counts with off-resonant scattering and dark-state switch-on,
  optimal integer threshold -- plus the threshold-1 closed form
  (Eq. A20) and the inverse tools `required_efficiency` and
  `required_window` with honest refusals.
- `remote`: the Barrett-Kok two-photon heralding budget
  (PRA 71, 060310(R) (2005)), with its intrinsic factor 1/2 stated
  as a ceiling, not an engineering loss.
- Anchors (asserted, never stored): Hermiticity and exact Kramers
  doublets; the closed-form splitting vs full diagonalisation; the
  measured SnV- landscape reproduced with no free parameters
  (902.98 GHz splitting, 3.677 GHz qubit frequency, cyclicity 2244
  and 8.6 at their measured field angles, MHz Rabi scale); the exact
  aligned-field cyclicity divergence; the Purcell identity as an
  exact round trip; exact Lorentzian and overlap limits; the bright
  distribution's machine-precision Poisson and geometric limits; the
  measured confocal operating point (n_b ~ 4 at eta = 0.2%)
  reproduced; inverse-tool round trips and refusals throughout.
