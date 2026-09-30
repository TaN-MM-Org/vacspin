# Changelog

Every physical claim added in any release is pinned by a test against
an exact result, a published measurement, or two independent code
paths; the release notes on GitHub carry the full anchor lists.

## v0.4.0 - 2026-09-30

Exact readout statistics for both spin states, error bars on anything
predicted from a fit, stricter inputs, and fits about five times
faster.

### Added

- `propagate_uncertainty(fit, func)` (`vacspin.lab`, exported at top
  level): the value, 1-sigma error bar and covariance of any number
  or array computed from a `SpinFit`'s parameters (a qubit frequency
  at an unmeasured field, a cyclicity, a splitting), by first-order
  error propagation J cov J^T with central-difference slopes.
  The step is relative to the parameter's value, or to its error bar
  when that is larger; the slope is one-sided only when the lower
  point is not a valid parameter set (e.g. at ups = 0), and any error
  raised by `func` itself reaches the caller. Tests: exact for linear
  quantities (own error bar to 1e-9 relative; sigma/sqrt(n) to
  1e-9 GHz; A C A^T to 1e-8 relative; one-sided at ups = 0 to 1e-6
  relative) and within 5 % of the spread of 4000 parameter sets
  sampled from the fit's Gaussian uncertainty for a nonlinear
  prediction.
- `fidelity` returns three more keys: `n_bright_total` and
  `n_dark_total` (means of the full bright and dark count
  distributions it uses) and `truncation_bound` (a guaranteed bound,
  at most 1e-9, on the error from tracking finitely many counts).
- README example 6 (error bars on predictions) and a leak case in
  example 3.

### Fixed

- `fidelity` left the detector background out of the bright state's
  count distribution, although detector dark counts and stray light
  do not depend on the spin and `readout_counts` includes the
  background in the bright mean. The background is now
  convolved exactly into the bright distribution. The old result was
  too low, badly so when the background is comparable to the signal.
  Test: without spin flips and leak, `fidelity` equals the
  best-threshold Poisson closed form to 1e-9 (fails on 0.3.1).
- The dark switch-on channel was a 24-point sum over switch times,
  added to the background error and capped at 1. When the switch
  happens early in the window the sum missed most of it. It is now
  evaluated exactly: the bright time after a switch has an atom at
  zero plus a two-exponential density, and its Poisson mixture is
  computed in closed form (a Poisson tail, or a stable recurrence
  when the switch-on rate is fast), then convolved with the
  background. Tests: direct double numerical integration (1e-6 per
  value), normalisation (1e-12), closed-form mean (1e-9 relative),
  and 400 000-shot Monte Carlo of the counting process in three
  cases (within 5 standard errors).
- The mean bright count eta (Lambda + 1)(1 - exp(-Gp tau)) lost
  digits for very large cyclicity; now computed with `expm1`
  (`readout_counts` and `fidelity`). Test: matches the series limit
  to 1e-12 for Lambda = 1e9 to 1e15.
- `required_window` returned 1 ns whenever the 1 ns grid point already
  reached the target; it now searches below it. Test: the closed-form
  shortest window of a flip-free emitter (to 1e-5 relative).
- Non-finite inputs are refused instead of producing NaN or infinity:
  every number in `SpinParameters`; `qubit_frequency_perpendicular`;
  `purcell_max`, `lorentzian_suppression`; `CavityInterface` (`Q`,
  `V`, `omega_q_hz`, and `delta_partner_hz`, which must also be
  positive); `leak`, `noise_rate`, mean counts, `polarization_rate`
  inputs (arrays still accepted), `tau_max`; `entanglement_rate`.
  `CavityInterface` with `Q = 0` raised `ZeroDivisionError`; it now
  raises `ValueError`.

### Changed

- `h_manifold` builds the Hamiltonian from Kronecker products
  computed once at import, and the `lab` observables diagonalise only
  the manifold they need, for eigenvalues only. Same matrices (tested
  against the term-by-term construction to 1e-12 GHz; the lab path
  against `solve` to 1e-9 GHz). The Monte Carlo error-bar test went
  from 89 s to 16 s; the whole suite from 93 s to about 22 s.
- `required_window` stops scanning at the first window that reaches
  the target (same result; fewer evaluations).
- `fidelity(nt_switch=...)` is ignored and gives a
  `DeprecationWarning`; the argument is kept so old calls still run.
- `fidelity` raises `ValueError` when keeping its 1e-9 accuracy bound
  would need more than 2^18 counts (a very long window with a very
  high background); 0.3.1 silently truncated at 8000 counts.
- `snv_emission().reference` now gives the full Goerlitz et al.
  citation (New J. Phys. 22, 013048 (2020)) and credits Lee et al.
  for the zero-phonon line. Values unchanged.
- Documentation: the confocal example's eta = 0.2 % is now described
  as chosen to reproduce the paper's ~4 bright counts with this
  package's mean-count formula; the paper's own fitted efficiency is
  about 0.1 % (0.3.1 implied 0.2 % was the paper's value).

- Documented (README Limits, `readout_counts` and module docstrings)
  and tested: `fidelity` gives the bright spin the background but no
  leak counts, because the leak is off-resonant scattering of the
  dark spin; `readout_counts` keeps its source's bookkeeping and adds
  the whole dark mean, leak included, to the bright mean. With
  leak > 0 they differ by exactly eta leak R tau.

### Behaviour changes

With `noise_rate = 0` and `leak = 0` the two distributions reduce to
the 0.3.1 ones (the background convolution is with a point mass at
zero counts and there is no switch-on), so those results are
unchanged up to round-off. With
background or leak, `fidelity` and the inverse tools change
(0.3.1 -> 0.4.0):

- 150 signal counts on 200 background counts, no flips
  (eta 0.5, tau 4.05 us, noise 4.94e7/s, s = 2): 0.500000 -> 0.999998.
- eta 0.05, tau 2 us, noise 2e5/s, s = 2, no flips: 0.98516 -> 0.98805.
- Confocal readout of example 3 (eta 0.002, Lambda 2244, tau 50 us,
  s = 10, noise 4000/s): 0.82566 -> 0.84076 (with f0 = 0.935:
  0.80449 -> 0.81861); window for 85 %: 9.35 us -> 8.81 us.
- Same with leak = 0.01: 0.7181 (threshold 2) -> 0.5001
  (threshold 14). A 400 000-shot Monte Carlo of the model's counting
  process gives 0.492 +/- 0.001 at the old threshold 2, which the old
  value missed, and 0.500 at the new threshold 14 (tested within 5
  standard errors).
- eta 0.3, Lambda 500, tau 2 us, leak 1e-3, no background:
  0.94544 -> 0.94425.
- Cavity design point (Q = 500, computed leak 0.0037, 1000 counts/s,
  s = 2), windows 10/20/30/50/100 ns: 0.97102/0.98182/0.97780/
  0.97014/0.95573 -> 0.97090/0.98134/0.97707/0.96893/0.95336; at
  100 ns with f0 = 0.99: 0.95117 -> 0.94883; window for 85 %:
  3.5282 ns -> 3.5291 ns. The integration test's thresholds (> 98 %
  at the best window, > 94 % at 100 ns) still hold.
- `readout_counts` bright mean at Lambda = 1e15 (eta 0.5,
  tau 4.05 us, s = 2): 149.99113 -> 149.99999999998 (series limit
  150 (1 - 7.5e-14)).
- `required_window(0.6, eta=0.9, lam=1e15, gamma=1.6e9, s=10)`:
  1.0e-9 s -> 3.409e-10 s (closed form 3.409e-10 s).
- `CavityInterface(q_loaded=0, ...)`: `ZeroDivisionError` ->
  `ValueError`; NaN or infinite inputs listed above now raise
  `ValueError`.

### Tests

- 84 tests (62 before); new file `tests/test_readout_exact.py`, new
  tests in `test_hamiltonian.py`, `test_lab.py` and
  `test_params_remote.py`. `test_observables_match_solver_paths` now
  compares the lab path with `qubit_frequency` to 1e-9 GHz (the
  tolerance the README already stated) instead of bit equality: the
  two now use different LAPACK routines and differ by ~2e-13 GHz.
- Verified locally on Python 3.9 and 3.11 (current NumPy) and on the
  oldest-dependencies setup (Python 3.10, NumPy 1.22.0,
  pytest 7.0.0); on 3.10 and 3.11 also with warnings turned into
  errors. All README examples re-run and match their printed output.

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
