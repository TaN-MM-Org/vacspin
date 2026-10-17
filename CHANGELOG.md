# Changelog

Every physical claim added in any release is pinned by a test against
an exact result, a published measurement, or two independent code
paths; the release notes on GitHub carry the full anchor lists.

## 0.7.0 (2026-10-17)

The readout window with the highest fidelity, error bars on anything
predicted from a readout calibration, a refusal for cavities with a
Purcell factor below 1, faster spin fits, and two worked tutorials.

### Added

- `optimal_window(eta, lam, gamma, s, leak, noise_rate, f0, tau_min,
  tau_max)` (`vacspin.readout`, exported at top level): the readout
  window with the highest single-shot fidelity, for readouts whose
  fidelity rises and then falls (background or leak). An 80-point
  log scan plus a golden-section search around the best point; returns
  the `fidelity` result at that window plus `tau`. It refuses when
  there is no peak inside the range (the fidelity at `tau_max` within
  1e-9 of the best, as without background and leak; or the best at
  `tau_min`), `eta = 0`, and bad `tau_min`/`tau_max`. Tests: with a
  threshold of one photon and background rate N, setting the slope of
  F(tau) = 1/2 + e^(-N tau) (1 - P0(tau)) / 2 to zero gives the closed
  form tau* = ln(1 + a/N) / a, a = R/(Lambda + 1) + eta R (derived in
  the test); in three cases the window matches it to 1e-5 relative and
  the fidelity to 1e-10. With a leaking dark spin and at the cavity
  design point, no window of a 400-point scan or of a fine scan within
  5 % beats it (beyond 1e-12).
- `propagate_uncertainty` now also takes a `ReadoutFit` from
  `fit_readout_histograms`; the function then receives a dict with
  `eta`, `lam` and `noise_rate`. Same first-order propagation
  J cov J^T, one-sided only where a step leaves the model's range.
  Tests: exact for linear quantities (own error bar to 1e-9 relative;
  full covariance and A C A^T to 1e-8 relative; one-sided at eta = 1);
  the Poisson-limit closed form 2 sqrt(m/N) for a predicted mean count
  (1e-6 relative); the fidelity at an unmeasured window within 5 % of
  the spread over 2000 parameter sets drawn from the fit's
  uncertainty; over 150 simulated experiments of the counting process
  the pulls have a spread within 20 % of 1 and a mean within 0.25.

### Fixed

- `CavityInterface` with an ideal Purcell factor below 1 and any
  overlap returned a negative photon share. Cause: with
  F_max = (3/4 pi^2) Q/V < 1, F_C = 1 + (F_max - 1) xi < 1, so
  `beta = a (F_C - 1) / zeta` < 0. Example: Q_L = 1000, V = 100
  (lam/n)^3 (F_max = 0.760), xi_pol = 2/3, xi_pos = 0.5, eta_wg = 1:
  before,
  `beta = eta = -0.0281` and `g_hz = cooperativity = NaN` (with a
  RuntimeWarning), and `require_valid()` blamed "weak_coupling";
  after, a `ValueError` naming F_max and the condition
  Q/V >= 4 pi^2/3. With no overlap (F_C = 1, beta = 0) and with
  `beta_form="transition"` nothing changes; a guard keeps F_max = 1 up
  to round-off at beta in [0, 1e-12] (tested). Every result with
  F_max >= 1 is unchanged. Test:
  `test_cavity_with_f_max_below_one_is_refused`.

### Changed

- A NaN or infinite `strain_scale` is refused with a `ValueError` that
  names it, in `solve`, `qubit_frequency`, `cyclicity`,
  `transition_table`, `rabi_rate` and the `vacspin.lab` fits (before:
  NumPy's `LinAlgError` "Eigenvalues did not converge"). Test:
  `test_non_finite_strain_scale_refused`.
- `rabi_rate` refuses a drive field with a NaN or infinite component
  (before: it returned NaN). Test:
  `test_rabi_rate_refuses_non_finite_drive`.
- Type hints on `solve`, `rabi_rate`, `propagate_uncertainty` and the
  new function.
- Documentation: README sections for the new tools, a Tutorials
  section, the refusals, the checks, a 0.7.0 correction note and a
  limit for `optimal_window`; the examples header now says the outputs
  were printed with 0.7.0 (it said 0.5.0, although example 2 already
  showed the 0.6.0 numbers). README examples 1 to 7 print the same
  numbers as in 0.6.0. Test count 122 (103 before).

### Performance

- The `vacspin.lab` fits evaluate all observations at once: one
  rotation of every field and one stacked `eigvalsh` per manifold,
  instead of a Python loop over points. Each Hamiltonian is built with
  the same arithmetic as `h_manifold`; on the maintainer's machine the
  results are bit-identical to 0.6.0 with NumPy 1.22.0 and 2.5.3, and
  the test holds them to the kept point-by-point path within 1e-12 GHz
  (150 random designs, three centers) and the fit, planner and
  propagation results to 1e-12 relative
  (`test_batched_observables_equal_the_pointwise_path`). Measured CPU
  time, median of 3 runs of 15 calls each, same machine, Python 3.13,
  NumPy 2.5.3: `fit_spin_parameters` (6 points, 2 parameters)
  68.2 ms -> 25.6 ms (2.7x); `parameter_information` (20 points,
  4 parameters) 7.5 ms -> 1.75 ms (4.3x); `design_fields` (8 of 20)
  17.2 ms -> 4.9 ms (3.5x); the 400-fit Monte Carlo test 15.6 s ->
  6.3 s (2.5x). The whole suite (wall time, median of 3 on a shared
  machine) went from 32 s for 103 tests to 18 s for 122.

### Tutorials

- `examples/tutorial_readout_window.py`: from simulated bright and
  dark count histograms (fixed seed) to the calibrated efficiency,
  cyclicity and background, the best window and its fidelity, each
  with an error bar, checked against the closed-form window.
- `examples/tutorial_spin_to_readout.py`: from simulated qubit and
  orbital frequencies of a strained SnV- center to the fitted
  spin-orbit and strain constants, the cyclicity and qubit frequency
  at a readout field, the Q_L = 500 cavity budget, the best window and
  the fidelity with the calibration's error bar carried through.
- Each runs in under a second; their output is in the README and in
  each script's docstring, and `tests/test_tutorials.py` runs both
  and checks the printed numbers.

## 0.6.0 (2026-10-10)

The photon share of the cavity mode now counts only the emission the
cavity adds, an opt-in full-Zeeman microwave drive, and citation
corrections.

### Changed

- `CavityInterface`: `beta`, `g_hz` and `cooperativity` now use
  `F_C - 1` instead of `F_C`. The coupled line decays `F_C` times faster
  with the cavity than without it (Lee et al., arXiv:2511.05740, Eq. 2);
  only the added part goes into the cavity mode, while its normal
  emission still goes where it went before. With no overlap at all
  (`xi_pos = 0`), 0.5.0 gave `beta = 0.342` (the radiative fraction) and
  a nonzero `g`; both are now exactly 0. The budget now obeys the
  bad-cavity relations `zeta = 1 + C` and `beta = C / (1 + C)`. For
  README example 2, `beta` goes from 0.909 to 0.862, `eta` from 0.648 to
  0.614, `g` from 7.52 to 7.32 GHz and the cooperativity from 6.60 to
  6.26; the 50 ns fidelity printed there goes from 0.99966 to 0.99964.
  For the companion study's design point (test suite), `eta` goes from
  0.6486 to 0.6150 and the best-window fidelity from 0.9813 to 0.9802.
  `f_max`, `f_c`, `f_c2`, `zeta`, `gamma_cav`, `lambda_cav` and
  `kappa_hz` are unchanged. `beta_form="transition"` gives the 0.5.0
  numbers. Test: `test_only_the_cavity_added_emission_goes_into_the_mode`.

### Added

- `rabi_rate(..., full_zeeman=True)`: drives with the same Zeeman terms
  as the static Hamiltonian, including the orbital factor `f_g` and the
  anisotropy `(1 + 2 delta_g)`. The default is unchanged: the
  spin-Zeeman drive of Rosenthal et al., PRX 13, 031022 (2023), Eq. B10,
  which the paper writes with `f_g = delta_g = 0`. For the SnV- set at
  the readout field and a 0.6 mT drive, the full form changes the rate
  by -3.6 % (lab x), +4.1 % (lab y), +1.7 % (lab z) and +18.8 % (along
  the symmetry axis). Test: `test_rabi_rate_full_zeeman_option` (equal
  to the default with no orbital or anisotropy terms, and equal to the
  matrix element of `H(b0 + b_drive) - H(b0)`).

### Fixed (documentation)

- `cavity.py` said it followed Lee et al. "Eqs. 1-10". Lee et al.
  give the decay-rate budget (Eqs. 1-3) and the polarization overlap
  (Eqs. 6-10); the quantum efficiency, the ideal Purcell factor and the
  position overlap are not in that paper. The text now says which part
  comes from where.
- `gev_bhaskar2017`: the splittings were called "unstrained", but
  Bhaskar et al., PRL 118, 223603 (2017) do not report the strain of
  their emitters. The word is removed, and the docstring says the set
  treats the splittings as pure spin-orbit. The numbers are unchanged.
- Rosenthal et al., arXiv:2403.13110 is now also cited by its journal
  version, Phys. Rev. X 14, 041008 (2024).
- Packaging: the license is now given as the SPDX expression
  `license = "Apache-2.0"` with `license-files = ["LICENSE"]`, and the
  old license table and license classifier are gone. Building no longer
  prints setuptools' deprecation warnings (their support ends on
  2027-02-18). Building from source now needs setuptools 77 or newer;
  installing the package is unchanged.

## 0.5.0 (2026-10-10)

Readout calibration from a lab's own photon-count histograms, a
goodness-of-fit test for both fits, and two fixes in the inverse
readout tools.

### Added

- `fit_readout_histograms` and `ReadoutFit` (`vacspin.lab`, exported
  at top level): fit the detection efficiency `eta`, the cyclicity
  `lam` and the background `noise_rate` (any subset) to measured
  bright and, optionally, dark single-shot count histograms by
  maximum likelihood, using the same count distributions as
  `fidelity`. Fisher scoring from a method-of-moments start;
  large-sample error bars (inverse expected information); a Pearson
  goodness-of-fit test with bins pooled to at least 5 expected shots.
  It refuses malformed histograms, fewer than 10 shots, parameters
  the data cannot tell apart, an error bar above a quarter of the
  fitted value, a best fit at eta = 1, and non-convergence. Tests:
  closed-form maximum-likelihood values and error bars in the Poisson
  limit, the joint bright/dark Poisson limit (full covariance) and the
  geometric limit (to 1e-6 relative); 300 experiments simulated from
  the counting process (pull spread within 15 % of 1); a wrong model
  (20 % failed preparations) gives p < 1e-6; the likelihood's count
  distributions equal `fidelity`'s to 1e-12 and keep relative accuracy
  in the far tail (log-space quadrature, 1e-6 relative); the
  moment start recovers exact parameters.
- `SpinFit.p_value` (chi-squared tail probability of the fit) and
  `SpinFit.residuals_ghz` (measured minus fitted values). The new
  field has a default, so a `SpinFit` built by hand as before still
  works. Tests: residuals against `qubit_frequency` (1e-9 GHz);
  p-value against exp(-chi2/2) for 2 degrees of freedom (1e-14); the
  chi-squared tail function against erfc and numerical integration.
- README example 7 (readout calibration) and a p-value line in
  example 5.

### Fixed

- `required_window` refused a reachable target when the fidelity
  peaks between two points of its 60-point search grid. Confocal
  readout of example 3 (eta 0.002, Lambda 2244, s = 10, background
  4000/s, tau_max 1 s), target 0.876: before, `ValueError` "unreachable
  (best 0.8757)"; after, 15.52 us (the peak is 0.8763 at 16.8 us). It
  now searches finely (log-spaced scan plus golden section) around
  the best grid point before refusing. Test:
  `test_required_window_finds_a_peak_between_grid_points` (closed-form
  threshold-1 fidelity; the returned window equals the closed-form
  crossing to 1e-5 relative).
- `required_efficiency` returned its search start eta = 1e-6 whenever
  that already reached the target. Lambda = 1e6, gamma = 1/(4.5 ns),
  tau = 0.1 s, s = 1, target 0.7: before 1e-6, after 6.668e-7
  (closed form 6.668e-7). The
  lower end now moves down by factors of 1000 until it falls short.
  Test: `test_required_efficiency_below_the_search_start`.

### Changed

- `required_window` and `required_efficiency` refuse a target
  fidelity that is not a real number with a `ValueError` naming it
  (a string such as "0.9" raised `TypeError` before).
- `xi_pol_overlap` refuses a NaN or infinite angle (it returned NaN).
- `load_observations_csv` reads the file as UTF-8 with an optional
  byte-order mark and skips empty lines, so a file saved by a
  spreadsheet as "CSV UTF-8" loads (its header was refused before).
- Type hints on `required_window`, `required_efficiency`,
  `xi_pol_overlap` and the new functions.
- Documentation: README sections for the new tools, their checks and
  refusals; test count 101 (84 before). No other result changes: the
  README examples 1 to 6 print the same numbers as in 0.4.0.

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
