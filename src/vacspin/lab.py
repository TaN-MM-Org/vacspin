"""Adapt vacspin to your own sample: plan, measure, calibrate.

Every sample is different -- strain, orientation and quenching move the
numbers -- so the parameter set that describes YOUR centre has to come
from YOUR measurements. This module closes that loop in three steps,
in the order a lab actually works:

1. `parameter_information`: before measuring anything, ask whether the
   observations you plan to take can determine the parameters you care
   about, and how small the error bars would be.
2. `design_fields`: given a list of candidate observations (field
   points and observable types), pick the subset that pins the
   parameters down best.
3. `fit_spin_parameters`: after measuring, fit the chosen parameters to
   the data, with uncertainties, and get back a ready-to-use
   `SpinParameters` whose `reference` records exactly where it came
   from.

Observables (`kinds`), all in GHz, fields in tesla (lab frame):

* "qubit"     : the ground-state qubit frequency (`qubit_frequency`).
* "orbital_g" : the ground-manifold orbital splitting, the gap between
  the lower and upper Kramers doublets. At zero field this is exactly
  Delta_g = sqrt(lam_g^2 + 4 ups_g^2).
* "orbital_e" : the same splitting of the excited manifold (measured
  optically as the spacing of the transition branches).

Statistics, stated plainly: with measurement errors `sigmas_ghz`, the
error bars come from the standard weighted-least-squares covariance
(J^T W J)^-1, and the pre-measurement predictions use the same matrix
(the Fisher information of independent Gaussian measurements; see e.g.
D. R. Cox and D. V. Hinkley, Theoretical Statistics (1974), or any
statistics text, under "Cramer-Rao bound"). The design tool maximizes
its determinant (D-optimal design; F. Pukelsheim, Optimal Design of
Experiments, SIAM (2006)). Nothing here invents a noise model: without
`sigmas_ghz` the fit scales its error bars from the residual scatter
and says so, and the planner reports information per unit variance.

A set of observations that cannot tell the varied parameters apart is
refused with an explanation, never silently pseudo-inverted. The
classic case: zero-field splittings alone can never separate lam from
ups, because they enter only through sqrt(lam^2 + 4 ups^2).

Since 0.5.0 two more tools check a lab's own data against the model:

* `SpinFit.p_value` and `SpinFit.residuals_ghz`: is the scatter of the
  measured frequencies about the fit consistent with the stated
  errors, and which point is off?
* `fit_readout_histograms`: fit the detection efficiency, cyclicity
  and background of the readout model (`vacspin.readout`) to measured
  single-shot photon-count histograms by maximum likelihood, with
  large-sample error bars and a Pearson goodness-of-fit test.
"""
from __future__ import annotations

import csv
import dataclasses
import math
from typing import Optional

import numpy as np

from .hamiltonian import h_manifold, lab_to_spin
from .params import SpinParameters
from .readout import (_check_background, _check_common,
                      _exp_poisson_integrals, _switch_on_pmf, poisson_pmf)

__all__ = ["SpinFit", "fit_spin_parameters", "parameter_information",
           "propagate_uncertainty",
           "design_fields", "save_observations_csv",
           "load_observations_csv", "ReadoutFit",
           "fit_readout_histograms"]

_KINDS = ("qubit", "orbital_g", "orbital_e")
_VARIABLE = ("lam_g", "ups_g", "lam_e", "ups_e", "f_g", "f_e",
             "delta_g", "delta_e", "theta_rad", "phi_rad")
_COND_MAX = 1e10
_SINGULAR_MSG = ("these observations cannot tell the varied parameters "
                 "apart (singular or near-singular information matrix). "
                 "Measure a combination that responds differently to "
                 "each parameter -- `parameter_information` shows which "
                 "designs work before you spend beam time")


def _invert_information(fisher, vary):
    """Scale-invariant inversion of an information matrix.

    Varied parameters can carry different units (energies in GHz,
    angles in radians), so the raw condition number partly measures
    the units. The identifiability verdict therefore uses the
    correlation-scaled matrix D^-1 F D^-1 with D = sqrt(diag F):
    exact functional degeneracies of the model survive the scaling,
    unit mismatches do not. Returns (identifiable, cond, cov, sigma).
    """
    d = np.sqrt(np.diag(fisher))
    if np.any(d <= 0.0) or not np.all(np.isfinite(d)):
        return False, np.inf, None, None
    fs = fisher / np.outer(d, d)
    sv = np.linalg.svd(fs, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    if not (np.isfinite(cond) and cond <= _COND_MAX):
        return False, cond, None, None
    cov = np.linalg.inv(fs) / np.outer(d, d)
    err = np.sqrt(np.diag(cov))
    return True, cond, cov, {k: float(s) for k, s in zip(vary, err)}


def _observe(params, kind, b_lab, strain_scale):
    """One observable (GHz). Only the manifold it needs is
    diagonalised, and only for its eigenvalues (`eigvalsh`); the tests
    check that this equals the full `solve` path."""
    if kind not in _KINDS:
        raise ValueError(f"unknown observable kind {kind!r}; "
                         f"choose from {_KINDS}")
    if not np.all(np.isfinite(b_lab)):
        raise ValueError("magnetic field must be finite")
    s = float(strain_scale)
    if s < 0:
        raise ValueError("strain_scale must be >= 0")
    bs = lab_to_spin(b_lab, params.theta_rad, params.phi_rad)
    if kind == "orbital_e":
        e = np.linalg.eigvalsh(h_manifold(params.lam_e, s * params.ups_e,
                                          0.0, params.f_e,
                                          params.delta_e, bs))
        return float(e[2] - e[0])
    e = np.linalg.eigvalsh(h_manifold(params.lam_g, s * params.ups_g,
                                      0.0, params.f_g, params.delta_g,
                                      bs))
    if kind == "qubit":
        return float(e[1] - e[0])
    return float(e[2] - e[0])


def _check_inputs(kinds, b_fields_t, values_ghz, sigmas_ghz, vary):
    kinds = [str(k) for k in kinds]
    b = np.atleast_2d(np.asarray(b_fields_t, dtype=float))
    if b.shape != (len(kinds), 3):
        raise ValueError("b_fields_t must have shape (n, 3): one lab-"
                         "frame field vector (tesla) per observation")
    if not np.all(np.isfinite(b)):
        raise ValueError("fields must be finite")
    for k in kinds:
        if k not in _KINDS:
            raise ValueError(f"unknown observable kind {k!r}; "
                             f"choose from {_KINDS}")
    if values_ghz is not None:
        values_ghz = np.asarray(values_ghz, dtype=float)
        if values_ghz.shape != (len(kinds),):
            raise ValueError("values_ghz must have one entry per "
                             "observation")
        if not np.all(np.isfinite(values_ghz)):
            raise ValueError("measured values must be finite")
    if sigmas_ghz is not None:
        sigmas_ghz = np.broadcast_to(
            np.asarray(sigmas_ghz, dtype=float), (len(kinds),)).copy()
        if not np.all(np.isfinite(sigmas_ghz)) or np.any(sigmas_ghz <= 0):
            raise ValueError("sigmas_ghz must be positive and finite")
    vary = tuple(vary)
    if len(vary) == 0:
        raise ValueError("vary must name at least one parameter")
    for name in vary:
        if name not in _VARIABLE:
            raise ValueError(f"cannot vary {name!r}; choose from "
                             f"{_VARIABLE}")
    if len(set(vary)) != len(vary):
        raise ValueError("vary contains a repeated name")
    return kinds, b, values_ghz, sigmas_ghz, vary


def _replace(params, vary, x):
    return dataclasses.replace(
        params, **{name: float(v) for name, v in zip(vary, x)})


def _model_vector(params, vary, x, kinds, b, strain_scale):
    p = _replace(params, vary, x)
    return np.array([_observe(p, k, bi, strain_scale)
                     for k, bi in zip(kinds, b)])


def _jacobian(params, vary, x, kinds, b, strain_scale):
    """Central-difference Jacobian d(observable)/d(parameter)."""
    n, p = len(kinds), len(x)
    jac = np.empty((n, p))
    for j in range(p):
        h = 1e-6 * max(abs(x[j]), 1e-3)
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        try:
            fp = _model_vector(params, vary, xp, kinds, b, strain_scale)
            fm = _model_vector(params, vary, xm, kinds, b, strain_scale)
        except ValueError:
            # one-sided step when the symmetric one leaves the allowed
            # parameter range (e.g. ups = 0)
            fp = _model_vector(params, vary, xp, kinds, b, strain_scale)
            fm = _model_vector(params, vary, x, kinds, b, strain_scale)
            jac[:, j] = (fp - fm) / h
            continue
        jac[:, j] = (fp - fm) / (2.0 * h)
    return jac


@dataclasses.dataclass
class SpinFit:
    """Result of `fit_spin_parameters`.

    params : the fitted `SpinParameters`, ready to use everywhere in
        this package; its `reference` records the fit and the base set.
    values : fitted value of each varied parameter, by name.
    sigma : 1-sigma uncertainty of each varied parameter, by name.
    cov : full covariance matrix in the order of `vary`.
    chi2, chi2_dof : goodness of fit (chi2 is None when no measurement
        errors were given; the error bars are then scaled from the
        residual scatter instead).
    condition_number : of the information matrix; large means barely
        identifiable.
    residuals_ghz : measured minus fitted value of each observation
        (GHz), in input order (since 0.5.0; None for a `SpinFit` built
        by hand).
    p_value : (property, since 0.5.0) the probability that a
        chi-squared variable with `chi2_dof` degrees of freedom is at
        least `chi2`; None without measurement errors or without spare
        degrees of freedom.
    """

    params: SpinParameters
    values: dict
    sigma: dict
    cov: np.ndarray
    chi2: float
    chi2_dof: int
    n_points: int
    condition_number: float
    n_iter: int
    converged: bool
    residuals_ghz: Optional[np.ndarray] = None

    @property
    def p_value(self) -> Optional[float]:
        """Goodness of fit: P(chi2_dof-distributed variable >= chi2).

        A very small value (say below 0.001) means the data scatter
        about the fitted model by more than the stated `sigmas_ghz`
        allow: the errors are underestimated, or the model (with the
        parameters held fixed) does not describe the sample."""
        if self.chi2 is None or self.chi2_dof is None or self.chi2_dof < 1:
            return None
        return _chi2_sf(self.chi2, self.chi2_dof)


def _chi2_sf(x, dof):
    """Survival function P(X >= x) of a chi-squared variable with an
    integer number of degrees of freedom, from the standard closed
    forms (the regularized upper incomplete gamma function Q(dof/2,
    x/2) at integer and half-integer order):

        dof even: exp(-x/2) sum_{j=0}^{dof/2-1} (x/2)^j / j!
        dof odd:  erfc(sqrt(x/2))
                  + exp(-x/2) sum_{j=1}^{(dof-1)/2} (x/2)^(j-1/2) / Gamma(j+1/2)

    Terms are summed in log space, so large x underflows to 0 instead
    of overflowing. The tests check it against exp(-x/2) (dof = 2),
    erfc(sqrt(x/2)) (dof = 1) and direct numerical integration of the
    density."""
    dof = int(dof)
    x = float(x)
    if dof < 1:
        raise ValueError("degrees of freedom must be >= 1")
    if not np.isfinite(x):
        raise ValueError("chi-squared value must be finite")
    if x <= 0.0:
        return 1.0
    h = 0.5 * x
    lh = math.log(h)
    if dof % 2 == 0:
        js = np.arange(dof // 2, dtype=float)
        logs = -h + js * lh - np.array([math.lgamma(j + 1.0) for j in js])
        total = float(np.sum(np.exp(logs)))
    else:
        total = math.erfc(math.sqrt(h))
        if dof > 1:
            js = np.arange(1, (dof - 1) // 2 + 1, dtype=float)
            logs = -h + (js - 0.5) * lh - np.array(
                [math.lgamma(j + 0.5) for j in js])
            total += float(np.sum(np.exp(logs)))
    return min(max(total, 0.0), 1.0)


def fit_spin_parameters(kinds, b_fields_t, values_ghz, params0,
                        vary=("lam_g", "ups_g"), sigmas_ghz=None,
                        strain_scale=1.0, max_iter=200, tol=1e-12):
    """Fit chosen spin parameters to your measured frequencies.

    kinds : sequence of observable names ("qubit", "orbital_g",
        "orbital_e"), one per measurement.
    b_fields_t : (n, 3) lab-frame field vectors in tesla.
    values_ghz : (n,) measured values in GHz.
    params0 : starting `SpinParameters` (a shipped set, or your best
        guess); everything not in `vary` is held fixed at its value.
    vary : names of the parameters to fit.
    sigmas_ghz : optional 1-sigma measurement errors (scalar or (n,)).
        With them the covariance is exact for the stated errors and a
        chi-squared is reported; without them the error bars are scaled
        from the residual scatter, which needs at least one spare
        observation.

    Returns a `SpinFit`. Refuses -- with an explanation -- designs that
    cannot identify the varied parameters, too few points, or a fit
    that does not converge.
    """
    kinds, b, values, sig, vary = _check_inputs(
        kinds, b_fields_t, values_ghz, sigmas_ghz, vary)
    n, p = len(kinds), len(vary)
    if sig is None and n < p + 1:
        raise ValueError(f"{n} observations cannot determine {p} "
                         "parameters and an error scale; add points or "
                         "supply sigmas_ghz")
    if n < p:
        raise ValueError(f"{n} observations cannot determine {p} "
                         "parameters")
    w = np.ones(n) if sig is None else 1.0 / sig
    x = np.array([float(getattr(params0, name)) for name in vary])

    def cost(xv):
        r = (_model_vector(params0, vary, xv, kinds, b, strain_scale)
             - values) * w
        return r, float(r @ r)

    r, c = cost(x)
    mu = 1e-3
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        jac = _jacobian(params0, vary, x, kinds, b, strain_scale) \
            * w[:, None]
        jtj = jac.T @ jac
        g = jac.T @ r
        stepped = False
        for _ in range(60):
            try:
                dx = np.linalg.solve(
                    jtj + mu * np.diag(np.maximum(np.diag(jtj), 1e-30)),
                    -g)
            except np.linalg.LinAlgError:
                mu *= 10.0
                continue
            try:
                r_new, c_new = cost(x + dx)
            except ValueError:      # step left the allowed range
                mu *= 10.0
                continue
            if c_new <= c:
                x, r, c_old, c = x + dx, r_new, c, c_new
                mu = max(mu / 10.0, 1e-15)
                stepped = True
                break
            mu *= 10.0
        if not stepped:
            converged = True        # no descent direction left
            break
        if abs(c_old - c) <= tol * (1.0 + c) and \
                float(np.max(np.abs(dx))) <= 1e-10 * (1.0 + float(np.max(np.abs(x)))):
            converged = True
            break
    if not converged:
        raise RuntimeError("fit did not converge; check the starting "
                           "parameters and that the observations vary "
                           "with the fitted parameters")

    jac = _jacobian(params0, vary, x, kinds, b, strain_scale) * w[:, None]
    jtj = jac.T @ jac
    ok, cond, cov, _ = _invert_information(jtj, vary)
    if not ok:
        raise ValueError(_SINGULAR_MSG)
    if sig is None:
        rss = c
        cov = cov * rss / (n - p)
        chi2, chi2_dof = None, None
    else:
        chi2, chi2_dof = c, n - p

    fitted = _replace(params0, vary, x)
    fitted = dataclasses.replace(
        fitted,
        reference=(f"fitted by vacspin.lab.fit_spin_parameters to {n} "
                   f"measured points (vary={','.join(vary)}); base set: "
                   f"{params0.reference}"))
    err = np.sqrt(np.diag(cov))
    resid = values - _model_vector(params0, vary, x, kinds, b,
                                   strain_scale)
    return SpinFit(params=fitted,
                   values={k: float(v) for k, v in zip(vary, x)},
                   sigma={k: float(s) for k, s in zip(vary, err)},
                   cov=cov, chi2=chi2, chi2_dof=chi2_dof, n_points=n,
                   condition_number=cond, n_iter=it, converged=True,
                   residuals_ghz=resid)


def propagate_uncertainty(fit, func, rel_step=1e-6):
    """Carry a fit's error bars to any quantity computed from it.

    fit : a `SpinFit` from `fit_spin_parameters`.
    func : a function of a `SpinParameters` returning a number or an
        array of numbers -- for example
        ``lambda p: vacspin.qubit_frequency(p, b)`` for the qubit
        frequency at a field you have not measured, or
        ``lambda p: vacspin.cyclicity(p, b)``.
    rel_step : step of the differences, relative to the parameter's
        value (or to its error bar when that is larger).

    Uses first-order ("delta method") propagation: with J the
    derivatives of `func` with respect to the fitted parameters (by
    central differences, one-sided at a parameter boundary such as
    ups = 0), the covariance of the result is J cov J^T. This is exact
    for a quantity that depends linearly on the fitted parameters and
    a good approximation when the error bars are small compared with
    the scale on which `func` curves; the tests compare it with direct
    sampling of the fit's Gaussian uncertainty.

    Returns dict(value, sigma, cov): `value` is func(fit.params)
    (a float, or an array for array output), `sigma` its 1-sigma
    uncertainty in the same shape, and `cov` the full covariance
    matrix (1 x 1 for a scalar).
    """
    if not isinstance(fit, SpinFit):
        raise TypeError("fit must be a SpinFit from fit_spin_parameters")
    if not (np.isfinite(rel_step) and 0.0 < rel_step < 1e-2):
        raise ValueError("rel_step must be in (0, 1e-2)")
    vary = tuple(fit.values)
    x = np.array([fit.values[k] for k in vary])
    base = fit.params

    def ev(xv):
        return np.atleast_1d(np.asarray(
            func(_replace(base, vary, xv)), dtype=float)).ravel()

    value = np.asarray(func(base), dtype=float)
    f0 = np.atleast_1d(value).ravel()
    if not np.all(np.isfinite(f0)):
        raise ValueError("func returned a non-finite value at the "
                         "fitted parameters")
    jac = np.empty((f0.size, x.size))
    sig = np.sqrt(np.maximum(np.diag(np.asarray(fit.cov, dtype=float)),
                             0.0))
    for j in range(x.size):
        # step relative to the value, or to the error bar when the
        # value is near zero (e.g. a parameter at ups = 0)
        h = rel_step * max(abs(x[j]), sig[j], 1e-3)
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        # one-sided only when the lower point is not a valid parameter
        # set (e.g. ups = 0 would step to negative strain); any error
        # raised by `func` itself is passed on to the caller
        try:
            _replace(base, vary, xm)
            lower_ok = True
        except ValueError:
            lower_ok = False
        if lower_ok:
            jac[:, j] = (ev(xp) - ev(xm)) / (2.0 * h)
        else:
            jac[:, j] = (ev(xp) - f0) / h
    if not np.all(np.isfinite(jac)):
        raise ValueError("func is not differentiable at the fitted "
                         "parameters (non-finite derivative)")
    cov = jac @ np.asarray(fit.cov) @ jac.T
    sigma = np.sqrt(np.maximum(np.diag(cov), 0.0))
    if value.ndim == 0:
        return {"value": float(value), "sigma": float(sigma[0]),
                "cov": cov}
    return {"value": value, "sigma": sigma.reshape(value.shape),
            "cov": cov}


def parameter_information(kinds, b_fields_t, params,
                          vary=("lam_g", "ups_g"), sigmas_ghz=None,
                          strain_scale=1.0):
    """Would this experiment determine these parameters? Ask first.

    Computes the information matrix J^T W J of the planned observations
    around `params` and, when it is invertible, the error bars the
    weighted fit would deliver. Without `sigmas_ghz` the answer is
    stated per unit measurement error (1 GHz): real error bars scale
    linearly with your sigma.

    Returns dict(fisher, identifiable, condition_number, sigma) where
    `sigma` maps each varied parameter to its predicted 1-sigma error
    bar, or is None when the design is not identifiable.
    """
    kinds, b, _, sig, vary = _check_inputs(
        kinds, b_fields_t, None, sigmas_ghz, vary)
    w = np.ones(len(kinds)) if sig is None else 1.0 / sig
    x = np.array([float(getattr(params, name)) for name in vary])
    jac = _jacobian(params, vary, x, kinds, b, strain_scale) * w[:, None]
    fisher = jac.T @ jac
    identifiable, cond, _, sigma = _invert_information(fisher, vary)
    if len(kinds) < len(vary):
        identifiable, sigma = False, None
    return {"fisher": fisher, "identifiable": identifiable,
            "condition_number": cond, "sigma": sigma}


def design_fields(kinds, b_fields_t, n_pick, params,
                  vary=("lam_g", "ups_g"), sigmas_ghz=None,
                  strain_scale=1.0):
    """Pick the most informative subset of candidate observations.

    Greedy D-optimal selection: from the candidate list, repeatedly add
    the observation that most increases the determinant of the
    information matrix (equivalently: shrinks the joint parameter
    uncertainty fastest). The greedy rule is transparent and each step
    can only add information, but it is a good-practice heuristic, not
    a proof of the globally best subset.

    Returns dict(indices, fisher, condition_number, sigma) with the
    chosen candidate indices in pick order and the same fields as
    `parameter_information` for the chosen set. Refuses when even the
    full candidate list cannot identify the varied parameters.
    """
    kinds, b, _, sig, vary = _check_inputs(
        kinds, b_fields_t, None, sigmas_ghz, vary)
    n, p = len(kinds), len(vary)
    n_pick = int(n_pick)
    if not p <= n_pick <= n:
        raise ValueError(f"n_pick must be between {p} (the number of "
                         f"varied parameters) and {n} (the number of "
                         "candidates)")
    info_all = parameter_information(kinds, b, params, vary, sigmas_ghz,
                                     strain_scale)
    if not info_all["identifiable"]:
        raise ValueError("even the full candidate list is not "
                         "identifiable: " + _SINGULAR_MSG)
    w = np.ones(n) if sig is None else 1.0 / sig
    x = np.array([float(getattr(params, name)) for name in vary])
    rows = _jacobian(params, vary, x, kinds, b, strain_scale) * w[:, None]
    # column-scaled (unit-free) greedy: scaling every column
    # identically multiplies every candidate determinant by the same
    # constant, so the choices are unchanged, while the tiny start-up
    # regularizer stays meaningful in every direction
    scale = np.sqrt(np.mean(rows * rows, axis=0))
    rs = rows / scale
    eps = 1e-12 * float(np.max(np.sum(rs * rs, axis=1)))
    fs = eps * np.eye(p)
    chosen = []
    for _ in range(n_pick):
        best_j, best_det = -1, -np.inf
        for j in range(n):
            if j in chosen:
                continue
            trial = fs + np.outer(rs[j], rs[j])
            det = float(np.linalg.slogdet(trial)[1])
            if det > best_det:
                best_j, best_det = j, det
        fs = fs + np.outer(rs[best_j], rs[best_j])
        chosen.append(best_j)
    fisher = (fs - eps * np.eye(p)) * np.outer(scale, scale)
    _, cond, _, sigma = _invert_information(fisher, vary)
    return {"indices": chosen, "fisher": fisher,
            "condition_number": cond, "sigma": sigma}


_HEADER = ("kind", "bx_t", "by_t", "bz_t", "value_ghz")
_HEADER_S = _HEADER + ("sigma_ghz",)


def save_observations_csv(path, kinds, b_fields_t, values_ghz,
                          sigmas_ghz=None):
    """Write an observation record; the exact inverse of
    `load_observations_csv` (values round-trip bit for bit)."""
    kinds, b, values, sig, _ = _check_inputs(
        kinds, b_fields_t, values_ghz, sigmas_ghz, ("lam_g",))
    header = _HEADER if sig is None else _HEADER_S
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        for i, k in enumerate(kinds):
            row = [k, repr(float(b[i, 0])), repr(float(b[i, 1])),
                   repr(float(b[i, 2])), repr(float(values[i]))]
            if sig is not None:
                row.append(repr(float(sig[i])))
            wr.writerow(row)


def load_observations_csv(path):
    """Read an observation record written by `save_observations_csv`.

    Returns (kinds, b_fields_t, values_ghz, sigmas_ghz) with
    sigmas_ghz None when the file has no sigma column. The header and
    every value are checked; a malformed file is refused, not guessed
    at. Since 0.5.0 the file is read as UTF-8 with an optional
    byte-order mark (spreadsheet programs often add one when they
    save "CSV UTF-8"), and completely empty lines are skipped.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [row for row in csv.reader(fh) if row]
    if not rows:
        raise ValueError("empty observation file")
    header = tuple(rows[0])
    if header not in (_HEADER, _HEADER_S):
        raise ValueError(f"observation header must be {_HEADER} or "
                         f"{_HEADER_S}; got {header}")
    has_sigma = header == _HEADER_S
    body = rows[1:]
    if not body:
        raise ValueError("observation file has no data rows")
    kinds, b, vals, sigs = [], [], [], []
    for row in body:
        if len(row) != len(header):
            raise ValueError(f"row {row!r} does not match the header")
        kinds.append(row[0])
        try:
            b.append([float(row[1]), float(row[2]), float(row[3])])
            vals.append(float(row[4]))
            if has_sigma:
                sigs.append(float(row[5]))
        except ValueError as exc:
            raise ValueError(f"non-numeric value in row {row!r}") \
                from exc
    kinds_out = list(kinds)
    b_out = np.array(b)
    v_out = np.array(vals)
    s_out = np.array(sigs) if has_sigma else None
    _check_inputs(kinds_out, b_out, v_out, s_out, ("lam_g",))
    return kinds_out, b_out, v_out, s_out


# ---------------------------------------------------------------------------
# Readout calibration from your own photon-count histograms (since 0.5.0)
# ---------------------------------------------------------------------------

_READOUT_VARY = ("eta", "lam", "noise_rate")
_MIN_SHOTS = 10
_POOL_MIN_EXPECTED = 5.0
_MAX_REL_SIGMA = 0.25


@dataclasses.dataclass
class ReadoutFit:
    """Result of `fit_readout_histograms`.

    values : all three parameters by name ("eta", "lam",
        "noise_rate"): the fitted values of the varied ones and the
        held values of the others.
    sigma : 1-sigma uncertainty of each VARIED parameter, by name.
    cov : covariance matrix of the varied parameters, in `vary` order.
    vary : the varied names, in order.
    log_likelihood : the maximized log-likelihood of the histograms,
        sum over k of (shots with k counts) log P(k); the constant
        multinomial factor is left out.
    chi2, chi2_dof, p_value : Pearson goodness of fit, after pooling
        neighboring count bins until each expects at least 5 shots
        (None when no degrees of freedom are left).
    n_bright_shots, n_dark_shots : shots in each histogram (0 when no
        dark histogram was given).
    condition_number : of the scale-free information matrix.
    n_iter : iterations used.
    """

    values: dict
    sigma: dict
    cov: np.ndarray
    vary: tuple
    log_likelihood: float
    chi2: Optional[float]
    chi2_dof: Optional[int]
    p_value: Optional[float]
    n_bright_shots: int
    n_dark_shots: int
    condition_number: float
    n_iter: int


def _convolve_positive(p, q, n):
    """First n entries of the convolution of two non-negative
    sequences, by direct summation over the non-zero part of q only.
    Every term is non-negative, so each entry keeps its relative
    accuracy, however small it is (an FFT keeps only absolute
    accuracy, about 1e-16, which is not enough for a likelihood)."""
    nz = np.nonzero(q[:n])[0]
    out = np.zeros(n)
    if nz.size == 0:
        return out
    lo, hi = int(nz[0]), int(nz[-1])
    out[lo:] = np.convolve(p[:n], q[lo:hi + 1])[:n - lo]
    return out


def _histogram_pmfs(eta, lam, noise_rate, gamma, tau, s, leak, kmax):
    """Bright and dark count pmfs (k = 0..kmax) of the readout model,
    the same distributions as `vacspin.readout.fidelity`, evaluated
    for a likelihood: every probability keeps its relative accuracy,
    including far in the tails. The flip-terminated bright signal is

        P(K = k) = gp T_k + e^{-gp tau} Po(k; c tau),
        T_k = int_0^tau e^{-gp t} Po(k; c t) dt
            = (1/a) (c/a)^k P[Po(a tau) >= k + 1],  a = gp + c,

    with c = eta R and gp = R / (lam + 1), evaluated in log space by
    `_exp_poisson_integrals` (the readout module's recurrence is exact
    in absolute terms only). The background is then convolved by
    direct summation. The tests check these pmfs against the readout
    module's (to 1e-12) and against log-space quadrature in the
    tails."""
    r = 0.5 * gamma * s / (1.0 + s)
    gp = r / (1.0 + lam)
    c = eta * r
    signal = gp * _exp_poisson_integrals(gp, c, tau, kmax) \
        + math.exp(-gp * tau) * poisson_pmf(c * tau, kmax)
    pb = _convolve_positive(signal, poisson_pmf(noise_rate * tau, kmax),
                            kmax + 1)
    n_dark = (noise_rate + eta * leak * r) * tau
    r_sw = leak * r * lam / (lam + 1.0)
    if r_sw > 0.0:
        pd = _convolve_positive(_switch_on_pmf(c, gp, r_sw, tau, kmax),
                                poisson_pmf(n_dark, kmax), kmax + 1)
    else:
        pd = poisson_pmf(n_dark, kmax)
    return pb, pd


def _as_histogram(h, name):
    """A 1-D array of non-negative whole numbers of shots."""
    try:
        a = np.asarray(h, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a 1-D array of shot numbers "
                         "(entry k = number of shots with k counts)") \
            from None
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a 1-D array of shot numbers "
                         "(entry k = number of shots with k counts); "
                         "from per-shot counts use numpy.bincount")
    if not np.all(np.isfinite(a)) or np.any(a < 0) \
            or np.any(a != np.round(a)):
        raise ValueError(f"{name} entries must be non-negative whole "
                         "numbers of shots")
    if a.sum() < _MIN_SHOTS:
        raise ValueError(f"{name} holds {int(a.sum())} shots; at least "
                         f"{_MIN_SHOTS} are needed (the error bars are "
                         "large-sample error bars)")
    return a


def _pooled_pearson(hist, pmf):
    """Pearson chi2 of one histogram against a pmf covering every count
    (the tail beyond the histogram is one more bin). Neighboring bins
    are pooled, from low counts upward, until each expects at least 5
    shots. Returns (chi2, number of pooled bins)."""
    n = float(hist.sum())
    k = hist.size
    obs = np.concatenate((hist, [0.0]))
    tail = max(1.0 - float(np.sum(pmf[:k])), 0.0)
    exp = n * np.concatenate((pmf[:k], [tail]))
    bins_o, bins_e = [], []
    acc_o = acc_e = 0.0
    for o, e in zip(obs, exp):
        acc_o += o
        acc_e += e
        if acc_e >= _POOL_MIN_EXPECTED:
            bins_o.append(acc_o)
            bins_e.append(acc_e)
            acc_o = acc_e = 0.0
    if bins_e:
        bins_o[-1] += acc_o
        bins_e[-1] += acc_e
    else:
        bins_o, bins_e = [acc_o], [acc_e]
    o = np.array(bins_o)
    e = np.array(bins_e)
    return float(np.sum((o - e) ** 2 / e)), len(bins_e)


def _moment_start(hb, hd, r, tau, eta, lam, noise_rate, vary):
    """Starting values from the histograms' means and variances.

    Background: the dark mean over tau. Bright signal (background
    removed): with D = min(T_flip, tau) the bright time and x = Gp tau,
    mean = c E[D] and variance - mean = c^2 Var(D), where
    E[D] = tau (1 - e^{-x}) / x and E[D^2] = 2 tau^2 (1 - e^{-x}(1 + x)) / x^2.
    So (variance - mean) / mean^2 = E[D^2] / E[D]^2 - 1, which rises
    from 0 (no flips) to 1 (flips always end the record) and fixes x;
    then c = mean / E[D]. Only a starting point for the likelihood fit:
    the moments are noisy, so values are clipped into range."""
    def stats(h):
        k = np.arange(h.size, dtype=float)
        n = h.sum()
        m = float(np.dot(k, h) / n)
        v = float(np.dot((k - m) ** 2, h) / max(n - 1.0, 1.0))
        return m, v

    out = {"eta": eta, "lam": lam, "noise_rate": noise_rate}
    if out["noise_rate"] is None:
        if "noise_rate" in vary and hd is not None:
            md, _ = stats(hd)
            if md <= 0.0:
                raise ValueError("the dark histogram has no counts, so a "
                                 "background cannot be fitted; hold "
                                 "noise_rate fixed (for example at 0)")
            out["noise_rate"] = md / tau
        elif "noise_rate" in vary:
            raise ValueError("to fit noise_rate without a dark histogram, "
                             "give a positive starting noise_rate (1/s)")
        else:
            out["noise_rate"] = 0.0
    nb = out["noise_rate"] * tau
    mb, vb = stats(hb)
    m_s, v_s = mb - nb, vb - nb
    if out["eta"] is None or out["lam"] is None:
        if m_s <= 0.0:
            raise ValueError("the bright histogram's mean count "
                             f"({mb:.4g}) is not above the background "
                             f"({nb:.4g}); there is no signal to fit")

        def ed(x):                                   # E[D] / tau
            return -math.expm1(-x) / x

        def ratio(x):                                # Var(D) / E[D]^2
            ed2 = 2.0 * (-math.expm1(-x) - x * math.exp(-x)) / (x * x)
            return ed2 / ed(x) ** 2 - 1.0

        if out["lam"] is None:
            rho = (v_s - m_s) / (m_s * m_s)
            lo, hi = 1e-4, 1e3
            if rho <= ratio(lo):
                x = lo
            elif rho >= ratio(hi):
                x = hi
            else:
                for _ in range(100):
                    mid = math.sqrt(lo * hi)
                    lo, hi = (mid, hi) if ratio(mid) < rho else (lo, mid)
                x = math.sqrt(lo * hi)
            out["lam"] = max(r * tau / x - 1.0, 1e-3)
        if out["eta"] is None:
            x = r * tau / (1.0 + out["lam"])
            out["eta"] = min(max(m_s / (r * tau * ed(x)), 1e-9), 0.99)
    return out


def fit_readout_histograms(bright_hist, gamma: float, tau: float,
                           s: float = 1.0, dark_hist=None,
                           eta: Optional[float] = None,
                           lam: Optional[float] = None,
                           noise_rate: Optional[float] = None,
                           leak: float = 0.0, vary=("eta", "lam"),
                           max_iter: int = 200) -> ReadoutFit:
    """Fit the readout model to your measured photon-count histograms.

    In a single-shot readout experiment you prepare the bright spin
    state many times, count the photons detected in a window `tau`
    each time, and histogram the counts; often you do the same for the
    dark state. This function finds the detection efficiency `eta`,
    the cyclicity `lam` and the background `noise_rate` (any subset)
    that make those histograms most probable under the same count
    distributions `fidelity` uses (maximum likelihood), with error
    bars and a goodness-of-fit test.

    bright_hist : histogram of the bright-state shots: entry k is the
        number of shots with exactly k detected photons (from per-shot
        counts: ``numpy.bincount(counts)``).
    gamma : optical decay rate (1/s) and s : saturation parameter of the
        drive, as in `fidelity`; tau : the counting window (s). They are
        held fixed: `eta` and the drive enter the counts only through
        the detected rate eta R, R = (gamma/2) s / (1 + s), so an error
        in gamma or s moves the fitted eta, not the fit quality.
    dark_hist : optional histogram of the dark-state shots, fitted
        jointly; it is what pins down `noise_rate` well.
    eta, lam, noise_rate : values of the fixed parameters, and
        optional starting values of the varied ones. A varied parameter
        left at None starts from the histograms' means and variances
        (method of moments; `noise_rate` from the dark histogram), which
        is usually close to the best fit; the likelihood can have more
        than one maximum, so a poor start can end on a worse one (a
        tiny `p_value` then shows it). A fixed `eta` or `lam` must be
        given; a fixed `noise_rate` left at None means no background.
    leak : the dark spin's off-resonant scattering ratio, held fixed
        (default 0; see `fidelity`).
    vary : which of "eta", "lam", "noise_rate" to fit.

    Method: Fisher scoring on the exact log-likelihood (the expected
    information matrix with central-difference derivatives of the
    count probabilities). The error bars are the inverse of the
    expected information at the optimum: the usual large-sample
    (Cramer-Rao) error bars, which the tests check against closed
    forms and against simulated experiments.

    Refuses, with the reason: histograms that are not 1-D arrays of
    whole numbers of shots, or that hold fewer than 10 shots; an
    unknown or repeated name in `vary`; a fixed `eta` or `lam` without
    a value; a bright mean count not above the background; a varied
    `noise_rate` with a dark histogram that has no counts, or with
    neither a dark histogram nor a starting value; starting values
    outside the model's range or that give zero probability to an
    observed count;
    parameters the data cannot tell apart (for example `lam` when spin
    flips are rare within the window, so the bright counts are Poisson
    whatever `lam` is); a fitted parameter whose error bar exceeds a
    quarter of its value (the likelihood is then too lopsided for
    large-sample error bars to be reliable); a best fit at the
    boundary eta = 1 (the counts exceed what gamma, s and tau allow);
    and a fit that does not converge (`RuntimeError`).
    """
    hb = _as_histogram(bright_hist, "bright_hist")
    hd = None if dark_hist is None else _as_histogram(dark_hist,
                                                      "dark_hist")
    vary = tuple(vary)
    if len(vary) == 0:
        raise ValueError("vary must name at least one parameter")
    for name in vary:
        if name not in _READOUT_VARY:
            raise ValueError(f"cannot vary {name!r}; choose from "
                             f"{_READOUT_VARY}")
    if len(set(vary)) != len(vary):
        raise ValueError("vary contains a repeated name")
    for name, v in (("eta", eta), ("lam", lam)):
        if v is None and name not in vary:
            raise ValueError(f"{name} is held fixed, so give its value")
    _check_common(0.5, 1.0, gamma, tau, s)
    _check_background(leak, 0.0)
    r = 0.5 * gamma * s / (1.0 + s)
    full = _moment_start(hb, hd, r, tau, eta, lam, noise_rate, vary)
    full = {k: float(v) for k, v in full.items()}
    _check_common(full["eta"], full["lam"], gamma, tau, s)
    _check_background(leak, full["noise_rate"])
    if full["eta"] <= 0.0:
        raise ValueError("eta must start above 0")
    if "noise_rate" in vary and full["noise_rate"] <= 0.0:
        raise ValueError("a varied noise_rate must start above 0 (1/s); "
                         "to fit without background, hold it fixed at 0")
    k_obs = max(hb.size, 0 if hd is None else hd.size) - 1

    def unpack(x):
        d = dict(full)
        d.update({k: float(v) for k, v in zip(vary, x)})
        return d

    def valid(x):
        d = unpack(x)
        return (0.0 < d["eta"] <= 1.0 and d["lam"] > 0.0
                and np.isfinite(d["lam"]) and d["noise_rate"] >= 0.0
                and (d["noise_rate"] > 0.0 or "noise_rate" not in vary)
                and np.isfinite(d["noise_rate"]))

    def pmfs(x):
        """Both pmfs to a count beyond which less than 1e-9 remains."""
        d = unpack(x)
        gp = r / (1.0 + d["lam"])
        m = (d["eta"] * (d["lam"] + 1.0) * (-math.expm1(-gp * tau))
             + 2.0 * d["noise_rate"] * tau
             + d["eta"] * r * tau * (1.0 + leak) * (leak > 0.0))
        kmax = int(max(k_obs, m + 12.0 * math.sqrt(m + 1.0) + 60))
        while True:
            pb, pd = _histogram_pmfs(d["eta"], d["lam"], d["noise_rate"],
                                     gamma, tau, s, leak, kmax)
            if 1.0 - pb.sum() < 1e-9 and 1.0 - pd.sum() < 1e-9:
                return pb, pd
            if kmax >= (1 << 18):
                raise ValueError("count distributions too wide to "
                                 "evaluate (more than 2^18 counts)")
            kmax = min(2 * kmax, 1 << 18)

    def loglik(pb, pd):
        total = 0.0
        for h, p in ((hb, pb), (hd, pd)):
            if h is None:
                continue
            nz = h > 0
            pk = p[:h.size][nz]
            if np.any(pk <= 0.0):
                return -np.inf
            total += float(np.sum(h[nz] * np.log(pk)))
        return total

    def score_info(x):
        """Score vector and expected information matrix at x."""
        pb, pd = pmfs(x)
        size_b, size_d = pb.size, pd.size
        dpb, dpd = [], []
        for j in range(len(x)):
            hj = 1e-5 * abs(x[j])
            xp, xm = x.copy(), x.copy()
            xp[j] += hj
            xm[j] -= hj
            if valid(xp):
                denom = 2.0 * hj
            else:                       # eta at 1: one-sided step
                xp, denom = x.copy(), hj
            pbp, pdp = pmfs(xp)
            pbm, pdm = pmfs(xm)
            n_b = min(pbp.size, pbm.size, size_b)
            n_d = min(pdp.size, pdm.size, size_d)
            dpb.append((pbp[:n_b] - pbm[:n_b]) / denom)
            dpd.append((pdp[:n_d] - pdm[:n_d]) / denom)
        p_dim = len(x)
        u = np.zeros(p_dim)
        info = np.zeros((p_dim, p_dim))
        for h, p, dp in ((hb, pb, dpb), (hd, pd, dpd)):
            if h is None:
                continue
            n_k = min(len(dpj) for dpj in dp)
            g = np.array([dpj[:n_k] for dpj in dp])        # (p, n_k)
            pk = p[:n_k]
            # counts with probability below 1e-10 carry a negligible
            # share of the information, and there the pmf's round-off
            # (about 1e-16 absolute) would dominate its derivative
            ok = pk > 1e-10
            w = np.zeros(n_k)
            w[ok] = 1.0 / pk[ok]
            info += float(h.sum()) * (g * w) @ g.T
            # the score uses every observed count, as the likelihood does
            m_k = min(h.size, n_k)
            seen = (h[:m_k] > 0) & (pk[:m_k] > 0.0)
            u += g[:, :m_k][:, seen] @ (h[:m_k][seen] / pk[:m_k][seen])
        return u, info, (pb, pd)

    x = np.array([full[k] for k in vary], dtype=float)
    pb, pd = pmfs(x)
    ll = loglik(pb, pd)
    if not np.isfinite(ll):
        gp0 = r / (1.0 + full["lam"])
        m_model = (full["eta"] * (full["lam"] + 1.0)
                   * (-math.expm1(-gp0 * tau)) + full["noise_rate"] * tau)
        m_data = float(np.dot(np.arange(hb.size), hb) / hb.sum())
        raise ValueError(
            "the starting values give zero probability to an observed "
            "count; start closer. The model's mean bright count "
            "eta (lam + 1)(1 - exp(-R tau / (lam + 1))) + noise_rate tau "
            f"is {m_model:.4g} at the start, the measured mean is "
            f"{m_data:.4g}")
    u, info, _ = score_info(x)
    singular_msg = ("these histograms cannot determine "
                    f"{', '.join(vary)} together (singular information "
                    "matrix). With rare spin flips in the window the "
                    "bright counts do not depend on lam: hold lam fixed, "
                    "or use a longer window")
    ok, _, _, _ = _invert_information(info, vary)
    if not ok:
        raise ValueError(singular_msg)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        d = np.sqrt(np.diag(info))
        step = None
        if np.all(np.isfinite(d)) and np.all(d > 0.0):
            try:
                step = np.linalg.solve(info / np.outer(d, d), u / d) / d
            except np.linalg.LinAlgError:
                step = None
        if step is None or not np.all(np.isfinite(step)):
            raise ValueError(singular_msg)
        alpha, moved = 1.0, False
        for _ in range(60):
            x_new = x + alpha * step
            if valid(x_new):
                pb_n, pd_n = pmfs(x_new)
                ll_new = loglik(pb_n, pd_n)
                if ll_new >= ll - 1e-12 * abs(ll):
                    moved = True
                    break
            alpha *= 0.5
        if not moved:
            converged = True            # no ascent direction left
            break
        dx = x_new - x
        gain = ll_new - ll
        x, ll = x_new, ll_new
        u, info, _ = score_info(x)
        # converged when the step is negligible, or when both the step
        # and the likelihood gain are at the round-off level
        if np.all(np.abs(dx) <= 1e-9 * np.abs(x)) or (
                gain <= 1e-12 * abs(ll)
                and np.all(np.abs(dx) <= 1e-6 * np.abs(x))):
            converged = True
            break
    if not converged:
        raise RuntimeError("readout fit did not converge; check gamma, s "
                           "and tau, start closer, or hold a poorly "
                           "determined parameter fixed")
    fitted = unpack(x)
    if fitted["eta"] > 1.0 - 1e-6:
        raise ValueError("the best fit sits at the boundary eta = 1: the "
                         "bright counts exceed what gamma, s and tau "
                         "allow. Check those inputs")
    ok, cond, cov, sig = _invert_information(info, vary)
    if not ok:
        raise ValueError(singular_msg)
    weak = [k for k in vary if sig[k] > _MAX_REL_SIGMA * abs(fitted[k])]
    if weak:
        found = ", ".join(f"{k} = {fitted[k]:.4g} +/- {sig[k]:.2g}"
                          for k in weak)
        raise ValueError(
            f"poorly determined: {found}. When an error bar exceeds a "
            "quarter of the value, the likelihood is too lopsided for "
            "large-sample error bars to be reliable. Hold that "
            "parameter fixed, or collect more shots (for lam: use a "
            "window with more spin flips, tau near (lam + 1) / R)")
    # goodness of fit
    pb, pd = pmfs(x)
    chi2, bins = _pooled_pearson(hb, pb)
    dof = bins - 1
    if hd is not None:
        c2, bd = _pooled_pearson(hd, pd)
        chi2 += c2
        dof += bd - 1
    dof -= len(vary)
    if dof >= 1:
        p_val = _chi2_sf(chi2, dof)
    else:
        chi2, dof, p_val = None, None, None
    return ReadoutFit(values=fitted, sigma=sig, cov=cov, vary=vary,
                      log_likelihood=float(ll), chi2=chi2, chi2_dof=dof,
                      p_value=p_val, n_bright_shots=int(hb.sum()),
                      n_dark_shots=0 if hd is None else int(hd.sum()),
                      condition_number=cond, n_iter=it)
