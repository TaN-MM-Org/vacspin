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
"""
from __future__ import annotations

import csv
import dataclasses

import numpy as np

from .hamiltonian import solve
from .params import SpinParameters

__all__ = ["SpinFit", "fit_spin_parameters", "parameter_information",
           "design_fields", "save_observations_csv",
           "load_observations_csv"]

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
    eg, _, ee, _ = solve(params, b_lab, strain_scale)
    if kind == "qubit":
        return float(eg[1] - eg[0])
    if kind == "orbital_g":
        return float(eg[2] - eg[0])
    if kind == "orbital_e":
        return float(ee[2] - ee[0])
    raise ValueError(f"unknown observable kind {kind!r}; "
                     f"choose from {_KINDS}")


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
    return SpinFit(params=fitted,
                   values={k: float(v) for k, v in zip(vary, x)},
                   sigma={k: float(s) for k, s in zip(vary, err)},
                   cov=cov, chi2=chi2, chi2_dof=chi2_dof, n_points=n,
                   condition_number=cond, n_iter=it, converged=True)


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
    at.
    """
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
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
