"""Tutorial: from photon-count histograms to the best readout window.

A lab calibrates its single-shot readout: it prepares the spin bright
5000 times and dark 5000 times, counts the photons in a 20 us window
each time, and wants to know (1) the detection efficiency, cyclicity
and background of the setup, with error bars, (2) the readout window
that gives the highest fidelity, and (3) how well that fidelity is
known. Here the "measured" counts are simulated with a fixed seed from
the counting process itself (eta = 0.2 %, cyclicity 2244, background
4000 counts/s, the confocal SnV- readout of README example 3); replace
`bright` and `dark` with your own per-shot counts.

Run it with ``python examples/tutorial_readout_window.py`` (under a
second).
It printed, with vacspin 0.7.0:

    calibration from 5000 bright and 5000 dark shots:
      eta        = 0.1968 +/- 0.0033 %   (simulated 0.2000 %)
      lam        = 2398 +/- 112   (simulated 2244)
      noise_rate = 3752 +/- 193 counts/s   (simulated 4000)
      goodness of fit: chi2 = 8.3 for 10 degrees of freedom, p = 0.60
    best window: 17.34 +/- 0.35 us (closed form for threshold 1: 17.34 us)
    fidelity there: 0.8807 +/- 0.0025 (threshold 1 photon)
    with the simulated values: 16.76 us, fidelity 0.8763
"""
import math

import numpy as np

import vacspin as vs

gamma = vs.snv_emission().gamma0          # optical decay rate, 1/s
tau, s = 20e-6, 10.0                      # counting window (s), drive
n_shots = 5000

# 1. The "measured" data: photon counts of each shot (simulated here).
eta_true, lam_true, noise_true = 0.002, 2244.0, 4000.0
rng = np.random.default_rng(2026)
r = 0.5 * gamma * s / (1.0 + s)                        # scattering rate
t_flip = rng.exponential((1.0 + lam_true) / r, n_shots)  # spin flips
bright = (rng.poisson(eta_true * r * np.minimum(t_flip, tau))
          + rng.poisson(noise_true * tau, n_shots))
dark = rng.poisson(noise_true * tau, n_shots)

# 2. Calibrate: maximum likelihood on both histograms at once.
fit = vs.fit_readout_histograms(
    np.bincount(bright), gamma, tau, s, dark_hist=np.bincount(dark),
    vary=("eta", "lam", "noise_rate"))
v, e = fit.values, fit.sigma
print(f"calibration from {fit.n_bright_shots} bright and "
      f"{fit.n_dark_shots} dark shots:")
print(f"  eta        = {100 * v['eta']:.4f} +/- {100 * e['eta']:.4f} %"
      f"   (simulated {100 * eta_true:.4f} %)")
print(f"  lam        = {v['lam']:.0f} +/- {e['lam']:.0f}"
      f"   (simulated {lam_true:.0f})")
print(f"  noise_rate = {v['noise_rate']:.0f} +/- {e['noise_rate']:.0f} "
      f"counts/s   (simulated {noise_true:.0f})")
print(f"  goodness of fit: chi2 = {fit.chi2:.1f} for {fit.chi2_dof} "
      f"degrees of freedom, p = {fit.p_value:.2f}")


# 3. The best window for the calibrated setup, and its error bar.
def best(p):
    return vs.optimal_window(p["eta"], p["lam"], gamma, s=s,
                             noise_rate=p["noise_rate"], tau_max=1e-3)


out = best(v)
# the window's error bar: a larger relative step, because the peak
# position is found only to about 1e-8 relative
win = vs.propagate_uncertainty(fit, lambda p: best(p)["tau"],
                               rel_step=1e-4)
# check: with threshold 1 the peak has the closed form
# tau* = ln(1 + a / N) / a, a = R / (lam + 1) + eta R
a = r / (1.0 + v["lam"]) + v["eta"] * r
tau_closed = math.log(1.0 + a / v["noise_rate"]) / a
print(f"best window: {1e6 * out['tau']:.2f} +/- {1e6 * win['sigma']:.2f} us"
      f" (closed form for threshold 1: {1e6 * tau_closed:.2f} us)")


def fid(p):
    return vs.fidelity(p["eta"], p["lam"], gamma, out["tau"], s,
                       noise_rate=p["noise_rate"])["fidelity"]


q = vs.propagate_uncertainty(fit, fid)
print(f"fidelity there: {q['value']:.4f} +/- {q['sigma']:.4f} "
      f"(threshold {out['threshold']} photon)")
ref = best({"eta": eta_true, "lam": lam_true, "noise_rate": noise_true})
print(f"with the simulated values: {1e6 * ref['tau']:.2f} us, "
      f"fidelity {ref['fidelity']:.4f}")
