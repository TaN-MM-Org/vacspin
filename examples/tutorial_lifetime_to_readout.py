"""Tutorial: from two lifetime histograms to the cavity budget and readout.

A lab has a tin-vacancy (SnV-) center in a Q_L = 500 cavity. It
measures the excited-state lifetime twice by time-correlated photon
counting: once with the cavity tuned far away and once on resonance.
It wants to know (1) both lifetimes, with error bars, (2) what they say
about the cavity: the Purcell factor, the cooperativity, the share of
photons that go into the cavity mode and the coupling rate, and whether
these match the design, and (3) what that predicts for single-shot
readout: the best window, its threshold, the fidelity with an error
bar, and how the errors split between the two spin states.

The "measured" histograms are simulated with a fixed seed: 30000 decays
(Poisson) per histogram with the lifetime of the cited SnV- budget
(4.5 ns) and the lifetime of the README example 2 design (0.620 ns),
excited at 2 ns with a 40 ps timing jitter, plus 3000 background
counts spread over the 50 ns delay axis, in 16 ps bins. The fits start
at 2.5 ns, after the jitter has died away. Replace `h_ref` and `h_cav`
with your own histograms. The detection chain and drive of the readout
are those of README example 2; the 1000 counts/s background and the
dark-spin leak are those of examples/tutorial_spin_to_readout.py. All
are illustrative only.

Run it with ``python examples/tutorial_lifetime_to_readout.py`` (under a
second).
It printed, with vacspin 0.8.0:

    lifetime, cavity detuned:      4.5466 +/- 0.0357 ns (simulated 4.5000), p = 0.94
    lifetime, cavity on resonance: 0.6107 +/- 0.0058 ns (simulated 0.6202), p = 0.41
    lifetime shortened 7.44 +/- 0.09 times: cooperativity 6.44 +/- 0.09, beta 0.8657 +/- 0.0017
    Purcell factor F_C = 19.84 +/- 0.27 (design 19.29), g = 7.39 +/- 0.08 GHz (design 7.32)
    readout: eta = 0.6168, best window 20.9 ns, threshold 2 photons
      fidelity 0.98014 +/- 0.00027: bright error 0.00838, dark error 0.03134
      at threshold 1 instead: fidelity 0.96795, bright error 0.00158, dark error 0.06251
    with the simulated lifetimes: fidelity 0.98051
"""
import math

import numpy as np

import vacspin as vs

budget = vs.snv_emission()                      # cited SnV- budget
spin = vs.snv_rosenthal2023()
f_q = vs.qubit_frequency(spin, vs.field_on_circle(147.0, 0.125))  # GHz
design = vs.CavityInterface(
    q_loaded=500, v_rel=0.68, eta_wg=0.9996, budget=budget,
    lambda0=2244.0, omega_q_hz=f_q * 1e9, xi_pol=vs.xi_pol_overlap(),
    xi_pos=0.5, t_link=0.99, eta_chip=0.8, eta_det=0.9).require_valid()

# 1. The "measured" histograms (simulated here).
rng = np.random.default_rng(18)
edges = np.arange(0.0, 50e-9 + 8e-12, 16e-12)   # delay axis, 16 ps bins
t_exc, jitter = 2e-9, 40e-12


def histogram(tau):
    n = rng.poisson(30000)
    t = t_exc + rng.exponential(tau, n) + jitter * rng.standard_normal(n)
    bg = rng.uniform(0.0, edges[-1], rng.poisson(3000))
    return np.histogram(np.concatenate([t, bg]), edges)[0]


h_ref = histogram(budget.tau0_s)                # cavity tuned away
h_cav = histogram(design.tau_cav_s)             # cavity on resonance

# 2. Fit both lifetimes from 2.5 ns on.
fit_ref = vs.fit_lifetime_histogram(h_ref, edges, t_start=2.5e-9)
fit_cav = vs.fit_lifetime_histogram(h_cav, edges, t_start=2.5e-9)
for name, fit, true in (("detuned", fit_ref, budget.tau0_s),
                        ("on resonance", fit_cav, design.tau_cav_s)):
    print(f"lifetime, cavity {name + ':':14s}{1e9 * fit.tau_s:.4f} +/- "
          f"{1e9 * fit.sigma_tau_s:.4f} ns (simulated "
          f"{1e9 * true:.4f}), p = {fit.p_value:.2f}")

# 3. The cavity budget these lifetimes measure. kappa = f_cav / Q_L,
#    with Q_L = 500 +/- 10 from a transmission spectrum (illustrative).
kappa = design.f_cav_hz / 500.0
cav = vs.purcell_from_lifetimes(fit_ref, fit_cav, budget=budget,
                                kappa_hz=kappa,
                                sigma_kappa_hz=kappa * 10.0 / 500.0)
v, e = cav["values"], cav["sigma"]
print(f"lifetime shortened {v['zeta']:.2f} +/- {e['zeta']:.2f} times: "
      f"cooperativity {v['cooperativity']:.2f} +/- "
      f"{e['cooperativity']:.2f}, beta {v['beta']:.4f} +/- "
      f"{e['beta']:.4f}")
print(f"Purcell factor F_C = {v['f_c']:.2f} +/- {e['f_c']:.2f} (design "
      f"{design.f_c:.2f}), g = {v['g_hz'] / 1e9:.2f} +/- "
      f"{e['g_hz'] / 1e9:.2f} GHz (design {design.g_hz / 1e9:.2f})")

# 4. The readout this predicts. The chain after the cavity is that of
#    README example 2. kappa (about 1 THz) is far wider than the qubit
#    frequency, so the cavity speeds up the spin-flipping line almost as
#    much as the bright one, and the cyclicity stays at its bare 2244
#    (the design's cavity-boosted value is 2244.1).
chain = 0.9996 * 0.99 * 0.8 * 0.9
s, noise, lam = 2.0, 1e3, 2244.0


def readout_inputs(tau_ref, tau_cav):
    """The `fidelity` inputs that two lifetimes give."""
    gamma = 1.0 / tau_cav
    eta = (1.0 - tau_cav / tau_ref) * chain     # beta times the chain
    # dark-spin leak: Eq. A8 of Rosenthal et al. at a detuning of one
    # qubit frequency (as in examples/tutorial_spin_to_readout.py)
    gamma_hz = gamma / (2.0 * math.pi)
    leak = (vs.polarization_rate(gamma, 0.0, s, f_q * 1e9 / gamma_hz)
            / vs.polarization_rate(gamma, 0.0, s))
    return dict(eta=eta, lam=lam, gamma=gamma, s=s, leak=leak,
                noise_rate=noise)


kw = readout_inputs(fit_ref.tau_s, fit_cav.tau_s)
best = vs.optimal_window(tau_max=1e-6, **kw)
win, thr = best["tau"], best["threshold"]


def fid(tau_ref, tau_cav):
    """Fidelity at this window and threshold."""
    return vs.fidelity(tau=win, threshold=thr,
                       **readout_inputs(tau_ref, tau_cav))["fidelity"]


# error bars from the two independent fits, at this window and threshold
q_ref = vs.propagate_uncertainty(fit_ref,
                                 lambda p: fid(p["tau_s"], fit_cav.tau_s))
q_cav = vs.propagate_uncertainty(fit_cav,
                                 lambda p: fid(fit_ref.tau_s, p["tau_s"]))
sig_f = math.hypot(q_ref["sigma"], q_cav["sigma"])
print(f"readout: eta = {kw['eta']:.4f}, best window {1e9 * win:.1f} ns, "
      f"threshold {thr} photons")
print(f"  fidelity {best['fidelity']:.5f} +/- {sig_f:.5f}: bright error "
      f"{best['error_bright']:.5f}, dark error {best['error_dark']:.5f}")
one = vs.fidelity(tau=win, threshold=1, **kw)
print(f"  at threshold 1 instead: fidelity {one['fidelity']:.5f}, bright "
      f"error {one['error_bright']:.5f}, dark error {one['error_dark']:.5f}")
truth = vs.fidelity(tau=win, threshold=thr,
                    **readout_inputs(budget.tau0_s, design.tau_cav_s))
print(f"with the simulated lifetimes: fidelity {truth['fidelity']:.5f}")
