"""Tutorial: from measured spin frequencies to a cavity readout plan.

A lab has a tin-vacancy (SnV-) center whose strain differs from the
published device. It measures the qubit frequency and the ground
orbital splitting at eight field angles, and wants to know (1) the
spin-orbit and strain constants of its own center, with error bars,
(2) the cyclicity and qubit frequency at its readout field, and
(3) what that predicts for a fast readout through a Q_L = 500 cavity:
the best window, its fidelity, and how much the calibration
uncertainty moves that fidelity. The "measured" frequencies are
simulated with a fixed seed from a center whose strain is 10 % above
the cited value, with 0.05 GHz noise; replace `measured` with your own
values. The cavity, detector and background numbers are those of
README example 2 and the test suite's design point, illustrative only.

Run it with ``python examples/tutorial_spin_to_readout.py`` (under a
second).
It printed, with vacspin 0.7.0:

    fitted ground state of this center:
      lam_g = 831.5 +/- 5.6 GHz   (simulated 830.1)
      ups_g = 194.0 +/- 6.0 GHz   (simulated 195.4)
      chi2 = 3.87 for 6 degrees of freedom, p = 0.69
    at the readout field: cyclicity 1827 +/- 143, qubit frequency 5.910 +/- 0.005 GHz
    cavity: F_C = 19.29, eta = 0.614, cavity-boosted cyclicity 1827
    best window 24.8 ns (threshold 2 photons): fidelity 0.99075 +/- 0.00008
    with the simulated center: fidelity 0.99073 at the same window
"""
import dataclasses

import numpy as np

import vacspin as vs

p0 = vs.snv_rosenthal2023()                 # cited starting point
truth = dataclasses.replace(p0, ups_g=1.1 * p0.ups_g)

# 1. The "measured" data: qubit frequency and ground orbital splitting
#    at 150 mT, eight angles in the lab x-z plane (simulated here).
kinds = ["qubit", "orbital_g"] * 4
angles = (15.0, 40.0, 65.0, 90.0, 115.0, 140.0, 165.0, 190.0)
fields = np.array([vs.field_on_circle(z, 0.150) for z in angles])
sigma = 0.05                                 # GHz, per point
rng = np.random.default_rng(7)
exact = []
for kind, b in zip(kinds, fields):
    eg = vs.solve(truth, b)[0]
    exact.append(eg[1] - eg[0] if kind == "qubit" else eg[2] - eg[0])
measured = np.array(exact) + sigma * rng.standard_normal(len(kinds))

# 2. Fit the two ground-state constants of this center.
fit = vs.fit_spin_parameters(kinds, fields, measured, p0,
                             vary=("lam_g", "ups_g"), sigmas_ghz=sigma)
print("fitted ground state of this center:")
for name in ("lam_g", "ups_g"):
    print(f"  {name} = {fit.values[name]:.1f} +/- {fit.sigma[name]:.1f} GHz"
          f"   (simulated {getattr(truth, name):.1f})")
print(f"  chi2 = {fit.chi2:.2f} for {fit.chi2_dof} degrees of freedom, "
      f"p = {fit.p_value:.2f}")

# 3. The readout field: 180 mT, 9 degrees from the center's axis.
th, ph = p0.theta_rad, p0.phi_rad
axis = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                 np.cos(th)])
side = np.cross(axis, [0.0, 0.0, 1.0])
side /= np.linalg.norm(side)
b_read = 0.180 * (np.cos(np.deg2rad(9.0)) * axis
                  + np.sin(np.deg2rad(9.0)) * side)
spin = vs.propagate_uncertainty(
    fit, lambda p: [vs.cyclicity(p, b_read), vs.qubit_frequency(p, b_read)])
(lam0, f_q), (dlam0, df_q) = spin["value"], spin["sigma"]
print(f"at the readout field: cyclicity {lam0:.0f} +/- {dlam0:.0f}, "
      f"qubit frequency {f_q:.3f} +/- {df_q:.3f} GHz")

# 4. The cavity readout this predicts.
s, noise = 2.0, 1e3                          # drive, background (1/s)


def readout(p, tau=None):
    """Cavity budget, dark-spin leak and fidelity for spin set p."""
    lam_p = vs.cyclicity(p, b_read)
    fq_hz = vs.qubit_frequency(p, b_read) * 1e9
    ci = vs.CavityInterface(
        q_loaded=500, v_rel=0.68, eta_wg=0.9996, budget=vs.snv_emission(),
        lambda0=lam_p, omega_q_hz=fq_hz, xi_pol=vs.xi_pol_overlap(),
        xi_pos=0.5, t_link=0.99, eta_chip=0.8, eta_det=0.9).require_valid()
    # leak: off-resonant over resonant scattering of the dark spin, the
    # ratio of the scattering rates of Rosenthal et al. Eq. A8 at a
    # detuning of one qubit frequency
    gamma_hz = ci.gamma_cav / (2.0 * np.pi)
    leak = (vs.polarization_rate(ci.gamma_cav, 0.0, s, fq_hz / gamma_hz)
            / vs.polarization_rate(ci.gamma_cav, 0.0, s))
    kw = dict(eta=ci.eta, lam=ci.lambda_cav, gamma=ci.gamma_cav, s=s,
              leak=leak, noise_rate=noise)
    if tau is None:
        return ci, vs.optimal_window(tau_max=1e-6, **kw)
    return vs.fidelity(tau=tau, **kw)["fidelity"]


ci, out = readout(fit.params)
# with kappa near 1 THz and a qubit frequency of a few GHz the cavity
# speeds up the spin-flipping line as much as the bright one, so the
# cavity-boosted cyclicity equals the bare one here
print(f"cavity: F_C = {ci.f_c:.2f}, eta = {ci.eta:.3f}, "
      f"cavity-boosted cyclicity {ci.lambda_cav:.0f}")
q = vs.propagate_uncertainty(fit, lambda p: readout(p, out["tau"]))
print(f"best window {1e9 * out['tau']:.1f} ns (threshold "
      f"{out['threshold']} photons): fidelity {out['fidelity']:.5f} "
      f"+/- {q['sigma']:.5f}")
print(f"with the simulated center: fidelity "
      f"{readout(truth, out['tau']):.5f} at the same window")
