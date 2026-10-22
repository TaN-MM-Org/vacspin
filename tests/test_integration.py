"""End to end: the package reproduces the design point of the ORIGINAL
SUBMISSION of the source study (Mahim, Rahman and Mohsin, Optics
Express (2026), submitted; the open pipeline at
Tanvir-Mahmud-Mahim/snv-tfln-cavity-interface, release v1.1.1) from
its stated inputs -- the overcoupled Q_L = 500 diamond nanocavity on
TFLN: F_C ~ 19, end-to-end efficiency ~ 0.65, single-shot fidelity
above 98% in a 100 ns window. The revised manuscript moved the design
point to Q_L = 870 (F_C = 28, efficiency 32 %) with a four-level
readout model; tests/test_revised_study.py checks that version. This
file stays as an end-to-end regression of the chain. The original
submission's efficiency counts the whole enhanced emission of the
coupled line as cavity photons (beta_form="transition"); the default
since 0.6.0 counts only the cavity-added part and gives 0.615. The chain runs spin model -> cavity ->
readout with no numbers passed by hand between stages."""
import numpy as np

from vacspin import (CavityInterface, cyclicity, fidelity,
                     field_on_circle, qubit_frequency, snv_emission,
                     snv_rosenthal2023, xi_pol_overlap)

P = snv_rosenthal2023()
BUD = snv_emission()


def _paper_interface(beta_form="purcell"):
    """The original submission's design point, from ITS stated inputs:
    Q_L = 500, V = 0.68 (lam/n)^3, xi_pos = 0.5, taper transfer 0.991, chip 0.8,
    detector 0.9, eta_wg = 1 - Q_L/Q_i with Q_i = 1.39e6 -- and the
    cyclicity and qubit frequency computed by THIS package's spin
    model at that version's operating field."""
    b = field_on_circle(147.0, 0.125)
    # that version uses the MEASURED cyclicity 2244 (arXiv:2403.13110) at
    # its operating point; the model's swept-circle value at the
    # nominal angles is smaller because of the reported ~10 deg
    # residual misalignment -- and the model DOES reproduce 2244 at
    # 9 deg from the axis (test_snv_measured), so the two are one
    # consistent story, checked from both ends.
    lam0 = 2244.0
    omega_q = qubit_frequency(P, b) * 1e9
    return CavityInterface(
        q_loaded=500.0, v_rel=0.68, eta_wg=1.0 - 500.0 / 1.39e6,
        budget=BUD, lambda0=lam0, omega_q_hz=omega_q,
        xi_pol=xi_pol_overlap(), xi_pos=0.5,
        t_link=0.991, eta_chip=0.8, eta_det=0.9,
        beta_form=beta_form), omega_q


def test_design_point_reproduced():
    ci, _ = _paper_interface()
    assert abs(ci.f_c - 19.0) < 0.05 * 19.0       # original submission: F_C ~ 19
    ci.require_valid()                            # in-regime by design
    old, _ = _paper_interface("transition")
    assert abs(old.eta - 0.65) < 0.05 * 0.65      # original submission: eta ~ 65%
    # the default differs only by the mode share: (F_C - 1) / F_C
    assert abs(ci.eta / old.eta - (ci.f_c - 1.0) / ci.f_c) < 1e-12
    assert abs(ci.eta - 0.6150) < 5e-5 and abs(old.eta - 0.6486) < 5e-5


def test_fast_readout_fidelity():
    """The original submission's headline regime: single-shot fidelity
    above 98% within a sub-100-ns window, with the dark leak computed
    from the cavity linewidth and qubit frequency (power broadening at
    s = 2), as that version did: the leak takes the spin-flipping line
    A2 at the full strength of A1, which overstates that one channel
    (A2 is weaker by about 1 / Lambda). That version left out the
    channel the revised study finds dominant, the dark spin's own line
    B2, so these numbers are a regression of the chain, not a bound on
    the real fidelity.
    Since 0.4.0 the dark switch-on channel is evaluated exactly (the
    source pipeline used a 9-point quadrature, vacspin 0.3 a 24-point
    sum); the numbers sit on the conservative side of that version's
    98.5%: 0.9802 at the best of the four windows (0.9813 in 0.5.0,
    whose efficiency used F_C rather than F_C - 1; 0.9818 in 0.3.1),
    0.9481 at a full 100 ns with f0 = 0.99 (0.9488 in 0.5.0, 0.9512 in
    0.3.1)."""
    ci, omega_q = _paper_interface()
    s = 2.0
    gamma_cav_hz = ci.gamma_cav / (2.0 * np.pi)
    leak = (1.0 + s) / (1.0 + s + (2.0 * omega_q / gamma_cav_hz) ** 2)
    best = max(fidelity(ci.eta, ci.lambda_cav, ci.gamma_cav, tau,
                        s=s, leak=leak, noise_rate=1e3)["fidelity"]
               for tau in (10e-9, 20e-9, 30e-9, 50e-9))
    assert best > 0.98
    out = fidelity(ci.eta, ci.lambda_cav, ci.gamma_cav, 0.1e-6,
                   s=s, leak=leak, noise_rate=1e3, f0=0.99)
    assert out["fidelity"] > 0.94
    # and the fidelity is honest about its budget: dropping the
    # detector efficiency must lower it
    out_worse = fidelity(ci.eta * 0.5, ci.lambda_cav, ci.gamma_cav,
                         0.1e-6, s=s, leak=leak, noise_rate=1e3,
                         f0=0.99)
    assert out_worse["fidelity"] < out["fidelity"]


def test_confocal_to_cavity_gain_is_orders_of_magnitude():
    """The whole point of the cavity: the same emitter that needs
    ~50 us confocally reaches higher fidelity in ~0.1 us -- a >100x
    speedup delivered by the chain, computed, not asserted."""
    from vacspin import required_window
    tau_confocal = required_window(0.85, eta=0.002, lam=2244.0,
                                   gamma=BUD.gamma0, s=10.0,
                                   noise_rate=0.2 / 50e-6, tau_max=1.0)
    ci, omega_q = _paper_interface()
    s = 2.0
    gamma_cav_hz = ci.gamma_cav / (2.0 * np.pi)
    leak = (1.0 + s) / (1.0 + s + (2.0 * omega_q / gamma_cav_hz) ** 2)
    tau_cavity = required_window(0.85, eta=ci.eta, lam=ci.lambda_cav,
                                 gamma=ci.gamma_cav, s=s, leak=leak,
                                 noise_rate=1e3, tau_max=1.0)
    assert tau_confocal / tau_cavity > 100.0
