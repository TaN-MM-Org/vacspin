"""vacspin: group-IV colour-centre spin-photon interfaces, end to end.

From the strained spin Hamiltonian of a vacancy centre in diamond
(SiV-, GeV-, SnV-, PbV-), through its optical transitions and
cyclicity, the transition-resolved Purcell budget of a coupled
cavity, to exact single-shot readout statistics and remote-
entanglement rates -- with every physics claim in the test suite
pinned to a closed form, a measured anchor, or two independent code
paths.

Methodological basis: T. M. Mahim, M. M. Rahman and A. S. M. Mohsin,
"Fast single-shot readout of tin-vacancy spins with an overcoupled
diamond nanocavity on thin-film lithium niobate" (submitted to Optics
Express, 2026), and the effective-Hamiltonian and readout formalism of
Rosenthal et al., PRX 13, 031022 (2023) and arXiv:2403.13110.
"""
from .params import (EmissionBudget, SpinParameters, gev_bhaskar2017,
                     siv_hepp2014, snv_emission, snv_rosenthal2023)
from .hamiltonian import (GAMMA_GHZ_PER_T, h_manifold, lab_to_spin,
                          qubit_frequency, qubit_frequency_perpendicular,
                          solve, zero_field_splitting)
from .transitions import (cyclicity, field_on_circle, rabi_rate,
                          transition_strength, transition_table)
from .cavity import (CavityInterface, lorentzian_suppression,
                     purcell_max, xi_pol_overlap)
from .readout import (fidelity, fidelity_threshold1, geometric_pmf,
                      poisson_pmf, polarization_rate, readout_counts,
                      required_efficiency, required_window)
from .remote import barrett_kok_success, entanglement_rate
from .lab import (SpinFit, design_fields, fit_spin_parameters,
                  load_observations_csv, parameter_information,
                  save_observations_csv)

__version__ = "0.3.0"
__all__ = [
    "SpinParameters", "EmissionBudget", "snv_rosenthal2023",
    "snv_emission", "siv_hepp2014", "gev_bhaskar2017",
    "GAMMA_GHZ_PER_T", "h_manifold", "lab_to_spin", "solve",
    "zero_field_splitting", "qubit_frequency",
    "qubit_frequency_perpendicular",
    "transition_strength", "transition_table", "cyclicity", "rabi_rate",
    "field_on_circle",
    "purcell_max", "lorentzian_suppression", "xi_pol_overlap",
    "CavityInterface",
    "polarization_rate", "readout_counts", "fidelity_threshold1",
    "geometric_pmf", "poisson_pmf", "fidelity", "required_efficiency",
    "required_window",
    "barrett_kok_success", "entanglement_rate",
    "SpinFit", "fit_spin_parameters", "parameter_information",
    "design_fields", "save_observations_csv", "load_observations_csv",
    "__version__",
]
