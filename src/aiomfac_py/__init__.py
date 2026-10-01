"""aiomfac_py: pure-Python port of the AIOMFAC model (activity-coefficient core).

Derived from AIOMFAC-web (Fortran) by A. Zuend et al.; GPL-3.0-or-later.
See README.md for the citation requirements of the original model.
"""
__version__ = "0.0.23"

#: commit of https://github.com/andizuend/AIOMFAC (AIOMFAC-web v3.14) this port is validated against
AIOMFAC_REFERENCE_COMMIT = "b9cb96d0eb22dc65a5e63edafe1ed97bd07662f2"

from .io import Component, InputCase, read_input_file  # noqa: E402
from .params import MRParams, SRParams, SubgroupParams, load_mr_params, load_sr_params, load_subgroup_params  # noqa: E402
from .system import SRSystem, build_sr_system  # noqa: E402
from .sr import SRTerms, sr_terms  # noqa: E402
from .completion import CompletionResult, complete_components  # noqa: E402
from .carbonate import CarbonateResult, CarbSulfResult, solve_carb_sulf, solve_carbonate  # noqa: E402
from .dissociation import BisulfateResult, ln_k_hso4_at_t, solve_bisulfate  # noqa: E402
from .model import ActivityModel, ActivityTerms, activity_coefficients  # noqa: E402
from .lle import AiomfacGFE, PhaseSplitResult, solve_pep, solve_pep_gfe  # noqa: E402
from .spinodal import reduced_hessian, stability_determinant, stability_map  # noqa: E402
from .gp_partition import VolatileSpecies, GPResult, gp_partition, distinct_phases  # noqa: E402
from .viscosity import (  # noqa: E402
    OrganicViscosityResult, ViscosityResult, aquelec_viscosity, aquorg_viscosity, electrolyte_viscosity,
    organic_mixture_viscosity, predict_tg_derieux2018, pure_organic_viscosity_vtf, water_viscosity_pas,
)

__all__ = ["Component", "InputCase", "read_input_file", "MRParams", "SRParams", "SubgroupParams", "load_mr_params", "load_sr_params", "load_subgroup_params",
           "SRSystem", "build_sr_system", "SRTerms", "sr_terms",
           "CompletionResult", "complete_components", "CarbonateResult", "solve_carbonate", "CarbSulfResult", "solve_carb_sulf", "BisulfateResult", "ln_k_hso4_at_t", "solve_bisulfate", "ActivityModel", "ActivityTerms", "activity_coefficients",
           "AiomfacGFE", "PhaseSplitResult", "solve_pep", "solve_pep_gfe",
           "reduced_hessian", "stability_determinant", "stability_map",
           "VolatileSpecies", "GPResult", "gp_partition", "distinct_phases",
           "ViscosityResult", "OrganicViscosityResult", "electrolyte_viscosity", "water_viscosity_pas",
           "organic_mixture_viscosity", "pure_organic_viscosity_vtf", "aquelec_viscosity", "aquorg_viscosity",
           "predict_tg_derieux2018",
           "AIOMFAC_REFERENCE_COMMIT", "__version__"]
