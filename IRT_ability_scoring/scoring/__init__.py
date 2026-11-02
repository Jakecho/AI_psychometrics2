"""Person scoring with known item parameters: MLE, EAP, MAP, WLE, TCC inversion, EAPsum."""
from .eap import convergence_table, eap, node_table, posterior, quadrature, reference_integral  # noqa: F401
from .eapsum import eapsum, recursion_trace, score_distribution, score_table  # noqa: F401
from .map_estimate import map_estimate  # noqa: F401
from .mle import mle  # noqa: F401
from .result import ScoreResult  # noqa: F401
from .settings import ScoringSettings  # noqa: F401
from .solvers import rows  # noqa: F401
from .tcc import tcc_inverse  # noqa: F401
from .wle import weighted_likelihood_mode, wle  # noqa: F401

METHODS = ("MLE", "EAP", "MAP", "WLE", "TCC", "EAPsum")
FUNCTIONS = dict(MLE=mle, EAP=eap, MAP=map_estimate, WLE=wle, TCC=tcc_inverse, EAPsum=eapsum)
UNCERTAINTY = dict(MLE="Information-based SE 1/√I", EAP="Posterior SD", MAP="Posterior-curvature SE 1/√(I + 1/σ²)",
                   WLE="SE 1/√I at the WLE", TCC="Delta-method SE √ΣVar / T′ (approximation)",
                   EAPsum="Posterior SD given the summed score")


def score_all(items, Y, settings, methods=METHODS):
    return {m: FUNCTIONS[m](items, Y, settings) for m in methods}
