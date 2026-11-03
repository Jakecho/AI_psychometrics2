"""Reference-validation cases shared by tests/test_reference.py and r_reference/run_validation.py.

Each case: (package, generating SimSpec, lab FitConfig, tolerance on the largest item-parameter difference).
"""
from core.em import FitConfig
from core.quadrature import illustrative_distribution, make_grid
from core.simulation import SimSpec

SKEWED_41 = list(illustrative_distribution(make_grid("equal", 41, 6.0).nodes, "skewed"))

TIGHT = dict(tol=1e-6, prior_tol=1e-6, max_cycles=3000)
CASES = [
    ("TAM", SimSpec(model="PCM", n_per_group=800, n_items=6, ncat=4, seed=1),
     FitConfig(model="PCM", prior="estimated_normal"), 1e-4),
    ("TAM", SimSpec(model="RSM", n_per_group=800, n_items=6, ncat=4, seed=2),
     FitConfig(model="RSM", prior="estimated_normal"), 1e-4),
    ("TAM", SimSpec(model="RSM", design="concurrent", groups=2, n_common=3, n_per_group=600, n_items=6, ncat=4, seed=2),
     FitConfig(model="RSM", design="concurrent", prior="estimated_normal"), 1e-4),
    ("TAM", SimSpec(model="PCM", design="fipc", n_anchor=4, n_per_group=800, n_items=5, ncat=3, group_means=[0.4],
                    group_sds=[1.1], seed=3), FitConfig(model="PCM", design="fipc", prior="estimated_normal"), 1e-4),
    ("TAM", SimSpec(model="RSM", design="mgfipc", groups=2, n_anchor=4, n_common=2, n_per_group=600, n_items=5, ncat=4,
                    seed=4), FitConfig(model="RSM", design="mgfipc", prior="estimated_normal"), 1e-4),
    ("TAM", SimSpec(model="GPCM", n_per_group=800, n_items=6, ncat=4, seed=5),
     FitConfig(model="GPCM", prior="fixed_normal"), 1e-4),
    ("irtQ", SimSpec(model="GPCM", n_per_group=800, n_items=6, ncat=4, seed=5),
     FitConfig(model="GPCM", prior="fixed_normal"), 1e-4),
    ("irtQ", SimSpec(model="GPCM", n_per_group=1000, n_items=6, ncat=4, shape="skewed", seed=10),
     FitConfig(model="GPCM", prior="fixed_empirical", fixed_weights=SKEWED_41), 1e-4),
    ("irtQ", SimSpec(model="GRM", design="concurrent", groups=2, n_common=3, n_per_group=800, n_items=6, ncat=4, seed=7),
     FitConfig(model="GRM", design="concurrent", prior="estimated_normal", normal_update="moments"), 1e-4),
    ("irtQ", SimSpec(model="GPCM", design="fipc", n_anchor=5, n_per_group=1000, n_items=5, ncat=4, group_means=[0.4],
                     group_sds=[1.1], seed=8),
     FitConfig(model="GPCM", design="fipc", prior="estimated_normal", normal_update="moments"), 1e-4),
    ("irtQ", SimSpec(model="GPCM", design="fipc", n_anchor=5, n_per_group=1000, n_items=5, ncat=4, group_means=[0.4],
                     group_sds=[1.1], seed=8),
     FitConfig(model="GPCM", design="fipc", prior="estimated_normal", normal_update="moments", fipc_method="OEM"), 1e-4),
    ("irtQ", SimSpec(model="GRM", design="mgfipc", groups=2, n_anchor=4, n_common=2, n_per_group=800, n_items=5, ncat=4,
                     seed=9), FitConfig(model="GRM", design="mgfipc", prior="estimated_normal", normal_update="moments"),
     1e-4),
    ("irtQ", SimSpec(model="GRM", n_per_group=1000, n_items=6, ncat=4, seed=6),
     FitConfig(model="GRM", prior="estimated_empirical", empirical_update="woods"), 5e-3),
    ("mirt", SimSpec(model="GPCM", n_per_group=800, n_items=6, ncat=4, seed=5), FitConfig(model="GPCM"), 1e-4),
    ("mirt", SimSpec(model="PCM", n_per_group=800, n_items=6, ncat=4, seed=1),
     FitConfig(model="PCM", prior="estimated_normal", normal_update="moments"), 1e-4),
    ("mirt", SimSpec(model="GRM", design="concurrent", groups=2, n_common=3, n_per_group=800, n_items=6, ncat=4, seed=7),
     FitConfig(model="GRM", design="concurrent", prior="estimated_normal", normal_update="moments"), 1e-4),
]
# Documented differences (run, reported, not asserted): estimators that differ by design.
DIFFERENCES = [
    ("irtQ", SimSpec(model="GRM", n_per_group=1000, n_items=6, ncat=4, seed=6),
     FitConfig(model="GRM", prior="estimated_empirical", empirical_update="constrained"),
     "Exact constrained histogram (lab) vs Woods rescaling (irtQ EmpHist)"),
    ("irtQ", SimSpec(model="PCM", design="fipc", n_anchor=4, n_per_group=800, n_items=5, ncat=3, group_means=[0.4],
                     group_sds=[1.1], seed=3), FitConfig(model="PCM", design="fipc", prior="estimated_empirical"),
     "Free histogram in FIPC: stopping rule differs (irtQ checks item changes only)"),
    ("TAM", SimSpec(model="PCM", n_per_group=800, n_items=6, ncat=4, seed=1),
     FitConfig(model="PCM", prior="estimated_normal", normal_update="exact", n_nodes=7),
     ("Coarse 7-node grid: TAM weights are φ(θ_q)·Δθ, not renormalised (they sum to 0.50 here); evaluated that way the lab "
     "reproduces TAM's ℓ exactly. Identical results from about 15 nodes")),
]
