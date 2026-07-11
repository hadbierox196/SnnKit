"""Week 14 deliverable: diffusion model validated against a reference
(exact matrix-exponential heat-equation solution), reporting correlation
and relative L2 error — not just visual plausibility."""

import jax.numpy as jnp

from snnkit.connectome.diffusion import (
    build_laplacian,
    diffusion_fit_quality,
    reference_diffusion_expm,
    simulate_diffusion_euler,
)
from snnkit.connectome.loader import load_white1986_connectome

CORRELATION_THRESHOLD = 0.999
REL_L2_ERROR_THRESHOLD = 0.02


def test_euler_diffusion_matches_matrix_exponential_reference():
    graph = load_white1986_connectome()
    laplacian = build_laplacian(graph, symmetrize=True)
    n = len(graph.node_names)

    source_idx = graph.node_names.index("ADAL")
    x0 = jnp.zeros(n).at[source_idx].set(1.0)

    diffusion_rate = 0.001
    t_final = 50.0
    dt = 0.1
    n_steps = int(t_final / dt)

    trace = simulate_diffusion_euler(x0, laplacian, diffusion_rate, dt, n_steps)
    euler_final = trace[-1]

    reference_final = reference_diffusion_expm(x0, laplacian, diffusion_rate, t_final)

    fit = diffusion_fit_quality(euler_final, reference_final)
    print(
        f"diffusion fit: correlation={fit['correlation']:.6f} "
        f"rel_l2_error={fit['relative_l2_error']:.4f}"
    )

    assert (
        fit["correlation"] > CORRELATION_THRESHOLD
    ), f"expected correlation > {CORRELATION_THRESHOLD}, got {fit['correlation']:.6f}"
    assert (
        fit["relative_l2_error"] < REL_L2_ERROR_THRESHOLD
    ), f"expected relative L2 error < {REL_L2_ERROR_THRESHOLD}, got {fit['relative_l2_error']:.4f}"


def test_total_pathology_mass_is_approximately_conserved():
    """The heat equation on a graph conserves total mass exactly (L has
    zero row/column sums); Euler integration should conserve it
    approximately, verified directly rather than assumed."""
    graph = load_white1986_connectome()
    laplacian = build_laplacian(graph, symmetrize=True)
    n = len(graph.node_names)

    x0 = jnp.ones(n) / n  # spread initial mass uniformly, sums to 1
    trace = simulate_diffusion_euler(x0, laplacian, diffusion_rate=0.001, dt=0.1, n_steps=500)

    initial_mass = float(x0.sum())
    final_mass = float(trace[-1].sum())
    assert abs(final_mass - initial_mass) / initial_mass < 0.01


def test_diffusion_spreads_from_source_over_time():
    """Pathology load at the source node should decrease over time as it
    spreads to neighbors (basic sanity check on the direction of spread)."""
    graph = load_white1986_connectome()
    laplacian = build_laplacian(graph, symmetrize=True)
    n = len(graph.node_names)
    source_idx = graph.node_names.index("ADAL")
    x0 = jnp.zeros(n).at[source_idx].set(1.0)

    trace = simulate_diffusion_euler(x0, laplacian, diffusion_rate=0.001, dt=0.1, n_steps=500)
    source_load_over_time = trace[:, source_idx]

    assert source_load_over_time[0] < 1.0, "source load should have started decreasing"
    assert source_load_over_time[-1] < source_load_over_time[0]
    # Some other node should have gained pathology load.
    assert jnp.max(trace[-1]) > 0 and jnp.sum(trace[-1] > 1e-6) > 1
