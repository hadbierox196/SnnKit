"""Network-diffusion pathology-spread model over the connectome graph
(Week 14), reusing `snnkit.connectome.loader`'s dense adjacency.

Model: the standard graph heat equation,

    dx/dt = -diffusion_rate * L @ x

where `L = D - A` is the combinatorial graph Laplacian (`D` = degree
matrix, `A` = symmetrized adjacency — pathology is modeled as spreading
along a connection regardless of the underlying synapse's signaling
direction, so we symmetrize before building `L`). `x` is a per-node
"pathology load" scalar.

**Validation (per the roadmap's critique — not just "looks plausible"):**
`simulate_diffusion_euler`'s forward-Euler integration is checked against
`reference_diffusion_expm`, the *exact* closed-form solution
`x(t) = expm(-diffusion_rate * L * t) @ x0` (matrix exponential — the
textbook solution to the linear heat equation). See
`tests/test_connectome_diffusion.py` for the quantitative comparison
(correlation, relative L2 error).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from snnkit.connectome.loader import ConnectomeGraph, dense_adjacency


def build_laplacian(graph: ConnectomeGraph, symmetrize: bool = True) -> jnp.ndarray:
    """Build the combinatorial graph Laplacian `L = D - A`.

    Args:
        graph: `ConnectomeGraph`.
        symmetrize: if `True` (default), use `A + A^T` before computing
            the Laplacian — appropriate for pathology spread, which
            propagates along a physical connection regardless of the
            chemical synapse's signaling direction. Set `False` to use
            the raw directed adjacency (e.g. for comparing against a
            directed-spread variant).

    Returns:
        `[n, n]` Laplacian matrix.
    """
    adj = dense_adjacency(graph)
    if symmetrize:
        adj = adj + adj.T
    degree = jnp.sum(adj, axis=1)
    return jnp.diag(degree) - adj


def simulate_diffusion_euler(
    x0: jnp.ndarray, laplacian: jnp.ndarray, diffusion_rate: float, dt: float, n_steps: int
) -> jnp.ndarray:
    """Forward-Euler integration of `dx/dt = -diffusion_rate * L @ x`.

    Args:
        x0: `[n]` initial pathology load per node.
        laplacian: `[n, n]` graph Laplacian (from `build_laplacian`).
        diffusion_rate: spread rate constant.
        dt: integration timestep.
        n_steps: number of steps.

    Returns:
        `[n_steps, n]` trace of pathology load over time.
    """

    def step(x, _):
        dx = -diffusion_rate * (laplacian @ x)
        x_next = x + dt * dx
        return x_next, x_next

    _, trace = jax.lax.scan(step, x0, None, length=n_steps)
    return trace


def reference_diffusion_expm(
    x0: jnp.ndarray, laplacian: jnp.ndarray, diffusion_rate: float, t: float
) -> np.ndarray:
    """Exact closed-form solution `x(t) = expm(-diffusion_rate * L * t) @ x0`,
    via `scipy.linalg.expm` — the reference this module's Euler integration
    is validated against. Runs on CPU/numpy (matrix exponential isn't a hot
    loop, and this is only used for validation, not the training/simulation
    path itself)."""
    from scipy.linalg import expm

    l_np = np.asarray(laplacian)
    x0_np = np.asarray(x0)
    return expm(-diffusion_rate * l_np * t) @ x0_np


def diffusion_fit_quality(euler_final: jnp.ndarray, reference_final: np.ndarray) -> dict:
    """Quantitative comparison metrics between the Euler simulation's final
    state and the exact reference solution's final state."""
    euler_np = np.asarray(euler_final)
    rel_l2_error = float(
        np.linalg.norm(euler_np - reference_final) / np.linalg.norm(reference_final)
    )
    correlation = float(np.corrcoef(euler_np, reference_final)[0, 1])
    return {"relative_l2_error": rel_l2_error, "correlation": correlation}
