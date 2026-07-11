"""Population + batch simulation, vectorized via `jax.vmap`.

Phase 0 decision: batch dimension is `[batch, neurons, ...]` from day one.
This module vectorizes `snnkit.core.neuron.simulate_lif` over the neuron
axis using `vmap` (rather than relying only on elementwise broadcasting),
so heterogeneous per-neuron parameters (e.g. a distinct `tau` per neuron)
work correctly and explicitly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from snnkit.core.neuron import LIFParams, simulate_lif


def _in_axes_for_params(params: LIFParams) -> LIFParams:
    """Build vmap `in_axes` for `LIFParams`: 0 for per-neuron arrays
    (ndim > 0), None for scalars shared across the population."""
    axes = []
    for field in params:
        arr = jnp.asarray(field)
        axes.append(0 if arr.ndim > 0 else None)
    return LIFParams(*axes)


def simulate_population(
    i_trace: jax.Array,
    params: LIFParams,
    v0: jax.Array | float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Simulate a population of neurons across a batch dimension.

    Args:
        i_trace: input current, shape `[time, batch, neurons]`.
        params: `LIFParams`; each field may be a Python scalar (shared
            across the whole population) or a `[neurons]`-shaped array for
            per-neuron heterogeneity.
        v0: initial potential, scalar or broadcastable to `[batch, neurons]`.

    Returns:
        (v_trace, spike_trace), each shape `[time, batch, neurons]`.
    """
    time_steps, batch, n_neurons = i_trace.shape
    v0_arr = jnp.broadcast_to(jnp.asarray(v0, dtype=i_trace.dtype), (batch, n_neurons))

    # Move the neuron axis to the front so vmap maps over it: [neurons, time, batch]
    i_by_neuron = jnp.moveaxis(i_trace, 2, 0)
    v0_by_neuron = jnp.moveaxis(v0_arr, 1, 0)  # [neurons, batch]

    in_axes_params = _in_axes_for_params(params)
    simulate_one_neuron = jax.vmap(simulate_lif, in_axes=(0, in_axes_params, 0), out_axes=(0, 0))
    v_by_neuron, spikes_by_neuron = simulate_one_neuron(i_by_neuron, params, v0_by_neuron)
    # v_by_neuron: [neurons, time, batch] -> back to [time, batch, neurons]
    v_trace = jnp.moveaxis(v_by_neuron, 0, 2)
    spike_trace = jnp.moveaxis(spikes_by_neuron, 0, 2)
    return v_trace, spike_trace
