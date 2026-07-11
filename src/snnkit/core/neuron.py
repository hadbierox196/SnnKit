"""Leaky integrate-and-fire (LIF) membrane dynamics as plain JAX functions.

Design choice (Phase 0): the core simulation primitives are plain functions
operating on arrays, not classes. The object-oriented `NeuronGroup` API
(`snnkit.api.neuron_group`) is a thin wrapper added in Week 5 — it should
never contain dynamics logic itself, only bookkeeping.

Model:
    tau * dv/dt = -(v - v_rest) + R * I(t)
    if v >= v_th: emit a spike, v <- v_reset

Discretized with forward Euler at step size `dt`:
    v[t+1] = v[t] + (dt / tau) * (-(v[t] - v_rest) + R * I[t])
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


class LIFParams(NamedTuple):
    """Parameters for a LIF neuron (or population, if arrays are broadcast).

    All fields may be scalars or arrays broadcastable to the neuron/batch
    shape being simulated, so the same struct works for a single neuron
    (Week 1) and for heterogeneous populations (Week 2+).
    """

    tau: float = 20e-3  # membrane time constant, seconds
    v_rest: float = 0.0  # resting potential
    v_th: float = 1.0  # spike threshold
    v_reset: float = 0.0  # post-spike reset potential
    r: float = 1.0  # membrane resistance
    dt: float = 1e-3  # simulation timestep, seconds


def lif_euler_step(v: jax.Array, i_in: jax.Array, params: LIFParams) -> tuple[jax.Array, jax.Array]:
    """One forward-Euler LIF update.

    Args:
        v: membrane potential(s), any shape.
        i_in: input current(s) at this timestep, same shape as `v`.
        params: `LIFParams`.

    Returns:
        (v_next, spiked) where `spiked` is a 0/1 float array (same shape as
        `v`, spike non-linearity is a hard threshold here — the
        differentiable surrogate version lives in `snnkit.training.surrogate`,
        deliberately kept out of the core engine so the core stays a plain,
        inspectable numerical reference).
    """
    dv = (params.dt / params.tau) * (-(v - params.v_rest) + params.r * i_in)
    v_pre_reset = v + dv
    spiked = (v_pre_reset >= params.v_th).astype(v.dtype)
    v_next = jnp.where(spiked > 0, params.v_reset, v_pre_reset)
    return v_next, spiked


def simulate_lif(
    i_trace: jax.Array,
    params: LIFParams,
    v0: jax.Array | float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Simulate a LIF neuron/population over time using `lax.scan`.

    Args:
        i_trace: input current, shape `[time, ...]` (`...` is any neuron /
            batch shape — this function is shape-agnostic and gets vectorized
            over population/batch dims in `snnkit.core.population`).
        params: `LIFParams`.
        v0: initial membrane potential, broadcastable to `i_trace[0].shape`.

    Returns:
        (v_trace, spike_trace), both shape `[time, ...]`.
    """
    v0_arr = jnp.broadcast_to(jnp.asarray(v0, dtype=i_trace.dtype), i_trace.shape[1:])

    def step(v, i_t):
        v_next, spiked = lif_euler_step(v, i_t, params)
        return v_next, (v_next, spiked)

    _, (v_trace, spike_trace) = lax.scan(step, v0_arr, i_trace)
    return v_trace, spike_trace


def analytical_firing_rate(i_in: jax.Array | float, params: LIFParams) -> jax.Array:
    """Closed-form steady-state LIF firing rate for constant input current.

    Standard result for an LIF neuron with `v_rest = 0` under constant
    current input, no refractory period:

        rate = 1 / (tau * ln(I / (I - v_th)))   if R*I > v_th
        rate = 0                                 otherwise

    Returns:
        Firing rate in Hz (assuming `tau`/`dt` are in seconds).
    """
    i_in = jnp.asarray(i_in, dtype=jnp.float32)
    r_i = params.r * i_in
    can_fire = r_i > params.v_th
    # Avoid log(<=0) / div-by-zero for sub-threshold currents by clamping;
    # result is discarded via `jnp.where` for those entries anyway.
    safe_ratio = jnp.where(can_fire, r_i / jnp.maximum(r_i - params.v_th, 1e-12), 2.0)
    rate = 1.0 / (params.tau * jnp.log(safe_ratio))
    return jnp.where(can_fire, rate, 0.0)
