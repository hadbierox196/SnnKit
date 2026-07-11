"""Ring-buffer synaptic delay delivery (Week 4).

A spike emitted at timestep `t` on a synapse with delay `d` should affect
the postsynaptic neuron's input current exactly at timestep `t + d`. We
implement this with a circular (ring) buffer of length `max_delay + 1`:
writing a contribution at slot `(t + d) mod buffer_len` and reading /
clearing the slot for the current timestep `t mod buffer_len` every step.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from snnkit.core.synapses import SparseWeights, inject_current


class DelayedWeights(NamedTuple):
    """`SparseWeights` extended with a per-synapse integer delay."""

    weights: SparseWeights
    delay_steps: jax.Array  # [n_synapses], int32, >= 1


class RingBufferState(NamedTuple):
    """State of the delay ring buffer.

    `buffer` has shape `[buffer_len, n_post]`: `buffer[slot]` holds the
    total current that should be delivered when the simulation clock's
    `t mod buffer_len == slot`.
    """

    buffer: jax.Array
    buffer_len: int  # static: max_delay + 1


def init_ring_buffer(n_post: int, max_delay_steps: int) -> RingBufferState:
    """Create an empty ring buffer sized for delays up to `max_delay_steps`."""
    buffer_len = max_delay_steps + 1
    return RingBufferState(buffer=jnp.zeros((buffer_len, n_post)), buffer_len=buffer_len)


def ring_buffer_step(
    state: RingBufferState,
    t: jax.Array,
    spikes_pre: jax.Array,
    delayed_weights: DelayedWeights,
) -> tuple[RingBufferState, jax.Array]:
    """Advance the ring buffer by one timestep.

    1. Read out (and clear) the slot for the *current* timestep `t` — this
       is the current to deliver *now*, from spikes emitted `delay` steps
       ago.
    2. Compute this timestep's new spike contributions and scatter-add them
       into the slot for `t + delay`, per-synapse.

    Args:
        state: current `RingBufferState`.
        t: current (scalar) timestep index.
        spikes_pre: `[n_pre]` presynaptic spikes at timestep `t`.
        delayed_weights: `DelayedWeights`.

    Returns:
        (new_state, current_out) where `current_out` is `[n_post]`, the
        current to inject into postsynaptic neurons *this* timestep.
    """
    buffer_len = state.buffer_len
    slot_now = jnp.mod(t, buffer_len)

    current_out = state.buffer[slot_now]
    buffer_cleared = state.buffer.at[slot_now].set(0.0)

    weights = delayed_weights.weights
    pre_spike_per_synapse = jnp.take(spikes_pre, weights.pre_idx, axis=-1)
    contribution = pre_spike_per_synapse * weights.weight  # [n_synapses]

    target_slot = jnp.mod(t + delayed_weights.delay_steps, buffer_len)  # [n_synapses]
    # Scatter each synapse's contribution to (target_slot, post_idx).
    buffer_updated = buffer_cleared.at[target_slot, weights.post_idx].add(contribution)

    return RingBufferState(buffer=buffer_updated, buffer_len=buffer_len), current_out


def simulate_with_delays(
    spike_trace_pre: jax.Array,
    delayed_weights: DelayedWeights,
    n_post: int,
    max_delay_steps: int,
) -> jax.Array:
    """Run the ring buffer across a full presynaptic spike trace.

    Args:
        spike_trace_pre: `[time, n_pre]` presynaptic spikes.
        delayed_weights: `DelayedWeights`.
        n_post: number of postsynaptic neurons.
        max_delay_steps: static upper bound on `delay_steps` values.

    Returns:
        `[time, n_post]` delivered current at each timestep.
    """
    time_steps = spike_trace_pre.shape[0]
    init_state = init_ring_buffer(n_post, max_delay_steps)

    def step(state, t_and_spikes):
        t, spikes_pre = t_and_spikes
        new_state, current_out = ring_buffer_step(state, t, spikes_pre, delayed_weights)
        return new_state, current_out

    ts = jnp.arange(time_steps)
    _, current_trace = jax.lax.scan(step, init_state, (ts, spike_trace_pre))
    return current_trace


def unused_inject_current_reference(spikes_pre: jax.Array, weights: SparseWeights) -> jax.Array:
    """Kept for cross-referencing against the no-delay path in tests."""
    return inject_current(spikes_pre, weights)
