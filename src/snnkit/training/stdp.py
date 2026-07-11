"""Spike-timing-dependent plasticity (STDP), Week 9.

Classic pair-based STDP with exponential eligibility traces (Song, Miller
& Abbott 2000 / Gerstner & Kistler 2002 formulation):

    if a presynaptic spike arrives, potentiate by the current postsynaptic
    trace; if a postsynaptic spike occurs, depress by the current
    presynaptic trace.

    dw = A_plus  * pre_trace  (on post spike)   -- LTP, pre-before-post
    dw = -A_minus * post_trace (on pre spike)    -- LTD, post-before-pre

Traces decay exponentially with time constants `tau_plus` / `tau_minus`
and are incremented by 1 on every spike of their respective neuron —
this makes them a running record of "how recently did this neuron spike".

This module is a local learning rule: weight updates depend only on each
synapse's local pre/post spike history, not on a global loss gradient
(contrast with `snnkit.training.bptt`).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from snnkit.core.synapses import SparseWeights


class STDPParams(NamedTuple):
    tau_plus: float = 20e-3  # presynaptic trace decay (s)
    tau_minus: float = 20e-3  # postsynaptic trace decay (s)
    a_plus: float = 0.01  # LTP step size
    a_minus: float = 0.012  # LTD step size (slightly > a_plus is a common
    # convention that biases the weight distribution towards depression,
    # keeping unpotentiated synapses from drifting up on noise alone)
    dt: float = 1e-3
    w_min: float = 0.0
    w_max: float = 1.0


class STDPState(NamedTuple):
    pre_trace: jax.Array  # [n_pre]
    post_trace: jax.Array  # [n_post]


def init_stdp_state(n_pre: int, n_post: int) -> STDPState:
    return STDPState(pre_trace=jnp.zeros(n_pre), post_trace=jnp.zeros(n_post))


def stdp_step(
    state: STDPState,
    weights: SparseWeights,
    spikes_pre: jax.Array,
    spikes_post: jax.Array,
    params: STDPParams,
) -> tuple[STDPState, SparseWeights]:
    """One STDP update step: decay traces, apply spike-triggered increments,
    and update synapse weights for every (pre, post) pair with a spike this step.

    Args:
        state: current `STDPState`.
        weights: current `SparseWeights` (COO format).
        spikes_pre: `[n_pre]` presynaptic spikes this timestep.
        spikes_post: `[n_post]` postsynaptic spikes this timestep.
        params: `STDPParams`.

    Returns:
        (new_state, new_weights).
    """
    decay_pre = jnp.exp(-params.dt / params.tau_plus)
    decay_post = jnp.exp(-params.dt / params.tau_minus)

    pre_trace = state.pre_trace * decay_pre + spikes_pre
    post_trace = state.post_trace * decay_post + spikes_post

    # For each synapse (pre_idx[k], post_idx[k]):
    #   LTP if post fired this step, using the (pre-update) pre_trace value.
    #   LTD if pre fired this step, using the (pre-update) post_trace value.
    pre_trace_per_syn = state.pre_trace[weights.pre_idx]
    post_trace_per_syn = state.post_trace[weights.post_idx]
    post_fired_per_syn = spikes_post[weights.post_idx]
    pre_fired_per_syn = spikes_pre[weights.pre_idx]

    dw_ltp = params.a_plus * pre_trace_per_syn * post_fired_per_syn
    dw_ltd = -params.a_minus * post_trace_per_syn * pre_fired_per_syn
    new_weight = jnp.clip(weights.weight + dw_ltp + dw_ltd, params.w_min, params.w_max)

    new_state = STDPState(pre_trace=pre_trace, post_trace=post_trace)
    new_weights = weights._replace(weight=new_weight)
    return new_state, new_weights


def stdp_weight_change_curve(delta_t: jax.Array, params: STDPParams) -> jax.Array:
    """Theoretical STDP weight-change curve `dw(delta_t)` for a single
    isolated pre/post spike pair, `delta_t = t_post - t_spike_pre`:

        dw = a_plus  * exp(-delta_t / tau_plus)   if delta_t > 0  (pre before post: LTP)
        dw = -a_minus * exp(delta_t / tau_minus)  if delta_t < 0  (post before pre: LTD)

    Used to validate the simulated (trace-based) implementation against
    the textbook closed-form curve — see `tests/test_stdp.py`.
    """
    ltp = params.a_plus * jnp.exp(-delta_t / params.tau_plus)
    ltd = -params.a_minus * jnp.exp(delta_t / params.tau_minus)
    return jnp.where(delta_t > 0, ltp, jnp.where(delta_t < 0, ltd, 0.0))
