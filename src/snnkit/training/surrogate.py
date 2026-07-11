"""Surrogate-gradient spike non-linearity + a differentiable LIF simulation
loop (Week 7). This is a standalone "is it differentiable end-to-end"
step, deliberately separated from actually training anything
(`snnkit.training.bptt`, Week 8), per the roadmap critique.

The hard spike threshold (`v >= v_th`) has zero gradient almost
everywhere, which kills backprop through a spiking network. The standard
fix (Zenke & Ganguli 2018, "SuperSpike") is a *surrogate gradient*: use
the true (non-differentiable) Heaviside step in the forward pass, but
substitute a smooth surrogate derivative in the backward pass. We
implement this with `jax.custom_vjp`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from snnkit.core.neuron import LIFParams


def fast_sigmoid_surrogate_grad(x: jax.Array, beta: float = 10.0) -> jax.Array:
    """Derivative of the "fast sigmoid" surrogate: `1 / (1 + beta*|x|)^2`.

    `x` is the pre-threshold value `(v - v_th)`. This is the surrogate
    used in Zenke & Ganguli's SuperSpike paper (also standard in
    surrogate-gradient BPTT for SNNs, e.g. Neftci et al. 2019).
    """
    return 1.0 / (1.0 + beta * jnp.abs(x)) ** 2


@jax.custom_vjp
def spike_fn(x: jax.Array) -> jax.Array:
    """Heaviside step spike function: `1` where `x >= 0`, else `0`.

    Forward pass is the true (non-differentiable) threshold. Gradient is
    overridden via `custom_vjp` — see `_spike_fn_bwd`.
    """
    return (x >= 0).astype(x.dtype)


def _spike_fn_fwd(x):
    return spike_fn(x), x  # save x for backward


def _spike_fn_bwd(residual_x, grad_output):
    surrogate = fast_sigmoid_surrogate_grad(residual_x)
    return (grad_output * surrogate,)


spike_fn.defvjp(_spike_fn_fwd, _spike_fn_bwd)


def differentiable_lif_step(
    v: jax.Array, i_in: jax.Array, params: LIFParams
) -> tuple[jax.Array, jax.Array]:
    """Like `snnkit.core.neuron.lif_euler_step`, but using the surrogate
    spike function so `jax.grad` can flow through the threshold."""
    dv = (params.dt / params.tau) * (-(v - params.v_rest) + params.r * i_in)
    v_pre_reset = v + dv
    spiked = spike_fn(v_pre_reset - params.v_th)
    # Soft reset via `spiked` (differentiable) rather than `jnp.where` on a
    # boolean, so the reset itself doesn't reintroduce a non-differentiable
    # branch: v_next = v_pre_reset * (1 - spiked) + v_reset * spiked.
    v_next = v_pre_reset * (1.0 - spiked) + params.v_reset * spiked
    return v_next, spiked


def differentiable_simulate_lif(
    i_trace: jax.Array, params: LIFParams, v0: jax.Array | float = 0.0
) -> tuple[jax.Array, jax.Array]:
    """Differentiable counterpart of `snnkit.core.neuron.simulate_lif`,
    using the surrogate-gradient spike function at every step so gradients
    flow from a loss on the output back through the full `lax.scan`."""
    v0_arr = jnp.broadcast_to(jnp.asarray(v0, dtype=i_trace.dtype), i_trace.shape[1:])

    def step(v, i_t):
        v_next, spiked = differentiable_lif_step(v, i_t, params)
        return v_next, (v_next, spiked)

    _, (v_trace, spike_trace) = lax.scan(step, v0_arr, i_trace)
    return v_trace, spike_trace
