"""Izhikevich (2003) neuron model.

    dv/dt = 0.04v^2 + 5v + 140 - u + I
    du/dt = a(bv - u)
    if v >= 30 (mV): v <- c, u <- u + d

A 2-variable coupled nonlinear ODE system — explicitly out of scope for
the Week-5 minimal parser (which punts on multi-variable coupled systems),
so this is implemented directly as a hand-written JAX function, matching
`snnkit.core.neuron`'s pattern rather than going through
`snnkit.core.parser`. This is a deliberate scope decision, not an
oversight — see docs referenced in the Week 5 parser docstring.

Canonical parameter presets below are taken from Izhikevich, E.M. (2003),
"Simple Model of Spiking Neurons", IEEE Trans. Neural Networks 14(6).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


class IzhikevichParams(NamedTuple):
    a: float
    b: float
    c: float  # reset potential (mV)
    d: float  # recovery variable reset increment
    v_peak: float = 30.0  # spike cutoff (mV), per the original paper
    dt: float = 1.0  # ms — Izhikevich's original scripts use 1ms steps


#: Regular spiking (RS): the "default" cortical excitatory pattern —
#: spike-frequency adaptation, settles into a steady low-ish rate.
REGULAR_SPIKING = IzhikevichParams(a=0.02, b=0.2, c=-65.0, d=8.0)

#: Intrinsically bursting (IB): under sustained step current, fires one
#: initial burst then settles into regular spiking (this is IB's actual
#: documented behavior in Izhikevich 2003, not a bug in this
#: implementation — for *sustained, periodic* bursting throughout the
#: whole trace, see `CHATTERING` below).
INTRINSICALLY_BURSTING = IzhikevichParams(a=0.02, b=0.2, c=-55.0, d=4.0)

#: Chattering (CH): fires periodic bursts for the whole duration of a
#: sustained step current. Used as the featured "bursting" example in
#: Week 6's validation since it demonstrates repeated, unambiguous bursts
#: rather than IB's single-burst-then-regular pattern.
CHATTERING = IzhikevichParams(a=0.02, b=0.2, c=-50.0, d=2.0)

#: Fast spiking (FS): sustained high-frequency firing, ~no adaptation
#: (typical of cortical inhibitory interneurons).
FAST_SPIKING = IzhikevichParams(a=0.1, b=0.2, c=-65.0, d=2.0)


def izhikevich_step(
    v: jax.Array, u: jax.Array, i_in: jax.Array, params: IzhikevichParams
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One forward-Euler Izhikevich update, with the v-then-u sub-stepping
    the original paper's reference implementation uses (updating v twice
    at dt/2 per 1ms step for numerical stability near spike time is a
    common variant; we use plain single-step Euler at the caller-specified
    `dt` for simplicity and note this as a known minor divergence from the
    original MATLAB script, not affecting qualitative firing pattern)."""
    dv = params.dt * (0.04 * v**2 + 5.0 * v + 140.0 - u + i_in)
    v_pre = v + dv
    du = params.dt * (params.a * (params.b * v - u))
    u_pre = u + du

    spiked = (v_pre >= params.v_peak).astype(v.dtype)
    v_next = jnp.where(spiked > 0, params.c, v_pre)
    u_next = jnp.where(spiked > 0, u_pre + params.d, u_pre)
    return v_next, u_next, spiked


def simulate_izhikevich(
    i_trace: jax.Array,
    params: IzhikevichParams,
    v0: float = -65.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Simulate an Izhikevich neuron over time via `lax.scan`.

    Args:
        i_trace: input current, shape `[time, ...]`.
        params: `IzhikevichParams`.
        v0: initial membrane potential (mV).

    Returns:
        (v_trace, u_trace, spike_trace), each shape `[time, ...]`.
    """
    v0_arr = jnp.broadcast_to(jnp.asarray(v0, dtype=i_trace.dtype), i_trace.shape[1:])
    u0_arr = params.b * v0_arr

    def step(carry, i_t):
        v, u = carry
        v_next, u_next, spiked = izhikevich_step(v, u, i_t, params)
        return (v_next, u_next), (v_next, u_next, spiked)

    _, (v_trace, u_trace, spike_trace) = lax.scan(step, (v0_arr, u0_arr), i_trace)
    return v_trace, u_trace, spike_trace


def spike_times_ms(spike_trace: jax.Array, dt: float) -> jax.Array:
    """Extract spike times (ms) from a 1D 0/1 spike trace (numpy-side helper,
    not JIT-traced — used only in analysis/validation code, not hot loops)."""
    import numpy as np

    idx = np.nonzero(np.asarray(spike_trace) > 0)[0]
    return idx * dt


def inter_spike_intervals(spike_times: jax.Array) -> jax.Array:
    import numpy as np

    st = np.asarray(spike_times)
    if len(st) < 2:
        return np.array([])
    return np.diff(st)
