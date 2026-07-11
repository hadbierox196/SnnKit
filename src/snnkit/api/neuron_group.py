"""`NeuronGroup`: object API for a population of neurons defined by an
equation string, wrapping `snnkit.core.parser` + the spike/reset logic
that lives in `snnkit.core.neuron` for the hand-written LIF path.

This class contains only bookkeeping (shapes, parameter threading) and
delegates all numerics to `snnkit.core` — per the Phase 0 API philosophy,
dynamics belong in core, structure belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from snnkit.core.parser import ParsedODE, parse_ode


@dataclass
class NeuronGroup:
    """A population of `n` neurons sharing one governing equation.

    Args:
        n: number of neurons in the group.
        equation: a `d<var>/dt = <expr>` string, e.g. `"dv/dt = (I - v) / tau"`.
        v_th: spike threshold (applied to the state variable).
        v_reset: post-spike reset value.
        dt: integration timestep.
    """

    n: int
    equation: str
    v_th: float = 1.0
    v_reset: float = 0.0
    dt: float = 1e-3
    _parsed: ParsedODE = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._parsed = parse_ode(self.equation)

    @property
    def param_names(self) -> tuple[str, ...]:
        """Names of parameters the governing equation requires (besides the
        state variable itself), e.g. `("I", "tau")` for the LIF equation."""
        return self._parsed.param_names

    def simulate(
        self,
        params_trace: dict[str, jax.Array | float],
        n_steps: int,
        v0: jax.Array | float = 0.0,
    ) -> tuple[jax.Array, jax.Array]:
        """Simulate the group for `n_steps`, threading params through the
        parsed equation and applying threshold/reset each step.

        Args:
            params_trace: maps each name in `self.param_names` to either a
                `[n_steps, n]` (or `[n_steps]`) time-varying array, or a
                scalar/constant `[n]` array (broadcast across time).
            n_steps: number of timesteps to simulate.
            v0: initial state value, scalar or `[n]`.

        Returns:
            (state_trace, spike_trace), each shape `[n_steps, n]`.
        """
        v0_arr = jnp.broadcast_to(jnp.asarray(v0, dtype=jnp.float32), (self.n,))

        def get_step_params(t: jax.Array) -> dict[str, jax.Array]:
            out = {}
            for name in self._parsed.param_names:
                val = jnp.asarray(params_trace[name])
                # Time-varying if leading dim matches n_steps; else treated
                # as constant across time (broadcast every step).
                is_time_varying = val.ndim > 0 and val.shape[0] == n_steps
                out[name] = val[t] if is_time_varying else val
            return out

        def step(v, t):
            p = get_step_params(t)
            deriv = self._parsed.derivative_fn(v, **p)
            v_pre_reset = v + self.dt * deriv
            spiked = (v_pre_reset >= self.v_th).astype(v.dtype)
            v_next = jnp.where(spiked > 0, self.v_reset, v_pre_reset)
            return v_next, (v_next, spiked)

        _, (v_trace, spike_trace) = jax.lax.scan(step, v0_arr, jnp.arange(n_steps))
        return v_trace, spike_trace
