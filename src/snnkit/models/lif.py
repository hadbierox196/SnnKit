"""LIF model: equation definition for use with the parser/API layer, plus
re-exports of the low-level `snnkit.core.neuron` functions for direct use.

Two ways to simulate a LIF population in snnkit, by design (Phase 0 hybrid
API): the low-level, hand-written `snnkit.core.neuron` path (fast, simple,
no parsing overhead — used internally e.g. in benchmarks), and the
equation-driven `NeuronGroup` path via `LIF_EQUATION` (flexible, supports
swapping in other first-order-ODE models with the same object API). Week 5
validates that the two paths agree numerically — see
`tests/test_parser.py::test_neuron_group_matches_hand_written_lif`.
"""

from __future__ import annotations

from snnkit.api.neuron_group import NeuronGroup
from snnkit.core.neuron import LIFParams, analytical_firing_rate, lif_euler_step, simulate_lif

#: The LIF governing equation, in the minimal parser's supported grammar.
#: v_rest is fixed at 0 here (matches `snnkit.core.neuron.LIFParams` default);
#: for nonzero v_rest use "dv/dt = (-(v - v_rest) + R*I) / tau" instead.
LIF_EQUATION = "dv/dt = (I - v) / tau"


def default_lif_params() -> dict[str, float]:
    """Default LIF parameters, matching `LIFParams()` defaults."""
    p = LIFParams()
    return {"tau": p.tau, "v_th": p.v_th, "v_reset": p.v_reset, "dt": p.dt}


def make_lif_group(
    n: int, tau: float = 20e-3, v_th: float = 1.0, v_reset: float = 0.0, dt: float = 1e-3
) -> NeuronGroup:
    """Build a `NeuronGroup` of `n` LIF neurons via the equation/parser path."""
    return NeuronGroup(n=n, equation=LIF_EQUATION, v_th=v_th, v_reset=v_reset, dt=dt)


__all__ = [
    "LIF_EQUATION",
    "default_lif_params",
    "make_lif_group",
    # Re-exports of the hand-written core path:
    "LIFParams",
    "lif_euler_step",
    "simulate_lif",
    "analytical_firing_rate",
]
