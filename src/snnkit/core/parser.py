"""A deliberately minimal equation parser (Week 5).

Scope, per the roadmap critique: supports **only first-order ODEs without
units** (e.g. `"dv/dt = (I - v) / tau"`). This is explicitly NOT a general
Brian2-equivalent compiler. Out of scope for v1, noted as future work,
not attempted here:

    - Physical units / dimensional analysis
    - Complex dependency ordering between multiple coupled equations
    - Multi-variable coupled ODE systems solved simultaneously

If this parser starts accumulating special cases to handle those, per the
roadmap's own risk flag for Week 5: **stop and re-scope rather than grow
it into a general compiler.**

Equation grammar accepted:

    "d<var>/dt = <expr>"

where `<expr>` is any sympy-parseable expression in `<var>` and any number
of named parameters (e.g. `tau`, `I`) that are supplied at call time.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import sympy
from sympy import symbols
from sympy.utilities.lambdify import lambdify

_ODE_PATTERN = re.compile(r"^\s*d(\w+)\s*/\s*dt\s*=\s*(.+)$")


class ParserScopeError(ValueError):
    """Raised when an equation string falls outside this parser's deliberately
    narrow scope (units, multi-variable systems, etc.) — by design, not a bug."""


@dataclass
class ParsedODE:
    """A parsed first-order ODE `d<state_var>/dt = f(state_var, **params)`.

    Attributes:
        state_var: name of the differentiated variable (e.g. "v").
        param_names: names of free symbols in the RHS other than the state
            variable — these must be supplied as kwargs to `derivative_fn`.
        expr: the parsed sympy expression (RHS).
        derivative_fn: a JAX-traceable function
            `(state_value, **params) -> d(state)/dt`.
    """

    state_var: str
    param_names: tuple[str, ...]
    expr: sympy.Expr
    derivative_fn: Callable[..., jax.Array]


def parse_ode(equation: str) -> ParsedODE:
    """Parse a first-order ODE string of the form `"dv/dt = (I - v) / tau"`.

    Raises:
        ParserScopeError: if the equation isn't of the supported
            `d<var>/dt = <expr>` form, or if units are detected (a bare
            physical-unit suffix like "20 * ms" is not supported — pass
            pre-converted numeric values instead).
    """
    match = _ODE_PATTERN.match(equation)
    if not match:
        raise ParserScopeError(
            f"Equation {equation!r} is not of the supported form 'd<var>/dt = <expr>'. "
            "This parser only handles single first-order ODEs (Week 5 scope)."
        )
    state_var, rhs_str = match.group(1), match.group(2)

    # Reject obvious unit suffixes (ms, mV, nS, ...) rather than silently
    # mis-parsing them as undefined symbols.
    unit_like = re.findall(r"\b\d+(?:\.\d+)?\s*\*?\s*(ms|mV|nS|pF|Hz|nA|pA)\b", rhs_str)

    if unit_like:
        raise ParserScopeError(
            f"Detected unit suffix(es) {unit_like} in {rhs_str!r}. "
            "This parser does not support units (Week 5 scope) — pass "
            "pre-converted numeric values (e.g. tau=0.02 for 20ms, not tau=20*ms)."
        )

    try:
        # sympy treats the bare symbol "I" as the imaginary unit by default,
        # which silently breaks the extremely common convention of naming
        # input current "I" in neuron equations. Override it (and a couple
        # of other common single-letter clashes) to plain real symbols.
        clashing_names = ("I", "E", "N", "O", "S")
        local_overrides = {name: symbols(name, real=True) for name in clashing_names}
        expr = sympy.sympify(rhs_str, locals=local_overrides)
    except (sympy.SympifyError, TypeError) as e:
        raise ParserScopeError(f"Could not parse RHS {rhs_str!r}: {e}") from e

    state_sym = symbols(state_var)
    other_symbols = sorted((expr.free_symbols - {state_sym}), key=str)
    param_names = tuple(str(s) for s in other_symbols)

    all_syms = (state_sym, *other_symbols)
    numeric_fn = lambdify(all_syms, expr, modules=[jnp, "jax"])

    def derivative_fn(state_value: jax.Array, **params: jax.Array) -> jax.Array:
        missing = set(param_names) - set(params)
        if missing:
            raise ValueError(f"Missing required parameters for ODE: {missing}")
        ordered_args = [state_value] + [params[name] for name in param_names]
        return numeric_fn(*ordered_args)

    return ParsedODE(
        state_var=state_var,
        param_names=param_names,
        expr=expr,
        derivative_fn=derivative_fn,
    )


def euler_integrate(
    parsed: ParsedODE,
    initial_value: jax.Array,
    dt: float,
    n_steps: int,
    params_trace: dict[str, jax.Array],
) -> jax.Array:
    """Integrate a `ParsedODE` forward with forward Euler over `n_steps`.

    Args:
        parsed: `ParsedODE` from `parse_ode`.
        initial_value: initial state value.
        dt: integration timestep.
        n_steps: number of steps to integrate.
        params_trace: dict mapping each param name to an array of shape
            `[n_steps, ...]` (its value at each step) or a scalar (constant
            across steps).

    Returns:
        State trace, shape `[n_steps, ...]`.
    """

    def get_step_params(t: int) -> dict[str, jax.Array]:
        out = {}
        for name in parsed.param_names:
            val = params_trace[name]
            out[name] = val[t] if jnp.ndim(val) > 0 else val
        return out

    def step(state, t):
        p = get_step_params(t)
        deriv = parsed.derivative_fn(state, **p)
        new_state = state + dt * deriv
        return new_state, new_state

    _, trace = jax.lax.scan(step, initial_value, jnp.arange(n_steps))
    return trace
