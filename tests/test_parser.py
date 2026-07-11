import jax
import jax.numpy as jnp
import pytest

from snnkit.core.neuron import LIFParams, simulate_lif
from snnkit.core.parser import ParserScopeError, euler_integrate, parse_ode
from snnkit.models.lif import make_lif_group


def test_parse_simple_lif_equation():
    parsed = parse_ode("dv/dt = (I - v) / tau")
    assert parsed.state_var == "v"
    assert set(parsed.param_names) == {"I", "tau"}


def test_current_symbol_I_is_not_confused_with_imaginary_unit():
    """Regression test: sympy treats bare 'I' as the imaginary unit by
    default. The parser must override this so 'I' means input current."""
    parsed = parse_ode("dv/dt = (I - v) / tau")
    result = parsed.derivative_fn(jnp.array(0.0), I=jnp.array(2.0), tau=jnp.array(1.0))
    assert jnp.isclose(result, 2.0)  # (2 - 0) / 1 = 2, not a complex number


@pytest.mark.parametrize(
    "bad_eq",
    [
        "v = I * R",  # not an ODE
        "dv/dt = (20*ms - v) / tau",  # units
        "dv/dt = ",  # empty RHS
    ],
)
def test_rejects_out_of_scope_equations(bad_eq):
    with pytest.raises(ParserScopeError):
        parse_ode(bad_eq)


def test_euler_integrate_converges_to_steady_state():
    """dv/dt = (I - v)/tau should converge to v = I at steady state."""
    parsed = parse_ode("dv/dt = (I - v) / tau")
    trace = euler_integrate(
        parsed, initial_value=0.0, dt=1e-3, n_steps=5000, params_trace={"I": 1.0, "tau": 0.02}
    )
    assert jnp.isclose(trace[-1], 1.0, atol=1e-3)


def test_neuron_group_matches_hand_written_lif():
    """Week 5 deliverable: reimplement the Week 1-4 network with the new
    parser + NeuronGroup API, and report max-abs-error / RMS-error vs the
    original hand-written `snnkit.core.neuron` path — not just 'matches'."""
    n = 20
    n_steps = 2000
    tau, v_th, v_reset, dt = 20e-3, 1.0, 0.0, 1e-3

    key = jax.random.PRNGKey(0)
    i_trace = jax.random.uniform(key, (n_steps, n), minval=0.5, maxval=2.0)

    # Hand-written path (Week 1).
    hw_params = LIFParams(tau=tau, v_th=v_th, v_reset=v_reset, dt=dt)
    v_hw, spikes_hw = simulate_lif(i_trace, hw_params)

    # Parser/API path (Week 5).
    group = make_lif_group(n=n, tau=tau, v_th=v_th, v_reset=v_reset, dt=dt)
    v_api, spikes_api = group.simulate(params_trace={"I": i_trace, "tau": tau}, n_steps=n_steps)

    max_abs_error = float(jnp.max(jnp.abs(v_hw - v_api)))
    rms_error = float(jnp.sqrt(jnp.mean((v_hw - v_api) ** 2)))
    spikes_match = bool(jnp.array_equal(spikes_hw, spikes_api))

    print(
        f"max_abs_error={max_abs_error:.2e}  rms_error={rms_error:.2e}  spikes_match={spikes_match}"
    )

    assert spikes_match, "spike trains diverged between hand-written and parser paths"
    assert max_abs_error < 1e-5
    assert rms_error < 1e-6
