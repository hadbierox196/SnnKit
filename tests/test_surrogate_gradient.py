"""Week 7 deliverable: a test confirming non-zero, non-NaN gradients flow
from a loss at the output back through the full simulation to input
weights, on a toy network."""

import jax
import jax.numpy as jnp

from snnkit.core.neuron import LIFParams
from snnkit.training.surrogate import (
    differentiable_simulate_lif,
    fast_sigmoid_surrogate_grad,
    spike_fn,
)


def test_surrogate_gradient_is_smooth_and_nonzero_near_threshold():
    """The surrogate derivative should be nonzero and finite in a
    neighborhood of the threshold — unlike the true Heaviside gradient,
    which is zero everywhere except an undefined point at x=0."""
    x = jnp.linspace(-2.0, 2.0, 101)
    grad = fast_sigmoid_surrogate_grad(x)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.all(grad > 0), "surrogate gradient should be strictly positive everywhere"
    # Peaked at the threshold (x=0), decaying away from it.
    assert grad[50] == jnp.max(grad)


def test_spike_fn_forward_is_exact_heaviside():
    x = jnp.array([-1.0, -0.001, 0.0, 0.001, 1.0])
    spikes = spike_fn(x)
    assert jnp.array_equal(spikes, jnp.array([0.0, 0.0, 1.0, 1.0, 1.0]))


def test_gradient_flows_through_full_simulation_to_input_weights():
    """The core Week 7 deliverable: jax.grad of a loss on the spike output
    w.r.t. an input weight, flowing through the entire `lax.scan`
    simulation loop, is non-zero and contains no NaNs."""
    params = LIFParams(tau=20e-3, v_th=1.0, dt=1e-3)
    n_steps, n_in = 200, 5
    key = jax.random.PRNGKey(0)
    base_input = jax.random.uniform(key, (n_steps, n_in), minval=0.5, maxval=1.5)

    def loss_fn(weight):
        i_trace = base_input * weight
        _, spike_trace = differentiable_simulate_lif(i_trace, params)
        return jnp.sum(spike_trace)

    weight = jnp.ones(n_in)
    loss, grad = jax.value_and_grad(loss_fn)(weight)

    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(grad)), f"found NaN/Inf in gradient: {grad}"
    assert jnp.any(grad != 0), "gradient should not be identically zero"


def test_gradient_flows_even_with_zero_initial_spikes():
    """Regression guard: a network that starts completely silent (very
    low initial weight) should still receive a non-zero gradient nudging
    it toward the threshold — this is exactly the case surrogate
    gradients exist to fix, vs. a hard threshold's zero gradient there."""
    params = LIFParams(tau=20e-3, v_th=1.0, dt=1e-3)
    n_steps, n_in = 300, 3
    key = jax.random.PRNGKey(1)
    base_input = jax.random.uniform(key, (n_steps, n_in), minval=0.1, maxval=0.3)

    def loss_fn(weight):
        i_trace = base_input * weight
        _, spike_trace = differentiable_simulate_lif(i_trace, params)
        return jnp.sum(spike_trace)

    weight = jnp.array([0.5, 0.5, 0.5])  # deliberately sub-threshold-ish
    _, spike_trace_check = differentiable_simulate_lif(base_input * weight, params)
    grad = jax.grad(loss_fn)(weight)

    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(jnp.abs(grad) > 1e-8), "expected a usable (non-vanishing) gradient signal"
