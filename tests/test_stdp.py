"""Week 9 deliverable: isolate a single synapse, drive pre/post spikes at
controlled time offsets, and confirm the resulting weight change matches
the theoretical STDP curve — reporting R^2 of fit, not just "looks right".
"""

import jax.numpy as jnp
import numpy as np
import pytest

from snnkit.core.synapses import SparseWeights
from snnkit.training.stdp import STDPParams, init_stdp_state, stdp_step, stdp_weight_change_curve

R2_THRESHOLD = 0.95


def _measure_weight_change(delta_steps: int, params: STDPParams, w0: float = 0.5) -> float:
    """Simulate a single synapse with one presynaptic spike and one
    postsynaptic spike offset by `delta_steps` timesteps; return the net
    weight change."""
    weights = SparseWeights(
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]), weight=jnp.array([w0]), n_pre=1, n_post=1
    )
    state = init_stdp_state(1, 1)
    t_pre = 80
    t_post = t_pre + delta_steps
    time_steps = 200
    assert 0 <= t_post < time_steps, "test offsets must stay within the simulated window"

    for t in range(time_steps):
        spikes_pre = jnp.array([1.0 if t == t_pre else 0.0])
        spikes_post = jnp.array([1.0 if t == t_post else 0.0])
        state, weights = stdp_step(state, weights, spikes_pre, spikes_post, params)
    return float(weights.weight[0] - w0)


def test_stdp_weight_change_matches_theoretical_curve_r2():
    params = STDPParams(tau_plus=20e-3, tau_minus=20e-3, a_plus=0.01, a_minus=0.012, dt=1e-3)

    delta_steps_range = [d for d in range(-60, 61, 3) if d != 0]
    measured, theoretical = [], []
    for d in delta_steps_range:
        measured.append(_measure_weight_change(d, params))
        theoretical.append(float(stdp_weight_change_curve(jnp.array(d * params.dt), params)))

    measured_arr = np.array(measured)
    theoretical_arr = np.array(theoretical)

    ss_res = np.sum((measured_arr - theoretical_arr) ** 2)
    ss_tot = np.sum((measured_arr - measured_arr.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"STDP simulated-vs-theoretical curve fit: R^2 = {r2:.4f}")
    assert r2 > R2_THRESHOLD, f"STDP curve fit R^2={r2:.4f} below threshold {R2_THRESHOLD}"


def test_ltp_dominates_for_pre_before_post():
    """Sanity check on the sign convention: pre-before-post (delta_t > 0)
    should potentiate (positive dw); post-before-pre should depress."""
    params = STDPParams()
    dw_ltp = _measure_weight_change(delta_steps=20, params=params)  # post after pre
    dw_ltd = _measure_weight_change(delta_steps=-20, params=params)  # post before pre
    assert dw_ltp > 0, "pre-before-post should potentiate the synapse"
    assert dw_ltd < 0, "post-before-pre should depress the synapse"


@pytest.mark.parametrize("w_min,w_max", [(0.0, 1.0)])
def test_weights_respect_bounds(w_min, w_max):
    """Repeated potentiation should saturate at w_max, not overshoot."""
    params = STDPParams(a_plus=0.5, a_minus=0.0, w_min=w_min, w_max=w_max)
    weights = SparseWeights(
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]), weight=jnp.array([0.9]), n_pre=1, n_post=1
    )
    state = init_stdp_state(1, 1)
    for _ in range(50):
        state, weights = stdp_step(state, weights, jnp.array([1.0]), jnp.array([1.0]), params)
    assert float(weights.weight[0]) <= w_max + 1e-6
