import jax
import jax.numpy as jnp
import pytest

from snnkit.core.delays import (
    DelayedWeights,
    init_ring_buffer,
    ring_buffer_step,
    simulate_with_delays,
)
from snnkit.core.neuron import LIFParams, lif_euler_step
from snnkit.core.synapses import SparseWeights, random_sparse_weights


@pytest.mark.parametrize("delay", [1, 3, 7])
def test_exact_delay_timing_single_synapse(delay):
    """A single spike through a single synapse with delay `d` must arrive
    at the postsynaptic neuron exactly `d` timesteps later — not d-1, not d+1."""
    weights = SparseWeights(
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]), weight=jnp.array([2.0]), n_pre=1, n_post=1
    )
    delayed = DelayedWeights(weights=weights, delay_steps=jnp.array([delay]))

    time_steps = 20
    spike_time = 5
    spike_trace = jnp.zeros((time_steps, 1)).at[spike_time, 0].set(1.0)

    current_trace = simulate_with_delays(
        spike_trace, delayed, n_post=1, max_delay_steps=max(delay, 1) + 2
    )
    nonzero = jnp.nonzero(current_trace[:, 0])[0]

    assert len(nonzero) == 1, f"expected exactly one delivery event, got {len(nonzero)}"
    assert int(nonzero[0]) == spike_time + delay
    assert jnp.isclose(current_trace[spike_time + delay, 0], 2.0)


def test_multiple_delays_dont_interfere():
    """Two synapses from the same presynaptic neuron with different delays
    should each deliver independently at their own arrival time."""
    weights = SparseWeights(
        pre_idx=jnp.array([0, 0]),
        post_idx=jnp.array([0, 1]),
        weight=jnp.array([1.0, -1.0]),
        n_pre=1,
        n_post=2,
    )
    delayed = DelayedWeights(weights=weights, delay_steps=jnp.array([2, 5]))

    time_steps = 15
    spike_trace = jnp.zeros((time_steps, 1)).at[3, 0].set(1.0)
    current_trace = simulate_with_delays(spike_trace, delayed, n_post=2, max_delay_steps=6)

    assert jnp.isclose(current_trace[3 + 2, 0], 1.0)
    assert jnp.isclose(current_trace[3 + 5, 1], -1.0)
    # nothing else should be nonzero
    mask = jnp.ones_like(current_trace, dtype=bool)
    mask = mask.at[3 + 2, 0].set(False).at[3 + 5, 1].set(False)
    assert jnp.allclose(current_trace[mask], 0.0)


def test_100_neuron_recurrent_network_is_stable():
    """A ~100-neuron sparse recurrent network with delayed synapses should
    settle into a non-silent, non-runaway firing regime.

    Parameters here (connection_prob, weight_scale, drive range) were tuned
    by hand per the Week 4 task; this test locks in that a network built
    with these parameters stays in a sane regime, guarding against future
    regressions in the delay/current-injection pipeline.
    """
    n = 100
    key = jax.random.PRNGKey(0)
    key, k_w, k_delay, k_drive = jax.random.split(key, 4)

    weights = random_sparse_weights(
        k_w, n, n, connection_prob=0.05, weight_scale=0.25, seed_self_connections=False
    )
    delay_steps = jax.random.randint(k_delay, weights.pre_idx.shape, 1, 4)
    delayed = DelayedWeights(weights=weights, delay_steps=delay_steps)

    params = LIFParams(tau=20e-3, v_th=1.0, dt=1e-3)
    time_steps = 1000
    ext_current = jax.random.uniform(k_drive, (time_steps, n), minval=0.9, maxval=1.3)

    ring_state0 = init_ring_buffer(n, max_delay_steps=5)

    def full_step(carry, t):
        v, ring_state, spikes_prev = carry
        ring_state, delayed_current = ring_buffer_step(ring_state, t, spikes_prev, delayed)
        total_current = ext_current[t] + delayed_current
        v_next, spiked = lif_euler_step(v, total_current, params)
        return (v_next, ring_state, spiked), spiked

    init_carry = (jnp.zeros(n), ring_state0, jnp.zeros(n))
    _, spike_trace = jax.lax.scan(full_step, init_carry, jnp.arange(time_steps))

    rates = spike_trace.sum(axis=0) / (time_steps * params.dt)

    # Non-silent: every neuron fires at least a handful of times.
    assert float(rates.min()) > 0.0, "network went silent"
    # Non-runaway: no neuron is firing anywhere near every timestep
    # (dt=1ms => max possible rate 1000 Hz; runaway would approach that).
    assert float(rates.max()) < 200.0, "network is firing away (runaway regime)"
    # Sanity: population mean rate in a biologically plausible ballpark.
    assert 1.0 < float(rates.mean()) < 100.0
