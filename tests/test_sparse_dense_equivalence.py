import jax
import jax.numpy as jnp
import pytest

from snnkit.core.spikes import dense_to_sparse, sparse_to_dense


@pytest.mark.parametrize("firing_prob", [0.01, 0.05, 0.2])
def test_dense_sparse_roundtrip_exact(firing_prob):
    """Converting dense -> sparse -> dense should exactly reproduce the
    original spike tensor, given a sufficiently large max_events buffer."""
    key = jax.random.PRNGKey(0)
    shape = (50, 4, 200)  # time, batch, neurons
    dense = (jax.random.uniform(key, shape) < firing_prob).astype(jnp.float32)

    n_true_spikes = int(dense.sum())
    sparse = dense_to_sparse(dense, max_events=n_true_spikes + 10)
    recon = sparse_to_dense(sparse, shape)

    assert jnp.array_equal(recon, dense)
    assert int(sparse.num_events) == n_true_spikes


def test_sparse_truncates_gracefully_when_buffer_too_small():
    """When max_events is smaller than the true spike count, dense_to_sparse
    should not crash — it silently truncates to the buffer size."""
    key = jax.random.PRNGKey(1)
    shape = (20, 2, 50)
    dense = (jax.random.uniform(key, shape) < 0.3).astype(jnp.float32)
    n_true = int(dense.sum())
    assert n_true > 5, "test setup should produce enough spikes to truncate"

    sparse = dense_to_sparse(dense, max_events=5)
    assert int(sparse.num_events) == 5


def test_population_population_dense_equivalence_via_population_sim():
    """The population simulator's dense output round-trips through the
    sparse representation, exercising the two Week-2 pieces together."""
    from snnkit.core.neuron import LIFParams
    from snnkit.core.population import simulate_population

    key = jax.random.PRNGKey(2)
    params = LIFParams(tau=20e-3, v_th=1.0, dt=1e-3)
    i_trace = jax.random.uniform(key, (300, 3, 10), minval=0.5, maxval=1.5)
    _, spike_trace = simulate_population(i_trace, params)

    n_true = int(spike_trace.sum())
    sparse = dense_to_sparse(spike_trace, max_events=n_true + 5)
    recon = sparse_to_dense(sparse, spike_trace.shape)
    assert jnp.array_equal(recon, spike_trace)
