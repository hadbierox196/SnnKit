import jax.numpy as jnp

from snnkit.core.synapses import SparseWeights, inject_current


def test_hand_verifiable_small_network():
    """5-neuron network with hand-picked weights; every output current is
    checked against a value computed by hand.

    Topology:
        0 -> 1  (w = 0.5)
        0 -> 2  (w = -0.3)
        1 -> 2  (w = 1.0)
        1 -> 3  (w = 0.2)
        3 -> 4  (w = -1.0)
        4 -> 4  (self, w = 0.1)  -- exercises a self-loop
    """
    weights = SparseWeights(
        pre_idx=jnp.array([0, 0, 1, 1, 3, 4]),
        post_idx=jnp.array([1, 2, 2, 3, 4, 4]),
        weight=jnp.array([0.5, -0.3, 1.0, 0.2, -1.0, 0.1]),
        n_pre=5,
        n_post=5,
    )

    # Case A: neurons 0 and 1 fire.
    spikes = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0])
    current = inject_current(spikes, weights)
    expected = jnp.array(
        [
            0.0,  # neuron 0: no incoming synapses
            0.5,  # neuron 1: from 0 (0.5)
            -0.3 + 1.0,  # neuron 2: from 0 (-0.3) + from 1 (1.0) = 0.7
            0.2,  # neuron 3: from 1 (0.2)
            0.0,  # neuron 4: from 3 (0, since 3 didn't fire) + from 4 (0, didn't fire)
        ]
    )
    assert jnp.allclose(current, expected)

    # Case B: neurons 3 and 4 fire (exercises the self-loop and negative weight).
    spikes_b = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    current_b = inject_current(spikes_b, weights)
    expected_b = jnp.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0 + 0.1,  # neuron 4: from 3 (-1.0) + self (0.1) = -0.9
        ]
    )
    assert jnp.allclose(current_b, expected_b)


def test_no_spikes_gives_zero_current():
    weights = SparseWeights(
        pre_idx=jnp.array([0, 1]),
        post_idx=jnp.array([1, 2]),
        weight=jnp.array([1.0, -1.0]),
        n_pre=3,
        n_post=3,
    )
    spikes = jnp.zeros(3)
    current = inject_current(spikes, weights)
    assert jnp.allclose(current, jnp.zeros(3))


def test_batched_current_injection():
    """inject_current should work over an arbitrary leading batch shape."""
    weights = SparseWeights(
        pre_idx=jnp.array([0]),
        post_idx=jnp.array([1]),
        weight=jnp.array([2.0]),
        n_pre=2,
        n_post=2,
    )
    spikes_batch = jnp.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])  # [batch=3, n_pre=2]
    current = inject_current(spikes_batch, weights)
    expected = jnp.array([[0.0, 2.0], [0.0, 0.0], [0.0, 2.0]])
    assert jnp.allclose(current, expected)
