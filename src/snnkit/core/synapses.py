"""Sparse synaptic connectivity and current injection (Week 3).

No delays yet — every spike affects its targets on the *next* timestep
(1-step latency, unavoidable in a discrete-time simulator). Transmission
delays beyond that are added in `snnkit.core.delays` (Week 4).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class SparseWeights(NamedTuple):
    """COO-format sparse synaptic weight matrix from `n_pre` to `n_post` neurons.

    Attributes:
        pre_idx: `[n_synapses]` int32, presynaptic neuron index.
        post_idx: `[n_synapses]` int32, postsynaptic neuron index.
        weight: `[n_synapses]` float32, synaptic weight (signed: negative
            for inhibitory synapses).
        n_pre: number of presynaptic neurons (static).
        n_post: number of postsynaptic neurons (static).
    """

    pre_idx: jax.Array
    post_idx: jax.Array
    weight: jax.Array
    n_pre: int
    n_post: int


def random_sparse_weights(
    key: jax.Array,
    n_pre: int,
    n_post: int,
    connection_prob: float,
    weight_scale: float = 1.0,
    seed_self_connections: bool = True,
) -> SparseWeights:
    """Build a random Erdos-Renyi sparse weight matrix.

    Args:
        key: JAX PRNG key.
        n_pre: number of presynaptic neurons.
        n_post: number of postsynaptic neurons.
        connection_prob: probability that a given (pre, post) pair is connected.
        weight_scale: weights drawn as `N(0, weight_scale)`.
        seed_self_connections: if `False` and `n_pre == n_post`, self-connections
            (`pre_idx == post_idx`) are removed (useful for recurrent layers).

    Returns:
        `SparseWeights`.
    """
    key_mask, key_w = jax.random.split(key)
    mask = jax.random.bernoulli(key_mask, p=connection_prob, shape=(n_pre, n_post))
    if not seed_self_connections and n_pre == n_post:
        mask = mask & ~jnp.eye(n_pre, n_post, dtype=bool)
    pre_idx, post_idx = jnp.nonzero(mask)
    weight = jax.random.normal(key_w, shape=pre_idx.shape) * weight_scale
    return SparseWeights(
        pre_idx=pre_idx.astype(jnp.int32),
        post_idx=post_idx.astype(jnp.int32),
        weight=weight,
        n_pre=n_pre,
        n_post=n_post,
    )


def inject_current(spikes_pre: jax.Array, weights: SparseWeights) -> jax.Array:
    """Compute postsynaptic input current from presynaptic spikes.

    Args:
        spikes_pre: `[..., n_pre]` — presynaptic spikes (any leading batch
            shape), 0/1.
        weights: `SparseWeights` from `n_pre` to `n_post`.

    Returns:
        `[..., n_post]` postsynaptic current: for each postsynaptic neuron,
        the sum of `weight * spike` over all incoming synapses.
    """
    # Gather presynaptic spike value for each synapse: [..., n_synapses]
    pre_spike_per_synapse = jnp.take(spikes_pre, weights.pre_idx, axis=-1)
    contribution = pre_spike_per_synapse * weights.weight  # [..., n_synapses]

    leading_shape = spikes_pre.shape[:-1]
    out = jnp.zeros(leading_shape + (weights.n_post,), dtype=spikes_pre.dtype)
    # Scatter-add each synapse's contribution into its postsynaptic target.
    out = out.at[..., weights.post_idx].add(contribution)
    return out
