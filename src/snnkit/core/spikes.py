"""Sparse spike-index representation, per `docs/spike-tensor-spec.md`.

Padded (not ragged) layout: `batch_idx`, `neuron_idx`, `time_idx` are all
`[max_events]` int32 arrays, front-filled with real spike events and
back-filled with the sentinel `neuron_idx == -1` for unused slots. This
keeps shapes static under JIT.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

#: Sentinel value marking an unused (padding) slot in a `SparseSpikes` batch.
PAD_SENTINEL = -1


class SparseSpikes(NamedTuple):
    """Padded sparse spike-event representation. See spike-tensor-spec.md."""

    batch_idx: jax.Array  # [max_events], int32
    neuron_idx: jax.Array  # [max_events], int32 (PAD_SENTINEL where unused)
    time_idx: jax.Array  # [max_events], int32

    @property
    def num_events(self) -> jax.Array:
        """Number of real (non-padding) events."""
        return jnp.sum(self.neuron_idx != PAD_SENTINEL)


def dense_to_sparse(spike_trace: jax.Array, max_events: int | None = None) -> SparseSpikes:
    """Convert a dense `[time, batch, neurons]` 0/1 spike trace to `SparseSpikes`.

    Args:
        spike_trace: dense spike tensor, shape `[time, batch, neurons]`.
        max_events: static upper bound on event count for this call. If
            `None`, defaults to `time * batch * neurons` (fully dense
            worst case) — pass a tighter bound when expected sparsity is
            known, to save memory.

    Returns:
        `SparseSpikes` with arrays of shape `[max_events]`.
    """
    time_steps, batch, n_neurons = spike_trace.shape
    total = time_steps * batch * n_neurons
    if max_events is None:
        max_events = total

    flat = spike_trace.reshape(-1)
    # Static-shape top-k trick: argsort descending puts all "1" entries
    # first (stable enough since spike_trace is 0/1), then slice to
    # max_events. This keeps the whole operation JIT-friendly (no
    # dynamic-shape boolean masking).
    order = jnp.argsort(-flat, stable=True)[:max_events]
    is_real = flat[order] > 0

    t_idx, b_idx, n_idx = jnp.unravel_index(order, spike_trace.shape)
    neuron_idx = jnp.where(is_real, n_idx, PAD_SENTINEL).astype(jnp.int32)
    batch_idx = jnp.where(is_real, b_idx, PAD_SENTINEL).astype(jnp.int32)
    time_idx = jnp.where(is_real, t_idx, PAD_SENTINEL).astype(jnp.int32)
    return SparseSpikes(batch_idx=batch_idx, neuron_idx=neuron_idx, time_idx=time_idx)


def sparse_to_dense(sparse: SparseSpikes, shape: tuple[int, int, int]) -> jax.Array:
    """Inverse of `dense_to_sparse`: scatter events back into a dense tensor.

    Args:
        sparse: `SparseSpikes`.
        shape: `(time, batch, neurons)` of the output dense tensor.

    Returns:
        Dense 0/1 float32 tensor, shape `shape`.
    """
    dense = jnp.zeros(shape, dtype=jnp.float32)
    valid = sparse.neuron_idx != PAD_SENTINEL
    # Clamp padding indices to 0 so the scatter is in-bounds; `valid` masks
    # their contribution out via the update value below.
    t = jnp.where(valid, sparse.time_idx, 0)
    b = jnp.where(valid, sparse.batch_idx, 0)
    n = jnp.where(valid, sparse.neuron_idx, 0)
    updates = valid.astype(jnp.float32)
    dense = dense.at[t, b, n].add(updates)
    return jnp.clip(dense, 0.0, 1.0)
