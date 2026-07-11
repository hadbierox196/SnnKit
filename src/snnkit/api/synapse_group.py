"""`SynapseGroup`: object API bundling a `SparseWeights` matrix with an
optional per-synapse delay, wrapping `snnkit.core.synapses` /
`snnkit.core.delays`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from snnkit.core.delays import DelayedWeights
from snnkit.core.synapses import SparseWeights, inject_current, random_sparse_weights


@dataclass
class SynapseGroup:
    """Connects a `NeuronGroup` of size `n_pre` to one of size `n_post`.

    If `delay_steps` is `None`, current injection is instantaneous
    (1-step latency inherent to discrete-time simulation, no additional
    delay). Otherwise, current is delivered via the ring-buffer mechanism.
    """

    weights: SparseWeights
    delay_steps: jax.Array | None = None

    @classmethod
    def random(
        cls,
        key: jax.Array,
        n_pre: int,
        n_post: int,
        connection_prob: float,
        weight_scale: float = 1.0,
        delay_steps: jax.Array | int | None = None,
        seed_self_connections: bool = True,
    ) -> SynapseGroup:
        """Convenience constructor: random sparse connectivity, optionally
        with a fixed or per-synapse-random delay."""
        weights = random_sparse_weights(
            key, n_pre, n_post, connection_prob, weight_scale, seed_self_connections
        )
        if isinstance(delay_steps, int):
            delay_steps = jnp.full(weights.pre_idx.shape, delay_steps, dtype=jnp.int32)
        return cls(weights=weights, delay_steps=delay_steps)

    @property
    def has_delay(self) -> bool:
        return self.delay_steps is not None

    def as_delayed_weights(self) -> DelayedWeights:
        if not self.has_delay:
            raise ValueError(
                "This SynapseGroup has no delays configured; use `inject_current` directly "
                "instead of the ring-buffer path."
            )
        return DelayedWeights(weights=self.weights, delay_steps=self.delay_steps)

    def instantaneous_current(self, spikes_pre: jax.Array) -> jax.Array:
        """Compute postsynaptic current with no additional delay (Week 3 path)."""
        return inject_current(spikes_pre, self.weights)
