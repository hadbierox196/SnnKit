"""Surrogate-gradient BPTT training loop, using `optax`.

Trains a small feedforward spiking network (one recurrent-free hidden
layer of LIF neurons) on a classification task, via backprop-through-time
through `snnkit.training.surrogate.differentiable_lif_step`.

Dataset note: this trains on `synthetic_shd_like_dataset` below — a
synthetic, SHD-*shaped* (Spiking Heidelberg Digits: spike-encoded audio,
700 input channels, 20 classes) dataset generated locally, NOT the real
SHD dataset. The real SHD dataset requires downloading from
https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/, which
this environment has no network access to. `docs/local-vs-bptt.md`
reports results on this synthetic stand-in and says so explicitly — swap
in `tonic.datasets.SHD` (see that doc) to reproduce on the real dataset.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from snnkit.core.neuron import LIFParams
from snnkit.training.surrogate import differentiable_lif_step


class MLPSNNParams(NamedTuple):
    """Weights for a 2-layer feedforward spiking network: input -> hidden
    (spiking) -> output (readout, non-spiking membrane potential)."""

    w_in_hidden: jax.Array  # [n_in, n_hidden]
    w_hidden_out: jax.Array  # [n_hidden, n_out]


def init_mlp_snn_params(key: jax.Array, n_in: int, n_hidden: int, n_out: int) -> MLPSNNParams:
    k1, k2 = jax.random.split(key)
    # Scale tuned empirically (see docs/local-vs-bptt.md) so the hidden
    # layer sits in a usable spiking regime (neither silent nor saturated)
    # against LIFParams(tau=10ms, v_th=0.3) used by this module's training
    # loop — LIF's default v_th=1.0/tau=20ms is too conservative for
    # single-timestep binary spike inputs to reliably cross threshold.
    w_in_hidden = jax.random.normal(k1, (n_in, n_hidden)) / jnp.sqrt(n_in) * 8.0
    w_hidden_out = jax.random.normal(k2, (n_hidden, n_out)) / jnp.sqrt(n_hidden)
    return MLPSNNParams(w_in_hidden=w_in_hidden, w_hidden_out=w_hidden_out)


#: LIF params tuned (empirically, see docs/local-vs-bptt.md) to give a
#: usable spiking regime for this module's networks specifically — not a
#: general-purpose default the way `LIFParams()` in `snnkit.core.neuron` is.
TRAINING_LIF_PARAMS = LIFParams(tau=10e-3, v_th=0.3, v_reset=0.0, dt=1e-3)


def forward(
    params: MLPSNNParams, input_spikes: jax.Array, lif_params: LIFParams
) -> tuple[jax.Array, jax.Array]:
    """Run the network forward over time.

    Args:
        params: `MLPSNNParams`.
        input_spikes: `[time, n_in]` binary input spike train.
        lif_params: `LIFParams` for the hidden layer.

    Returns:
        (readout_trace, hidden_spike_trace): `readout_trace` is
        `[time, n_out]`, the (non-spiking, leaky-integrated) output-layer
        membrane potential at each step — used as class-logits after
        summing/averaging over time. `hidden_spike_trace` is
        `[time, n_hidden]`, for logging/regularization (e.g. spike-count
        penalties).
    """
    n_hidden = params.w_in_hidden.shape[1]
    n_out = params.w_hidden_out.shape[1]

    # Readout layer: simple leaky integrator (no spiking, no reset) so the
    # output is a smooth function of hidden spikes throughout training.
    readout_tau = 20e-3

    def step(carry, x_t):
        v_hidden, v_out = carry
        i_hidden = x_t @ params.w_in_hidden  # [n_hidden]
        v_hidden_next, spikes_hidden = differentiable_lif_step(v_hidden, i_hidden, lif_params)

        i_out = spikes_hidden @ params.w_hidden_out  # [n_out]
        dv_out = (lif_params.dt / readout_tau) * (-v_out + i_out)
        v_out_next = v_out + dv_out

        return (v_hidden_next, v_out_next), (v_out_next, spikes_hidden)

    init_carry = (jnp.zeros(n_hidden), jnp.zeros(n_out))
    _, (readout_trace, hidden_spike_trace) = jax.lax.scan(step, init_carry, input_spikes)
    return readout_trace, hidden_spike_trace


def loss_fn(
    params: MLPSNNParams,
    input_spikes: jax.Array,
    label: jax.Array,
    lif_params: LIFParams,
    spike_reg_weight: float = 1e-3,
) -> tuple[jax.Array, jax.Array]:
    """Cross-entropy on the time-averaged readout, + a small spike-count
    regularizer (discourages runaway hidden-layer firing, standard
    practice in surrogate-gradient SNN training, e.g. Neftci et al. 2019).
    """
    readout_trace, hidden_spike_trace = forward(params, input_spikes, lif_params)
    logits = jnp.mean(readout_trace, axis=0)  # [n_out], time-averaged
    log_probs = jax.nn.log_softmax(logits)
    ce_loss = -log_probs[label]
    spike_reg = spike_reg_weight * jnp.mean(hidden_spike_trace)
    return ce_loss + spike_reg, logits


def batched_loss_and_acc(
    params: MLPSNNParams,
    input_batch: jax.Array,
    labels: jax.Array,
    lif_params: LIFParams,
) -> tuple[jax.Array, jax.Array]:
    """Vmap `loss_fn` over a batch. Returns (mean_loss, accuracy)."""

    def single(x, y):
        loss, logits = loss_fn(params, x, y, lif_params)
        return loss, logits

    losses, logits_batch = jax.vmap(single)(input_batch, labels)
    preds = jnp.argmax(logits_batch, axis=-1)
    acc = jnp.mean((preds == labels).astype(jnp.float32))
    return jnp.mean(losses), acc


def train_step(
    params: MLPSNNParams,
    opt_state,
    optimizer: optax.GradientTransformation,
    input_batch: jax.Array,
    labels: jax.Array,
    lif_params: LIFParams,
):
    def loss_only(p):
        loss, _ = batched_loss_and_acc(p, input_batch, labels, lif_params)
        return loss

    loss, grads = jax.value_and_grad(loss_only)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def make_shd_templates(key: jax.Array, n_classes: int, n_channels: int) -> jax.Array:
    """Fixed per-class channel "template" (which channels are the elevated-
    rate signature for each class), to be generated ONCE and reused across
    train/test splits — see `synthetic_shd_like_dataset`'s docstring for
    why this must be shared, not regenerated per split."""
    return jax.random.bernoulli(key, p=0.1, shape=(n_classes, n_channels))


def synthetic_shd_like_dataset(
    key: jax.Array,
    n_samples: int,
    n_classes: int = 20,
    n_channels: int = 700,
    n_timesteps: int = 100,
    firing_prob: float = 0.05,
    templates: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Generate a synthetic, SHD-*shaped* spike dataset (SHD: Spiking
    Heidelberg Digits — 700 input channels, 20 classes; see module
    docstring for why this is synthetic rather than the real dataset).

    Each class gets a fixed random "template" set of active channels;
    samples are Poisson-ish spike trains with elevated firing probability
    on the class's template channels and low background firing elsewhere
    — a crude but learnable stand-in for the real dataset's structure.

    Args:
        templates: pre-computed `[n_classes, n_channels]` template matrix
            from `make_shd_templates`. **Required when generating more
            than one split (train/test) of the same task** — if `None`,
            a fresh (different) template set is derived from `key`, which
            is only correct for a single, self-contained dataset call.
            Regenerating templates independently per split silently
            creates two *unrelated* tasks (train and test would use
            different class/channel mappings) — a real bug this project
            hit once already; don't reintroduce it.

    Returns:
        (spikes, labels): spikes `[n_samples, n_timesteps, n_channels]`,
        labels `[n_samples]` int in `[0, n_classes)`.
    """
    key_templates, key_labels, key_spikes = jax.random.split(key, 3)
    if templates is None:
        templates = make_shd_templates(key_templates, n_classes, n_channels)
    labels = jax.random.randint(key_labels, (n_samples,), 0, n_classes)

    background_rate = 0.01
    template_rate = firing_prob

    class_templates = templates[labels]  # [n_samples, n_channels]
    rate_map = jnp.where(class_templates, template_rate, background_rate)
    rate_map = jnp.broadcast_to(rate_map[:, None, :], (n_samples, n_timesteps, n_channels))
    spikes = jax.random.bernoulli(key_spikes, p=rate_map).astype(jnp.float32)
    return spikes, labels
