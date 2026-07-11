"""SuperSpike (Zenke & Ganguli 2018)-style local learning rule, Week 11.

De-risked per the roadmap: SuperSpike (simpler, well-established) rather
than jumping straight to full e-prop. **E-prop was not attempted** — noted
explicitly here as a stretch goal not reached, per the roadmap's
instruction to record this honestly either way.

**Scope note (simplified vs. the original paper):** this implements a
two-layer feedback-alignment-style variant: per-synapse eligibility traces
are a single-exponential presynaptic trace multiplied by the surrogate
derivative of the postsynaptic membrane potential (matching the paper's
core idea), and credit is assigned to the hidden layer via a *fixed random
feedback matrix* (feedback alignment, Lillicrap et al. 2016) rather than
the exact symmetric/optimal feedback — this keeps every weight update a
function of purely local, causal quantities (no backprop through time,
no backprop through layers), which is the property that actually matters
for the BPTT-vs-local comparison in Week 12. The original paper's
double-exponential trace and exact derivation are not reproduced exactly;
this is a deliberate simplification, not an oversight.

Target convention: classification is framed as regression to a fixed
per-class target held constant across the whole trial
(`target_k = 1` for the correct class, `0` otherwise) — the same
"regress the leaky-integrator readout toward a target" idea used
in Zenke's own SNN training tutorials, adapted here for a fully local
per-timestep error signal `error(t) = target - readout(t)`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from snnkit.core.neuron import LIFParams
from snnkit.training.surrogate import fast_sigmoid_surrogate_grad


class SuperSpikeParams(NamedTuple):
    w_in: jax.Array  # [n_channels, n_hidden]
    w_out: jax.Array  # [n_hidden, n_classes]


class SuperSpikeHyper(NamedTuple):
    lif_params: LIFParams
    feedback: jax.Array  # [n_classes, n_hidden], fixed (not trained)
    tau_trace: float = 20e-3
    tau_out: float = 20e-3
    eta_in: float = 1e-3
    eta_out: float = 1e-2


def init_superspike_params(
    key: jax.Array, n_channels: int, n_hidden: int, n_classes: int
) -> tuple[SuperSpikeParams, jax.Array]:
    """Init trainable weights + the fixed random feedback matrix.

    Scale tuned empirically (same reasoning as `bptt.init_mlp_snn_params`):
    the hidden layer needs to start in a firing regime that produces
    nonzero surrogate-derivative signal, or the eligibility trace is zero
    everywhere and nothing learns.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    w_in = jax.random.normal(k1, (n_channels, n_hidden)) / jnp.sqrt(n_channels) * 8.0
    w_out = jax.random.normal(k2, (n_hidden, n_classes)) / jnp.sqrt(n_hidden)
    feedback = jax.random.normal(k3, (n_classes, n_hidden)) / jnp.sqrt(n_classes)
    return SuperSpikeParams(w_in=w_in, w_out=w_out), feedback


def process_sample_and_update(
    params: SuperSpikeParams, hyper: SuperSpikeHyper, spikes_in: jax.Array, target: jax.Array
) -> tuple[SuperSpikeParams, jax.Array]:
    """Run one sample through the network, applying the local SuperSpike
    weight update online, per timestep, within the `lax.scan` itself (no
    `jax.grad` anywhere in this function — every update is a explicit,
    local, causal computation).

    Args:
        params: current `SuperSpikeParams`.
        hyper: `SuperSpikeHyper` (includes the fixed feedback matrix).
        spikes_in: `[time, n_channels]` input spike train.
        target: `[n_classes]` fixed target (e.g. one-hot label), held
            constant across the trial.

    Returns:
        (updated_params, readout_trace) where `readout_trace` is
        `[time, n_classes]` (for computing logits/accuracy — `max` over
        time is used elsewhere, matching `snnkit.training.bptt`'s
        convention for comparability).
    """
    lif = hyper.lif_params
    decay_trace = jnp.exp(-lif.dt / hyper.tau_trace)
    n_hidden = params.w_in.shape[1]
    n_channels = params.w_in.shape[0]
    n_classes = params.w_out.shape[1]

    def step(carry, x_t):
        v_hidden, v_out, tr_in, tr_hidden, w_in, w_out = carry

        i_hidden = x_t @ w_in
        dv = (lif.dt / lif.tau) * (-(v_hidden - lif.v_rest) + i_hidden)
        v_pre_reset = v_hidden + dv
        spiked = (v_pre_reset >= lif.v_th).astype(jnp.float32)
        surrogate = fast_sigmoid_surrogate_grad(v_pre_reset - lif.v_th)
        v_hidden_next = jnp.where(spiked > 0, lif.v_reset, v_pre_reset)

        i_out = spiked @ w_out
        v_out_next = v_out + (lif.dt / hyper.tau_out) * (-v_out + i_out)
        error = target - v_out_next  # [n_classes], purely local (no lookahead)

        # Traces updated with the PRE-update (this-step) presynaptic
        # activity, so the eligibility below uses "how active was this
        # synapse's input just before this update" — causal, no
        # information from the future leaks in.
        tr_in_next = tr_in * decay_trace + x_t
        tr_hidden_next = tr_hidden * decay_trace + spiked

        elig_in = tr_in[:, None] * surrogate[None, :]  # [n_channels, n_hidden]
        feedback_signal = error @ hyper.feedback  # [n_hidden], via FIXED feedback
        dw_in = hyper.eta_in * elig_in * feedback_signal[None, :]
        dw_out = hyper.eta_out * tr_hidden[:, None] * error[None, :]  # [n_hidden, n_classes]

        w_in_next = w_in + dw_in
        w_out_next = w_out + dw_out

        new_carry = (v_hidden_next, v_out_next, tr_in_next, tr_hidden_next, w_in_next, w_out_next)
        return new_carry, v_out_next

    init_carry = (
        jnp.zeros(n_hidden),
        jnp.zeros(n_classes),
        jnp.zeros(n_channels),
        jnp.zeros(n_hidden),
        params.w_in,
        params.w_out,
    )
    final_carry, readout_trace = lax.scan(step, init_carry, spikes_in)
    _, _, _, _, w_in_final, w_out_final = final_carry
    return SuperSpikeParams(w_in=w_in_final, w_out=w_out_final), readout_trace


def forward_no_update(
    params: SuperSpikeParams, lif: LIFParams, tau_out: float, spikes_in: jax.Array
) -> jax.Array:
    """Forward pass only (no weight updates) — used for evaluation."""
    n_hidden = params.w_in.shape[1]
    n_classes = params.w_out.shape[1]

    def step(carry, x_t):
        v_hidden, v_out = carry
        i_hidden = x_t @ params.w_in
        dv = (lif.dt / lif.tau) * (-(v_hidden - lif.v_rest) + i_hidden)
        v_pre_reset = v_hidden + dv
        spiked = (v_pre_reset >= lif.v_th).astype(jnp.float32)
        v_hidden_next = jnp.where(spiked > 0, lif.v_reset, v_pre_reset)
        i_out = spiked @ params.w_out
        v_out_next = v_out + (lif.dt / tau_out) * (-v_out + i_out)
        return (v_hidden_next, v_out_next), v_out_next

    init_carry = (jnp.zeros(n_hidden), jnp.zeros(n_classes))
    _, readout_trace = lax.scan(step, init_carry, spikes_in)
    return readout_trace


def accuracy(
    params: SuperSpikeParams, hyper: SuperSpikeHyper, spikes: jax.Array, labels: jax.Array
) -> jax.Array:
    def logits_for(x):
        trace = forward_no_update(params, hyper.lif_params, hyper.tau_out, x)
        return jnp.max(trace, axis=0)

    logits = jax.vmap(logits_for)(spikes)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean((preds == labels).astype(jnp.float32))


def train(
    key: jax.Array,
    train_spikes: jax.Array,
    train_labels: jax.Array,
    test_spikes: jax.Array,
    test_labels: jax.Array,
    n_channels: int,
    n_hidden: int,
    n_classes: int,
    lif_params: LIFParams,
    n_epochs: int = 15,
    eta_in: float = 1e-3,
    eta_out: float = 1e-2,
    verbose: bool = False,
) -> dict:
    """Full local-online training loop: one sample at a time (true to the
    "local, online" nature of the rule — no batching/averaging of updates
    across samples, unlike the BPTT loop's minibatch SGD)."""
    params, feedback = init_superspike_params(key, n_channels, n_hidden, n_classes)
    hyper = SuperSpikeHyper(
        lif_params=lif_params, feedback=feedback, eta_in=eta_in, eta_out=eta_out
    )
    targets = jax.nn.one_hot(train_labels, n_classes)

    process_jit = jax.jit(lambda p, x, t: process_sample_and_update(p, hyper, x, t))

    train_acc_history, test_acc_history = [], []
    n_train = train_spikes.shape[0]

    for epoch in range(n_epochs):
        for i in range(n_train):
            params, _ = process_jit(params, train_spikes[i], targets[i])

        train_acc = float(accuracy(params, hyper, train_spikes, train_labels))
        test_acc = float(accuracy(params, hyper, test_spikes, test_labels))
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        if verbose and (epoch % 3 == 0 or epoch == n_epochs - 1):
            print(f"epoch {epoch}: train_acc={train_acc:.3f} test_acc={test_acc:.3f}")

    return {
        "params": params,
        "hyper": hyper,
        "train_acc_history": train_acc_history,
        "test_acc_history": test_acc_history,
    }
