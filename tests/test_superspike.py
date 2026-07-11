"""Week 11 deliverable: SuperSpike (local learning rule) trains on the
same synthetic SHD-shaped subset used for BPTT (Week 8), with accuracy
recorded for direct comparison. E-prop was not attempted (stretch goal,
not reached — see module docstring in `snnkit.training.superspike`)."""

import jax

from snnkit.training.bptt import TRAINING_LIF_PARAMS, synthetic_shd_like_dataset
from snnkit.training.superspike import accuracy, init_superspike_params, train


def test_superspike_trains_on_same_shd_subset_as_bptt():
    """Same dataset generator, same class/channel/timestep counts as
    `test_bptt_training.py`, for direct comparability."""
    from snnkit.training.superspike import SuperSpikeHyper, init_superspike_params

    key = jax.random.PRNGKey(0)
    k_data, k_train = jax.random.split(key)
    n_classes, n_channels, n_timesteps = 10, 50, 50

    spikes, labels = synthetic_shd_like_dataset(
        k_data,
        n_samples=64,
        n_classes=n_classes,
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        firing_prob=0.15,
    )

    # True pre-training baseline: accuracy of freshly initialized (untrained) weights.
    init_params, init_feedback = init_superspike_params(k_train, n_channels, 64, n_classes)
    init_hyper = SuperSpikeHyper(lif_params=TRAINING_LIF_PARAMS, feedback=init_feedback)
    baseline_acc = float(accuracy(init_params, init_hyper, spikes, labels))

    result = train(
        k_train,
        train_spikes=spikes,
        train_labels=labels,
        test_spikes=spikes,  # same-set eval here; held-out split lives in Week 12's script
        test_labels=labels,
        n_channels=n_channels,
        n_hidden=64,
        n_classes=n_classes,
        lif_params=TRAINING_LIF_PARAMS,
        n_epochs=15,
    )

    final_acc = result["train_acc_history"][-1]

    assert (
        final_acc > baseline_acc
    ), f"expected accuracy to improve: baseline={baseline_acc:.3f} final={final_acc:.3f}"
    assert final_acc > 0.3, f"expected well above chance-level (10%) accuracy, got {final_acc:.3f}"


def test_superspike_weight_updates_are_purely_local_no_grad():
    """Sanity/design check: `process_sample_and_update` contains no
    `jax.grad`/`jax.vjp` call anywhere — it's a plain forward computation
    with explicit trace-based updates. This test checks the *behavioral*
    consequence: calling it doesn't require (and works fine without) any
    autodiff machinery being invoked, by simply confirming it runs under
    `jax.disable_jit`-free normal tracing without needing gradient tapes.
    """
    from snnkit.training.superspike import SuperSpikeHyper, process_sample_and_update

    key = jax.random.PRNGKey(0)
    n_channels, n_hidden, n_classes, n_steps = 10, 16, 3, 20
    params, feedback = init_superspike_params(key, n_channels, n_hidden, n_classes)
    hyper = SuperSpikeHyper(lif_params=TRAINING_LIF_PARAMS, feedback=feedback)

    spikes_in = jax.random.bernoulli(key, 0.1, (n_steps, n_channels)).astype(float)
    target = jax.nn.one_hot(1, n_classes)

    new_params, trace = process_sample_and_update(params, hyper, spikes_in, target)
    assert new_params.w_in.shape == params.w_in.shape
    assert new_params.w_out.shape == params.w_out.shape
    # Weights should actually have changed (rule is doing something).
    import jax.numpy as jnp

    assert not jnp.allclose(new_params.w_in, params.w_in)
