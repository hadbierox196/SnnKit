"""Week 8 deliverable: BPTT training loop trains a small SNN classifier
via optax, with loss decreasing and accuracy improving over a fixed,
deterministic (seeded) number of steps."""

import jax
import optax

from snnkit.training.bptt import (
    TRAINING_LIF_PARAMS,
    batched_loss_and_acc,
    init_mlp_snn_params,
    synthetic_shd_like_dataset,
    train_step,
)


def test_bptt_loss_decreases_and_accuracy_improves():
    key = jax.random.PRNGKey(0)
    k_data, k_params = jax.random.split(key)
    n_classes, n_channels, n_timesteps = 10, 50, 50

    spikes, labels = synthetic_shd_like_dataset(
        k_data,
        n_samples=64,
        n_classes=n_classes,
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        firing_prob=0.15,
    )

    params = init_mlp_snn_params(k_params, n_in=n_channels, n_hidden=64, n_out=n_classes)
    optimizer = optax.adam(5e-3)
    opt_state = optimizer.init(params)

    initial_loss, initial_acc = batched_loss_and_acc(params, spikes, labels, TRAINING_LIF_PARAMS)

    n_steps = 40
    for _ in range(n_steps):
        params, opt_state, loss = train_step(
            params, opt_state, optimizer, spikes, labels, TRAINING_LIF_PARAMS
        )

    final_loss, final_acc = batched_loss_and_acc(params, spikes, labels, TRAINING_LIF_PARAMS)

    assert final_loss < initial_loss, (
        f"expected loss to decrease over {n_steps} steps: "
        f"initial={float(initial_loss):.3f} final={float(final_loss):.3f}"
    )
    assert final_acc > initial_acc, (
        f"expected accuracy to improve over {n_steps} steps: "
        f"initial={float(initial_acc):.3f} final={float(final_acc):.3f}"
    )
    # Chance level for 10 classes is 10%; a working training loop should
    # clear that comfortably within 40 steps on this easy synthetic task.
    assert final_acc > 0.3, f"expected well above chance-level accuracy, got {float(final_acc):.3f}"


def test_synthetic_dataset_shape_and_labels():
    key = jax.random.PRNGKey(0)
    spikes, labels = synthetic_shd_like_dataset(
        key, n_samples=10, n_classes=20, n_channels=700, n_timesteps=30
    )
    assert spikes.shape == (10, 30, 700)
    assert labels.shape == (10,)
    assert int(labels.min()) >= 0
    assert int(labels.max()) < 20


def test_shared_templates_give_a_consistent_task_across_splits():
    """Regression test: train/test splits generated with independently
    derived templates would silently be *different, unrelated tasks*
    (this repo hit exactly this bug once). Passing the same `templates`
    to both calls must make both splits share the same class->channel
    signature, verified directly rather than just via downstream accuracy."""
    from snnkit.training.bptt import make_shd_templates

    n_classes, n_channels = 5, 30
    key = jax.random.PRNGKey(0)
    k_templates, k_train, k_test = jax.random.split(key, 3)
    templates = make_shd_templates(k_templates, n_classes, n_channels)

    train_spikes, train_labels = synthetic_shd_like_dataset(
        k_train,
        n_samples=200,
        n_classes=n_classes,
        n_channels=n_channels,
        n_timesteps=50,
        templates=templates,
    )
    test_spikes, test_labels = synthetic_shd_like_dataset(
        k_test,
        n_samples=200,
        n_classes=n_classes,
        n_channels=n_channels,
        n_timesteps=50,
        templates=templates,
    )

    # Without a shared task, there's no reason class-average channel
    # activity would correlate between splits. With shared templates, the
    # per-class mean channel-activity pattern should be strongly correlated
    # between train and test (same signature channels drive both).
    import jax.numpy as jnp
    import numpy as np

    def class_channel_profile(spikes, labels):
        return jnp.stack([spikes[labels == c].mean(axis=(0, 1)) for c in range(n_classes)])

    train_profile = np.asarray(class_channel_profile(train_spikes, train_labels))
    test_profile = np.asarray(class_channel_profile(test_spikes, test_labels))
    corr = np.corrcoef(train_profile.flatten(), test_profile.flatten())[0, 1]
    assert (
        corr > 0.8
    ), f"expected strongly correlated class signatures across splits, got r={corr:.3f}"
