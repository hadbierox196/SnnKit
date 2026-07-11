import jax.numpy as jnp
import pytest

from snnkit.core.neuron import LIFParams, analytical_firing_rate, simulate_lif


@pytest.mark.parametrize("i_in", [1.2, 1.5, 2.0, 3.0])
def test_lif_firing_rate_matches_analytical(i_in):
    """Simulated firing rate under constant input should match the
    closed-form LIF firing-rate formula within tolerance."""
    params = LIFParams(tau=20e-3, v_th=1.0, v_reset=0.0, r=1.0, dt=1e-4)
    time_steps = 200_000  # 20 s of simulated time at dt=0.1ms, for a tight estimate
    i_trace = jnp.full((time_steps,), i_in)

    _, spikes = simulate_lif(i_trace, params)
    duration_s = time_steps * params.dt
    sim_rate = spikes.sum() / duration_s

    analytic_rate = analytical_firing_rate(i_in, params)

    # Euler discretization + finite-duration estimate both introduce small
    # error; 5% relative tolerance is tight enough to catch real bugs but
    # loose enough to not be flaky from discretization alone.
    assert jnp.isclose(sim_rate, analytic_rate, rtol=0.05), (
        f"sim_rate={float(sim_rate):.3f} Hz vs analytic={float(analytic_rate):.3f} Hz "
        f"for I={i_in}"
    )


def test_subthreshold_current_never_fires():
    """Current below threshold should produce zero spikes and zero analytical rate."""
    params = LIFParams(tau=20e-3, v_th=1.0, v_reset=0.0, r=1.0, dt=1e-3)
    i_trace = jnp.full((5000,), 0.5)
    _, spikes = simulate_lif(i_trace, params)
    assert float(spikes.sum()) == 0.0
    assert float(analytical_firing_rate(0.5, params)) == 0.0
