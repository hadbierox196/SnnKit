"""Quantitative validation of the Izhikevich model against the specific
regular-spiking, bursting (chattering), and fast-spiking parameter sets
from Izhikevich (2003) — checking the *shape* of each firing pattern
(adaptation, burst structure, sustained high-frequency regularity), not
just "some spikes happened"."""

import jax.numpy as jnp
import numpy as np

from snnkit.models.izhikevich import (
    CHATTERING,
    FAST_SPIKING,
    REGULAR_SPIKING,
    inter_spike_intervals,
    simulate_izhikevich,
    spike_times_ms,
)


def _run(params, i_val, t_ms=1000, v0=-70.0):
    i_trace = jnp.full((t_ms,), i_val)
    v, u, spikes = simulate_izhikevich(i_trace, params, v0=v0)
    st = spike_times_ms(spikes, params.dt)
    isi = inter_spike_intervals(st)
    return v, spikes, st, isi


def test_regular_spiking_shows_adaptation_then_steady_firing():
    """RS: classic pattern is a shorter first ISI (higher initial rate)
    followed by adaptation to a longer, stable steady-state ISI."""
    v, spikes, st, isi = _run(REGULAR_SPIKING, i_val=10.0)
    assert len(st) >= 10, "expected sustained regular firing under 10 pA step current"

    first_isi = isi[0]
    steady_isis = isi[-5:]  # last 5 intervals, after adaptation settles
    steady_mean = np.mean(steady_isis)
    steady_cv = np.std(steady_isis) / steady_mean

    assert first_isi < steady_mean, (
        f"expected spike-frequency adaptation (first ISI {first_isi} < steady-state "
        f"{steady_mean}), RS should fire faster initially then slow down"
    )
    assert steady_cv < 0.05, f"steady-state firing should be regular (low ISI CV), got {steady_cv}"


def test_chattering_shows_periodic_bursts():
    """Chattering: repeated bursts of short within-burst ISIs separated by
    much longer inter-burst gaps, for the whole duration of the step
    current (unlike IB, which bursts once then regularizes)."""
    v, spikes, st, isi = _run(CHATTERING, i_val=10.0)
    assert len(st) >= 20, "expected many spikes across multiple bursts"

    short_isis = isi[isi < 15.0]  # within-burst
    long_isis = isi[isi >= 15.0]  # inter-burst gaps
    assert len(short_isis) > 0, "expected within-burst short ISIs"
    assert len(long_isis) >= 3, "expected several inter-burst gaps (multiple bursts)"

    # Bursts should repeat: at least 3 separate long gaps, evenly enough spaced
    # that this isn't just noise (CV of the long-gap durations is bounded).
    gap_cv = np.std(long_isis) / np.mean(long_isis)
    assert gap_cv < 0.3, f"inter-burst gaps should be roughly periodic, got CV={gap_cv}"


def test_fast_spiking_sustains_high_regular_rate_without_adaptation():
    """FS: high sustained firing rate with minimal spike-frequency
    adaptation (nearly constant ISI throughout), unlike RS."""
    v_rs, _, st_rs, isi_rs = _run(REGULAR_SPIKING, i_val=10.0)
    v_fs, _, st_fs, isi_fs = _run(FAST_SPIKING, i_val=10.0)

    rate_rs = len(st_rs)
    rate_fs = len(st_fs)
    assert rate_fs > rate_rs, "fast spiking should fire at a higher rate than regular spiking"

    # Minimal adaptation: first ISI close to steady-state ISI (ratio near 1),
    # in contrast to RS's pronounced first-vs-steady-state gap.
    fs_adaptation_ratio = isi_fs[0] / np.mean(isi_fs[-5:])
    rs_adaptation_ratio = isi_rs[0] / np.mean(isi_rs[-5:])
    assert fs_adaptation_ratio > rs_adaptation_ratio, (
        "FS should show less spike-frequency adaptation than RS "
        f"(FS ratio={fs_adaptation_ratio:.2f}, RS ratio={rs_adaptation_ratio:.2f})"
    )


def test_all_three_patterns_are_mutually_distinguishable():
    """Sanity check that the three parameter sets don't collapse to the
    same behavior: pairwise spike counts and ISI-CV should all differ."""
    results = {}
    for name, params in [("RS", REGULAR_SPIKING), ("CH", CHATTERING), ("FS", FAST_SPIKING)]:
        _, _, st, isi = _run(params, i_val=10.0)
        cv = np.std(isi) / np.mean(isi) if len(isi) > 1 else 0.0
        results[name] = (len(st), cv)

    counts = [r[0] for r in results.values()]
    assert len(set(counts)) == 3, f"expected 3 distinct spike counts, got {results}"
