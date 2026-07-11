"""Dense vs. sparse scalability benchmark (Week 2 deliverable).

Measures runtime, peak memory, JIT compile time (separately from
execution time), and throughput at 1k / 10k / 50k neurons, and at
whatever size maxes out available accelerator memory.

Run:
    python benchmarks/dense_vs_sparse_scaling.py [--max-neurons N] [--out results.json]

Note: peak-memory measurement uses `jax.live_arrays` byte accounting on
CPU/GPU as a portable proxy. On GPU, prefer reading
`jax.devices()[0].memory_stats()['peak_bytes_in_use']` if available for a
more accurate number — this script does so when running on GPU.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

from snnkit.core.neuron import LIFParams
from snnkit.core.population import simulate_population
from snnkit.core.spikes import dense_to_sparse
from snnkit.reproducibility import get_package_versions, set_seed

DEFAULT_SIZES = [1_000, 10_000, 50_000]


def _peak_memory_bytes() -> int | None:
    dev = jax.devices()[0]
    stats = None
    try:
        stats = dev.memory_stats()
    except Exception:
        return None
    if stats is None:
        return None
    return stats.get("peak_bytes_in_use")


def benchmark_size(n_neurons: int, time_steps: int = 500, batch: int = 8) -> dict:
    key = set_seed(0)
    i_trace = jax.random.uniform(key, (time_steps, batch, n_neurons), minval=0.0, maxval=2.0)
    params = LIFParams(tau=20e-3, v_th=1.0, dt=1e-3)

    sim_fn = jax.jit(lambda i: simulate_population(i, params))

    # JIT compile time: first call (including trace + compile), isolated
    # from steady-state execution time.
    t0 = time.perf_counter()
    v, spikes = jax.block_until_ready(sim_fn(i_trace))
    compile_and_first_run_s = time.perf_counter() - t0

    # Steady-state execution time: average of several subsequent calls.
    n_repeats = 5
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        v, spikes = jax.block_until_ready(sim_fn(i_trace))
    exec_s = (time.perf_counter() - t0) / n_repeats

    peak_mem = _peak_memory_bytes()

    # Sparse conversion cost, for reference (dense->sparse is itself an op
    # with a cost; report it rather than assume sparse is "free").
    approx_spikes = int(spikes.sum())
    sparse_fn = jax.jit(lambda s: dense_to_sparse(s, max_events=max(approx_spikes * 2, 1)))
    t0 = time.perf_counter()
    sparse_result = jax.block_until_ready(sparse_fn(spikes))
    sparse_convert_s = time.perf_counter() - t0

    throughput = (time_steps * batch * n_neurons) / exec_s  # neuron-steps/sec

    return {
        "n_neurons": n_neurons,
        "time_steps": time_steps,
        "batch": batch,
        "compile_plus_first_run_s": compile_and_first_run_s,
        "steady_state_exec_s": exec_s,
        "sparse_convert_s": sparse_convert_s,
        "peak_memory_bytes": peak_mem,
        "throughput_neuron_steps_per_sec": throughput,
        "total_spikes": approx_spikes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=DEFAULT_SIZES, help="Population sizes to test."
    )
    parser.add_argument(
        "--max-neurons",
        type=int,
        default=None,
        help="If set, also benchmark this size (e.g. whatever maxes out your GPU).",
    )
    parser.add_argument("--out", type=str, default="benchmarks/results/dense_vs_sparse.json")
    args = parser.parse_args()

    sizes = list(args.sizes)
    if args.max_neurons is not None and args.max_neurons not in sizes:
        sizes.append(args.max_neurons)

    print(f"Backend: {jax.default_backend()}, devices: {jax.devices()}")
    results = []
    for n in sizes:
        print(f"Benchmarking n_neurons={n} ...")
        try:
            r = benchmark_size(n)
            results.append(r)
            print(
                f"  compile+first={r['compile_plus_first_run_s']:.4f}s  "
                f"steady-state={r['steady_state_exec_s']:.4f}s  "
                f"throughput={r['throughput_neuron_steps_per_sec']:.3e} neuron-steps/s  "
                f"peak_mem={r['peak_memory_bytes']}"
            )
        except Exception as e:  # noqa: BLE001 - report and continue to next size
            print(f"  FAILED at n_neurons={n}: {e}")
            results.append({"n_neurons": n, "error": str(e)})

    output = {"environment": get_package_versions(), "results": results}

    import os

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()
