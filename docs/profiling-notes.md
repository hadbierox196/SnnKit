# Profiling Notes

Week 10 deliverable. Profiled on this repo's dev sandbox: **CPU only**
(`jax.default_backend() == "cpu"`, single `CpuDevice`) — absolute numbers
here should not be read as GPU performance; the *relative* findings
(compile-vs-exec ratio, where time actually goes) are the useful part and
should be re-checked once this runs on a Colab GPU.

Methodology: every measurement below separates **JIT compile + first call**
(traced once with `time.perf_counter`, `jax.block_until_ready` after) from
**steady-state execution** (mean of several subsequent calls to the same
jitted function, same input shapes — so no retracing).

## 1. Core population simulation (`simulate_population`)

| n_neurons | compile + first call | steady-state exec | ratio |
|-----------|----------------------|--------------------|-------|
| 1,000     | 0.54 s                | 0.056 s            | 9.6×  |
| 10,000    | 2.27 s                | 0.587 s            | 3.9×  |

(T=500 steps, batch=8, forward-Euler LIF, `vmap` over the neuron axis.)

**Finding:** JIT compile time dominates for a one-off or infrequently
repeated simulation call — at 1,000 neurons, compiling costs *almost 10×*
what a single run costs once compiled. This isn't a bug, it's the
standard JAX tradeoff, but it means: **benchmark scripts and notebooks
that report "runtime" should always warm up (discard the first call)
before timing**, which `benchmarks/dense_vs_sparse_scaling.py` already
does. It also means single-shot simulations (e.g. someone exploring one
network configuration interactively) pay this cost every time the input
*shape* changes — reusing a jitted function across many calls with the
same shape (as a training loop does) is where the compile cost actually
amortizes away.

## 2. BPTT training step (`snnkit.training.bptt.train_step`)

| | compile + first call | steady-state exec | ratio |
|-|----------------------|--------------------|-------|
| single `train_step` | 3.42 s | 0.785 s | 4.4× |

(64 samples, 50 timesteps, 50 input channels, 64 hidden units, full
`lax.scan` + surrogate-gradient backward pass via `custom_vjp`.)

**Finding:** the backward pass through the full `lax.scan` (custom VJP at
every one of 50 timesteps) is the dominant per-step cost on CPU — 785ms
for a genuinely tiny network (64 hidden units) is high. This is very
likely CPU-specific: there's no parallelism across the batch/hidden
dimensions being exploited the way a GPU would, and `lax.scan`'s
sequential nature means the backward pass can't overlap across
timesteps. **Decision: not addressed now.** Two reasons: (a) this
environment has no GPU to confirm whether the bottleneck persists there
— re-profile on Colab GPU before optimizing anything, since the fix
(if any is needed) may differ; (b) Phase 2's experiments are
deliberately toy-scale (documented in `docs/local-vs-bptt.md`), so
absolute training speed isn't yet a claim this repo is making. Revisit
if/when GPU numbers are in hand or network sizes grow.

## 3. Python dispatch overhead between training steps

| | total (20 steps) | per-step |
|-|-------------------|----------|
| async-dispatched (block once, at the end) | 15.33 s | 766 ms |
| blocking every step (forces sync each call) | 16.24 s | 812 ms |

**Finding:** per-call Python/dispatch overhead is ~46 ms/step, about 6%
of total step time at this scale. **Not currently a bottleneck** — it's
dwarfed by the backward-pass compute itself (finding #2). Worth
revisiting only if/when GPU execution shrinks the compute-per-step
enough that dispatch overhead becomes comparably sized — at that point,
batching multiple training steps into a single `lax.scan`-wrapped
"epoch" (rather than a Python `for` loop calling a jitted single-step
function) would amortize it away.

## Top bottlenecks, ranked, and what (if anything) to do about them

1. **BPTT backward-pass compute per step (CPU)** — largest single cost
   found. Action: defer; re-profile on GPU first (see finding #2).
2. **JIT compile time on shape changes** — real but well-understood and
   already mitigated procedurally (warm-up-before-timing convention used
   throughout this repo's benchmarks/tests).
3. **Python dispatch overhead** — smallest of the three, not worth
   addressing at current scale.

No memory-side bottleneck was identified as urgent: the largest dense
trace tested here (T=500, batch=8, n=10,000) is ~160 MB per array, well
within reach of both this sandbox and a free-tier Colab GPU. Memory
becomes a real constraint only well past where this repo's Phase 2
experiments currently operate — tracked as a benchmark-scaling question
(`benchmarks/dense_vs_sparse_scaling.py`), not a training-loop one.
