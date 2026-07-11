# Local Learning (SuperSpike) vs. BPTT: A Statistically Grounded Comparison

Week 12 deliverable. Both training methods (Week 8's surrogate-gradient
BPTT, Week 11's SuperSpike-style local rule) trained on the **same**
synthetic SHD-shaped task, across **3 random seeds each** (not a single
run), with mean ± std reported. Raw results:
`benchmarks/results/local_vs_bptt.json`, produced by
`benchmarks/local_vs_bptt_comparison.py` (rerun it to reproduce these
numbers exactly, or with different seeds/settings).

## Task and honesty note on the dataset

**This is not the real SHD (Spiking Heidelberg Digits) dataset.**
Downloading it requires network access to
`https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/`, which
this environment's network allowlist doesn't include. Both methods here
train on `snnkit.training.bptt.synthetic_shd_like_dataset`: a locally
generated, SHD-*shaped* task (multi-channel spike trains, per-class
channel signatures, held-out train/test splits sharing the same
class↔channel task definition) — same shape of problem, not the real
benchmark. **To reproduce on real SHD:** install `tonic`
(`pip install tonic`), load `tonic.datasets.SHD`, bin events into the
same `[time, channels]` dense array format, and everything downstream
(`snnkit.training.bptt`, `snnkit.training.superspike`) should work
unchanged. Numbers below should be read as "does the training pipeline
work and how do the two methods compare to each other," not as a
claim about performance on the real benchmark.

Task settings used here: 10 classes, 50 input channels, 50 timesteps,
64 hidden units, 64 train / 32 test samples per seed.

## Results: mean ± std test accuracy across 3 seeds

| Method | Test accuracy (mean ± std) | Individual seeds | Wall time (mean) |
|---|---|---|---|
| BPTT (Week 8) | **0.979 ± 0.029** | 0.938, 1.000, 1.000 | 61.9 s |
| SuperSpike (Week 11) | **0.958 ± 0.039** | 1.000, 0.906, 0.969 | 6.9 s |

(Wall time is this repo's CPU sandbox, not representative of GPU
performance — see caveat below and `docs/profiling-notes.md`.)

**Interpretation:** both methods solve this task reliably and land within
each other's std — on a task this small/easy, this comparison mainly
demonstrates that both training pipelines *work end-to-end* and converge
consistently across seeds (the original goal), rather than establishing
a meaningful accuracy gap. SuperSpike's wall-clock advantage here (~9×
faster) is a real, interesting signal — it reflects that per-sample
online local updates on this tiny network are cheap on CPU. **Do not
over-read this speed gap**: BPTT's real advantage (better scaling
to harder, longer-horizon credit-assignment problems) doesn't show up on
a task this easy and this short (50 timesteps); SuperSpike's per-sample
Python-loop training (see `superspike.train`) also isn't yet batched the
way BPTT's minibatch loop is, which is a training-loop implementation
difference, not an inherent property of the learning rule. A harder task
and an apples-to-apples batched implementation of both would be needed
before drawing a real speed conclusion.

## E-prop: not attempted

Per the roadmap's own risk flag (Week 11 was the second-highest-risk week
in the plan), e-prop was explicitly scoped as a stretch goal, attempted
only if SuperSpike went smoothly and time allowed. **It was not
attempted in this repo.** SuperSpike trained cleanly and consistently
enough (see table above) that there wasn't a clear signal something
harder was needed to demonstrate a working local rule — recorded here
honestly rather than silently dropped.

## Competitive reference point: Brian2

Per the roadmap, a rough comparison against an existing framework, for
context. Same LIF dynamics (`tau=20ms`, `v_th=1`, constant input current
`I=1.5`, `dt=1ms`, 500ms duration), run through both Brian2 and
`snnkit.core.population.simulate_population`, at 3 population sizes:

| n_neurons | Brian2 (s) | snnkit (s) | Spike counts match |
|-----------|-----------|-----------|---------------------|
| 100 | 0.452 | 0.0002 | ✓ (2,200 = 2,200) |
| 1,000 | 0.391 | 0.0028 | ✓ (22,000 = 22,000) |
| 5,000 | 0.457 | 0.0133 | ✓ (110,000 = 110,000) |

**Caveats, read before drawing conclusions:**
- Spike counts matching exactly across all 3 sizes is the important
  correctness signal here — both simulators implement the same LIF
  dynamics and agree exactly, which is reassuring independent of speed.
- Brian2 ran with its **`numpy` codegen target**, because this sandbox
  has no C compiler available for Brian2's (typically faster) C++
  target. Treat this speed comparison as conservative/favorable to
  snnkit, not definitive — re-run with `brian2.prefs.codegen.target =
  'cython'` or the default C++ target on a machine that has a compiler
  for a fairer number.
- snnkit's timings are **steady-state, post-JIT-warmup** (the standard
  JAX convention, see `docs/profiling-notes.md`); Brian2's timings
  include its own code-generation overhead on `run()`, which doesn't
  have a separate "warm up and discard" step in the same way. This makes
  the comparison closer to "first real use" for Brian2 vs. "amortized
  steady-state" for snnkit — not perfectly apples-to-apples, noted rather
  than hidden.
- This is a **single, trivial, embarrassingly-parallel workload**
  (unconnected neurons, constant input) — it says nothing about relative
  performance on the sparse recurrent connectivity this repo actually
  cares about (Week 3-4). A fairer follow-up benchmark would replicate
  Week 4's ~100-neuron delayed recurrent network in both frameworks.

## Reproducing

```bash
pip install -e ".[dev]"
pip install brian2   # optional, for the reference-point section only
python benchmarks/local_vs_bptt_comparison.py --seeds 0 1 2
```
