# snnkit

A JAX-native spiking neural network simulator: biologically faithful,
equation-based modeling with GPU-scale performance via JAX, trainable via
both surrogate-gradient BPTT and local learning rules (STDP, SuperSpike).

[![CI](https://github.com/hadbierox196/snnkit/actions/workflows/ci.yml/badge.svg)](https://github.com/hadbierox196/snnkit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Replace `hadbierox196` above (and in `pyproject.toml`) once this repo
> has a real home on GitHub, and the CI badge will start reporting
> real status.

## What this is

Built following a 22-week roadmap (Phases 0-5), this repo implements
Phases 0-3: a core simulation engine, a small model library with two
training paradigms, and an optional connectome extension. Phases 4-5
(open-source community process, company formation, fundraising) are
business/process phases outside a codebase's scope and aren't
represented here as code — see "What's not in this repo" below.

**50 passing tests.** Every deliverable below is falsifiable — run the
tests, run the benchmarks, read the docs; nothing here is asserted
without a corresponding check.

## Quickstart

```bash
git clone https://github.com/hadbierox196/snnkit.git
cd snnkit
pip install -e ".[dev,connectome]"
pytest -v
```

```python
import jax.numpy as jnp
from snnkit.core.neuron import LIFParams, simulate_lif

params = LIFParams(tau=20e-3, v_th=1.0, dt=1e-3)
i_trace = jnp.full((1000,), 1.5)  # constant input current
v_trace, spike_trace = simulate_lif(i_trace, params)
print(f"{int(spike_trace.sum())} spikes in {len(i_trace)} steps")
```

## Repo structure

```
src/snnkit/
  core/         Simulation engine: neuron dynamics, populations, spikes,
                synapses, delays, the minimal equation parser
  api/          NeuronGroup/SynapseGroup object API wrapping core
  models/       Model library: LIF, Izhikevich
  training/     Differentiable engine (surrogate gradients), BPTT, STDP,
                SuperSpike
  connectome/   Optional extension: connectome loading + diffusion
tests/          50 tests, one file per major deliverable
benchmarks/     Scalability + comparison scripts, with saved results
docs/           Spec docs, profiling notes, writeups
notebooks/      Runnable Colab notebooks for key deliverables
```

## Phase 1: Core Engine (Weeks 1-5)

- **Week 1** — LIF neuron as a plain JAX function, validated against the
  analytical firing-rate formula. `src/snnkit/core/neuron.py`,
  `tests/test_lif_firing_rate.py`, `notebooks/week01_single_neuron.ipynb`.
- **Week 2** — Population/batch simulation via `vmap`, sparse spike-index
  representation. `docs/spike-tensor-spec.md`,
  `src/snnkit/core/population.py`, `src/snnkit/core/spikes.py`,
  `benchmarks/dense_vs_sparse_scaling.py`.
- **Week 3** — Sparse synaptic connectivity + current injection, hand
  -verified on a small network. `src/snnkit/core/synapses.py`,
  `tests/test_synapse_current.py`.
- **Week 4** — Ring-buffer delay mechanism, exact-timing tests, a stable
  ~100-neuron recurrent network. `src/snnkit/core/delays.py`,
  `tests/test_synapse_delay.py`.
- **Week 5** — Deliberately narrow equation parser (first-order ODEs, no
  units — see the parser's own docstring for what it explicitly punts
  on), `NeuronGroup`/`SynapseGroup` object API, numerically validated
  (max abs error < 1e-5) against the hand-written path.
  `src/snnkit/core/parser.py`, `src/snnkit/api/`, `tests/test_parser.py`.

## Phase 2: Model Library & Local Learning (Weeks 6-12)

- **Week 6** — Izhikevich model, validated against Izhikevich (2003)'s
  regular-spiking/chattering/fast-spiking parameter sets.
  `src/snnkit/models/izhikevich.py`, `tests/test_izhikevich.py`,
  `notebooks/week06_izhikevich_validation.ipynb`.
- **Week 7** — Surrogate-gradient spike function; confirmed gradients
  flow non-zero and NaN-free through a full `lax.scan` simulation.
  `src/snnkit/training/surrogate.py`, `tests/test_surrogate_gradient.py`.
- **Week 8** — BPTT training loop (optax) on a synthetic, SHD-*shaped*
  task (see honesty note below). `src/snnkit/training/bptt.py`,
  `tests/test_bptt_training.py`.
- **Week 9** — STDP with pre/post eligibility traces, fit against the
  theoretical curve (R² > 0.95). `src/snnkit/training/stdp.py`,
  `tests/test_stdp.py`.
- **Week 10** — Profiling: top bottlenecks identified, with an explicit
  decision on what's addressed now vs. deferred. `docs/profiling-notes.md`.
- **Week 11** — SuperSpike (feedback-alignment-style local learning
  rule), trained on the same task as Week 8. E-prop was a stretch goal
  and **was not attempted** — recorded honestly, not silently dropped.
  `src/snnkit/training/superspike.py`, `tests/test_superspike.py`.
- **Week 12** — BPTT vs. SuperSpike across 3 seeds (mean ± std), plus a
  rough Brian2 reference point for the underlying engine.
  `docs/local-vs-bptt.md`, `benchmarks/local_vs_bptt_comparison.py`.

**Honesty note on Weeks 8/11/12:** these train on
`snnkit.training.bptt.synthetic_shd_like_dataset` — a locally generated,
SHD-*shaped* task, not the real SHD (Spiking Heidelberg Digits) dataset
(downloading it requires network access this environment didn't have).
See `docs/local-vs-bptt.md` for exactly what that means for how to read
the results, and how to swap in the real dataset via `tonic`.

## Phase 3: Connectome Extension (Weeks 13-16, optional)

Gated on Phase 1-2 being solid (they are — 50/50 tests passing). See
`docs/connectome-narrative.md` for why this belongs in the same repo
rather than being a bolted-on demo.

- **Week 13** — Real connectome (White et al. 1986, 309 neurons, 2,961
  synapses) loaded from a named, licensed source (OpenWorm's
  ConnectomeToolbox, MIT) directly into snnkit's existing sparse graph
  format. `src/snnkit/connectome/loader.py`,
  `src/snnkit/connectome/data/SOURCE.md`, `tests/test_connectome_loader.py`,
  `docs/connectome_static_visualization.png`.
- **Week 14** — Diffusion (graph heat equation) model, validated against
  the exact matrix-exponential solution (correlation > 0.999).
  `src/snnkit/connectome/diffusion.py`, `tests/test_connectome_diffusion.py`.
- **Week 15-16** — Interactive Plotly visualization + standalone demo
  notebook + scope-limits writeup.
  `notebooks/connectome_disease_spread_demo.ipynb`,
  `docs/connectome-demo.md`.

## What's not in this repo

Phases 4 (ongoing open-source process) and 5 (customer discovery,
positioning, incorporation, fundraising) are real, valuable parts of the
original roadmap — but they're actions in the world (conversations,
legal filings, applications), not code. This repo is the Phase 1-3
technical deliverable the roadmap says should exist *before* any of that
— "ship a working demo before fundraising... code is the pitch."

Also out of scope, tracked rather than ignored (per the roadmap's own
"explicit gaps" list): API stability/versioning policy, experiment
logging (TensorBoard/W&B) integration, a docs site (MkDocs), and
config-file-based parameter management. None of these block using or
extending what's here.

## Reproducing results

Every number reported in `docs/` comes from a script or test you can
re-run:

```bash
pip install -e ".[dev,connectome]"
pytest -v                                              # all 50 tests
python benchmarks/dense_vs_sparse_scaling.py           # Week 2 scaling table
python benchmarks/local_vs_bptt_comparison.py          # Week 12 multi-seed comparison
```

Reproducibility convention (used throughout): `snnkit.reproducibility.set_seed`
for a fixed seed, `snnkit.reproducibility.get_package_versions` recorded
alongside any reported number.

## License

MIT, see `LICENSE`. The vendored connectome dataset
(`src/snnkit/connectome/data/aconnectome_white_1986_whole.csv`) is
separately MIT-licensed by the OpenWorm project — see
`src/snnkit/connectome/data/SOURCE.md` for full attribution.
