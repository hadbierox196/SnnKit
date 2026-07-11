# Why a Connectome / Disease-Spread Demo Belongs in This Repo

Week 13 deliverable. This doc exists because "SNN simulator" and "disease
spread on a worm connectome" sound, on first read, like two different
products bolted together for a demo. They're not — here's the actual
throughline, spelled out rather than left implicit.

## The throughline

**1. General sparse graph engine.** `snnkit.core.synapses.SparseWeights`
(Week 3) is not "neuron connectivity" as a hardcoded concept — it's a
generic sparse weighted directed graph representation: `(pre_idx,
post_idx, weight, n_pre, n_post)`. It was built to answer "how do I inject
current from spiking neuron A into neuron B," but nothing about the
representation is neuron-specific.

**2. Connectome representation.** A biological connectome — literally a
measured, real graph of which neurons connect to which, with what
strength (synapse count) — slots into that same representation with zero
new infrastructure. `snnkit.connectome.loader.load_white1986_connectome`
parses a real dataset directly into `SparseWeights`. This is the first
proof point: the engine's core data structure wasn't over-fit to "small
toy networks I built by hand" (Weeks 3-4) — it holds up on a real,
externally-sourced 309-node, 2,961-edge graph without modification.

**3. Diffusion dynamics.** Once you have a weighted graph, "how does
something spread across it over time" is a natural question — and it's
the same *kind* of question as "how does a spike's effect propagate
through a network," just with different per-node dynamics (a scalar
diffusing quantity via the heat equation, vs. a spiking threshold
neuron). `snnkit.connectome.diffusion` reuses the loaded graph's adjacency
directly; no new graph infrastructure, only new *dynamics* on top of it —
consistent with Phase 0's original design decision to separate dynamics
from structure.

**4. Disease modeling.** Pathology spread along neural connectivity
(e.g. prion-like propagation of misfolded proteins along synaptically
connected pathways, a real and actively studied hypothesis in
neurodegenerative disease research — see e.g. work on tau propagation in
Alzheimer's along connectome pathways) is a genuine, motivated instance
of "diffusion on a connectome." It's not a metaphor stretched to fit;
network-diffusion models of pathology spread are an actual subfield.

## Why this is the same company, not two companies

The pitch isn't "we simulate neurons AND we simulate disease." It's: **we
built a general, fast, differentiable engine for dynamics on sparse
biological graphs, and both spiking neural circuits and pathology-spread
networks are instances of that same underlying problem.** The core
engine's value (Weeks 1-12: correctness, speed, trainability) is the
moat; the connectome extension is a demonstration that the moat
generalizes past the exact use case it was first built for; that's a
*stronger* signal of a real platform than a simulator that only ever
does the one thing it was designed for.

## What this section does NOT claim

- No claim of clinical validity. `snnkit.connectome.diffusion` is a
  simplified linear diffusion model, validated against a mathematical
  reference (Week 14), not against real disease-progression data — see
  `docs/connectome-demo.md` for the explicit scope limits.
- No claim that this specific connectome (a 1986 electron-microscopy
  reconstruction of *C. elegans*, a 302-neuron worm) is directly relevant
  to human disease. It's a real, properly-sourced, appropriately-scaled
  graph for demonstrating the engine generalizes — not a human
  connectome, and not presented as one.
