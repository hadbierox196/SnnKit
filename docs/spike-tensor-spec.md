# Spike Tensor Spec (v1)

Written before implementation, per Phase 0 / Week 2 of the roadmap. This is
the contract `snnkit.core.spikes` implements and that later
compiler-facing work (Phase 6, hardware deployment) should be able to
consume without redefinition.

## Why sparse

Dense spike tensors (`[time, batch, neurons]` of 0/1) are simple but waste
memory and bandwidth once networks get large and firing rates are low
(typical for biologically plausible regimes, <20 Hz). A sparse
index-based representation only stores which neurons fired.

## Representation

A sparse spike batch is a small struct of three arrays, all sharing a
common leading "event" dimension `E` (the number of spikes in the batch):

| Field       | Shape | Dtype     | Meaning                                   |
|-------------|-------|-----------|--------------------------------------------|
| `batch_idx` | `[E]` | `int32`   | which batch element this spike belongs to |
| `neuron_idx`| `[E]` | `int32`   | which neuron fired                         |
| `time_idx`  | `[E]` | `int32`   | which timestep the spike occurred at       |

We use **timestep index**, not wall-clock time, so the representation is
independent of `dt` — converting to physical time is a one-line
multiplication (`time_idx * dt`) done at the boundary, not baked into the
core format.

## Layout: padded, not ragged

`E` (event count) varies per batch element and isn't known until runtime,
which is awkward under JIT (JAX requires static shapes). We use a
**padded** layout: arrays are shape `[max_events]`, front-filled with real
events and back-filled with a sentinel (`neuron_idx = -1`) for unused
slots. This was chosen over a ragged/jagged representation specifically
because it JIT-compiles with a static shape; `max_events` is a
conservative upper bound (e.g. `n_neurons * n_timesteps` for the fully
dense case, or a smaller estimate when expected sparsity is known).

Ragged (`jax.experimental.sparse` / variable-length lists per batch
element) was considered and rejected for v1: it requires re-tracing on
event-count changes, which defeats JIT caching during training loops
where the same shapes get called repeatedly. Revisit if padding overhead
becomes the dominant cost at larger scale.

## Conversion

- `dense_to_sparse(spike_trace) -> SparseSpikes`: takes the
  `[time, batch, neurons]` dense output of `simulate_population` and
  extracts nonzero entries into the padded format above.
- `sparse_to_dense(sparse_spikes, shape) -> dense array`: inverse, used
  for correctness testing (`tests/test_sparse_dense_equivalence.py`) and
  for feeding back into dense-only code paths (e.g. plotting).

## Non-goals for v1

- No support for multi-valued events (e.g. graded/analog spikes) — binary
  spikes only.
- No on-device dynamic memory growth — `max_events` is fixed per call.
- Not yet exposed as the primary I/O format for training loops (Phase 2
  trains on dense traces); sparse is currently a storage/benchmarking
  format. Revisit if a training loop is shown to be I/O-bound on dense
  traces.
