"""Object-oriented API wrapping the core simulation engine.

Per Phase 0's "hybrid API philosophy": equations describe neuron/synapse
*dynamics* (`snnkit.core.parser`), while `NeuronGroup` / `SynapseGroup`
here provide the object API for network *structure* (how groups connect,
how simulation state is threaded and stepped together). This module must
never contain dynamics logic itself — only bookkeeping and delegation to
`snnkit.core`.
"""
