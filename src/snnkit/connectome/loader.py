"""Load the White et al. 1986 *C. elegans* whole-worm connectome (sourced
from OpenWorm's ConnectomeToolbox, MIT-licensed — see `data/SOURCE.md`)
into snnkit's existing sparse connectivity format.

Reuses `snnkit.core.synapses.SparseWeights` — the same sparse
representation the core engine already uses for synaptic connectivity —
rather than inventing a separate graph format for this extension. This is
the throughline documented in `docs/connectome-narrative.md`: the
connectome is just another sparse weighted graph to the engine.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from snnkit.core.synapses import SparseWeights

DATA_PATH = Path(__file__).parent / "data" / "aconnectome_white_1986_whole.csv"


class ConnectomeGraph(NamedTuple):
    """A loaded connectome: node names + sparse weighted connectivity.

    Attributes:
        node_names: `[n_nodes]`, index -> neuron/target name (e.g. "ADAL").
        weights: `SparseWeights` over indices into `node_names`; `weight`
            is the raw synapse count for that (pre, post) pair (i.e.
            "connection strength", not a signed excitatory/inhibitory
            weight the way `snnkit.core.synapses` uses it for trainable
            networks — diffusion doesn't need sign, just non-negative
            coupling strength).
        edge_type: `[n_synapses]` int32, `0` = chemical (directed),
            `1` = electrical (gap junction, physically bidirectional
            though listed directionally in the source data).
    """

    node_names: list[str]
    weights: SparseWeights
    edge_type: jnp.ndarray


def load_white1986_connectome(path: Path | str = DATA_PATH) -> ConnectomeGraph:
    """Load `aconnectome_white_1986_whole.csv` into a `ConnectomeGraph`.

    Args:
        path: path to the CSV (defaults to the vendored copy in
            `snnkit/connectome/data/`).

    Returns:
        `ConnectomeGraph`.
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    node_set = sorted({row["pre"] for row in rows} | {row["post"] for row in rows})
    node_index = {name: i for i, name in enumerate(node_set)}

    pre_idx = [node_index[row["pre"]] for row in rows]
    post_idx = [node_index[row["post"]] for row in rows]
    weight = [float(row["synapses"]) for row in rows]
    edge_type = [0 if row["type"] == "chemical" else 1 for row in rows]

    n = len(node_set)
    weights = SparseWeights(
        pre_idx=jnp.array(pre_idx, dtype=jnp.int32),
        post_idx=jnp.array(post_idx, dtype=jnp.int32),
        weight=jnp.array(weight, dtype=jnp.float32),
        n_pre=n,
        n_post=n,
    )
    return ConnectomeGraph(
        node_names=node_set, weights=weights, edge_type=jnp.array(edge_type, dtype=jnp.int32)
    )


def to_networkx(graph: ConnectomeGraph, chemical_only: bool = False):
    """Convert to a `networkx.DiGraph` for visualization / graph analysis.

    Args:
        graph: `ConnectomeGraph`.
        chemical_only: if `True`, exclude electrical (gap junction) edges.
    """
    import networkx as nx

    g = nx.DiGraph()
    g.add_nodes_from(graph.node_names)

    pre_idx = graph.weights.pre_idx.tolist()
    post_idx = graph.weights.post_idx.tolist()
    weight = graph.weights.weight.tolist()
    edge_type = graph.edge_type.tolist()

    for p, q, w, et in zip(pre_idx, post_idx, weight, edge_type):
        if chemical_only and et == 1:
            continue
        u, v = graph.node_names[p], graph.node_names[q]
        if g.has_edge(u, v):
            g[u][v]["weight"] += w
        else:
            g.add_edge(u, v, weight=w, electrical=(et == 1))
    return g


def dense_adjacency(graph: ConnectomeGraph) -> jnp.ndarray:
    """Dense `[n, n]` weighted adjacency matrix (sum of synapse counts for
    each (pre, post) pair, chemical + electrical combined). Convenient for
    the diffusion model, which operates on a dense small graph (309 nodes
    is comfortably dense-representable)."""
    n = len(graph.node_names)
    adj = jnp.zeros((n, n))
    adj = adj.at[graph.weights.pre_idx, graph.weights.post_idx].add(graph.weights.weight)
    return adj
