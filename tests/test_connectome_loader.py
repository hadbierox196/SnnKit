"""Week 13 deliverable: connectome loaded from a named, licensed source
(OpenWorm's ConnectomeToolbox), converted into snnkit's existing sparse
connectivity format."""

import jax.numpy as jnp

from snnkit.connectome.loader import dense_adjacency, load_white1986_connectome, to_networkx


def test_connectome_loads_with_expected_scale():
    """Sanity check on scale: the White et al. 1986 whole-worm connectome
    has a few hundred nodes and a few thousand edges — catch a parsing
    regression that silently drops most of the data."""
    graph = load_white1986_connectome()
    n_nodes = len(graph.node_names)
    n_edges = graph.weights.pre_idx.shape[0]

    assert 250 <= n_nodes <= 400, f"expected ~309 nodes, got {n_nodes}"
    assert 2000 <= n_edges <= 4000, f"expected ~2961 edges, got {n_edges}"
    assert len(set(graph.node_names)) == n_nodes, "node names should be unique"


def test_known_neurons_present():
    """A few well-known, frequently-cited C. elegans neurons should be
    present by name (catches wholesale mis-parsing of the name column)."""
    graph = load_white1986_connectome()
    node_set = set(graph.node_names)
    # AVAL/AVAR (command interneurons), ADAL (sensory), PLML (touch receptor)
    # are all textbook C. elegans neurons that should appear in any correct
    # parse of this dataset.
    for name in ["AVAL", "AVAR", "ADAL", "PLML"]:
        assert name in node_set, f"expected well-known neuron {name!r} in loaded connectome"


def test_edge_types_are_chemical_or_electrical_only():
    graph = load_white1986_connectome()
    assert set(graph.edge_type.tolist()) <= {0, 1}


def test_weights_are_positive_synapse_counts():
    graph = load_white1986_connectome()
    assert jnp.all(graph.weights.weight > 0), "synapse counts should all be positive"


def test_to_networkx_matches_loaded_scale():
    graph = load_white1986_connectome()
    g = to_networkx(graph)
    assert g.number_of_nodes() == len(graph.node_names)
    # to_networkx merges parallel (pre,post) edges of different type into
    # one graph edge, so edge count can be <= the raw edge count.
    assert g.number_of_edges() <= graph.weights.pre_idx.shape[0]
    assert g.number_of_edges() > 0


def test_dense_adjacency_matches_sparse_total_weight():
    """Total weight in the dense adjacency should equal total weight in
    the sparse representation (no mass lost/gained in conversion)."""
    graph = load_white1986_connectome()
    adj = dense_adjacency(graph)
    assert jnp.isclose(adj.sum(), graph.weights.weight.sum())
