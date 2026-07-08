"""Numeric-valued functionals on GUDHI simplex trees.

Every public function in this module takes a ``gudhi.SimplexTree`` as its
first argument and returns one integer- or real-valued scalar.
"""

import gudhi
import numpy as np


def simplex_count(simplex_tree: gudhi.SimplexTree, dimension: int) -> int:
    """Return the number of simplices of the requested dimension."""
    if not isinstance(dimension, (int, np.integer)) or dimension < 0:
        raise ValueError("dimension must be a nonnegative integer.")
    return int(sum(
        1
        for simplex, _ in simplex_tree.get_skeleton(int(dimension))
        if len(simplex) == dimension + 1
    ))


def vertex_count(simplex_tree: gudhi.SimplexTree) -> int:
    """Return the number of vertices."""
    return int(simplex_tree.num_vertices())


def edge_count(simplex_tree: gudhi.SimplexTree) -> int:
    """Return the number of edges."""
    return simplex_count(simplex_tree, dimension=1)


def triangle_count(simplex_tree: gudhi.SimplexTree) -> int:
    """Return the number of two-simplices."""
    return simplex_count(simplex_tree, dimension=2)


def betti_number(simplex_tree: gudhi.SimplexTree, dimension: int) -> int:
    """Return the Betti number in the requested dimension."""
    if not isinstance(dimension, (int, np.integer)) or dimension < 0:
        raise ValueError("dimension must be a nonnegative integer.")
    simplex_tree.compute_persistence(
        homology_coeff_field=2,
        persistence_dim_max=True,
    )
    betti = simplex_tree.betti_numbers()
    return int(betti[dimension]) if dimension < len(betti) else 0


def beta_0(simplex_tree: gudhi.SimplexTree) -> int:
    """Return the number of connected components."""
    return betti_number(simplex_tree, dimension=0)


def beta_1(simplex_tree: gudhi.SimplexTree) -> int:
    """Return the first Betti number."""
    return betti_number(simplex_tree, dimension=1)


def point_complexity(simplex_tree: gudhi.SimplexTree) -> float:
    """Return sum_x [dim(K) - max_{sigma containing x} dim(sigma)]."""
    simplices = list(simplex_tree.get_filtration())
    if not simplices:
        return 0.0

    complex_dimension = max(len(simplex) - 1 for simplex, _ in simplices)
    vertex_max_dimension = {}
    for simplex, _ in simplices:
        dimension = len(simplex) - 1
        for vertex in simplex:
            vertex_max_dimension[vertex] = max(
                vertex_max_dimension.get(vertex, 0),
                dimension,
            )

    return float(sum(
        complex_dimension - maximum_dimension
        for maximum_dimension in vertex_max_dimension.values()
    ))


def log_component_edge_functional(
    simplex_tree: gudhi.SimplexTree,
    alpha: float = 1.0,
    xi: float = 1.0,
    tilde_delta: float = 0.0,
) -> float:
    """Return tilde_delta + alpha log(beta_0) + xi log(max(1, |E|))."""
    components = beta_0(simplex_tree)
    if components < 1:
        raise ValueError("The logarithmic functional requires at least one vertex.")
    edges = edge_count(simplex_tree)
    return float(
        tilde_delta
        + alpha * np.log(components)
        + xi * np.log(max(1, edges))
    )


def functional_value(simplex_tree: gudhi.SimplexTree, name: str) -> float:
    """Evaluate one named basic scalar functional."""
    functionals = {
        "f0 (vertices)": vertex_count,
        "f1 (edges)": edge_count,
        "f2 (triangles)": triangle_count,
        "beta0": beta_0,
        "beta1": beta_1,
        "point complexity": point_complexity,
    }
    try:
        functional = functionals[name]
    except KeyError as exc:
        raise ValueError(f"Unknown functional: {name}") from exc
    return float(functional(simplex_tree))
