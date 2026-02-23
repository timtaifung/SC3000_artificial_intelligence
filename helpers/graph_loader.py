"""Utilities for loading the NYC road network dictionaries.

Supports JSON or pickle formats and falls back to a small
S-1-2-T example if the expected files are not present.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class GraphData:
    """Container for graph-related data.

    Attributes
    ----------
    G: Adjacency list mapping node id (str) -> list of neighbour ids (str).
    Coord: Mapping node id (str) -> (x, y) coordinates (floats).
    Dist: Mapping (u, v) edge tuple -> distance (float).
    Cost: Mapping (u, v) edge tuple -> energy cost (float).
    source: Start node id.
    target: Goal node id.
    energy_budget: Energy budget for resource-constrained tasks.
    using_example: True if the built-in toy graph is used.
    """

    G: Dict[str, List[str]]
    Coord: Dict[str, Tuple[float, float]]
    Dist: Dict[Tuple[str, str], float]
    Cost: Dict[Tuple[str, str], float]
    source: str
    target: str
    energy_budget: Optional[float]
    using_example: bool


def _find_data_file(base_dir: str, basename: str) -> Optional[str]:
    """Return the first existing path for a given base name and known extensions.

    Known extensions: .json, .pkl, .pickle
    """
    for ext in (".json", ".pkl", ".pickle"):
        candidate = os.path.join(base_dir, basename + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def _load_mapping(path: str) -> Dict[Any, Any]:
    """Load a mapping from a JSON or pickle file based on extension."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as f:  # use binary to support pickle
        if ext == ".json":
            # Re-open as text for JSON
            f.close()
            with open(path, "r", encoding="utf-8") as ft:
                return json.load(ft)
        if ext in (".pkl", ".pickle"):
            return pickle.load(f)
    raise ValueError(f"Unsupported file extension for {path!r}. Use .json or .pkl")


def _normalise_adj_list(raw_G: Dict[Any, Iterable[Any]]) -> Dict[str, List[str]]:
    """Convert adjacency list keys and neighbours to strings."""
    G: Dict[str, List[str]] = {}
    for node, neighbours in raw_G.items():
        node_id = str(node)
        G[node_id] = [str(n) for n in neighbours]
    return G


def _normalise_coord(raw_coord: Dict[Any, Any]) -> Dict[str, Tuple[float, float]]:
    """Convert coordinate mapping keys to strings and values to (float, float)."""
    coord: Dict[str, Tuple[float, float]] = {}
    for node, value in raw_coord.items():
        node_id = str(node)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            x, y = value
        else:
            raise ValueError(f"Invalid coordinate for node {node_id!r}: {value!r}")
        coord[node_id] = (float(x), float(y))
    return coord


def _normalise_edge_mapping(raw_edges: Dict[Any, Any]) -> Dict[Tuple[str, str], float]:
    """Normalise edge dictionaries to use (u, v) tuple keys and float values.

    Accepts keys of the form:
    - (u, v) tuples
    - "u,v" strings
    """
    edges: Dict[Tuple[str, str], float] = {}
    for key, value in raw_edges.items():
        u: str
        v: str
        if isinstance(key, tuple) and len(key) == 2:
            u, v = key  # type: ignore[assignment]
            u, v = str(u), str(v)
        else:
            key_str = str(key)
            if "," not in key_str:
                raise ValueError(
                    f"Edge key {key!r} is not a (u, v) tuple or 'u,v' string. "
                    "Please check the input dictionaries."
                )
            u, v = [part.strip() for part in key_str.split(",", 1)]
        edges[(u, v)] = float(value)
    return edges


def load_graph_data(base_dir: str = ".") -> GraphData:
    """Load graph dictionaries from *base_dir* or fall back to a toy example.

    The loader looks for the four standard files with either JSON or pickle
    extensions: ``G``, ``Coord``, ``Dist``, and ``Cost``. If any of them is
    missing, a message is printed and a small built-in S-1-2-T example graph
    is returned instead so that the program remains runnable.
    """
    g_path = _find_data_file(base_dir, "G")
    coord_path = _find_data_file(base_dir, "Coord")
    dist_path = _find_data_file(base_dir, "Dist")
    cost_path = _find_data_file(base_dir, "Cost")

    paths = [g_path, coord_path, dist_path, cost_path]

    if all(p is None for p in paths):
        print(
            "NYC instance files (G, Coord, Dist, Cost) not found in "
            f"{os.path.abspath(base_dir)}. Using built-in small example graph."
        )
        return _build_example_graph()

    missing = [name for name, path in zip(["G", "Coord", "Dist", "Cost"], paths) if path is None]
    if missing:
        print(
            "Warning: some graph files are missing in "
            f"{os.path.abspath(base_dir)}: {', '.join(missing)}. "
            "Falling back to built-in small example graph."
        )
        return _build_example_graph()

    assert g_path and coord_path and dist_path and cost_path

    raw_G = _load_mapping(g_path)
    raw_coord = _load_mapping(coord_path)
    raw_dist = _load_mapping(dist_path)
    raw_cost = _load_mapping(cost_path)

    G = _normalise_adj_list(raw_G)
    Coord = _normalise_coord(raw_coord)
    Dist = _normalise_edge_mapping(raw_dist)
    Cost = _normalise_edge_mapping(raw_cost)

    return GraphData(
        G=G,
        Coord=Coord,
        Dist=Dist,
        Cost=Cost,
        source="1",
        target="50",
        energy_budget=287932.0,
        using_example=False,
    )


def _build_example_graph() -> GraphData:
    """Return the small S-1-2-T example from the assignment handout.

    The instance encodes two S-T paths:
    - S -> 2 -> T with distance 10 and energy 12 (infeasible for budget 11)
    - S -> 1 -> T with distance 12 and energy 10 (feasible and optimal)
    """
    G: Dict[str, List[str]] = {
        "S": ["1", "2"],
        "1": ["S", "T"],
        "2": ["S", "T"],
        "T": ["1", "2"],
    }

    Coord: Dict[str, Tuple[float, float]] = {
        "S": (0.0, 0.0),
        "1": (1.0, 0.0),
        "2": (0.0, 1.0),
        "T": (1.0, 1.0),
    }

    # Distances
    Dist: Dict[Tuple[str, str], float] = {
        ("S", "1"): 6.0,
        ("1", "S"): 6.0,
        ("1", "T"): 6.0,
        ("T", "1"): 6.0,
        ("S", "2"): 5.0,
        ("2", "S"): 5.0,
        ("2", "T"): 5.0,
        ("T", "2"): 5.0,
    }

    # Energy costs
    Cost: Dict[Tuple[str, str], float] = {
        ("S", "1"): 4.0,
        ("1", "S"): 4.0,
        ("1", "T"): 6.0,
        ("T", "1"): 6.0,
        ("S", "2"): 8.0,
        ("2", "S"): 8.0,
        ("2", "T"): 4.0,
        ("T", "2"): 4.0,
    }

    return GraphData(
        G=G,
        Coord=Coord,
        Dist=Dist,
        Cost=Cost,
        source="S",
        target="T",
        energy_budget=11.0,
        using_example=True,
    )
