"""Shortest-path and resource-constrained search algorithms for Part 1.

This module implements:
- Dijkstra's algorithm on Dist weights (no energy constraint).
- Uniform Cost Search with an energy budget constraint.
- A* search with an admissible Euclidean heuristic and label-setting
  treatment of the (distance, energy) trade-off.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


Graph = Dict[str, List[str]]
CoordDict = Dict[str, Tuple[float, float]]
EdgeWeights = Dict[Tuple[str, str], float]


@dataclass
class SearchResult:
    """Container for search results."""

    path: Optional[List[str]]
    distance: Optional[float]
    energy: Optional[float]
    expanded_nodes: int


def _edge_weight(weights: EdgeWeights, u: str, v: str) -> Optional[float]:
    """Return edge weight for (u, v) if present."""
    return weights.get((u, v))


def _compute_path_cost(
    path: List[str],
    dist: EdgeWeights,
    cost: EdgeWeights,
) -> Tuple[float, float]:
    """Compute total distance and energy for a given path."""
    total_dist = 0.0
    total_energy = 0.0
    for u, v in zip(path, path[1:]):
        d = _edge_weight(dist, u, v)
        c = _edge_weight(cost, u, v)
        if d is None or c is None:
            raise KeyError(f"Missing Dist or Cost for edge ({u}, {v})")
        total_dist += d
        total_energy += c
    return total_dist, total_energy


def dijkstra_shortest_path(
    G: Graph,
    Dist: EdgeWeights,
    Cost: EdgeWeights,
    start: str,
    goal: str,
) -> SearchResult:
    """Standard Dijkstra on Dist weights, ignoring the energy constraint.

    Returns the shortest-distance path from *start* to *goal*, together with
    its distance and corresponding energy cost.
    """
    distances: Dict[str, float] = {start: 0.0}
    predecessors: Dict[str, str] = {}
    visited = set()

    heap: List[Tuple[float, str]] = [(0.0, start)]
    expanded = 0

    while heap:
        d_u, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        expanded += 1

        if u == goal:
            break

        for v in G.get(u, []):
            w = _edge_weight(Dist, u, v)
            if w is None:
                continue
            alt = d_u + w
            old = distances.get(v)
            if old is None or alt < old:
                distances[v] = alt
                predecessors[v] = u
                heapq.heappush(heap, (alt, v))

    if goal not in distances:
        return SearchResult(path=None, distance=None, energy=None, expanded_nodes=expanded)

    # Reconstruct path
    path: List[str] = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = predecessors[cur]
    path.append(start)
    path.reverse()

    total_dist, total_energy = _compute_path_cost(path, Dist, Cost)
    return SearchResult(path=path, distance=total_dist, energy=total_energy, expanded_nodes=expanded)


@dataclass(order=True)
class _Label:
    """Label for resource-constrained search.

    The first field ``priority`` is used by ``heapq`` as the key.
    """

    priority: float
    distance: float
    energy: float
    node: str
    parent: Optional["_Label"] = None


def _euclidean_heuristic(
    Coord: CoordDict,
    node: str,
    goal: str,
) -> float:
    """Admissible Euclidean heuristic based on node coordinates.

    If coordinates are missing for a node, fall back to 0 (still admissible).
    """
    try:
        x1, y1 = Coord[node]
        x2, y2 = Coord[goal]
    except KeyError:
        return 0.0
    dx = x1 - x2
    dy = y1 - y2
    return math.hypot(dx, dy)


def _rcsp_search(
    G: Graph,
    Dist: EdgeWeights,
    Cost: EdgeWeights,
    start: str,
    goal: str,
    energy_budget: float,
    heuristic,
) -> SearchResult:
    """Generic label-setting search for the resource-constrained problem.

    Parameters
    ----------
    heuristic: callable(node: str) -> float
        If ``heuristic`` returns 0 for all nodes, this reduces to a
        Uniform Cost Search over distance with an energy constraint.
        If ``heuristic`` returns the Euclidean distance, this becomes an
        A* search with an admissible heuristic.
    """
    start_label = _Label(
        priority=float(heuristic(start)),
        distance=0.0,
        energy=0.0,
        node=start,
        parent=None,
    )

    frontier: List[_Label] = [start_label]
    heapq.heapify(frontier)

    # For each node, keep only Pareto-optimal labels in (distance, energy).
    labels_at_node: Dict[str, List[_Label]] = {start: [start_label]}

    expanded = 0

    while frontier:
        current = heapq.heappop(frontier)
        expanded += 1

        if current.node == goal:
            # First time we pop the goal: optimal under admissible heuristic.
            return SearchResult(
                path=_reconstruct_path_from_label(current),
                distance=current.distance,
                energy=current.energy,
                expanded_nodes=expanded,
            )

        for v in G.get(current.node, []):
            d_edge = _edge_weight(Dist, current.node, v)
            c_edge = _edge_weight(Cost, current.node, v)
            if d_edge is None or c_edge is None:
                continue

            new_energy = current.energy + c_edge
            if new_energy > energy_budget:
                continue

            new_distance = current.distance + d_edge
            new_priority = new_distance + float(heuristic(v))
            new_label = _Label(
                priority=new_priority,
                distance=new_distance,
                energy=new_energy,
                node=v,
                parent=current,
            )

            # Dominance check at node v.
            existing = labels_at_node.get(v, [])
            dominated = False
            survivors: List[_Label] = []
            for lab in existing:
                # lab dominates new_label
                if lab.distance <= new_label.distance and lab.energy <= new_label.energy:
                    dominated = True
                    break
                # new_label dominates lab
                if new_label.distance <= lab.distance and new_label.energy <= lab.energy:
                    continue
                survivors.append(lab)

            if dominated:
                continue

            survivors.append(new_label)
            labels_at_node[v] = survivors
            heapq.heappush(frontier, new_label)

    # No feasible path within the given energy budget.
    return SearchResult(path=None, distance=None, energy=None, expanded_nodes=expanded)


def _reconstruct_path_from_label(label: _Label) -> List[str]:
    path: List[str] = []
    cur: Optional[_Label] = label
    while cur is not None:
        path.append(cur.node)
        cur = cur.parent
    path.reverse()
    return path


def ucs_shortest_path_with_energy(
    G: Graph,
    Dist: EdgeWeights,
    Cost: EdgeWeights,
    start: str,
    goal: str,
    energy_budget: float,
) -> SearchResult:
    """Uniform Cost Search with an energy budget.

    This is implemented as label-setting RCSP search with a zero heuristic,
    which reduces to UCS over distance while enforcing the energy
    constraint via label pruning.
    """

    def zero_heuristic(_node: str) -> float:
        return 0.0

    return _rcsp_search(G, Dist, Cost, start, goal, energy_budget, zero_heuristic)


def astar_shortest_path_with_energy(
    G: Graph,
    Coord: CoordDict,
    Dist: EdgeWeights,
    Cost: EdgeWeights,
    start: str,
    goal: str,
    energy_budget: float,
) -> SearchResult:
    """A* search with Euclidean heuristic and energy budget.

    Uses a label-setting approach with Pareto-optimal labels in the
    (distance, energy) space. The priority key is ``f = g + h``, where
    ``g`` is the distance-so-far and ``h`` is the Euclidean straight-line
    distance from the current node to the goal.
    """

    def heuristic(node: str) -> float:
        return _euclidean_heuristic(Coord, node, goal)

    return _rcsp_search(G, Dist, Cost, start, goal, energy_budget, heuristic)
