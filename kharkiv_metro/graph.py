"""Побудова графів NetworkX зі статичних даних метро."""

from typing import Dict, List, Set, Tuple

import networkx as nx

from .data import Line, Transfer


def build_indexes(
    lines: List[Line],
    transfers: List[Transfer],
) -> Tuple[Dict[str, str], Dict[str, Line], Set[str]]:
    """Передобчислює мапи для швидкого пошуку O(1).

    Returns:
        color_of: станція → hex-колір її лінії.
        line_of: станція → об'єкт Line, якому вона належить.
        transfer_stations: множина всіх пересадкових станцій.
    """
    color_of = {st: line.color for line in lines for st in line.stations}
    line_of = {st: line for line in lines for st in line.stations}
    transfer_stations = {s for t in transfers for s in t.stations}
    return color_of, line_of, transfer_stations


def build_graph(lines: List[Line], transfers: List[Transfer]) -> nx.Graph:
    """Будує незважений граф мережі: вершини = станції, ребра = перегони + пересадки."""
    G = nx.Graph()
    for line in lines:
        G.add_nodes_from(line.stations)
        G.add_edges_from(zip(line.stations, line.stations[1:]))
    for t in transfers:
        G.add_edge(*t.stations)
    return G


def build_weighted_graph(
    lines: List[Line],
    transfers: List[Transfer],
    weights: Dict[Tuple[str, str], int],
) -> nx.Graph:
    """Будує зважений граф з вагою кожного ребра = час руху (хв).

    Args:
        lines: список ліній метро.
        transfers: список пересадок.
        weights: мапа (u, v) → час у хвилинах. Порядок вершин не важливий —
            пошук виконується в обидвох напрямках.
    """
    G = nx.Graph()
    for line in lines:
        G.add_nodes_from(line.stations)
        for u, v in zip(line.stations, line.stations[1:]):
            w = weights.get((u, v)) or weights[(v, u)]
            G.add_edge(u, v, weight=w)
    for t in transfers:
        u, v = t.stations
        w = weights.get((u, v)) or weights[(v, u)]
        G.add_edge(u, v, weight=w)
    return G
