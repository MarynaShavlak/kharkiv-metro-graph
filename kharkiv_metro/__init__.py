"""Аналіз графа Харківського метрополітену.

Публічне API:
    LINES, TRANSFERS, EDGE_WEIGHTS — статичні дані мережі.
    build_graph, build_weighted_graph — побудова графів.
    compute_geographic_layout, compute_circular_layout — розклади координат.
    bfs_path, dfs_path, shortest_path — алгоритми пошуку шляхів.
    print_analysis, print_planarity — друк звітів.
    draw_metro, draw_planarity, draw_fragility_map тощо — рендер.
"""

from .algorithms import (
    all_pairs_shortest_times,
    bfs_path,
    dfs_path,
    dijkstra,
    path_time,
    reconstruct_path,
    shortest_path,
)
from .analysis import (
    basic_stats,
    degree_stats,
    line_distribution,
    planarity_info,
    print_analysis,
    print_planarity,
    structure_stats,
    top_betweenness,
)
from .data import EDGE_WEIGHTS, LINES, TRANSFERS, Line, Transfer
from .graph import build_graph, build_indexes, build_weighted_graph
from .layout import compute_circular_layout, compute_geographic_layout
from .visualization import (
    draw_circular_schematic,
    draw_fragility_map,
    draw_metro,
    draw_planarity,
    draw_time_heatmap,
    visualize_bfs_dfs,
    visualize_bfs_vs_dijkstra,
    visualize_dijkstra,
)

__all__ = [
    # data
    "Line", "Transfer", "LINES", "TRANSFERS", "EDGE_WEIGHTS",
    # graph
    "build_graph", "build_weighted_graph", "build_indexes",
    # layout
    "compute_geographic_layout", "compute_circular_layout",
    # algorithms
    "bfs_path", "dfs_path", "dijkstra", "shortest_path",
    "reconstruct_path", "all_pairs_shortest_times", "path_time",
    # analysis
    "basic_stats", "degree_stats", "top_betweenness", "structure_stats",
    "line_distribution", "planarity_info",
    "print_analysis", "print_planarity",
    # visualization
    "draw_metro", "draw_planarity", "draw_fragility_map",
    "draw_time_heatmap", "draw_circular_schematic",
    "visualize_bfs_dfs", "visualize_dijkstra", "visualize_bfs_vs_dijkstra",
]

__version__ = "0.1.0"
