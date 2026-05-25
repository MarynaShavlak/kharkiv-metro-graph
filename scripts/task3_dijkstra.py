"""Завдання 3: алгоритм Дейкстри на зваженому графі.

Запуск:
    python scripts/task3_dijkstra.py
"""

from pathlib import Path

from kharkiv_metro import (
    EDGE_WEIGHTS,
    LINES,
    TRANSFERS,
    all_pairs_shortest_times,
    bfs_path,
    build_weighted_graph,
    compute_geographic_layout,
    draw_time_heatmap,
    path_time,
    shortest_path,
    visualize_bfs_vs_dijkstra,
    visualize_dijkstra,
)

IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)


TEST_PAIRS = [
    ("Холодна гора", "Салтівська"),
    ("Університет", "Спортивна"),
    ("Перемога", "Індустріальна"),
    ("Вокзальна", "Тракторний завод"),
    ("Холодна гора", "Індустріальна"),
]


def print_dijkstra_table(GW):
    """Час та число перегонів для тестових пар."""
    results = []
    for s, e in TEST_PAIRS:
        path, t = shortest_path(GW, s, e)
        results.append((f"{s} → {e}", t, len(path) - 1))

    w = max(len(r[0]) for r in results)
    header = f"{'Маршрут':<{w}}  {'Час':>5}  {'Перегонів':>10}"
    print(header)
    print("=" * len(header))
    for route, t, n in results:
        print(f"{route:<{w}}  {t:>3} хв  {n:>10}")


def print_detailed_route(GW, start, end):
    """Друкує маршрут перегон за перегоном з вагами."""
    path, total = shortest_path(GW, start, end)
    print(f"\nДетально: {start} → {end}  (загальний час {total} хв)\n")
    for u, v in zip(path, path[1:]):
        print(f"  {u}  ──{GW[u][v]['weight']} хв──→  {v}")


def print_weighted_diameter(GW):
    """Найдовший найкоротший шлях за часом."""
    all_times = all_pairs_shortest_times(GW)
    diam_pair = max(
        ((s, e) for s in all_times for e in all_times[s] if s != e),
        key=lambda p: all_times[p[0]][p[1]],
    )
    print(
        f"\nЗважений діаметр: {diam_pair[0]} → {diam_pair[1]} "
        f"= {all_times[diam_pair[0]][diam_pair[1]]} хв"
    )


def print_bfs_dijkstra_diffs(GW):
    """Знаходить пари, де BFS і Дейкстра дають різні маршрути."""
    diff_pairs = []
    nodes = list(GW.nodes())
    for i, s in enumerate(nodes):
        for e in nodes[i + 1:]:
            bfs = bfs_path(GW, s, e)
            dijk, _ = shortest_path(GW, s, e)
            if bfs != dijk and bfs[::-1] != dijk:
                diff_pairs.append(
                    (
                        s, e,
                        len(bfs) - 1, path_time(GW, bfs),
                        len(dijk) - 1, path_time(GW, dijk),
                    )
                )

    total_pairs = len(nodes) * (len(nodes) - 1) // 2
    print(
        f"\nПар з різними маршрутами: {len(diff_pairs)} "
        f"з {total_pairs} можливих\n"
    )
    print(f"{'Пара':<55} {'BFS':>14}  {'Дейкстра':>14}  {'Економія':>9}")
    print("=" * 100)
    for s, e, n_b, t_b, n_d, t_d in diff_pairs:
        pair = f"{s} → {e}"
        print(
            f"{pair:<55} {n_b} рб, {t_b:>2} хв     "
            f"{n_d} рб, {t_d:>2} хв   {t_b - t_d:>4} хв"
        )


def main() -> None:
    GW = build_weighted_graph(LINES, TRANSFERS, EDGE_WEIGHTS)
    pos = compute_geographic_layout(LINES)

    print(
        f"Зважений граф: {GW.number_of_nodes()} вершин, "
        f"{GW.number_of_edges()} ребер"
    )

    print("\n" + "=" * 60)
    print("ДЕЙКСТРА: найкоротші маршрути за часом")
    print("=" * 60)
    print_dijkstra_table(GW)
    print_detailed_route(GW, "Холодна гора", "Салтівська")
    print_weighted_diameter(GW)

    # Візуалізація одного маршруту
    visualize_dijkstra(
        GW, LINES, TRANSFERS, pos,
        "Холодна гора", "Салтівська",
        savepath=str(IMAGES_DIR / "06_dijkstra_route.png"),
    )

    # Теплова карта часів
    draw_time_heatmap(
        GW, LINES,
        savepath=str(IMAGES_DIR / "07_time_heatmap.png"),
    )

    # Порівняння BFS vs Дейкстра
    print("\n" + "=" * 60)
    print("BFS vs ДЕЙКСТРА — де результати розходяться")
    print("=" * 60)
    print_bfs_dijkstra_diffs(GW)

    visualize_bfs_vs_dijkstra(
        GW, LINES, TRANSFERS, pos,
        "Майдан Конституції", "Архітектора Бекетова",
        savepath=str(IMAGES_DIR / "08_bfs_vs_dijkstra.png"),
    )


if __name__ == "__main__":
    main()
