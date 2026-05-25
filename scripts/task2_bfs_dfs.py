"""Завдання 2: пошук шляхів через BFS і DFS, порівняння результатів.

Запуск:
    python scripts/task2_bfs_dfs.py
"""

from pathlib import Path

from kharkiv_metro import (
    LINES,
    TRANSFERS,
    bfs_path,
    build_graph,
    compute_geographic_layout,
    dfs_path,
    visualize_bfs_dfs,
)

IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)


# Тестові пари — підібрані так, щоб покрити різні топологічні ситуації.
# Деталі добору описані в README та в коментарях у ноутбуці.
TEST_PAIRS = [
    ("Холодна гора", "Салтівська"),         # через увесь центр
    ("Університет", "Спортивна"),           # обхід циклу з різних боків
    ("Перемога", "Індустріальна"),          # кінець-в-кінець через все місто
    ("Вокзальна", "Тракторний завод"),      # обидві на M1
]


def print_comparison_table(G):
    """Друкує таблицю порівняння BFS і DFS по тестових парах."""
    results = []
    for s, e in TEST_PAIRS:
        bfs = bfs_path(G, s, e)
        dfs = dfs_path(G, s, e)
        same = "так" if bfs == dfs else "ні"
        results.append((f"{s} → {e}", len(bfs) - 1, len(dfs) - 1, same))

    w = max(len(r[0]) for r in results)
    header = f"{'Маршрут':<{w}}  {'BFS':>4}  {'DFS':>4}  Збіг"
    print(header)
    print("=" * len(header))
    for route, bfs_len, dfs_len, same in results:
        print(f"{route:<{w}}  {bfs_len:>4}  {dfs_len:>4}  {same}")


def print_detailed_example(G, start, end):
    """Друкує повний маршрут обох алгоритмів для однієї пари."""
    print(f"\nДеталі: {start} → {end}\n")
    bfs = bfs_path(G, start, end)
    dfs = dfs_path(G, start, end)
    print(f"BFS ({len(bfs) - 1} перегонів):")
    print("  " + " → ".join(bfs))
    print(f"\nDFS ({len(dfs) - 1} перегонів):")
    print("  " + " → ".join(dfs))


def main() -> None:
    G = build_graph(LINES, TRANSFERS)
    pos = compute_geographic_layout(LINES)

    print("=" * 60)
    print("BFS vs DFS — порівняння на чотирьох тестових парах")
    print("=" * 60)
    print_comparison_table(G)
    print_detailed_example(G, "Університет", "Спортивна")

    # Дві ключові візуалізації
    visualize_bfs_dfs(
        G, LINES, TRANSFERS, pos,
        "Університет", "Спортивна",
        savepath=str(IMAGES_DIR / "04_bfs_vs_dfs_universitet.png"),
    )
    visualize_bfs_dfs(
        G, LINES, TRANSFERS, pos,
        "Вокзальна", "Тракторний завод",
        savepath=str(IMAGES_DIR / "05_bfs_vs_dfs_vokzalna.png"),
    )


if __name__ == "__main__":
    main()
