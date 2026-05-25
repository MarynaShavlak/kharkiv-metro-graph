"""Завдання 1: побудова графа, візуалізація, аналіз мережі та планарності.

Запуск:
    python scripts/task1_network_analysis.py
"""

from pathlib import Path

from kharkiv_metro import (
    LINES,
    TRANSFERS,
    build_graph,
    compute_geographic_layout,
    draw_fragility_map,
    draw_metro,
    draw_planarity,
    print_analysis,
    print_planarity,
)

IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)


def main() -> None:
    G = build_graph(LINES, TRANSFERS)
    pos = compute_geographic_layout(LINES)

    # 1. Схема метро
    draw_metro(
        G, LINES, TRANSFERS, pos,
        savepath=str(IMAGES_DIR / "01_metro_schema.png"),
    )

    # 2. Аналіз мережі — статистика, центральності
    print_analysis(G, LINES)

    # 3. Планарність + формула Ейлера
    print("\n" + "=" * 60)
    print("ПЛАНАРНІСТЬ")
    print("=" * 60)
    print_planarity(G)
    draw_planarity(
        G, LINES, TRANSFERS,
        savepath=str(IMAGES_DIR / "02_planarity.png"),
    )

    # 4. Підсумкова карта крихкості мережі
    draw_fragility_map(
        G, LINES, TRANSFERS, pos,
        savepath=str(IMAGES_DIR / "03_fragility_map.png"),
    )


if __name__ == "__main__":
    main()
