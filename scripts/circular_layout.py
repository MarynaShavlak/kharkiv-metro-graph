"""Альтернативна схема: пересадки як окремі станції на 9-вершинному полігоні.

Запуск:
    python scripts/circular_layout.py
"""

from pathlib import Path

from kharkiv_metro import (
    LINES,
    TRANSFERS,
    build_graph,
    compute_circular_layout,
    draw_circular_schematic,
)

IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)


def main() -> None:
    G = build_graph(LINES, TRANSFERS)
    pos = compute_circular_layout(LINES)

    print(f"Stations: {G.number_of_nodes()}")
    print(f"Edges:    {G.number_of_edges()}")

    draw_circular_schematic(
        G, LINES, TRANSFERS, pos,
        savepath=str(IMAGES_DIR / "09_circular_schema.png"),
    )


if __name__ == "__main__":
    main()
