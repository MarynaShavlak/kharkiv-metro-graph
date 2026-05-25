"""Аналіз структури графа метро: статистика, центральності, планарність."""

from typing import Any, Dict, List, Tuple

import networkx as nx

from .data import Line


def basic_stats(G: nx.Graph) -> Dict[str, Any]:
    """Базові характеристики графа."""
    return {
        "Кількість станцій": G.number_of_nodes(),
        "Кількість з'єднань": G.number_of_edges(),
        "Щільність": f"{nx.density(G):.4f}",
        "Зв'язний": "Так" if nx.is_connected(G) else "Ні",
    }


def degree_stats(G: nx.Graph) -> Dict[str, Any]:
    """Розподіл ступенів вершин."""
    degrees = [d for _, d in G.degree()]
    return {
        "Середній": f"{sum(degrees) / len(degrees):.2f}",
        "Максимальний": max(degrees),
        "Мінімальний": min(degrees),
    }


def top_betweenness(G: nx.Graph, n: int = 5) -> List[Tuple[str, float]]:
    """ТОП-n станцій за посередництвом (betweenness centrality).

    Чим вище значення — тим більше найкоротших шляхів проходить через станцію.
    """
    bc = nx.betweenness_centrality(G)
    return sorted(bc.items(), key=lambda kv: kv[1], reverse=True)[:n]


def structure_stats(G: nx.Graph) -> Dict[str, Any]:
    """Структурні характеристики: діаметр і середня відстань."""
    return {
        "Діаметр (перегонів)": nx.diameter(G),
        "Середня відстань": f"{nx.average_shortest_path_length(G):.2f}",
    }


def line_distribution(lines: List[Line]) -> Dict[str, int]:
    """Розподіл станцій між лініями."""
    return {f"{line.code} ({line.name})": len(line.stations) for line in lines}


def planarity_info(G: nx.Graph) -> Dict[str, Any]:
    """Інформація про планарність графа та формулу Ейлера.

    Returns:
        is_planar: чи планарний граф.
        V, E, F: число вершин, ребер і граней.
        euler: значення V − E + F (має бути 2 для зв'язного планарного графа).
        cycles: базис циклів.
    """
    is_planar, _ = nx.check_planarity(G)
    V, E = G.number_of_nodes(), G.number_of_edges()
    F = 2 - V + E
    return {
        "is_planar": is_planar,
        "V": V,
        "E": E,
        "F": F,
        "euler": V - E + F,
        "cycles": nx.cycle_basis(G),
    }


# ---------------------------------------------------------------------------
# Друк
# ---------------------------------------------------------------------------

def _print_section(emoji: str, title: str, items: Dict[str, Any]) -> None:
    print(f"\n{emoji} {title}:")
    for k, v in items.items():
        print(f"   {k}: {v}")


def print_analysis(G: nx.Graph, lines: List[Line]) -> None:
    """Друкує повний звіт з аналізу мережі."""
    print("=" * 60)
    print("АНАЛІЗ МЕРЕЖІ ХАРКІВСЬКОГО МЕТРО")
    print("=" * 60)

    _print_section("📊", "Базові характеристики", basic_stats(G))
    _print_section("📈", "Ступені вершин", degree_stats(G))

    print("\n🚪 ТОП-5 ключових станцій (за посередництвом):")
    for st, score in top_betweenness(G):
        print(f"   {st}: {score:.4f}")

    _print_section("🗺️ ", "Структура", structure_stats(G))
    _print_section("🚇", "Розподіл станцій", line_distribution(lines))
    print("=" * 60)


def print_planarity(G: nx.Graph) -> None:
    """Друкує аналіз планарності та формулу Ейлера."""
    info = planarity_info(G)
    V, E, F = info["V"], info["E"], info["F"]
    print(f"Граф планарний: {info['is_planar']}")
    print(f"V = {V},  E = {E},  F = {F}")
    print(
        f"Формула Ейлера: V − E + F = {V} − {E} + {F} = {info['euler']}  ✓"
    )
    print(f"\nЦиклів у базисі: {len(info['cycles'])}")
    for i, c in enumerate(info["cycles"], 1):
        print(f"  Цикл {i} ({len(c)} вершин): {' → '.join(c)} → {c[0]}")
