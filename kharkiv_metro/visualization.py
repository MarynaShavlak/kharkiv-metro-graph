"""Рендер схем і візуалізацій усіх алгоритмів."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Wedge

from .algorithms import (
    all_pairs_shortest_times,
    bfs_path,
    dfs_path,
    path_time,
    shortest_path,
)
from .data import Line, Transfer
from .graph import build_indexes
from .layout import CircularLayout, compute_geographic_layout


# ===========================================================================
# Загальні стилі та геометричні метадані
# ===========================================================================

class Style:
    """Налаштування стилю для рендеру."""

    FIGSIZE = (22, 16)
    STATION_SIZE = 280
    STATION_EDGE_WIDTH = 3
    LINE_WIDTH = 7
    LINE_ALPHA = 0.85
    LABEL_FONTSIZE = 9
    TRANSFER_FONTSIZE = 10
    TITLE_FONTSIZE = 18
    LEGEND_FONTSIZE = 12


# Геометрія пересадкових кружечків для географічного layout-у.
# split_angle — кут поділу wedge-у; label_offsets — (dx, dy, ha, va) для підписів.
TRANSFER_GEOMETRY: Dict[Tuple[str, str], Dict] = {
    ("Майдан Конституції", "Історичний музей"): {
        "split_angle": 45,
        "label_offsets": {
            "Майдан Конституції": (-0.65, -0.35, "right", "top"),
            "Історичний музей": (0.70, 1.80, "right", "bottom"),
        },
    },
    ("Університет", "Держпром"): {
        "split_angle": 0,
        "label_offsets": {
            "Університет": (-0.70, 0.50, "right", "bottom"),
            "Держпром": (0.70, -0.40, "left", "bottom"),
        },
    },
    ("Спортивна", "Метробудівників"): {
        "split_angle": 135,
        "label_offsets": {
            "Спортивна": (-0.65, -0.35, "right", "top"),
            "Метробудівників": (0.60, 0.60, "left", "top"),
        },
    },
}

# Підписи окремих станцій, що потребують ручної правки
SPECIAL_LABELS: Dict[str, Tuple[float, float, str, str]] = {
    "Ярослава Мудрого": (0.55, -0.45, "left", "bottom"),
}


# ===========================================================================
# Базові примітиви рендеру
# ===========================================================================

def setup_axes(figsize: Tuple[float, float] = Style.FIGSIZE):
    """Створює fig/ax з білим тлом, без осей, з рівним масштабом."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.margins(0.12)
    return fig, ax


def draw_lines(ax, G, pos, lines: List[Line]) -> None:
    """Малює сегменти ліній метро у власному кольорі."""
    for line in lines:
        edges = list(zip(line.stations, line.stations[1:]))
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax,
            edge_color=line.color, width=Style.LINE_WIDTH,
            alpha=Style.LINE_ALPHA,
        )


def draw_regular_stations(ax, G, pos, transfer_stations, color_of) -> None:
    """Малює звичайні (не пересадкові) станції — білі кружечки з обводкою."""
    for st in G.nodes():
        if st in transfer_stations:
            continue
        x, y = pos[st]
        ax.scatter(
            x, y, s=Style.STATION_SIZE, c="white",
            edgecolors=color_of[st],
            linewidths=Style.STATION_EDGE_WIDTH, zorder=3,
        )


def draw_transfer_circles(ax, pos, transfers: List[Transfer], color_of) -> None:
    """Малює пересадкові станції як двокольорові кружечки (wedge'и)."""
    for t in transfers:
        geom = TRANSFER_GEOMETRY[t.stations]
        s1, s2 = t.stations
        center = pos[s1]
        for half_offset, st in [(0, s1), (180, s2)]:
            ax.add_patch(
                Wedge(
                    center, 0.4,
                    geom["split_angle"] + half_offset,
                    geom["split_angle"] + half_offset + 180,
                    facecolor=color_of[st], edgecolor="black",
                    linewidth=2, zorder=5,
                )
            )


# ===========================================================================
# Підписи
# ===========================================================================

def compute_label_position(
    station: str, pos, line_of, transfers: List[Transfer]
) -> Tuple[float, float, str, str]:
    """Повертає (x, y, ha, va) для підпису станції в географічному layout."""
    x, y = pos[station]

    # Пересадки: офсет рахується від центра пересадкового кружка
    for t in transfers:
        if station in t.stations:
            cx, cy = pos[t.stations[0]]
            dx, dy, ha, va = TRANSFER_GEOMETRY[t.stations]["label_offsets"][station]
            return cx + dx, cy + dy, ha, va

    # Спецвипадки
    if station in SPECIAL_LABELS:
        dx, dy, ha, va = SPECIAL_LABELS[station]
        return x + dx, y + dy, ha, va

    # За кодом лінії
    code = line_of[station].code
    if code == "M1":   # червона ↘ → підпис вниз-вліво (під лінією)
        return x - 0.15, y - 0.35, "right", "top"
    if code == "M2":   # синя ↗ → підпис вправо-нижче (під лінією)
        return x + 0.45, y - 0.35, "left", "bottom"
    return x + 0.4, y, "left", "center"  # зелена: праворуч


def _label_style(is_transfer: bool, color: str) -> Dict:
    if is_transfer:
        return {
            "fontsize": Style.TRANSFER_FONTSIZE,
            "fontweight": "bold",
            "color": color,
            "bbox": dict(
                boxstyle="round,pad=0.3", facecolor="white",
                edgecolor=color, alpha=0.95, linewidth=1.5,
            ),
        }
    return {
        "fontsize": Style.LABEL_FONTSIZE,
        "fontweight": "normal",
        "color": "black",
        "bbox": dict(
            boxstyle="round,pad=0.15", facecolor="white",
            edgecolor="none", alpha=0.85,
        ),
    }


def draw_labels(ax, pos, line_of, color_of, transfers, transfer_stations) -> None:
    for st in pos:
        lx, ly, ha, va = compute_label_position(st, pos, line_of, transfers)
        style = _label_style(st in transfer_stations, color_of[st])
        ax.text(lx, ly, st, ha=ha, va=va, zorder=6, **style)


def draw_legend(ax, lines: List[Line]) -> None:
    handles = [
        Line2D(
            [0], [0], color=line.color, linewidth=Style.LINE_WIDTH,
            label=f"{line.code} — {line.name}",
        )
        for line in lines
    ]
    ax.legend(
        handles=handles, loc="upper right",
        fontsize=Style.LEGEND_FONTSIZE, framealpha=0.95, edgecolor="gray",
    )


def draw_title(ax, G, n_transfers: int) -> None:
    ax.set_title(
        f"Схема Харківського метрополітену\n"
        f"{G.number_of_nodes()} станцій • "
        f"{G.number_of_edges()} з'єднань • 3 лінії • {n_transfers} пересадки",
        fontsize=Style.TITLE_FONTSIZE, fontweight="bold", pad=20,
    )


# ===========================================================================
# Головні функції рендеру схеми
# ===========================================================================

def draw_metro(
    G, lines: List[Line], transfers: List[Transfer], pos,
    savepath: Optional[str] = None,
) -> None:
    """Малює повну схему метро у географічному layout-і."""
    color_of, line_of, transfer_stations = build_indexes(lines, transfers)

    fig, ax = setup_axes()
    draw_lines(ax, G, pos, lines)
    draw_regular_stations(ax, G, pos, transfer_stations, color_of)
    draw_transfer_circles(ax, pos, transfers, color_of)
    draw_labels(ax, pos, line_of, color_of, transfers, transfer_stations)
    draw_legend(ax, lines)
    draw_title(ax, G, len(transfers))
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ===========================================================================
# Планарність
# ===========================================================================

def _unique_consecutive(coords):
    """Видаляє послідовні дублікати — пересадки мають однакові координати."""
    coords = [tuple(c) for c in coords]
    return [c for i, c in enumerate(coords) if i == 0 or coords[i - 1] != c]


def draw_planarity(
    G, lines: List[Line], transfers: List[Transfer],
    savepath: Optional[str] = None,
) -> None:
    """Дві панелі: географічний layout з підсвіченою гранню + алгоритмічний планарний."""
    color_of, _, _ = build_indexes(lines, transfers)
    cycle = nx.cycle_basis(G)[0]
    transfer_edges = [t.stations for t in transfers]
    V, E = G.number_of_nodes(), G.number_of_edges()
    F = 2 - V + E

    fig, axes = plt.subplots(1, 2, figsize=(22, 11), facecolor="white")

    # === ЛІВА: географічний layout з підсвіченою гранню ===
    ax = axes[0]
    pos_geo = compute_geographic_layout(lines)
    face = _unique_consecutive([pos_geo[st] for st in cycle])
    ax.add_patch(
        Polygon(
            face, facecolor="#FFD93D", edgecolor="#E8A317",
            alpha=0.4, linewidth=2.5, zorder=1,
            label="Обмежена грань F₁",
        )
    )
    for line in lines:
        nx.draw_networkx_edges(
            G, pos_geo, ax=ax,
            edgelist=list(zip(line.stations, line.stations[1:])),
            edge_color=line.color, width=5, alpha=0.9,
        )
    nx.draw_networkx_edges(
        G, pos_geo, edgelist=transfer_edges, ax=ax,
        edge_color="#222", width=2, style="dashed", alpha=0.9,
    )
    nx.draw_networkx_nodes(
        G, pos_geo, ax=ax, node_size=180, node_color="white",
        edgecolors=[color_of[n] for n in G.nodes()], linewidths=2,
    )
    nx.draw_networkx_labels(
        G, pos_geo, ax=ax,
        labels={st: st for st in cycle},
        font_size=8, font_weight="bold",
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white",
            edgecolor="gray", alpha=0.9,
        ),
    )
    ax.set_title(
        "Географічний layout\nЖовтим — обмежена грань (єдиний цикл)",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.margins(0.1)

    # === ПРАВА: алгоритмічний планарний layout ===
    ax = axes[1]
    pos_planar = nx.planar_layout(G, scale=2)
    for line in lines:
        nx.draw_networkx_edges(
            G, pos_planar, ax=ax,
            edgelist=list(zip(line.stations, line.stations[1:])),
            edge_color=line.color, width=3, alpha=0.9,
        )
    nx.draw_networkx_edges(
        G, pos_planar, edgelist=transfer_edges, ax=ax,
        edge_color="#222", width=1.5, style="dashed", alpha=0.9,
    )
    nx.draw_networkx_nodes(
        G, pos_planar, ax=ax, node_size=120, node_color="white",
        edgecolors=[color_of[n] for n in G.nodes()], linewidths=1.5,
    )
    nx.draw_networkx_labels(
        G, pos_planar, ax=ax, font_size=6.5,
        bbox=dict(
            boxstyle="round,pad=0.1", facecolor="white",
            edgecolor="none", alpha=0.85,
        ),
    )
    ax.set_title(
        "Алгоритмічний планарний layout (nx.planar_layout)\n"
        "Те ж розкладання без географії — жодного перетину",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.set_axis_off()
    ax.margins(0.15)

    fig.suptitle(
        f"Планарність графа Харківського метрополітену\n"
        f"V = {V},  E = {E},  F = {F}    →    "
        f"формула Ейлера:  V − E + F = {V}−{E}+{F} = 2  ✓",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ===========================================================================
# Візуалізація BFS / DFS
# ===========================================================================

def visualize_bfs_dfs(
    G, lines: List[Line], transfers: List[Transfer], pos,
    start: str, end: str, savepath: Optional[str] = None,
) -> None:
    """Накладає шляхи BFS і DFS на схему мережі."""
    color_of, _, _ = build_indexes(lines, transfers)
    bfs = bfs_path(G, start, end)
    dfs = dfs_path(G, start, end)
    bfs_edges = list(zip(bfs, bfs[1:]))
    dfs_edges = list(zip(dfs, dfs[1:]))

    fig, ax = plt.subplots(figsize=(16, 12), facecolor="white")

    # Сіра підкладка — вся мережа
    for line in lines:
        edges = list(zip(line.stations, line.stations[1:]))
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax,
            edge_color="lightgray", width=4, alpha=0.6,
        )
    transfer_edges = [t.stations for t in transfers]
    nx.draw_networkx_edges(
        G, pos, edgelist=transfer_edges, ax=ax,
        edge_color="lightgray", width=2, style="dashed", alpha=0.6,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=140, node_color="white",
        edgecolors=[color_of[n] for n in G.nodes()], linewidths=1.5,
    )

    # DFS товстіший знизу, BFS поверх
    nx.draw_networkx_edges(
        G, pos, edgelist=dfs_edges, ax=ax,
        edge_color="#F39C12", width=10, alpha=0.85,
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=bfs_edges, ax=ax,
        edge_color="#27AE60", width=6, alpha=0.95,
    )

    # Старт / фініш
    for st, color in [(start, "#2C3E50"), (end, "#C0392B")]:
        x, y = pos[st]
        ax.scatter(
            x, y, s=500, c=color, edgecolors="white",
            linewidths=3, zorder=10,
        )

    # Підписи лише для задіяних станцій
    used = set(bfs) | set(dfs)
    nx.draw_networkx_labels(
        G, pos, labels={st: st for st in used}, ax=ax,
        font_size=8, font_weight="bold",
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white",
            edgecolor="gray", alpha=0.9,
        ),
    )

    legend_handles = [
        Line2D(
            [0], [0], color="#27AE60", linewidth=6,
            label=f"BFS — {len(bfs)-1} перегонів (найкоротший)",
        ),
        Line2D(
            [0], [0], color="#F39C12", linewidth=8,
            label=f"DFS — {len(dfs)-1} перегонів",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#2C3E50",
            markersize=15, label=f"Старт: {start}",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#C0392B",
            markersize=15, label=f"Фініш: {end}",
        ),
    ]
    ax.legend(
        handles=legend_handles, loc="upper right", fontsize=11,
        framealpha=0.95, edgecolor="gray",
    )
    ax.set_title(
        f"BFS vs DFS:  {start}  →  {end}",
        fontsize=16, fontweight="bold", pad=15,
    )
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.margins(0.1)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ===========================================================================
# Візуалізація Дейкстри
# ===========================================================================

def visualize_dijkstra(
    G, lines: List[Line], transfers: List[Transfer], pos,
    start: str, end: str, savepath: Optional[str] = None,
) -> None:
    """Малює найкоротший за часом шлях, що знайшла Дейкстра."""
    color_of, _, _ = build_indexes(lines, transfers)
    path, total = shortest_path(G, start, end)
    path_edges = list(zip(path, path[1:]))

    fig, ax = plt.subplots(figsize=(18, 13), facecolor="white")

    # Сіра підкладка
    for line in lines:
        edges = list(zip(line.stations, line.stations[1:]))
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax,
            edge_color="lightgray", width=4, alpha=0.5,
        )
    transfer_edges = [t.stations for t in transfers]
    nx.draw_networkx_edges(
        G, pos, edgelist=transfer_edges, ax=ax,
        edge_color="lightgray", width=2, style="dashed", alpha=0.5,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=140, node_color="white",
        edgecolors=[color_of[n] for n in G.nodes()], linewidths=1.5,
    )

    # Маршрут
    nx.draw_networkx_edges(
        G, pos, edgelist=path_edges, ax=ax,
        edge_color="#8E44AD", width=8, alpha=0.95,
    )

    # Ваги ребер маршруту — текст у середині кожного ребра
    for u, v in path_edges:
        mx = (pos[u][0] + pos[v][0]) / 2
        my = (pos[u][1] + pos[v][1]) / 2
        ax.text(
            mx, my, f"{G[u][v]['weight']} хв",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#8E44AD", zorder=8,
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="#F5E8FA",
                edgecolor="#8E44AD", linewidth=1,
            ),
        )

    # Підписи станцій маршруту
    nx.draw_networkx_labels(
        G, pos, labels={st: st for st in path}, ax=ax,
        font_size=8, font_weight="bold",
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white",
            edgecolor="gray", alpha=0.95,
        ),
    )
    # Старт / фініш
    for st, col in [(start, "#2C3E50"), (end, "#C0392B")]:
        x, y = pos[st]
        ax.scatter(
            x, y, s=500, c=col, edgecolors="white",
            linewidths=3, zorder=10,
        )

    legend_handles = [
        Line2D(
            [0], [0], color="#8E44AD", linewidth=7,
            label=f"Найкоротший маршрут — {total} хв ({len(path)-1} перегонів)",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#2C3E50",
            markersize=15, label=f"Старт: {start}",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#C0392B",
            markersize=15, label=f"Фініш: {end}",
        ),
    ]
    ax.legend(
        handles=legend_handles, loc="upper right", fontsize=11,
        framealpha=0.95, edgecolor="gray",
    )
    ax.set_title(
        f"Дейкстра:  {start}  →  {end}  =  {total} хв",
        fontsize=16, fontweight="bold", pad=15,
    )
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.margins(0.1)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ===========================================================================
# Теплова карта часів
# ===========================================================================

def draw_time_heatmap(
    G, lines: List[Line], savepath: Optional[str] = None,
) -> None:
    """Матриця V×V найкоротших часів між усіма парами станцій."""
    stations, seen = [], set()
    for line in lines:
        for st in line.stations:
            if st not in seen:
                stations.append(st)
                seen.add(st)

    all_times = all_pairs_shortest_times(G)
    n = len(stations)
    M = np.array([[all_times[s][e] for e in stations] for s in stations])

    fig, ax = plt.subplots(figsize=(14, 12), facecolor="white")
    im = ax.imshow(M, cmap="YlOrRd", aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(stations, rotation=90, fontsize=7)
    ax.set_yticklabels(stations, fontsize=7)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            if M[i, j] == 0:
                continue
            color = "white" if M[i, j] > M.max() * 0.55 else "black"
            ax.text(
                j, i, int(M[i, j]), ha="center", va="center",
                fontsize=6, color=color,
            )

    plt.colorbar(im, ax=ax, shrink=0.7).set_label(
        "Час у дорозі (хв)", fontsize=11
    )
    ax.set_title(
        "Матриця найкоротших часів між усіма станціями\n"
        "(алгоритм Дейкстри, ваги = час руху + пересадки)",
        fontsize=13, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ===========================================================================
# BFS vs Дейкстра
# ===========================================================================

def visualize_bfs_vs_dijkstra(
    G, lines: List[Line], transfers: List[Transfer], pos,
    start: str, end: str, savepath: Optional[str] = None,
) -> None:
    """Накладає на схему обидва маршрути: BFS (за перегонами) і Дейкстра (за часом)."""
    color_of, _, _ = build_indexes(lines, transfers)
    bfs = bfs_path(G, start, end)
    dijk, _ = shortest_path(G, start, end)
    bfs_t = path_time(G, bfs)
    dijk_t = path_time(G, dijk)
    bfs_e = list(zip(bfs, bfs[1:]))
    dijk_e = list(zip(dijk, dijk[1:]))

    fig, ax = plt.subplots(figsize=(16, 12), facecolor="white")

    # Сіра підкладка
    for line in lines:
        edges = list(zip(line.stations, line.stations[1:]))
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax,
            edge_color="lightgray", width=4, alpha=0.5,
        )
    transfer_edges = [t.stations for t in transfers]
    nx.draw_networkx_edges(
        G, pos, edgelist=transfer_edges, ax=ax,
        edge_color="lightgray", width=2, style="dashed", alpha=0.5,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=140, node_color="white",
        edgecolors=[color_of[n] for n in G.nodes()], linewidths=1.5,
    )

    # BFS під сподом, Дейкстра поверх
    nx.draw_networkx_edges(
        G, pos, edgelist=bfs_e, ax=ax,
        edge_color="#27AE60", width=10, alpha=0.85,
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=dijk_e, ax=ax,
        edge_color="#8E44AD", width=6, alpha=0.95,
    )

    # Ваги ребер обох маршрутів (об'єднання)
    drawn = set()
    for u, v in bfs_e + dijk_e:
        key = frozenset([u, v])
        if key in drawn:
            continue
        drawn.add(key)
        mx = (pos[u][0] + pos[v][0]) / 2
        my = (pos[u][1] + pos[v][1]) / 2
        ax.text(
            mx, my, f"{G[u][v]['weight']}хв",
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color="black", zorder=8,
            bbox=dict(
                boxstyle="round,pad=0.18", facecolor="#FFF8DC",
                edgecolor="#888", linewidth=0.7,
            ),
        )

    # Старт / фініш
    for st, col in [(start, "#2C3E50"), (end, "#C0392B")]:
        x, y = pos[st]
        ax.scatter(
            x, y, s=500, c=col, edgecolors="white",
            linewidths=3, zorder=10,
        )

    # Підписи задіяних станцій
    used = set(bfs) | set(dijk)
    nx.draw_networkx_labels(
        G, pos, labels={st: st for st in used}, ax=ax,
        font_size=8, font_weight="bold",
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white",
            edgecolor="gray", alpha=0.95,
        ),
    )

    legend_handles = [
        Line2D(
            [0], [0], color="#27AE60", linewidth=8,
            label=f"BFS — {len(bfs)-1} перегонів, {bfs_t} хв",
        ),
        Line2D(
            [0], [0], color="#8E44AD", linewidth=6,
            label=f"Дейкстра — {len(dijk)-1} перегонів, {dijk_t} хв",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#2C3E50",
            markersize=15, label=f"Старт: {start}",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#C0392B",
            markersize=15, label=f"Фініш: {end}",
        ),
    ]
    ax.legend(
        handles=legend_handles, loc="upper right", fontsize=11,
        framealpha=0.95, edgecolor="gray",
    )

    saved = bfs_t - dijk_t
    ax.set_title(
        f"BFS vs Дейкстра:  {start}  →  {end}\n"
        f"Дейкстра економить {saved} хв, хоча проходить "
        f"{len(dijk) - len(bfs):+d} перегонів",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.margins(0.1)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ===========================================================================
# Карта крихкості (підсумкова)
# ===========================================================================

def draw_fragility_map(
    G, lines: List[Line], transfers: List[Transfer], pos,
    top_n: int = 5, savepath: Optional[str] = None,
) -> None:
    """Підсумкова карта мережі з підсвіченою циклічною зоною та ТОП-N вузлами."""
    color_of, _, _ = build_indexes(lines, transfers)

    # 9 ребер циклу — те, що буде підсвічено
    bridges = set(map(frozenset, nx.bridges(G)))
    cycle_edges = [e for e in G.edges() if frozenset(e) not in bridges]

    bc = nx.betweenness_centrality(G)
    top_stations = sorted(bc.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    top_set = {st for st, _ in top_stations}

    def edge_color(edge):
        u, v = edge
        for line in lines:
            if u in line.stations and v in line.stations:
                return line.color
        return "#2C3E50"

    by_color = defaultdict(list)
    for e in G.edges():
        by_color[edge_color(e)].append(e)

    BG, TEXT, SUB = "#F4F4EF", "#1A1A1A", "#4B5563"
    TOP_FILL, TOP_EDGE = "#FFD93D", "#1A1A1A"
    BOX_EDGE = "#9CA3AF"
    CALLOUT_CLR = "#6B7280"
    HALO_CLR = "#FFE89C"

    label_offsets = {
        "Університет": (-6.5, 3.0, "right", "center"),
        "Держпром": (6.0, 1.8, "left", "center"),
        "Майдан Конституції": (-4.0, -2.5, "right", "center"),
        "Спортивна": (-4.5, -3.5, "right", "center"),
        "Заводська": (5.5, -1.5, "left", "center"),
    }

    fig, ax = plt.subplots(figsize=(16, 13), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_axis_off()
    ax.set_aspect("equal")

    all_xs = [pos[n][0] for n in G.nodes()] + [
        pos[st][0] + dx for st, (dx, _, _, _) in label_offsets.items()
    ]
    all_ys = [pos[n][1] for n in G.nodes()] + [
        pos[st][1] + dy for st, (_, dy, _, _) in label_offsets.items()
    ]
    pad = 1.5
    ax.set_xlim(min(all_xs) - pad, max(all_xs) + pad)
    ax.set_ylim(min(all_ys) - pad, max(all_ys) + pad)

    # HALO — лежить під усім
    nx.draw_networkx_edges(
        G, pos, edgelist=cycle_edges, ax=ax,
        edge_color=HALO_CLR, width=26, alpha=0.7,
    )

    # Ребра в натуральних кольорах
    for clr, edges in by_color.items():
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax,
            edge_color=clr, width=9, alpha=0.95,
        )

    # Звичайні вузли
    regular = [n for n in G.nodes() if n not in top_set]
    nx.draw_networkx_nodes(
        G, pos, nodelist=regular, ax=ax,
        node_size=320, node_color=BG,
        edgecolors=[color_of[n] for n in regular],
        linewidths=2.5,
    )

    # Топ-5 — золоті
    top_nodes = [st for st, _ in top_stations]
    nx.draw_networkx_nodes(
        G, pos, nodelist=top_nodes, ax=ax,
        node_size=2100, node_color=TOP_FILL,
        edgecolors=TOP_EDGE, linewidths=4,
    )

    # Callout-лейбли
    for st, bc_val in top_stations:
        x, y = pos[st]
        dx, dy, ha, va = label_offsets[st]
        ax.annotate(
            f"{st}\n{bc_val*100:.0f}%",
            xy=(x, y),
            xytext=(x + dx, y + dy),
            fontsize=14, fontweight="bold",
            color=TEXT, ha=ha, va=va, linespacing=1.3,
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white",
                edgecolor=BOX_EDGE, alpha=0.92, linewidth=1,
            ),
            arrowprops=dict(
                arrowstyle="-", color=CALLOUT_CLR,
                linewidth=1.5, shrinkA=2, shrinkB=24,
            ),
            zorder=10,
        )

    fig.text(
        0.5, 0.075,
        "Підсвічена зона — 9 з 30 ребер з альтернативним маршрутом",
        ha="center", va="center", fontsize=13,
        color=SUB, fontstyle="italic",
    )
    fig.text(
        0.5, 0.040,
        "Золоті кружки — Топ-5 станцій за впливом",
        ha="center", va="center", fontsize=13,
        color=SUB, fontstyle="italic",
    )

    fig.text(
        0.5, 0.975, "Карта крихкості мережі Харківського метро",
        ha="center", va="top", fontsize=26,
        fontweight="bold", color=TEXT,
    )
    fig.text(
        0.5, 0.928,
        "21 з 30 ребер — мости без альтернативи   ·   "
        "5 станцій тримають майже половину мережі",
        ha="center", va="top", fontsize=16, color=SUB,
    )

    plt.subplots_adjust(top=0.89, bottom=0.11, left=0.02, right=0.98)

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.show()
    plt.close(fig)


# ===========================================================================
# Круговий (полігональний) layout — друга візуалізація
# ===========================================================================

class CircularStyle:
    """Налаштування стилю для кругового layout-у."""

    FIGSIZE = (22, 12)
    STATION_SIZE = 280
    STATION_EDGE_WIDTH = 3
    LINE_WIDTH = 7
    LINE_ALPHA = 0.85

    TRANSFER_EDGE_COLOR = "#555555"
    TRANSFER_EDGE_WIDTH = 3.0
    TRANSFER_EDGE_STYLE = (0, (5, 3))
    TRANSFER_EDGE_ALPHA = 0.85

    LABEL_FONTSIZE = 10
    TRANSFER_FONTSIZE = 11


# Лейбли для пересадкових станцій у круговому layout-і.
# (angle_deg, distance, ha, va)
CIRCULAR_LABEL_OFFSETS: Dict[str, Tuple[float, float, str, str]] = {
    "Майдан Конституції": (0, 0.55, "left", "center"),
    "Університет": (270, 0.55, "right", "top"),
    "Держпром": (90, 0.95, "center", "bottom"),
    "Спортивна": (315, 0.55, "left", "top"),
    "Архітектора Бекетова": (120, 0.85, "right", "bottom"),
    "Захисників України": (80, 0.85, "center", "bottom"),
    "Метробудівників": (40, 0.55, "left", "bottom"),
    "Левада": (320, 0.85, "left", "top"),
}


def draw_circular_schematic(
    G, lines: List[Line], transfers: List[Transfer], pos,
    savepath: Optional[str] = None,
) -> None:
    """Альтернативна схема: пересадки як окремі станції на 9-вершинному полігоні."""
    import math

    color_of, _, transfer_stations = build_indexes(lines, transfers)

    fig, ax = plt.subplots(figsize=CircularStyle.FIGSIZE, facecolor="white")
    ax.set_facecolor("white")
    ax.set_axis_off()
    ax.set_aspect("equal")

    # Сегменти ліній
    for line in lines:
        edges = list(zip(line.stations, line.stations[1:]))
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax,
            edge_color=line.color,
            width=CircularStyle.LINE_WIDTH,
            alpha=CircularStyle.LINE_ALPHA,
        )

    # Пунктирні ребра-пересадки
    transfer_edges = [t.stations for t in transfers]
    nx.draw_networkx_edges(
        G, pos, edgelist=transfer_edges, ax=ax,
        edge_color=CircularStyle.TRANSFER_EDGE_COLOR,
        width=CircularStyle.TRANSFER_EDGE_WIDTH,
        style=CircularStyle.TRANSFER_EDGE_STYLE,
        alpha=CircularStyle.TRANSFER_EDGE_ALPHA,
    )

    # Усі станції — однакові кружечки
    for st, (x, y) in pos.items():
        ax.scatter(
            x, y,
            s=CircularStyle.STATION_SIZE, c="white",
            edgecolors=color_of[st],
            linewidths=CircularStyle.STATION_EDGE_WIDTH,
            zorder=5,
        )

    # Підписи лише для CIRCULAR_LABEL_OFFSETS
    for st in pos:
        if st not in CIRCULAR_LABEL_OFFSETS:
            continue
        x, y = pos[st]
        angle_deg, dist, ha, va = CIRCULAR_LABEL_OFFSETS[st]
        a = math.radians(angle_deg)
        lx, ly = x + dist * math.cos(a), y + dist * math.sin(a)
        is_transfer = st in transfer_stations
        style = (
            {
                "fontsize": CircularStyle.TRANSFER_FONTSIZE,
                "fontweight": "bold",
                "color": color_of[st],
                "bbox": dict(
                    boxstyle="round,pad=0.3", facecolor="white",
                    edgecolor=color_of[st], alpha=0.95, linewidth=1.5,
                ),
            }
            if is_transfer
            else {
                "fontsize": CircularStyle.LABEL_FONTSIZE,
                "fontweight": "normal",
                "color": "black",
                "bbox": dict(
                    boxstyle="round,pad=0.15", facecolor="white",
                    edgecolor="none", alpha=0.85,
                ),
            }
        )
        ax.text(lx, ly, st, ha=ha, va=va, zorder=6, **style)

    # Легенда
    handles = [
        Line2D(
            [0], [0], color=line.color,
            linewidth=CircularStyle.LINE_WIDTH,
            label=f"{line.code} — {line.name}",
        )
        for line in lines
    ]
    handles.append(
        Line2D(
            [0], [0], color=CircularStyle.TRANSFER_EDGE_COLOR,
            linewidth=CircularStyle.TRANSFER_EDGE_WIDTH,
            linestyle="--", label="Пересадка",
        )
    )
    ax.legend(
        handles=handles, loc="upper right",
        fontsize=Style.LEGEND_FONTSIZE,
        framealpha=0.95, edgecolor="gray",
    )

    # Заголовок
    ax.set_title(
        f"Схема Харківського метрополітену\n"
        f"{G.number_of_nodes()} станцій • "
        f"{G.number_of_edges()} з'єднань • 3 лінії • {len(transfers)} пересадки",
        fontsize=Style.TITLE_FONTSIZE, fontweight="bold", pad=20,
    )

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.show()
    plt.close(fig)
