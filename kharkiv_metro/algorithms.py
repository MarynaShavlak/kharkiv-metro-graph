"""Алгоритми пошуку шляхів на графі: BFS, DFS, Дейкстра."""

import heapq
from collections import deque
from typing import Dict, List, Optional, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# BFS / DFS — для незважених графів
# ---------------------------------------------------------------------------

def bfs_path(G: nx.Graph, start: str, end: str) -> Optional[List[str]]:
    """Найкоротший шлях через BFS (пошук у ширину).

    Гарантує мінімальну кількість ребер. Складність O(V + E).
    """
    if start == end:
        return [start]
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        node, path = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor in visited:
                continue
            if neighbor == end:
                return path + [neighbor]
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    return None


def dfs_path(G: nx.Graph, start: str, end: str) -> Optional[List[str]]:
    """Шлях через DFS (пошук у глибину), ітеративна реалізація через стек.

    Не гарантує найкоротшого шляху — повертає перший знайдений. Складність O(V + E).
    """
    if start == end:
        return [start]
    visited = {start}
    stack = [(start, [start])]
    while stack:
        node, path = stack.pop()
        for neighbor in G.neighbors(node):
            if neighbor in visited:
                continue
            if neighbor == end:
                return path + [neighbor]
            visited.add(neighbor)
            stack.append((neighbor, path + [neighbor]))
    return None


# ---------------------------------------------------------------------------
# Дейкстра — для зважених графів
# ---------------------------------------------------------------------------

def dijkstra(
    G: nx.Graph, start: str
) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
    """Найкоротші відстані від start до всіх інших вершин.

    Класична реалізація з бінарною купою. Складність O((V + E) · log V).

    Returns:
        distances: вершина → найкоротша відстань від start.
        predecessors: вершина → попередник у найкоротшому шляху (для відновлення).
    """
    distances: Dict[str, float] = {node: float("inf") for node in G.nodes()}
    distances[start] = 0
    predecessors: Dict[str, Optional[str]] = {node: None for node in G.nodes()}
    heap: List[Tuple[float, str]] = [(0, start)]
    visited = set()

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v in G.neighbors(u):
            if v in visited:
                continue
            new_d = d + G[u][v]["weight"]
            if new_d < distances[v]:
                distances[v] = new_d
                predecessors[v] = u
                heapq.heappush(heap, (new_d, v))

    return distances, predecessors


def reconstruct_path(
    predecessors: Dict[str, Optional[str]], end: str
) -> List[str]:
    """Відновлює шлях від старту до end, рухаючись по мапі предків."""
    path: List[str] = []
    cur: Optional[str] = end
    while cur is not None:
        path.append(cur)
        cur = predecessors[cur]
    return list(reversed(path))


def shortest_path(
    G: nx.Graph, start: str, end: str
) -> Tuple[Optional[List[str]], Optional[float]]:
    """Найкоротший шлях за вагами та його загальний час."""
    distances, predecessors = dijkstra(G, start)
    if distances[end] == float("inf"):
        return None, None
    return reconstruct_path(predecessors, end), distances[end]


def all_pairs_shortest_times(G: nx.Graph) -> Dict[str, Dict[str, float]]:
    """Найкоротші часи між усіма парами вершин — матриця V×V."""
    return {start: dijkstra(G, start)[0] for start in G.nodes()}


def path_time(G: nx.Graph, path: List[str]) -> int:
    """Загальний час маршруту (сума ваг ребер)."""
    return sum(G[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))
