"""Розклади координат станцій для рендеру схеми.

Два варіанти:
1. ``compute_geographic_layout`` — географічно правдоподібна розкладка
   (як у першому ноутбуці): червона ↘, синя ↗, зелена через центральний трикутник.
2. ``compute_circular_layout`` — схематична розкладка з 9-вершинним полігоном
   у центрі (як у one_circle), де пересадки рендеряться окремими станціями.
"""

import math
from typing import Dict, List, Tuple

from .data import Line


# ===========================================================================
# Геометричні хелпери
# ===========================================================================

def place_along_line(
    pos: Dict[str, Tuple[float, float]],
    start: Tuple[float, float],
    direction: Tuple[float, float],
    stations: Tuple[str, ...],
    offsets: List[float],
) -> None:
    """Розставляє станції від start у напрямку direction за offset'ами."""
    sx, sy = start
    dx, dy = direction
    for st, off in zip(stations, offsets):
        pos[st] = (sx + off * dx, sy + off * dy)


def place_between(
    pos: Dict[str, Tuple[float, float]],
    start: Tuple[float, float],
    end: Tuple[float, float],
    stations: Tuple[str, ...],
) -> None:
    """Розставляє станції рівномірно на відрізку (start, end), не включно."""
    sx, sy = start
    ex, ey = end
    n = len(stations) + 1
    for i, st in enumerate(stations, start=1):
        t = i / n
        pos[st] = (sx + t * (ex - sx), sy + t * (ey - sy))


# ===========================================================================
# 1. Географічний layout
# ===========================================================================

class GeographicLayout:
    """Сталі для географічного layout-у."""

    STEP = 2.0
    RED_DIR = (1.0, -0.6)   # напрямок червоної ↘
    BLUE_DIR = (1.3, 0.6)   # напрямок синьої ↗
    TRIANGLE_HEIGHT = 2.5
    TRANSFER_RADIUS = 0.4


def compute_geographic_layout(
    lines: List[Line],
) -> Dict[str, Tuple[float, float]]:
    """Будує географічно-правдоподібний layout: червона ↘, синя ↗, зелена ↓."""
    step = GeographicLayout.STEP
    red, blue, green = lines

    # 1) Червона ↘ — центр у "Майдан Конституції" (offset = 0)
    konst_idx = red.stations.index("Майдан Конституції")
    konst_pos = (0.0, 0.0)
    red_dir = (
        GeographicLayout.RED_DIR[0] * step,
        GeographicLayout.RED_DIR[1] * step,
    )
    red_offsets = [i - konst_idx for i in range(len(red.stations))]

    pos: Dict[str, Tuple[float, float]] = {}
    place_along_line(pos, konst_pos, red_dir, red.stations, red_offsets)

    sport_pos = pos["Спортивна"]

    # 2) Вершина трикутника (Університет / Держпром)
    univ_pos = (
        (konst_pos[0] + sport_pos[0]) / 2,
        (konst_pos[1] + sport_pos[1]) / 2
        + step * GeographicLayout.TRIANGLE_HEIGHT,
    )

    # 3) Синя ↗ — від Університету вгору-вправо
    pos["Університет"] = univ_pos
    pos["Історичний музей"] = konst_pos
    blue_dir = (
        GeographicLayout.BLUE_DIR[0] * step,
        GeographicLayout.BLUE_DIR[1] * step,
    )
    blue_before = blue.stations[:6]  # без двох пересадкових
    blue_offsets = list(range(len(blue_before), 0, -1))  # [6,5,4,3,2,1]
    place_along_line(pos, univ_pos, blue_dir, blue_before, blue_offsets)

    # 4) Зелена: вертикальна частина зверху + діагональна знизу
    pos["Держпром"] = univ_pos
    pos["Метробудівників"] = sport_pos
    green_above = green.stations[:5]
    above_offsets = [(len(green_above) - i) * step for i in range(len(green_above))]
    for st, dy in zip(green_above, above_offsets):
        pos[st] = (univ_pos[0], univ_pos[1] + dy)

    green_below = green.stations[6:8]  # між Держпромом і Метробудівниками
    place_between(pos, univ_pos, sport_pos, green_below)

    return pos


# ===========================================================================
# 2. Круговий (полігональний) layout
# ===========================================================================

class CircularLayout:
    """Сталі для кругового layout-у з 9-вершинним центральним полігоном."""

    POLYGON_CENTER = (0.0, 0.0)
    POLYGON_RADIUS = 3.0
    STEP_EXT = 1.5  # крок між зовнішніми станціями

    # 9 станцій циклу — рівномірно по колу проти годинникової стрілки.
    # Спортивна на 0° (справа), щоб найдовший хвіст (M1→Індустріальна) пішов праворуч.
    CYCLE_STATIONS: List[Tuple[str, float]] = [
        ("Спортивна", 0.0),
        ("Метробудівників", 40.0),
        ("Захисників України", 80.0),
        ("Архітектора Бекетова", 120.0),
        ("Держпром", 160.0),
        ("Університет", 200.0),
        ("Історичний музей", 240.0),
        ("Майдан Конституції", 280.0),
        ("Левада", 320.0),
    ]


CYCLE_STATION_SET = {s for s, _ in CircularLayout.CYCLE_STATIONS}


def compute_circular_layout(
    lines: List[Line],
) -> Dict[str, Tuple[float, float]]:
    """Будує круговий layout: 9 пересадково-сусідніх станцій по колу,
    решта — радіально назовні."""
    red, blue, green = lines
    cx, cy = CircularLayout.POLYGON_CENTER
    R = CircularLayout.POLYGON_RADIUS
    step_ext = CircularLayout.STEP_EXT

    pos: Dict[str, Tuple[float, float]] = {}

    # Крок 1: розставити 9 станцій циклу по колу
    for st, angle_deg in CircularLayout.CYCLE_STATIONS:
        a = math.radians(angle_deg)
        pos[st] = (cx + R * math.cos(a), cy + R * math.sin(a))

    def outward(st: str) -> Tuple[float, float]:
        x, y = pos[st]
        dx, dy = x - cx, y - cy
        m = math.hypot(dx, dy)
        return (dx / m, dy / m)

    # Крок 2: продовжити зовнішні «хвости» РАДІАЛЬНО назовні
    def extend(anchor_st, line_stations, anchor_idx, indices):
        ax, ay = pos[anchor_st]
        odx, ody = outward(anchor_st)
        for i in indices:
            st = line_stations[i]
            k = abs(i - anchor_idx)
            pos[st] = (ax + k * step_ext * odx, ay + k * step_ext * ody)

    konst_i = red.stations.index("Майдан Конституції")
    sport_i = red.stations.index("Спортивна")
    univ_i = blue.stations.index("Університет")
    derzh_i = green.stations.index("Держпром")

    extend("Майдан Конституції", red.stations, konst_i, range(konst_i))
    extend(
        "Спортивна", red.stations, sport_i,
        range(sport_i + 1, len(red.stations)),
    )
    extend("Університет", blue.stations, univ_i, range(univ_i))
    extend("Держпром", green.stations, derzh_i, range(derzh_i))

    return pos
