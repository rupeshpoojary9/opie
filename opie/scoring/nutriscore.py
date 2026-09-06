"""Base composite health score: the 2017 Nutri-Score algorithm for general foods.

Public, independent benchmark. We compute it from our extracted attributes so the eval
harness can measure agreement between OPIE's grade and OFF's published `nutriscore_grade`.
Beverages and some categories use different tables; this implements the general-food table.
"""
from __future__ import annotations

from typing import Optional

from opie.schemas import NutritionPanel

_KCAL_TO_KJ = 4.184

# (threshold, points) — first threshold the value is <= wins.
_ENERGY_KJ = [(335, 0), (670, 1), (1005, 2), (1340, 3), (1675, 4),
              (2010, 5), (2345, 6), (2680, 7), (3015, 8), (3350, 9)]
_SUGARS = [(4.5, 0), (9, 1), (13.5, 2), (18, 3), (22.5, 4),
           (27, 5), (31, 6), (36, 7), (40, 8), (45, 9)]
_SAT_FAT = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8), (10, 9)]
_SODIUM_MG = [(90, 0), (180, 1), (270, 2), (360, 3), (450, 4),
              (540, 5), (630, 6), (720, 7), (810, 8), (900, 9)]
_FIBER = [(0.9, 0), (1.9, 1), (2.8, 2), (3.7, 3), (4.7, 4)]
_PROTEIN = [(1.6, 0), (3.2, 1), (4.8, 2), (6.4, 3), (8.0, 4)]


def _points(value: Optional[float], table: list[tuple[float, int]], max_points: int) -> int:
    if value is None:
        return 0
    for threshold, pts in table:
        if value <= threshold:
            return pts
    return max_points


def _sodium_mg(panel: NutritionPanel) -> Optional[float]:
    if panel.sodium_100g.value is not None:
        return panel.sodium_100g.value * 1000.0
    if panel.salt_100g.value is not None:
        return (panel.salt_100g.value / 2.5) * 1000.0
    return None


def nutriscore(panel: NutritionPanel, fruit_veg_pct: float = 0.0) -> tuple[Optional[int], Optional[str]]:
    """Return (raw points, grade A-E). Points None if no nutrition data at all."""
    energy_kcal = panel.energy_kcal_100g.value
    energy_kj = energy_kcal * _KCAL_TO_KJ if energy_kcal is not None else None

    have_any = any(
        getattr(panel, f).value is not None
        for f in ("energy_kcal_100g", "sugars_100g", "saturated_fat_100g", "salt_100g", "sodium_100g")
    )
    if not have_any:
        return None, None

    neg = (
        _points(energy_kj, _ENERGY_KJ, 10)
        + _points(panel.sugars_100g.value, _SUGARS, 10)
        + _points(panel.saturated_fat_100g.value, _SAT_FAT, 10)
        + _points(_sodium_mg(panel), _SODIUM_MG, 10)
    )

    fiber_pts = _points(panel.fiber_100g.value, _FIBER, 5)
    protein_pts = _points(panel.proteins_100g.value, _PROTEIN, 5)
    fruit_pts = _fruit_points(fruit_veg_pct)

    # Protein points don't count when negatives are high unless fruit points maxed.
    if neg >= 11 and fruit_pts < 5:
        pos = fiber_pts + fruit_pts
    else:
        pos = fiber_pts + fruit_pts + protein_pts

    total = neg - pos
    return total, _grade(total)


def _fruit_points(pct: float) -> int:
    if pct > 80:
        return 5
    if pct > 60:
        return 2
    if pct > 40:
        return 1
    return 0


def _grade(total: int) -> str:
    if total <= -1:
        return "A"
    if total <= 2:
        return "B"
    if total <= 10:
        return "C"
    if total <= 18:
        return "D"
    return "E"
