"""Ingredient taxonomy normalization.

Maps each extracted ingredient to a coarse taxonomy class, tags recognized additive
E-numbers, flags EU major allergens, and marks NOVA-style ultra-processing markers.
The tables are configurable JSON data files, not hard-coded logic, so the taxonomy can
be swapped or extended per client without touching code.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from opie.schemas import Ingredient

_DATA = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=1)
def _load() -> dict:
    with (_DATA / "allergens.json").open() as fh:
        allergens = json.load(fh)["allergens"]
    with (_DATA / "additives.json").open() as fh:
        additives = json.load(fh)["additives"]
    with (_DATA / "categories.json").open() as fh:
        cats = json.load(fh)
    return {
        "allergens": allergens,
        "additives": additives,
        "categories": cats["categories"],
        "ultra_processing": cats["ultra_processing_markers"],
    }


class Taxonomy:
    def __init__(self, tables: dict | None = None) -> None:
        self.t = tables or _load()

    def classify(self, name: str) -> str | None:
        n = name.lower()
        for cls, markers in self.t["categories"].items():
            if any(m in n for m in markers):
                return cls
        return None

    def additive(self, name: str) -> tuple[str | None, str | None]:
        n = name.lower()
        # Longest key first so 'sodium citrate' wins over 'citric acid' substrings.
        for key in sorted(self.t["additives"], key=len, reverse=True):
            if key in n:
                a = self.t["additives"][key]
                return a["enumber"], a["function"]
        return None, None

    def allergen(self, name: str) -> str | None:
        n = name.lower()
        for cls, markers in self.t["allergens"].items():
            if any(m in n for m in markers):
                return cls
        return None

    def is_ultra_processing(self, name: str) -> bool:
        n = name.lower()
        return any(m in n for m in self.t["ultra_processing"])

    def annotate(self, ing: Ingredient) -> Ingredient:
        enumber, _fn = self.additive(ing.name)
        ing.taxonomy = self.classify(ing.name)
        ing.enumber = enumber
        ing.is_allergen = self.allergen(ing.name) is not None
        ing.ultra_processing_marker = self.is_ultra_processing(ing.name)
        return ing


def normalize_ingredients(ingredients: list[Ingredient], taxonomy: Taxonomy | None = None) -> list[Ingredient]:
    tax = taxonomy or Taxonomy()
    return [tax.annotate(i) for i in ingredients]
