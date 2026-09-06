"""Deterministic product corpus for the flywheel demo.

Default: a seeded synthetic corpus of plausible packaged foods (public-domain-style,
no company data), so `opie feedback-eval` runs with zero network and zero key and is
fully reproducible. If a downloaded real Open Food Facts subset exists, that is used
instead — the oracle and loop are identical either way.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from opie.config import OFF_SUBSET_DIR
from opie.data.off import OFFProduct, read_subset

# category -> (nutrition ranges per 100g, ingredient template pool)
_CATEGORIES: dict[str, dict] = {
    "biscuits": {
        "ranges": {"energy_kcal_100g": (430, 520), "fat_100g": (18, 28), "saturated_fat_100g": (8, 15),
                   "carbohydrates_100g": (58, 70), "sugars_100g": (25, 40), "fiber_100g": (1.5, 4),
                   "proteins_100g": (5, 8), "salt_100g": (0.4, 1.1)},
        "ingredients": ["wheat flour", "sugar", "palm oil", "glucose-fructose syrup", "raising agent",
                        "salt", "soy lecithin", "flavouring"],
    },
    "cereal": {
        "ranges": {"energy_kcal_100g": (350, 400), "fat_100g": (2, 9), "saturated_fat_100g": (0.5, 3),
                   "carbohydrates_100g": (70, 84), "sugars_100g": (15, 35), "fiber_100g": (5, 12),
                   "proteins_100g": (6, 11), "salt_100g": (0.2, 1.0)},
        "ingredients": ["maize", "sugar", "wheat", "barley malt extract", "salt", "vitamin"],
    },
    "soda": {
        "ranges": {"energy_kcal_100g": (30, 55), "fat_100g": (0, 0.2), "saturated_fat_100g": (0, 0.1),
                   "carbohydrates_100g": (7, 13), "sugars_100g": (7, 13), "fiber_100g": (0, 0.2),
                   "proteins_100g": (0, 0.3), "salt_100g": (0.0, 0.1)},
        "ingredients": ["water", "sugar", "carbon dioxide", "acid", "flavouring", "caffeine"],
    },
    "chips": {
        "ranges": {"energy_kcal_100g": (480, 560), "fat_100g": (28, 38), "saturated_fat_100g": (2.5, 6),
                   "carbohydrates_100g": (48, 58), "sugars_100g": (1, 4), "fiber_100g": (3, 6),
                   "proteins_100g": (5, 8), "salt_100g": (1.0, 1.8)},
        "ingredients": ["potato", "sunflower oil", "salt", "flavouring", "monosodium glutamate"],
    },
    "yogurt": {
        "ranges": {"energy_kcal_100g": (55, 110), "fat_100g": (0.1, 5), "saturated_fat_100g": (0.1, 3),
                   "carbohydrates_100g": (5, 16), "sugars_100g": (5, 16), "fiber_100g": (0, 1),
                   "proteins_100g": (3, 6), "salt_100g": (0.1, 0.3)},
        "ingredients": ["milk", "sugar", "fruit", "modified starch", "pectin", "flavouring"],
    },
    "bread": {
        "ranges": {"energy_kcal_100g": (230, 280), "fat_100g": (1.5, 5), "saturated_fat_100g": (0.3, 1.5),
                   "carbohydrates_100g": (44, 52), "sugars_100g": (2, 6), "fiber_100g": (3, 7),
                   "proteins_100g": (8, 11), "salt_100g": (0.8, 1.3)},
        "ingredients": ["wheat flour", "water", "yeast", "salt", "sunflower oil"],
    },
    "chocolate": {
        "ranges": {"energy_kcal_100g": (520, 580), "fat_100g": (30, 40), "saturated_fat_100g": (18, 24),
                   "carbohydrates_100g": (48, 60), "sugars_100g": (45, 58), "fiber_100g": (2, 7),
                   "proteins_100g": (5, 9), "salt_100g": (0.1, 0.4)},
        "ingredients": ["sugar", "cocoa", "cocoa butter", "milk", "soy lecithin", "flavouring"],
    },
    "sauce": {
        "ranges": {"energy_kcal_100g": (90, 140), "fat_100g": (0.1, 3), "saturated_fat_100g": (0.0, 1),
                   "carbohydrates_100g": (18, 28), "sugars_100g": (18, 27), "fiber_100g": (0.5, 2),
                   "proteins_100g": (1, 2), "salt_100g": (1.5, 3.0)},
        "ingredients": ["tomato", "sugar", "vinegar", "salt", "modified starch", "acid"],
    },
}


@dataclass
class CorpusProduct:
    """A product with ground truth, plus its category (used by the simulated extractor)."""
    code: str
    category: str
    gt_nutrition: dict[str, float]
    gt_ingredients: list[str]
    nutriscore_grade: Optional[str] = None
    off: Optional[OFFProduct] = None


def _round_dp(value: float, dp: int = 2) -> float:
    return round(value, dp)


def synthetic_corpus(n: int, seed: int = 7) -> list[CorpusProduct]:
    rng = random.Random(seed)
    cats = list(_CATEGORIES)
    products: list[CorpusProduct] = []
    for i in range(n):
        cat = cats[i % len(cats)]
        spec = _CATEGORIES[cat]
        nutrition = {}
        for field_name, (lo, hi) in spec["ranges"].items():
            nutrition[field_name] = _round_dp(rng.uniform(lo, hi))
        # sodium is derived from salt (real OFF relationship), so the validator can check it
        nutrition["sodium_100g"] = _round_dp(nutrition["salt_100g"] / 2.5, 3)
        # pick a deterministic ingredient subset (keep order = descending quantity)
        pool = spec["ingredients"]
        k = rng.randint(max(3, len(pool) - 3), len(pool))
        ingredients = pool[:k]
        products.append(CorpusProduct(
            code=f"SYN-{cat}-{i:04d}",
            category=cat,
            gt_nutrition=nutrition,
            gt_ingredients=list(ingredients),
        ))
    from opie.scoring.nutriscore import nutriscore
    from opie.feedback.simulate import panel_from_values
    for p in products:
        _, grade = nutriscore(panel_from_values(p.gt_nutrition))
        p.nutriscore_grade = grade
    return products


def _category_of(off: OFFProduct) -> str:
    tags = " ".join(off.labels_tags) + " " + (off.product_name or "")
    tags = tags.lower()
    for cat in _CATEGORIES:
        if cat[:-1] in tags or cat in tags:
            return cat
    return "biscuits"  # arbitrary default so the simulated error model still has a mode


def load_corpus(n: int, seed: int = 7, subset_path: Optional[Path] = None) -> list[CorpusProduct]:
    """Real OFF subset if present, else the deterministic synthetic corpus."""
    path = subset_path or (OFF_SUBSET_DIR / "subset.jsonl")
    if path.exists():
        out: list[CorpusProduct] = []
        for off in read_subset(path):
            if not off.has_ground_truth():
                continue
            out.append(CorpusProduct(
                code=off.code,
                category=_category_of(off),
                gt_nutrition={k: v for k, v in off.gt_nutrition().items() if v is not None},
                gt_ingredients=off.gt_ingredient_names(),
                nutriscore_grade=off.nutriscore_grade,
                off=off,
            ))
            if len(out) >= n:
                break
        if out:
            return out
    return synthetic_corpus(n, seed=seed)
