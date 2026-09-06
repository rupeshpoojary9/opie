"""Open Food Facts loader + ground-truth mapping.

OFF is the public data source AND the ground truth: real product images we run the
pipeline on, and the structured nutrition/ingredient/Nutri-Score fields we score the
extraction against. OFF nutriment keys use hyphens (`energy-kcal_100g`); OPIE's schema
uses underscores. This module owns that mapping so the rest of the code never sees OFF's
raw shape.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

# OPIE schema field  ->  OFF nutriments key
OFF_NUTRIMENT_KEYS: dict[str, str] = {
    "energy_kcal_100g": "energy-kcal_100g",
    "fat_100g": "fat_100g",
    "saturated_fat_100g": "saturated-fat_100g",
    "carbohydrates_100g": "carbohydrates_100g",
    "sugars_100g": "sugars_100g",
    "fiber_100g": "fiber_100g",
    "proteins_100g": "proteins_100g",
    "salt_100g": "salt_100g",
    "sodium_100g": "sodium_100g",
}


@dataclass
class OFFProduct:
    """A single Open Food Facts product, both input (image) and ground truth."""
    code: str
    product_name: str = ""
    image_url: Optional[str] = None
    local_image: Optional[Path] = None
    nutriments: dict[str, Any] = field(default_factory=dict)
    ingredients_text: str = ""
    ingredients: list[dict[str, Any]] = field(default_factory=list)
    nutriscore_grade: Optional[str] = None
    labels_tags: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    # --- ground truth accessors -------------------------------------------
    def gt_nutrition(self) -> dict[str, Optional[float]]:
        """Ground-truth per-100g values, keyed by OPIE schema field name."""
        out: dict[str, Optional[float]] = {}
        for schema_key, off_key in OFF_NUTRIMENT_KEYS.items():
            val = self.nutriments.get(off_key)
            out[schema_key] = _to_float(val)
        return out

    def gt_ingredient_names(self) -> list[str]:
        """Ordered, normalized ingredient tokens from OFF."""
        if self.ingredients:
            return [
                _norm(i.get("text") or i.get("id", "").split(":")[-1])
                for i in self.ingredients
                if (i.get("text") or i.get("id"))
            ]
        # Fallback: split the raw ingredients_text on commas.
        return [_norm(tok) for tok in _split_ingredients(self.ingredients_text) if tok.strip()]

    def has_ground_truth(self) -> bool:
        nut = self.gt_nutrition()
        return any(v is not None for v in nut.values()) and bool(self.gt_ingredient_names())

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "OFFProduct":
        p = obj.get("product", obj)
        return cls(
            code=str(p.get("code") or p.get("_id") or ""),
            product_name=p.get("product_name", "") or "",
            image_url=(
                p.get("image_nutrition_url")
                or p.get("image_front_url")
                or p.get("image_url")
            ),
            local_image=Path(p["local_image"]) if p.get("local_image") else None,
            nutriments=p.get("nutriments", {}) or {},
            ingredients_text=p.get("ingredients_text", "") or "",
            ingredients=p.get("ingredients", []) or [],
            nutriscore_grade=(p.get("nutriscore_grade") or None),
            labels_tags=p.get("labels_tags", []) or [],
            raw=p,
        )


def _to_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().replace("_", " ").split())


def _split_ingredients(text: str) -> list[str]:
    # Ingredient lists nest with parentheses/brackets; a full parse is out of scope.
    # Splitting on top-level commas is enough for OFF's flat `ingredients` fallback.
    parts, depth, cur = [], 0, []
    for ch in text or "":
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p for p in parts if p.strip()]


# --- persistence -----------------------------------------------------------

def write_subset(products: list[OFFProduct], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for p in products:
            rec = dict(p.raw)
            rec["code"] = p.code
            if p.local_image:
                rec["local_image"] = str(p.local_image)
            fh.write(json.dumps(rec) + "\n")


def read_subset(path: Path) -> Iterator[OFFProduct]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield OFFProduct.from_json(json.loads(line))
