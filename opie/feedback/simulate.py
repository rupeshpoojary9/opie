"""Deterministic simulated extractor with systematic, learnable errors.

Models the error modes a real OCR/vision extractor makes — decimal-comma scale errors,
salt/sodium unit swaps, digit-drops, OCR token corruptions, synonym mismatches, spurious
insertions, and occasional drops — as a *seeded pure function* of (product, field). The
errors are systematic (a given mode shares a correction ratio / token map across products),
so a retrieval-based correction layer can learn to invert them and generalize to unseen
products. Confidence is emitted noisily correlated with correctness, so recalibration has
signal to fit. This is the reproducible, offline stand-in for the Claude-vision extractor;
the same correction layer wraps the real backend when a key + images are present.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any

from opie.config import NUTRITION_FIELDS
from opie.feedback.types import FieldObservation
from opie.feedback.vectorize import magnitude_bucket
from opie.schemas import NumericAttribute, NutritionPanel

# --- systematic numeric error modes: (category, field) -> (ratio, trigger_prob) ---------
# `ratio` = correct_value / extracted_value, i.e. what a correction must multiply by.
NUMERIC_MODES: dict[tuple[str, str], tuple[float, float]] = {
    ("soda", "sugars_100g"): (10.0, 0.75),        # decimal comma: 10.5g read as 1.05
    ("chocolate", "sugars_100g"): (10.0, 0.55),
    ("yogurt", "sugars_100g"): (10.0, 0.45),
    ("biscuits", "energy_kcal_100g"): (10.0, 0.55),  # dropped digit: 482 -> 48
    ("cereal", "energy_kcal_100g"): (10.0, 0.45),
    ("chips", "salt_100g"): (2.5, 0.70),          # sodium printed where salt expected
    ("sauce", "salt_100g"): (2.5, 0.65),
    ("bread", "salt_100g"): (2.5, 0.55),
    ("chips", "saturated_fat_100g"): (10.0, 0.35),
}

# --- systematic ingredient token corruptions: correct_token -> (corrupted, trigger_prob) --
TOKEN_CORRUPTIONS: dict[str, tuple[str, float]] = {
    "sugar": ("sugarr", 0.6),
    "maize": ("corn", 0.7),
    "flavouring": ("flavour", 0.5),
    "sunflower oil": ("sunflwr oil", 0.55),
    "wheat flour": ("wheat flr", 0.45),
    "soy lecithin": ("soya lecithin", 0.5),
    "modified starch": ("modified starcn", 0.5),
    "palm oil": ("paim oil", 0.4),
}
_SPURIOUS_TOKEN = "e000 unknown"
_SPURIOUS_PROB = 0.10
_DROP_PROB = 0.07
_IDIOSYNCRATIC_PROB = 0.05   # non-generalizable noise -> caps achievable F1 below 1.0


def _seed(*parts: Any) -> random.Random:
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def panel_from_values(values: dict[str, float | None], confidence: float = 1.0) -> NutritionPanel:
    panel = NutritionPanel()
    for name, val in values.items():
        if val is not None and name in NUTRITION_FIELDS:
            unit = "kcal" if name == "energy_kcal_100g" else "g"
            setattr(panel, name, NumericAttribute(value=float(val), unit=unit, confidence=confidence))
    return panel


class SimulatedExtractor:
    backend_name = "simulated"

    def extract(self, product: Any) -> list[FieldObservation]:
        obs: list[FieldObservation] = []
        obs.extend(self._nutrition(product))
        obs.extend(self._ingredients(product))
        return obs

    # --- nutrition --------------------------------------------------------
    def _nutrition(self, product: Any) -> list[FieldObservation]:
        out: list[FieldObservation] = []
        for field_name, gt in product.gt_nutrition.items():
            if gt is None:
                continue
            rng = _seed(product.code, field_name, "num")
            value = float(gt)
            corrupted = False

            mode = NUMERIC_MODES.get((product.category, field_name))
            if mode and rng.random() < mode[1]:
                ratio, _ = mode
                jitter = 1.0 + rng.uniform(-0.02, 0.02)   # small realistic noise
                value = round((gt / ratio) * jitter, 3)
                corrupted = True
            elif rng.random() < _IDIOSYNCRATIC_PROB:
                value = round(gt * rng.choice([0.5, 2.0, 1.4]), 3)  # non-generalizable
                corrupted = True

            conf = rng.uniform(0.30, 0.66) if corrupted else rng.uniform(0.72, 0.98)
            unit = "kcal" if field_name == "energy_kcal_100g" else "g"
            out.append(FieldObservation(
                product_id=product.code, kind="nutrition", name=field_name,
                extracted_value=value, raw_confidence=round(conf, 4),
                features={"name": field_name, "category": product.category,
                          "mag": magnitude_bucket(value), "unit": unit},
                evidence=f"{field_name} {value}{unit}",
                gt_value=float(gt),
            ))
        return out

    # --- ingredients ------------------------------------------------------
    def _ingredients(self, product: Any) -> list[FieldObservation]:
        out: list[FieldObservation] = []
        for rank, token in enumerate(product.gt_ingredients, start=1):
            rng = _seed(product.code, token, rank, "ing")
            if rng.random() < _DROP_PROB:
                continue  # unrecoverable recall loss
            emitted = token
            corrupted = False
            corr = TOKEN_CORRUPTIONS.get(token)
            if corr and rng.random() < corr[1]:
                emitted = corr[0]
                corrupted = True
            conf = rng.uniform(0.30, 0.65) if corrupted else rng.uniform(0.70, 0.97)
            out.append(FieldObservation(
                product_id=product.code, kind="ingredient", name=f"ing{rank}",
                extracted_value=emitted, raw_confidence=round(conf, 4),
                features={"name": "ingredient", "category": product.category, "token": emitted},
                evidence=emitted, gt_value=token,
            ))
            # occasional spurious insertion (a token with no GT match)
            if rng.random() < _SPURIOUS_PROB:
                out.append(FieldObservation(
                    product_id=product.code, kind="ingredient", name=f"ing{rank}s",
                    extracted_value=_SPURIOUS_TOKEN, raw_confidence=round(rng.uniform(0.3, 0.6), 4),
                    features={"name": "ingredient", "category": product.category, "token": _SPURIOUS_TOKEN},
                    evidence=_SPURIOUS_TOKEN, gt_value=None,
                ))
        return out
