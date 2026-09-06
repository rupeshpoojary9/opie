from __future__ import annotations

import pytest

from opie.schemas import Claim, Evidence, Ingredient, NumericAttribute, NutritionPanel


def attr(value, unit="g", confidence=1.0, text="evidence"):
    return NumericAttribute(value=value, unit=unit, confidence=confidence,
                            evidence=Evidence(text=text))


def panel(**kwargs) -> NutritionPanel:
    """Build a NutritionPanel from field=value kwargs, confidence 1.0."""
    p = NutritionPanel()
    for k, v in kwargs.items():
        setattr(p, k, attr(v, unit="kcal" if k == "energy_kcal_100g" else "g"))
    return p


class FakeExtractor:
    """Deterministic extractor for tests — no OCR, no key, no network."""
    backend_name = "fake"

    def __init__(self, nutrition: dict | None = None,
                 ingredients: list[str] | None = None,
                 claims: list[str] | None = None):
        self._nutrition = nutrition or {}
        self._ingredients = ingredients or []
        self._claims = claims or []

    def ocr(self, image_bytes):
        return "fake ocr text"

    def extract_nutrition(self, image_bytes, ocr_text):
        return panel(**self._nutrition)

    def extract_ingredients(self, image_bytes, ocr_text):
        return [Ingredient(name=n, rank=i + 1, confidence=0.9,
                           evidence=Evidence(text=n))
                for i, n in enumerate(self._ingredients)]

    def extract_claims(self, image_bytes, ocr_text):
        return [Claim(text=c, normalized=c.replace(" ", "_"), confidence=0.9,
                      evidence=Evidence(text=c)) for c in self._claims]

    def cost_usd(self):
        return 0.0

    def reset_cost(self):
        pass


def make_off_product(code, nutrition: dict, ingredients: list[str], grade: str):
    """Construct a raw OFF-shaped dict for OFFProduct.from_json."""
    off_keys = {
        "energy_kcal_100g": "energy-kcal_100g",
        "saturated_fat_100g": "saturated-fat_100g",
    }
    nutriments = {off_keys.get(k, k): v for k, v in nutrition.items()}
    return {
        "code": code,
        "product_name": f"Test {code}",
        "nutriments": nutriments,
        "ingredients": [{"text": n, "rank": i + 1} for i, n in enumerate(ingredients)],
        "ingredients_text": ", ".join(ingredients),
        "nutriscore_grade": grade,
        "image_front_url": f"http://example/{code}.jpg",
    }


@pytest.fixture
def fake_extractor():
    return FakeExtractor
