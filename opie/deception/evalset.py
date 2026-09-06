"""Reproducible labeled eval set for deception detection.

Construction (documented + deterministic):
  * A fixed roster of front-of-pack claims is attached to every product. This yields both
    honest cases (claims a product's facts satisfy) and ADVERSARIAL cases (a claim attached
    to a product whose facts violate its definition — e.g. "sugar free" on a biscuit).
  * Ground truth for each (product, claim) = the PUBLIC regulatory definition adjudicated on
    the product's TRUE facts (GT nutrition + GT ingredients). No hand labeling; the label
    follows from the cited standard.
  * At eval time the detector runs on EXTRACTED facts (with realistic extraction noise), so
    the metric measures the end-to-end deception catch-rate, not just rule correctness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opie.deception.adjudicator import Adjudicator
from opie.deception.claims import normalize_claim
from opie.feedback.simulate import SimulatedExtractor, panel_from_values
from opie.schemas import ClaimStatus, Ingredient, NutritionPanel
from opie.scoring.nutriscore import nutriscore
from opie.taxonomy.taxonomy import normalize_ingredients

# Claims attached to every product. Mix of nutrient-content, ingredient, and unverifiable.
CLAIM_ROSTER = (
    "sugar free", "no added sugar", "low sugar",
    "low fat", "fat free", "low salt",
    "high in protein", "source of fibre", "high fibre",
    "gluten free", "vegan", "100% natural", "low calorie",
    "organic", "fortified",
)


@dataclass
class LabeledCase:
    product_code: str
    category: str
    claim_text: str
    claim_id: str
    gt_status: ClaimStatus


def facts_from_product(product: Any, mode: str) -> tuple[NutritionPanel, list[Ingredient], str | None]:
    """Return (nutrition, ingredients, base_grade) from either true GT ('oracle') or the
    simulated extractor ('extracted')."""
    if mode == "oracle":
        panel = panel_from_values(product.gt_nutrition, confidence=1.0)
        names = list(product.gt_ingredients)
    elif mode == "extracted":
        obs = SimulatedExtractor().extract(product)
        values = {o.name: float(o.extracted_value) for o in obs
                  if o.kind == "nutrition" and o.extracted_value is not None}
        panel = panel_from_values(values, confidence=1.0)
        names = [str(o.extracted_value) for o in obs
                 if o.kind == "ingredient" and o.extracted_value is not None]
    else:
        raise ValueError(f"mode must be 'oracle' or 'extracted', got {mode!r}")
    ingredients = normalize_ingredients(
        [Ingredient(name=n.strip().lower(), rank=i + 1) for i, n in enumerate(names)])
    _, base_grade = nutriscore(panel)
    return panel, ingredients, base_grade


def build_labeled_set(products: list[Any]) -> list[LabeledCase]:
    """Ground-truth label = definition adjudicated on TRUE facts."""
    adj = Adjudicator()
    cases: list[LabeledCase] = []
    for product in products:
        panel, ingredients, base_grade = facts_from_product(product, "oracle")
        for claim_text in CLAIM_ROSTER:
            claim_id = normalize_claim(claim_text)
            if claim_id is None:
                continue
            verdict = adj.adjudicate(claim_text, panel, ingredients, base_grade)
            cases.append(LabeledCase(
                product_code=product.code, category=product.category,
                claim_text=claim_text, claim_id=claim_id, gt_status=verdict.status))
    return cases
