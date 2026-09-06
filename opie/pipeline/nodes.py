"""Pipeline node functions (backend-agnostic).

Each node takes the shared state dict and returns the keys it produced. The three
extraction agents (nutrition / ingredients / claims) are independent and fan out from
the OCR node; validation, taxonomy, and scoring run after they fan back in.
"""
from __future__ import annotations

from typing import Any, TypedDict

from opie.llm.base import Extractor
from opie.rules.engine import RulesEngine
from opie.schemas import (
    Claim,
    Ingredient,
    NutritionPanel,
    ProductIntelligence,
    Score,
)
from opie.scoring.nutriscore import nutriscore
from opie.scoring.personalization import personalize
from opie.taxonomy.taxonomy import Taxonomy


class PipelineState(TypedDict, total=False):
    product_id: str
    image_bytes: bytes
    profiles: list[str]
    ocr_text: str
    nutrition: NutritionPanel
    ingredients: list[Ingredient]
    claims: list[Claim]
    result: ProductIntelligence


class Nodes:
    """Holds the injected dependencies (extractor, rules, taxonomy) for the graph."""

    def __init__(self, extractor: Extractor, rules: RulesEngine, taxonomy: Taxonomy) -> None:
        self.extractor = extractor
        self.rules = rules
        self.taxonomy = taxonomy

    # [2] OCR + layout
    def ocr(self, state: PipelineState) -> dict[str, Any]:
        return {"ocr_text": self.extractor.ocr(state["image_bytes"])}

    # [3] Multi-agent extraction — three independent agents
    def nutrition_agent(self, state: PipelineState) -> dict[str, Any]:
        return {"nutrition": self.extractor.extract_nutrition(state["image_bytes"], state.get("ocr_text", ""))}

    def ingredient_agent(self, state: PipelineState) -> dict[str, Any]:
        return {"ingredients": self.extractor.extract_ingredients(state["image_bytes"], state.get("ocr_text", ""))}

    def claims_agent(self, state: PipelineState) -> dict[str, Any]:
        return {"claims": self.extractor.extract_claims(state["image_bytes"], state.get("ocr_text", ""))}

    # [5] Taxonomy mapping
    def taxonomy_node(self, state: PipelineState) -> dict[str, Any]:
        annotated = [self.taxonomy.annotate(i) for i in state.get("ingredients", [])]
        return {"ingredients": annotated}

    # [4] Validation + [6] Scoring + assembly
    def assemble(self, state: PipelineState) -> dict[str, Any]:
        panel: NutritionPanel = state.get("nutrition") or NutritionPanel()
        product = ProductIntelligence(
            product_id=state["product_id"],
            backend=self.extractor.backend_name,
            nutrition=panel,
            ingredients=state.get("ingredients", []),
            claims=state.get("claims", []),
        )
        product.validation = self.rules.validate(product)

        points, grade = nutriscore(panel)
        score = Score(base_grade=grade, base_points=points)
        for profile in state.get("profiles", ["general"]):
            score.personalized.append(personalize(panel, profile))
        product.score = score

        try:
            product.cost_usd = self.extractor.cost_usd()
        except Exception:
            product.cost_usd = None
        return {"result": product}
