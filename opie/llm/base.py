"""Extractor protocol shared by both backends, plus token-cost accounting.

The pipeline is genuinely multi-agent: nutrition, ingredients, and claims are three
independent extraction calls (three LangGraph nodes). Both backends implement the same
seams so the graph is backend-agnostic.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from opie.schemas import Claim, Ingredient, NutritionPanel

# Anthropic list price per 1M tokens (Opus 4.8), used for the "cost per product" metric.
# Overridden per-model in AnthropicVisionExtractor.
_PRICE_PER_MTOK = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


class CostMeter:
    """Accumulates token usage across a product's agent calls and prices it."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)

    def cost_usd(self) -> float:
        in_price, out_price = _PRICE_PER_MTOK.get(self.model, (0.0, 0.0))
        return (self.input_tokens * in_price + self.output_tokens * out_price) / 1_000_000

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0


@runtime_checkable
class Extractor(Protocol):
    backend_name: str

    def ocr(self, image_bytes: bytes) -> str:
        """Transcribe the label to text (OCR + layout pass)."""
        ...

    def extract_nutrition(self, image_bytes: bytes, ocr_text: str) -> NutritionPanel:
        ...

    def extract_ingredients(self, image_bytes: bytes, ocr_text: str) -> list[Ingredient]:
        ...

    def extract_claims(self, image_bytes: bytes, ocr_text: str) -> list[Claim]:
        ...

    def cost_usd(self) -> float:
        """Accumulated cost for the most recent product (0.0 for offline)."""
        ...

    def reset_cost(self) -> None:
        ...
