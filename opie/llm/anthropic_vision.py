"""Claude vision extractor: OCR + per-attribute extraction with evidence + confidence.

Each of the three extraction agents (nutrition, ingredients, claims) makes a focused
vision call with structured output, so every field comes back typed, with a confidence
and the text span it was read from. The zero-arg Anthropic() client resolves credentials
from ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / an `ant auth login` profile — add any one
and this backend produces real, measured numbers.
"""
from __future__ import annotations

import base64
from typing import Optional

from pydantic import BaseModel

from opie.llm.base import CostMeter
from opie.schemas import Claim, Ingredient, NutritionPanel

_OCR_PROMPT = (
    "You are an OCR + layout engine for a packaged-food label. Transcribe ALL legible "
    "text on this label verbatim, preserving line breaks and reading order. Include the "
    "nutrition table, the ingredient list, and any front-of-pack claims. Output only the "
    "transcribed text, no commentary."
)

_NUTRITION_PROMPT = (
    "Extract the per-100g nutrition panel from this food label. For each field, give the "
    "numeric value, its unit as printed, a confidence in [0,1], and an evidence.text span "
    "copied verbatim from the label. If a nutrient is not present, leave value null and "
    "confidence 0. Energy must be in kcal per 100g. Use the transcribed text as a guide "
    "but trust the image. Do not invent values.\n\nTranscribed text:\n{ocr}"
)

_INGREDIENTS_PROMPT = (
    "Extract the ordered ingredient list from this food label. Ingredients are listed in "
    "descending quantity; set rank = 1-based position. For each ingredient give the name "
    "(lowercase, as printed), rank, a confidence in [0,1], and an evidence.text span. Do "
    "not fill taxonomy, enumber, is_allergen, or ultra_processing_marker. Do not invent "
    "ingredients.\n\nTranscribed text:\n{ocr}"
)

_CLAIMS_PROMPT = (
    "Extract front-of-pack marketing/health claims from this food label (e.g. 'high in "
    "protein', 'no added sugar', 'gluten free', 'organic'). For each, give the text as "
    "printed, a normalized snake_case key, a confidence in [0,1], and an evidence.text "
    "span. Only include claims actually printed on the label.\n\nTranscribed text:\n{ocr}"
)


class _IngredientsResponse(BaseModel):
    ingredients: list[Ingredient]


class _ClaimsResponse(BaseModel):
    claims: list[Claim]


class AnthropicVisionExtractor:
    backend_name = "anthropic"

    def __init__(self, model: str = "claude-opus-4-8") -> None:
        self.model = model
        self._meter = CostMeter(model)
        self._client = None  # lazy — importing this module must not require a key

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def cost_usd(self) -> float:
        return self._meter.cost_usd()

    def reset_cost(self) -> None:
        self._meter.reset()

    # --- calls -------------------------------------------------------------

    def _image_block(self, image_bytes: bytes) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
            },
        }

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            self._meter.add(getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))

    def ocr(self, image_bytes: bytes) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [self._image_block(image_bytes), {"type": "text", "text": _OCR_PROMPT}],
            }],
        )
        self._record_usage(resp)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def _parse(self, image_bytes: bytes, prompt: str, ocr_text: str, output_model):
        resp = self.client.messages.parse(
            model=self.model,
            max_tokens=8000,
            messages=[{
                "role": "user",
                "content": [
                    self._image_block(image_bytes),
                    {"type": "text", "text": prompt.format(ocr=ocr_text[:6000])},
                ],
            }],
            output_format=output_model,
        )
        self._record_usage(resp)
        return resp.parsed_output

    def extract_nutrition(self, image_bytes: bytes, ocr_text: str) -> NutritionPanel:
        return self._parse(image_bytes, _NUTRITION_PROMPT, ocr_text, NutritionPanel)

    def extract_ingredients(self, image_bytes: bytes, ocr_text: str) -> list[Ingredient]:
        parsed: _IngredientsResponse = self._parse(image_bytes, _INGREDIENTS_PROMPT, ocr_text, _IngredientsResponse)
        return parsed.ingredients

    def extract_claims(self, image_bytes: bytes, ocr_text: str) -> list[Claim]:
        parsed: _ClaimsResponse = self._parse(image_bytes, _CLAIMS_PROMPT, ocr_text, _ClaimsResponse)
        return parsed.claims
