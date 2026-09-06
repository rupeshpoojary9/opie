"""Offline extractor: local Tesseract OCR + deterministic heuristic parsing.

This runs with no API key so the whole graph, rules, taxonomy, scoring, API, and eval
harness are testable end-to-end. It is a genuine (weak) extractor, not an oracle — it
parses OCR text with regexes and never reads OFF ground truth. Offline eval numbers
therefore reflect the heuristic parser, NOT the Claude-vision pipeline, and are not the
metrics that go on a resume. Swap in the Anthropic backend for real numbers.
"""
from __future__ import annotations

import io
import re
from typing import Optional

from opie.schemas import Claim, Evidence, Ingredient, NumericAttribute, NutritionPanel

_OFFLINE_CONFIDENCE = 0.5  # heuristic parses are marked uncertain on purpose

# Nutrition keyword -> schema field. Order matters: check "saturated" before "fat".
_NUTRITION_PATTERNS: list[tuple[str, str]] = [
    (r"energy[^0-9]*?([0-9]+(?:\.[0-9]+)?)\s*kcal", "energy_kcal_100g"),
    (r"satur\w*(?:\s*fat)?[^0-9]*?([0-9]+(?:\.[0-9]+)?)", "saturated_fat_100g"),
    (r"(?<!satur)(?<!satur )(?<!satu)fat[^0-9]*?([0-9]+(?:\.[0-9]+)?)", "fat_100g"),
    (r"sugars?[^0-9]*?([0-9]+(?:\.[0-9]+)?)", "sugars_100g"),
    (r"carbo\w*[^0-9]*?([0-9]+(?:\.[0-9]+)?)", "carbohydrates_100g"),
    (r"fib(?:re|er)[^0-9]*?([0-9]+(?:\.[0-9]+)?)", "fiber_100g"),
    (r"prote\w*[^0-9]*?([0-9]+(?:\.[0-9]+)?)", "proteins_100g"),
    (r"salt[^0-9]*?([0-9]+(?:\.[0-9]+)?)", "salt_100g"),
    (r"sodium[^0-9]*?([0-9]+(?:\.[0-9]+)?)", "sodium_100g"),
]

_UNIT = {
    "energy_kcal_100g": "kcal",
    "sodium_100g": "g",
    "salt_100g": "g",
}

_CLAIM_PATTERNS: list[tuple[str, str]] = [
    (r"high\s+(?:in\s+)?protein", "high_protein"),
    (r"(?:source\s+of|high\s+(?:in\s+)?)\s*fib(?:re|er)", "high_fiber"),
    (r"(?:no|zero|0%)\s+added\s+sugar", "no_added_sugar"),
    (r"low\s+fat", "low_fat"),
    (r"sugar[- ]free", "sugar_free"),
    (r"gluten[- ]free", "gluten_free"),
    (r"\borganic\b", "organic"),
    (r"\bvegan\b", "vegan"),
]


class OfflineExtractor:
    backend_name = "offline"

    def cost_usd(self) -> float:
        return 0.0

    def reset_cost(self) -> None:
        pass

    def ocr(self, image_bytes: bytes) -> str:
        try:
            import pytesseract
            from PIL import Image
        except Exception:  # pragma: no cover - env without tesseract/PIL
            return ""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return pytesseract.image_to_string(img)
        except Exception:  # pragma: no cover - unreadable image
            return ""

    def extract_nutrition(self, image_bytes: bytes, ocr_text: str) -> NutritionPanel:
        text = ocr_text.lower()
        panel = NutritionPanel()
        for pattern, field_name in _NUTRITION_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if not m:
                continue
            attr = getattr(panel, field_name)
            if attr.value is not None:
                continue
            value = _safe_float(m.group(1))
            line = _line_of(ocr_text, m.start())
            setattr(
                panel,
                field_name,
                NumericAttribute(
                    value=value,
                    unit=_UNIT.get(field_name, "g"),
                    confidence=_OFFLINE_CONFIDENCE if value is not None else 0.0,
                    evidence=Evidence(text=line) if line else None,
                ),
            )
        return panel

    def extract_ingredients(self, image_bytes: bytes, ocr_text: str) -> list[Ingredient]:
        m = re.search(r"ingredients?\s*[:\-]?\s*(.+)", ocr_text, re.IGNORECASE | re.DOTALL)
        if not m:
            return []
        segment = m.group(1)
        # Stop at the next section header if one appears.
        segment = re.split(r"(?i)\b(nutrition|allergen|contains|storage|net weight)\b", segment)[0]
        tokens = _split_top_level(segment)
        out: list[Ingredient] = []
        for i, tok in enumerate(tokens, start=1):
            name = _clean_token(tok)
            if not name or len(name) > 60:
                continue
            out.append(
                Ingredient(
                    name=name,
                    rank=len(out) + 1,
                    confidence=_OFFLINE_CONFIDENCE,
                    evidence=Evidence(text=tok.strip()),
                )
            )
            if len(out) >= 40:
                break
        return out

    def extract_claims(self, image_bytes: bytes, ocr_text: str) -> list[Claim]:
        out: list[Claim] = []
        for pattern, normalized in _CLAIM_PATTERNS:
            m = re.search(pattern, ocr_text, re.IGNORECASE)
            if m:
                out.append(
                    Claim(
                        text=m.group(0),
                        normalized=normalized,
                        confidence=_OFFLINE_CONFIDENCE,
                        evidence=Evidence(text=_line_of(ocr_text, m.start())),
                    )
                )
        return out


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _line_of(text: str, idx: int) -> str:
    start = text.rfind("\n", 0, idx) + 1
    end = text.find("\n", idx)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def _split_top_level(text: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in text:
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
    return parts


def _clean_token(tok: str) -> str:
    tok = re.sub(r"\([^)]*\)", "", tok)          # drop parenthetical sub-lists
    tok = re.sub(r"[0-9]+(\.[0-9]+)?\s*%", "", tok)  # drop percentages
    tok = re.sub(r"[^a-zA-Z \-']", " ", tok)
    return " ".join(tok.strip().lower().split())
