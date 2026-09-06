"""Typed, evidence-backed intelligence record — the contract every OPIE layer speaks.

Every extracted field carries a value, a confidence in [0, 1], and an evidence
span (the text the model saw, plus an optional bounding box) so a downstream
consumer can trace any claim back to the label. This traceability is the point:
you cannot make a health claim off an LLM extraction you cannot audit.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Evidence & attributes -------------------------------------------------

class Evidence(BaseModel):
    """Where a value came from on the label."""
    text: str = Field(description="The exact text span the value was read from.")
    bbox: Optional[list[float]] = Field(
        default=None,
        description="Optional [x0, y0, x1, y1] normalized bounding box, if the OCR layer provides one.",
    )


class NumericAttribute(BaseModel):
    value: Optional[float] = Field(default=None, description="Numeric value, or null if absent/unreadable.")
    unit: Optional[str] = Field(default=None, description="Unit as printed, e.g. 'kcal', 'g', 'mg'.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: Optional[Evidence] = None


class Ingredient(BaseModel):
    name: str = Field(description="Ingredient token as printed (normalized to lowercase).")
    rank: int = Field(description="1-based position in the ingredient list (order = descending quantity).")
    taxonomy: Optional[str] = Field(default=None, description="Normalized taxonomy class, filled by the taxonomy layer.")
    enumber: Optional[str] = Field(default=None, description="E-number if this is a recognized additive, e.g. 'E322'.")
    is_allergen: bool = False
    ultra_processing_marker: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: Optional[Evidence] = None


class Claim(BaseModel):
    text: str = Field(description="Front-of-pack claim as printed, e.g. 'HIGH IN PROTEIN'.")
    normalized: Optional[str] = Field(default=None, description="Canonical claim key, e.g. 'high_protein'.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: Optional[Evidence] = None


class NutritionPanel(BaseModel):
    """Per-100g nutrition attributes. Names mirror Open Food Facts `*_100g` fields."""
    energy_kcal_100g: NumericAttribute = Field(default_factory=NumericAttribute)
    fat_100g: NumericAttribute = Field(default_factory=NumericAttribute)
    saturated_fat_100g: NumericAttribute = Field(default_factory=NumericAttribute)
    carbohydrates_100g: NumericAttribute = Field(default_factory=NumericAttribute)
    sugars_100g: NumericAttribute = Field(default_factory=NumericAttribute)
    fiber_100g: NumericAttribute = Field(default_factory=NumericAttribute)
    proteins_100g: NumericAttribute = Field(default_factory=NumericAttribute)
    salt_100g: NumericAttribute = Field(default_factory=NumericAttribute)
    sodium_100g: NumericAttribute = Field(default_factory=NumericAttribute)

    def as_dict(self) -> dict[str, NumericAttribute]:
        return {k: getattr(self, k) for k in type(self).model_fields}


# --- Validation ------------------------------------------------------------

class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ValidationFlag(BaseModel):
    rule: str = Field(description="Rule id that fired, e.g. 'sugars_gt_threshold'.")
    severity: Severity
    message: str
    field: Optional[str] = None


class ValidationResult(BaseModel):
    passed: bool = True
    flags: list[ValidationFlag] = Field(default_factory=list)
    review_required: bool = Field(
        default=False,
        description="True when low-confidence or conflicting fields need human review (exception routing).",
    )


# --- Scoring ---------------------------------------------------------------

class TrafficLight(str, Enum):
    green = "green"
    yellow = "yellow"
    red = "red"


class PersonalizedScore(BaseModel):
    profile: str = Field(description="Profile id the score was computed for, e.g. 'diabetic'.")
    flag: TrafficLight
    why: list[str] = Field(default_factory=list, description="Explainable reasons driving the flag.")


class Score(BaseModel):
    base_grade: Optional[str] = Field(default=None, description="Nutri-Score-style grade A-E.")
    base_points: Optional[int] = Field(default=None, description="Raw Nutri-Score points (lower is healthier).")
    personalized: list[PersonalizedScore] = Field(default_factory=list)


# --- Top-level record ------------------------------------------------------

class ProductIntelligence(BaseModel):
    """The full image -> trustworthy structured data -> personalized decision record."""
    product_id: str
    source: str = Field(default="openfoodfacts", description="Provenance of the input.")
    backend: str = Field(default="offline", description="Extraction backend used ('anthropic' or 'offline').")
    nutrition: NutritionPanel = Field(default_factory=NutritionPanel)
    ingredients: list[Ingredient] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    validation: ValidationResult = Field(default_factory=ValidationResult)
    score: Score = Field(default_factory=Score)
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
