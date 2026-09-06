"""Runtime configuration. Everything tunable lives here or in the rules/taxonomy data files."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OFF_SUBSET_DIR = DATA_DIR / "off_subset"
IMAGES_DIR = DATA_DIR / "images"
RESULTS_DIR = REPO_ROOT / "results"

# The nutrition attributes OPIE extracts and evaluates, and their OFF ground-truth key.
NUTRITION_FIELDS: tuple[str, ...] = (
    "energy_kcal_100g",
    "fat_100g",
    "saturated_fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
    "sodium_100g",
)

# Tolerance for counting a numeric extraction "correct" vs OFF ground truth.
# Relative tolerance with an absolute floor so tiny values (e.g. salt 0.1g) aren't
# unfairly penalized by rounding.
NUMERIC_REL_TOL = 0.05      # 5%
NUMERIC_ABS_TOL = 0.5       # absolute floor in the field's unit

# Confidence below this routes a field to human review (exception routing).
REVIEW_CONFIDENCE_THRESHOLD = 0.60


def _resolve_backend() -> str:
    explicit = os.environ.get("OPIE_BACKEND")
    if explicit:
        return explicit.lower()
    # Auto: use Claude vision when a credential is discoverable, else offline.
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    # An `ant auth login` profile also counts, but detecting it reliably without a
    # network call is not worth it here; users set OPIE_BACKEND=anthropic in that case.
    return "offline"


@dataclass
class Settings:
    backend: str = field(default_factory=_resolve_backend)
    model: str = field(default_factory=lambda: os.environ.get("OPIE_MODEL", "claude-opus-4-8"))
    review_confidence_threshold: float = REVIEW_CONFIDENCE_THRESHOLD

    @property
    def uses_llm(self) -> bool:
        return self.backend == "anthropic"


SETTINGS = Settings()
