"""Claim normalization: free-text front-of-pack claims -> typed claim ids.

Deterministic, ordered regex matching (specific patterns before general ones). Returns the
canonical claim id, or None when the text is not a recognized nutrition/content claim.
"""
from __future__ import annotations

import re

# Canonical claim ids
SUGAR_FREE = "sugar_free"
NO_ADDED_SUGAR = "no_added_sugar"
LOW_SUGAR = "low_sugar"
FAT_FREE = "fat_free"
LOW_FAT = "low_fat"
LOW_SATURATED_FAT = "low_saturated_fat"
SALT_FREE = "salt_free"
LOW_SALT = "low_salt"
HIGH_PROTEIN = "high_protein"
SOURCE_OF_PROTEIN = "source_of_protein"
HIGH_FIBRE = "high_fibre"
SOURCE_OF_FIBRE = "source_of_fibre"
LOW_ENERGY = "low_energy"
ENERGY_FREE = "energy_free"
GLUTEN_FREE = "gluten_free"
VEGAN = "vegan"
NATURAL = "natural"
ORGANIC = "organic"
FORTIFIED = "fortified"
SOURCE_OF_VITAMINS = "source_of_vitamins"

CANONICAL_CLAIM_IDS = (
    SUGAR_FREE, NO_ADDED_SUGAR, LOW_SUGAR, FAT_FREE, LOW_FAT, LOW_SATURATED_FAT,
    SALT_FREE, LOW_SALT, HIGH_PROTEIN, SOURCE_OF_PROTEIN, HIGH_FIBRE, SOURCE_OF_FIBRE,
    LOW_ENERGY, ENERGY_FREE, GLUTEN_FREE, VEGAN, NATURAL, ORGANIC, FORTIFIED,
    SOURCE_OF_VITAMINS,
)

# Ordered (specific -> general). First match wins.
_PATTERNS: list[tuple[str, str]] = [
    (r"\b(no|zero|0%?)\s+added\s+sugar", NO_ADDED_SUGAR),
    (r"\bsugar[\s-]*free\b|\bfree\s+from\s+sugar", SUGAR_FREE),
    (r"\b(low|reduced|light)\s+(in\s+)?sugar|\blow[\s-]sugar", LOW_SUGAR),
    (r"\bfat[\s-]*free\b", FAT_FREE),
    (r"\blow\s+(in\s+)?saturat", LOW_SATURATED_FAT),
    (r"\b(low|reduced|light)\s+(in\s+)?fat|\blow[\s-]fat|\b\d+%\s+less\s+fat", LOW_FAT),
    (r"\b(salt|sodium)[\s-]*free\b", SALT_FREE),
    (r"\blow\s+(in\s+)?(salt|sodium)|\blow[\s-](salt|sodium)", LOW_SALT),
    (r"\bhigh\s+(in\s+)?protein|\bprotein[\s-]*rich|\bpacked\s+with\s+protein", HIGH_PROTEIN),
    (r"\bsource\s+of\s+protein|\bwith\s+protein", SOURCE_OF_PROTEIN),
    (r"\bhigh\s+(in\s+)?fib(re|er)|\bfib(re|er)[\s-]*rich", HIGH_FIBRE),
    (r"\bsource\s+of\s+fib(re|er)|\bwith\s+fib(re|er)", SOURCE_OF_FIBRE),
    (r"\b(energy|calorie|calories)[\s-]*free\b", ENERGY_FREE),
    (r"\blow\s+(in\s+)?(energy|calorie|calories)|\blow[\s-](cal|calorie)", LOW_ENERGY),
    (r"\bgluten[\s-]*free\b", GLUTEN_FREE),
    (r"\bvegan\b|\bplant[\s-]*based\b", VEGAN),
    (r"\b100%?\s*natural\b|\ball[\s-]natural\b|\bnatural\b", NATURAL),
    (r"\borganic\b|\bbio\b", ORGANIC),
    (r"\bfortified\b|\benriched\b", FORTIFIED),
    (r"\bsource\s+of\s+(vitamin|mineral)|\bhigh\s+in\s+(vitamin|mineral)|\bwith\s+(added\s+)?vitamin", SOURCE_OF_VITAMINS),
]


def normalize_claim(text: str) -> str | None:
    """Map a free-text claim to a canonical claim id, or None if unrecognized."""
    if not text:
        return None
    t = text.strip().lower()
    # already a canonical id?
    if t in CANONICAL_CLAIM_IDS:
        return t
    t = t.replace("_", " ")
    for pattern, claim_id in _PATTERNS:
        if re.search(pattern, t):
            return claim_id
    return None
