"""Validation agent: a configurable rules engine over the extracted attributes.

Runs threshold rules (regulatory / plausibility limits) and built-in cross-field
consistency checks (energy vs macros, salt vs sodium, sugars <= carbs, ...). Fields
whose confidence is below the review threshold, or that trip a consistency check, route
to human review (exception routing). This is what makes the extraction trustworthy: an
LLM number that violates arithmetic gets caught, not shipped.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from opie.config import SETTINGS
from opie.schemas import (
    NutritionPanel,
    ProductIntelligence,
    Severity,
    ValidationFlag,
    ValidationResult,
)

_RULES_PATH = Path(__file__).resolve().parent / "rules.yaml"
_OPS = {
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
}


class RulesEngine:
    def __init__(self, rules_path: Path | None = None, review_threshold: float | None = None) -> None:
        with (rules_path or _RULES_PATH).open() as fh:
            self.cfg = yaml.safe_load(fh)
        self.review_threshold = (
            review_threshold if review_threshold is not None else SETTINGS.review_confidence_threshold
        )

    # --- public -----------------------------------------------------------

    def validate(self, product: ProductIntelligence) -> ValidationResult:
        flags: list[ValidationFlag] = []
        panel = product.nutrition
        vals = {k: attr.value for k, attr in panel.as_dict().items()}

        flags.extend(self._threshold_flags(vals))
        flags.extend(self._builtin_flags(vals))

        review = self._needs_review(panel)
        # A high-severity flag also means the record shouldn't pass unattended.
        passed = not any(f.severity == Severity.high for f in flags)
        return ValidationResult(passed=passed, flags=flags, review_required=review or not passed)

    # --- threshold rules --------------------------------------------------

    def _threshold_flags(self, vals: dict[str, float | None]) -> list[ValidationFlag]:
        out: list[ValidationFlag] = []
        for rule in self.cfg.get("thresholds", []):
            v = vals.get(rule["field"])
            if v is None:
                continue
            if _OPS[rule["op"]](v, rule["value"]):
                out.append(ValidationFlag(
                    rule=rule["id"],
                    severity=Severity(rule["severity"]),
                    message=rule["message"],
                    field=rule["field"],
                ))
        return out

    # --- built-in consistency checks --------------------------------------

    def _builtin_flags(self, vals: dict[str, float | None]) -> list[ValidationFlag]:
        out: list[ValidationFlag] = []
        by_id = {b["id"]: b for b in self.cfg.get("builtins", []) if b.get("enabled", True)}

        if "percent_bounds" in by_id:
            b = by_id["percent_bounds"]
            for field_name in ("fat_100g", "saturated_fat_100g", "carbohydrates_100g",
                               "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g", "sodium_100g"):
                v = vals.get(field_name)
                if v is not None and v > 100.0:
                    out.append(ValidationFlag(
                        rule="percent_bounds", severity=Severity(b["severity"]),
                        message=f"{field_name} = {v} exceeds 100g per 100g.", field=field_name))

        if "sugars_le_carbs" in by_id:
            b = by_id["sugars_le_carbs"]
            s, c = vals.get("sugars_100g"), vals.get("carbohydrates_100g")
            if s is not None and c is not None and s > c + 0.5:
                out.append(ValidationFlag(
                    rule="sugars_le_carbs", severity=Severity(b["severity"]),
                    message=f"Sugars ({s}g) exceed carbohydrates ({c}g).", field="sugars_100g"))

        if "saturated_le_fat" in by_id:
            b = by_id["saturated_le_fat"]
            sat, fat = vals.get("saturated_fat_100g"), vals.get("fat_100g")
            if sat is not None and fat is not None and sat > fat + 0.5:
                out.append(ValidationFlag(
                    rule="saturated_le_fat", severity=Severity(b["severity"]),
                    message=f"Saturated fat ({sat}g) exceeds total fat ({fat}g).", field="saturated_fat_100g"))

        if "salt_sodium_consistency" in by_id:
            b = by_id["salt_sodium_consistency"]
            salt, sodium = vals.get("salt_100g"), vals.get("sodium_100g")
            if salt is not None and sodium is not None and sodium > 0:
                expected = sodium * 2.5
                if abs(salt - expected) > b["tolerance"] * max(expected, 0.1):
                    out.append(ValidationFlag(
                        rule="salt_sodium_consistency", severity=Severity(b["severity"]),
                        message=f"Salt ({salt}g) inconsistent with sodium ({sodium}g); expected ~{expected:.2f}g.",
                        field="salt_100g"))

        if "energy_macro_balance" in by_id:
            b = by_id["energy_macro_balance"]
            energy = vals.get("energy_kcal_100g")
            carb, prot, fat = vals.get("carbohydrates_100g"), vals.get("proteins_100g"), vals.get("fat_100g")
            if energy is not None and None not in (carb, prot, fat) and energy > 0:
                computed = 4 * carb + 4 * prot + 9 * fat
                if abs(computed - energy) > b["tolerance"] * energy:
                    out.append(ValidationFlag(
                        rule="energy_macro_balance", severity=Severity(b["severity"]),
                        message=f"Energy {energy}kcal inconsistent with macros (~{computed:.0f}kcal from 4/4/9).",
                        field="energy_kcal_100g"))
        return out

    def _needs_review(self, panel: NutritionPanel) -> bool:
        for attr in panel.as_dict().values():
            if attr.value is not None and attr.confidence < self.review_threshold:
                return True
        return False
