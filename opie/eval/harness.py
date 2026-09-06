"""Eval harness: runs OPIE over an OFF held-out set and reports real, measured numbers.

Metrics:
  * attribute extraction precision / recall / F1 (nutrition fields + ingredients) vs OFF
  * validation catch-rate: inject known inconsistencies into ground-truth nutrition and
    measure the share the rules engine flags (per error type + overall)
  * score agreement: OPIE base grade vs OFF `nutriscore_grade` (accuracy, within-one, kappa)
  * latency p50/p95 and cost per product
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from opie.config import NUTRITION_FIELDS
from opie.data.off import OFFProduct
from opie.eval.metrics import PRF, cohen_kappa, ingredient_prf, nutrition_prf
from opie.pipeline.graph import Pipeline
from opie.rules.engine import RulesEngine
from opie.schemas import NumericAttribute, NutritionPanel, ProductIntelligence

GRADE_LABELS = ["A", "B", "C", "D", "E"]

# name -> (field it targets, mutator(panel)->None). Each returns the rule id we expect to fire.
_INJECTIONS: dict[str, tuple[list[str], str, Callable[[NutritionPanel], None]]] = {
    "sugars_gt_carbs": (["carbohydrates_100g"], "sugars_le_carbs",
                        lambda p: _set(p, "sugars_100g", (p.carbohydrates_100g.value or 0) + 20)),
    "saturated_gt_fat": (["fat_100g"], "saturated_le_fat",
                         lambda p: _set(p, "saturated_fat_100g", (p.fat_100g.value or 0) + 10)),
    "impossible_percent": ([], "percent_bounds",
                           lambda p: _set(p, "fat_100g", 150.0)),
    "salt_sodium_mismatch": (["sodium_100g"], "salt_sodium_consistency",
                             lambda p: _set(p, "salt_100g", (p.sodium_100g.value or 0.1) * 10)),
    "energy_macro_mismatch": (["carbohydrates_100g", "proteins_100g", "fat_100g"], "energy_macro_balance",
                              lambda p: _set(p, "energy_kcal_100g",
                                             ((p.carbohydrates_100g.value or 0) * 4
                                              + (p.proteins_100g.value or 0) * 4
                                              + (p.fat_100g.value or 0) * 9) * 3 + 300)),
}


def _set(panel: NutritionPanel, field_name: str, value: float) -> None:
    setattr(panel, field_name, NumericAttribute(value=value, unit="g", confidence=1.0))


def _panel_from_gt(gt: dict[str, Optional[float]]) -> NutritionPanel:
    panel = NutritionPanel()
    for field_name, val in gt.items():
        if val is not None:
            setattr(panel, field_name, NumericAttribute(value=val, unit="g", confidence=1.0))
    return panel


@dataclass
class EvalReport:
    n_products: int = 0
    n_scored_grades: int = 0
    nutrition: dict = field(default_factory=dict)
    ingredients: dict = field(default_factory=dict)
    validation_catch_rate: dict = field(default_factory=dict)
    score_agreement: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)
    cost_usd: dict = field(default_factory=dict)
    backend: str = "offline"
    engine: str = "sequential"

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    def to_markdown(self) -> str:
        n = self.nutrition
        ing = self.ingredients
        sa = self.score_agreement
        lat = self.latency_ms
        lines = [
            f"# OPIE eval report ({self.backend} backend, {self.engine} engine)",
            "",
            f"- Products evaluated: **{self.n_products}**",
            "",
            "## Attribute extraction vs Open Food Facts",
            "",
            "| Attribute set | Precision | Recall | F1 |",
            "|---|---|---|---|",
            f"| Nutrition fields | {n.get('precision', 0):.3f} | {n.get('recall', 0):.3f} | {n.get('f1', 0):.3f} |",
            f"| Ingredients | {ing.get('precision', 0):.3f} | {ing.get('recall', 0):.3f} | {ing.get('f1', 0):.3f} |",
            "",
            "## Validation catch-rate (injected errors)",
            "",
            "| Error type | Applicable | Caught | Rate |",
            "|---|---|---|---|",
        ]
        for name, d in self.validation_catch_rate.get("by_type", {}).items():
            lines.append(f"| {name} | {d['applicable']} | {d['caught']} | {d['rate']:.3f} |")
        overall = self.validation_catch_rate.get("overall", {})
        lines += [
            f"| **overall** | {overall.get('applicable', 0)} | {overall.get('caught', 0)} | {overall.get('rate', 0):.3f} |",
            "",
            "## Score agreement vs Nutri-Score",
            "",
            f"- Exact grade accuracy: **{sa.get('accuracy', 0):.3f}** (n={self.n_scored_grades})",
            f"- Within-one-grade accuracy: **{sa.get('within_one', 0):.3f}**",
            f"- Cohen's kappa: **{sa.get('cohen_kappa', 0):.3f}**",
            "",
            "## Performance",
            "",
            f"- Latency p50: {lat.get('p50', 0):.1f} ms · p95: {lat.get('p95', 0):.1f} ms",
            f"- Cost per product (mean): ${self.cost_usd.get('mean', 0):.5f}",
        ]
        return "\n".join(lines)


def _image_bytes(product: OFFProduct) -> Optional[bytes]:
    if product.local_image and Path(product.local_image).exists():
        return Path(product.local_image).read_bytes()
    return None


def run_eval(
    products: Iterable[OFFProduct],
    pipeline: Optional[Pipeline] = None,
    image_loader: Callable[[OFFProduct], Optional[bytes]] = _image_bytes,
    rules: Optional[RulesEngine] = None,
    profiles: Optional[list[str]] = None,
) -> EvalReport:
    pipeline = pipeline or Pipeline()
    rules = rules or RulesEngine()
    profiles = profiles or ["general", "diabetic", "hypertension"]

    nut_prf, ing_prf = PRF(), PRF()
    pred_grades: list[str] = []
    gt_grades: list[str] = []
    latencies: list[float] = []
    costs: list[float] = []
    catch = {name: {"applicable": 0, "caught": 0} for name in _INJECTIONS}
    n_products = 0

    products = list(products)
    for product in products:
        gt_nut = product.gt_nutrition()

        # --- validation catch-rate (injection-based, uses GT, no image needed) ---
        for name, (needed, rule_id, mutate) in _INJECTIONS.items():
            base = _panel_from_gt(gt_nut)
            if any(getattr(base, f).value is None for f in needed):
                continue
            catch[name]["applicable"] += 1
            mutate(base)
            corrupt = ProductIntelligence(product_id=product.code, nutrition=base)
            result = rules.validate(corrupt)
            if any(fl.rule == rule_id for fl in result.flags):
                catch[name]["caught"] += 1

        # --- extraction metrics (needs an image) ---
        img = image_loader(product)
        if img is not None:
            pi = pipeline.run(product.code, img, profiles=profiles)
            n_products += 1
            pred_nut = {k: attr.value for k, attr in pi.nutrition.as_dict().items()}
            nut_prf.add(nutrition_prf(pred_nut, gt_nut))
            ing_prf.add(ingredient_prf([i.name for i in pi.ingredients], product.gt_ingredient_names()))
            if pi.latency_ms is not None:
                latencies.append(pi.latency_ms)
            if pi.cost_usd is not None:
                costs.append(pi.cost_usd)
            if pi.score.base_grade and product.nutriscore_grade:
                pred_grades.append(pi.score.base_grade.upper())
                gt_grades.append(product.nutriscore_grade.upper())

    report = EvalReport(
        n_products=n_products,
        n_scored_grades=len(pred_grades),
        backend=pipeline.extractor.backend_name,
        engine=pipeline.engine,
        nutrition=nut_prf.as_dict(),
        ingredients=ing_prf.as_dict(),
        validation_catch_rate=_catch_summary(catch),
        score_agreement=_grade_agreement(pred_grades, gt_grades),
        latency_ms=_pctiles(latencies),
        cost_usd={"mean": round(statistics.mean(costs), 6) if costs else 0.0,
                  "total": round(sum(costs), 6)},
    )
    return report


def _catch_summary(catch: dict) -> dict:
    by_type = {}
    tot_app = tot_caught = 0
    for name, d in catch.items():
        rate = d["caught"] / d["applicable"] if d["applicable"] else 0.0
        by_type[name] = {"applicable": d["applicable"], "caught": d["caught"], "rate": round(rate, 4)}
        tot_app += d["applicable"]
        tot_caught += d["caught"]
    return {
        "by_type": by_type,
        "overall": {"applicable": tot_app, "caught": tot_caught,
                    "rate": round(tot_caught / tot_app, 4) if tot_app else 0.0},
    }


def _grade_agreement(pred: list[str], gt: list[str]) -> dict:
    if not pred:
        return {"accuracy": 0.0, "within_one": 0.0, "cohen_kappa": 0.0}
    exact = sum(p == g for p, g in zip(pred, gt)) / len(pred)
    order = {lab: i for i, lab in enumerate(GRADE_LABELS)}
    within = sum(abs(order[p] - order[g]) <= 1 for p, g in zip(pred, gt) if p in order and g in order) / len(pred)
    return {
        "accuracy": round(exact, 4),
        "within_one": round(within, 4),
        "cohen_kappa": round(cohen_kappa(pred, gt, GRADE_LABELS), 4),
    }


def _pctiles(xs: list[float]) -> dict:
    if not xs:
        return {"p50": 0.0, "p95": 0.0}
    s = sorted(xs)
    return {
        "p50": round(s[int(0.50 * (len(s) - 1))], 2),
        "p95": round(s[int(0.95 * (len(s) - 1))], 2),
    }


# keep the field order used by callers building GT dicts
assert set(NUTRITION_FIELDS) == set(OFFProduct(code="x").gt_nutrition().keys())
