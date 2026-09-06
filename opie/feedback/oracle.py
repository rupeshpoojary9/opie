"""Reviewer oracle: the Open Food Facts record stands in for a human reviewer.

Labels every observation's correctness against ground truth (for metrics + calibration),
and turns queued fields into correction records (field, extracted, correct). For nutrition
the correct value is the GT number; for ingredients it aligns a predicted token to the GT
token it most likely corrupts (or marks it spurious). Works for both the simulated corpus
(GT carried on the observation) and real OFF products (alignment by fuzzy match).
"""
from __future__ import annotations

import difflib
from typing import Any, Optional

from opie.eval.metrics import numeric_match
from opie.feedback.types import CorrectionCase, FieldObservation


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("_", " ").split())


class ReviewOracle:
    def __init__(self, match_cutoff: float = 0.82) -> None:
        self.match_cutoff = match_cutoff

    def label(self, obs: FieldObservation, product: Any) -> None:
        """Fill obs.is_correct (and obs.gt_value if not already set)."""
        if obs.kind == "nutrition":
            gt = product.gt_nutrition.get(obs.name)
            obs.gt_value = gt
            obs.is_correct = numeric_match(_as_float(obs.extracted_value), gt) if gt is not None else False
            return
        # ingredient
        gt_set = {_norm(t) for t in product.gt_ingredients}
        pred = _norm(obs.extracted_value)
        obs.is_correct = pred in gt_set
        if obs.gt_value is None and not obs.is_correct:
            obs.gt_value = self._align(pred, product.gt_ingredients)

    def correct_value(self, obs: FieldObservation, product: Any) -> Any:
        """The oracle's correct value for a queued field (None => drop, for spurious tokens)."""
        if obs.kind == "nutrition":
            return product.gt_nutrition.get(obs.name)
        pred = _norm(obs.extracted_value)
        if pred in {_norm(t) for t in product.gt_ingredients}:
            return pred                      # already correct — confirmation
        if obs.gt_value:
            return _norm(obs.gt_value)       # simulated truth
        return self._align(pred, product.gt_ingredients)

    def make_correction(self, obs: FieldObservation, product: Any, round_index: int) -> CorrectionCase:
        return CorrectionCase(
            product_id=obs.product_id, kind=obs.kind, name=obs.name,
            extracted_value=obs.extracted_value,
            correct_value=self.correct_value(obs, product),
            features=obs.features, round_index=round_index,
        )

    def _align(self, pred: str, gt_tokens: list[str]) -> Optional[str]:
        """Nearest GT token by string similarity, or None if nothing is close (=> spurious)."""
        best, best_score = None, 0.0
        for t in gt_tokens:
            score = difflib.SequenceMatcher(None, pred, _norm(t)).ratio()
            if score > best_score:
                best, best_score = _norm(t), score
        return best if best_score >= self.match_cutoff else None


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
