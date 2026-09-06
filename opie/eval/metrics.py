"""Attribute-level evaluation metrics against Open Food Facts ground truth."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from opie.config import NUMERIC_ABS_TOL, NUMERIC_REL_TOL


def numeric_match(pred: Optional[float], gt: Optional[float],
                  rel_tol: float = NUMERIC_REL_TOL, abs_tol: float = NUMERIC_ABS_TOL) -> bool:
    """A predicted number counts as correct within a relative tolerance (with an abs floor)."""
    if pred is None or gt is None:
        return False
    return abs(pred - gt) <= max(abs_tol, rel_tol * abs(gt))


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "PRF") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def nutrition_prf(pred: dict[str, Optional[float]], gt: dict[str, Optional[float]]) -> PRF:
    """Per-field numeric PRF for one product (micro-accumulate across products)."""
    prf = PRF()
    for field_name, gt_val in gt.items():
        pred_val = pred.get(field_name)
        if gt_val is not None and pred_val is not None:
            if numeric_match(pred_val, gt_val):
                prf.tp += 1
            else:
                prf.fp += 1   # predicted a value, but wrong
                prf.fn += 1   # and missed the real value
        elif gt_val is not None and pred_val is None:
            prf.fn += 1       # missed a value that exists
        elif gt_val is None and pred_val is not None:
            prf.fp += 1       # hallucinated a value
    return prf


def ingredient_prf(pred_names: list[str], gt_names: list[str]) -> PRF:
    """Set-based ingredient PRF on normalized names for one product."""
    pred_set = {_norm(n) for n in pred_names if n}
    gt_set = {_norm(n) for n in gt_names if n}
    tp = len(pred_set & gt_set)
    return PRF(tp=tp, fp=len(pred_set - gt_set), fn=len(gt_set - pred_set))


def cohen_kappa(pred: list[str], gt: list[str], labels: list[str]) -> float:
    """Cohen's kappa for categorical grade agreement."""
    n = len(pred)
    if n == 0:
        return 0.0
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    conf = [[0] * k for _ in range(k)]
    for p, g in zip(pred, gt):
        if p in idx and g in idx:
            conf[idx[g]][idx[p]] += 1
    total = sum(sum(row) for row in conf)
    if total == 0:
        return 0.0
    po = sum(conf[i][i] for i in range(k)) / total
    row_tot = [sum(conf[i]) for i in range(k)]
    col_tot = [sum(conf[r][c] for r in range(k)) for c in range(k)]
    pe = sum((row_tot[i] / total) * (col_tot[i] / total) for i in range(k))
    return (po - pe) / (1 - pe) if (1 - pe) else 1.0


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().replace("_", " ").split())
