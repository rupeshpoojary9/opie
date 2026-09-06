"""Correction memory — learns to invert extraction errors from accumulated corrections.

Numeric fields: k-NN retrieval over past correction cases (embedded features), then apply
the median correct/extracted ratio when neighbors agree with enough support. This learns
scale/unit errors as transforms that generalize to unseen products sharing the error mode.

Ingredient tokens: a learned normalization map (corrupted -> correct) and a spurious-token
set, both built from corrections (exact-key retrieval).

Also exposes few-shot `exemplars()` for retrieval-augmented prompting of the real
Claude-vision agents. Nothing here memorizes product identity — only transferable
transforms — so measuring on disjoint rounds is leakage-free.
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any, Optional

from opie.feedback.types import CorrectionCase
from opie.feedback.vectorize import cosine, embed, topk


class CorrectionMemory:
    def __init__(self, k: int = 12, min_support: int = 3,
                 sim_threshold: float = 0.45, ratio_tol: float = 0.25) -> None:
        self.k = k
        self.min_support = min_support
        self.sim_threshold = sim_threshold
        self.ratio_tol = ratio_tol
        # numeric cases: parallel lists
        self._num_vecs: list[list[float]] = []
        self._num_ratio: list[float] = []
        self._num_meta: list[CorrectionCase] = []
        # ingredient token learning
        self._token_map: dict[str, Counter] = defaultdict(Counter)
        self._spurious: Counter = Counter()
        self._confirmed: Counter = Counter()   # tokens confirmed correct
        # exemplar index (by field/category)
        self._exemplars: dict[tuple[str, str], list[CorrectionCase]] = defaultdict(list)

    def __len__(self) -> int:
        return len(self._num_meta) + sum(len(c) for c in self._token_map.values())

    # --- learning ---------------------------------------------------------

    def add(self, case: CorrectionCase) -> None:
        key = (case.name if case.kind == "nutrition" else "ingredient", case.features.get("category", "?"))
        self._exemplars[key].append(case)
        if case.kind == "nutrition":
            self._add_numeric(case)
        else:
            self._add_token(case)

    def add_many(self, cases: list[CorrectionCase]) -> None:
        for c in cases:
            self.add(c)

    def _add_numeric(self, case: CorrectionCase) -> None:
        ext, cor = _f(case.extracted_value), _f(case.correct_value)
        if ext is None or cor is None or abs(ext) < 1e-9:
            return
        self._num_vecs.append(embed(case.features))
        self._num_ratio.append(cor / ext)
        self._num_meta.append(case)

    def _add_token(self, case: CorrectionCase) -> None:
        pred = _norm(case.extracted_value)
        if not pred:
            return
        if case.correct_value is None:
            self._spurious[pred] += 1
        elif _norm(case.correct_value) == pred:
            self._confirmed[pred] += 1
        else:
            self._token_map[pred][_norm(case.correct_value)] += 1

    # --- applying corrections --------------------------------------------

    def correct_numeric(self, extracted: float, features: dict) -> tuple[Optional[float], int]:
        """Return (corrected_value, effective_support). None if not confident."""
        if not self._num_vecs or extracted is None:
            return None, 0
        q = embed(features)
        neigh = topk(q, enumerate(self._num_vecs), self.k)
        ratios = [self._num_ratio[i] for i, sim in neigh if sim >= self.sim_threshold]
        if len(ratios) < self.min_support:
            return None, len(ratios)
        med = statistics.median(ratios)
        support = sum(1 for r in ratios if abs(r - med) <= self.ratio_tol * max(abs(med), 1e-9))
        if support < self.min_support:
            return None, support
        return round(float(extracted) * med, 3), support

    def correct_token(self, pred: str) -> tuple[str, Optional[str]]:
        """Return (action, value): ('keep', tok) | ('map', correct) | ('drop', None)."""
        p = _norm(pred)
        if self._spurious[p] >= 2 and self._spurious[p] > self._confirmed[p]:
            return "drop", None
        if p in self._token_map and sum(self._token_map[p].values()) >= 2:
            return "map", self._token_map[p].most_common(1)[0][0]
        return "keep", p

    # --- retrieval-augmented prompting (real backends) --------------------

    def exemplars(self, field_name: str, category: str, k: int = 3) -> list[str]:
        cases = self._exemplars.get((field_name, category), [])
        out = []
        for c in cases[-k:]:
            out.append(f"saw '{c.extracted_value}' -> correct '{c.correct_value}'")
        return out

    def stats(self) -> dict:
        return {
            "numeric_cases": len(self._num_meta),
            "token_mappings": len(self._token_map),
            "spurious_tokens": len([t for t, n in self._spurious.items() if n >= 2]),
        }


def _norm(s: Any) -> str:
    return " ".join(str(s).strip().lower().replace("_", " ").split())


def _f(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
