"""Confidence recalibration + calibration error (ECE).

Reliability-binning calibrator: maps a raw extractor confidence to the empirical
probability-correct observed for that confidence band, fit from accumulated review
outcomes. Per-field calibrators with a global fallback so sparse fields still calibrate.
The review-queue threshold is a fixed target on the *calibrated* scale, so as calibration
sharpens the raw-confidence cut that queues a field moves on its own — the "recalibrated
threshold" the flywheel is meant to learn.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

N_BINS = 10


class _Reliability:
    def __init__(self, n_bins: int = N_BINS) -> None:
        self.n_bins = n_bins
        self.count = [0] * n_bins
        self.correct = [0] * n_bins
        self.n = 0

    def _bin(self, p: float) -> int:
        return min(self.n_bins - 1, max(0, int(p * self.n_bins)))

    def fit(self, pairs: Iterable[tuple[float, bool]]) -> None:
        self.count = [0] * self.n_bins
        self.correct = [0] * self.n_bins
        self.n = 0
        for conf, ok in pairs:
            b = self._bin(conf)
            self.count[b] += 1
            self.correct[b] += 1 if ok else 0
            self.n += 1

    def prob(self, raw: float) -> float:
        b = self._bin(raw)
        # Laplace-smoothed empirical accuracy; fall back to raw when the bin is empty.
        if self.count[b] == 0:
            return raw
        return (self.correct[b] + 1) / (self.count[b] + 2)


class CalibratorSet:
    """Per-field reliability calibrators with a global fallback."""

    def __init__(self, target_prob: float = 0.85, min_field_count: int = 15) -> None:
        self.target_prob = target_prob
        self.min_field_count = min_field_count
        self._global = _Reliability()
        self._per_field: dict[str, _Reliability] = defaultdict(_Reliability)
        self._fitted = False

    def fit(self, observations) -> None:
        """observations: objects with .name, .raw_confidence, .is_correct."""
        by_field: dict[str, list[tuple[float, bool]]] = defaultdict(list)
        allpairs: list[tuple[float, bool]] = []
        for o in observations:
            if o.is_correct is None:
                continue
            pair = (float(o.raw_confidence), bool(o.is_correct))
            by_field[o.name].append(pair)
            allpairs.append(pair)
        self._global.fit(allpairs)
        self._per_field = defaultdict(_Reliability)
        for field_name, pairs in by_field.items():
            r = _Reliability()
            r.fit(pairs)
            self._per_field[field_name] = r
        self._fitted = bool(allpairs)

    def calibrate(self, field_name: str, raw: float) -> float:
        if not self._fitted:
            return raw  # identity until we have outcomes
        r = self._per_field.get(field_name)
        if r is not None and r.n >= self.min_field_count:
            return r.prob(raw)
        return self._global.prob(raw)

    def should_queue(self, field_name: str, calibrated: float) -> bool:
        return calibrated < self.target_prob

    def ece(self, observations) -> float:
        """Expected Calibration Error on the calibrated probabilities."""
        bins_conf = [0.0] * N_BINS
        bins_correct = [0] * N_BINS
        bins_count = [0] * N_BINS
        n = 0
        for o in observations:
            if o.is_correct is None:
                continue
            p = self.calibrate(o.name, float(o.raw_confidence))
            b = min(N_BINS - 1, max(0, int(p * N_BINS)))
            bins_conf[b] += p
            bins_correct[b] += 1 if o.is_correct else 0
            bins_count[b] += 1
            n += 1
        if n == 0:
            return 0.0
        ece = 0.0
        for b in range(N_BINS):
            if bins_count[b] == 0:
                continue
            avg_conf = bins_conf[b] / bins_count[b]
            acc = bins_correct[b] / bins_count[b]
            ece += (bins_count[b] / n) * abs(acc - avg_conf)
        return ece
