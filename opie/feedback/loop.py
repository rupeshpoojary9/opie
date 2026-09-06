"""FeedbackLoop: process the dataset in N sequential rounds and prove the loop lifts accuracy.

Each round is measured COLD using only state learned from earlier rounds (products are
disjoint across rounds -> no leakage), then that round's review-queue corrections are folded
into the correction memory and the calibrator is refit. A frozen no-feedback baseline runs
the same rounds with learning disabled, so the round-over-round delta isolates the loop's
effect: attribute F1 should rise and the review queue should shrink for the learned system
while the baseline stays flat.
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from opie.eval.metrics import PRF, ingredient_prf, nutrition_prf
from opie.feedback.calibration import CalibratorSet
from opie.feedback.memory import CorrectionMemory
from opie.feedback.oracle import ReviewOracle
from opie.feedback.simulate import SimulatedExtractor, panel_from_values
from opie.feedback.store import FeedbackStore
from opie.feedback.types import FieldObservation
from opie.rules.engine import RulesEngine

CORRECTED_CONFIDENCE = 0.95


@dataclass
class RoundResult:
    round_index: int
    n_products: int
    # learned system
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    nutrition_f1: float = 0.0
    ingredient_f1: float = 0.0
    queue_size: int = 0
    queue_per_product: float = 0.0
    queue_error_recall: float = 0.0
    queue_precision: float = 0.0
    ece: float = 0.0
    memory_size: int = 0
    # frozen baseline
    base_f1: float = 0.0
    base_queue_size: int = 0
    base_ece: float = 0.0


@dataclass
class LoopState:
    memory: CorrectionMemory = field(default_factory=CorrectionMemory)
    calibrator: CalibratorSet = field(default_factory=CalibratorSet)
    calib_data: list = field(default_factory=list)


class FeedbackLoop:
    def __init__(
        self,
        products: list[Any],
        rounds: int = 5,
        seed: int = 13,
        emit_fn: Optional[Callable[[Any], list[FieldObservation]]] = None,
        store: Optional[FeedbackStore] = None,
        rules: Optional[RulesEngine] = None,
        oracle: Optional[ReviewOracle] = None,
    ) -> None:
        self.rounds = rounds
        self.seed = seed
        self.emit_fn = emit_fn or SimulatedExtractor().extract
        self.store = store
        self.rules = rules or RulesEngine()
        self.oracle = oracle or ReviewOracle()
        self._round_products = self._split(products, rounds, seed)

    # --- round splitting (deterministic, disjoint) ------------------------

    @staticmethod
    def _split(products: list[Any], rounds: int, seed: int) -> list[list[Any]]:
        items = list(products)
        random.Random(seed).shuffle(items)
        buckets: list[list[Any]] = [[] for _ in range(rounds)]
        for i, p in enumerate(items):
            buckets[i % rounds].append(p)
        return buckets

    # --- public -----------------------------------------------------------

    def run(self) -> list[RoundResult]:
        learned = LoopState()
        frozen = LoopState()   # never updated
        results: list[RoundResult] = []

        for r, prods in enumerate(self._round_products):
            # measure learned COLD (state from rounds < r)
            learned_obs = self._infer(prods, learned, apply_learning=True)
            base_obs = self._infer(prods, frozen, apply_learning=False)

            res = self._score(r, prods, learned_obs, base_obs, learned)

            if self.store is not None:
                run_id = uuid.uuid4().hex[:12]
                self.store.start_run(run_id, r, "learned", len(prods))
                self.store.save_observations(run_id, learned_obs)

            # fold this round's corrections into memory + refit calibrator
            self._learn(r, prods, learned_obs, learned)
            res.memory_size = len(learned.memory)
            results.append(res)
        return results

    # --- inference over one round -----------------------------------------

    def _infer(self, products: list[Any], state: LoopState, apply_learning: bool) -> list[FieldObservation]:
        out: list[FieldObservation] = []
        for product in products:
            obs = self.emit_fn(product)
            self._apply(product, obs, state, apply_learning)
            for o in obs:
                self.oracle.label(o, product)   # ground-truth label on the POST value
            out.extend(obs)
        return out

    def _apply(self, product: Any, obs: list[FieldObservation], state: LoopState, learn: bool) -> None:
        by_pid: dict[str, list[FieldObservation]] = {}
        for o in obs:
            by_pid.setdefault(o.product_id, []).append(o)

        for pid, group in by_pid.items():
            flagged_raw = self._validator_flags(group)
            for o in group:
                cal_raw = state.calibrator.calibrate(o.name, o.raw_confidence)
                candidate = state.calibrator.should_queue(o.name, cal_raw) or (o.name in flagged_raw)
                post_conf = cal_raw
                if learn and candidate:
                    post_conf = self._correct(o, state) or cal_raw
                o.calibrated_confidence = post_conf

            # queue decision on the corrected panel
            flagged_after = self._validator_flags(group)
            for o in group:
                o.validator_flagged = o.name in flagged_after and o.kind == "nutrition"
                below = state.calibrator.should_queue(o.name, o.calibrated_confidence or 0.0)
                o.queued = bool(below or o.validator_flagged)

    def _correct(self, o: FieldObservation, state: LoopState) -> Optional[float]:
        """Apply a learned correction; return the post-correction confidence, or None if unchanged."""
        if o.kind == "nutrition":
            val, support = state.memory.correct_numeric(_f(o.extracted_value), o.features)
            if val is not None:
                o.extracted_value = val
                o.corrected = True
                return min(0.97, 0.80 + 0.03 * support)
            return None
        # ingredient
        action, value = state.memory.correct_token(str(o.extracted_value))
        if action == "map":
            o.extracted_value = value
            o.corrected = True
            return CORRECTED_CONFIDENCE
        if action == "drop":
            o.extracted_value = None   # excluded from the predicted set
            o.corrected = True
            return CORRECTED_CONFIDENCE
        return None

    def _validator_flags(self, group: list[FieldObservation]) -> set[str]:
        values = {o.name: _f(o.extracted_value) for o in group if o.kind == "nutrition"}
        panel = panel_from_values({k: v for k, v in values.items() if v is not None})
        from opie.schemas import ProductIntelligence
        result = self.rules.validate(ProductIntelligence(product_id="_", nutrition=panel))
        return {fl.field for fl in result.flags if fl.field}

    # --- learning (fold round corrections) --------------------------------

    def _learn(self, round_index: int, products: list[Any],
               obs: list[FieldObservation], state: LoopState) -> None:
        by_pid = {p.code: p for p in products}
        cases = []
        for o in obs:
            if o.queued:
                product = by_pid[o.product_id]
                cases.append(self.oracle.make_correction(o, product, round_index))
        state.memory.add_many(cases)
        if self.store is not None:
            self.store.save_corrections(cases)
        # refit calibrator on all outcomes seen so far
        state.calib_data.extend(obs)
        state.calibrator.fit(state.calib_data)

    # --- scoring ----------------------------------------------------------

    def _score(self, r: int, products: list[Any],
               learned_obs: list[FieldObservation], base_obs: list[FieldObservation],
               state: LoopState) -> RoundResult:
        l_nut, l_ing = self._prf(products, learned_obs)
        b_nut, b_ing = self._prf(products, base_obs)
        combined = PRF()
        combined.add(l_nut)
        combined.add(l_ing)
        base_combined = PRF()
        base_combined.add(b_nut)
        base_combined.add(b_ing)

        queued = [o for o in learned_obs if o.queued]
        errors = [o for o in learned_obs if o.is_correct is False]
        caught = [o for o in errors if o.queued]
        queue_errors = [o for o in queued if o.is_correct is False]

        return RoundResult(
            round_index=r,
            n_products=len(products),
            f1=round(combined.f1, 4),
            precision=round(combined.precision, 4),
            recall=round(combined.recall, 4),
            nutrition_f1=round(l_nut.f1, 4),
            ingredient_f1=round(l_ing.f1, 4),
            queue_size=len(queued),
            queue_per_product=round(len(queued) / max(1, len(products)), 3),
            queue_error_recall=round(len(caught) / len(errors), 4) if errors else 1.0,
            queue_precision=round(len(queue_errors) / len(queued), 4) if queued else 0.0,
            ece=round(state.calibrator.ece(learned_obs), 4),
            base_f1=round(base_combined.f1, 4),
            base_queue_size=len([o for o in base_obs if o.queued]),
            base_ece=round(CalibratorSet().ece(base_obs), 4),
        )

    def _prf(self, products: list[Any], obs: list[FieldObservation]) -> tuple[PRF, PRF]:
        by_pid: dict[str, list[FieldObservation]] = {}
        for o in obs:
            by_pid.setdefault(o.product_id, []).append(o)
        nut, ing = PRF(), PRF()
        for product in products:
            group = by_pid.get(product.code, [])
            pred_nut = {o.name: _f(o.extracted_value) for o in group if o.kind == "nutrition"}
            nut.add(nutrition_prf(pred_nut, product.gt_nutrition))
            pred_ing = [str(o.extracted_value) for o in group
                        if o.kind == "ingredient" and o.extracted_value is not None]
            ing.add(ingredient_prf(pred_ing, product.gt_ingredients))
        return nut, ing


def _f(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
