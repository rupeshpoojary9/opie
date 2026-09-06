"""Round-over-round feedback report: the rising-F1 / shrinking-queue table + verdict."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from opie.feedback.loop import RoundResult


class FeedbackReport:
    def __init__(self, rounds: list[RoundResult], backend: str = "simulated") -> None:
        self.rounds = rounds
        self.backend = backend

    # --- derived headline deltas -----------------------------------------

    @property
    def first(self) -> RoundResult:
        return self.rounds[0]

    @property
    def last(self) -> RoundResult:
        return self.rounds[-1]

    def f1_lift(self) -> float:
        return round(self.last.f1 - self.first.f1, 4)

    def f1_vs_baseline(self) -> float:
        return round(self.last.f1 - self.last.base_f1, 4)

    def queue_drop(self) -> float:
        if self.first.queue_size == 0:
            return 0.0
        return round((self.first.queue_size - self.last.queue_size) / self.first.queue_size, 4)

    def ece_improvement(self) -> float:
        return round(self.first.ece - self.last.ece, 4)

    def passed(self) -> bool:
        """Acceptance: learned F1 rises across rounds AND beats the frozen baseline AND queue shrinks."""
        return (self.last.f1 > self.first.f1
                and self.last.f1 > self.last.base_f1
                and self.last.queue_size < self.first.queue_size)

    # --- output -----------------------------------------------------------

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "backend": self.backend,
            "rounds": [asdict(r) for r in self.rounds],
            "summary": {
                "f1_lift_across_rounds": self.f1_lift(),
                "f1_vs_frozen_baseline": self.f1_vs_baseline(),
                "queue_drop_fraction": self.queue_drop(),
                "ece_improvement": self.ece_improvement(),
                "passed": self.passed(),
            },
        }, indent=2))

    def to_markdown(self) -> str:
        lines = [
            f"# OPIE feedback loop — round-over-round ({self.backend} extractor)",
            "",
            "Each round is measured **cold** (state learned only from earlier rounds; products "
            "are disjoint across rounds, so there is no leakage). The frozen baseline runs the "
            "same rounds with the feedback loop disabled.",
            "",
            "| Round | Products | Learned F1 | Baseline F1 | Nutrition F1 | Ingredient F1 | "
            "Review queue | Queue/product | ECE | Baseline ECE | Corrections learned |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in self.rounds:
            lines.append(
                f"| {r.round_index + 1} | {r.n_products} | **{r.f1:.3f}** | {r.base_f1:.3f} | "
                f"{r.nutrition_f1:.3f} | {r.ingredient_f1:.3f} | {r.queue_size} | "
                f"{r.queue_per_product:.2f} | {r.ece:.3f} | {r.base_ece:.3f} | {r.memory_size} |")
        lines += [
            "",
            "## Verdict",
            "",
            f"- Attribute F1: **{self.first.f1:.3f} → {self.last.f1:.3f}** "
            f"(+{self.f1_lift():.3f} across rounds; +{self.f1_vs_baseline():.3f} vs frozen baseline)",
            f"- Review queue: **{self.first.queue_size} → {self.last.queue_size}** "
            f"({self.queue_drop() * 100:.0f}% smaller)",
            f"- Calibration (ECE): **{self.first.ece:.3f} → {self.last.ece:.3f}** "
            f"(−{self.ece_improvement():.3f})",
            "",
            f"**Flywheel {'CONFIRMED' if self.passed() else 'NOT confirmed'}:** "
            f"F1 rises across rounds and beats the no-feedback baseline while the review queue shrinks."
            if self.passed() else
            f"**Flywheel NOT confirmed** on this run — inspect the per-round table.",
        ]
        return "\n".join(lines)
