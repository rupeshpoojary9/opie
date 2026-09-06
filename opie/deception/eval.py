"""Deception eval: run the detector over the labeled set and report catch-rate.

Headline metric is RECALL on MISLEADING (how many deceptive claims we catch). Also reports
the full 3x3 confusion matrix and per-class precision/recall/F1, plus worked examples.
Ground truth is the public definition on true facts; predictions run on extracted facts, so
the MISLEADING recall gap from 1.0 is exactly the extraction-induced miss rate (on true
facts the rule set is self-consistent, i.e. recall 1.0 by construction).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from opie.deception.adjudicator import Adjudicator
from opie.deception.evalset import build_labeled_set, facts_from_product
from opie.schemas import ClaimStatus

CLASSES = [ClaimStatus.supported, ClaimStatus.misleading, ClaimStatus.unverifiable]
_ORDER = [c.value for c in CLASSES]


@dataclass
class DeceptionReport:
    mode: str = "extracted"
    n_cases: int = 0
    confusion: dict = field(default_factory=dict)     # gt -> pred -> count
    per_class: dict = field(default_factory=dict)
    misleading_recall: float = 0.0
    misleading_precision: float = 0.0
    misleading_f1: float = 0.0
    macro_f1: float = 0.0
    worked_examples: list = field(default_factory=list)

    def passed(self, target_recall: float = 0.85) -> bool:
        return self.misleading_recall >= target_recall

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))

    def to_markdown(self) -> str:
        lines = [
            f"# OPIE deception detector — labeled eval ({self.mode} facts)",
            "",
            f"Cases: **{self.n_cases}** · ground truth = public regulatory definition on true "
            "facts; predictions run on extracted facts.",
            "",
            "## Confusion matrix (rows = ground truth, cols = predicted)",
            "",
            "| GT \\ Pred | " + " | ".join(_ORDER) + " |",
            "|" + "---|" * (len(_ORDER) + 1),
        ]
        for g in _ORDER:
            row = self.confusion.get(g, {})
            lines.append(f"| **{g}** | " + " | ".join(str(row.get(p, 0)) for p in _ORDER) + " |")
        lines += ["", "## Per-class metrics", "",
                  "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for c in _ORDER:
            m = self.per_class.get(c, {})
            lines.append(f"| {c} | {m.get('precision', 0):.3f} | {m.get('recall', 0):.3f} | "
                         f"{m.get('f1', 0):.3f} | {m.get('support', 0)} |")
        lines += [
            "",
            "## Headline — deception catch-rate",
            "",
            f"- **MISLEADING recall: {self.misleading_recall:.3f}** (deceptive claims caught)",
            f"- MISLEADING precision: {self.misleading_precision:.3f} · F1: {self.misleading_f1:.3f}",
            f"- Macro-F1: {self.macro_f1:.3f}",
            "",
            "## Worked examples",
            "",
        ]
        for ex in self.worked_examples:
            lines += [
                f"**{ex['claim_text']}** on a *{ex['category']}* product "
                f"(GT {ex['gt']} / predicted {ex['pred']}){' — HEALTH-HALO' if ex['health_halo'] else ''}",
                f"- Verdict: **{ex['pred']}** · severity {ex['severity']}",
                f"- Basis: {ex['basis']}",
            ]
            if ex.get("offending_fact"):
                lines.append(f"- Offending fact: {ex['offending_fact']}")
            for r in ex.get("reasons", []):
                lines.append(f"- {r}")
            lines.append("")
        return "\n".join(lines)


def run_deception_eval(products: list[Any], mode: str = "extracted") -> DeceptionReport:
    labeled = build_labeled_set(products)
    by_code = {p.code: p for p in products}
    adj = Adjudicator()

    facts_cache: dict[str, tuple] = {}

    def facts(code: str):
        if code not in facts_cache:
            facts_cache[code] = facts_from_product(by_code[code], mode)
        return facts_cache[code]

    confusion = {g: {p: 0 for p in _ORDER} for g in _ORDER}
    verdict_cache: list[tuple] = []
    for case in labeled:
        panel, ingredients, base_grade = facts(case.product_code)
        verdict = adj.adjudicate(case.claim_text, panel, ingredients, base_grade)
        confusion[case.gt_status.value][verdict.status.value] += 1
        verdict_cache.append((case, verdict))

    per_class = _per_class_metrics(confusion)
    mis = per_class[ClaimStatus.misleading.value]
    report = DeceptionReport(
        mode=mode,
        n_cases=len(labeled),
        confusion=confusion,
        per_class=per_class,
        misleading_recall=mis["recall"],
        misleading_precision=mis["precision"],
        misleading_f1=mis["f1"],
        macro_f1=round(sum(per_class[c]["f1"] for c in _ORDER) / len(_ORDER), 4),
        worked_examples=_worked_examples(verdict_cache),
    )
    return report


def _per_class_metrics(confusion: dict) -> dict:
    out = {}
    for c in _ORDER:
        tp = confusion[c][c]
        fp = sum(confusion[g][c] for g in _ORDER if g != c)
        fn = sum(confusion[c][p] for p in _ORDER if p != c)
        support = sum(confusion[c][p] for p in _ORDER)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[c] = {"precision": round(precision, 4), "recall": round(recall, 4),
                  "f1": round(f1, 4), "support": support}
    return out


def _worked_examples(verdict_cache: list[tuple]) -> list[dict]:
    """Pick a caught deception, a health-halo, a missed deception, and an unverifiable."""
    picks: list[dict] = []
    seen = set()

    def add(case, verdict, tag):
        if tag in seen:
            return
        seen.add(tag)
        picks.append({
            "tag": tag, "claim_text": case.claim_text, "category": case.category,
            "gt": case.gt_status.value, "pred": verdict.status.value,
            "basis": verdict.basis, "offending_fact": verdict.offending_fact,
            "reasons": verdict.reasons, "severity": verdict.severity.value,
            "health_halo": verdict.health_halo,
        })

    for case, verdict in verdict_cache:
        if case.gt_status == ClaimStatus.misleading and verdict.status == ClaimStatus.misleading:
            add(case, verdict, "caught_deception")
        if verdict.health_halo:
            add(case, verdict, "health_halo")
        if case.gt_status == ClaimStatus.misleading and verdict.status != ClaimStatus.misleading:
            add(case, verdict, "missed_deception")
        if verdict.status == ClaimStatus.unverifiable and case.claim_id == "organic":
            add(case, verdict, "unverifiable")
    order = ["caught_deception", "health_halo", "missed_deception", "unverifiable"]
    return sorted(picks, key=lambda x: order.index(x["tag"]) if x["tag"] in order else 99)
