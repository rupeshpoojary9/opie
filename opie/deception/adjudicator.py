"""Adjudicator agent: cross-reference each claim against its regulatory definition.

Produces an explainable ClaimVerdict (SUPPORTED | MISLEADING | UNVERIFIABLE) citing the
rule and the offending fact, and flags health-halo claims — technically-true claims on a
product with a poor overall profile (base grade D/E or very high sugar).
"""
from __future__ import annotations

from typing import Optional

from opie.deception.claims import normalize_claim
from opie.deception.definitions import DEFINITIONS, Facts
from opie.schemas import (
    Claim,
    ClaimStatus,
    ClaimVerdict,
    Evidence,
    Ingredient,
    NutritionPanel,
    ProductIntelligence,
    Severity,
)

HALO_SUGAR_THRESHOLD = 22.5   # UK FoP "high sugar" (g/100g)
HALO_GRADES = {"D", "E"}


class Adjudicator:
    def adjudicate(
        self,
        claim_text: str,
        nutrition: NutritionPanel,
        ingredients: list[Ingredient],
        base_grade: Optional[str] = None,
        claim_evidence: Optional[Evidence] = None,
    ) -> ClaimVerdict:
        claim_id = normalize_claim(claim_text)
        facts = Facts(nutrition=nutrition, ingredients=ingredients)

        if claim_id is None or claim_id not in DEFINITIONS:
            return ClaimVerdict(
                claim_text=claim_text, claim_id=claim_id, status=ClaimStatus.unverifiable,
                basis="No regulatory definition for this claim.",
                reasons=["Claim text did not map to a known nutrition/content claim."],
                evidence=[claim_evidence] if claim_evidence else [], severity=Severity.low,
            )

        outcome = DEFINITIONS[claim_id].check(facts)
        evidence = [Evidence(text=t) for t in outcome.evidence]
        if claim_evidence:
            evidence.insert(0, claim_evidence)

        verdict = ClaimVerdict(
            claim_text=claim_text, claim_id=claim_id, status=outcome.status,
            basis=outcome.basis, offending_fact=outcome.offending_fact,
            reasons=list(outcome.reasons), evidence=evidence, severity=outcome.severity,
        )
        self._apply_health_halo(verdict, nutrition, base_grade)
        return verdict

    def _apply_health_halo(self, verdict: ClaimVerdict, nutrition: NutritionPanel,
                           base_grade: Optional[str]) -> None:
        if verdict.status != ClaimStatus.supported:
            return
        sugars = nutrition.sugars_100g.value
        poor_grade = base_grade is not None and base_grade.upper() in HALO_GRADES
        high_sugar = sugars is not None and sugars >= HALO_SUGAR_THRESHOLD
        if poor_grade or high_sugar:
            verdict.health_halo = True
            why = []
            if poor_grade:
                why.append(f"overall Nutri-Score grade {base_grade.upper()}")
            if high_sugar:
                why.append(f"sugars {sugars:g}g/100g")
            verdict.reasons.append(
                "Health-halo: the claim is technically true but distracts from a poor overall "
                f"profile ({', '.join(why)}).")
            if verdict.severity == Severity.low:
                verdict.severity = Severity.medium


def adjudicate_product(product: ProductIntelligence) -> list[ClaimVerdict]:
    """Fill and return claim verdicts for a full intelligence record."""
    adj = Adjudicator()
    verdicts = [
        adj.adjudicate(
            claim_text=c.text or c.normalized or "",
            nutrition=product.nutrition,
            ingredients=product.ingredients,
            base_grade=product.score.base_grade,
            claim_evidence=c.evidence,
        )
        for c in product.claims
    ]
    product.claim_verdicts = verdicts
    return verdicts
