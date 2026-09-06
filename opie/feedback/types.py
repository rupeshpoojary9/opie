"""Shared types for the feedback loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

Value = Union[float, str, None]


@dataclass
class FieldObservation:
    """One extracted field decision — the unit the flywheel captures, reviews, and learns from."""
    product_id: str
    kind: str                       # "nutrition" | "ingredient"
    name: str                       # nutrition field name, or ingredient token slot id
    extracted_value: Value          # what the extractor produced (post-correction if applied)
    raw_confidence: float           # extractor's own confidence
    features: dict = field(default_factory=dict)   # category, magnitude bucket, etc. (for retrieval)
    evidence: Optional[str] = None

    # filled downstream:
    calibrated_confidence: Optional[float] = None
    corrected: bool = False         # did the correction layer change the value?
    validator_flagged: bool = False
    queued: bool = False            # routed to the review queue this round

    # oracle-labeled (ground truth), used for metrics + learning:
    gt_value: Value = None
    is_correct: Optional[bool] = None


@dataclass
class CorrectionCase:
    """A reviewed correction: what the extractor said vs. the oracle's truth, plus retrieval features."""
    product_id: str
    kind: str
    name: str
    extracted_value: Value
    correct_value: Value
    features: dict = field(default_factory=dict)
    round_index: int = 0
