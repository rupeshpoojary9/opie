"""Real-world validation feedback loop (the data flywheel).

The system gets measurably more accurate over time as corrections flow back in, and its
low-confidence review queue shrinks. Corrections are learned by retrieval (k-NN over past
correction cases -> learned numeric transforms + token-normalization maps) and by
confidence recalibration — no fine-tune required for v1. See `loop.FeedbackLoop`.
"""
from opie.feedback.loop import FeedbackLoop
from opie.feedback.report import FeedbackReport

__all__ = ["FeedbackLoop", "FeedbackReport"]
