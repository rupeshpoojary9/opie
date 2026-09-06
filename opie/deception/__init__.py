"""Truth & deception layer.

For every front-of-pack marketing claim, decide whether the product's actual nutrition +
ingredients SUPPORT it, and flag misleading claims with an explainable, evidence-backed
verdict grounded in PUBLIC regulatory definitions (EU Reg 1924/2006, Codex CAC/GL 23-1997,
FSSAI Labelling & Display Regs 2020). Optimized to catch deceptive claims (recall on
MISLEADING). See `adjudicator.Adjudicator`.
"""
from opie.deception.adjudicator import Adjudicator, adjudicate_product

__all__ = ["Adjudicator", "adjudicate_product"]
