"""Extraction backends: Claude vision (real) and offline (Tesseract + heuristics)."""
from __future__ import annotations

from opie.config import Settings
from opie.llm.base import Extractor


def build_extractor(settings: Settings) -> Extractor:
    if settings.backend == "anthropic":
        from opie.llm.anthropic_vision import AnthropicVisionExtractor
        return AnthropicVisionExtractor(model=settings.model)
    if settings.backend == "offline":
        from opie.llm.offline import OfflineExtractor
        return OfflineExtractor()
    raise ValueError(f"Unknown backend: {settings.backend!r} (expected 'anthropic' or 'offline')")
