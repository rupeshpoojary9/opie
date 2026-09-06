"""Adapter: run the real extraction Pipeline and convert its output to FieldObservations,
so the exact same feedback loop wraps the Claude-vision (or offline Tesseract) backend when
a key + product images are available. Retrieval-augmented exemplars from the correction
memory can be injected into the vision agents' prompts via memory.exemplars()."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from opie.feedback.types import FieldObservation
from opie.feedback.vectorize import magnitude_bucket
from opie.pipeline.graph import Pipeline


def pipeline_emit_fn(pipeline: Pipeline) -> Callable[[Any], list[FieldObservation]]:
    def emit(product: Any) -> list[FieldObservation]:
        off = getattr(product, "off", None)
        if off is None or not off.local_image or not Path(off.local_image).exists():
            raise FileNotFoundError(
                f"Product {product.code} has no local image; run scripts.download_off_subset first.")
        image_bytes = Path(off.local_image).read_bytes()
        result = pipeline.run(product.code, image_bytes)
        obs: list[FieldObservation] = []
        for name, attr in result.nutrition.as_dict().items():
            if attr.value is None:
                continue
            obs.append(FieldObservation(
                product_id=product.code, kind="nutrition", name=name,
                extracted_value=attr.value, raw_confidence=attr.confidence,
                features={"name": name, "category": product.category,
                          "mag": magnitude_bucket(attr.value),
                          "unit": attr.unit or "g"},
                evidence=attr.evidence.text if attr.evidence else None))
        for rank, ing in enumerate(result.ingredients, start=1):
            obs.append(FieldObservation(
                product_id=product.code, kind="ingredient", name=f"ing{rank}",
                extracted_value=ing.name, raw_confidence=ing.confidence,
                features={"name": "ingredient", "category": product.category, "token": ing.name},
                evidence=ing.evidence.text if ing.evidence else None))
        return obs
    return emit
