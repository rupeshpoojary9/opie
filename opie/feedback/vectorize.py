"""Deterministic feature embedding + cosine retrieval (stdlib only, no model, no key).

Each correction case is embedded into a fixed-dim hashed vector over its features
(field name, category, value-magnitude bucket, unit, context tokens). Retrieval is
top-k cosine. A real embedding model could replace `embed()` without changing the
retrieval interface; this keeps the flywheel demo fully offline and reproducible.
"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable

DIM = 256

# Feature weights: field name and category dominate so same-error-mode cases are nearest.
_WEIGHTS = {
    "name": 3.0,
    "category": 2.5,
    "mag": 1.5,
    "unit": 1.0,
    "token": 1.0,
    "ctx": 0.7,
}


def _hash_index(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % DIM


def magnitude_bucket(value) -> str:
    try:
        v = abs(float(value))
    except (TypeError, ValueError):
        return "na"
    if v <= 0:
        return "z"
    return str(int(math.floor(math.log10(v + 1e-9))))


def embed(features: dict) -> list[float]:
    """Hashed bag-of-features embedding, L2-normalized."""
    vec = [0.0] * DIM
    for key, weight in _WEIGHTS.items():
        val = features.get(key)
        if val is None:
            continue
        vals = val if isinstance(val, (list, tuple, set)) else [val]
        for item in vals:
            idx = _hash_index(f"{key}={item}")
            vec[idx] += weight
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def topk(query_vec: list[float], candidates: Iterable[tuple[int, list[float]]], k: int) -> list[tuple[int, float]]:
    """Return [(candidate_id, similarity), ...] for the k most similar, stable tie-break by id."""
    scored = [(cid, cosine(query_vec, cvec)) for cid, cvec in candidates]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored[:k]
