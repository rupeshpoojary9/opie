"""Personalized scoring: a configurable profile matrix -> red/yellow/green + reasons.

Each profile is data (per-field thresholds), so adding a condition is a JSON edit, not a
code change. The overall flag is the worst per-field flag, and every flag carries an
explainable reason string (which field, which threshold, the actual value).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from opie.schemas import NutritionPanel, PersonalizedScore, TrafficLight

_PROFILES_PATH = Path(__file__).resolve().parent / "profiles.json"
_RANK = {TrafficLight.green: 0, TrafficLight.yellow: 1, TrafficLight.red: 2}


@lru_cache(maxsize=1)
def _load() -> dict:
    with _PROFILES_PATH.open() as fh:
        return json.load(fh)["profiles"]


PROFILES = tuple(_load().keys())


def personalize(panel: NutritionPanel, profile: str) -> PersonalizedScore:
    profiles = _load()
    if profile not in profiles:
        raise ValueError(f"Unknown profile {profile!r}. Known: {', '.join(profiles)}")
    spec = profiles[profile]
    vals = {k: attr.value for k, attr in panel.as_dict().items()}

    overall = TrafficLight.green
    reasons: list[str] = []

    for field_name, thr in spec["fields"].items():
        v = vals.get(field_name)
        if v is None:
            continue
        flag, reason = _flag_field(field_name, v, thr)
        if flag != TrafficLight.green:
            reasons.append(reason)
        if _RANK[flag] > _RANK[overall]:
            overall = flag

    if not reasons:
        reasons.append(f"No {spec['label']} concerns in the extracted nutrition.")
    return PersonalizedScore(profile=profile, flag=overall, why=reasons)


def _flag_field(field_name: str, v: float, thr: dict) -> tuple[TrafficLight, str]:
    unit = thr.get("unit", field_name)
    # "higher is worse" fields
    if "red" in thr and v >= thr["red"]:
        return TrafficLight.red, f"{v:g} {unit} is high (>= {thr['red']:g})."
    if "yellow" in thr and v >= thr["yellow"]:
        return TrafficLight.yellow, f"{v:g} {unit} is moderate (>= {thr['yellow']:g})."
    # "higher is better" fields (e.g. fibre): low is a concern
    if "yellow_below" in thr and v < thr["yellow_below"]:
        return TrafficLight.yellow, f"{v:g} {unit} is low (< {thr['yellow_below']:g})."
    return TrafficLight.green, ""
