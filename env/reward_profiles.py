"""Opt-in reward shaping; observations and vehicle controls remain unchanged."""
import json
import os
from pathlib import Path

import numpy as np


def reward_profile():
    name = os.environ.get("YEOUIDO_REWARD_PROFILE", "legacy")
    if name not in ("legacy", "clear_idle_v1"):
        raise ValueError(f"Unknown reward profile: {name}")
    return name


def clear_idle_candidate(obs, vmax, visibility, route_range):
    """Only clearly open road, away from a junction. No expert queries."""
    if obs is None or len(obs) not in (42, 64) or not np.isfinite(obs).all():
        return False
    return bool(obs[0] * vmax < 0.5
                and obs[9] * visibility >= 15.0
                and obs[31] > 0.5
                and obs[33] * route_range >= 30.0
                and obs[23] > 0.5)


def apply_reward_profile(reward):
    out = dict(reward)
    if reward_profile() == "clear_idle_v1":
        weight = float(os.environ.get("YEOUIDO_CLEAR_IDLE_WEIGHT", "0.005"))
        if not np.isfinite(weight) or not 0.0 <= weight <= 0.05:
            raise ValueError("Clear-idle penalty must be finite and between 0 and 0.05")
        out["clear_idle_penalty"] = weight
    return out


def restore_reward_profile(model_path):
    """Explicit environment overrides win; otherwise restore a saved run profile."""
    for parent in list(Path(model_path).resolve().parents)[:4]:
        path = parent / "environment.json"
        if path.exists():
            data = json.loads(path.read_text())
            effective=data.get('deployment',{}).get('effective',data.get('effective',{}))
            gate=effective.get('lc_safety_enabled',effective.get('lane_change_safety'))
            if gate is not None:
                os.environ.setdefault('YEOUIDO_LANE_CHANGE_GATE','1' if gate else '0')
            os.environ.setdefault("YEOUIDO_REWARD_PROFILE", data.get("reward_profile", "legacy"))
            if "clear_idle_penalty" in data.get("effective", {}):
                os.environ.setdefault("YEOUIDO_CLEAR_IDLE_WEIGHT", str(data["effective"]["clear_idle_penalty"]))
            if "gamma" in data:
                os.environ.setdefault("YEOUIDO_DISCOUNT_GAMMA", str(data["gamma"]))
            return
