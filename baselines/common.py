"""
baselines/common.py

Shared infrastructure for all baseline methods:
  - Canonical parameter list and mirror map (imported from RL code)
  - project_theta: enforce bounds + anatomical constraints on a (N,) config
  - build_full_state: expand (N,) theta to the 62-key state dict / ndarray
  - evaluate_config: project → render on Furhat → DAN reward (single oracle call)
  - save_outputs: write configs.npy, rewards.npy, metadata.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ── Repo root on path ─────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from policy_gradient.constants import base_expression as _BASE_EXPR
from policy_gradient.mixins.core import CoreMixin

# ── Public constants ──────────────────────────────────────────────────────────

# 62-key ordered list — defines index mapping for env.step()
BASE_EXPR_KEYS: list[str] = list(_BASE_EXPR.keys())

# Bilateral mirror map: LEFT param name → RIGHT param name
MIRROR_MAP: dict[str, str] = CoreMixin.MIRROR_MAP

# Default 26-D selected-parameter list (order matches RL SELECTED_PARAMS in config.yaml)
SELECTED_PARAMS: list[str] = [
    "MOUTH_LEFT", "MOUTH_DIMPLE_LEFT", "MOUTH_STRETCH_LEFT",
    "BROW_INNER_UP", "BROW_OUTER_UP_LEFT",
    "MOUTH_FROWN_LEFT", "MOUTH_FUNNEL",
    "MOUTH_LOWER_DOWN_LEFT", "MOUTH_PRESS_LEFT", "MOUTH_PUCKER",
    "MOUTH_ROLL_LOWER", "MOUTH_ROLL_UPPER",
    "MOUTH_SHRUG_LOWER", "MOUTH_SHRUG_UPPER",
    "MOUTH_UPPER_UP_LEFT",
    "BROW_DOWN_LEFT", "BROW_IN_LEFT", "BROW_UP_LEFT",
    "EYE_SQUINT_LEFT", "EYE_WIDE_LEFT", "EYE_BLINK_LEFT",
    "CHEEK_SQUINT_LEFT", "CHEEK_PUFF",
    "NOSE_SNEER_LEFT", "JAW_OPEN", "JAW_FORWARD",
]


# ── Projection ────────────────────────────────────────────────────────────────

def project_theta(theta: np.ndarray, selected_params: list[str] = SELECTED_PARAMS) -> np.ndarray:
    """
    Quantise to the 0.01 grid then project into the feasible set.

    theta:           (N,) float — raw sampled config, any range
    selected_params: list[str] length N — param name for each dimension
    Returns:         (N,) float32 — values in {0.00, 0.01, …, 1.00},
                     all anatomical constraints satisfied

    All constraint arithmetic is done on the integer 0-100 scale so there are
    no float32 rounding issues.  The result is bitwise-idempotent.
    """
    # ── 1. Quantise: round to nearest 0.01, clamp to [0, 100] ────────────────
    t = np.clip(
        np.round(np.asarray(theta, dtype=np.float64) * 100), 0, 100
    ).astype(np.int32)                                          # (N,) int32 in [0, 100]
    idx = {name: i for i, name in enumerate(selected_params)}

    # ── 2. Mouth height: upper_int – lower_int ≥ 7  (== gap ≥ 0.07) ─────────
    for upper_name, lower_name in [
        ("MOUTH_SHRUG_UPPER",   "MOUTH_SHRUG_LOWER"),
        ("MOUTH_UPPER_UP_LEFT", "MOUTH_LOWER_DOWN_LEFT"),
        ("MOUTH_ROLL_UPPER",    "MOUTH_ROLL_LOWER"),
    ]:
        if upper_name not in idx or lower_name not in idx:
            continue
        u, l = idx[upper_name], idx[lower_name]
        if t[u] - t[l] < 7:
            t[l] = t[u] - 7
            if t[l] < 0:
                t[l] = 0
                if t[u] < 7:
                    t[u] = 7

    # ── 3. Eye occlusion ─────────────────────────────────────────────────────
    # squint ≤ 60, blink ≤ 30, squint + 2*blink ≤ 33
    # (derived from squint/60 + blink/30 ≤ 0.55 on the 0–100 integer grid)
    sq_key, bl_key = "EYE_SQUINT_LEFT", "EYE_BLINK_LEFT"
    if sq_key in idx and bl_key in idx:
        sq, bl = idx[sq_key], idx[bl_key]
        t[sq] = min(int(t[sq]), 60)
        t[bl] = min(int(t[bl]), 30)
        combined = int(t[sq]) + 2 * int(t[bl])
        if combined > 33:
            t[sq] = int(t[sq]) * 33 // combined   # floor keeps constraint
            t[bl] = int(t[bl]) * 33 // combined
            # floor can leave combined at 32 or 33; one extra guard pass
            while int(t[sq]) + 2 * int(t[bl]) > 33:
                if int(t[bl]) > 0:
                    t[bl] -= 1
                else:
                    t[sq] -= 1

    # ── 4. Back to float32 on the 0.01 grid ──────────────────────────────────
    return (t / 100.0).astype(np.float32)


# ── State construction ────────────────────────────────────────────────────────

def build_full_state(
    theta: np.ndarray,
    selected_params: list[str] = SELECTED_PARAMS,
) -> np.ndarray:
    """
    Expand a (N,) selected-param config to the full 62-D state vector.

    Sets mirrored RIGHT params to the same value as their LEFT counterparts.
    All unselected params stay at 0 (neutral).

    theta:           (N,) float32 — projected config in SELECTED_PARAMS order
    selected_params: list[str] length N
    Returns:         (61,) float32 — indexed by BASE_EXPR_KEYS
    """
    state = np.zeros(len(BASE_EXPR_KEYS), dtype=np.float32)
    for i, param in enumerate(selected_params):
        j = BASE_EXPR_KEYS.index(param)
        state[j] = float(theta[i])
        mirror = MIRROR_MAP.get(param)
        if mirror is not None and mirror in BASE_EXPR_KEYS:
            state[BASE_EXPR_KEYS.index(mirror)] = float(theta[i])
    return state


# ── Evaluation oracle ─────────────────────────────────────────────────────────

_eval_count = [0]  # module-level call counter; reset by run_baseline


def evaluate_config(
    theta: np.ndarray,
    env,
    tmp_img_path: str,
    selected_params: list[str] = SELECTED_PARAMS,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Project theta, render on Furhat, classify with DAN, return reward + full probs.

    theta:          (N,) float — will be projected before evaluation
    env:            FurhatEnv instance (already initialised with correct character/emotion)
    tmp_img_path:   str — screenshot saved here when step_num % 10 == 0 (DAN internal)
    selected_params: list[str] length N
    Returns:
      reward:     float
      theta_proj: (N,) float32 — the actually-used (projected) config
      probs:      (7,) float32 — full emotion probability vector (zeros on classifier failure)
    """
    _eval_count[0] += 1
    theta_proj = project_theta(theta, selected_params)
    state = build_full_state(theta_proj, selected_params)

    _, reward, _, probs, _, _ = env.step(
        state,
        step_num=_eval_count[0],
        episode_num=0,
        save_path=tmp_img_path,
        annotate_save=False,
    )
    probs_arr = (
        np.asarray(probs, dtype=np.float32).reshape(-1)
        if probs is not None
        else np.zeros(7, dtype=np.float32)
    )
    return float(reward), theta_proj, probs_arr


# ── Output saving ─────────────────────────────────────────────────────────────

def save_outputs(
    configs: np.ndarray,
    rewards: np.ndarray,
    metadata: dict,
    out_dir: Path,
) -> None:
    """
    Persist top-N results to disk.

    configs:  (N, dim) float32 — best configs in descending reward order
    rewards:  (N,)     float32 — corresponding rewards
    metadata: dict             — run metadata written as JSON
    out_dir:  Path             — destination directory (created if absent)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(out_dir / "configs.npy"), configs.astype(np.float32))
    np.save(str(out_dir / "rewards.npy"), rewards.astype(np.float32))

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[save] {len(rewards)} configs saved → {out_dir}")
