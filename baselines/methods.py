"""
baselines/methods.py

Uniform random search baseline (non-RL).
Samples full configs theta in [0,1]^N, projects into the feasible set,
evaluates via the oracle, and accumulates all evaluated configs/rewards/probs.
The caller (run_baseline.py) selects the top-N from the returned arrays.

No actions, episodes, trajectories, actor, critic, or replay buffer.
Deduplication: keeps a `seen` set of already-evaluated projected configs
(keyed by bytes). Duplicate projections are resampled before evaluation,
so every oracle call is guaranteed to be on a fresh, unseen config.
"""

from __future__ import annotations

import numpy as np

from common import evaluate_config, project_theta

_MAX_RESAMPLE = 500   # safety limit for deduplication retries


def _unique_project(sample_fn, selected_params, seen):
    """
    Call sample_fn() repeatedly until project_theta gives an unseen config.

    sample_fn:       callable() -> (N,) float array (raw, unprojected)
    selected_params: list[str] length N
    seen:            set of bytes keys of already-evaluated projected configs
    Returns:         (N,) float32 — projected, unique config; key added to seen
    Raises:          RuntimeError after _MAX_RESAMPLE failed attempts
    """
    for attempt in range(_MAX_RESAMPLE):
        theta_proj = project_theta(sample_fn(), selected_params)
        key = theta_proj.tobytes()
        if key not in seen:
            seen.add(key)
            return theta_proj
        if attempt == 0:
            print("[dedup] collision detected — resampling")
    raise RuntimeError(
        f"Could not find a unique config after {_MAX_RESAMPLE} attempts. "
        "The feasible region may be too small for the requested budget."
    )


# ── Uniform random search ───────────────────────────────────────────────────

def uniform(
    cfg: dict,
    env,
    tmp_img: str,
    selected_params: list[str],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample theta_i ~ Uniform({0, 1, …, 100}/100) independently per dimension.
    Samples directly from the 101-point discrete grid — no continuous rounding.

    Returns:
      all_configs: (budget, N) float32
      all_rewards: (budget,)   float32
      all_probs:   (budget, 7) float32
    """
    budget = int(cfg["budget"])
    N = len(selected_params)
    all_configs = np.zeros((budget, N), dtype=np.float32)
    all_rewards = np.zeros(budget, dtype=np.float32)
    all_probs   = np.zeros((budget, 7), dtype=np.float32)
    seen: set[bytes] = set()

    for i in range(budget):
        theta_proj = _unique_project(
            lambda: (rng.integers(0, 101, size=N) / 100.0).astype(np.float32),
            selected_params, seen,
        )
        r, theta_proj, probs = evaluate_config(theta_proj, env, tmp_img, selected_params)
        all_configs[i] = theta_proj
        all_rewards[i] = r
        all_probs[i]   = probs
        print(f"[uniform] {i+1}/{budget}  reward={r:.4f}")

    return all_configs, all_rewards, all_probs
