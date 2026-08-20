"""
baselines/run_baseline.py

Entry point for the uniform random search baseline.  All settings are read
from baselines/config_baseline.yaml — no argparse, no command-line flags.

Usage:
    python baselines/run_baseline.py          (from repo root)
    python -m baselines.run_baseline          (from repo root)

Output structure (inside out_dir/<job_id>/):
    all_configs_log.csv       — every evaluated config + probs, sorted best→worst
    top_<n_top>_log.csv       — top-N configs + probs from the re-rendered images
    Top_<n_top>/top_01.jpg … — annotated images, rank 1 = best
    configs.npy               — (n_top, N) float32
    rewards.npy               — (n_top,)   float32
    metadata.json
    config_baseline.yaml      — copy of the run config
"""

from __future__ import annotations

import csv
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import yaml

# ── Repo root on path (must be before any policy_gradient imports) ─────────────
_REPO = Path(__file__).resolve().parents[1]
_BASELINES = Path(__file__).resolve().parent
for _p in (_BASELINES, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from policy_gradient.constants import base_expression
from policy_gradient.env import ExpressionUtils, FurhatEnv

import common as _common
from common import build_full_state
import methods as _methods

_CANONICAL_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


# ── Config loading ────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    cfg_path = Path(__file__).parent / "config_baseline.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Environment construction ──────────────────────────────────────────────────

def _build_env(cfg: dict):
    """Instantiate FurhatEnv + ExpressionUtils from config_baseline.yaml keys."""
    emotion_str = str(cfg["emotion"]).strip().lower()
    if emotion_str not in _CANONICAL_EMOTIONS:
        raise ValueError(
            f"Unknown emotion '{emotion_str}'. Expected one of: {_CANONICAL_EMOTIONS}"
        )
    emotion_index = _CANONICAL_EMOTIONS.index(emotion_str)
    emotion_name  = emotion_str.title()

    character = str(cfg["character"])

    rl_cfg = {
        "BASE_EXPRESSION":          base_expression,
        "DESIGNATED_EMOTION":       emotion_index,
        "DESIGNATED_EMOTION_INDEX": emotion_index,
        "DESIGNATED_EMOTION_NAME":  emotion_name,
        "DESIGNATED_EMOTION_SLUG":  emotion_str,
        "GAP_BASED_REWARD":         bool(cfg.get("gap_based_reward", True)),
        "WEIGHTED_GAP_BASED_REWARD": bool(cfg.get("weighted_gap_based_reward", True)),
        "WEIGHTED_GAP_ALPHA":       float(cfg.get("weighted_gap_alpha", 0.5)),
    }

    classifier_init_kwargs = {
        "dan_preset":   cfg.get("dan_preset"),
        "detect_face":  cfg.get("dan_detect_face", True),
        "if_no_face":   cfg.get("dan_if_no_face", "fail"),
    }

    # Maurice uses a dedicated template (matches RL trainer.py behaviour)
    default_template = (
        "furhat_template_maurice.png"
        if character.strip().lower() == "maurice"
        else "furhat_template.png"
    )
    template_path = str(cfg.get("template_path") or default_template)

    expr_utils = ExpressionUtils(
        classifier_type=str(cfg.get("classifier_type", "dan")),
        furhat_ip=str(cfg["furhat_ip"]),
        fer_model_path=str(cfg["fer_model_path"]),
        template_path=template_path,
        capture_from_monitor=bool(cfg.get("capture_from_monitor", False)),
        furhat_texture=character,
        furhat_texture_settle_s=float(cfg.get("furhat_texture_settle_s", 0.35)),
        furhat_texture_strict=bool(cfg.get("furhat_texture_strict", True)),
        classifier_module=cfg.get("classifier_module") or None,
        classifier_class=cfg.get("classifier_class") or None,
        classifier_init_kwargs=classifier_init_kwargs,
        disable_microexpressions=bool(cfg.get("furhat_disable_microexpressions", True)),
        disable_blinking=bool(cfg.get("furhat_disable_blinking", True)),
        disable_head_sway=bool(cfg.get("furhat_disable_head_sway", True)),
        reset_neutral_after_texture=bool(cfg.get("furhat_reset_neutral_after_texture", True)),
        set_head_pose=bool(cfg.get("furhat_set_head_pose", True)),
        head_yaw=float(cfg.get("furhat_head_yaw", 0.0)),
        head_pitch=float(cfg.get("furhat_head_pitch", 0.0)),
        head_roll=float(cfg.get("furhat_head_roll", 0.0)),
    )

    env = FurhatEnv(rl_cfg, expr_utils)
    return env


# ── CSV helper ────────────────────────────────────────────────────────────────

def _write_csv(
    path: Path,
    selected_params: list[str],
    configs: np.ndarray,
    rewards: np.ndarray,
    probs: np.ndarray,
    rank_offset: int = 1,
) -> None:
    """
    Write a CSV with columns: rank, reward, <param_0..N-1>, angry..neutral.
    Rows are written in the order supplied; caller is responsible for sorting.
    """
    header = ["rank", "reward"] + list(selected_params) + _CANONICAL_EMOTIONS

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, (theta, r, p) in enumerate(zip(configs, rewards, probs)):
            row = (
                [rank_offset + i, f"{r:.6f}"]
                + [f"{v:.2f}" for v in theta.tolist()]
                + [f"{v:.6f}" for v in p.tolist()]
            )
            writer.writerow(row)

    print(f"[log] {len(rewards)} rows → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = _load_cfg()

    method          = str(cfg["method"]).strip().lower()
    emotion         = str(cfg["emotion"]).strip().lower()
    character       = str(cfg["character"])
    seed            = int(cfg.get("seed", 42))
    budget          = int(cfg["budget"])
    n_top           = int(cfg.get("n_top", 10))
    selected_params = list(cfg.get("selected_params", _common.SELECTED_PARAMS))
    N               = len(selected_params)

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    _common._eval_count[0] = 0

    print(
        f"[run_baseline] method={method}  emotion={emotion}  "
        f"character={character}  seed={seed}  budget={budget}  N={N}"
    )

    job_id  = str(uuid.uuid4())[:8]
    out_dir = Path(cfg.get("out_dir", "output_baseline")) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_img = str(out_dir / "tmp_eval.jpg")
    print(f"[run_baseline] output → {out_dir}")

    env = _build_env(cfg)

    # ── Method dispatch ───────────────────────────────────────────────────────
    t0 = time.time()

    if method == "uniform":
        all_configs, all_rewards, all_probs = _methods.uniform(
            cfg, env, tmp_img, selected_params, rng
        )

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: uniform"
        )

    elapsed = time.time() - t0

    # ── All-configs log (sorted best → worst reward) ──────────────────────────
    sort_idx     = np.argsort(all_rewards)[::-1]
    sorted_cfgs  = all_configs[sort_idx]
    sorted_rews  = all_rewards[sort_idx]
    sorted_probs = all_probs[sort_idx]

    _write_csv(
        out_dir / "all_configs_log.csv",
        selected_params,
        sorted_cfgs,
        sorted_rews,
        sorted_probs,
    )

    # ── Top-N selection ───────────────────────────────────────────────────────
    n_save      = min(n_top, len(all_rewards))
    top_configs  = sorted_cfgs[:n_save]
    top_rewards  = sorted_rews[:n_save]
    top_probs_arr = sorted_probs[:n_save]          # (n_save, 7) — search-phase probs, no re-classify

    best_reward = float(top_rewards[0]) if n_save > 0 else 0.0
    emotion_idx = _CANONICAL_EMOTIONS.index(emotion)

    # ── Render top-N images → Top_<n_top>/ ───────────────────────────────────
    # Render the expression on Furhat and capture the face image, but DO NOT
    # re-run DAN inference.  Annotation uses the probs already captured during
    # the search phase (stored in top_probs_arr) so the image and CSV agree
    # with the data that drove the ranking.
    top_img_dir = out_dir / f"Top_{n_top}"
    top_img_dir.mkdir(exist_ok=True)

    _dan           = getattr(getattr(env, "expr_utils", None), "emotion_classifier", None)
    _from_monitor  = getattr(getattr(env, "expr_utils", None), "capture_from_monitor", False)
    _template_path = getattr(getattr(env, "expr_utils", None), "template_path", "furhat_template.png")

    top_probs_list: list[list[float]] = []

    for rank, (theta, stored_probs) in enumerate(zip(top_configs, top_probs_arr), start=1):
        img_path = str(top_img_dir / f"top_{rank:02d}.jpg")

        # 1. Render expression on Furhat (no classification)
        state = build_full_state(theta, selected_params)
        expr  = {k: float(v) for k, v in zip(_common.BASE_EXPR_KEYS, state)}
        env.expr_utils.perform_expression(expr)

        # 2. Capture face image only — no DAN inference
        if _dan is not None:
            if _from_monitor:
                _dan.capture_face_image_right_screen(_template_path, img_path)
            else:
                _dan.capture_face_image(_template_path, img_path)

        # 3. Annotate with search-phase probs (same data that ranked this config)
        p_target = float(stored_probs[emotion_idx])
        env._annotate_saved_image(img_path, stored_probs, p_target)

        p_vec = stored_probs.tolist()
        top_probs_list.append(p_vec)
        print(f"[run_baseline] top-{rank:02d} → {img_path}  p_{emotion}={p_target:.4f}")

    # ── Top-N log — reward and probs both from search phase, match image annotation ──
    _write_csv(
        out_dir / f"top_{n_top}_log.csv",
        selected_params,
        top_configs,
        top_rewards,
        top_probs_arr,
    )

    # ── Copy config for reproducibility ──────────────────────────────────────
    shutil.copy2(
        Path(__file__).parent / "config_baseline.yaml",
        out_dir / "config_baseline.yaml",
    )

    # ── Metadata JSON ─────────────────────────────────────────────────────────
    metadata = {
        "method":          method,
        "character":       character,
        "emotion":         emotion,
        "seed":            seed,
        "budget":          budget,
        "n_top":           n_top,
        "n_evaluated":     int(len(all_rewards)),
        "best_reward":     best_reward,
        "top_probs":       top_probs_list,
        "job_id":          job_id,
        "elapsed_s":       round(elapsed, 2),
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "selected_params": selected_params,
    }

    _common.save_outputs(top_configs, top_rewards, metadata, out_dir)
    print(
        f"[run_baseline] done in {elapsed:.1f}s  "
        f"best_reward={best_reward:.4f}  job_id={job_id}"
    )


if __name__ == "__main__":
    main()
