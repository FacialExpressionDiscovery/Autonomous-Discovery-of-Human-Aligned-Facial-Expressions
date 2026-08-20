"""
baselines/run_bayesian.py

Entry point for the BORFEO-inspired Bayesian Optimization baseline.
All settings are read from baselines/config_bayesian.yaml.

Usage (from repo root):
    python baselines/run_bayesian.py
    python -m baselines.run_bayesian

Output structure (inside out_dir/<job_id>/):
    bo_eval_log.csv           -- per-evaluation log with BO extras (mu, sigma, ucb, is_init)
    all_configs_log.csv       -- all evaluations sorted best -> worst (same format as other baselines)
    top_<n_top>_log.csv       -- top-N configs (compatible format)
    eval_images/              -- per-evaluation face crop images (eval_0001.jpg ... eval_NNNN.jpg)
    Top_<n_top>/top_01.jpg .. -- annotated images of top-N expressions
    best_expression.jpg       -- annotated image of the single best expression
    best_expression.json      -- best expression: selected vector + full mirrored state + score
    convergence.png           -- best-so-far target probability vs. evaluation index
    configs.npy               -- (n_top, D) float32 top-N parameter vectors
    rewards.npy               -- (n_top,)   float32 top-N target probabilities
    metadata.json             -- complete run metadata (JSON)
    config_bayesian.yaml      -- copy of the run configuration
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import sys
import time
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe on headless machines
import matplotlib.pyplot as plt
import numpy as np
import yaml

# ── Repo root on path ─────────────────────────────────────────────────────────
_REPO     = Path(__file__).resolve().parents[1]
_BASELINES = Path(__file__).resolve().parent
for _p in (_BASELINES, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from policy_gradient.constants import base_expression
from policy_gradient.env import ExpressionUtils, FurhatEnv

import common as _common
from common import BASE_EXPR_KEYS, build_full_state
from run_baseline import _build_env   # reuse identical environment construction
import bayesian_method as _bo

_CANONICAL_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


# ── Config loading ────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    cfg_path = Path(__file__).parent / "config_bayesian.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _write_standard_csv(
    path: Path,
    selected_params: list[str],
    configs: np.ndarray,
    rewards: np.ndarray,
    probs: np.ndarray,
    eval_indices: np.ndarray,
    rank_offset: int = 1,
) -> None:
    """
    Write CSV compatible with the other baselines' all_configs_log.csv format.

    Rows are sorted best -> worst (rank), which loses the original search
    order -- eval_idx restores it: the 1-indexed BO evaluation number
    (init or BO iteration) at which each row's config was actually evaluated
    during the search, so it's always possible to answer "which iteration
    did this top config come from" without cross-referencing bo_eval_log.csv.
    """
    header = ["rank", "eval_idx", "reward"] + list(selected_params) + _CANONICAL_EMOTIONS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, (theta, r, p, ei) in enumerate(zip(configs, rewards, probs, eval_indices)):
            row = (
                [rank_offset + i, int(ei), f"{r:.6f}"]
                + [f"{v:.2f}" for v in theta.tolist()]
                + [f"{v:.6f}" for v in p.tolist()]
            )
            writer.writerow(row)
    print(f"[log] {len(rewards)} rows -> {path}")


def _write_bo_eval_log(
    path: Path,
    selected_params: list[str],
    all_configs: np.ndarray,
    all_rewards: np.ndarray,
    all_probs: np.ndarray,
    bo_extras: list[dict],
) -> None:
    """
    Write the per-evaluation BO log with GP metadata.

    Columns: eval_idx, is_init, target_prob, best_so_far,
             gp_mu, gp_sigma, ucb_val,
             t_perform_s, t_classify_s, t_total_s,
             img_path,
             raw_<param_0..N>       -- bayes_opt's continuous suggestion (pre-projection)
             proj_<param_0..N>      -- actually-rendered projected config (== all_configs row)
             angry..neutral

    raw_ and proj_ columns are kept separate (never overwritten into one
    another) so the continuous BO suggestion and the anatomically-projected,
    actually-evaluated configuration both remain inspectable per evaluation.
    """
    header = (
        ["eval_idx", "is_init", "target_prob", "best_so_far",
         "gp_mu", "gp_sigma", "ucb_val",
         "t_perform_s", "t_classify_s", "t_total_s", "img_path"]
        + [f"raw_{p}" for p in selected_params]
        + [f"proj_{p}" for p in selected_params]
        + _CANONICAL_EMOTIONS
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, (theta, r, p, ex) in enumerate(
            zip(all_configs, all_rewards, all_probs, bo_extras)
        ):
            def _fmt(v):
                return f"{v:.6f}" if v is not None else ""

            row = (
                [ex["eval_idx"], int(ex["is_init"]),
                 f"{ex['target_prob']:.6f}", f"{ex['best_so_far']:.6f}",
                 _fmt(ex["gp_mu"]), _fmt(ex["gp_sigma"]), _fmt(ex["ucb_val"]),
                 f"{ex['t_perform_s']:.3f}", f"{ex['t_classify_s']:.3f}",
                 f"{ex['t_total_s']:.3f}", ex["img_path"]]
                + [f"{v:.4f}" for v in ex["raw_suggested_configuration"]]
                + [f"{v:.2f}" for v in ex["projected_evaluated_configuration"]]
                + [f"{v:.6f}" for v in p.tolist()]
            )
            writer.writerow(row)
    print(f"[log] bo_eval_log -> {path}")


# ── Convergence plot ──────────────────────────────────────────────────────────

def _plot_convergence(
    bo_extras: list[dict],
    n_init: int,
    emotion: str,
    out_path: Path,
) -> None:
    """Plot best-so-far target probability over evaluation index."""
    evals      = [ex["eval_idx"]    for ex in bo_extras]
    best_curve = [ex["best_so_far"] for ex in bo_extras]
    raw_probs  = [ex["target_prob"] for ex in bo_extras]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(evals, raw_probs,  "o", color="steelblue", alpha=0.35, markersize=4,
            label="Evaluation")
    ax.plot(evals, best_curve, "-", color="crimson",   linewidth=2,
            label="Best so far")
    if n_init < len(evals):
        ax.axvline(x=n_init + 0.5, color="grey", linestyle="--", linewidth=1,
                   label=f"Init / BO boundary (n_init={n_init})")
    ax.set_xlabel("Evaluation index")
    ax.set_ylabel(f"DAN p({emotion.title()})")
    ax.set_title(f"BORFEO-inspired BO — {emotion.title()} convergence")
    ax.legend(loc="lower right")
    ax.set_xlim(1, len(evals))
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[plot] convergence -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = _load_cfg()

    method          = "bayesian"
    emotion         = str(cfg["emotion"]).strip().lower()
    character       = str(cfg["character"])
    seed            = int(cfg.get("seed", 42))
    n_top           = int(cfg.get("n_top", 20))
    selected_params = list(cfg.get("selected_params", _common.SELECTED_PARAMS))
    D               = len(selected_params)
    total           = int(cfg.get("BO_TOTAL_EVALUATIONS", 100))
    n_init          = int(cfg.get("BO_INIT_POINTS", 10))

    if emotion not in _CANONICAL_EMOTIONS:
        raise ValueError(f"Unknown emotion '{emotion}'. Expected: {_CANONICAL_EMOTIONS}")
    target_idx = _CANONICAL_EMOTIONS.index(emotion)

    # ── Seeds ─────────────────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    print(
        f"[run_bayesian] emotion={emotion}  character={character}  "
        f"seed={seed}  D={D}  total={total}  n_init={n_init}"
    )

    # ── Output directory ──────────────────────────────────────────────────────
    job_id  = str(uuid.uuid4())[:8]
    out_dir = Path(cfg.get("out_dir", "output_baseline_bo")) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_img_dir = str(out_dir / "eval_images")
    print(f"[run_bayesian] output -> {out_dir}")

    # ── Build Furhat+DAN environment ──────────────────────────────────────────
    env = _build_env(cfg)

    # Per-evaluation face-crop images are OFF by default (BO_SAVE_EVAL_IMAGES:
    # false). Every config + score is always fully logged regardless; images
    # are a debugging convenience only, and best_expression.jpg / Top-N are
    # re-rendered fresh from the logged configs below rather than relying on
    # a saved per-eval file, so no per-eval image is ever required. The actual
    # per-eval save/skip decision is made in bayesian_bo() (bayesian_method.py)
    # via img_path; this only configures the save FREQUENCY for when enabled.
    bo_save_eval_images = bool(cfg.get("BO_SAVE_EVAL_IMAGES", False))
    bo_image_save_every = int(cfg.get("BO_IMAGE_SAVE_EVERY", 10))
    if bo_image_save_every < 1:
        raise ValueError(
            f"BO_IMAGE_SAVE_EVERY must be >= 1, got {bo_image_save_every}"
        )
    if bo_save_eval_images:
        clf = getattr(getattr(env, "expr_utils", None), "emotion_classifier", None)
        if clf is not None and hasattr(clf, "image_save_every"):
            clf.image_save_every = bo_image_save_every

    # ── Run Bayesian Optimisation ─────────────────────────────────────────────
    t0 = time.time()
    all_configs, all_rewards, all_probs, bo_extras = _bo.bayesian_bo(
        cfg=cfg,
        env=env,
        eval_img_dir=eval_img_dir,
        selected_params=selected_params,
        rng=rng,
        target_idx=target_idx,
    )
    elapsed = time.time() - t0

    # ── Per-evaluation BO log (rich CSV with GP metadata) ─────────────────────
    _write_bo_eval_log(
        out_dir / "bo_eval_log.csv",
        selected_params,
        all_configs,
        all_rewards,
        all_probs,
        bo_extras,
    )

    # ── All-configs log (sorted best -> worst, compatible format) ─────────────
    # eval_indices_all[i] is the 1-indexed BO evaluation number at which
    # all_configs[i]/all_rewards[i]/all_probs[i] (bo_extras[i]) was actually
    # evaluated -- sorting by reward below would otherwise lose that mapping.
    eval_indices_all = np.array([ex["eval_idx"] for ex in bo_extras], dtype=int)

    sort_idx        = np.argsort(all_rewards)[::-1]
    sorted_cfgs     = all_configs[sort_idx]
    sorted_rews     = all_rewards[sort_idx]
    sorted_probs    = all_probs[sort_idx]
    sorted_eval_idx = eval_indices_all[sort_idx]

    _write_standard_csv(
        out_dir / "all_configs_log.csv",
        selected_params,
        sorted_cfgs,
        sorted_rews,
        sorted_probs,
        sorted_eval_idx,
    )

    # ── Convergence plot ──────────────────────────────────────────────────────
    _plot_convergence(bo_extras, n_init, emotion, out_dir / "convergence.png")

    # Furhat/DAN capture handles, shared by the best-expression re-render
    # below and the top-N re-render loop further down.
    _dan          = getattr(getattr(env, "expr_utils", None), "emotion_classifier", None)
    _from_monitor = getattr(getattr(env, "expr_utils", None), "capture_from_monitor", False)
    _tmpl         = getattr(getattr(env, "expr_utils", None), "template_path", "furhat_template.png")

    # ── Best expression ───────────────────────────────────────────────────────
    best_flat_idx = int(np.argmax(all_rewards))
    best_theta    = all_configs[best_flat_idx]
    best_prob     = float(all_rewards[best_flat_idx])
    best_probs    = all_probs[best_flat_idx]

    # Full mirrored 62-D state for the best expression
    best_state = build_full_state(best_theta, selected_params)
    best_expr_dict = {k: float(v) for k, v in zip(BASE_EXPR_KEYS, best_state)}

    best_eval_idx_val = int(eval_indices_all[best_flat_idx])
    best_expr_json = {
        "eval_idx":          best_eval_idx_val,
        "target_prob":       best_prob,
        "emotion":           emotion,
        "character":         character,
        "seed":              seed,
        "selected_params":   selected_params,
        "selected_values":   best_theta.tolist(),
        "full_expression":   best_expr_dict,
        "all_probs":         {e: float(best_probs[i]) for i, e in enumerate(_CANONICAL_EMOTIONS)},
    }
    with open(out_dir / "best_expression.json", "w", encoding="utf-8") as f:
        json.dump(best_expr_json, f, indent=2)
    print(f"[best] eval_idx={best_eval_idx_val}  p_{emotion}={best_prob:.4f}")
    print(f"[best] -> {out_dir / 'best_expression.json'}")

    # Re-render the best expression fresh on Furhat rather than relying on a
    # saved per-eval image existing (with BO_IMAGE_SAVE_EVERY > 1, most
    # evaluations -- including possibly this one -- are never saved to disk).
    best_img_dst = str(out_dir / "best_expression.jpg")
    env.expr_utils.perform_expression(best_expr_dict)
    if _dan is not None:
        if _from_monitor:
            _dan.capture_face_image_right_screen(_tmpl, best_img_dst)
        else:
            _dan.capture_face_image(_tmpl, best_img_dst)
        env._annotate_saved_image(
            best_img_dst, best_probs, best_prob, note=f"eval #{best_eval_idx_val}"
        )
        print(f"[best] image -> {best_img_dst}")

    # ── Top-N images (re-render on Furhat for clean captures) ─────────────────
    # Re-renders top-N expressions and captures fresh images, then annotates
    # with the search-phase probabilities (same approach as run_baseline.py).
    n_save      = min(n_top, len(all_rewards))
    top_cfgs    = sorted_cfgs[:n_save]
    top_rews    = sorted_rews[:n_save]
    top_probs   = sorted_probs[:n_save]
    top_eval_idx = sorted_eval_idx[:n_save]   # which BO evaluation each top-N config came from

    top_img_dir = out_dir / f"Top_{n_top}"
    top_img_dir.mkdir(exist_ok=True)
    top_probs_list: list[list[float]] = []

    for rank, (theta, stored_probs, eval_idx_val) in enumerate(
        zip(top_cfgs, top_probs, top_eval_idx), start=1
    ):
        img_path = str(top_img_dir / f"top_{rank:02d}.jpg")
        state = build_full_state(theta, selected_params)
        expr  = {k: float(v) for k, v in zip(BASE_EXPR_KEYS, state)}
        env.expr_utils.perform_expression(expr)

        if _dan is not None:
            if _from_monitor:
                _dan.capture_face_image_right_screen(_tmpl, img_path)
            else:
                _dan.capture_face_image(_tmpl, img_path)

        p_target = float(stored_probs[target_idx])
        env._annotate_saved_image(
            img_path, stored_probs, p_target, note=f"eval #{int(eval_idx_val)}"
        )
        top_probs_list.append(stored_probs.tolist())
        print(
            f"[top] rank {rank:02d} -> {img_path}  p_{emotion}={p_target:.4f}  "
            f"eval_idx={int(eval_idx_val)}"
        )

    _write_standard_csv(
        out_dir / f"top_{n_top}_log.csv",
        selected_params,
        top_cfgs,
        top_rews,
        top_probs,
        top_eval_idx,
    )

    # ── Copy config for reproducibility ──────────────────────────────────────
    shutil.copy2(
        Path(__file__).parent / "config_bayesian.yaml",
        out_dir / "config_bayesian.yaml",
    )

    # ── Time-limited BO bookkeeping ───────────────────────────────────────────
    # n_evaluated_actual is the ACTUAL number of Furhat+DAN evaluations
    # completed. When BO_TIME_LIMITED is false, it equals BO_TOTAL_EVALUATIONS.
    # When BO_TIME_LIMITED is true, BO_TOTAL_EVALUATIONS is NOT a ceiling at
    # all (see bayesian_method.py's bayesian_bo docstring) -- the run continues
    # past it until the wall-clock budget elapses, so n_evaluated_actual is
    # typically LARGER than BO_TOTAL_EVALUATIONS, not smaller. Whether the
    # time limit was the active stopping mode is therefore just whether
    # BO_TIME_LIMITED was configured, not a comparison against `total`.
    time_limited      = bool(cfg.get("BO_TIME_LIMITED", False))
    time_limit_hours  = float(cfg.get("BO_TIME_LIMIT_HOURS", 0))
    time_limit_minutes = float(cfg.get("BO_TIME_LIMIT_MINUTES", 0))
    n_evaluated_actual = int(len(all_rewards))
    stopped_by_time_limit = time_limited

    # ── Metadata JSON ─────────────────────────────────────────────────────────
    metadata = {
        "method":               method,
        "character":            character,
        "emotion":              emotion,
        "seed":                 seed,
        # BO_TOTAL_EVALUATIONS/BO_ITERATIONS are the NOMINAL config values.
        # When BO_TIME_LIMITED is true, BO_TOTAL_EVALUATIONS is not a ceiling
        # (see bayesian_bo's docstring) -- BO_ACTUAL_EVALUATIONS /
        # BO_ACTUAL_BO_ITERATIONS below are what actually ran and are the
        # numbers that matter in that mode.
        "BO_TOTAL_EVALUATIONS": total,
        "BO_INIT_POINTS":       n_init,
        "BO_ITERATIONS":        None if time_limited else (total - n_init),
        "BO_ACTUAL_EVALUATIONS": n_evaluated_actual,
        "BO_ACTUAL_BO_ITERATIONS": max(0, n_evaluated_actual - n_init),
        "BO_TIME_LIMITED":      time_limited,
        "BO_TIME_LIMIT_HOURS":  time_limit_hours,
        "BO_TIME_LIMIT_MINUTES": time_limit_minutes,
        "BO_SAVE_EVAL_IMAGES":  bo_save_eval_images,
        "BO_IMAGE_SAVE_EVERY":  bo_image_save_every,
        "BO_STOPPED_EARLY_BY_TIME_LIMIT": stopped_by_time_limit,
        "BO_KAPPA":             float(cfg.get("BO_KAPPA", 2.576)),
        "BO_SEED":              int(cfg.get("BO_SEED", seed)),
        "BO_ACQUISITION":       str(cfg.get("BO_ACQUISITION", "ucb")),
        "BO_OBJECTIVE":         str(cfg.get("BO_OBJECTIVE", "target_probability")),
        # GP hyperparameters below are the vendored bayes_opt package's own
        # hardcoded internal defaults (bayesian_optimization.py), not
        # independently configurable through the public maximize() API this
        # project calls, and are therefore not overridden here.
        "gp_kernel":            "Matern(nu=2.5)",
        "gp_alpha":             1e-6,
        "gp_normalize_y":       True,
        "gp_n_restarts":        5,
        "acq_max_n_warmup":     10_000,
        "acq_max_n_iter_lbfgs": 10,
        "n_top":                n_save,
        "n_evaluated":          int(len(all_rewards)),
        "best_reward":          best_prob,
        "best_eval_idx":        best_eval_idx_val,
        "top_probs":            top_probs_list,
        "top_eval_indices":     [int(v) for v in top_eval_idx.tolist()],
        "job_id":               job_id,
        "elapsed_s":            round(elapsed, 2),
        "timestamp":            time.strftime("%Y-%m-%dT%H:%M:%S"),
        "selected_params":      selected_params,
        "D":                    D,
        "ucb_formula":          "mu(x) + kappa * sigma(x)  [maximising; paper Eq.2 sign appears inverted; not reopened]",
        "dimensionality_note":  (
            "Runtime D=26 from config selected_params. "
            "The paper's supplementary material states '28-dimensional' but Table 1 lists 26 "
            "independent params. Runtime config is used for execution."
        ),
        **_bo.get_backend_metadata(),
    }

    _common.save_outputs(top_cfgs, top_rews, metadata, out_dir)

    if stopped_by_time_limit:
        print(
            f"[run_bayesian] time-limited run -- {n_evaluated_actual} evaluations "
            f"completed in the {time_limit_hours:.2f}h {time_limit_minutes:.0f}m "
            f"budget (BO_TOTAL_EVALUATIONS={total} was not used as a ceiling)"
        )
    print(
        f"\n[run_bayesian] done in {elapsed:.1f}s  "
        f"best_p_{emotion}={best_prob:.4f}  job_id={job_id}"
    )

    # ── Restore Furhat idle behaviour ─────────────────────────────────────────
    try:
        env.expr_utils.restore_idle_behaviors()
    except Exception as exc:
        print(f"[run_bayesian] restore_idle_behaviors failed: {exc}")


if __name__ == "__main__":
    main()
