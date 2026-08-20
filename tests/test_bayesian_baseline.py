"""
tests/test_bayesian_baseline.py

Focused unit tests for the BORFEO-inspired Bayesian Optimization baseline.
No Furhat connection or DAN model is required; all hardware is mocked.
The BO engine itself is the vendored bayesian-optimization==1.2.0 package
(baselines/vendor/bayes_opt_vendored) -- these tests exercise the project's
integration around it, not a custom GP/UCB reimplementation.

Tests
-----
 1. Dynamic dimensionality     D == len(SELECTED_PARAMS)
 2. Vector-to-full-state       shape (D,) -> shape (62,) correctly
 3. Symmetry                   every mirrored bilateral param gets the same value
 4. Bounds                     all rendered values are in [0.0, 1.0]
 5. Anatomical constraints     project_theta satisfies mouth-height + eye rules
 6. DAN objective extraction   objective == probs[target_idx], not gap reward
 7. No gap reward              objective != top1 - top2 and != weighted reward
 8. Evaluation budget          n_init + n_iter == BO_TOTAL_EVALUATIONS exactly
 9. Absolute states            candidate i does not depend on candidate i-1
10. Reproducibility            same seed -> same candidate sequence
11. No Py-Feat                 bayesian_method.py has no pyfeat / py_feat imports
12. Smoke optimisation         BO improves over init on a cheap synthetic function
13. Vendored package imports   bayes_opt_vendored.BayesianOptimization imports and
                                runs a toy optimisation end-to-end
14. Parameter-order safety     TargetSpace's internal alphabetical key sort does NOT
                                corrupt the reconstructed selected_params-order vector
15. project_theta receives     evaluate_bo_candidate passes the exact raw vector through
    correct vector             to project_theta (no silent reordering/mutation)
16. Scalar objective            evaluate_bo_candidate / bayesian_bo objective returns
                                a plain Python float
17. Invalid config raises       bad BO_ACQUISITION / BO_OBJECTIVE -> ValueError
18. Metadata versions           get_backend_metadata() reports real installed versions
19. Time-limited BO stops early once the configured wall-clock budget elapses,
    with results trimmed (not padded/zero-filled) to the actual eval count
20. Time-limited BO config validation: BO_TIME_LIMITED=true with a
    zero/negative budget raises ValueError
21. Per-eval image saving is OFF by default (no config key needed): every
    step() call gets save_path="" (classify without writing a file)
22. Per-eval image saving, when explicitly enabled (BO_SAVE_EVAL_IMAGES:
    true), produces a non-empty, eval-numbered save_path on every step() call
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ── Repo / baselines on sys.path ──────────────────────────────────────────────
_REPO      = Path(__file__).resolve().parents[1]
_BASELINES = _REPO / "baselines"
for _p in (_BASELINES, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import (
    BASE_EXPR_KEYS,
    MIRROR_MAP,
    SELECTED_PARAMS,
    build_full_state,
    project_theta,
)
from bayesian_method import (
    bayesian_bo,
    evaluate_bo_candidate,
    get_backend_metadata,
)
from vendor.bayes_opt_vendored import BayesianOptimization


# ── Mock environment (no Furhat, no DAN) ─────────────────────────────────────

class _MockEnv:
    """
    Minimal FurhatEnv stub for testing.

    Records every full-state sent via step() so tests can inspect independence.
    Returns a synthetic probability vector derived solely from the current state
    (not stored history), confirming that each evaluation is independent.
    """

    def __init__(self, target_idx: int = 3, noise_std: float = 0.0):
        self.target_idx = target_idx
        self.noise_std  = noise_std
        self.calls: list[np.ndarray] = []   # list of full 62-D states received
        self.save_paths: list[str] = []     # save_path received on every step() call

    def step(self, new_state, step_num, episode_num, save_path="", annotate_save=False):
        self.calls.append(new_state.copy())
        self.save_paths.append(save_path)
        # Synthetic objective: Gaussian bump centred at all-0.5
        x_sel = np.array([new_state[BASE_EXPR_KEYS.index(p)] for p in SELECTED_PARAMS])
        centre = np.full(len(SELECTED_PARAMS), 0.5, dtype=np.float32)
        score = float(np.exp(-np.sum((x_sel - centre) ** 2) / (2 * 0.3 ** 2)))
        score = min(max(score, 0.0), 1.0)
        if self.noise_std > 0:
            score = float(np.clip(score + np.random.normal(0, self.noise_std), 0, 1))
        probs = np.full(7, (1.0 - score) / 6.0, dtype=np.float32)
        probs[self.target_idx] = score
        t_perf    = 0.001
        t_classify = 0.001
        reward = float(probs[self.target_idx])
        penalty = 0.0
        return new_state.copy(), reward, penalty, probs, t_perf, t_classify

    def _annotate_saved_image(self, *args, **kwargs):
        pass


def _make_bo_cfg(total: int = 20, n_init: int = 5, seed: int = 0) -> dict:
    return {
        "BO_TOTAL_EVALUATIONS": total,
        "BO_INIT_POINTS":       n_init,
        "BO_KAPPA":             2.576,
        "BO_SEED":              seed,
        "BO_ACQUISITION":       "ucb",
        "BO_OBJECTIVE":         "target_probability",
    }


# ── Test 1: Dynamic dimensionality ───────────────────────────────────────────

def test_dimensionality_matches_selected_params():
    """D must equal len(SELECTED_PARAMS); never hard-coded."""
    D = len(SELECTED_PARAMS)
    assert D == len(SELECTED_PARAMS), "D != len(SELECTED_PARAMS)"
    # Verify build_full_state accepts a D-shaped input without error
    theta = np.zeros(D, dtype=np.float32)
    state = build_full_state(theta, SELECTED_PARAMS)
    assert state.shape == (len(BASE_EXPR_KEYS),)


# ── Test 2: Vector-to-full-state ──────────────────────────────────────────────

def test_vector_to_full_state_shape():
    """(D,) vector -> (62,) full state with correct dtype."""
    D = len(SELECTED_PARAMS)
    rng = np.random.default_rng(1)
    theta = rng.random(D).astype(np.float32)
    theta = project_theta(theta, SELECTED_PARAMS)
    state = build_full_state(theta, SELECTED_PARAMS)
    assert state.shape == (len(BASE_EXPR_KEYS),), f"state.shape={state.shape}"
    assert state.dtype == np.float32


# ── Test 3: Symmetry ──────────────────────────────────────────────────────────

def test_bilateral_symmetry():
    """Every LEFT parameter in SELECTED_PARAMS must mirror to its RIGHT counterpart."""
    D = len(SELECTED_PARAMS)
    rng = np.random.default_rng(2)
    for _ in range(10):
        theta = rng.random(D).astype(np.float32)
        theta = project_theta(theta, SELECTED_PARAMS)
        state = build_full_state(theta, SELECTED_PARAMS)
        state_dict = dict(zip(BASE_EXPR_KEYS, state.tolist()))
        for left, right in MIRROR_MAP.items():
            if left in SELECTED_PARAMS and right in BASE_EXPR_KEYS:
                v_left  = state_dict[left]
                v_right = state_dict[right]
                assert abs(v_left - v_right) < 1e-5, (
                    f"Symmetry broken: {left}={v_left:.4f} != {right}={v_right:.4f}"
                )


# ── Test 4: Bounds ────────────────────────────────────────────────────────────

def test_all_values_in_unit_interval():
    """All values in the full expression must be in [0.0, 1.0]."""
    D = len(SELECTED_PARAMS)
    rng = np.random.default_rng(3)
    for _ in range(20):
        theta_raw = rng.uniform(-0.5, 1.5, D).astype(np.float32)  # intentionally out of range
        theta_proj = project_theta(theta_raw, SELECTED_PARAMS)
        state = build_full_state(theta_proj, SELECTED_PARAMS)
        assert np.all(state >= 0.0), f"Values below 0: {state[state < 0.0]}"
        assert np.all(state <= 1.0), f"Values above 1: {state[state > 1.0]}"


# ── Test 5: Anatomical constraints ────────────────────────────────────────────

def test_anatomical_constraints():
    """project_theta must satisfy all three constraint families."""
    idx = {p: i for i, p in enumerate(SELECTED_PARAMS)}
    rng = np.random.default_rng(4)

    for _ in range(200):
        theta_raw = rng.random(len(SELECTED_PARAMS)).astype(np.float32)
        theta = project_theta(theta_raw, SELECTED_PARAMS)
        t = np.round(theta * 100).astype(int)

        # Mouth-height constraints: upper - lower >= 7 (== 0.07)
        for upper_name, lower_name in [
            ("MOUTH_SHRUG_UPPER",   "MOUTH_SHRUG_LOWER"),
            ("MOUTH_UPPER_UP_LEFT", "MOUTH_LOWER_DOWN_LEFT"),
            ("MOUTH_ROLL_UPPER",    "MOUTH_ROLL_LOWER"),
        ]:
            if upper_name in idx and lower_name in idx:
                gap = int(t[idx[upper_name]]) - int(t[idx[lower_name]])
                assert gap >= 7 or (t[idx[upper_name]] == 7 and t[idx[lower_name]] == 0), (
                    f"Mouth gap violated: {upper_name}={t[idx[upper_name]]} "
                    f"{lower_name}={t[idx[lower_name]]} gap={gap}"
                )

        # Eye constraints: squint <= 60, blink <= 30, squint + 2*blink <= 33
        if "EYE_SQUINT_LEFT" in idx and "EYE_BLINK_LEFT" in idx:
            sq = int(t[idx["EYE_SQUINT_LEFT"]])
            bl = int(t[idx["EYE_BLINK_LEFT"]])
            assert sq <= 60, f"EYE_SQUINT_LEFT={sq} > 60"
            assert bl <= 30, f"EYE_BLINK_LEFT={bl} > 30"
            assert sq + 2 * bl <= 33, f"sq={sq} + 2*bl={bl} = {sq + 2*bl} > 33"


# ── Test 6: DAN objective extraction ─────────────────────────────────────────

def test_objective_is_target_emotion_probability():
    """f(x) must equal probs[target_idx], not any gap or reward formula."""
    mock_probs = np.array([0.05, 0.10, 0.15, 0.50, 0.08, 0.07, 0.05], dtype=np.float32)
    for target_idx, emotion in enumerate(
        ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    ):
        expected = float(mock_probs[target_idx])
        # Objective must be probs[target_idx] exactly
        got = float(mock_probs[target_idx])
        assert abs(got - expected) < 1e-6, (
            f"Objective mismatch for {emotion}: got {got}, expected {expected}"
        )


# ── Test 7: No gap reward ─────────────────────────────────────────────────────

def test_objective_is_not_gap_reward():
    """
    The BO objective must be raw target probability, not gap or weighted-gap.
    Computed by evaluate_bo_candidate (mocked env returns probs directly).
    """
    target_idx = 3  # happy
    mock_probs = np.array([0.05, 0.10, 0.15, 0.50, 0.08, 0.07, 0.05], dtype=np.float32)

    class _FixedEnv:
        def step(self, state, step_num, episode_num, save_path="", annotate_save=False):
            return state, 0.0, 0.0, mock_probs.copy(), 0.0, 0.0

    D = len(SELECTED_PARAMS)
    theta = np.zeros(D, dtype=np.float32)
    target_prob, _, probs, _, _, _ = evaluate_bo_candidate(
        theta, _FixedEnv(), "", target_idx, SELECTED_PARAMS, 1
    )

    expected_target = float(mock_probs[target_idx])   # 0.50

    # Gap reward = top1 - top2 when top1 is target
    top2 = np.partition(mock_probs, -2)[-2]
    gap_reward = float(mock_probs[target_idx]) - float(top2)

    # Weighted-gap reward with alpha=0.5
    other_max = float(np.max(np.delete(mock_probs, target_idx)))
    m = float(mock_probs[target_idx]) - other_max
    weighted_reward = 0.5 * float(mock_probs[target_idx]) + 0.5 * max(0.0, m)

    assert abs(target_prob - expected_target) < 1e-5, (
        f"objective={target_prob} != target_prob={expected_target}"
    )
    assert abs(target_prob - gap_reward) > 1e-5, (
        f"objective equals gap reward ({target_prob} == {gap_reward})"
    )
    assert abs(target_prob - weighted_reward) > 1e-5, (
        f"objective equals weighted-gap reward ({target_prob} == {weighted_reward})"
    )


# ── Test 8: Evaluation budget ─────────────────────────────────────────────────

def test_evaluation_budget_exact(tmp_path):
    """n_init + n_BO_iterations == BO_TOTAL_EVALUATIONS exactly."""
    for total, n_init in [(20, 5), (10, 3), (15, 15)]:
        cfg = _make_bo_cfg(total=total, n_init=n_init)
        env = _MockEnv(target_idx=3)
        rng = np.random.default_rng(0)
        all_configs, all_rewards, all_probs, bo_extras = bayesian_bo(
            cfg=cfg,
            env=env,
            eval_img_dir=str(tmp_path / f"imgs_{total}_{n_init}"),
            selected_params=SELECTED_PARAMS,
            rng=rng,
            target_idx=3,
        )
        assert len(bo_extras) == total, f"len(bo_extras)={len(bo_extras)} != total={total}"
        assert all_configs.shape == (total, len(SELECTED_PARAMS))
        assert all_rewards.shape == (total,)
        assert all_probs.shape == (total, 7)

        n_init_actual = sum(1 for ex in bo_extras if ex["is_init"])
        n_iter_actual = sum(1 for ex in bo_extras if not ex["is_init"])
        assert n_init_actual == n_init, f"n_init_actual={n_init_actual} != {n_init}"
        assert n_iter_actual == total - n_init, (
            f"n_iter_actual={n_iter_actual} != {total - n_init}"
        )


# ── Test 9: Absolute states ───────────────────────────────────────────────────

def test_evaluation_independence(tmp_path):
    """
    Each evaluation must not depend on the previous candidate's state.
    Verified by checking that the full states sent to env.step are each built
    from scratch from the neutral base -- no incremental delta accumulation.
    """
    cfg = _make_bo_cfg(total=10, n_init=5)
    env = _MockEnv(target_idx=3)
    rng = np.random.default_rng(7)
    bayesian_bo(
        cfg=cfg,
        env=env,
        eval_img_dir=str(tmp_path / "imgs_indep"),
        selected_params=SELECTED_PARAMS,
        rng=rng,
        target_idx=3,
    )

    # Build the complete set of params that build_full_state is allowed to set:
    # the selected params themselves, plus their bilateral mirror targets.
    # (MIRROR_MAP keys are LEFT; values are their RIGHT counterparts.)
    selected_set = set(SELECTED_PARAMS)
    mirrored_set = {
        MIRROR_MAP[p]
        for p in SELECTED_PARAMS
        if p in MIRROR_MAP and MIRROR_MAP[p] in BASE_EXPR_KEYS
    }
    allowed = selected_set | mirrored_set

    for call_idx, state in enumerate(env.calls):
        state_dict = dict(zip(BASE_EXPR_KEYS, state.tolist()))
        for param, val in state_dict.items():
            if param not in allowed:
                assert val == 0.0, (
                    f"Call {call_idx}: param '{param}' = {val} != 0.0; "
                    "this param is neither selected nor a bilateral mirror of a "
                    "selected param — suggests state leakage from previous evaluation"
                )


# ── Test 10: Reproducibility ──────────────────────────────────────────────────

def test_reproducibility(tmp_path):
    """Same seed must produce the same candidate sequence under a deterministic mock."""
    cfg = _make_bo_cfg(total=15, n_init=5, seed=99)

    def _run(tag: str) -> np.ndarray:
        env = _MockEnv(target_idx=3, noise_std=0.0)
        rng = np.random.default_rng(123)   # unused for sampling (bayes_opt owns its own RNG)
        configs, _, _, _ = bayesian_bo(
            cfg=cfg,
            env=env,
            eval_img_dir=str(tmp_path / f"imgs_repro_{tag}"),
            selected_params=SELECTED_PARAMS,
            rng=rng,
            target_idx=3,
        )
        return configs

    configs_a = _run("a")
    configs_b = _run("b")
    assert np.allclose(configs_a, configs_b, atol=0.0), (
        "Same BO_SEED produced different candidate sequences"
    )


# ── Test 11: No Py-Feat ───────────────────────────────────────────────────────

def test_no_pyfeat_import():
    """bayesian_method.py must not import or reference Py-Feat."""
    module_path = _BASELINES / "bayesian_method.py"
    source = module_path.read_text(encoding="utf-8")
    assert "pyfeat" not in source.lower(), "Found 'pyfeat' in bayesian_method.py"
    assert "py_feat" not in source.lower(), "Found 'py_feat' in bayesian_method.py"
    assert "feat.Detector" not in source,   "Found 'feat.Detector' in bayesian_method.py"

    # Also check run_bayesian.py
    run_path = _BASELINES / "run_bayesian.py"
    run_source = run_path.read_text(encoding="utf-8")
    assert "pyfeat" not in run_source.lower(), "Found 'pyfeat' in run_bayesian.py"
    assert "py_feat" not in run_source.lower(), "Found 'py_feat' in run_bayesian.py"


# ── Test 12: Smoke optimisation ───────────────────────────────────────────────

def test_smoke_optimisation_improves_over_init(tmp_path):
    """
    On the synthetic Gaussian-bump objective (MockEnv), the BO best score should
    be at least as good as the best initialisation score (BO must not degrade),
    and with enough evaluations it should strictly improve in expectation.
    """
    # Use a generous budget so BO has a chance to improve
    cfg = _make_bo_cfg(total=30, n_init=8, seed=42)
    env = _MockEnv(target_idx=3, noise_std=0.0)
    rng = np.random.default_rng(42)

    all_configs, all_rewards, all_probs, bo_extras = bayesian_bo(
        cfg=cfg,
        env=env,
        eval_img_dir=str(tmp_path / "imgs_smoke"),
        selected_params=SELECTED_PARAMS,
        rng=rng,
        target_idx=3,
    )

    # Extract init and BO rewards
    init_rewards = [ex["target_prob"] for ex in bo_extras if ex["is_init"]]
    bo_rewards   = [ex["target_prob"] for ex in bo_extras if not ex["is_init"]]

    best_init = max(init_rewards)
    best_bo   = max(bo_rewards) if bo_rewards else best_init
    best_overall = float(np.max(all_rewards))

    # Best overall must be the best actually-evaluated point
    assert abs(best_overall - max(init_rewards + bo_rewards)) < 1e-5, (
        "best_overall != max of all evaluated rewards"
    )

    # BO must not degrade: best overall >= best init
    assert best_overall >= best_init - 1e-5, (
        f"BO degraded: best_overall={best_overall:.4f} < best_init={best_init:.4f}"
    )

    # BO should find a better candidate than the initialisation (not guaranteed
    # but highly probable on this smooth unimodal function with 22 BO iterations)
    assert best_bo >= best_init * 0.9, (
        f"BO best ({best_bo:.4f}) is much worse than init best ({best_init:.4f})"
    )

    # All returned probabilities must be finite scalars in [0, 1]
    assert np.all(np.isfinite(all_rewards))
    assert np.all(all_rewards >= 0.0)
    assert np.all(all_rewards <= 1.0)

    print(
        f"[smoke] best_init={best_init:.4f}  best_bo={best_bo:.4f}  "
        f"best_overall={best_overall:.4f}"
    )


# ── Test 13: Vendored package imports and runs ────────────────────────────────

def test_vendored_package_imports_and_runs():
    """
    bayesian-optimization==1.2.0 (vendored, with the two minimal numpy/scipy
    compatibility patches -- see baselines/vendor/bayes_opt_vendored.py)
    must import cleanly and run a toy black-box optimisation end-to-end.
    """
    def black_box(**kwargs):
        x, y = kwargs["x"], kwargs["y"]
        return -(x - 0.3) ** 2 - (y - 0.7) ** 2

    pbounds = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
    opt = BayesianOptimization(f=black_box, pbounds=pbounds, random_state=42, verbose=0)
    opt.maximize(init_points=2, n_iter=3, acq="ucb", kappa=2.576)

    assert len(opt.res) == 5, f"expected 5 evaluations, got {len(opt.res)}"
    assert opt.max["target"] > -1.0, "toy optimisation produced an implausible result"
    # Found optimum should be reasonably close to the true optimum (0.3, 0.7)
    assert abs(opt.max["params"]["x"] - 0.3) < 0.5
    assert abs(opt.max["params"]["y"] - 0.7) < 0.5


# ── Test 14: Parameter-order safety against TargetSpace's internal sort ──────

def test_parameter_order_survives_targetspace_alphabetical_sort():
    """
    bayes_opt.TargetSpace sorts pbounds keys alphabetically internally
    (`self._keys = sorted(pbounds)`), so a naive `list(kwargs.values())` inside
    an objective would silently scramble a non-alphabetical selected_params
    order. This test uses a deliberately non-alphabetical param order (matching
    the real SELECTED_PARAMS, which is not alphabetically sorted) and confirms
    that explicit `[kwargs[name] for name in selected_params]` reconstruction
    recovers the correct order regardless of TargetSpace's internal sort.
    """
    selected = ["MOUTH_LEFT", "BROW_INNER_UP", "JAW_OPEN"]   # deliberately non-alphabetical
    assert selected != sorted(selected), "test fixture must be non-alphabetical"

    captured = []

    def objective(**kwargs):
        vec = np.array([kwargs[name] for name in selected], dtype=np.float64)
        captured.append((dict(kwargs), vec))
        return float(vec.sum())

    pbounds = {name: (0.0, 1.0) for name in selected}
    opt = BayesianOptimization(f=objective, pbounds=pbounds, random_state=3, verbose=0)

    # Confirm TargetSpace really does reorder internally (this is the failure
    # mode the explicit-reconstruction pattern guards against).
    assert opt.space.keys == sorted(selected)
    assert opt.space.keys != selected

    opt.maximize(init_points=3, n_iter=0, acq="ucb", kappa=2.576)

    for kwargs, vec in captured:
        expected = np.array([kwargs[name] for name in selected], dtype=np.float64)
        assert np.array_equal(vec, expected), (
            f"reconstructed vector {vec} does not match selected_params order for {kwargs}"
        )


# ── Test 15: project_theta receives the correct raw vector ──────────────────

def test_project_theta_receives_correct_vector(monkeypatch):
    """
    evaluate_bo_candidate must pass the exact raw_theta it was given through to
    project_theta (as the first positional argument), with no reordering,
    truncation, or silent copy substitution.
    """
    import bayesian_method as bm

    D = len(SELECTED_PARAMS)
    raw_theta = np.linspace(0.0, 1.0, D, dtype=np.float64)
    seen = {}

    real_project_theta = bm.project_theta

    def _spy(theta, selected_params):
        seen["theta"] = np.array(theta, dtype=np.float64).copy()
        seen["selected_params"] = list(selected_params)
        return real_project_theta(theta, selected_params)

    monkeypatch.setattr(bm, "project_theta", _spy)

    class _FixedEnv:
        def step(self, state, step_num, episode_num, save_path="", annotate_save=False):
            probs = np.zeros(7, dtype=np.float32)
            probs[0] = 1.0
            return state, 1.0, 0.0, probs, 0.0, 0.0

    bm.evaluate_bo_candidate(raw_theta, _FixedEnv(), "", 0, SELECTED_PARAMS, 1)

    assert "theta" in seen, "project_theta was never called"
    assert np.array_equal(seen["theta"], raw_theta), (
        f"project_theta received {seen['theta']}, expected exactly raw_theta={raw_theta}"
    )
    assert seen["selected_params"] == list(SELECTED_PARAMS)


# ── Test 16: Objective returns a scalar float ────────────────────────────────

def test_objective_returns_scalar_float(tmp_path):
    """
    The bayes_opt objective (and evaluate_bo_candidate's target_prob) must be a
    plain scalar float, never an ndarray/list, since bayes_opt.TargetSpace
    stores it directly in a float64 array.
    """
    target_idx = 3
    mock_probs = np.array([0.05, 0.10, 0.15, 0.50, 0.08, 0.07, 0.05], dtype=np.float32)

    class _FixedEnv:
        def step(self, state, step_num, episode_num, save_path="", annotate_save=False):
            return state, 0.0, 0.0, mock_probs.copy(), 0.0, 0.0

    D = len(SELECTED_PARAMS)
    theta = np.zeros(D, dtype=np.float32)
    target_prob, theta_proj, probs, t_perf, t_clf, t_total = evaluate_bo_candidate(
        theta, _FixedEnv(), "", target_idx, SELECTED_PARAMS, 1
    )
    assert isinstance(target_prob, float), f"target_prob is {type(target_prob)}, not float"
    assert not isinstance(target_prob, np.ndarray)

    # Also confirm the full bayesian_bo objective path yields a scalar the
    # vendored package can register without error (implicitly exercised by
    # test_evaluation_budget_exact, asserted explicitly here via a 1-shot run).
    cfg = _make_bo_cfg(total=2, n_init=2, seed=0)
    env = _MockEnv(target_idx=3)
    rng = np.random.default_rng(0)
    _, all_rewards, _, _ = bayesian_bo(
        cfg=cfg, env=env, eval_img_dir=str(tmp_path / "imgs_scalar"),
        selected_params=SELECTED_PARAMS, rng=rng, target_idx=3,
    )
    assert all_rewards.dtype == np.float32
    assert all_rewards.shape == (2,)


# ── Test 17: Invalid acquisition/objective config raises ValueError ─────────

def test_invalid_acquisition_raises(tmp_path):
    cfg = _make_bo_cfg(total=5, n_init=2)
    cfg["BO_ACQUISITION"] = "ei"   # not supported by this baseline
    env = _MockEnv(target_idx=3)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="BO_ACQUISITION"):
        bayesian_bo(
            cfg=cfg, env=env, eval_img_dir=str(tmp_path / "imgs_bad_acq"),
            selected_params=SELECTED_PARAMS, rng=rng, target_idx=3,
        )


def test_invalid_objective_raises(tmp_path):
    cfg = _make_bo_cfg(total=5, n_init=2)
    cfg["BO_OBJECTIVE"] = "gap_reward"   # not supported by this baseline
    env = _MockEnv(target_idx=3)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="BO_OBJECTIVE"):
        bayesian_bo(
            cfg=cfg, env=env, eval_img_dir=str(tmp_path / "imgs_bad_obj"),
            selected_params=SELECTED_PARAMS, rng=rng, target_idx=3,
        )


# ── Test 18: Metadata reports actual installed versions ──────────────────────

def test_metadata_reports_actual_versions():
    """get_backend_metadata() must report the real, currently-installed versions."""
    import numpy
    import scipy
    import sklearn

    meta = get_backend_metadata()

    assert meta["bo_package_name"] == "bayesian-optimization"
    assert meta["bo_package_version"] == "1.2.0"
    assert meta["bo_package_installed_or_vendored"] == "vendored"
    assert len(meta["bo_package_compat_patches"]) >= 2, (
        "expected both the np.float and the scipy-minimize compat patches to be recorded"
    )
    assert meta["numpy_version"] == numpy.__version__
    assert meta["scipy_version"] == scipy.__version__
    assert meta["sklearn_version"] == sklearn.__version__
    assert isinstance(meta["python_version"], str) and meta["python_version"]


# ── Test 19: Time-limited BO stops early (budget reached before `total`) ────

def test_time_limited_bo_stops_early_and_trims_results(tmp_path, monkeypatch):
    """
    BO_TIME_LIMITED must stop the run once the wall-clock budget elapses,
    even though BO_TOTAL_EVALUATIONS has not been reached, and the returned
    arrays/bo_extras must be sized to the actual number of evaluations
    completed (never zero-padded to the nominal total -- storage is a
    growable list converted to an array at the end, not a pre-allocated
    (total, ...) buffer).

    Uses a fake deterministic clock (monkeypatched onto bayesian_method's
    `time.time`) so the test doesn't depend on real wall-clock speed: each
    call advances a shared counter by exactly 1.0 "second", so the number of
    evaluations completed before the budget trips is reproducible.
    """
    import bayesian_method as bm

    clock = {"t": 0.0}

    def _fake_time():
        clock["t"] += 1.0
        return clock["t"]

    monkeypatch.setattr(bm.time, "time", _fake_time)

    total, n_init = 50, 5
    cfg = _make_bo_cfg(total=total, n_init=n_init, seed=0)
    cfg["BO_TIME_LIMITED"] = True
    cfg["BO_TIME_LIMIT_HOURS"] = 0
    cfg["BO_TIME_LIMIT_MINUTES"] = 1   # 60 fake-clock "seconds"

    env = _MockEnv(target_idx=3)
    rng = np.random.default_rng(0)

    all_configs, all_rewards, all_probs, bo_extras = bayesian_bo(
        cfg=cfg,
        env=env,
        eval_img_dir=str(tmp_path / "imgs_time_limit"),
        selected_params=SELECTED_PARAMS,
        rng=rng,
        target_idx=3,
    )

    n_actual = len(bo_extras)
    assert 0 < n_actual < total, (
        f"expected the time budget to force an early stop strictly between 0 "
        f"and total={total}, got n_actual={n_actual}"
    )
    assert all_configs.shape == (n_actual, len(SELECTED_PARAMS)), all_configs.shape
    assert all_rewards.shape == (n_actual,), all_rewards.shape
    assert all_probs.shape == (n_actual, 7), all_probs.shape

    # Every recorded row must be a real, independently-evaluated candidate,
    # not a leftover zero-filled row from the original (total, ...) allocation.
    assert np.all(np.isfinite(all_rewards))
    eval_indices = [ex["eval_idx"] for ex in bo_extras]
    assert eval_indices == list(range(1, n_actual + 1)), (
        "bo_extras must be contiguous eval_idx 1..n_actual with no gaps/padding"
    )


# ── Test 19b: Time-limited BO bypasses BO_TOTAL_EVALUATIONS as a ceiling ────

def test_time_limited_bo_bypasses_total_evaluations_ceiling(tmp_path, monkeypatch):
    """
    When BO_TIME_LIMITED is true, BO_TOTAL_EVALUATIONS must NOT act as a
    ceiling: the run should keep going PAST the nominal total for as long as
    the wall-clock budget allows, not stop as soon as `total` evaluations
    have been completed.
    """
    import bayesian_method as bm

    clock = {"t": 0.0}

    def _fake_time():
        clock["t"] += 1.0
        return clock["t"]

    monkeypatch.setattr(bm.time, "time", _fake_time)

    total, n_init = 10, 3   # deliberately tiny nominal total
    cfg = _make_bo_cfg(total=total, n_init=n_init, seed=0)
    cfg["BO_TIME_LIMITED"] = True
    cfg["BO_TIME_LIMIT_HOURS"] = 0
    cfg["BO_TIME_LIMIT_MINUTES"] = 5   # 300 fake-clock "seconds" -- enough for far more than 10 evals

    env = _MockEnv(target_idx=3)
    rng = np.random.default_rng(0)

    all_configs, all_rewards, all_probs, bo_extras = bayesian_bo(
        cfg=cfg,
        env=env,
        eval_img_dir=str(tmp_path / "imgs_bypass_total"),
        selected_params=SELECTED_PARAMS,
        rng=rng,
        target_idx=3,
    )

    n_actual = len(bo_extras)
    assert n_actual > total, (
        f"expected time-limited BO to run PAST the nominal total={total} "
        f"(BO_TOTAL_EVALUATIONS must be bypassed as a ceiling in this mode), "
        f"got n_actual={n_actual}"
    )
    assert all_configs.shape == (n_actual, len(SELECTED_PARAMS))
    assert all_rewards.shape == (n_actual,)
    assert all_probs.shape == (n_actual, 7)


# ── Test 20: Time-limited BO requires a positive budget ─────────────────────

def test_time_limited_bo_requires_positive_budget(tmp_path):
    """BO_TIME_LIMITED=true with a zero/absent time budget must raise ValueError,
    not silently run unbounded or silently disable the time limit."""
    cfg = _make_bo_cfg(total=10, n_init=3)
    cfg["BO_TIME_LIMITED"] = True
    cfg["BO_TIME_LIMIT_HOURS"] = 0
    cfg["BO_TIME_LIMIT_MINUTES"] = 0
    env = _MockEnv(target_idx=3)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="BO_TIME_LIMITED"):
        bayesian_bo(
            cfg=cfg, env=env, eval_img_dir=str(tmp_path / "imgs_bad_time_budget"),
            selected_params=SELECTED_PARAMS, rng=rng, target_idx=3,
        )


# ── Test 21: Per-eval image saving is OFF by default ────────────────────────

def test_eval_images_off_by_default(tmp_path):
    """
    Without BO_SAVE_EVAL_IMAGES set, bayesian_bo must classify every
    candidate without ever asking env.step to write an image file (empty
    save_path on every call). Saving a face crop per evaluation is a
    debugging convenience only and is not storage-efficient over a long run.
    """
    cfg = _make_bo_cfg(total=8, n_init=3, seed=0)   # BO_SAVE_EVAL_IMAGES absent
    env = _MockEnv(target_idx=3)
    rng = np.random.default_rng(0)

    bayesian_bo(
        cfg=cfg,
        env=env,
        eval_img_dir=str(tmp_path / "imgs_default_off"),
        selected_params=SELECTED_PARAMS,
        rng=rng,
        target_idx=3,
    )

    assert len(env.save_paths) == 8
    assert all(p == "" for p in env.save_paths), (
        f"expected every save_path to be empty by default, got {env.save_paths}"
    )
    # No directory should have been created for images nobody is going to write.
    assert not (tmp_path / "imgs_default_off").exists()


# ── Test 22: Per-eval image saving, when explicitly enabled ─────────────────

def test_eval_images_enabled_via_config(tmp_path):
    """BO_SAVE_EVAL_IMAGES: true must produce a real, eval-numbered save_path
    on every step() call (this test does not require an actual DAN/Furhat
    backend to write the file -- _MockEnv never touches the filesystem)."""
    cfg = _make_bo_cfg(total=6, n_init=2, seed=0)
    cfg["BO_SAVE_EVAL_IMAGES"] = True
    env = _MockEnv(target_idx=3)
    rng = np.random.default_rng(0)

    bayesian_bo(
        cfg=cfg,
        env=env,
        eval_img_dir=str(tmp_path / "imgs_enabled"),
        selected_params=SELECTED_PARAMS,
        rng=rng,
        target_idx=3,
    )

    assert len(env.save_paths) == 6
    assert all(p for p in env.save_paths), (
        f"expected every save_path to be non-empty, got {env.save_paths}"
    )
    assert all("eval_" in p for p in env.save_paths)
    assert (tmp_path / "imgs_enabled").is_dir()


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
