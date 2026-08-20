"""
baselines/bayesian_method.py

BORFEO-inspired Bayesian Optimization baseline for Furhat facial-expression
generation, adapted from:

  Yang et al., "Optimizing Facial Expressions of an Android Robot Effectively:
  a Bayesian Optimization Approach", IEEE Humanoids 2022.

Key adaptations Nikola -> Furhat
---------------------------------
  Nikola                            Furhat
  ─────────────────────────────────────────────────────────────────────
  35 pneumatic actuators            ARKit-style + Furhat BasicParam controls
  Integer values in [0, 255]        Continuous values in [0.0, 1.0]
  14-D reduced search space         D = len(SELECTED_PARAMS) dimensions
  Py-Feat target-emotion score      DAN target-emotion probability
  Symmetric actuator pairs          Existing MIRROR_MAP bilateral mirroring
  Safety/fragile-pair exclusions    Existing project_theta constraints

GP / acquisition engine
------------------------
Reference [30] in Yang et al. (2022) is fmfn/BayesianOptimization
(github.com/fmfn/BayesianOptimization), published on PyPI as
`bayesian-optimization`. This module calls that package directly
(bayes_opt.BayesianOptimization.maximize) rather than reimplementing the GP /
UCB / acquisition-maximization logic, so GP fitting and acquisition
maximization are performed by the cited historical implementation.

Historical version anchor: bayesian-optimization==1.2.0. Yang et al. do not
report an exact package version, commit, or dependency pins, so 1.2.0 is used
as a stable, reproducible anchor -- NOT a claim about the authors' actual
environment. Rationale:
  - 1.2.0 (PyPI, released 2020-05-16) was still the latest published release
    on July 15, 2022 (the paper's stated deadline); the next release, v1.3.0,
    did not appear until 2022-10-24. An ordinary `pip install
    bayesian-optimization` at any point before the deadline would have
    resolved to 1.2.0.
  - Cross-checked against the upstream GitHub history up to the deadline
    (commit 5bc8ad4 on the fmfn/BayesianOptimization master branch,
    2022-06-26, the latest commit before 2022-07-15): the GP constructor
    (Matern(nu=2.5) kernel, alpha=1e-6, normalize_y=True,
    n_restarts_optimizer=5), the UCB formula (mean + kappa * std), target
    normalization, and the suggest/register/probe/maximize control flow are
    IDENTICAL to the 1.2.0 tag. No relevant core-behavior divergence exists
    between 1.2.0 and the latest pre-deadline commit. (The only pre-deadline
    commits touching bayes_opt/util.py or target_space.py were two forward
    compatibility fixes -- see the module docstring in
    baselines/vendor/bayes_opt_vendored.py -- neither of which changes
    GP/UCB/acquisition semantics; both were
    unreleased on PyPI until v1.3.0, months after the deadline.)

Because 1.2.0 was written against pre-2022 NumPy/SciPy, installing it into
this project's current environment (numpy 1.26.4, scipy 1.15.3,
scikit-learn 1.8.0) fails outright on import (removed `np.float` alias) and
would fail again inside acq_max() (SciPy now requires 1-D `x0` for
`minimize`). Per project policy this is handled by VENDORING the 1.2.0
source with the minimum patches needed to run, rather than installing a
newer bayes_opt release or hand-rolling a replacement GP/UCB. See the module
docstring in baselines/vendor/bayes_opt_vendored.py for the full provenance
and patch log (both patches are dtype/shape compatibility fixes only, and one
of them is textually identical to the upstream project's own pre-deadline
fix for the same line).

UCB sign note
─────────────
Yang et al. (2022) Eq. (2) prints:

    a_UCB(x; beta) = mu(x) - beta * delta(x)   with beta > 0

A subtraction of uncertainty directs search toward exploitation only and is
the appropriate form for MINIMIZATION.  The paper's stated objective (Eq. 1)
is argmax f(x), i.e. MAXIMIZATION of the target-probability score. The
vendored bayes_opt package's UCB (bayes_opt/util.py, UtilityFunction._ucb)
implements the PLUS form:

    a_UCB(x; kappa) = mu(x) + kappa * sigma(x)

which is the standard maximizing UCB and the form fmfn/BayesianOptimization
(reference [30]) has always used. We use the package's own UCB unmodified;
this sign question is not reopened here.

Parameter ordering
-------------------
bayes_opt.BayesianOptimization's internal TargetSpace SORTS parameter names
alphabetically (`self._keys = sorted(pbounds)` in target_space.py) -- verified
by direct inspection and by a toy run with deliberately-unsorted parameter
names (BROW_INNER_UP/JAW_OPEN/MOUTH_LEFT come back in that alphabetical
order, not SELECTED_PARAMS order). The objective callback below therefore
NEVER relies on `list(kwargs.values())` or dict iteration order; it always
reconstructs the vector explicitly via
`[kwargs[name] for name in selected_params]`.

Raw vs. projected configurations
----------------------------------
The package proposes a continuous point in [0, 1]^D (raw_theta). Furhat can
only render the anatomically-feasible, 0.01-grid-quantised projection of that
point (project_theta). The GP therefore models the composite black-box
function raw_theta -> project_theta -> build_full_state -> Furhat render ->
DAN -> target probability, exactly as permitted by the project's design
notes; both raw_theta and its projection are recorded per evaluation (see
bo_extras below) and neither overwrites the other in the logs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ── Repo / baselines on sys.path (so this module can be imported standalone,
#    e.g. from tests, without relying on run_bayesian.py having run first) ──
_BASELINES_DIR = Path(__file__).resolve().parent
_REPO_DIR = _BASELINES_DIR.parent
for _p in (_BASELINES_DIR, _REPO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vendor.bayes_opt_vendored import BayesianOptimization
from vendor.bayes_opt_vendored import (
    VENDORED_PACKAGE_NAME,
    VENDORED_PACKAGE_VERSION,
    VENDORED_SOURCE_URL,
    VENDORED_PATCHES,
)

# Shared baseline infrastructure
from common import (
    build_full_state,
    project_theta,
)

_SUPPORTED_ACQUISITIONS = {"ucb"}
_SUPPORTED_OBJECTIVES = {"target_probability"}

# Sentinel n_iter passed to the vendored optimizer.maximize() in time-limited
# mode (BO_TIME_LIMITED), so BO_TOTAL_EVALUATIONS is never the reason the run
# stops. Not a real cap and allocates no memory (bayes_opt's own TargetSpace
# grows one row per observation regardless of n_iter) -- at a lower bound of
# multiple seconds per expensive Furhat+DAN evaluation, this value could
# never realistically be reached; only the wall-clock check in objective()
# (via _BOTimeLimitReached) is expected to ever stop a time-limited run.
_UNBOUNDED_N_ITER = 10 ** 9


class _BOTimeLimitReached(Exception):
    """
    Internal control-flow signal for time-limited BO (BO_TIME_LIMITED).

    The vendored bayes_opt.BayesianOptimization.maximize() (see
    baselines/vendor/bayes_opt_vendored.py) runs its own while-loop and
    exposes no early-stop hook, so there is no way to ask it to stop from
    outside. This exception is raised from inside our own `objective()`
    callback -- after a candidate evaluation has fully completed and its
    result has already been recorded -- and is caught immediately around the
    `optimizer.maximize(...)` call in `bayesian_bo()` below. It never
    interrupts an in-flight Furhat render or DAN inference, and it never
    touches the vendored package's internals (no signal, no process kill,
    no monkeypatching of its loop) -- it is ordinary Python exception-based
    unwinding of a call stack we already own at every frame it passes
    through (objective -> TargetSpace.probe -> BayesianOptimization.probe ->
    BayesianOptimization.maximize), none of which catches or alters it.
    """


def _parse_time_budget_s(cfg: dict) -> float | None:
    """
    Returns the configured wall-clock BO budget in seconds, or None if
    BO_TIME_LIMITED is false/absent. Raises ValueError if BO_TIME_LIMITED is
    true but the budget is zero or negative.
    """
    if not bool(cfg.get("BO_TIME_LIMITED", False)):
        return None

    hours = float(cfg.get("BO_TIME_LIMIT_HOURS", 0))
    minutes = float(cfg.get("BO_TIME_LIMIT_MINUTES", 0))
    budget_s = hours * 3600.0 + minutes * 60.0

    if budget_s <= 0:
        raise ValueError(
            "BO_TIME_LIMITED is true but BO_TIME_LIMIT_HOURS "
            f"({hours}) + BO_TIME_LIMIT_MINUTES ({minutes}) resolves to a "
            "non-positive budget. Set a positive time budget or set "
            "BO_TIME_LIMITED: false."
        )
    return budget_s


# ── Single-candidate evaluation oracle ───────────────────────────────────────

def evaluate_bo_candidate(
    theta: np.ndarray,
    env,
    img_path: str,
    target_idx: int,
    selected_params: list[str],
    eval_num: int,
) -> tuple[float, np.ndarray, np.ndarray, float, float, float]:
    """
    Project theta into the feasible set, render on Furhat, classify with DAN.

    This is the Furhat objective wrapper: raw_theta -> project_theta ->
    build_full_state -> Furhat rendering -> DAN probabilities -> target
    probability (scalar). It is called once per bayes_opt evaluation (both
    during random initialisation and during BO iterations) and is the only
    place that touches the robot/classifier.

    Every call is an independent absolute expression evaluation:
      neutral base -> set selected params to theta -> mirror bilaterals -> render.
    There is no delta from the previous candidate; state leakage is prevented
    because the full 26-D expression dict is sent on every call (env.step
    replaces the entire expression unconditionally).

    Parameters
    ----------
    theta           : (D,) float -- raw candidate from bayes_opt; any range accepted
    env             : FurhatEnv -- already initialised with correct character/emotion
    img_path        : str -- unique per-evaluation save path for the face crop,
                      or "" to classify without writing an image file
                      (BO_SAVE_EVAL_IMAGES: false, the default -- see bayesian_bo)
    target_idx      : int -- index of target emotion in DAN's 7-class output
    selected_params : list[str] length D -- parameter names in optimisation order
    eval_num        : int -- 1-indexed evaluation counter (passed as step_num)

    Returns
    -------
    target_prob  : float -- DAN probability of target emotion (BO objective f(x))
    theta_proj   : (D,) float32 -- the actually-evaluated projected vector
    probs        : (7,) float32 -- all seven DAN emotion probabilities
    t_perform    : float -- Furhat gesture render time in seconds
    t_classify   : float -- DAN capture + inference time in seconds
    t_total      : float -- total wall-clock time for this evaluation in seconds

    On classifier failure: zeros for probs, 0.0 for target_prob.
    Matches evaluate_config in common.py so failure behaviour is consistent
    across all baselines.
    """
    t_wall = time.time()

    # 1. Project to feasible region: 0.01 grid + [0,1] bounds + anatomical constraints
    theta_proj = project_theta(theta, selected_params)

    # 2. Expand D selected params to full 26-D Furhat expression with bilateral mirroring
    state = build_full_state(theta_proj, selected_params)

    # 3. Render absolute expression, capture face crop, run DAN
    #    annotate_save=False: never draw on the image before or during DAN inference
    _, _, _, probs_raw, t_perform, t_classify = env.step(
        state,
        step_num=eval_num,
        episode_num=0,
        save_path=img_path,
        annotate_save=False,
    )

    t_total = time.time() - t_wall

    # 4. Validate DAN output and extract scalar BO objective
    if probs_raw is None:
        probs = np.zeros(7, dtype=np.float32)
        target_prob = 0.0
        print(f"[bo][WARN] classifier returned None at eval {eval_num} -- recording 0.0")
    else:
        probs = np.asarray(probs_raw, dtype=np.float32).reshape(-1)
        if probs.size != 7 or not np.all(np.isfinite(probs)):
            print(f"[bo][WARN] invalid probs at eval {eval_num}: {probs} -- recording zeros")
            probs = np.zeros(7, dtype=np.float32)
        # Objective: target-emotion probability only (no gap, no weighted-gap reward)
        target_prob = float(probs[target_idx]) if 0 <= target_idx < 7 else 0.0

    return target_prob, theta_proj, probs, float(t_perform), float(t_classify), t_total


# ── Backend provenance (for run metadata) ────────────────────────────────────

def get_backend_metadata() -> dict:
    """
    Runtime provenance for the BO engine actually used this run. Describes
    what actually ran, not what the paper's authors are claimed to have run.
    """
    import platform
    import scipy
    import sklearn

    return {
        "bo_package_name": VENDORED_PACKAGE_NAME,
        "bo_package_version": VENDORED_PACKAGE_VERSION,
        "bo_package_source_url": VENDORED_SOURCE_URL,
        "bo_package_installed_or_vendored": "vendored",
        "bo_package_vendor_path": "baselines/vendor/bayes_opt_vendored.py",
        "bo_package_compat_patches": list(VENDORED_PATCHES),
        "bo_package_version_note": (
            "1.2.0 was the latest PyPI release as of the paper's stated "
            "2022-07-15 deadline (next release v1.3.0 shipped 2022-10-24); "
            "used here as a reproducible historical anchor, not a claim "
            "about the authors' exact environment."
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
    }


# ── Main BO loop ──────────────────────────────────────────────────────────────

def bayesian_bo(
    cfg: dict,
    env,
    eval_img_dir: str,
    selected_params: list[str],
    rng: np.random.Generator,
    target_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """
    BORFEO-inspired sequential GP-UCB optimisation of Furhat facial expressions,
    delegating all GP fitting and acquisition maximisation to the vendored
    bayes_opt.BayesianOptimization (bayesian-optimization==1.2.0).

    Two modes, selected by BO_TIME_LIMITED:

      - BO_TIME_LIMITED false (default): performs exactly BO_TOTAL_EVALUATIONS
        expensive Furhat+DAN evaluations. BO_TOTAL_EVALUATIONS is a hard
        evaluation-count ceiling.
      - BO_TIME_LIMITED true: BO_TOTAL_EVALUATIONS is NOT used as a ceiling
        at all -- the run continues PAST that count, for as long as it takes,
        until the configured wall-clock budget (BO_TIME_LIMIT_HOURS +
        BO_TIME_LIMIT_MINUTES) elapses (see _BOTimeLimitReached). The only
        thing BO_TOTAL_EVALUATIONS affects in this mode is nothing -- it is
        simply ignored as a stopping condition.

    In both modes:
      - BO_INIT_POINTS evaluations : random (uniform-in-bounds) initialisation,
                                     drawn by bayes_opt's own TargetSpace.random_sample
      - remaining evaluations      : sequential BO with UCB acquisition,
                                     via bayes_opt's suggest()/acq_max()

    Every evaluated candidate is an ABSOLUTE final expression (not a delta).
    The GP input is the D-dimensional raw (pre-projection) selected-parameter
    vector; the GP output is the scalar DAN target-emotion probability of the
    PROJECTED, actually-rendered configuration (see module docstring: "Raw vs.
    projected configurations").

    Parameters
    ----------
    cfg             : dict -- loaded from config_bayesian.yaml
    env             : FurhatEnv -- already initialised (correct character/emotion)
    eval_img_dir    : str -- directory for per-evaluation face-crop images
    selected_params : list[str] length D
    rng             : seeded numpy Generator (unused for candidate sampling --
                      sampling is delegated to bayes_opt's own seeded
                      RandomState via BO_SEED -- kept in the signature for
                      interface parity with the other baselines)
    target_idx      : int -- index of target emotion in 7-class DAN output

    Returns
    -------
    all_configs : (N, D) float32 -- PROJECTED param vectors actually rendered
    all_rewards : (N,)   float32 -- target DAN probabilities
    all_probs   : (N, 7) float32 -- all DAN probabilities
    bo_extras   : list[dict] length N -- per-eval metadata

    N == BO_TOTAL_EVALUATIONS when BO_TIME_LIMITED is false. When
    BO_TIME_LIMITED is true, N is simply how many evaluations fit in the
    configured wall-clock budget -- it can be less than, equal to, or
    (typically) greater than BO_TOTAL_EVALUATIONS, which is not a ceiling
    in that mode.
        Keys: eval_idx, is_init, raw_suggested_configuration,
              projected_evaluated_configuration, target_prob, best_so_far,
              gp_mu, gp_sigma, ucb_val,
              t_perform_s, t_classify_s, t_total_s, img_path
    """
    D = len(selected_params)

    # ── Configuration ─────────────────────────────────────────────────────────
    total   = int(cfg.get("BO_TOTAL_EVALUATIONS", 100))
    n_init  = int(cfg.get("BO_INIT_POINTS", 10))
    kappa   = float(cfg.get("BO_KAPPA", 2.576))
    bo_seed = int(cfg.get("BO_SEED", int(rng.integers(0, 2 ** 31))))

    acquisition = str(cfg.get("BO_ACQUISITION", "ucb")).strip().lower()
    objective_name = str(cfg.get("BO_OBJECTIVE", "target_probability")).strip().lower()

    if acquisition not in _SUPPORTED_ACQUISITIONS:
        raise ValueError(
            f"BO_ACQUISITION={acquisition!r} is not supported. Only "
            f"{sorted(_SUPPORTED_ACQUISITIONS)} is implemented, matching "
            "Yang et al. Sec. III-B.2 (UCB is the only acquisition function "
            "the paper describes). EI/POI are not enabled for this baseline."
        )
    if objective_name not in _SUPPORTED_OBJECTIVES:
        raise ValueError(
            f"BO_OBJECTIVE={objective_name!r} is not supported. Only "
            f"{sorted(_SUPPORTED_OBJECTIVES)} is implemented: f(x) = raw DAN "
            "probability of the target emotion, with no gap or weighted-gap "
            "reward variants for this baseline."
        )
    if n_init < 1:
        raise ValueError(f"BO_INIT_POINTS must be >= 1, got {n_init}")

    # Raises ValueError here (not silently) if BO_TIME_LIMITED is true but no
    # positive budget was given.
    time_budget_s = _parse_time_budget_s(cfg)

    if time_budget_s is not None:
        # Time-limited mode: BO_TOTAL_EVALUATIONS is NOT a ceiling here. Pass
        # the vendored package a sentinel n_iter large enough that its own
        # iteration-count loop condition (`while ... or iteration < n_iter`)
        # never becomes the limiting factor -- this is not a real cap (it
        # allocates no memory; bayes_opt's internal arrays grow one row at a
        # time regardless of n_iter), the wall-clock check inside objective()
        # below is the only thing that will ever stop this run.
        n_iter = _UNBOUNDED_N_ITER
    else:
        n_iter = total - n_init   # sequential BO iterations after init
        if n_iter < 0:
            raise ValueError(
                f"BO_TOTAL_EVALUATIONS={total} < BO_INIT_POINTS={n_init}; "
                "BO_INIT_POINTS must be strictly less than BO_TOTAL_EVALUATIONS."
            )

    print(
        f"[bo] D={D}  n_init={n_init}  kappa={kappa}  bo_seed={bo_seed}"
    )
    if time_budget_s is not None:
        print(
            f"[bo] time-limited run: BO_TOTAL_EVALUATIONS ({total}) is NOT "
            f"used as a ceiling in this mode -- will keep running past it "
            f"until {time_budget_s / 3600.0:.3f}h of BO wall-clock time has "
            "elapsed (checked once per completed evaluation)"
        )
    else:
        print(f"[bo] fixed-budget run: total={total}  n_iter={n_iter}")
    print(
        f"[bo] engine: vendored {VENDORED_PACKAGE_NAME}=={VENDORED_PACKAGE_VERSION} "
        f"({VENDORED_SOURCE_URL}) -- GP fit + UCB + acquisition maximisation "
        "performed by the package, not a project reimplementation"
    )
    print(
        "[bo] UCB = mu(x) + kappa * sigma(x)  [package default for acq='ucb'; "
        "maximising form; Yang et al. Eq.2 minus sign appears inconsistent "
        "with the paper's stated maximisation goal -- not reopened here]"
    )
    print(f"[bo] objective: DAN probs[{target_idx}] (target-emotion probability only)")
    print("[bo] no gap reward, no weighted-gap reward, no Py-Feat")

    # Per-evaluation face-crop images are OFF by default: every config + score
    # is always fully logged (bo_extras/CSV/JSON below) regardless of this
    # setting, and best/Top-N images are re-rendered fresh from the logged
    # configs at the end of the run (run_bayesian.py) using logged scores, not
    # a fresh re-classification -- so no per-eval image is ever required for
    # correctness. Saving one for every evaluation is a debugging convenience
    # that is not storage-efficient over a long run; see config_bayesian.yaml.
    save_eval_images = bool(cfg.get("BO_SAVE_EVAL_IMAGES", False))
    print(
        f"[bo] per-eval image saving: "
        + (f"ON (every {int(cfg.get('BO_IMAGE_SAVE_EVERY', 10))}th eval)"
           if save_eval_images else "OFF (best/Top-N are re-rendered at the end)")
    )

    # ── Storage ───────────────────────────────────────────────────────────────
    # Growable (not pre-allocated to `total`): in time-limited mode the
    # eventual evaluation count is not known in advance and is routinely
    # larger than BO_TOTAL_EVALUATIONS. Converted to fixed-size numpy arrays
    # once the run finishes, whichever way it finishes.
    all_configs_list: list[np.ndarray] = []
    all_rewards_list: list[float] = []
    all_probs_list:   list[np.ndarray] = []
    bo_extras: list[dict] = []

    eval_count = [0]           # mutable closure counter (1-indexed once incremented)
    _holder: dict = {"optimizer": None}   # set once the optimizer object exists

    if save_eval_images:
        Path(eval_img_dir).mkdir(parents=True, exist_ok=True)

    # bayes_opt.BayesianOptimization requires pbounds: {name: (low, high)}.
    # Furhat parameter bounds are fixed at [0, 1] for every selected param.
    pbounds = {name: (0.0, 1.0) for name in selected_params}

    # ── Objective wrapper ────────────────────────────────────────────────────
    def objective(**kwargs) -> float:
        """
        bayes_opt calls this as target_func(**params), where `params` keys are
        `selected_params` names but the VALUE ORDER of kwargs.items() is NOT
        guaranteed to match selected_params (TargetSpace sorts keys
        alphabetically internally). The vector is always reconstructed
        explicitly below -- never via list(kwargs.values()).
        """
        raw_theta = np.array(
            [kwargs[name] for name in selected_params], dtype=np.float64
        )

        eval_count[0] += 1
        eval_num = eval_count[0]
        is_init = eval_num <= n_init

        # Empty img_path when save_eval_images is off: env.step's underlying
        # capture call treats a falsy save_path as "classify but don't write
        # a file" (see dan_classifier.py's `if save_path and (step % ...)`),
        # so classification is unaffected either way -- only the image write
        # is skipped.
        img_path = (
            str(Path(eval_img_dir) / f"eval_{eval_num:04d}.jpg")
            if save_eval_images else ""
        )
        target_prob, theta_proj, probs, t_perf, t_clf, t_total = evaluate_bo_candidate(
            raw_theta, env, img_path, target_idx, selected_params, eval_num
        )

        all_configs_list.append(theta_proj.astype(np.float32))
        all_rewards_list.append(float(target_prob))
        all_probs_list.append(probs.astype(np.float32))
        best_so_far = max(all_rewards_list)

        # Read-only introspection of the package's own fitted GP (as it stood
        # when this candidate was chosen) for diagnostic logging only. Does
        # not alter GP/UCB/acquisition behaviour; during init evaluations the
        # GP has not been fit yet (bayes_opt only fits inside suggest(), which
        # is not called while the init queue is being drained), so these stay
        # None for is_init rows -- matching the previous custom implementation.
        optimizer = _holder["optimizer"]
        gp = getattr(optimizer, "_gp", None) if optimizer is not None else None
        gp_mu = gp_sigma = ucb_val = None
        if gp is not None and hasattr(gp, "X_train_"):
            mu_arr, sigma_arr = gp.predict(raw_theta.reshape(1, -1), return_std=True)
            gp_mu = float(mu_arr[0])
            gp_sigma = float(sigma_arr[0])
            ucb_val = gp_mu + kappa * gp_sigma

        bo_extras.append({
            "eval_idx":     eval_num,
            "is_init":      is_init,
            "raw_suggested_configuration":       raw_theta.astype(np.float32).tolist(),
            "projected_evaluated_configuration": theta_proj.tolist(),
            "target_prob":  target_prob,
            "best_so_far":  best_so_far,
            "gp_mu":        gp_mu,
            "gp_sigma":     gp_sigma,
            "ucb_val":      ucb_val,
            "t_perform_s":  t_perf,
            "t_classify_s": t_clf,
            "t_total_s":    t_total,
            "img_path":     img_path,
        })

        eval_of = "?" if time_budget_s is not None else str(total)
        print(
            f"[bo] {'init' if is_init else 'bo  '} eval {eval_num:4d}/{eval_of}  "
            f"p_target={target_prob:.4f}  best={best_so_far:.4f}  t={t_total:.1f}s"
        )

        # ── Time-limit check (once per completed evaluation) ─────────────────
        # This candidate's result is already fully recorded above (all_configs/
        # all_rewards/all_probs/bo_extras), so raising here never loses or
        # truncates a partially-evaluated candidate -- it only ever stops the
        # NEXT evaluation from starting. See _BOTimeLimitReached's docstring
        # for why an exception is the right (and only available) way to
        # interrupt the vendored package's own maximize() loop from here.
        if time_budget_s is not None:
            elapsed_run_s = time.time() - run_start_time
            if elapsed_run_s >= time_budget_s:
                print(
                    f"[bo][time-limit] {elapsed_run_s / 3600.0:.3f}h elapsed >= "
                    f"budget {time_budget_s / 3600.0:.3f}h after eval {eval_num} "
                    "-- stopping (BO_TOTAL_EVALUATIONS was not used as a "
                    "ceiling in this time-limited run)"
                )
                raise _BOTimeLimitReached(eval_num)

        return target_prob

    # ── Delegate entirely to the vendored historical package ────────────────
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=bo_seed,
        verbose=0,   # cosmetic only: the package's own ScreenLogger is silenced
                     # because `objective` already prints one line per evaluation
                     # above; this does not affect GP/UCB/acquisition behaviour.
    )
    _holder["optimizer"] = optimizer

    # Clock starts here: right before the first evaluation, whether it is a
    # random-init draw or a BO iteration. Does not include env/Furhat/DAN
    # setup time, which happens before bayesian_bo() is even called.
    run_start_time = time.time()
    stopped_early_by_time = False
    try:
        optimizer.maximize(
            init_points=n_init,
            n_iter=n_iter,
            acq="ucb",
            kappa=kappa,
        )
    except _BOTimeLimitReached:
        stopped_early_by_time = True

    actual_evals = eval_count[0]
    assert len(bo_extras) == actual_evals == len(all_rewards_list)

    # Build fixed-size arrays from the growable lists now that the run is
    # over; sized to whatever actually ran, not to `total`.
    all_configs = np.array(all_configs_list, dtype=np.float32)
    all_rewards = np.array(all_rewards_list, dtype=np.float32)
    all_probs   = np.array(all_probs_list, dtype=np.float32)

    if not stopped_early_by_time and time_budget_s is None:
        assert actual_evals == total, (
            f"eval_count={actual_evals} != total={total}. The only way this can "
            "happen with bayes_opt 1.2.0 is TargetSpace.probe()'s exact-duplicate "
            "cache short-circuit (identical raw float64 point probed twice), which "
            "returns the cached target without re-calling the objective; this is "
            "vanishingly unlikely with continuous seeded draws but is the one "
            "known way this invariant could be violated."
        )

    best_idx = int(np.argmax(all_rewards))
    if stopped_early_by_time:
        print(
            f"\n[bo] stopped early by time limit -- {actual_evals} evaluations "
            f"completed (BO_TOTAL_EVALUATIONS={total} was not a ceiling)  "
            f"best_p_target={float(all_rewards[best_idx]):.4f}  "
            f"best_eval_idx={best_idx + 1}"
        )
    else:
        print(
            f"\n[bo] completed -- {actual_evals} evaluations  "
            f"best_p_target={float(all_rewards[best_idx]):.4f}  "
            f"best_eval_idx={best_idx + 1}  "
            f"(init: evals 1-{n_init}, BO: evals {n_init + 1}-{actual_evals})"
        )

    return all_configs, all_rewards, all_probs, bo_extras
