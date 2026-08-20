from __future__ import annotations

import copy
import random
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .classifier_loader import _get_classifier_labels
from .emotions import _CANONICAL_EMOTION_LABELS, _resolve_emotion_index_from_name, _slugify_emotion_name
from .env import ExpressionUtils, FurhatEnv
from .logging_utils import LoggerManager
from .models import FactorizedDiscretePolicy, ValueNet
from .paths import PG_DIR, resolve_pg_path
from .mixins import (
    ActionPhasesMixin,
    CoreMixin,
    EvaluationMixin,
    PlottingMixin,
    SamplingMixin,
    TrainingMixin,
)


class PGTrainer(
    CoreMixin,
    SamplingMixin,
    ActionPhasesMixin,
    EvaluationMixin,
    PlottingMixin,
    TrainingMixin,
):
    def __init__(self, config: dict):
        random.seed(config.get("SEED", 42))
        np.random.seed(config.get("SEED", 42))
        torch.manual_seed(config.get("SEED", 42))
        torch.cuda.manual_seed_all(config.get("SEED", 42))

        self.config = config
        labels = _get_classifier_labels(
            str(config.get("CLASSIFIER_TYPE", "dan")),
            classifier_module=config.get("CLASSIFIER_MODULE"),
            classifier_class=config.get("CLASSIFIER_CLASS"),
        )

        raw_designated = config.get(
            "DESIGNATED_EMOTION",
            config.get("DESIGNATED_EMOTION_INDEX", config.get("HAPPY_INDEX", config.get("DESIGNATED_EMOTION_NAME", 3))),
        )
        if isinstance(raw_designated, str):
            self.designated_emotion_index = _resolve_emotion_index_from_name(raw_designated, labels=labels)
        else:
            self.designated_emotion_index = int(raw_designated)

        if labels and 0 <= self.designated_emotion_index < len(labels):
            designated_name = labels[self.designated_emotion_index]
        elif 0 <= self.designated_emotion_index < len(_CANONICAL_EMOTION_LABELS):
            designated_name = _CANONICAL_EMOTION_LABELS[self.designated_emotion_index].title()
        else:
            designated_name = f"Emotion{self.designated_emotion_index}"

        self.designated_emotion_name = str(designated_name)
        self.designated_emotion_slug = _slugify_emotion_name(self.designated_emotion_name)
        config["DESIGNATED_EMOTION"] = int(self.designated_emotion_index)
        config["DESIGNATED_EMOTION_INDEX"] = int(self.designated_emotion_index)
        config["DESIGNATED_EMOTION_NAME"] = self.designated_emotion_name
        config["DESIGNATED_EMOTION_SLUG"] = self.designated_emotion_slug
        config.setdefault("HAPPY_INDEX", int(self.designated_emotion_index))

        # IO
        self.job_id = str(uuid.uuid4())[:8]
        self.output_dir = (PG_DIR / "output_REINFORCE" / self.job_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = LoggerManager(self.output_dir, self.job_id, config)
        self.logger.log_config()
        self._snapshot_self_script()

        # Device
        self.device = torch.device(self.config["DEVICE"] if torch.cuda.is_available() else "cpu")
        self.logger.log_message(f"Using device: {self.device}")

        # DAN classifier init kwargs
        classifier_init_kwargs = {
            "dan_preset": config.get("DAN_PRESET"),
            "detect_face": config.get("DAN_DETECT_FACE", True),
            "if_no_face": config.get("DAN_IF_NO_FACE", "fail"),
        }

        expr_utils = ExpressionUtils(
            config["CLASSIFIER_TYPE"],
            config["FURHAT_IP"],
            config["FER_MODEL_PATH"],
            template_path=config.get("TEMPLATE_PATH") or (
                "furhat_template_maurice.png"
                if str(config.get("FURHAT_TEXTURE") or "").strip().lower() == "maurice"
                else "furhat_template.png"
            ),
            capture_from_monitor=config["CAPTURE_FROM_MONITOR"],
            furhat_texture=config.get("FURHAT_TEXTURE"),
            furhat_texture_settle_s=float(config.get("FURHAT_TEXTURE_SETTLE_S", 0.35)),
            furhat_texture_strict=bool(config.get("FURHAT_TEXTURE_STRICT", True)),
            classifier_module=config.get("CLASSIFIER_MODULE"),
            classifier_class=config.get("CLASSIFIER_CLASS"),
            classifier_init_kwargs=classifier_init_kwargs,
            disable_microexpressions=bool(config.get("FURHAT_DISABLE_MICROEXPRESSIONS", False)),
            disable_blinking=bool(config.get("FURHAT_DISABLE_BLINKING", False)),
            disable_head_sway=bool(config.get("FURHAT_DISABLE_HEAD_SWAY", False)),
            reset_neutral_after_texture=bool(config.get("FURHAT_RESET_NEUTRAL_AFTER_TEXTURE", True)),
            set_head_pose=bool(config.get("FURHAT_SET_HEAD_POSE", False)),
            head_yaw=float(config.get("FURHAT_HEAD_YAW", 0.0)),
            head_pitch=float(config.get("FURHAT_HEAD_PITCH", 0.0)),
            head_roll=float(config.get("FURHAT_HEAD_ROLL", 0.0)),
        )
        self.env = FurhatEnv(config, expr_utils)

        # Selected params
        self.selected_params = config["SELECTED_PARAMS"]
        self.param_indices = [self.env.expression_keys.index(p) for p in self.selected_params]
        self.obs_dim = len(self.param_indices)
        self.name_to_idx = {name: i for i, name in enumerate(self.env.expression_keys)}

        # REINFORCE hyperparams
        self.max_steps = config.get("MAX_STEPS_PER_EPISODE", 6)
        self.episodes = config.get("EPISODES", 100)
        self.gamma = float(config.get("GAMMA", 0.9))
        self.lr = float(config.get("LR", 1e-3))

        # Discrete delta values per parameter: [-0.25, 0.25] step 0.01 (51 values)
        self.K = 51
        self.delta_values = torch.linspace(-0.25, 0.25, steps=self.K, device=self.device)

        # Policy
        self.policy = FactorizedDiscretePolicy(self.obs_dim, n_actions_per_dim=self.K, hidden=int(config.get("HIDDEN", 128))).to(self.device)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # Actor-Critic
        self.use_actor_critic = bool(config.get("USE_ACTOR_CRITIC", False))
        if self.use_actor_critic:
            self.value_net = ValueNet(self.obs_dim, hidden=int(config.get("HIDDEN", 128))).to(self.device)
            critic_lr = float(config.get("CRITIC_LR", self.lr))
            self.critic_optim = torch.optim.Adam(self.value_net.parameters(), lr=critic_lr)
            self.critic_loss_coef = float(config.get("CRITIC_LOSS_COEF", 0.5))

            # Critic replay buffer
            self.use_critic_replay = bool(config.get("USE_CRITIC_REPLAY", False))
            self.critic_buffer_capacity = int(config.get("CRITIC_BUFFER_CAPACITY", 50000))
            self.critic_batch_size = int(config.get("CRITIC_BATCH_SIZE", 256))
            self.critic_updates_per_ep = int(config.get("CRITIC_UPDATES_PER_EP", 4))

            self.critic_buffer_states = []
            self.critic_buffer_targets = []

            self.critic_use_full_traj = bool(config.get("CRITIC_USE_FULL_TRAJ", False))
            self.critic_target_mode = str(config.get("CRITIC_TARGET_MODE", "return")).lower()
        else:
            self.value_net = None
            self.critic_optim = None
            self.critic_loss_coef = 0.0
            self.use_critic_replay = False
            self.critic_buffer_states = []
            self.critic_buffer_targets = []

        # Moving baseline
        self.baseline_mean = 0.0
        self.baseline_beta = 0.98
        self.final_happies = []
        self.final_rewards = []
        self.episode_best_rewards = []

        # Top-episode evaluation
        self.eval_top_episodes = bool(config.get("EVAL_TOP_EPISODES", False))
        self.eval_top_k = int(config.get("EVAL_TOP_K", 20))
        self.eval_top_steps = int(
            config.get("EVAL_TOP_MAX_STEPS_PER_EPISODE", config.get("EVAL_MAX_STEPS_PER_EPISODE", self.max_steps))
        )
        self.eval_top_greedy = bool(config.get("EVAL_TOP_GREEDY", True))
        self.eval_top_output_subdir = str(config.get("EVAL_TOP_OUTPUT_SUBDIR", "evaluation"))
        self.eval_top_best_images_subdir = str(config.get("EVAL_TOP_BEST_IMAGES_SUBDIR", "best_images_sorted"))

        self.expl_use_eps_mix = bool(config.get("EXPL_USE_EPS_MIX", True))

        from collections import defaultdict
        self.visit_counts = defaultdict(int)

        # Full action grid (no action levels)
        grid_lo, grid_hi = -0.25, 0.25
        self.action_phases = [
            {
                "end_ep": int(self.episodes),
                "type": "range",
                "min": grid_lo,
                "max": grid_hi,
            }
        ]

        self.eval_ignore_action_phases = bool(config.get("EVAL_IGNORE_ACTION_PHASES", True))

        # Inter-episode exploration gating
        self.use_episode_expl = bool(config.get("USE_EPISODE_EXPL", True))
        self.ep_expl_prob_start = float(config.get("EP_EXPL_PROB_START", 1.0))
        self.ep_expl_prob_end   = float(config.get("EP_EXPL_PROB_END", 0.001))
        self.ep_expl_decay_eps  = int(config.get("EP_EXPL_DECAY_EPISODES", self.episodes))

        # Intra-episode step epsilon schedule (full-episode linear decay)
        self.step_eps_start = float(config.get("STEP_EPS_START", 0.90))
        self.step_eps_end   = float(config.get("STEP_EPS_END", 0.05))

        # Clip peak trajectory
        self.clip_peak_trajectory = bool(config.get("CLIP_PEAK_TRAJECTORY", False))

        # Return mode
        self.return_mode = str(config.get("RETURN_MODE", "rtg")).lower()
        self.blend_terminal_weight = float(config.get("BLEND_TERMINAL_WEIGHT", 0.5))
        self.blend_immediate_weight = float(config.get("BLEND_IMMEDIATE_WEIGHT", 0.5))

        # Plotting config
        self.emotion_plot_ylim = tuple(self.config.get("EMOTION_PLOT_YLIM", [0.0, 1.0]))
        self.plot_final_metrics_every = int(self.config.get("PLOT_FINAL_METRICS_EVERY", 100))
        self.plot_loss_every = int(self.config.get("PLOT_LOSS_EVERY", 10))

        # Cutoff mode
        self.cutoff_trajectory_mode = str(self.config.get("CUTOFF_TRAJECTORY_MODE", "reward")).strip().lower()

        # Loss histories
        self.policy_loss_history = []
        self.critic_train_loss_history = []
        self.critic_eval_loss_history = []

        def _sched(start, end, horizon, ep):
            k = min(1.0, ep / max(1, horizon))
            return start + k * (end - start)

        self._sched = _sched

    def load_policy(self, path: str):
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        state = torch.load(str(p), map_location=self.device)
        self.policy.load_state_dict(state)
        self.logger.log_message(f"[CKPT] Loaded policy from {p}")
