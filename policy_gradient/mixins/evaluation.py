from __future__ import annotations
from __future__ import annotations

import copy
import shutil
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch


class EvaluationMixin:
    @torch.no_grad()
    def evaluate(self):
        import matplotlib.pyplot as plt

        eval_eps  = int(self.config.get("EVAL_EPISODES", 10))
        eval_steps= int(self.config.get("EVAL_MAX_STEPS_PER_EPISODE", self.max_steps))
        greedy    = bool(self.config.get("EVAL_GREEDY", True))
        subdir    = str(self.config.get("EVAL_OUTPUT_SUBDIR", "eval"))

        eval_dir = (self.output_dir / subdir)
        eval_dir.mkdir(parents=True, exist_ok=True)

        self.logger.switch_log_file(eval_dir / "eval.log")
        self.logger.log_message(f"[EVAL] episodes={eval_eps} steps={eval_steps} greedy={greedy}")

        self.final_happies = []
        ep_avgs = []

        for ep in range(1, eval_eps + 1):
            self.current_episode = ep
            self.logger.log_message(f"[EVAL][ActionPhase] apply={not self.eval_ignore_action_phases} {self._action_phase_desc(ep)}")
            s = self.env.reset()
            happy_trace = []
            num_states = int(eval_steps)
            num_actions = max(0, num_states - 1)

            for t in range(num_states):
                if t < num_actions:
                    deltas_np, _chosen_idxs, aux = self._sample_action(
                        s,
                        step_idx=t,
                        exploration_on=False,
                        deterministic=greedy,
                        apply_action_phase=not self.eval_ignore_action_phases,
                    )
                    action_dict = OrderedDict((self.selected_params[i], float(deltas_np[i])) for i in range(self.obs_dim))
                    self.logger.log_message(f"[EVAL] t={t} eps_used={aux['eps']:.4f} greedy={greedy}")
                else:
                    deltas_np = None
                    action_dict = OrderedDict((self.selected_params[i], 0.0) for i in range(self.obs_dim))

                img_path = eval_dir / f"ep{ep:04d}_step_{t}.jpg"
                s_eval, r_eval, _, probs_eval, t_perf, t_cls = self.env.step(s, t, ep, save_path=str(img_path))
                happy_curr = float(probs_eval[self.designated_emotion_index]) if probs_eval is not None else 0.0
                happy_trace.append(happy_curr)

                state_dict = OrderedDict((k, float(v)) for k, v in zip(self.env.expression_keys, s_eval))
                self.logger.log_step(t, state_dict, action_dict, r_eval, 0.0, happy_curr, t_perf, t_cls)

                if t < num_actions:
                    s = self._apply_action(s, deltas_np)

            final_happy = float(happy_trace[-1]) if happy_trace else 0.0
            self.final_happies.append(final_happy)
            ep_avgs.append(np.mean(happy_trace) if happy_trace else 0.0)
            self.logger.log_message(f"[EVAL] ep={ep} final_happy={final_happy:.4f} avg_happy={ep_avgs[-1]:.4f}")

        self._plot_final_happiness(eval_dir, fin=False)
        plt.figure(); plt.plot(ep_avgs); plt.title(f"Eval: Average {self.designated_emotion_name} per Episode")
        plt.xlabel("Eval Episode"); plt.ylabel(f"Avg {self.designated_emotion_name}"); plt.grid()
        plt.savefig(eval_dir / f"avg_{self.designated_emotion_slug}_over_eval_episodes.png"); plt.close()
        self.logger.log_message("[EVAL] done")

    def _get_emotion_name_for_index(self, idx: int) -> str:
        env_obj = getattr(self, "env", None)
        clf = getattr(env_obj, "emotion_classifier", None)
        if clf is None:
            clf = getattr(getattr(env_obj, "expr_utils", None), "emotion_classifier", None)
        labels = list(getattr(clf, "EMOTION_LABELS", []) or [])
        if 0 <= int(idx) < len(labels):
            return str(labels[int(idx)])
        get_name = getattr(clf, "get_emotion_name", None)
        if callable(get_name):
            try:
                return str(get_name(int(idx)))
            except Exception:
                pass
        return f"Emotion{int(idx)}"

    def _build_emotion_annotation_lines(self, happy_prob: float, emotion_probs=None) -> list[str]:
        designated_idx = int(self.designated_emotion_index)
        designated_name = str(self.designated_emotion_name)
        fallback_line = f"{designated_name}: {float(happy_prob):.4f}"

        if emotion_probs is None:
            return [fallback_line]

        try:
            probs = np.asarray(emotion_probs, dtype=np.float32).reshape(-1).copy()
        except Exception:
            return [fallback_line]
        if probs.size == 0:
            return [fallback_line]

        if 0 <= designated_idx < probs.size:
            probs[designated_idx] = float(happy_prob)
            selected = [designated_idx]
        else:
            selected = []

        remaining = [int(i) for i in np.argsort(probs)[::-1] if int(i) not in selected]
        selected.extend(remaining[: max(0, 3 - len(selected))])
        if not selected:
            return [fallback_line]

        return [
            f"{self._get_emotion_name_for_index(i)}: {float(probs[i]):.4f}"
            for i in selected[:3]
        ]

    def _annotate_image_with_scores(self, image_path: Path, happy_prob: float, emotion_probs=None):
        img = cv2.imread(str(image_path))
        if img is None:
            self.logger.log_message(f"[WARN] Could not read image for annotation: {image_path}")
            return

        lines = self._build_emotion_annotation_lines(happy_prob, emotion_probs=emotion_probs)

        x, y = 10, 30
        dy = 28
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        for line in lines:
            cv2.putText(img, line, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y += dy

        cv2.imwrite(str(image_path), img)

    def _capture_state_image(self, state_np, episode_num, out_path: Path, override_happy=None):
        if state_np is None:
            return None

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        _, _, _, probs_eval, _, _ = self.env.step(state_np, 0, episode_num, save_path=str(out_path), annotate_save=False)
        happy_prob = float(probs_eval[self.designated_emotion_index]) if probs_eval is not None else 0.0

        if override_happy is not None:
            happy_prob = float(override_happy)

        self._annotate_image_with_scores(out_path, happy_prob, emotion_probs=probs_eval)
        return happy_prob

    def _select_peak_index(self, rewards, happy_probs, naturalness_scores=None):
        if not rewards:
            return None, "reward", None

        mode = self.cutoff_trajectory_mode
        metric_name = "reward"
        metric_arr = np.asanyarray(rewards, dtype=np.float32)

        if mode in ("happiness", "emotion", "designated_emotion") and happy_probs:
            metric_name = self.designated_emotion_slug
            metric_arr = np.asanyarray(happy_probs, dtype=np.float32)

        idx = int(np.argmax(metric_arr))
        return idx, metric_name, metric_arr

    @torch.no_grad()
    def _evaluate_top_episodes(self):
        if not self.eval_top_episodes:
            return
        if not self.episode_best_rewards:
            self.logger.log_message("[TOP-EVAL] No episode rewards recorded; skipping.")
            return

        ranked = sorted(self.episode_best_rewards, key=lambda r: r["best_reward"], reverse=True)
        top_k = max(0, min(self.eval_top_k, len(ranked)))
        if top_k == 0:
            self.logger.log_message("[TOP-EVAL] No episodes to evaluate.")
            return

        eval_root = self.output_dir / self.eval_top_output_subdir
        eval_root.mkdir(parents=True, exist_ok=True)
        best_images_dir = eval_root / self.eval_top_best_images_subdir
        best_images_dir.mkdir(parents=True, exist_ok=True)

        eval_results = []
        policy_backup = copy.deepcopy(self.policy.state_dict())
        try:
            for train_rank, rec in enumerate(ranked[:top_k], start=1):
                ep = int(rec["episode"])
                train_best_reward = float(rec["best_reward"])
                policy_path = self.output_dir / f"episode_{ep}" / "policy.pt"
                if not policy_path.is_file():
                    self.logger.log_message(f"[TOP-EVAL][WARN] Missing {policy_path}, skipping.")
                    continue

                state = torch.load(str(policy_path), map_location=self.device)
                self.policy.load_state_dict(state)
                self.policy.eval()

                eval_ep_dir = eval_root / f"episode_{ep}"
                eval_ep_dir.mkdir(parents=True, exist_ok=True)
                self.logger.switch_log_file(eval_ep_dir / "eval.log")
                self.logger.log_message(
                    f"[TOP-EVAL] ep={ep} train_rank={train_rank} train_best_reward={train_best_reward:.4f}"
                )

                num_states = int(self.eval_top_steps)
                if num_states <= 0:
                    self.logger.log_message(f"[TOP-EVAL][WARN] ep={ep} eval_top_steps={num_states}; skipping.")
                    continue

                num_actions = max(0, num_states - 1)
                s = self.env.reset()
                rewards: list[float] = []
                happy_probs: list[float] = []
                states: list[np.ndarray] = []

                for t in range(num_states):
                    if t < num_actions:
                        deltas_np, chosen_idxs, aux = self._sample_action(
                            s, step_idx=t, exploration_on=False,
                            deterministic=self.eval_top_greedy,
                            apply_action_phase=not self.eval_ignore_action_phases,
                        )
                        action_dict = OrderedDict((self.selected_params[i], float(deltas_np[i])) for i in range(self.obs_dim))
                    else:
                        deltas_np = None
                        action_dict = OrderedDict((self.selected_params[i], 0.0) for i in range(self.obs_dim))

                    img_path = eval_ep_dir / f"step_{t}.jpg"
                    s_eval, r_eval, _, probs_eval, t_perf, t_cls = self.env.step(s, t, ep, save_path=str(img_path))

                    happy_curr = float(probs_eval[self.designated_emotion_index]) if probs_eval is not None else 0.0

                    state_dict = OrderedDict((k, float(v)) for k, v in zip(self.env.expression_keys, s_eval))
                    self.logger.log_step(t, state_dict, action_dict, r_eval, 0.0, happy_curr, t_perf, t_cls)
                    self.logger.log_message(f"[TOP-EVAL] t={t} greedy={self.eval_top_greedy}")

                    states.append(s_eval.copy())
                    rewards.append(float(r_eval))
                    happy_probs.append(happy_curr)

                    if t < num_actions:
                        s = self._apply_action(s, deltas_np)

                peak_idx, metric_name, metric_arr = self._select_peak_index(rewards, happy_probs)
                if peak_idx is None:
                    self.logger.log_message(f"[TOP-EVAL][WARN] ep={ep} no rewards recorded.")
                    continue

                peak_state = states[peak_idx]
                peak_metric = float(metric_arr[peak_idx]) if metric_arr is not None else float("nan")
                peak_reward = float(rewards[peak_idx])

                best_img_path = eval_ep_dir / "best_state.jpg"
                peak_happy = happy_probs[peak_idx] if peak_idx < len(happy_probs) else None
                self._capture_state_image(peak_state, ep, best_img_path, override_happy=peak_happy)

                self.logger.log_message(
                    f"[TOP-EVAL] ep={ep} peak_idx={peak_idx} metric={metric_name} value={peak_metric:.4f}"
                )

                if best_img_path.is_file():
                    eval_results.append({
                        "episode": ep,
                        "best_img": best_img_path,
                        "eval_peak_reward": peak_reward,
                        "train_best_reward": train_best_reward,
                        "train_rank": train_rank,
                    })
                else:
                    self.logger.log_message(f"[TOP-EVAL][WARN] Missing best_state image for ep={ep}, skipping sort entry.")

            if eval_results:
                sorted_eval = sorted(eval_results, key=lambda r: r["eval_peak_reward"], reverse=True)
                for eval_rank, rec in enumerate(sorted_eval, start=1):
                    out_name = (
                        f"evalrank_{eval_rank:02d}_ep{rec['episode']}_eval{rec['eval_peak_reward']:.4f}_"
                        f"train{rec['train_best_reward']:.4f}.jpg"
                    )
                    shutil.copy2(rec["best_img"], best_images_dir / out_name)
            else:
                self.logger.log_message("[TOP-EVAL] No best_state images to sort.")
        finally:
            self.policy.load_state_dict(policy_backup)
            self.policy.train()
