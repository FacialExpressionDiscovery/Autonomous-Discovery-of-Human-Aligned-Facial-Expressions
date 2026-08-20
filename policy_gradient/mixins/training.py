from __future__ import annotations

import copy
import random
from collections import OrderedDict
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class TrainingMixin:
    def train(self):
        episode_avg_happies = []

        for ep in range(1, self.episodes + 1):
            self.current_episode = ep

            ep_expl_prob = self._episode_exploration_prob(ep) if self.use_episode_expl else 0.0
            exploration_on = (random.random() < ep_expl_prob) if self.use_episode_expl else self.expl_use_eps_mix

            step_aux = []
            saved_full_states = []
            ep_dir = self.output_dir / f"episode_{ep}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            self.logger.switch_log_file(ep_dir / "episode.log")
            self.logger.log_message(f"=== EPISODE {ep} START ===")
            self.logger.log_message(f"[Expl] ep={ep} ep_prob={ep_expl_prob:.3f} exploration_on={exploration_on}")

            eps_first = self._step_eps(0)
            eps_last = self._step_eps(int(self.max_steps) - 1)
            self.logger.log_message(f"[Expl] ep={ep} eps_first_step={eps_first:.3f} eps_last_step={eps_last:.3f}")

            s = self.env.reset()
            time.sleep(0.2)
            happy_probs = []

            saved_states_sel = []
            saved_choices    = []
            rewards          = []
            actions_taken    = []
            states_logged    = []
            critic_loss_values = []

            happy_curr = 0.0
            t_perf_curr = 0.0
            t_classify_curr = 0.0

            num_states = int(self.max_steps)
            num_actions = max(0, num_states - 1)

            state_rewards = []
            state_happies = []
            states_per_step = []

            for t in range(num_states):
                if t < num_actions:
                    saved_full_states.append(s.copy())
                    s_selected = self._selected_from_full(s)

                    deltas_np, chosen_idxs, aux = self._sample_action(
                        s, step_idx=t, exploration_on=exploration_on, apply_action_phase=True
                    )
                    action_dict = OrderedDict((self.selected_params[i], float(deltas_np[i])) for i in range(self.obs_dim))
                    self.logger.log_message(f"[StepExpl] t={t} eps_used={aux['eps']:.4f} mean_entropy={aux['mean_entropy']:.4f}")

                    saved_states_sel.append(s_selected.copy())
                    saved_choices.append(chosen_idxs)
                    actions_taken.append(deltas_np.copy())
                    step_aux.append(aux)
                else:
                    deltas_np = None
                    action_dict = OrderedDict((self.selected_params[i], 0.0) for i in range(self.obs_dim))

                img_path = ep_dir / f"step_{t}.jpg"
                s_eval, r_eval, penalty_eval, probs_eval, t_perf_curr, t_classify_curr = self.env.step(
                    s, t, ep, save_path=str(img_path)
                )

                happy_curr = float(probs_eval[self.designated_emotion_index]) if probs_eval is not None else 0.0

                state_dict = OrderedDict((k, float(v)) for k, v in zip(self.env.expression_keys, s_eval))
                self.logger.log_step(
                    t,
                    state_dict,
                    action_dict,
                    r_eval,
                    0.0,
                    happy_curr,
                    t_perf_curr,
                    t_classify_curr,
                )

                state_rewards.append(float(r_eval))
                state_happies.append(float(happy_curr))
                states_per_step.append(s_eval.copy())

                if t < num_actions:
                    s = self._apply_action(s, deltas_np)

            rewards = state_rewards[1:]
            happy_probs = state_happies[1:]

            rewards_all = list(rewards)
            states_all = states_per_step[1:] if len(states_per_step) > 1 else []
            happy_all = state_happies[1:]

            if rewards_all and states_all:
                T_best = min(len(rewards_all), len(states_all), len(happy_all))
                best_idx = int(np.argmax(np.asanyarray(rewards_all[:T_best], dtype=np.float32)))
                best_reward = float(rewards_all[best_idx])
                self.episode_best_rewards.append({"episode": ep, "best_reward": best_reward})
                best_state = states_all[best_idx].copy()
                best_img_path = ep_dir / "best_reward.jpg"
                self._capture_state_image(
                    best_state,
                    ep,
                    best_img_path,
                    override_happy=happy_all[best_idx],
                )
                self.logger.log_message(
                    f"[BEST-STATE] ep={ep} best_reward={best_reward:.4f} step={best_idx + 1} saved={best_img_path.name}"
                )
            else:
                self.episode_best_rewards.append({"episode": ep, "best_reward": float("-inf")})

            # Save full trajectory for critic (before clipping)
            rewards_full = list(rewards)
            saved_states_sel_full = [s.copy() for s in saved_states_sel]
            rewards_full_np = np.asanyarray(rewards_full, dtype=np.float32)
            skip_actor_update = bool(
                rewards_full_np.size > 0 and np.all(np.isclose(rewards_full_np, 0.0, atol=1e-8))
            )

            if self.clip_peak_trajectory:
                rewards_np = np.asanyarray(rewards, dtype=np.float32)
                T_full = len(rewards)

                mode = self.cutoff_trajectory_mode
                if mode in ("happiness", "emotion", "designated_emotion"):
                    metric_arr = np.asanyarray(happy_probs[:T_full], dtype=np.float32)
                    metric_name = self.designated_emotion_slug
                else:
                    metric_arr = rewards_np
                    metric_name = "reward"

                peak_idx = int(np.argmax(metric_arr))

                if peak_idx < T_full - 1:
                    keep_T = peak_idx + 1
                    self.logger.log_message(
                        f"[Clip] peak at step={peak_idx + 1} ({metric_name}={float(metric_arr[peak_idx]):.4f}); "
                        f"truncating T={T_full} -> {keep_T}"
                    )
                    self.logger.log_message(f"The Reward array is : {rewards}")
                    rewards            = rewards[:keep_T]
                    self.logger.log_message(f"The length of the new trajectory is : {len(rewards)}")
                    self.saved_full_states  = saved_full_states[:keep_T]
                    saved_states_sel   = saved_states_sel[:keep_T]
                    saved_choices      = saved_choices[:keep_T]
                    actions_taken      = actions_taken[:keep_T]
                    step_aux           = step_aux[:keep_T]
                    states_logged      = states_logged[:keep_T]
                    happy_probs        = happy_probs[:keep_T]

                    final_reward = float(rewards[-1])
                    final_happy_for_plot = float(happy_probs[-1]) if happy_probs else float(happy_curr)

                    self.final_rewards.append(final_reward)
                    self.final_happies.append(final_happy_for_plot)
                    episode_score = final_reward
                    self.logger.log_message(f"Episode {ep} ended. Final {self.designated_emotion_name}: {final_happy_for_plot:.4f}")
                else:
                    effective_final_happy_for_best = float(rewards[-1])
                    episode_score = effective_final_happy_for_best
                    final_reward = float(rewards[-1])
                    final_happy_for_plot = float(happy_probs[-1]) if happy_probs else float(happy_curr)
                    self.final_rewards.append(final_reward)
                    self.final_happies.append(final_happy_for_plot)
                    self.logger.log_message(f"Episode {ep} ended. Final {self.designated_emotion_name}: {final_happy_for_plot:.4f}")
            else:
                final_happy = float(happy_curr)
                episode_score = final_happy
                final_reward = float(rewards[-1])
                final_happy_for_plot = float(happy_probs[-1]) if happy_probs else float(happy_curr)
                self.final_rewards.append(final_reward)
                self.final_happies.append(final_happy_for_plot)
                self.logger.log_message(f"Episode {ep} ended. Final {self.designated_emotion_name}: {final_happy_for_plot:.4f}")

            T = len(rewards)

            if T == 0:
                self.logger.log_message(f"[WARN] Episode {ep} has T=0, skipping update")
                continue

            # Compute returns
            if self.return_mode == "terminal":
                final_r = float(rewards[-1])
                returns_policy = [(self.gamma**(T - 1 - t)) * final_r for t in range(T)]
            elif self.return_mode == "immediate":
                returns_policy = [float(r) for r in rewards]
            elif self.return_mode == "blend":
                final_r = float(rewards[-1])
                returns_policy = [
                    self.blend_terminal_weight * (self.gamma**(T - 1 - t)) * final_r
                    + self.blend_immediate_weight * float(rewards[t])
                    for t in range(T)
                ]
            else:
                raise ValueError(f"Unknown RETURN_MODE='{self.return_mode}'. Choose: terminal, immediate, blend")

            R = torch.as_tensor(returns_policy, dtype=torch.float32, device=self.device)
            T = R.shape[0]

            advantages = None
            value_loss_item = 0.0
            values_policy = None

            if self.use_actor_critic:
                states_np_policy = np.stack(saved_states_sel, axis=0).astype(np.float32)
                states_t_policy = torch.as_tensor(states_np_policy, dtype=torch.float32, device=self.device)

                if self.critic_use_full_traj:
                    states_np_critic = np.stack(saved_states_sel_full, axis=0).astype(np.float32)
                    rewards_for_critic = np.asanyarray(rewards_full, dtype=np.float32)
                else:
                    states_np_critic = states_np_policy
                    rewards_for_critic = np.asanyarray(rewards, dtype=np.float32)

                if self.critic_target_mode == "reward":
                    critic_targets_np = rewards_for_critic
                elif self.critic_target_mode == "blend":
                    T_c = len(rewards_for_critic)
                    final_r_c = float(rewards_for_critic[-1])
                    critic_targets_np = np.asanyarray(
                        [self.blend_terminal_weight * (self.gamma ** (T_c - 1 - t)) * final_r_c
                         + self.blend_immediate_weight * float(rewards_for_critic[t]) for t in range(T_c)],
                        dtype=np.float32
                    )
                else:  # "return"
                    if self.return_mode == "terminal":
                        T_c = len(rewards_for_critic)
                        final_r_c = float(rewards_for_critic[-1])
                        critic_targets_np = np.asanyarray(
                            [(self.gamma ** (T_c - 1 - t)) * final_r_c for t in range(T_c)],
                            dtype=np.float32
                        )
                    elif self.return_mode == "immediate":
                        critic_targets_np = rewards_for_critic.copy()
                    elif self.return_mode == "blend":
                        T_c = len(rewards_for_critic)
                        final_r_c = float(rewards_for_critic[-1])
                        critic_targets_np = np.asanyarray(
                            [self.blend_terminal_weight * (self.gamma ** (T_c - 1 - t)) * final_r_c
                             + self.blend_immediate_weight * float(rewards_for_critic[t]) for t in range(T_c)],
                            dtype=np.float32
                        )
                    else:
                        raise ValueError(f"Unknown RETURN_MODE='{self.return_mode}' for critic.")

                if self.use_critic_replay:
                    T_c = states_np_critic.shape[0]
                    for i in range(T_c):
                        self.critic_buffer_states.append(states_np_critic[i].copy())
                        self.critic_buffer_targets.append(float(critic_targets_np[i]))

                    if len(self.critic_buffer_states) > self.critic_buffer_capacity:
                        overflow = len(self.critic_buffer_states) - self.critic_buffer_capacity
                        if overflow > 0:
                            self.critic_buffer_states = self.critic_buffer_states[overflow:]
                            self.critic_buffer_targets = self.critic_buffer_targets[overflow:]

                    if len(self.critic_buffer_states) >= self.critic_batch_size:
                        for _ in range(self.critic_updates_per_ep):
                            idx = np.random.choice(len(self.critic_buffer_states), size=self.critic_batch_size, replace=False)
                            batch_s = torch.as_tensor(
                                np.stack([self.critic_buffer_states[j] for j in idx], axis=0),
                                dtype=torch.float32, device=self.device
                            )
                            batch_y = torch.as_tensor(
                                [self.critic_buffer_targets[j] for j in idx],
                                dtype=torch.float32, device=self.device
                            )
                            pred = self.value_net(batch_s)
                            value_loss = 0.5 * (pred - batch_y).pow(2).mean()
                            critic_loss_values.append(float(value_loss.item()))
                            self.critic_optim.zero_grad()
                            (self.critic_loss_coef * value_loss).backward()
                            nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
                            self.critic_optim.step()

                    critic_train = float(np.mean(critic_loss_values)) if critic_loss_values else 0.0
                    self.critic_train_loss_history.append(critic_train)

                    values_policy = self.value_net(states_t_policy)
                    value_loss_item = float(0.5 * (values_policy - R).pow(2).mean().item())
                    self.critic_eval_loss_history.append(float(value_loss_item))
                    advantages = (R - values_policy).detach()

                else:
                    K = int(self.config.get("CRITIC_EPOCHS_PER_EP", 1))
                    last_values = None
                    for _ in range(K):
                        values_policy = self.value_net(states_t_policy)
                        value_loss = 0.5 * (values_policy - R).pow(2).mean()
                        critic_loss_values.append(float(value_loss.item()))
                        self.critic_optim.zero_grad()
                        (self.critic_loss_coef * value_loss).backward()
                        nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
                        self.critic_optim.step()
                        last_values = values_policy
                        value_loss_item = float(value_loss.item())

                    critic_train = float(np.mean(critic_loss_values)) if critic_loss_values else 0.0
                    self.critic_train_loss_history.append(critic_train)

                    values_policy = last_values
                    self.critic_eval_loss_history.append(float(value_loss_item))
                    advantages = (R - values_policy).detach()

            # Policy gradient update
            loss_values = []

            if skip_actor_update:
                self.logger.log_message(
                    f"[ACTOR] Episode {ep}: full trajectory rewards are all zero; skipping actor/policy update."
                )
            else:
                for t in range(T):
                    logpt = self._logprob_of_choice_masked_mix(saved_full_states[t], saved_choices[t], step_aux[t])

                    if self.use_actor_critic:
                        weight = advantages[t]
                    else:
                        weight = R[t]

                    loss_t = -(weight * logpt)

                    self.optim.zero_grad()
                    loss_t.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                    self.optim.step()

                    loss_values.append(float(loss_t.item()))

            self.logger.log_message(
                f"Episode {ep} | Return: {sum(rewards):.4f} | "
                f"Avg{self.designated_emotion_name}: {np.mean(happy_probs) if happy_probs else 0.0:.4f} | "
                f"Loss(last step): {loss_values[-1] if loss_values else 0.0:.4f}"
            )

            if self.use_actor_critic:
                self.logger.log_message(f"[AC] Episode {ep}: value_loss={value_loss_item:.6f}")

            critic_eval = float(value_loss_item) if self.use_actor_critic else None
            self._plot_episode_metrics(ep, loss_values, happy_probs, rewards, ep_dir, critic_loss_values=critic_loss_values, critic_eval_loss=critic_eval)

            policy_loss_scalar = float(np.mean(loss_values)) if loss_values else 0.0
            self.policy_loss_history.append(policy_loss_scalar)

            torch.save(self.policy.state_dict(), ep_dir / "policy.pt")
            episode_avg_happies.append(np.mean(happy_probs) if happy_probs else 0.0)

            if (ep) % self.plot_final_metrics_every == 0:
                self._plot_final_happiness(ep_dir, False)
                self._plot_max_rewards(ep_dir, False)

            if (ep) % self.plot_loss_every == 0:
                self._plot_loss_curves(ep_dir, False)

        # End of all episodes
        self._plot_training_summary(episode_avg_happies)
        self._plot_final_happiness(None, True)
        self._plot_max_rewards(None, True)
        self._plot_loss_curves(None, True)
        if self.eval_top_episodes:
            self._evaluate_top_episodes()
