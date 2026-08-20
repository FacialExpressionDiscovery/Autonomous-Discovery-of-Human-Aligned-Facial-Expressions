from __future__ import annotations


class PlottingMixin:
    def _write_series_log(self, out_path, title, x_label, y_label, xs, ys, empty_message=None):
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8") as f:
            f.write(f"{title}\n")
            f.write(f"{x_label}\t{y_label}\n")
            if ys:
                for x, y in zip(xs, ys):
                    f.write(f"{x}\t{float(y):.10f}\n")
            elif empty_message is not None:
                f.write(f"# {empty_message}\n")

    def _plot_episode_metrics(
        self,
        episode,
        losses,
        happy_probs,
        rewards,
        out_dir,
        critic_loss_values=None,
        critic_eval_loss=None,
    ):
        import matplotlib.pyplot as plt

        if rewards is None:
            rewards = []
        if critic_loss_values is None:
            critic_loss_values = []

        # 1) Policy loss
        if losses is not None and len(losses) > 0:
            policy_xs = list(range(len(losses)))
            fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
            ax.plot(policy_xs, losses)
            ax.set_title(f"Episode {episode} Policy Loss")
            ax.set_xlabel("Policy update idx (1 per kept step)")
            ax.set_ylabel("Loss")
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(out_dir / f"loss_ep{episode}.png", bbox_inches="tight")
            plt.close(fig)
            self._write_series_log(
                out_dir / f"policy_loss_ep{episode}.log",
                f"Episode {episode} Policy Loss",
                "policy_update_idx",
                "loss",
                policy_xs,
                losses,
            )
        else:
            self._write_series_log(
                out_dir / f"policy_loss_ep{episode}.log",
                f"Episode {episode} Policy Loss",
                "policy_update_idx",
                "loss",
                [],
                [],
                empty_message="No policy loss available",
            )

        # 2) Metrics
        T = min(len(happy_probs), len(rewards))
        if T > 0:
            happy_plot = list(happy_probs[:T])
            xs = list(range(1, T + 1))
            fig, ax = plt.subplots(figsize=(7.5, 4), dpi=100)
            ax.plot(xs, happy_plot, label=self.designated_emotion_name)
            ax.set_title(f"Episode {episode} Metrics")
            ax.set_xlabel("Step")
            ax.set_ylabel("Value (0..1)")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True)
            fig.subplots_adjust(right=0.78)
            ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), borderaxespad=0.0)
            fig.savefig(out_dir / f"metrics_ep{episode}.png", bbox_inches="tight")
            plt.close(fig)

        # 3) Critic loss
        fig, ax = plt.subplots(figsize=(7, 4), dpi=100)

        if self.use_actor_critic:
            if len(critic_loss_values) > 0:
                critic_xs = list(range(len(critic_loss_values)))
                ax.plot(critic_xs, critic_loss_values)
                ax.set_title(f"Critic Loss Updates (Episode {episode})")
                ax.set_xlabel("Critic update idx")
                ax.set_ylabel("Value loss")
                ax.grid(True)
                critic_log_title = f"Critic Loss Updates (Episode {episode})"
                critic_log_xs = critic_xs
                critic_log_ys = critic_loss_values
                critic_empty_message = None
            elif critic_eval_loss is not None:
                ax.plot([0], [critic_eval_loss])
                ax.set_title(f"Critic Loss (Episode {episode})")
                ax.set_xlabel("No critic updates; showing eval loss")
                ax.set_ylabel("Value loss")
                ax.grid(True)
                critic_log_title = f"Critic Loss (Episode {episode})"
                critic_log_xs = [0]
                critic_log_ys = [critic_eval_loss]
                critic_empty_message = None
            else:
                ax.text(0.5, 0.5, "No critic loss available", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                critic_log_title = f"Critic Loss (Episode {episode})"
                critic_log_xs = []
                critic_log_ys = []
                critic_empty_message = "No critic loss available"
        else:
            ax.text(0.5, 0.5, "Actor-critic disabled", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            critic_log_title = f"Critic Loss (Episode {episode})"
            critic_log_xs = []
            critic_log_ys = []
            critic_empty_message = "Actor-critic disabled"

        fig.tight_layout()
        fig.savefig(out_dir / f"critic_loss_ep{episode}.png", bbox_inches="tight")
        plt.close(fig)
        self._write_series_log(
            out_dir / f"critic_loss_ep{episode}.log",
            critic_log_title,
            "critic_update_idx",
            "value_loss",
            critic_log_xs,
            critic_log_ys,
            empty_message=critic_empty_message,
        )

        # 4) Reward curve
        if rewards is not None and len(rewards) > 0:
            xs = list(range(1, len(rewards) + 1))
            fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
            ax.plot(xs, rewards)
            ax.set_title(f"Episode {episode} Reward")
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(out_dir / f"reward_ep{episode}.png", bbox_inches="tight")
            plt.close(fig)

    def _plot_training_summary(self, avg_happies):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(avg_happies)
        plt.title(f"Average {self.designated_emotion_name} over Episodes")
        plt.xlabel("Episode"); plt.ylabel(f"Average {self.designated_emotion_name}")
        plt.ylim(self.emotion_plot_ylim[0], self.emotion_plot_ylim[1])
        plt.grid(); plt.savefig(self.output_dir / f"avg_{self.designated_emotion_slug}_over_episodes.png"); plt.close()

    def _plot_final_happiness(self, ep_dir, fin):
        import matplotlib.pyplot as plt

        if not self.final_happies:
            return

        xs = list(range(1, len(self.final_happies) + 1))
        plt.figure()
        plt.plot(xs, self.final_happies, label=self.designated_emotion_name)
        plt.title(f"Final {self.designated_emotion_name} per Episode")
        plt.ylabel(f"{self.designated_emotion_name} (0..1)")
        plt.xlabel("Episode")
        plt.ylim(self.emotion_plot_ylim[0], self.emotion_plot_ylim[1])
        plt.grid()

        if fin:
            plt.savefig(self.output_dir / f"final_{self.designated_emotion_slug}_over_episodes_final.png", bbox_inches="tight")
        else:
            plt.savefig(ep_dir / f"final_{self.designated_emotion_slug}_over_episodes_intermediate.png", bbox_inches="tight")
        plt.close()

    def _plot_loss_curves(self, ep_dir, fin):
        import matplotlib.pyplot as plt

        if self.policy_loss_history:
            xs = list(range(1, len(self.policy_loss_history) + 1))
            plt.figure()
            plt.plot(xs, self.policy_loss_history)
            plt.title("Policy Loss (mean per episode)")
            plt.xlabel("Episode")
            plt.ylabel("Policy loss")
            plt.grid()
            if fin:
                plt.savefig(self.output_dir / "policy_loss_over_episodes_final.png", bbox_inches="tight")
                policy_log_path = self.output_dir / "policy_loss_over_episodes_final.log"
            else:
                plt.savefig(ep_dir / "policy_loss_over_episodes_intermediate.png", bbox_inches="tight")
                policy_log_path = ep_dir / "policy_loss_over_episodes_intermediate.log"
            plt.close()
            self._write_series_log(
                policy_log_path,
                "Policy Loss (mean per episode)",
                "episode",
                "policy_loss",
                xs,
                self.policy_loss_history,
            )

        if self.use_actor_critic and self.critic_eval_loss_history:
            xs = list(range(1, len(self.critic_eval_loss_history) + 1))
            plt.figure()
            plt.plot(xs, self.critic_eval_loss_history)
            plt.title("Critic Eval Loss (mean per episode)")
            plt.xlabel("Episode")
            plt.ylabel("Value loss")
            plt.grid()
            if fin:
                plt.savefig(self.output_dir / "critic_loss_over_episodes_final_eval.png", bbox_inches="tight")
                critic_log_path = self.output_dir / "critic_loss_over_episodes_final_eval.log"
            else:
                plt.savefig(ep_dir / "critic_loss_over_episodes_intermediate_eval.png", bbox_inches="tight")
                critic_log_path = ep_dir / "critic_loss_over_episodes_intermediate_eval.log"
            plt.close()
            self._write_series_log(
                critic_log_path,
                "Critic Eval Loss (mean per episode)",
                "episode",
                "value_loss",
                xs,
                self.critic_eval_loss_history,
            )

    def _plot_max_rewards(self, ep_dir, fin):
        import matplotlib.pyplot as plt

        if not hasattr(self, "episode_best_rewards") or not self.episode_best_rewards:
            return

        episodes = []
        best_rewards = []
        for row in self.episode_best_rewards:
            ep = row.get("episode")
            best_reward = row.get("best_reward")
            if ep is None or best_reward is None:
                continue
            if best_reward == float("-inf"):
                continue
            episodes.append(int(ep))
            best_rewards.append(float(best_reward))

        if not best_rewards:
            return

        plt.figure()
        plt.plot(episodes, best_rewards)
        plt.title("Maximum Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Maximum reward")
        plt.ylim(0.0, 1.0)
        plt.grid()
        if fin:
            plt.savefig(self.output_dir / "max_reward_over_episodes_final.png", bbox_inches="tight")
        else:
            plt.savefig(ep_dir / "max_reward_over_episodes_intermediate.png", bbox_inches="tight")
        plt.close()
