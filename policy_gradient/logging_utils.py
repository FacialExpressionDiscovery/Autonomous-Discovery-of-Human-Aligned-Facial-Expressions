from __future__ import annotations

import logging
from pathlib import Path

import numpy as np


class LoggerManager:
    def __init__(self, output_dir: Path, job_id: str, config: dict):
        self.output_dir = output_dir
        self.job_id = job_id
        self.config = config
        self.designated_emotion_name = str(config.get("DESIGNATED_EMOTION_NAME", "Happy"))
        self.log_file = output_dir / f"{job_id}.log"

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.log_file, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(fh)

    def switch_log_file(self, new_log_path: Path):
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)
            try:
                h.flush()
                h.close()
            except Exception:
                pass
        fh = logging.FileHandler(new_log_path, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(fh)
        self.log_file = new_log_path
        self.logger = root

    def log_message(self, message: str):
        self.logger.info(message)

    def log_config(self):
        self.logger.info("=" * 60)
        self.logger.info("         Furhat REINFORCE Experiment Configuration")
        self.logger.info("-" * 60)
        for key, value in self.config.items():
            self.logger.info(f"{key:<25}: {value}")
        self.logger.info("-" * 60)
        self.logger.info(f"Output Directory       : {self.output_dir}")
        self.logger.info("=" * 60)

    def log_step(
        self,
        step,
        state,
        action,
        reward,
        penalty,
        designated_emotion_prob,
        t_perf=None,
        t_classify=None,
    ):
        self.logger.info(f"Step {step}")

        if isinstance(state, dict):
            self.logger.info("State (per-parameter values):")
            self.logger.info("{")
            for k, v in state.items():
                self.logger.info(f'  "{k}": {float(v):.2f},')
            self.logger.info("}")
        else:
            self.logger.info(f"State:  {np.array2string(state, separator=', ')}")

        if isinstance(action, dict):
            self.logger.info("Action (per-parameter deltas):")
            self.logger.info("{")
            for k, v in action.items():
                if float(v) == 0.0:
                    sign = ""
                elif float(v) > 0:
                    sign = "+"
                elif float(v) < 0:
                    sign = ""
                self.logger.info(f'  "{k}": {sign}{float(v):.2f}')
            self.logger.info("}")
        else:
            self.logger.info(f"Action: {np.array2string(np.array(action), separator=', ')}")

        self.logger.info(f"{self.designated_emotion_name}:  {designated_emotion_prob:.4f}")
        self.logger.info(f"Penalty: {penalty:.4f}")
        self.logger.info(f"Reward: {reward:.4f}")
        if t_perf is not None and t_classify is not None:
            self.logger.info(f"T_perform: {t_perf:.3f}s")
            self.logger.info(f"T_classify: {t_classify:.3f}s")
        self.logger.info("-" * 40)
