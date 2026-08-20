from __future__ import annotations

import argparse
import sys

from .config_loader import ConfigLoader
from .constants import base_expression
from .paths import PG_DIR, resolve_pg_path
from .trainer import PGTrainer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Policy Gradient runner")
    parser.add_argument(
        "-c",
        "--config",
        required=False,
        default="config.yaml",
        help="Path to config YAML (relative paths are resolved against Policy Gradient/ and CWD).",
    )
    args = parser.parse_args(argv)

    pg_dir_str = str(PG_DIR)
    if pg_dir_str not in sys.path:
        sys.path.insert(0, pg_dir_str)

    cfg = ConfigLoader(args.config).load()
    cfg["BASE_EXPRESSION"] = base_expression

    cfg.setdefault("DEVICE", "cuda")
    cfg.setdefault("CLASSIFIER_TYPE", "dan")
    if "DESIGNATED_EMOTION" not in cfg:
        cfg["DESIGNATED_EMOTION"] = cfg.get("DESIGNATED_EMOTION_INDEX", cfg.get("HAPPY_INDEX", 3))
    if not isinstance(cfg.get("DESIGNATED_EMOTION"), str):
        cfg.setdefault("DESIGNATED_EMOTION_INDEX", cfg["DESIGNATED_EMOTION"])
        cfg.setdefault("HAPPY_INDEX", cfg["DESIGNATED_EMOTION_INDEX"])
    cfg.setdefault("MAX_STEPS_PER_EPISODE", 6)
    cfg.setdefault("EPISODES", 100)
    cfg.setdefault("GAMMA", 0.9)
    cfg.setdefault("LR", 1e-3)

    # episode exploration gating defaults
    cfg.setdefault("USE_EPISODE_EXPL", True)
    cfg.setdefault("EP_EXPL_PROB_START", 1.0)
    cfg.setdefault("EP_EXPL_PROB_END", 0.001)
    cfg.setdefault("EP_EXPL_DECAY_EPISODES", cfg.get("EPISODES", 1000))
    cfg.setdefault("STEP_EPS_START", 0.90)
    cfg.setdefault("STEP_EPS_END", 0.05)

    cfg.setdefault("RUN_MODE", "train")
    cfg.setdefault("RESUME_FROM_PATH", "")
    cfg.setdefault("EVAL_EPISODES", 10)
    cfg.setdefault("EVAL_MAX_STEPS_PER_EPISODE", cfg.get("MAX_STEPS_PER_EPISODE", 6))
    cfg.setdefault("EVAL_GREEDY", True)
    cfg.setdefault("EVAL_OUTPUT_SUBDIR", "eval")

    trainer = PGTrainer(cfg)

    ckpt_path = str(cfg.get("RESUME_FROM_PATH") or "").strip()
    if ckpt_path:
        trainer.load_policy(str(resolve_pg_path(ckpt_path)))

    mode = str(cfg.get("RUN_MODE", "train")).lower()
    if mode == "eval":
        trainer.evaluate()
    elif mode == "train_and_eval":
        trainer.evaluate()
        trainer.train()
    else:
        trainer.train()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
