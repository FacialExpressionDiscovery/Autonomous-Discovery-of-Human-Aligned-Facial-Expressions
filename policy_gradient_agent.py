"""Compatibility entrypoint.

The original repository used a monolithic `policy_gradient_agent.py` (~3000 LOC).
This file now delegates to the refactored `policy_gradient` package.

Run:
  python policy_gradient_agent.py -c config.yaml
"""

from __future__ import annotations

from policy_gradient.cli import main
from policy_gradient.config_loader import ConfigLoader  # backward-compatible import surface
from policy_gradient.constants import base_expression  # backward-compatible import surface
from policy_gradient.trainer import PGTrainer  # backward-compatible import surface


if __name__ == "__main__":
    raise SystemExit(main())
