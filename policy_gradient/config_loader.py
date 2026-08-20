from __future__ import annotations

from pathlib import Path

import yaml

from .paths import resolve_pg_path


class ConfigLoader:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.config = None

    def load(self) -> dict:
        cfg_path = resolve_pg_path(self.path)
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        return self.config

