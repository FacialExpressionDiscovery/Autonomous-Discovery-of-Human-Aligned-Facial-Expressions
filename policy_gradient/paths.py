from __future__ import annotations

from pathlib import Path

PG_DIR = Path(__file__).resolve().parents[1]


def resolve_pg_path(path: str | Path) -> Path:
    """
    Resolve a path used by the Policy Gradient code.

    - Absolute paths are returned as-is.
    - Relative paths are first tried relative to CWD, then relative to `PG_DIR`.
    - If the path doesn't exist, returns the `PG_DIR`-relative candidate (so callers
      can still produce a helpful error message).
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p

    candidates = [Path.cwd() / p, PG_DIR / p]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return (PG_DIR / p).resolve()

