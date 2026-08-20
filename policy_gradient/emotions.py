from __future__ import annotations

import re
from typing import Optional


_CANONICAL_EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
_EMOTION_ALIASES = {
    "anger": "angry",
    "happiness": "happy",
    "surprised": "surprise",
}


def _slugify_emotion_name(name: str) -> str:
    s = str(name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "emotion"


def _norm_emotion_label(name: str) -> str:
    s = str(name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return _EMOTION_ALIASES.get(s, s)


def _resolve_emotion_index_from_name(name: str, labels: Optional[list] = None) -> int:
    target = _norm_emotion_label(name)
    if labels:
        normed_labels = [_norm_emotion_label(x) for x in labels]
        if target in normed_labels:
            return int(normed_labels.index(target))
    if target in _CANONICAL_EMOTION_LABELS:
        return int(_CANONICAL_EMOTION_LABELS.index(target))
    raise ValueError(
        f"Unknown designated emotion '{name}'. Expected one of: {', '.join(_CANONICAL_EMOTION_LABELS)} "
        f"(also accepts aliases: {', '.join(sorted(_EMOTION_ALIASES.keys()))})."
    )


def _get_designated_emotion_index(config: dict) -> int:
    raw = config.get("DESIGNATED_EMOTION", config.get("DESIGNATED_EMOTION_INDEX", config.get("HAPPY_INDEX", 3)))
    if isinstance(raw, str):
        return _resolve_emotion_index_from_name(raw, labels=None)
    return int(raw)

