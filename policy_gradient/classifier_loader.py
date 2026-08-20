from __future__ import annotations

import importlib
from typing import Optional


# Only DAN classifier is supported.
_CLASSIFIER_REGISTRY = {
    "dan": ("dan_classifier", "DANEmotionClassifier"),
}


def _resolve_classifier_class(
    classifier_type: str,
    classifier_module: Optional[str] = None,
    classifier_class: Optional[str] = None,
):
    ctype = str(classifier_type or "").lower().strip()
    module_name = (classifier_module or "").strip() or None
    class_name = (classifier_class or "").strip() or None

    if module_name and class_name:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name, None)
        if cls is None:
            raise ImportError(f"Classifier class '{class_name}' not found in module '{module_name}'.")
        return cls

    if ctype not in _CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier type '{classifier_type}'. Known: {sorted(_CLASSIFIER_REGISTRY.keys())}. "
            "Or set CLASSIFIER_MODULE + CLASSIFIER_CLASS in config."
        )

    mod_name, cls_name = _CLASSIFIER_REGISTRY[ctype]
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name, None)
    if cls is None:
        raise ImportError(f"Classifier class '{cls_name}' not found in module '{mod_name}'.")
    return cls


def _create_emotion_classifier(
    classifier_type: str,
    fer_model_path: str,
    classifier_module: Optional[str] = None,
    classifier_class: Optional[str] = None,
    init_kwargs: Optional[dict] = None,
):
    cls = _resolve_classifier_class(
        classifier_type, classifier_module=classifier_module, classifier_class=classifier_class
    )
    init_kwargs = {k: v for k, v in dict(init_kwargs or {}).items() if v is not None}

    try:
        return cls(model_path=fer_model_path, **init_kwargs)
    except TypeError as e:
        msg = str(e)
        if "unexpected keyword argument" not in msg:
            raise
        if init_kwargs:
            try:
                return cls(model_path=fer_model_path)
            except TypeError as e2:
                if "unexpected keyword argument" not in str(e2):
                    raise
        return cls(fer_model_path)


def _get_classifier_labels(
    classifier_type: str,
    classifier_module: Optional[str] = None,
    classifier_class: Optional[str] = None,
):
    try:
        cls = _resolve_classifier_class(
            classifier_type, classifier_module=classifier_module, classifier_class=classifier_class
        )
    except Exception:
        return None
    labels = list(getattr(cls, "EMOTION_LABELS", []) or [])
    return labels or None
