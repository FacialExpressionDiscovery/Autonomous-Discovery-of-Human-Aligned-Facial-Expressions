from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image


def bgr_to_rgb_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).convert("RGB")


def demo_crop_face_pil_from_bgr(
    img_bgr: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    *,
    detect_face: bool = True,
    if_no_face: str = "fail",
) -> Optional[Image.Image]:
    """
    Implements the exact high-level preprocessing from `DAN/demo.py`:
      - face_cascade.detectMultiScale(img_bgr)
      - take faces[0]
      - PIL crop((x, y, x+w, y+h))

    Returns a RGB PIL Image ready for DAN's torchvision transforms, or None if failing.
    """
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        return None

    img0 = bgr_to_rgb_pil(img_bgr)

    if not detect_face:
        return img0

    faces = face_cascade.detectMultiScale(img_bgr)
    if faces is None or len(faces) == 0:
        policy = str(if_no_face).strip().lower()
        if policy == "full_image":
            return img0
        if policy == "fail":
            return None
        raise ValueError(f"Unknown if_no_face policy: {if_no_face!r} (expected 'fail' or 'full_image')")

    x, y, w, h = faces[0]
    return img0.crop((int(x), int(y), int(x + w), int(y + h)))

