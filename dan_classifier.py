import os
import sys
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import importlib.util
import cv2
import numpy as np
import pyautogui
import mss

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


class DANEmotionClassifier:
    """
    Screenshot-based emotion classifier using DAN (AffectNet) with a MediaPipe-compatible output order.

    Output probabilities are always in this order (same as MediaPipeEmotionClassifier):
      ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    """

    EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    _MP7_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    _LABEL_ALIASES = {"anger": "angry", "happiness": "happy"}

    # Label orders used by common DAN checkpoints / training scripts.
    # These are the class-index orders that DAN outputs before we remap to _MP7_LABELS.
    _DAN_PRESET_OUTPUT_LABELS = {
        # DAN/demo.py
        "affectnet8": ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger", "contempt"],
        # AffectNet-7 is AffectNet-8 with "contempt" removed.
        "affectnet7": ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"],
        # From DAN/rafdb.py: 0 surprise, 1 fear, 2 disgust, 3 happiness, 4 sadness, 5 anger, 6 neutral
        "rafdb7": ["surprise", "fear", "disgust", "happy", "sad", "anger", "neutral"],
    }

    def __init__(
        self,
        model_path: str,
        dan_preset: Optional[str] = None,
        detect_face: bool = True,
        if_no_face: str = "fail",
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        repo_root = Path(__file__).resolve().parent
        dan_root = (repo_root / "DAN").resolve()
        if not dan_root.exists():
            raise RuntimeError(f"DAN folder not found at: {dan_root}")
        if str(dan_root) not in sys.path:
            sys.path.insert(0, str(dan_root))
        try:
            from dan_demo_preprocess import demo_crop_face_pil_from_bgr  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to import DAN demo preprocessing helper from: {dan_root}") from e
        self._demo_crop_face_pil_from_bgr = demo_crop_face_pil_from_bgr

        # Load DAN model definition from file path (no reliance on sys.path / packages).
        dan_py = dan_root / "networks" / "dan.py"
        if not dan_py.exists():
            raise RuntimeError(f"DAN network definition not found: {dan_py}")
        spec = importlib.util.spec_from_file_location("_dan_networks_dan", str(dan_py))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to create import spec for: {dan_py}")
        dan_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dan_module)  # type: ignore[attr-defined]
        DAN = getattr(dan_module, "DAN", None)
        if DAN is None:
            raise RuntimeError(f"'DAN' class not found in: {dan_py}")

        resolved_model_path = self._resolve_model_path(model_path, repo_root=repo_root, dan_root=dan_root)
        checkpoint = torch.load(str(resolved_model_path), map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = self._strip_module_prefix(state_dict)

        num_class = self._infer_num_class(state_dict) or 7
        self.model_num_class = int(num_class)

        self.dan_preset = self._infer_dan_preset(resolved_model_path, self.model_num_class, dan_preset=dan_preset)
        self._OUT_TO_MP7 = self._build_out_to_mp7_map(self.dan_preset)

        self.model = DAN(num_head=4, num_class=self.model_num_class, pretrained=False)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.data_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.detect_face = bool(detect_face)
        self.if_no_face = str(if_no_face).strip().lower()
        if self.if_no_face not in ("fail", "full_image"):
            raise ValueError(f"if_no_face must be 'fail' or 'full_image', got: {if_no_face!r}")

        self.face_roi = None  # (x, y, w, h) in screen coords
        self.timelapse_frames = deque(maxlen=10)
        self._face_mesh = None  # lazily created (only needed when return_landmarks=True)
        self.image_save_every: int = 10  # save face crop every N steps; set to 1 to save every call

    @staticmethod
    def _resolve_model_path(model_path: str, repo_root: Path, dan_root: Path) -> Path:
        p = Path(str(model_path)).expanduser()
        if p.is_absolute() and p.exists():
            return p

        # Try relative to the RL folder caller, then repo root, then DAN folder.
        candidates = [
            (Path.cwd() / p),
            (repo_root / p),
            (dan_root / p),
            (dan_root / "models" / p),
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        raise RuntimeError(
            f"DAN model checkpoint not found: {model_path}. Tried: {', '.join(str(c) for c in candidates)}"
        )

    @staticmethod
    def _strip_module_prefix(state_dict: dict) -> dict:
        # Handle DataParallel checkpoints with "module." prefix.
        if not isinstance(state_dict, dict):
            return state_dict
        if not any(k.startswith("module.") for k in state_dict.keys()):
            return state_dict
        return {k[len("module.") :]: v for k, v in state_dict.items()}

    @staticmethod
    def _infer_num_class(state_dict: dict) -> Optional[int]:
        if not isinstance(state_dict, dict):
            return None
        for k, v in state_dict.items():
            if k.endswith("fc.weight") and hasattr(v, "shape") and len(getattr(v, "shape")) == 2:
                return int(v.shape[0])
        return None

    @classmethod
    def _norm_label(cls, name: str) -> str:
        s = str(name or "").strip().lower()
        return cls._LABEL_ALIASES.get(s, s)

    @classmethod
    def _build_out_to_mp7_map(cls, preset: str) -> np.ndarray:
        if preset not in cls._DAN_PRESET_OUTPUT_LABELS:
            raise ValueError(f"Unknown DAN preset '{preset}'. Known: {sorted(cls._DAN_PRESET_OUTPUT_LABELS.keys())}")
        out_labels = cls._DAN_PRESET_OUTPUT_LABELS[preset][:7]
        mapped = [cls._MP7_LABELS.index(cls._norm_label(lbl)) for lbl in out_labels]
        return np.array(mapped, dtype=np.int64)

    @classmethod
    def _infer_dan_preset(cls, resolved_model_path: Path, num_class: int, dan_preset: Optional[str] = None) -> str:
        if dan_preset:
            p = str(dan_preset).strip().lower()
            if p in cls._DAN_PRESET_OUTPUT_LABELS:
                return p
            if p in ("raf", "rafdb"):
                return "rafdb7"
            if p in ("affectnet7", "affecnet7", "affect7"):
                return "affectnet7"
            if p in ("affectnet8", "affecnet8", "affect8"):
                return "affectnet8"
            raise ValueError(
                f"Unknown dan_preset '{dan_preset}'. Known: {sorted(cls._DAN_PRESET_OUTPUT_LABELS.keys())}"
            )
        name = resolved_model_path.name.lower()
        if "raf" in name:
            return "rafdb7"
        if "affecnet8" in name or "affectnet8" in name:
            return "affectnet8"
        if "affecnet7" in name or "affectnet7" in name:
            return "affectnet7"
        if int(num_class) == 8:
            return "affectnet8"
        return "affectnet7"

    def _init_face_roi(self, screen: np.ndarray, template_path: str = "furhat_template.png", threshold: float = 0.8):
        template = cv2.imread(template_path)
        if template is None:
            raise RuntimeError(f"Error: Couldn't read template from {template_path}")

        res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < threshold:
            raise RuntimeError(f"Template match too weak during ROI init: {max_val:.2f}")

        t_h, t_w = template.shape[:2]
        x, y = max_loc
        self.face_roi = (x, y, t_w, t_h)

    def _detect_first_face_bbox(self, img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if img_bgr is None or img_bgr.size == 0:
            return None
        faces = self.face_cascade.detectMultiScale(img_bgr)
        if faces is None or len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        return int(x), int(y), int(w), int(h)

    def _detect_largest_face_bbox(self, img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        # Backward-compat: DAN/demo.py takes faces[0] without sorting.
        return self._detect_first_face_bbox(img_bgr)

    def _predict_probs_mp7(self, face_img_bgr: np.ndarray) -> Optional[np.ndarray]:
        if face_img_bgr is None or face_img_bgr.size == 0:
            return None

        img0 = self._demo_crop_face_pil_from_bgr(
            face_img_bgr,
            self.face_cascade,
            detect_face=self.detect_face,
            if_no_face=self.if_no_face,
        )
        if img0 is None:
            return None

        img = self.data_transforms(img0).view(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            out, _, _ = self.model(img)
            probs = F.softmax(out, dim=1)[0].detach().cpu().numpy().astype(np.float32)

        if probs.shape[0] == 7:
            dan_probs = probs
        elif probs.shape[0] == 8:
            # Map AffectNet-8 -> AffectNet-7 by folding "contempt" into "neutral"
            dan_probs = probs[:7].copy()
            dan_probs[0] += float(probs[7])
        else:
            raise RuntimeError(f"Unexpected DAN output classes: {probs.shape[0]}")

        mapped = np.zeros(7, dtype=np.float32)
        mapped[self._OUT_TO_MP7] = dan_probs[:7]
        return mapped

    def _ensure_facemesh(self):
        if self._face_mesh is not None:
            return
        try:
            import mediapipe as mp  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "MediaPipe is required to return landmarks (used by HUMANLIKE_NATURALNESS). "
                "Install mediapipe or disable HUMANLIKE_NATURALNESS."
            ) from e
        mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    def _extract_landmarks_xy(self, face_img_bgr: np.ndarray) -> Optional[np.ndarray]:
        self._ensure_facemesh()
        rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        facial_landmarks = result.multi_face_landmarks[0]
        landmarks = np.array([(pt.x, pt.y) for pt in facial_landmarks.landmark], dtype=np.float32)
        if landmarks.shape != (468, 2):
            return None
        return landmarks

    def predict_emotion(self, face_img_bgr: np.ndarray, return_landmarks: bool = False):
        probs = self._predict_probs_mp7(face_img_bgr)
        if probs is None:
            return None
        if return_landmarks:
            landmarks = self._extract_landmarks_xy(face_img_bgr)
            if landmarks is None:
                return None
            return probs, landmarks
        return probs

    def capture_virtual_furhat_face(
        self,
        template_path: str = "furhat_template.png",
        save_path: str = "current.jpg",
        step: int = 0,
        episode: int = 0,
        return_landmarks: bool = False,
    ):
        try:
            screenshot = pyautogui.screenshot()
            screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            if self.face_roi is None:
                self._init_face_roi(screen, template_path=template_path)

            x, y, t_w, t_h = self.face_roi
            face_img = screen[y : y + t_h, x : x + t_w]

            if save_path and (step % self.image_save_every == 0):
                dirn = os.path.dirname(save_path)
                if dirn:
                    os.makedirs(dirn, exist_ok=True)
                cv2.imwrite(save_path, face_img)

            if len(self.timelapse_frames) < self.timelapse_frames.maxlen:
                annotated = screen.copy()
                text = f"Episode {episode}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(text, font, 1.2, 2)
                cv2.putText(
                    annotated,
                    text,
                    (annotated.shape[1] - tw - 20, annotated.shape[0] - 20),
                    font,
                    1.2,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                self.timelapse_frames.append(annotated)

            return self.predict_emotion(face_img, return_landmarks=return_landmarks)
        except Exception as e:
            print(f"Screenshot emotion capture error: {e}")
            return None

    def capture_face_image(
        self,
        template_path: str = "furhat_template.png",
        save_path: str = "current.jpg",
    ) -> None:
        """Capture and save the face crop without running DAN inference."""
        try:
            screenshot = pyautogui.screenshot()
            screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            if self.face_roi is None:
                self._init_face_roi(screen, template_path=template_path)
            x, y, t_w, t_h = self.face_roi
            face_img = screen[y : y + t_h, x : x + t_w]
            if save_path:
                dirn = os.path.dirname(save_path)
                if dirn:
                    os.makedirs(dirn, exist_ok=True)
                cv2.imwrite(save_path, face_img)
        except Exception as e:
            print(f"[DAN] capture_face_image error: {e}")

    def capture_face_image_right_screen(
        self,
        template_path: str = "furhat_template.png",
        save_path: str = "current.jpg",
    ) -> None:
        """Capture and save the face crop from right monitor without DAN inference."""
        try:
            with mss.mss() as sct:
                monitor = {"top": 0, "left": 1920, "width": 1920, "height": 1080}
                screenshot = sct.grab(monitor)
                screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)
            if self.face_roi is None:
                self._init_face_roi(screen, template_path=template_path)
            x, y, t_w, t_h = self.face_roi
            face_img = screen[y : y + t_h, x : x + t_w]
            if save_path:
                dirn = os.path.dirname(save_path)
                if dirn:
                    os.makedirs(dirn, exist_ok=True)
                cv2.imwrite(save_path, face_img)
        except Exception as e:
            print(f"[DAN] capture_face_image_right_screen error: {e}")

    def capture_virtual_furhat_face_right_screen(
        self,
        template_path: str = "furhat_template.png",
        save_path: str = "current.jpg",
        step: int = 0,
        episode: int = 0,
        return_landmarks: bool = False,
    ):
        try:
            with mss.mss() as sct:
                monitor = {
                    "top": 0,
                    "left": 1920,
                    "width": 1920,
                    "height": 1080,
                }
                screenshot = sct.grab(monitor)
                screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)

            if self.face_roi is None:
                self._init_face_roi(screen, template_path=template_path)

            x, y, t_w, t_h = self.face_roi
            face_img = screen[y : y + t_h, x : x + t_w]

            if save_path and step % self.image_save_every == 0:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, face_img)

            if step < 10:
                annotated = screen.copy()
                text = f"Episode {episode}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(text, font, 1.2, 2)
                cv2.putText(
                    annotated,
                    text,
                    (annotated.shape[1] - tw - 20, annotated.shape[0] - 20),
                    font,
                    1.2,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                self.timelapse_frames.append(annotated)

            return self.predict_emotion(face_img, return_landmarks=return_landmarks)
        except Exception as e:
            print(f"Screenshot emotion capture error: {e}")
            return None
