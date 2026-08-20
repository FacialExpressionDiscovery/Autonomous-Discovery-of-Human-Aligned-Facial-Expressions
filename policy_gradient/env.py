from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from furhat_remote_api import FurhatRemoteAPI

from .classifier_loader import _create_emotion_classifier
from .emotions import _get_designated_emotion_index
from .paths import resolve_pg_path

# All face AU params zeroed out — used to ensure a clean neutral state
# before locking idle behaviors via the RT API.
_FULL_NEUTRAL_PARAMS: dict[str, float] = {
    "BROW_DOWN_LEFT": 0.0, "BROW_DOWN_RIGHT": 0.0,
    "BROW_IN_LEFT": 0.0, "BROW_IN_RIGHT": 0.0,
    "BROW_INNER_UP": 0.0,
    "BROW_OUTER_UP_LEFT": 0.0, "BROW_OUTER_UP_RIGHT": 0.0,
    "BROW_UP_LEFT": 0.0, "BROW_UP_RIGHT": 0.0,
    "BLINK_LEFT": 0.0, "BLINK_RIGHT": 0.0,
    "EYE_BLINK_LEFT": 0.0, "EYE_BLINK_RIGHT": 0.0,
    "EYE_LOOK_DOWN_LEFT": 0.0, "EYE_LOOK_DOWN_RIGHT": 0.0,
    "EYE_LOOK_IN_LEFT": 0.0, "EYE_LOOK_IN_RIGHT": 0.0,
    "EYE_LOOK_OUT_LEFT": 0.0, "EYE_LOOK_OUT_RIGHT": 0.0,
    "EYE_LOOK_UP_LEFT": 0.0, "EYE_LOOK_UP_RIGHT": 0.0,
    "EYE_SQUINT_LEFT": 0.0, "EYE_SQUINT_RIGHT": 0.0,
    "EYE_WIDE_LEFT": 0.0, "EYE_WIDE_RIGHT": 0.0,
    "MOUTH_CLOSE": 0.0,
    "MOUTH_DIMPLE_LEFT": 0.0, "MOUTH_DIMPLE_RIGHT": 0.0,
    "MOUTH_FROWN_LEFT": 0.0, "MOUTH_FROWN_RIGHT": 0.0,
    "MOUTH_FUNNEL": 0.0,
    "MOUTH_LEFT": 0.0, "MOUTH_RIGHT": 0.0,
    "MOUTH_LOWER_DOWN_LEFT": 0.0, "MOUTH_LOWER_DOWN_RIGHT": 0.0,
    "MOUTH_PRESS_LEFT": 0.0, "MOUTH_PRESS_RIGHT": 0.0,
    "MOUTH_PUCKER": 0.0,
    "MOUTH_ROLL_LOWER": 0.0, "MOUTH_ROLL_UPPER": 0.0,
    "MOUTH_SHRUG_LOWER": 0.0, "MOUTH_SHRUG_UPPER": 0.0,
    "MOUTH_SMILE_LEFT": 0.0, "MOUTH_SMILE_RIGHT": 0.0,
    "MOUTH_STRETCH_LEFT": 0.0, "MOUTH_STRETCH_RIGHT": 0.0,
    "MOUTH_UPPER_UP_LEFT": 0.0, "MOUTH_UPPER_UP_RIGHT": 0.0,
    "CHEEK_PUFF": 0.0,
    "CHEEK_SQUINT_LEFT": 0.0, "CHEEK_SQUINT_RIGHT": 0.0,
    "JAW_FORWARD": 0.0, "JAW_LEFT": 0.0, "JAW_OPEN": 0.0, "JAW_RIGHT": 0.0,
    "LOOK_LEFT": 0.0, "LOOK_RIGHT": 0.0, "LOOK_UP": 0.0, "LOOK_DOWN": 0.0,
    "NOSE_SNEER_LEFT": 0.0, "NOSE_SNEER_RIGHT": 0.0,
    "EXPR_ANGER": 0.0, "EXPR_DISGUST": 0.0, "EXPR_FEAR": 0.0, "EXPR_SAD": 0.0,
    "SMILE_OPEN": 0.0, "SMILE_CLOSED": 0.0, "SURPRISE": 0.0,
}


class ExpressionUtils:
    _KNOWN_TEXTURES = [
        "Alex", "Brooklyn", "Chen", "Dorothy", "Expresivo", "Fedora", "Fernando",
        "Gyeong", "Hayden", "Isabel", "James", "Jamie", "Jane", "Kione", "Lamin",
        "Marty", "Maurice", "Nazar", "Omar", "Patricia", "Rania", "Samuel", "Titan",
        "Vinnie", "Yi", "Yumi", "Zhen",
    ]

    def __init__(
        self,
        classifier_type: str,
        furhat_ip: str,
        fer_model_path: str,
        template_path: str,
        capture_from_monitor: bool = True,
        furhat_texture: Optional[str] = None,
        furhat_texture_settle_s: float = 0.35,
        furhat_texture_strict: bool = True,
        classifier_module: Optional[str] = None,
        classifier_class: Optional[str] = None,
        classifier_init_kwargs: Optional[dict] = None,
        disable_microexpressions: bool = False,
        disable_blinking: bool = False,
        disable_head_sway: bool = False,
        reset_neutral_after_texture: bool = True,
        set_head_pose: bool = False,
        head_yaw: float = 0.0,
        head_pitch: float = 0.0,
        head_roll: float = 0.0,
    ):
        self.classifier_type = classifier_type.lower()
        self.furhat_ip = furhat_ip
        self.furhat = FurhatRemoteAPI(furhat_ip)
        self.furhat_texture = str(furhat_texture or "").strip()
        self.furhat_texture_settle_s = float(furhat_texture_settle_s)
        self.furhat_texture_strict = bool(furhat_texture_strict)
        self.template_path = str(resolve_pg_path(template_path))
        self.capture_from_monitor = capture_from_monitor
        self.disable_microexpressions = bool(disable_microexpressions)
        self.disable_blinking = bool(disable_blinking)
        self.disable_head_sway = bool(disable_head_sway)
        self.reset_neutral_after_texture = bool(reset_neutral_after_texture)
        self.set_head_pose = bool(set_head_pose)
        self.head_yaw = float(head_yaw)
        self.head_pitch = float(head_pitch)
        self.head_roll = float(head_roll)

        # Connect RT API when needed for idle control or head pose
        self._furhat_rt = None
        if self.disable_microexpressions or self.disable_blinking or self.disable_head_sway or self.set_head_pose:
            self._furhat_rt = self._connect_realtime(furhat_ip)

        # Order matters: texture switch resets Furhat's idle state (re-enables
        # microexpressions/blinking). So we apply texture first, then neutralise
        # expression state, then lock idle behaviors via the RT API.
        self._apply_texture_best_effort()
        self.emotion_classifier = _create_emotion_classifier(
            classifier_type=self.classifier_type,
            fer_model_path=fer_model_path,
            classifier_module=classifier_module,
            classifier_class=classifier_class,
            init_kwargs=classifier_init_kwargs,
        )

    @staticmethod
    def _slugify(name: str) -> str:
        s = str(name or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    def _canonical_texture_name(self, texture_name: str) -> str:
        raw = str(texture_name or "").strip()
        if not raw:
            return ""
        target_slug = self._slugify(raw)
        for tex in self._KNOWN_TEXTURES:
            if self._slugify(tex) == target_slug:
                return tex
        return raw

    # ------------------------------------------------------------------
    # Realtime API helpers (idle behavior control)
    # ------------------------------------------------------------------

    def _connect_realtime(self, ip: str):
        """Connect to Furhat Realtime API. Returns client or None on failure."""
        try:
            from furhat_realtime_api import FurhatClient as _RTClient  # type: ignore
            client = _RTClient(ip)
            client.connect()
            print(f"[RT-API] Connected to {ip}")
            return client
        except Exception as exc:
            print(f"[RT-API] Could not connect to {ip}: {exc}")
            return None

    def _rt_send(self, payload: dict) -> None:
        rt = self._furhat_rt
        if rt is None:
            return
        if hasattr(rt, "_run_coroutine") and hasattr(rt, "async_client"):
            rt._run_coroutine(rt.async_client.send_event(payload))
        else:
            rt.send(payload)

    def _apply_idle_config(self) -> None:
        """Send request.face.config based on disable_* flags. No-op if RT API not connected."""
        if self._furhat_rt is None:
            return
        try:
            self._rt_send({
                "type": "request.face.config",
                "blinking": not self.disable_blinking,
                "microexpressions": not self.disable_microexpressions,
                "head_sway": not self.disable_head_sway,
            })
            print(
                f"[RT-API] Idle config applied — "
                f"blinking={'off' if self.disable_blinking else 'on'}, "
                f"microexpressions={'off' if self.disable_microexpressions else 'on'}, "
                f"head_sway={'off' if self.disable_head_sway else 'on'}"
            )
        except Exception as exc:
            print(f"[RT-API] _apply_idle_config failed: {exc}")

    def restore_idle_behaviors(self) -> None:
        """Re-enable all idle behaviors. Call on cleanup/exit when idle was disabled."""
        if self._furhat_rt is None:
            return
        try:
            self._rt_send({
                "type": "request.face.config",
                "blinking": True,
                "microexpressions": True,
                "head_sway": True,
            })
            print("[RT-API] Idle behaviors restored.")
        except Exception as exc:
            print(f"[RT-API] restore_idle_behaviors failed: {exc}")

    def _set_head_pose(self) -> None:
        """Point head straight (or to configured yaw/pitch/roll) using the RT API."""
        if self._furhat_rt is None:
            return
        try:
            self._rt_send({"type": "request.attend.nobody"})
            time.sleep(0.3)
            self._furhat_rt.request_face_headpose(
                yaw=self.head_yaw,
                pitch=self.head_pitch,
                roll=self.head_roll,
                relative=False,
            )
            print(f"[RT-API] Head pose set — yaw={self.head_yaw}, pitch={self.head_pitch}, roll={self.head_roll}")
        except Exception as exc:
            print(f"[RT-API] _set_head_pose failed: {exc}")

    def _send_full_neutral(self) -> None:
        """Send a full-neutral gesture (all AU params = 0) to clear any expression
        state left over by the texture-switch animation before locking idle."""
        try:
            self.furhat.gesture(
                body={
                    "frames": [{"time": [0.25], "params": dict(_FULL_NEUTRAL_PARAMS)}],
                    "name": "FullNeutral",
                    "class": "furhatos.gestures.Gesture",
                },
                blocking=True,
            )
        except Exception as exc:
            print(f"[FURHAT] _send_full_neutral failed: {exc}")

    # ------------------------------------------------------------------
    # Texture application
    # ------------------------------------------------------------------

    def _apply_texture_best_effort(self) -> None:
        if not self.furhat_texture:
            # No texture to switch — still apply idle config if needed.
            self._apply_idle_config()
            return

        texture_name = self._canonical_texture_name(self.furhat_texture)
        applied = False

        g1 = {
            "frames": [{"time": [0.1], "texture": texture_name}],
            "name": "SetTextureDynamic",
            "class": "furhatos.gestures.Gesture",
        }
        try:
            self.furhat.gesture(body=g1, blocking=True)
            print(f"[FURHAT] texture applied: {texture_name}")
            applied = True
        except Exception:
            pass

        if not applied:
            g2 = {
                "frames": [{"time": [0.1], "params": {"texture": texture_name}}],
                "name": "SetTextureDynamic",
                "class": "furhatos.gestures.Gesture",
            }
            try:
                self.furhat.gesture(body=g2, blocking=True)
                print(f"[FURHAT] texture applied (fallback): {texture_name}")
                applied = True
            except Exception as e:
                msg = f"[FURHAT] could not apply texture '{texture_name}': {e}"
                if self.furhat_texture_strict:
                    raise RuntimeError(msg)
                print(f"[FURHAT][WARN] {msg}")

        if applied and self.furhat_texture_settle_s > 0:
            time.sleep(self.furhat_texture_settle_s)

        # Texture switch resets Furhat's idle state. Send a clean full-neutral
        # gesture first so no texture-transition expression is "frozen" when we
        # then disable idle behaviors via the RT API.
        if applied and self.reset_neutral_after_texture:
            self._send_full_neutral()

        self._apply_idle_config()
        if self.set_head_pose:
            self._set_head_pose()

    def capture_and_classify_expression(
        self,
        how_many_steps,
        episode,
        save_path="current.jpg",
        return_landmarks: bool = False,
    ):
        if self.capture_from_monitor:
            fn = getattr(self.emotion_classifier, "capture_virtual_furhat_face_right_screen", None)
        else:
            fn = getattr(self.emotion_classifier, "capture_virtual_furhat_face", None)

        if fn is None:
            raise RuntimeError(
                f"Classifier '{type(self.emotion_classifier).__name__}' does not implement the required screenshot capture API."
            )

        return fn(
            template_path=self.template_path,
            save_path=save_path,
            step=how_many_steps,
            episode=episode,
            return_landmarks=return_landmarks,
        )

    def perform_expression(self, expr: dict, duration=0.0):
        gesture = {"frames": [{"time": [0.25], "params": expr}], "name": "Dynamic Expression", "class": "furhatos.gestures.Gesture"}
        self.furhat.gesture(body=gesture, blocking=True)


class FurhatEnv:
    def __init__(self, config, expr_utils):
        self.config = config
        self.expr_utils = expr_utils
        self.expression_keys = list(config["BASE_EXPRESSION"].keys())
        self.designated_emotion_index = _get_designated_emotion_index(config)
        self.gap_based_reward = bool(config.get("GAP_BASED_REWARD", False))
        self.weighted_gap_based_reward = bool(config.get("WEIGHTED_GAP_BASED_REWARD", False))
        self.weighted_gap_alpha = float(config.get("WEIGHTED_GAP_ALPHA", 0.5))
        self.weighted_gap_alpha = float(max(0.0, min(1.0, self.weighted_gap_alpha)))
        self.state = np.zeros(len(self.expression_keys), dtype=np.float32)
        self.last_valid_probs = None

    def _get_emotion_name_for_index(self, idx: int) -> str:
        clf = getattr(self.expr_utils, "emotion_classifier", None)
        labels = list(getattr(clf, "EMOTION_LABELS", []) or [])
        idx = int(idx)
        if 0 <= idx < len(labels):
            return str(labels[idx])
        get_name = getattr(clf, "get_emotion_name", None)
        if callable(get_name):
            try:
                return str(get_name(idx))
            except Exception:
                pass
        return f"Emotion{idx}"

    def _build_annotation_lines(self, probs, designated_prob: float, note: Optional[str] = None) -> list[str]:
        designated_idx = int(self.designated_emotion_index)
        designated_name = self._get_emotion_name_for_index(designated_idx)
        fallback_line = f"{designated_name}: {float(designated_prob):.4f}"

        if probs is None:
            lines = [fallback_line]
        else:
            try:
                arr = np.asarray(probs, dtype=np.float32).reshape(-1).copy()
            except Exception:
                arr = np.zeros(0, dtype=np.float32)

            if arr.size == 0:
                lines = [fallback_line]
            else:
                selected: list[int] = []
                if 0 <= designated_idx < arr.size:
                    arr[designated_idx] = float(designated_prob)
                    selected.append(designated_idx)
                remaining = [int(i) for i in np.argsort(arr)[::-1] if int(i) not in selected]
                selected.extend(remaining[: max(0, 3 - len(selected))])
                if selected:
                    lines = [
                        f"{self._get_emotion_name_for_index(i)}: {float(arr[i]):.4f}"
                        for i in selected[:3]
                    ]
                else:
                    lines = [fallback_line]
        if note:
            lines.append(str(note))

        return lines

    def _annotate_saved_image(self, save_path: str, probs, designated_prob: float, note: Optional[str] = None) -> None:
        if not save_path:
            return
        try:
            if not Path(save_path).is_file():
                return
            img = cv2.imread(str(save_path))
            if img is None:
                return
            lines = self._build_annotation_lines(probs, designated_prob, note=note)
            if not lines:
                return
            x, y = 10, 30
            dy = 28
            font = cv2.FONT_HERSHEY_SIMPLEX
            for line in lines:
                cv2.putText(img, line, (x, y), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                y += dy
            cv2.imwrite(str(save_path), img)
        except Exception:
            return

    def _compute_emotion_reward(self, probs, designated_emotion_prob: float) -> float:
        arr = np.asarray(probs, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return 0.0

        designated_idx = int(self.designated_emotion_index)
        pt = float(designated_emotion_prob)

        other_probs = np.delete(arr, designated_idx) if (0 <= designated_idx < arr.size) else arr
        other_max = float(np.max(other_probs)) if other_probs.size > 0 else 0.0
        m = float(pt - other_max)

        # Precedence: weighted gap overrides plain gap when enabled.
        if self.weighted_gap_based_reward:
            alpha = float(self.weighted_gap_alpha)
            return float((1.0 - alpha) * pt + alpha * max(0.0, m))

        if not self.gap_based_reward:
            return pt

        max_prob = float(np.max(arr))
        if pt < max_prob:
            return 0.0

        if arr.size == 1:
            return float(max_prob)

        top2 = np.partition(arr, -2)[-2:]
        second_highest = float(np.min(top2))
        return float(max_prob - second_highest)

    def step(self, new_state, step_num, episode_num, save_path="current.jpg", annotate_save: bool = True):
        self.state = new_state
        expr = {k: float(v) for k, v in zip(self.expression_keys, self.state)}

        start_perf = time.perf_counter()
        self.expr_utils.perform_expression(expr)
        t_perf = time.perf_counter() - start_perf

        if step_num == 0:
            time.sleep(0.2)

        start_classify = time.perf_counter()
        out = self.expr_utils.capture_and_classify_expression(
            step_num,
            episode_num,
            save_path=save_path,
            return_landmarks=False,
        )
        t_classify = time.perf_counter() - start_classify

        if out is None:
            if annotate_save:
                fallback_probs = self.last_valid_probs
                if fallback_probs is not None:
                    arr_fb = np.asarray(fallback_probs, dtype=np.float32).reshape(-1)
                    idx = int(self.designated_emotion_index)
                    fb_designated = float(arr_fb[idx]) if 0 <= idx < arr_fb.size else 0.0
                    self._annotate_saved_image(save_path=save_path, probs=arr_fb, designated_prob=fb_designated, note="[fallback: previous valid probs]")
                else:
                    self._annotate_saved_image(save_path=save_path, probs=None, designated_prob=0.0, note="[no classifier output]")
            return self.state.copy(), 0.0, 0.0, None, t_perf, t_classify

        probs = out

        if probs is None:
            if annotate_save:
                fallback_probs = self.last_valid_probs
                if fallback_probs is not None:
                    arr_fb = np.asarray(fallback_probs, dtype=np.float32).reshape(-1)
                    idx = int(self.designated_emotion_index)
                    fb_designated = float(arr_fb[idx]) if 0 <= idx < arr_fb.size else 0.0
                    self._annotate_saved_image(save_path=save_path, probs=arr_fb, designated_prob=fb_designated, note="[fallback: previous valid probs]")
                else:
                    self._annotate_saved_image(save_path=save_path, probs=None, designated_prob=0.0, note="[no classifier output]")
            return self.state.copy(), 0.0, 0.0, None, t_perf, t_classify

        self.last_valid_probs = np.asarray(probs, dtype=np.float32).reshape(-1).copy()
        designated_emotion_prob = float(probs[self.designated_emotion_index])
        reward = self._compute_emotion_reward(probs=probs, designated_emotion_prob=designated_emotion_prob)

        if annotate_save:
            self._annotate_saved_image(save_path=save_path, probs=probs, designated_prob=designated_emotion_prob)

        penalty = 0.0
        return self.state.copy(), reward, penalty, probs, t_perf, t_classify

    def reset(self):
        self.state.fill(0.0)
        return self.state.copy()
