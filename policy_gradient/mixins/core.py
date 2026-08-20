from __future__ import annotations

import inspect
import shutil
import sys
from pathlib import Path

import numpy as np


class CoreMixin:
    MIRROR_MAP = {
        "BROW_DOWN_LEFT": "BROW_DOWN_RIGHT",
        "BROW_IN_LEFT": "BROW_IN_RIGHT",
        "BROW_UP_LEFT": "BROW_UP_RIGHT",
        "MOUTH_STRETCH_LEFT": "MOUTH_STRETCH_RIGHT",
        "MOUTH_UPPER_UP_LEFT": "MOUTH_UPPER_UP_RIGHT",
        "MOUTH_LOWER_DOWN_LEFT": "MOUTH_LOWER_DOWN_RIGHT",
        "MOUTH_FROWN_LEFT": "MOUTH_FROWN_RIGHT",
        "MOUTH_PRESS_LEFT": "MOUTH_PRESS_RIGHT",
        "MOUTH_DIMPLE_LEFT": "MOUTH_DIMPLE_RIGHT",
        "MOUTH_LEFT": "MOUTH_RIGHT",
        "BROW_OUTER_UP_LEFT": "BROW_OUTER_UP_RIGHT",
        "BLINK_LEFT": "BLINK_RIGHT",
        "CHEEK_SQUINT_LEFT": "CHEEK_SQUINT_RIGHT",
        "EYE_BLINK_LEFT": "EYE_BLINK_RIGHT",
        "EYE_LOOK_DOWN_LEFT": "EYE_LOOK_DOWN_RIGHT",
        "EYE_LOOK_IN_LEFT": "EYE_LOOK_IN_RIGHT",
        "EYE_LOOK_OUT_LEFT": "EYE_LOOK_OUT_RIGHT",
        "EYE_LOOK_UP_LEFT": "EYE_LOOK_UP_RIGHT",
        "EYE_SQUINT_LEFT": "EYE_SQUINT_RIGHT",
        "EYE_WIDE_LEFT": "EYE_WIDE_RIGHT",
        "NOSE_SNEER_LEFT": "NOSE_SNEER_RIGHT",
        # Note: BROW_INNER_UP is a shared (midline) parameter; no mirroring needed.
    }

    def _snapshot_self_script(self):
        """
        Save a copy of the current Python source into self.output_dir at startup.
        Works whether this file is run as __main__ or imported.
        """
        try:
            module = sys.modules.get("__main__", None)
            # Prefer the real file path if available
            src_path = None
            if module is not None:
                src_file = getattr(module, "__file__", None)
                if src_file:
                    p = Path(src_file).resolve()
                    if p.is_file():
                        src_path = p

            if src_path is not None:
                dst = self.output_dir / f"{src_path.stem}_snapshot_{self.job_id}.py"
                shutil.copy2(src_path, dst)
            else:
                # Fallback: grab the source text from the loaded module
                mod = module if module is not None else sys.modules[__name__]
                code = inspect.getsource(mod)
                dst = self.output_dir / f"reinforce_agent_snapshot_{self.job_id}.py"
                with open(dst, "w", encoding="utf-8") as f:
                    f.write(code)

            self.logger.log_message(f"[SNAPSHOT] Saved source to: {dst}")
        except Exception as e:
            self.logger.log_message(f"[SNAPSHOT][WARN] Could not snapshot source: {e}")


    def _selected_from_full(self, full_state_np):
        return full_state_np[self.param_indices]  # shape [N]

    def _apply_action(self, state_np, deltas_np):
        """Apply deltas to selected LEFT parameters and mirror to RIGHT ones."""
        new_vals = state_np[self.param_indices] + deltas_np
        new_vals = np.clip(new_vals, 0.0, 1.0)
        new_state = state_np.copy()

        for i, idx in enumerate(self.param_indices):
            param_name = self.selected_params[i]
            new_state[idx] = new_vals[i]

            if param_name in self.MIRROR_MAP:
                mirror_name = self.MIRROR_MAP[param_name]
                if mirror_name in self.env.expression_keys:
                    mirror_idx = self.env.expression_keys.index(mirror_name)
                    new_state[mirror_idx] = new_vals[i]  # Mirror the same value
        return new_state



    def _compute_returns(self, rewards):
        """
        rewards: list of float length T
        returns: list of float length T, reward-to-go with discount
        """
        G = 0.0
        rets = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            rets.append(G)
        rets.reverse()
        return rets


