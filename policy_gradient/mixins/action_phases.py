from __future__ import annotations

import numpy as np
import torch


class ActionPhasesMixin:
    def _action_phase_for(self, ep: int):
        for ph in self.action_phases:
            if "end_ep" in ph:
                try:
                    if int(ep) <= int(ph["end_ep"]):
                        return ph
                except Exception:
                    pass
        frac = float(ep) / max(1.0, float(self.episodes))
        for ph in self.action_phases:
            if frac <= float(ph.get("end_frac", 1.0)) + 1e-12:
                return ph
        return {"type": "range", "min": -0.25, "max": 0.25}

    def _action_phase_mask(self, ep: int, apply_phase: bool = True):
        if not apply_phase:
            return torch.ones_like(self.delta_values, dtype=torch.bool, device=self.device)
        ph = self._action_phase_for(ep)
        if ph.get("type", "range") == "set":
            mask = torch.zeros_like(self.delta_values, dtype=torch.bool, device=self.device)
            for v in ph.get("values", []):
                mask |= torch.isclose(self.delta_values, torch.tensor(float(v), device=self.device), atol=1e-6)
            return mask
        mn = float(ph.get("min", -0.25)); mx = float(ph.get("max", 0.25))
        return (self.delta_values >= mn) & (self.delta_values <= mx)

    def _action_phase_desc(self, ep: int) -> str:
        ph = self._action_phase_for(ep)
        if ph.get("type", "range") == "set":
            vals = ph.get('values', [])
            return f"phase=set(n={len(vals)})"
        return f"phase=range[{ph.get('min',-0.25):.2f},{ph.get('max',0.25):.2f}]"

    def _episode_exploration_prob(self, ep: int) -> float:
        """Linear decay of episode exploration probability over training."""
        horizon = max(1, self.ep_expl_decay_eps)
        k = min(1.0, (ep - 1) / horizon)
        return self.ep_expl_prob_start + k * (self.ep_expl_prob_end - self.ep_expl_prob_start)

    def _step_eps(self, step_idx: int) -> float:
        """Linear decay of epsilon within an episode: 0.90 -> 0.05 from step 0 to last step."""
        last_idx = max(0, int(self.max_steps) - 1)
        if last_idx == 0:
            return float(self.step_eps_end)
        k = float(step_idx) / float(last_idx)
        k = max(0.0, min(1.0, k))
        return float(self.step_eps_start + k * (self.step_eps_end - self.step_eps_start))
