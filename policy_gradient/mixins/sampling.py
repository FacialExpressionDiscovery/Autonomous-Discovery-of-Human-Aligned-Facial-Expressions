from __future__ import annotations

import random

import numpy as np
import torch
from torch.distributions import Categorical


class SamplingMixin:
    def _logprob_of_choice_masked_mix(self, s_full_np, chosen_idxs, aux):
        x_jit_np, eps, M_count = aux["s_jit"], float(aux["eps"]), aux["valid_counts"]
        zero_idx = (self.K - 1)//2
        idxs = [(zero_idx if x is None else int(x)) for x in chosen_idxs]
        x_t = torch.as_tensor(x_jit_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits_all = self.policy(x_t)[0]  # [N,K]
        mask_phase = self._action_phase_mask(self.current_episode, apply_phase=True)  # training always applies

        constrained = [n for n in self._constraint_order() if n in self.selected_params]
        remaining   = [n for n in self.selected_params if n not in constrained]
        seq = constrained + remaining

        planned = np.zeros(self.obs_dim, dtype=np.float32)
        def cur_val(name):
            si = self._idx_of(name)
            if si is not None: return float(x_jit_np[si] + planned[si])
            return float(s_full_np[self.name_to_idx[name]]) if name in self.name_to_idx else 0.0
        def vect_for_current(cur_name, cand_vec):
            out = {}
            keys = [
                "MOUTH_SHRUG_UPPER",
                "MOUTH_SHRUG_LOWER",
                "MOUTH_UPPER_UP_LEFT",
                "MOUTH_LOWER_DOWN_LEFT",
                "MOUTH_UPPER_UP_RIGHT",
                "MOUTH_LOWER_DOWN_RIGHT",
                "MOUTH_ROLL_UPPER",
                "MOUTH_ROLL_LOWER",
            ]
            for key in keys:
                si = self._idx_of(key)
                if key == cur_name and si is not None:
                    out[key] = cand_vec
                else:
                    v = cur_val(key)
                    out[key] = torch.full_like(self.delta_values, fill_value=v)
            return out

        logp_sum = torch.tensor(0.0, device=self.device)
        for name in seq:
            j = self._idx_of(name)
            if j is None: continue
            base = float(x_jit_np[j])
            raw_logits = logits_all[j]
            final_vals = base + self.delta_values
            mask_range = (final_vals >= 0.0) & (final_vals <= 1.0)

            apply_h1 = name in ("MOUTH_SHRUG_UPPER","MOUTH_SHRUG_LOWER")
            apply_h2l = name in ("MOUTH_UPPER_UP_LEFT", "MOUTH_LOWER_DOWN_LEFT")
            apply_h2r = name in ("MOUTH_UPPER_UP_RIGHT", "MOUTH_LOWER_DOWN_RIGHT")
            apply_h2 = apply_h2l or apply_h2r
            apply_h3 = name in ("MOUTH_ROLL_UPPER","MOUTH_ROLL_LOWER")
            apply_eye_squint_left = name == "EYE_SQUINT_LEFT"
            apply_eye_blink_left  = name == "EYE_BLINK_LEFT"
            apply_eye_left = apply_eye_squint_left or apply_eye_blink_left
            if apply_h1 or apply_h2 or apply_h3:
                vecs = vect_for_current(name, base + self.delta_values)
                su, sl  = vecs["MOUTH_SHRUG_UPPER"],  vecs["MOUTH_SHRUG_LOWER"]
                uul, ldl  = vecs["MOUTH_UPPER_UP_LEFT"], vecs["MOUTH_LOWER_DOWN_LEFT"]
                uur, ldr  = vecs["MOUTH_UPPER_UP_RIGHT"], vecs["MOUTH_LOWER_DOWN_RIGHT"]
                ru, rl  = vecs["MOUTH_ROLL_UPPER"],   vecs["MOUTH_ROLL_LOWER"]
                mask_h1  = (su  - sl)  >= 0.07 if apply_h1  else torch.ones_like(mask_range)
                mask_h2l = (uul - ldl) >= 0.07 if apply_h2l else torch.ones_like(mask_range)
                mask_h2r = (uur - ldr) >= 0.07 if apply_h2r else torch.ones_like(mask_range)
                mask_h3  = (ru  - rl)  >= 0.07 if apply_h3  else torch.ones_like(mask_range)
                mask_all = mask_range & mask_h1 & mask_h2l & mask_h2r & mask_h3
            elif apply_eye_left:
                cand = base + self.delta_values
                squint_v = cand if apply_eye_squint_left else torch.full_like(cand, cur_val("EYE_SQUINT_LEFT"))
                blink_v  = cand if apply_eye_blink_left  else torch.full_like(cand, cur_val("EYE_BLINK_LEFT"))
                mask_squint = squint_v <= 0.60
                mask_blink  = blink_v  <= 0.30
                mask_sum    = (squint_v / 0.60 + blink_v / 0.30) <= 0.55
                mask_all = mask_range & mask_squint & mask_blink & mask_sum
            else:
                mask_all = mask_range

            mask_phase = self._action_phase_mask(self.current_episode, apply_phase=True) if j == 0 else mask_phase  # compute once
            valid_mask = mask_all & mask_phase

            masked_logits = raw_logits.clone()
            if valid_mask.any():
                masked_logits[~valid_mask] = -1e9
                dist = Categorical(logits=masked_logits)
            else:
                # fallback matched to sampling fallback
                masked_logits[~mask_range] = -1e9
                dist = Categorical(logits=masked_logits)

            probs = dist.probs
            idx = torch.as_tensor(int(idxs[j]), dtype=torch.long, device=self.device)

            M = max(1, int(valid_mask.sum().item()))  # consistent with sampling
            uni = 1.0 / float(M)
            qi = (1.0 - eps) * probs[idx] + eps * uni
            logp_sum = logp_sum + torch.log(qi.clamp_min(1e-12))

            planned[j] = float(self.delta_values[idx].item())
        return logp_sum


    def _constraint_order(self):
        # sample upper-side first, then lower-side; squint before blink for eye sum constraint
        return [
            "MOUTH_SHRUG_UPPER",
            "MOUTH_UPPER_UP_LEFT",
            "MOUTH_UPPER_UP_RIGHT",
            "MOUTH_ROLL_UPPER",
            "MOUTH_LOWER_DOWN_LEFT",
            "MOUTH_LOWER_DOWN_RIGHT",
            "MOUTH_ROLL_LOWER",
            "MOUTH_SHRUG_LOWER",
            "EYE_SQUINT_LEFT",
            "EYE_BLINK_LEFT",
        ]

    def _idx_of(self, name):
        try:
            return self.selected_params.index(name)
        except ValueError:
            return None
    
    def _sample_action(self, s_full_np, step_idx=None, exploration_on=False, deterministic=False,  apply_action_phase=True):
        """
        s_full_np: np.ndarray [D] float32
        Returns:
        deltas_np:  np.ndarray [N] float32
        chosen_idxs: list[int] length N
        aux: dict with:
            's_jit': np.ndarray [N] float32 (slice used to sample)
            'T': float (effective temperature used)
            'eps': float (epsilon used)
            'valid_counts': list[int] length N (#valid buckets each dim)
            'mean_entropy': float (mean per-dim entropy under temp-softmax, not mixture)
        """
        N = self.obs_dim
        x_np = s_full_np[self.param_indices]  # [N], float32
        ep = getattr(self, "current_episode", 1)

        base_eps = self._step_eps(step_idx or 0)

        if self.use_episode_expl:
            use_eps_mix = bool(exploration_on)
            eps = base_eps if exploration_on else 0.0
        else:
            use_eps_mix = self.expl_use_eps_mix
            eps = base_eps if self.expl_use_eps_mix else 0.0

        # jittered slice for sampling
        x_t = torch.as_tensor(x_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1,N]
        x_jit_np = x_t.squeeze(0).detach().cpu().numpy().astype(np.float32)  # [N]

        with torch.no_grad():
            logits_all = self.policy(x_t)[0].clone()  # [N,K]
        
        # action-phase mask is shared across dims (per-episode)
        mask_phase = self._action_phase_mask(self.current_episode, apply_phase=apply_action_phase)  # [K] bool
 

        planned  = np.zeros(N, dtype=np.float32)
        chosen   = [None]*N
        M_count  = [0]*N
        entropies = []  # per-dim entropy under temp-softmax (masked)

        zero_idx = int((self.K - 1)//2)

        constrained = [n for n in self._constraint_order() if n in self.selected_params]
        remaining   = [n for n in self.selected_params if n not in constrained]
        seq = constrained + remaining

        def cur_val(name):
            si = self._idx_of(name)
            if si is not None: return float(x_jit_np[si] + planned[si])
            return float(s_full_np[self.name_to_idx[name]]) if name in self.name_to_idx else 0.0

        def vect_for_current(cur_name, cand_vec):
            out = {}
            keys = [
                "MOUTH_SHRUG_UPPER",
                "MOUTH_SHRUG_LOWER",
                "MOUTH_UPPER_UP_LEFT",
                "MOUTH_LOWER_DOWN_LEFT",
                "MOUTH_UPPER_UP_RIGHT",
                "MOUTH_LOWER_DOWN_RIGHT",
                "MOUTH_ROLL_UPPER",
                "MOUTH_ROLL_LOWER",
            ]
            for key in keys:
                si = self._idx_of(key)
                if key == cur_name and si is not None:
                    out[key] = cand_vec
                else:
                    v = cur_val(key)
                    out[key] = torch.full_like(self.delta_values, fill_value=v)
            return out

        for name in seq:
            j = self._idx_of(name)
            if j is None: continue

            base = float(x_jit_np[j])
            raw_logits = logits_all[j].clone()                       # [K]
            final_vals = base + self.delta_values                    # [K]
            mask_range = (final_vals >= 0.0) & (final_vals <= 1.0)

            apply_h1 = name in ("MOUTH_SHRUG_UPPER","MOUTH_SHRUG_LOWER")
            apply_h2l = name in ("MOUTH_UPPER_UP_LEFT", "MOUTH_LOWER_DOWN_LEFT")
            apply_h2r = name in ("MOUTH_UPPER_UP_RIGHT", "MOUTH_LOWER_DOWN_RIGHT")
            apply_h2 = apply_h2l or apply_h2r
            apply_h3 = name in ("MOUTH_ROLL_UPPER","MOUTH_ROLL_LOWER")
            apply_eye_squint_left = name == "EYE_SQUINT_LEFT"
            apply_eye_blink_left  = name == "EYE_BLINK_LEFT"
            apply_eye_left = apply_eye_squint_left or apply_eye_blink_left

            if apply_h1 or apply_h2 or apply_h3:
                vecs = vect_for_current(name, base + self.delta_values)
                su, sl  = vecs["MOUTH_SHRUG_UPPER"],  vecs["MOUTH_SHRUG_LOWER"]
                uul, ldl  = vecs["MOUTH_UPPER_UP_LEFT"], vecs["MOUTH_LOWER_DOWN_LEFT"]
                uur, ldr  = vecs["MOUTH_UPPER_UP_RIGHT"], vecs["MOUTH_LOWER_DOWN_RIGHT"]
                ru, rl  = vecs["MOUTH_ROLL_UPPER"],   vecs["MOUTH_ROLL_LOWER"]
                mask_h1  = (su  - sl)  >= 0.07 if apply_h1  else torch.ones_like(mask_range, dtype=torch.bool)
                mask_h2l = (uul - ldl) >= 0.07 if apply_h2l else torch.ones_like(mask_range, dtype=torch.bool)
                mask_h2r = (uur - ldr) >= 0.07 if apply_h2r else torch.ones_like(mask_range, dtype=torch.bool)
                mask_h3  = (ru  - rl)  >= 0.07 if apply_h3  else torch.ones_like(mask_range, dtype=torch.bool)
                mask_all = mask_range & mask_h1 & mask_h2l & mask_h2r & mask_h3
            elif apply_eye_left:
                cand = base + self.delta_values
                squint_v = cand if apply_eye_squint_left else torch.full_like(cand, cur_val("EYE_SQUINT_LEFT"))
                blink_v  = cand if apply_eye_blink_left  else torch.full_like(cand, cur_val("EYE_BLINK_LEFT"))
                mask_squint = squint_v <= 0.60
                mask_blink  = blink_v  <= 0.30
                mask_sum    = (squint_v / 0.60 + blink_v / 0.30) <= 0.55
                mask_all = mask_range & mask_squint & mask_blink & mask_sum
            else:
                mask_all = mask_range

            masked_logits = raw_logits.clone()
            # Enforce both geometry/range and action-phase constraints in logits
            valid_mask = mask_all & mask_phase
            masked_logits[~valid_mask] = -1e9
            M = int(valid_mask.sum().item())
            M_count[j] = M
            #M = int(mask_all.sum().item())

            if M == 0:
                # No valid actions under phase+geometry. Prefer zero if allowed under both.
                zero_ok = bool((mask_range & mask_phase)[zero_idx].item())
                if zero_ok:
                    idx = zero_idx
                else:
                    # Try argmax within phase+geometry; if none finite, fallback to geometry only.
                    logits_valid = raw_logits.clone()
                    logits_valid[~(mask_all & mask_phase)] = -1e9
                    if torch.isfinite(logits_valid).any():
                        idx = int(torch.argmax(logits_valid).item())
                    else:
                        logits_range = raw_logits.clone()
                        logits_range[~mask_range] = -1e9
                        idx = int(torch.argmax(logits_range).item())
                # entropy undefined here; treat as zero
                entropies.append(0.0)

            else:
                if deterministic:
                    idx = int(torch.argmax(masked_logits).item())
                else:
                    if use_eps_mix and (random.random() < eps):
                        valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)  # [M]
                        pick = int(torch.randint(M, (1,)).item())  # 0..M-1
                        idx = int(valid_idx[pick].item())
                    else:
                        dist = Categorical(logits=masked_logits)
                        idx = int(dist.sample().item())


            chosen[j]  = idx
            planned[j] = float(self.delta_values[idx].item())

        for i in range(N):
            if chosen[i] is None:
                chosen[i] = zero_idx
                planned[i] = 0.0

        mean_entropy = float(np.mean(entropies)) if len(entropies) else 0.0

        aux = {
            "s_jit": x_jit_np,                   # np.ndarray [N], float32
            "T": 1.0,        # scalar float
            "eps": float(eps),                   # scalar float
            "valid_counts": M_count,             # list[int] length N
            "mean_entropy": mean_entropy         # float
        }
        return planned.astype(np.float32), [int(x) for x in chosen], aux


