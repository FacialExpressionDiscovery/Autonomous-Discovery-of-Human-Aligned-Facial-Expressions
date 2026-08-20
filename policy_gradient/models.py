from __future__ import annotations

import torch.nn as nn


# Factorized discrete policy: per-parameter K-way categorical over deltas in [-0.25, +0.25]
class FactorizedDiscretePolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions_per_dim: int = 51, hidden=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions_per_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim * n_actions_per_dim),
        )

    def forward(self, obs):  # obs: [B, obs_dim]
        logits = self.net(obs)  # [B, obs_dim * K]
        B = logits.shape[0]
        logits = logits.view(B, self.obs_dim, self.n_actions)  # [B, obs_dim, K]
        return logits


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        """
        obs: [B, obs_dim] float32
        return: [B] float32 (state value)
        """
        v = self.net(obs)  # [B, 1]
        return v.squeeze(-1)  # [B]

