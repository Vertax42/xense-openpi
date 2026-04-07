"""Off-policy replay buffer for RLT.

Stores chunk-level transitions for TD3 training. Each transition represents
one action chunk execution (C timesteps) with the associated RL token state,
proprioceptive state, reference actions, and reward.

Per-step rewards are stored as [C] to enable proper within-chunk discounting:
    Q̂ = Σ_{l'=1}^{C} γ^{l'-1} r_{l'} + γ^C E_{a'~π_θ}[Q_ψ'(x', a')]

next_ref_actions stores the VLA reference chunk at x_{t+1}, needed to condition
the target actor correctly when computing the TD target.
"""

import torch
from torch import Tensor


class ReplayBuffer:
    """Pre-allocated CPU replay buffer for chunk-level transitions."""

    def __init__(self, config, rl_token_dim: int, proprio_dim: int, action_dim: int, action_chunk: int):
        self.capacity = config.capacity
        self.rl_token_dim = rl_token_dim
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.action_chunk = action_chunk

        # Pre-allocate CPU tensors
        self.z_rl = torch.zeros(self.capacity, rl_token_dim)
        self.proprio = torch.zeros(self.capacity, proprio_dim)
        self.actions = torch.zeros(self.capacity, action_chunk, action_dim)
        self.ref_actions = torch.zeros(self.capacity, action_chunk, action_dim)
        # Per-step rewards [capacity, C] for within-chunk discounting (paper Eq. 3)
        self.rewards = torch.zeros(self.capacity, action_chunk)
        self.next_z_rl = torch.zeros(self.capacity, rl_token_dim)
        self.next_proprio = torch.zeros(self.capacity, proprio_dim)
        # VLA reference chunk at next state, used to condition target actor
        self.next_ref_actions = torch.zeros(self.capacity, action_chunk, action_dim)
        self.dones = torch.zeros(self.capacity, 1)

        self._size = 0
        self._ptr = 0

    def add(
        self,
        z_rl: Tensor,
        proprio: Tensor,
        actions: Tensor,
        ref_actions: Tensor,
        rewards: Tensor,
        next_z_rl: Tensor,
        next_proprio: Tensor,
        next_ref_actions: Tensor,
        dones: Tensor,
    ):
        """Add a single transition to the buffer.

        Args:
            z_rl: [D] or [B, D] RL token at current state
            proprio: [p] or [B, p] proprioceptive state
            actions: [C, d] or [B, C, d] executed action chunk
            ref_actions: [C, d] or [B, C, d] VLA reference chunk at current state
            rewards: [C] or [B, C] per-step rewards within the chunk
            next_z_rl: [D] or [B, D] RL token at next state
            next_proprio: [p] or [B, p] proprioceptive state at next state
            next_ref_actions: [C, d] or [B, C, d] VLA reference chunk at next state
            dones: scalar or [B, 1] episode termination flag
        """
        if z_rl.dim() == 1:
            self.z_rl[self._ptr] = z_rl.cpu()
            self.proprio[self._ptr] = proprio.cpu()
            self.actions[self._ptr] = actions.cpu()
            self.ref_actions[self._ptr] = ref_actions.cpu()
            self.rewards[self._ptr] = rewards.cpu()
            self.next_z_rl[self._ptr] = next_z_rl.cpu()
            self.next_proprio[self._ptr] = next_proprio.cpu()
            self.next_ref_actions[self._ptr] = next_ref_actions.cpu()
            self.dones[self._ptr] = dones.cpu()

            self._ptr = (self._ptr + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
        else:
            batch_size = z_rl.shape[0]
            for i in range(batch_size):
                self.z_rl[self._ptr] = z_rl[i].cpu()
                self.proprio[self._ptr] = proprio[i].cpu()
                self.actions[self._ptr] = actions[i].cpu()
                self.ref_actions[self._ptr] = ref_actions[i].cpu()
                self.rewards[self._ptr] = rewards[i].cpu()
                self.next_z_rl[self._ptr] = next_z_rl[i].cpu()
                self.next_proprio[self._ptr] = next_proprio[i].cpu()
                self.next_ref_actions[self._ptr] = next_ref_actions[i].cpu()
                self.dones[self._ptr] = dones[i].cpu()

                self._ptr = (self._ptr + 1) % self.capacity
                self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, Tensor]:
        """Sample a random batch from the buffer.

        Args:
            batch_size: Number of transitions to sample
            device: Target device for tensors

        Returns:
            Dictionary of batched tensors on the target device.
            'rewards' has shape [B, C] (per-step, not pre-summed).
        """
        indices = torch.randint(0, self._size, (batch_size,))
        return {
            "z_rl": self.z_rl[indices].to(device),
            "proprio": self.proprio[indices].to(device),
            "actions": self.actions[indices].to(device),
            "ref_actions": self.ref_actions[indices].to(device),
            "rewards": self.rewards[indices].to(device),
            "next_z_rl": self.next_z_rl[indices].to(device),
            "next_proprio": self.next_proprio[indices].to(device),
            "next_ref_actions": self.next_ref_actions[indices].to(device),
            "dones": self.dones[indices].to(device),
        }

    def __len__(self) -> int:
        return self._size
