from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """
    Circular buffer storing (state, action, reward, next_state, done) transitions.
    Randomly samples mini-batches to break temporal correlations during training.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)


if __name__ == "__main__":
    buf = ReplayBuffer(capacity=100)

    for i in range(50):
        state      = np.random.randn(22).astype(np.float32)
        next_state = np.random.randn(22).astype(np.float32)
        buf.push(state, action=1, reward=0.5, next_state=next_state, done=0.0)

    print("Buffer size:", len(buf))

    states, actions, rewards, next_states, dones = buf.sample(batch_size=8)
    print("states shape    :", states.shape)
    print("actions shape   :", actions.shape)
    print("rewards shape   :", rewards.shape)
    print("next_states shape:", next_states.shape)
    print("dones shape     :", dones.shape)
    print("replay_buffer.py OK")
