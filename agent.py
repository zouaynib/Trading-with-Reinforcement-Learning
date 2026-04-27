from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import QNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim:     int,
        n_actions:     int   = 3,
        lr:            float = 1e-3,
        gamma:         float = 0.99,
        buffer_size:   int   = 10_000,
        batch_size:    int   = 64,
        target_update: int   = 100,
    ) -> None:
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self.step_count    = 0

        self.q_net      = QNetwork(state_dim, n_actions)
        self.target_net = QNetwork(state_dim, n_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            q = self.q_net(torch.FloatTensor(state).unsqueeze(0))
            return int(q.argmax().item())

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, float(done))

    def update(self) -> None:
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1).values
            td_target  = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


if __name__ == "__main__":
    agent = DQNAgent(state_dim=22)

    for _ in range(200):
        state      = np.random.randn(22).astype(np.float32)
        action     = agent.select_action(state, epsilon=0.5)
        next_state = np.random.randn(22).astype(np.float32)
        agent.push(state, action, reward=0.1, next_state=next_state, done=False)
        agent.update()

    print("Buffer size :", len(agent.buffer))
    print("Steps taken :", agent.step_count)
    print("Target syncs:", agent.step_count // agent.target_update)

    state  = np.random.randn(22).astype(np.float32)
    action = agent.select_action(state, epsilon=0.0)
    print("Greedy action:", action, "(should be 0, 1, or 2)")
    print("agent.py OK")
