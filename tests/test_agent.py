"""Unit tests for DQNAgent."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from agent import DQNAgent


STATE_DIM = 22


def make_agent() -> DQNAgent:
    return DQNAgent(state_dim=STATE_DIM)


def random_state() -> np.ndarray:
    return np.random.randn(STATE_DIM).astype(np.float32)


# ── select_action ─────────────────────────────────────────────────────────────

def test_select_action_valid_range():
    agent = make_agent()
    for _ in range(50):
        a = agent.select_action(random_state(), epsilon=0.5)
        assert a in (0, 1, 2)


def test_greedy_action_deterministic():
    torch.manual_seed(0)
    agent = make_agent()
    state = random_state()
    a1 = agent.select_action(state, epsilon=0.0)
    a2 = agent.select_action(state, epsilon=0.0)
    assert a1 == a2


def test_random_action_at_epsilon_one():
    """At ε=1 every action is random — check all three are reachable."""
    np.random.seed(42)
    agent = make_agent()
    actions = {agent.select_action(random_state(), epsilon=1.0) for _ in range(200)}
    assert actions == {0, 1, 2}


# ── push & buffer ─────────────────────────────────────────────────────────────

def test_push_increases_buffer():
    agent = make_agent()
    assert len(agent.buffer) == 0
    agent.push(random_state(), 1, 0.5, random_state(), False)
    assert len(agent.buffer) == 1


# ── update ────────────────────────────────────────────────────────────────────

def test_update_noop_when_buffer_too_small():
    agent = make_agent()
    agent.push(random_state(), 0, 0.0, random_state(), False)
    agent.update()   # should silently return — buffer < batch_size
    assert agent.step_count == 0


def test_update_runs_gradient_step():
    agent = make_agent()
    for _ in range(agent.batch_size + 10):
        agent.push(random_state(), 1, 1.0, random_state(), False)
    agent.update()
    assert agent.step_count == 1


def test_target_network_synced_every_c_steps():
    agent = make_agent()
    for _ in range(agent.batch_size + 10):
        agent.push(random_state(), 0, 0.0, random_state(), False)

    # Run exactly target_update steps
    for _ in range(agent.target_update):
        agent.update()

    # After sync, online and target weights should be identical
    for p_online, p_target in zip(
        agent.q_net.parameters(), agent.target_net.parameters()
    ):
        assert torch.allclose(p_online, p_target)


def test_q_net_output_shape():
    agent = make_agent()
    state = torch.FloatTensor(random_state()).unsqueeze(0)
    q_vals = agent.q_net(state)
    assert q_vals.shape == (1, 3)
