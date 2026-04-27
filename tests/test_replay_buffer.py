"""Unit tests for ReplayBuffer."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from replay_buffer import ReplayBuffer


def _push_n(buf: ReplayBuffer, n: int) -> None:
    for _ in range(n):
        s  = np.random.randn(4).astype(np.float32)
        ns = np.random.randn(4).astype(np.float32)
        buf.push(s, action=1, reward=0.5, next_state=ns, done=0.0)


def test_len_grows_with_pushes():
    buf = ReplayBuffer(capacity=100)
    assert len(buf) == 0
    _push_n(buf, 10)
    assert len(buf) == 10


def test_capacity_capped():
    buf = ReplayBuffer(capacity=20)
    _push_n(buf, 50)
    assert len(buf) == 20


def test_sample_returns_correct_shapes():
    buf = ReplayBuffer(capacity=100)
    _push_n(buf, 32)
    states, actions, rewards, next_states, dones = buf.sample(16)
    assert states.shape      == (16, 4)
    assert actions.shape     == (16,)
    assert rewards.shape     == (16,)
    assert next_states.shape == (16, 4)
    assert dones.shape       == (16,)


def test_sample_returns_float_tensors():
    buf = ReplayBuffer(capacity=100)
    _push_n(buf, 32)
    states, _, rewards, next_states, dones = buf.sample(8)
    assert states.dtype      == torch.float32
    assert rewards.dtype     == torch.float32
    assert next_states.dtype == torch.float32
    assert dones.dtype       == torch.float32


def test_sample_actions_are_long():
    buf = ReplayBuffer(capacity=100)
    _push_n(buf, 32)
    _, actions, _, _, _ = buf.sample(8)
    assert actions.dtype == torch.int64


def test_sample_raises_when_too_few_transitions():
    buf = ReplayBuffer(capacity=100)
    _push_n(buf, 5)
    with pytest.raises(ValueError):
        buf.sample(10)


def test_done_flag_stored_correctly():
    buf = ReplayBuffer(capacity=10)
    s  = np.zeros(4, dtype=np.float32)
    ns = np.zeros(4, dtype=np.float32)
    buf.push(s, 0, 0.0, ns, done=1.0)
    _, _, _, _, dones = buf.sample(1)
    assert dones[0].item() == pytest.approx(1.0)
