"""Unit tests for tabular Q-learning helpers."""
from __future__ import annotations

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from q_learning import discretize_state, split_data, buy_and_hold


PRICES = np.linspace(100, 200, 100)


# ── discretize_state ─────────────────────────────────────────────────────────

def test_discretize_returns_valid_state_index():
    for pos in (-1, 0, 1):
        idx = discretize_state(PRICES, t=50, position=pos)
        assert 0 <= idx < 12, f"State index {idx} out of range for position={pos}"


def test_discretize_early_t_uses_neutral_bucket():
    # t < 20 → neutral bucket (indices 6-8)
    idx = discretize_state(PRICES, t=5, position=0)
    assert 6 <= idx <= 8


def test_discretize_uptrend_long_position():
    # Uptrend → 5d and 20d return both positive → s0=1, s1=1
    prices = np.linspace(100, 200, 100)
    idx = discretize_state(prices, t=50, position=1)   # long = s2=2
    expected = 1 * 6 + 1 * 3 + 2
    assert idx == expected


def test_discretize_position_mapping():
    prices = np.linspace(100, 200, 100)
    idx_short = discretize_state(prices, t=50, position=-1)
    idx_flat  = discretize_state(prices, t=50, position=0)
    idx_long  = discretize_state(prices, t=50, position=1)
    # Short < Flat < Long within the same return-sign bucket
    assert idx_short < idx_flat < idx_long


# ── split_data ────────────────────────────────────────────────────────────────

def test_split_default_70_30():
    train, test = split_data(PRICES)
    assert len(train) == 70
    assert len(test)  == 30


def test_split_no_overlap():
    train, test = split_data(PRICES)
    assert len(np.intersect1d(train, test)) == 0


def test_split_custom_ratio():
    train, test = split_data(PRICES, train_ratio=0.8)
    assert len(train) == 80
    assert len(test)  == 20


def test_split_preserves_order():
    train, test = split_data(PRICES)
    assert train[-1] < test[0], "Train prices should come before test prices"


# ── buy_and_hold ──────────────────────────────────────────────────────────────

def test_buy_and_hold_length():
    bnh = buy_and_hold(PRICES)
    assert len(bnh) == len(PRICES) - 1


def test_buy_and_hold_uptrend_is_positive():
    bnh = buy_and_hold(PRICES)
    assert bnh[-1] > 0


def test_buy_and_hold_is_cumulative():
    prices = np.array([100.0, 101.0, 103.0, 100.0])
    bnh = buy_and_hold(prices)
    expected = np.cumsum(np.diff(prices))
    np.testing.assert_allclose(bnh, expected)
