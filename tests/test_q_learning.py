"""Unit tests for tabular Q-learning helpers."""
from __future__ import annotations

import numpy as np
import pytest

from q_learning import (
    discretize_state,
    split_data,
    buy_and_hold,
    compute_metrics,
    evaluate_policy,
    train_q_learning,
    print_q_table,
)


PRICES = np.linspace(100, 200, 100)


# ── discretize_state ─────────────────────────────────────────────────────────

def test_discretize_returns_valid_state_index():
    for pos in (-1, 0, 1):
        idx = discretize_state(PRICES, t=50, position=pos)
        assert 0 <= idx < 12, f"State index {idx} out of range for position={pos}"


def test_discretize_early_t_uses_neutral_bucket():
    idx = discretize_state(PRICES, t=5, position=0)
    assert 6 <= idx <= 8


def test_discretize_uptrend_long_position():
    prices = np.linspace(100, 200, 100)
    idx = discretize_state(prices, t=50, position=1)
    expected = 1 * 6 + 1 * 3 + 2
    assert idx == expected


def test_discretize_position_mapping():
    prices = np.linspace(100, 200, 100)
    idx_short = discretize_state(prices, t=50, position=-1)
    idx_flat  = discretize_state(prices, t=50, position=0)
    idx_long  = discretize_state(prices, t=50, position=1)
    assert idx_short < idx_flat < idx_long


def test_discretize_downtrend():
    prices = np.linspace(200, 100, 100)
    idx = discretize_state(prices, t=50, position=0)
    # 5d and 20d returns both negative → s0=0, s1=0, s2=1 → index = 1
    assert idx == 0 * 6 + 0 * 3 + 1


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


# ── compute_metrics ───────────────────────────────────────────────────────────

def test_compute_metrics_returns_all_keys():
    pnl = np.array([0.0, 10.0, 5.0, 20.0, 15.0])
    m = compute_metrics(pnl, n_trades=3, label="test")
    assert set(m.keys()) == {"total_return", "sharpe", "mdd", "n_trades"}


def test_compute_metrics_total_return_is_last_value():
    pnl = np.array([1.0, 2.0, 3.0, 42.0])
    m = compute_metrics(pnl, n_trades=1, label="")
    assert m["total_return"] == pytest.approx(42.0)


def test_compute_metrics_mdd_is_non_positive():
    pnl = np.array([10.0, 20.0, 5.0, 15.0])
    m = compute_metrics(pnl, n_trades=0, label="")
    assert m["mdd"] <= 0


def test_compute_metrics_monotone_pnl_has_zero_drawdown():
    pnl = np.linspace(0, 100, 50)
    m = compute_metrics(pnl, n_trades=1, label="")
    assert m["mdd"] == pytest.approx(0.0, abs=1e-6)


def test_compute_metrics_sharpe_positive_in_uptrend():
    pnl = np.linspace(0, 100, 252)
    m = compute_metrics(pnl, n_trades=1, label="")
    assert m["sharpe"] > 0


# ── evaluate_policy ───────────────────────────────────────────────────────────

def test_evaluate_policy_returns_metrics_and_pnl():
    Q = np.zeros((12, 3))          # always hold
    metrics, pnl = evaluate_policy(PRICES, Q, label="test")
    assert isinstance(pnl, np.ndarray)
    assert len(pnl) > 0
    assert "total_return" in metrics


def test_evaluate_policy_all_hold_earns_zero():
    Q = np.zeros((12, 3))
    Q[:, 0] = 1.0                  # strongly prefer hold from any state
    _, pnl = evaluate_policy(PRICES, Q, label="hold-only")
    assert pnl[-1] == pytest.approx(0.0, abs=1e-6)


def test_evaluate_policy_pnl_length_matches_env_steps():
    Q = np.zeros((12, 3))
    _, pnl = evaluate_policy(PRICES, Q, label="")
    # env runs from window-1 to n_steps-1
    assert len(pnl) == len(PRICES) - 20   # window=20


# ── train_q_learning ─────────────────────────────────────────────────────────

def test_train_q_learning_returns_qtable_shape():
    Q, rewards = train_q_learning(PRICES, n_episodes=3, verbose=False)
    assert Q.shape == (12, 3)


def test_train_q_learning_returns_episode_rewards():
    _, rewards = train_q_learning(PRICES, n_episodes=5, verbose=False)
    assert len(rewards) == 5
    assert all(isinstance(r, float) for r in rewards)


def test_train_q_learning_qtable_changes_from_zero():
    Q, _ = train_q_learning(PRICES, n_episodes=10, verbose=False)
    assert not np.all(Q == 0), "Q-table should be updated from initial zeros"


# ── print_q_table ─────────────────────────────────────────────────────────────

def test_print_q_table_runs_without_error():
    Q = np.random.randn(12, 3)
    print_q_table(Q)   # should not raise
