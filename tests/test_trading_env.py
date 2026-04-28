"""Unit tests for TradingEnv."""
from __future__ import annotations

import numpy as np
import pytest

from trading_env import TradingEnv, run_random_episodes


PRICES = np.linspace(100, 200, 60)  # 60 synthetic days, steady uptrend


def make_env() -> TradingEnv:
    return TradingEnv(PRICES, window=20, cost=0.001)


# ── reset ────────────────────────────────────────────────────────────────────

def test_reset_returns_correct_shape():
    env = make_env()
    state = env.reset()
    assert state.shape == (env.state_dim,), f"Expected ({env.state_dim},), got {state.shape}"


def test_reset_starts_flat():
    env = make_env()
    env.reset()
    assert env.position == 0
    assert env.entry_price is None


def test_state_dim_equals_window_plus_two():
    env = TradingEnv(PRICES, window=10)
    assert env.state_dim == 12


def test_reset_is_idempotent():
    env = make_env()
    s1 = env.reset()
    env.step(TradingEnv.BUY)
    s2 = env.reset()
    np.testing.assert_array_equal(s1, s2)


# ── step: position transitions ───────────────────────────────────────────────

def test_buy_from_flat_goes_long():
    env = make_env()
    env.reset()
    env.step(TradingEnv.BUY)
    assert env.position == +1


def test_sell_from_flat_goes_short():
    env = make_env()
    env.reset()
    env.step(TradingEnv.SELL)
    assert env.position == -1


def test_sell_from_long_goes_flat():
    env = make_env()
    env.reset()
    env.step(TradingEnv.BUY)
    env.step(TradingEnv.SELL)
    assert env.position == 0


def test_buy_from_short_goes_flat():
    env = make_env()
    env.reset()
    env.step(TradingEnv.SELL)
    env.step(TradingEnv.BUY)
    assert env.position == 0


def test_hold_does_not_change_long_position():
    env = make_env()
    env.reset()
    env.step(TradingEnv.BUY)
    env.step(TradingEnv.HOLD)
    assert env.position == +1


def test_hold_does_not_change_short_position():
    env = make_env()
    env.reset()
    env.step(TradingEnv.SELL)
    env.step(TradingEnv.HOLD)
    assert env.position == -1


def test_buy_while_long_stays_long():
    env = make_env()
    env.reset()
    env.step(TradingEnv.BUY)
    env.step(TradingEnv.BUY)   # redundant buy — position unchanged
    assert env.position == +1


def test_sell_while_short_stays_short():
    env = make_env()
    env.reset()
    env.step(TradingEnv.SELL)
    env.step(TradingEnv.SELL)  # redundant sell — position unchanged
    assert env.position == -1


# ── step: entry price ────────────────────────────────────────────────────────

def test_entry_price_set_on_buy():
    env = make_env()
    env.reset()
    p = env.prices[env.t]
    env.step(TradingEnv.BUY)
    assert env.entry_price == pytest.approx(p)


def test_entry_price_cleared_on_close():
    env = make_env()
    env.reset()
    env.step(TradingEnv.BUY)
    env.step(TradingEnv.SELL)
    assert env.entry_price is None


# ── step: reward ─────────────────────────────────────────────────────────────

def test_flat_position_earns_zero_reward_on_hold():
    env = make_env()
    env.reset()
    _, reward, _, _ = env.step(TradingEnv.HOLD)
    assert reward == pytest.approx(0.0)


def test_long_position_earns_positive_reward_in_uptrend():
    env = make_env()
    env.reset()
    env.step(TradingEnv.BUY)
    _, reward, _, _ = env.step(TradingEnv.HOLD)
    assert reward > 0, "Long position should profit in uptrend"


def test_short_position_loses_in_uptrend():
    env = make_env()
    env.reset()
    env.step(TradingEnv.SELL)
    _, reward, _, _ = env.step(TradingEnv.HOLD)
    assert reward < 0, "Short position should lose in uptrend"


def test_transaction_cost_applied_on_trade():
    env = make_env()
    env.reset()
    p_now = env.prices[env.t]
    _, _, _, info = env.step(TradingEnv.BUY)
    assert info["transaction_cost"] == pytest.approx(env.cost * p_now)


def test_no_transaction_cost_on_hold():
    env = make_env()
    env.reset()
    _, _, _, info = env.step(TradingEnv.HOLD)
    assert info["transaction_cost"] == pytest.approx(0.0)


# ── episode termination ───────────────────────────────────────────────────────

def test_episode_terminates():
    env = make_env()
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(TradingEnv.HOLD)
        steps += 1
    assert steps > 0
    assert done is True


def test_done_state_is_zeros():
    env = make_env()
    env.reset()
    last_state = None
    done = False
    while not done:
        last_state, _, done, _ = env.step(TradingEnv.HOLD)
    assert np.all(last_state == 0)


# ── invalid action ────────────────────────────────────────────────────────────

def test_invalid_action_raises():
    env = make_env()
    env.reset()
    with pytest.raises(AssertionError):
        env.step(99)


# ── run_random_episodes ───────────────────────────────────────────────────────

def test_run_random_episodes_count():
    env = make_env()
    rewards = run_random_episodes(env, n_episodes=10)
    assert len(rewards) == 10


def test_run_random_episodes_returns_floats():
    env = make_env()
    rewards = run_random_episodes(env, n_episodes=5)
    assert all(isinstance(r, float) for r in rewards)


def test_run_random_episodes_env_resets_between_episodes():
    env = make_env()
    rewards_a = run_random_episodes(env, n_episodes=3)
    rewards_b = run_random_episodes(env, n_episodes=3)
    # Both runs should complete without error (env properly resets)
    assert len(rewards_a) == len(rewards_b) == 3
