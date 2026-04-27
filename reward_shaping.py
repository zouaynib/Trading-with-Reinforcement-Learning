from __future__ import annotations

import copy
import logging
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from trading_env import download_data, TradingEnv
from q_learning import split_data
from agent import DQNAgent
from evaluate import run_greedy, buy_and_hold, compute_metrics

logger = logging.getLogger(__name__)


def train_shaped(
    train_prices:  np.ndarray,
    val_prices:    np.ndarray,
    reward_lambda: float,
    n_episodes:    int   = 500,
    window:        int   = 20,
    cost:          float = 0.001,
    eps_decay_eps: int   = 300,
    patience:      int   = 50,
    val_every:     int   = 10,
) -> tuple[DQNAgent, list[float]]:
    """Train DQN with risk-adjusted reward: standard reward - λ * q_t²"""
    env   = TradingEnv(train_prices, window=window, cost=cost)
    agent = DQNAgent(state_dim=env.state_dim)

    best_val_reward     = -np.inf
    best_weights        = None
    episodes_no_improve = 0
    episode_rewards:    list[float] = []
    val_rewards:        list[tuple[int, float]] = []

    for ep in range(n_episodes):
        frac    = min(ep / max(eps_decay_eps - 1, 1), 1.0)
        epsilon = 1.0 - (1.0 - 0.01) * frac

        state   = env.reset()
        total_r = 0.0
        done    = False

        while not done:
            action                      = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            shaped = reward - reward_lambda * (env.position ** 2)

            agent.push(state, action, shaped, next_state, done)
            agent.update()
            state    = next_state
            total_r += reward   # track unshaped reward for fair comparison

        episode_rewards.append(total_r)

        if (ep + 1) % val_every == 0:
            val_r = _val_reward(agent, val_prices, window, cost)
            val_rewards.append((ep + 1, val_r))

            if val_r > best_val_reward:
                best_val_reward     = val_r
                best_weights        = copy.deepcopy(agent.q_net.state_dict())
                episodes_no_improve = 0
            else:
                episodes_no_improve += val_every

            if episodes_no_improve >= patience:
                logger.info("λ=%.4f early stopping at episode %d", reward_lambda, ep + 1)
                break

    if best_weights is not None:
        agent.q_net.load_state_dict(best_weights)
        agent.target_net.load_state_dict(best_weights)

    return agent, episode_rewards


def _val_reward(agent: DQNAgent, prices: np.ndarray, window: int, cost: float) -> float:
    env   = TradingEnv(prices, window=window, cost=cost)
    state = env.reset()
    total = 0.0
    done  = False
    agent.q_net.eval()
    with torch.no_grad():
        while not done:
            q      = agent.q_net(torch.FloatTensor(state).unsqueeze(0))
            action = int(q.argmax().item())
            state, reward, done, _ = env.step(action)
            total += reward
    agent.q_net.train()
    return total


def run_reward_shaping(
    train_prices: np.ndarray,
    val_prices: np.ndarray,
    test_prices: np.ndarray,
    lambdas: list[float] | None = None,
) -> dict[float, tuple[dict, np.ndarray]]:
    if lambdas is None:
        lambdas = [0.0001, 0.001, 0.01]

    results: dict[float, tuple[dict, np.ndarray]] = {}

    print(f"\n{'λ':>10s} | {'Return($)':>10s} | {'Sharpe':>8s} | {'MDD($)':>10s} | {'# Trades':>9s}")
    print("─" * 57)

    for lam in lambdas:
        logger.info("Training shaped agent λ=%.4f", lam)
        agent, _ = train_shaped(train_prices, val_prices, reward_lambda=lam)
        pnl, _, _, n_trades = run_greedy(agent, test_prices)
        metrics = compute_metrics(pnl, n_trades, label=f"λ={lam}")
        results[lam] = (metrics, pnl)
        print(f"{lam:>10.4f} | {metrics['total_return']:>10.2f} | "
              f"{metrics['sharpe']:>8.3f} | {metrics['mdd']:>10.2f} | {n_trades:>9d}")

    return results


def plot_shaped(
    test_prices: np.ndarray,
    results: dict[float, tuple[dict, np.ndarray]],
    fname: str = "dqn_reward_shaping.png",
) -> None:
    bnh = buy_and_hold(test_prices)
    n   = len(bnh)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["steelblue", "green", "red"]
    for (lam, (_, pnl)), color in zip(results.items(), colors):
        ax.plot(pnl[:n], label=f"λ={lam}", linewidth=1.2, color=color)
    ax.plot(bnh, label="Buy & Hold", linewidth=1.2, linestyle="--", color="orange")
    ax.axhline(0, color="black", linewidth=0.4, linestyle=":")
    ax.set_xlabel("Time step (test set)")
    ax.set_ylabel("Cumulated PnL ($)")
    ax.set_title("Reward shaping: effect of position penalty λ")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    logger.info("Saved → %s", fname)
    print(f"\nSaved → {fname}")
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    df     = download_data("BTC-USD", "2019-01-01", "2024-12-31")
    prices = df["price"].values
    train_prices, test_prices = split_data(prices)
    val_split    = int(len(train_prices) * 0.85)
    val_prices   = train_prices[val_split:]
    train_prices = train_prices[:val_split]

    print("Training 3 agents with different λ values (this takes ~10 min)…")
    results = run_reward_shaping(train_prices, val_prices, test_prices)
    plot_shaped(test_prices, results)

    print("\nreward_shaping.py OK")
