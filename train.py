from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from trading_env import download_data, TradingEnv
from q_learning import split_data
from agent import DQNAgent

logger = logging.getLogger(__name__)


def _val_reward(agent: DQNAgent, prices: np.ndarray, window: int, cost: float) -> float:
    """Run one greedy episode on prices. Returns total reward (no exploration)."""
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


def train(
    train_prices:  np.ndarray,
    val_prices:    np.ndarray,
    n_episodes:    int   = 500,
    window:        int   = 20,
    cost:          float = 0.001,
    eps_start:     float = 1.0,
    eps_end:       float = 0.01,
    eps_decay_eps: int   = 300,
    patience:      int   = 50,
    val_every:     int   = 10,
) -> tuple[DQNAgent, list[float], list[tuple[int, float]]]:
    """Train DQN with early stopping on a held-out validation set."""
    env   = TradingEnv(train_prices, window=window, cost=cost)
    agent = DQNAgent(state_dim=env.state_dim)

    best_val_reward     = -np.inf
    best_weights        = None
    episodes_no_improve = 0
    episode_rewards:    list[float]             = []
    val_rewards:        list[tuple[int, float]] = []

    for ep in range(n_episodes):
        frac    = min(ep / max(eps_decay_eps - 1, 1), 1.0)
        epsilon = eps_start - (eps_start - eps_end) * frac

        state   = env.reset()
        total_r = 0.0
        done    = False

        while not done:
            action                      = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.push(state, action, reward, next_state, done)
            agent.update()
            state    = next_state
            total_r += reward

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

            if (ep + 1) % 50 == 0:
                logger.info(
                    "Ep %4d/%d | ε=%.3f | train=%.2f | val=%.2f | best_val=%.2f",
                    ep + 1, n_episodes, epsilon, total_r, val_r, best_val_reward,
                )
                print(f"  Ep {ep+1:4d}/{n_episodes} | ε={epsilon:.3f} | "
                      f"train={total_r:.2f} | val={val_r:.2f} | best_val={best_val_reward:.2f}")

            if episodes_no_improve >= patience:
                logger.info("Early stopping at episode %d", ep + 1)
                print(f"\n  Early stopping at episode {ep+1} "
                      f"(no val improvement for {patience} episodes)")
                break

    if best_weights is not None:
        agent.q_net.load_state_dict(best_weights)
        agent.target_net.load_state_dict(best_weights)
        logger.info("Restored best weights (val reward = %.2f)", best_val_reward)
        print(f"  Restored best weights (val reward = {best_val_reward:.2f})")

    return agent, episode_rewards, val_rewards


def plot_training(
    episode_rewards: list[float],
    val_rewards: list[tuple[int, float]],
    fname: str = "dqn_training.png",
) -> None:
    rewards = np.array(episode_rewards)
    rolling = pd.Series(rewards).rolling(20, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    ax1.plot(rewards, alpha=0.35, color="steelblue", label="Train reward")
    ax1.plot(rolling, color="steelblue", linewidth=2, label="Rolling avg (20)")
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("Total reward ($)")
    ax1.set_title("DQN training reward")
    ax1.legend()

    val_eps, val_r = zip(*val_rewards) if val_rewards else ([], [])
    ax2.plot(val_eps, val_r, color="orange", linewidth=1.5, label="Val reward")
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Val reward ($)")
    ax2.set_title("Validation reward (greedy)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    logger.info("Saved → %s", fname)
    print(f"Saved → {fname}")
    plt.close()


if __name__ == "__main__":
    import random
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    df     = download_data("BTC-USD", "2019-01-01", "2024-12-31")
    prices = df["price"].values
    train_prices, test_prices = split_data(prices)
    val_split    = int(len(train_prices) * 0.85)
    train_prices, val_prices = train_prices[:val_split], train_prices[val_split:]
    print(f"Train: {len(train_prices)} days | Val: {len(val_prices)} days\n")

    agent, train_rewards, val_rewards = train(train_prices, val_prices, n_episodes=500)
    plot_training(train_rewards, val_rewards)

    print(f"\nBest val reward : {max(r for _, r in val_rewards):.2f}")
    print("train.py OK")
