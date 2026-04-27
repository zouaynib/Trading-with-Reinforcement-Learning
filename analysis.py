import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import torch
import random

from trading_env import download_data, TradingEnv
from q_learning import split_data
from evaluate import run_greedy
from train import train


def action_distribution(q_hist: np.ndarray):
    """4.1 — What fraction of time does the agent buy/sell/hold?"""
    actions = np.argmax(q_hist, axis=1)
    labels  = ["Hold", "Buy", "Sell"]
    counts  = [np.sum(actions == i) for i in range(3)]
    T       = len(actions)

    print("\n── 4.1 Action distribution (test set) ──")
    for lbl, cnt in zip(labels, counts):
        print(f"  {lbl:>5s}: {cnt:4d}  ({100*cnt/T:.1f}%)")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, counts, color=["grey", "green", "red"])
    ax.set_ylabel("Count")
    ax.set_title("Action distribution (test set)")
    plt.tight_layout()
    plt.savefig("dqn_action_dist.png", dpi=150)
    print("Saved → dqn_action_dist.png")
    plt.close()


def plot_q_values(q_hist: np.ndarray):
    """4.2 — Q-values over time. Large gap between best/worst = high conviction."""
    labels = ["Hold", "Buy", "Sell"]
    best   = q_hist.max(axis=1)
    worst  = q_hist.min(axis=1)
    gap    = best - worst   # conviction: how certain is the agent?

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for i, lbl in enumerate(labels):
        ax1.plot(q_hist[:, i], linewidth=0.7, label=f"Q({lbl})", alpha=0.8)
    ax1.set_ylabel("Q-value ($)")
    ax1.set_title("Q-values over time (test set)")
    ax1.legend()

    ax2.plot(gap, linewidth=0.7, color="purple")
    ax2.set_ylabel("Max − Min Q ($)")
    ax2.set_xlabel("Time step (test set)")
    ax2.set_title("Agent conviction (larger = more certain)")

    plt.tight_layout()
    plt.savefig("dqn_q_values.png", dpi=150)
    print("Saved → dqn_q_values.png")
    plt.close()


def trend_vs_mean_revert(positions: np.ndarray, test_prices: np.ndarray, window: int = 20):
    """
    4.3 — Compute correlation between position and past 5-day return.
    Positive → trend-following (buy after rises)
    Negative → mean-reverting (buy after falls)
    """
    # 5-day return at each test step, aligned with position array
    t_start  = window - 1
    ret5     = []
    for t in range(t_start, t_start + len(positions)):
        if t >= 5:
            r = (test_prices[t] - test_prices[t - 5]) / test_prices[t - 5]
        else:
            r = 0.0
        ret5.append(r)

    ret5 = np.array(ret5[:len(positions)])
    corr = np.corrcoef(positions, ret5)[0, 1]
    direction = "trend-following" if corr > 0 else "mean-reverting"

    print(f"\n── 4.3 Correlation(position, 5-day return) = {corr:.4f}  → {direction} ──")
    return corr


def kendall_tau(positions: np.ndarray, test_prices: np.ndarray, window: int = 20):
    """
    4.4 — Kendall τ between position and next-day return.
    Tests whether the agent's position has statistically significant
    predictive power over the next day's price move.
    """
    next_day_ret = np.diff(test_prices) / test_prices[:-1]
    t_start      = window - 1
    aligned      = next_day_ret[t_start: t_start + len(positions)]
    n            = min(len(positions), len(aligned))

    tau, p_value = stats.kendalltau(positions[:n], aligned[:n])

    print(f"── 4.4 Kendall τ = {tau:.4f}  p-value = {p_value:.4f} ──")
    if p_value < 0.05:
        print("     → Statistically significant at 5%: positions predict next-day returns.")
    else:
        print("     → Not statistically significant: positions do NOT reliably predict returns.")

    return tau, p_value


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Data
    df     = download_data("BTC-USD", "2019-01-01", "2024-12-31")
    prices = df["price"].values
    train_prices, test_prices = split_data(prices)
    val_split    = int(len(train_prices) * 0.85)
    val_prices   = train_prices[val_split:]
    train_prices = train_prices[:val_split]

    # Train
    print("Training DQN…")
    agent, _, _ = train(train_prices, val_prices, n_episodes=500)

    # Run greedy on test set
    _, positions, q_hist, _ = run_greedy(agent, test_prices)

    # Section 4 analysis
    action_distribution(q_hist)
    plot_q_values(q_hist)
    trend_vs_mean_revert(positions, test_prices)
    kendall_tau(positions, test_prices)

    print("\nanalysis.py OK")
