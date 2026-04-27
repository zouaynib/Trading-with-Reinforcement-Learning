"""
Section 2 — Tabular Q-learning
Discretized state: (sign of 5-day return, sign of 20-day return, position)
Q-table: 12 states × 3 actions
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves plots without displaying
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from trading_env import download_data, TradingEnv


# ──────────────────────────────────────────────
# 2.1  Discrete state encoder
# ──────────────────────────────────────────────

def discretize_state(prices: np.ndarray, t: int, position: int) -> int:
    """
    Map (sign of 5-day return, sign of 20-day return, position) → state index 0..11

    Features:
        s0 : sign of 5-day return  → 0 (negative/zero) or 1 (positive)
        s1 : sign of 20-day return → 0 (negative/zero) or 1 (positive)
        s2 : position ∈ {-1, 0, +1} → mapped to {0, 1, 2}

    Index: s0 * 6 + s1 * 3 + s2  ∈ {0, …, 11}
    """
    if t < 20:
        return 6 + {-1: 0, 0: 1, 1: 2}[position]   # neutral bucket

    ret5  = (prices[t] - prices[t - 5])  / prices[t - 5]
    ret20 = (prices[t] - prices[t - 20]) / prices[t - 20]

    s0 = 1 if ret5  > 0 else 0
    s1 = 1 if ret20 > 0 else 0
    s2 = {-1: 0, 0: 1, 1: 2}[position]

    return s0 * 6 + s1 * 3 + s2


# ──────────────────────────────────────────────
# 2.3  Train / test split (chronological 70/30)
# ──────────────────────────────────────────────

def split_data(prices: np.ndarray, train_ratio: float = 0.70):
    split = int(len(prices) * train_ratio)
    return prices[:split], prices[split:]


# ──────────────────────────────────────────────
# 2.2 + 2.3  Q-learning agent & training loop
# ──────────────────────────────────────────────

def train_q_learning(
    train_prices: np.ndarray,
    n_episodes:   int   = 500,
    alpha:        float = 0.1,
    gamma:        float = 0.99,
    eps_start:    float = 1.0,
    eps_end:      float = 0.01,
    window:       int   = 20,
    cost:         float = 0.001,
    verbose:      bool  = True,
):
    """Train tabular Q-learning. Returns Q-table and per-episode reward list."""
    n_states  = 12
    n_actions = 3
    Q = np.zeros((n_states, n_actions))

    env = TradingEnv(train_prices, window=window, cost=cost)
    episode_rewards = []

    for ep in range(n_episodes):
        # Linear epsilon decay
        eps = eps_start - (eps_start - eps_end) * ep / (n_episodes - 1)

        env.reset()
        total_reward = 0.0
        done = False

        while not done:
            t   = env.t
            pos = env.position
            s   = discretize_state(env.prices, t, pos)

            # ε-greedy action
            if np.random.rand() < eps:
                a = np.random.randint(n_actions)
            else:
                a = int(np.argmax(Q[s]))

            _, reward, done, _ = env.step(a)
            total_reward += reward

            # Next discrete state
            s_next = discretize_state(env.prices, env.t, env.position)

            # Q-learning update
            td_target = reward + gamma * np.max(Q[s_next]) * (1 - done)
            Q[s, a]  += alpha * (td_target - Q[s, a])

        episode_rewards.append(total_reward)

        if verbose and (ep + 1) % 100 == 0:
            print(f"  Episode {ep+1:4d}/{n_episodes} | ε={eps:.3f} | "
                  f"reward={total_reward:.2f} | avg(last50)={np.mean(episode_rewards[-50:]):.2f}")

    return Q, episode_rewards


# ──────────────────────────────────────────────
# 2.4  Plot training reward curve
# ──────────────────────────────────────────────

def plot_training_curve(episode_rewards: list, title: str = "Q-learning training"):
    rewards = np.array(episode_rewards)
    rolling = pd.Series(rewards).rolling(20, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, alpha=0.4, color="steelblue", label="Episode reward")
    ax.plot(rolling, color="steelblue", linewidth=2, label="Rolling avg (20)")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward ($)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig("q_learning_training.png", dpi=150)
    print("Saved → q_learning_training.png")


# ──────────────────────────────────────────────
# 2.5  Evaluation metrics
# ──────────────────────────────────────────────

def compute_metrics(cum_pnl: np.ndarray, n_trades: int, label: str = "") -> dict:
    """Total return, Sharpe ratio (annualised daily), max drawdown."""
    total_return = cum_pnl[-1]

    daily_pnl = np.diff(np.insert(cum_pnl, 0, 0.0))
    sharpe = (daily_pnl.mean() / (daily_pnl.std() + 1e-8)) * np.sqrt(252)

    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns   = cum_pnl - running_max
    mdd         = drawdowns.min()

    print(f"\n{'─'*40}")
    print(f"  {label}")
    print(f"{'─'*40}")
    print(f"  Total return  : ${total_return:>9.2f}")
    print(f"  Sharpe ratio  : {sharpe:>9.3f}")
    print(f"  Max drawdown  : ${mdd:>9.2f}")
    print(f"  # Trades      : {n_trades:>9d}")

    return dict(total_return=total_return, sharpe=sharpe, mdd=mdd, n_trades=n_trades)


def evaluate_policy(prices: np.ndarray, Q: np.ndarray, window: int = 20,
                    cost: float = 0.001, label: str = "Q-learning") -> dict:
    """Run greedy policy (ε=0) on prices. Return metrics dict and arrays for plotting."""
    env = TradingEnv(prices, window=window, cost=cost)
    env.reset()

    cum_pnl  = []
    total    = 0.0
    n_trades = 0
    done     = False

    while not done:
        t   = env.t
        pos = env.position
        s   = discretize_state(env.prices, t, pos)
        a   = int(np.argmax(Q[s]))

        prev_pos = env.position
        _, reward, done, _ = env.step(a)
        total += reward
        cum_pnl.append(total)
        if env.position != prev_pos:
            n_trades += 1

    cum_pnl = np.array(cum_pnl)
    metrics = compute_metrics(cum_pnl, n_trades, label=label)
    return metrics, cum_pnl


def buy_and_hold(prices: np.ndarray) -> np.ndarray:
    """Baseline: buy at first step, hold to end. Returns cumulated PnL."""
    daily_pnl = np.diff(prices)
    return np.cumsum(daily_pnl)


# ──────────────────────────────────────────────
# 2.5 cont.  Plot cumulated PnL vs buy-and-hold
# ──────────────────────────────────────────────

def plot_eval(test_prices: np.ndarray, agent_cum_pnl: np.ndarray,
              label: str = "Q-learning", fname: str = "q_eval.png"):
    bnh = buy_and_hold(test_prices)
    n   = min(len(agent_cum_pnl), len(bnh))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(agent_cum_pnl[:n], label=label, linewidth=1.2)
    ax.plot(bnh[:n],           label="Buy & Hold", linewidth=1.2, linestyle="--", color="orange")
    ax.axhline(0, color="black", linewidth=0.4, linestyle=":")
    ax.set_xlabel("Time step (test set)")
    ax.set_ylabel("Cumulated PnL ($)")
    ax.set_title(f"{label} vs Buy-and-Hold (test set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    print(f"Saved → {fname}")


# ──────────────────────────────────────────────
# 2.6  Print & interpret the Q-table
# ──────────────────────────────────────────────

def print_q_table(Q: np.ndarray):
    """Pretty-print the 12×3 Q-table with human-readable state labels."""
    sign_labels  = ["neg/flat", "positive"]
    pos_labels   = ["Short(-1)", "Flat(0)", "Long(+1)"]
    action_names = ["Hold", "Buy", "Sell"]

    header = f"{'State':>30s} | {'Hold':>8s} | {'Buy':>8s} | {'Sell':>8s} | Best action"
    print("\n" + "─" * len(header))
    print("  Q-TABLE  (12 states × 3 actions)")
    print("─" * len(header))
    print(header)
    print("─" * len(header))

    for s0 in range(2):
        for s1 in range(2):
            for s2 in range(3):
                idx   = s0 * 6 + s1 * 3 + s2
                label = f"5d={sign_labels[s0]}, 20d={sign_labels[s1]}, pos={pos_labels[s2]}"
                best  = action_names[int(np.argmax(Q[idx]))]
                print(f"  [{idx:2d}] {label:>38s} | "
                      f"{Q[idx,0]:>8.3f} | {Q[idx,1]:>8.3f} | {Q[idx,2]:>8.3f} | {best}")
    print("─" * len(header))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Load data
    df = download_data("AAPL", "2019-01-01", "2024-12-31")
    prices = df["price"].values

    # Split
    train_prices, test_prices = split_data(prices, train_ratio=0.70)
    print(f"\nTrain: {len(train_prices)} days | Test: {len(test_prices)} days")

    # Train
    print("\nTraining Q-learning (500 episodes)…")
    Q, ep_rewards = train_q_learning(train_prices, n_episodes=500)

    # 2.4 Plot training curve
    plot_training_curve(ep_rewards, title="Q-learning training reward — AAPL")

    # 2.5 Evaluate on test set
    metrics, agent_pnl = evaluate_policy(test_prices, Q, label="Q-learning (test)")

    # Buy-and-hold metrics
    bnh_pnl = buy_and_hold(test_prices)
    bnh_daily = np.diff(np.insert(bnh_pnl, 0, 0.0))
    bnh_sharpe = (bnh_daily.mean() / (bnh_daily.std() + 1e-8)) * np.sqrt(252)
    bnh_mdd = (bnh_pnl - np.maximum.accumulate(bnh_pnl)).min()
    print(f"\n{'─'*40}\n  Buy & Hold\n{'─'*40}")
    print(f"  Total return  : ${bnh_pnl[-1]:>9.2f}")
    print(f"  Sharpe ratio  : {bnh_sharpe:>9.3f}")
    print(f"  Max drawdown  : ${bnh_mdd:>9.2f}")
    print(f"  # Trades      : {'1':>9s}")

    # 2.5 Plot
    plot_eval(test_prices, agent_pnl, label="Q-learning", fname="q_eval.png")

    # 2.6 Q-table
    print_q_table(Q)
