from __future__ import annotations

import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from trading_env import TradingEnv
from agent import DQNAgent

logger = logging.getLogger(__name__)


def buy_and_hold(prices: np.ndarray) -> np.ndarray:
    return np.cumsum(np.diff(prices))


def compute_metrics(
    cum_pnl: np.ndarray,
    n_trades: int,
    label: str = "",
) -> dict[str, float]:
    total_return = cum_pnl[-1]
    daily_pnl    = np.diff(np.insert(cum_pnl, 0, 0.0))
    sharpe       = (daily_pnl.mean() / (daily_pnl.std() + 1e-8)) * np.sqrt(252)
    mdd          = (cum_pnl - np.maximum.accumulate(cum_pnl)).min()

    print(f"\n{'─'*40}")
    print(f"  {label}")
    print(f"{'─'*40}")
    print(f"  Total return : ${total_return:>9.2f}")
    print(f"  Sharpe ratio : {sharpe:>10.3f}")
    print(f"  Max drawdown : ${mdd:>9.2f}")
    print(f"  # Trades     : {n_trades:>10d}")

    logger.info(
        "%s | return=$%.2f | sharpe=%.3f | mdd=$%.2f | trades=%d",
        label, total_return, sharpe, mdd, n_trades,
    )
    return dict(total_return=total_return, sharpe=sharpe, mdd=mdd, n_trades=n_trades)


def run_greedy(
    agent: DQNAgent,
    prices: np.ndarray,
    window: int = 20,
    cost: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Run the agent with epsilon=0 on prices. Returns (cum_pnl, positions, q_hist, n_trades)."""
    env   = TradingEnv(prices, window=window, cost=cost)
    state = env.reset()

    cum_pnl, positions, q_hist = [], [], []
    total, n_trades = 0.0, 0
    done = False

    agent.q_net.eval()
    with torch.no_grad():
        while not done:
            q_vals = agent.q_net(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
            action = int(np.argmax(q_vals))
            q_hist.append(q_vals.copy())

            prev_pos                  = env.position
            state, reward, done, info = env.step(action)

            total += reward
            cum_pnl.append(total)
            positions.append(prev_pos)
            if env.position != prev_pos:
                n_trades += 1

    agent.q_net.train()
    return np.array(cum_pnl), np.array(positions), np.array(q_hist), n_trades


def plot_comparison(
    test_prices: np.ndarray,
    dqn_pnl: np.ndarray,
    ql_pnl: np.ndarray | None = None,
    fname: str = "dqn_eval.png",
) -> None:
    bnh = buy_and_hold(test_prices)
    n   = min(len(dqn_pnl), len(bnh))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dqn_pnl[:n], label="DQN",         linewidth=1.2, color="steelblue")
    if ql_pnl is not None:
        ax.plot(ql_pnl[:n], label="Q-learning", linewidth=1.2, color="green")
    ax.plot(bnh[:n],     label="Buy & Hold",  linewidth=1.2, linestyle="--", color="orange")
    ax.axhline(0, color="black", linewidth=0.4, linestyle=":")
    ax.set_xlabel("Time step (test set)")
    ax.set_ylabel("Cumulated PnL ($)")
    ax.set_title("DQN vs Q-learning vs Buy-and-Hold (test set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    logger.info("Saved → %s", fname)
    print(f"Saved → {fname}")
    plt.close()


if __name__ == "__main__":
    import random
    from train import train
    from q_learning import split_data, download_data, train_q_learning, evaluate_policy

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
    print(f"Train: {len(train_prices)} | Val: {len(val_prices)} | Test: {len(test_prices)}\n")

    print("Training DQN…")
    agent, _, _ = train(train_prices, val_prices, n_episodes=500)

    dqn_pnl, _, _, n_trades = run_greedy(agent, test_prices)
    compute_metrics(dqn_pnl, n_trades, label="DQN")

    full_train, _ = split_data(prices)
    print("\nTraining Q-learning…")
    Q, _ = train_q_learning(full_train, n_episodes=500, verbose=False)
    _, ql_pnl = evaluate_policy(test_prices, Q, label="Q-learning")

    bnh_pnl   = buy_and_hold(test_prices)
    bnh_daily = np.diff(np.insert(bnh_pnl, 0, 0.0))
    bnh_sharpe = (bnh_daily.mean() / (bnh_daily.std() + 1e-8)) * np.sqrt(252)
    bnh_mdd    = (bnh_pnl - np.maximum.accumulate(bnh_pnl)).min()
    print(f"\n{'─'*40}\n  Buy & Hold\n{'─'*40}")
    print(f"  Total return : ${bnh_pnl[-1]:>9.2f}")
    print(f"  Sharpe ratio : {bnh_sharpe:>10.3f}")
    print(f"  Max drawdown : ${bnh_mdd:>9.2f}")
    print(f"  # Trades     : {'1':>10s}")

    plot_comparison(test_prices, dqn_pnl, ql_pnl)
    print("\nevaluate.py OK")
