"""
Trading Environment (OpenAI Gym-style)
Section 1 of the RL Trading Lab
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# 1.1  Download data & compute log returns
# ──────────────────────────────────────────────

def download_data(ticker: str = "AAPL", start: str = "2019-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Download daily adjusted close prices and compute log returns."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}. Check your internet connection and try again.")
    df = df[["Close"]].rename(columns={"Close": "price"})
    # Flatten MultiIndex columns that yfinance may return for single-ticker downloads
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["price"] = df["price"].squeeze()   # ensure 1-D Series
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    df = df.dropna().copy()
    print(f"Downloaded {len(df)} trading days for {ticker} ({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ──────────────────────────────────────────────
# 1.2 – 1.6  TradingEnv class
# ──────────────────────────────────────────────

class TradingEnv:
    """
    Single-asset trading environment.

    State  : (W+2,) vector
             - W z-scored recent log-returns (rolling window)
             - current position  qt ∈ {-1, 0, +1}
             - unrealized PnL of current position (normalized by price)

    Actions: 0=hold, 1=buy, 2=sell

    Reward : qt * (p_{t+1} - p_t) - c * I[q_{t+1} != qt]
    """

    HOLD = 0
    BUY  = 1
    SELL = 2

    def __init__(self, prices: np.ndarray, window: int = 20, cost: float = 0.001):
        """
        Parameters
        ----------
        prices : array of daily closing prices
        window : look-back for state features (W)
        cost   : transaction cost as fraction of price
        """
        self.prices  = np.array(prices, dtype=np.float64).ravel()
        self.window  = window
        self.cost    = cost
        self.n_steps = len(prices)

        # Pre-compute log returns
        self.log_returns = np.zeros(self.n_steps)
        self.log_returns[1:] = np.log(self.prices[1:] / self.prices[:-1])

        self.state_dim  = window + 2
        self.action_dim = 3

        self.t          = None   # current time index
        self.position   = None   # qt ∈ {-1, 0, +1}
        self.entry_price = None  # price when position was opened

    # ── helpers ──────────────────────────────

    def _zscore_returns(self, t: int) -> np.ndarray:
        """Z-score the W most recent log returns using a rolling window."""
        window_returns = self.log_returns[t - self.window + 1: t + 1]
        mu  = window_returns.mean()
        std = window_returns.std() + 1e-8
        return (window_returns - mu) / std

    def _unrealized_pnl(self) -> float:
        """Normalized unrealized PnL."""
        if self.position == 0 or self.entry_price is None:
            return 0.0
        pnl = self.position * (self.prices[self.t] - self.entry_price)
        return pnl / self.prices[self.t]   # normalize by current price

    def _build_state(self) -> np.ndarray:
        z = self._zscore_returns(self.t)
        return np.append(z, [float(self.position), self._unrealized_pnl()])

    # ── public interface ──────────────────────

    def reset(self) -> np.ndarray:
        """Reset environment to start. Returns initial state."""
        self.t            = self.window - 1   # first usable index
        self.position     = 0
        self.entry_price  = None
        return self._build_state()

    def step(self, action: int):
        """
        Execute action and advance one time step.

        Returns
        -------
        next_state : np.ndarray
        reward     : float
        done       : bool
        info       : dict
        """
        assert action in (0, 1, 2), f"Invalid action {action}"

        prev_position = self.position
        p_now         = self.prices[self.t]

        # ── position transition ──
        if self.position == 0:
            if action == self.BUY:
                self.position  = +1
                self.entry_price = p_now
            elif action == self.SELL:
                self.position  = -1
                self.entry_price = p_now
            # else HOLD → no change

        elif self.position == +1:
            if action == self.SELL:
                self.position  = 0
                self.entry_price = None
            # BUY / HOLD while long → keep

        elif self.position == -1:
            if action == self.BUY:
                self.position  = 0
                self.entry_price = None
            # SELL / HOLD while short → keep

        # ── reward ──
        if self.t + 1 < self.n_steps:
            p_next = self.prices[self.t + 1]
        else:
            p_next = p_now

        transaction_cost = self.cost * p_now if (self.position != prev_position) else 0.0
        reward = prev_position * (p_next - p_now) - transaction_cost

        # ── advance time ──
        self.t += 1
        done = self.t >= self.n_steps - 1

        next_state = self._build_state() if not done else np.zeros(self.state_dim)

        info = {
            "position"       : self.position,
            "price"          : p_now,
            "reward"         : reward,
            "transaction_cost": transaction_cost,
        }
        return next_state, reward, done, info


# ──────────────────────────────────────────────
# 1.6  Test: 100 random episodes, plot one
# ──────────────────────────────────────────────

def run_random_episodes(env: TradingEnv, n_episodes: int = 100):
    """Run n_episodes with random actions; return per-episode total rewards."""
    episode_rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = np.random.randint(0, 3)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)
    return episode_rewards


def plot_episode(env: TradingEnv, title: str = "Random-policy episode"):
    """Run one episode with random actions and plot price, position, cumulated PnL."""
    state = env.reset()
    prices, positions, cum_pnl = [], [], []
    total_pnl = 0.0
    done = False

    while not done:
        action = np.random.randint(0, 3)
        state, reward, done, info = env.step(action)
        prices.append(info["price"])
        positions.append(info["position"])
        total_pnl += info["reward"]
        cum_pnl.append(total_pnl)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    t_axis = range(len(prices))

    axes[0].plot(t_axis, prices, linewidth=0.8)
    axes[0].set_ylabel("Price ($)")
    axes[0].set_title(title)

    axes[1].step(t_axis, positions, where="post", linewidth=0.8, color="orange")
    axes[1].axhline(0, color="black", linewidth=0.4, linestyle="--")
    axes[1].set_ylabel("Position")
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_yticklabels(["Short", "Flat", "Long"])

    axes[2].plot(t_axis, cum_pnl, linewidth=0.8, color="green")
    axes[2].axhline(0, color="black", linewidth=0.4, linestyle="--")
    axes[2].set_ylabel("Cumulated PnL ($)")
    axes[2].set_xlabel("Time step")

    plt.tight_layout()
    plt.savefig("random_episode.png", dpi=150)
    print("Plot saved → random_episode.png")


if __name__ == "__main__":
    # Download data
    df = download_data("AAPL", "2019-01-01", "2024-12-31")
    prices = df["price"].values

    # Build environment
    env = TradingEnv(prices, window=20, cost=0.001)

    # 100 random episodes
    rewards = run_random_episodes(env, n_episodes=100)
    print(f"\n100 random episodes — reward stats:")
    print(f"  mean  : {np.mean(rewards):.2f}")
    print(f"  std   : {np.std(rewards):.2f}")
    print(f"  min   : {np.min(rewards):.2f}")
    print(f"  max   : {np.max(rewards):.2f}")

    # Plot one episode
    plot_episode(env, title="Random-policy episode — AAPL")
