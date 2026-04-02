import torch
import numpy as np
import matplotlib.pyplot as plt
from .env import TradingEnv
from .config import *

def backtest(trainer, states_list, prices_list, tickers=None, steps_per_day=1):
    trainer.actor.eval()
    results = {}
    for i, (states, prices) in enumerate(zip(states_list, prices_list)):
        ticker = tickers[i] if tickers else f"Asset_{i}"
        env = TradingEnv(states, prices)
        state, _ = env.reset()
        done = False
        equity_curve = [INITIAL_CAPITAL]
        returns = []
        action_history = []
        while not done:
            state_t = torch.tensor(
                state,
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)
            with torch.no_grad():
                logits = trainer.actor(state_t)
                action = torch.argmax(logits, dim=-1).item()
            next_state, reward, done, _, info = env.step(action)
            action_history.append(action)
            returns.append(reward)
            equity_curve.append(info["portfolio_value"])
            state = next_state
        equity_curve = np.array(equity_curve)
        returns = np.array(returns)
        action_history = np.array(action_history)

        # ================= Metrics =================
        final_value = equity_curve[-1]
        total_return = (final_value / INITIAL_CAPITAL - 1) * 100
        winrate = (returns > 0).mean() * 100

        # ================= Action Distribution =================
        total_actions = len(action_history)
        action_counts = {
            ACTION_NAMES[a]: {
                "count": int((action_history == a).sum()),
                "percentage": round(
                    100 * (action_history == a).sum() / total_actions, 2
                )
            }
            for a in range(ACT_DIM)
        }
        results[ticker] = {
            "Total Return (%)": round(total_return, 4),
            "Win Rate (%)": round(winrate, 2),
            "Final Value": round(final_value, 6),
            "Action Distribution": action_counts,
            "Equity Curve": equity_curve
        }
    trainer.actor.train()
    return results


def plot_equity(equity_curve,
                title="Equity Curve",
                benchmark=None):
    """
    Vẽ equity curve của agent.
    Args:
        equity_curve: np.array portfolio value
        title: tiêu đề biểu đồ
        benchmark: buy & hold curve (optional)
        action_history: np.array actions (optional)
    """

    plt.figure(figsize=(12, 5))
    plt.plot(equity_curve, label="Agent", linewidth=1.5)
    if benchmark is not None:
        plt.plot(
            benchmark,
            label="Buy & Hold",
            linewidth=1.5,
            linestyle="--",
            alpha=0.8
        )
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Portfolio Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_buyhold_equity(prices,
                           initial_capital=INITIAL_CAPITAL,
                           transaction_cost=TRANSACTION_COST):
    """
    Tính equity curve của chiến lược Buy & Hold.
    Args:
        prices: np.array giá đóng cửa
        initial_capital: vốn ban đầu
        transaction_cost: phí giao dịch (mua lần đầu)
    Returns:
        np.array equity curve
    """

    prices = np.array(prices)
    # Mua toàn bộ tại bước đầu
    buy_cost = initial_capital * transaction_cost
    capital_after_cost = initial_capital - buy_cost
    shares = capital_after_cost / prices[0]
    equity = shares * prices
    # Thêm initial capital để align với equity_curve agent
    equity = np.insert(equity, 0, initial_capital)
    return equity