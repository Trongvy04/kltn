import numpy as np

ACTION_NAMES = {
    0: "ALLOC_0",
    1: "ALLOC_35",
    2: "ALLOC_70"
}

class TradingEnvEngine:

    def __init__(self, transaction_cost=0.002, initial_capital=100000):
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        self.reset()

    # ==========================================================
    # RESET
    # ==========================================================
    def reset(self):
        self.cash = float(self.initial_capital)
        self.shares = 0.0
        self.portfolio_value = float(self.initial_capital)
        self.max_value = float(self.initial_capital)

    # ==========================================================
    # STEP
    # ==========================================================
    def step(self, action, close_price):

        action_name = ACTION_NAMES.get(action, "UNKNOWN")

        price = close_price

        value_before = self.cash + self.shares * price
        current_position_value = self.shares * price

        # ===== Target allocation =====
        if action == 0:
            target_ratio = 0.0
        elif action == 1:
            target_ratio = 0.35
        elif action == 2:
            target_ratio = 0.7
        else:
            target_ratio = 0.0

        target_position_value = target_ratio * value_before
        delta_value = target_position_value - current_position_value

        trade_ratio = abs(delta_value) / (value_before + 1e-8)

        # ======================================================
        # REBALANCE
        # ======================================================
        if abs(delta_value) > 1e-8:

            # BUY
            if delta_value > 0:
                trade_value = min(delta_value, self.cash)
                cost = trade_value * self.transaction_cost
                total_spent = trade_value + cost

                if total_spent > self.cash:
                    trade_value = self.cash / (1 + self.transaction_cost)
                    cost = trade_value * self.transaction_cost
                    total_spent = trade_value + cost

                shares_delta = trade_value / price
                self.shares += shares_delta
                self.cash -= total_spent

            # SELL
            else:
                sell_value = abs(delta_value)
                sell_value = min(sell_value, self.shares * price)

                cost = sell_value * self.transaction_cost
                shares_delta = sell_value / price

                self.shares -= shares_delta
                self.cash += sell_value - cost

        # ======================================================
        # PORTFOLIO AFTER MOVE
        # ======================================================
        portfolio_value = self.cash + self.shares * price
        self.max_value = max(self.max_value, portfolio_value)

        if portfolio_value <= 0:
            reward = -1.0
        else:
            reward = (
                (portfolio_value - value_before) / (value_before + 1e-8)
                - 0.001 * trade_ratio
            )

        self.portfolio_value = portfolio_value

        return {
            "reward": float(reward),
            "action_name": action_name,
            "shares": float(self.shares),
            "cash": float(self.cash),
            "portfolio_value": float(portfolio_value)
        }