import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from .config import *

class TradingEnv(gym.Env):
    def __init__(self, states, prices, random_start=True):
        super().__init__()
        self.states = torch.tensor(states, dtype=torch.float32)
        self.prices = torch.tensor(prices, dtype=torch.float32)
        self.random_start = random_start
        self.max_step = len(prices) - 1

        self.action_space = spaces.Discrete(ACT_DIM)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(STATE_DIM,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        data_len = len(self.states)
        if self.random_start:
            start_idx = np.random.randint(0, max(1, data_len - EPISODE_LEN - 1))
            self.end_idx = min(start_idx + EPISODE_LEN, data_len - 1)
        else:
            start_idx = 0
            self.end_idx = data_len - 1

        self.t = start_idx
        self.cash = float(INITIAL_CAPITAL)
        self.shares = 0.0
        self.max_value = float(INITIAL_CAPITAL)

        # Tạo state đầu tiên
        portfolio_value = self.cash
        cash_ratio = 1.0
        position_ratio = 0.0
        portfolio = torch.tensor([cash_ratio, position_ratio])
        self.current_state = torch.cat([self.states[self.t], portfolio])
        return self.current_state.clone().numpy(), {}

    def step(self, action):
        price = self.prices[self.t]
        next_price = self.prices[min(self.t + 1, self.max_step)]

        # Giá trị portfolio trước khi trade
        value_before = self.cash + self.shares * price
        current_position_value = self.shares * price

        # Xác định mục tiêu
        if action == 0:
            target_ratio = 0.0
        elif action == 1:
            target_ratio = 0.35
        elif action == 2:
            target_ratio = 0.7

        target_position_value = target_ratio * value_before
        delta_value = target_position_value - current_position_value
        trade_ratio = abs(delta_value) / (value_before + 1e-8)

        # Thực hiện giao dịch
        if abs(delta_value) > 1e-8:
            if delta_value > 0:  # Mua
                trade_value = min(delta_value, self.cash)
                cost = trade_value * TRANSACTION_COST
                total_spent = trade_value + cost
                shares_delta = trade_value / price
                self.shares += shares_delta
                self.cash -= total_spent
            else:  # Bán
                sell_value = abs(delta_value)
                cost = sell_value * TRANSACTION_COST
                shares_delta = sell_value / price
                self.shares -= shares_delta
                self.cash += sell_value - cost

        # Giá trị portfolio sau khi price move
        value_after = self.cash + self.shares * next_price
        self.max_value = max(self.max_value, value_after)

        # Tính reward
        if value_after <= 0:
            reward = -1.0
            done = True
        else:
            reward = (value_after - value_before) / (value_before + 1e-8) - 0.001 * trade_ratio
            done = self.t >= self.end_idx - 1

        # Cập nhật state hiện tại
        portfolio_value = self.cash + self.shares * next_price
        cash_ratio = self.cash / (portfolio_value + 1e-8)
        position_ratio = (self.shares * next_price) / (portfolio_value + 1e-8)
        portfolio = torch.tensor([cash_ratio, position_ratio])
        self.current_state = torch.cat([self.states[self.t], portfolio])

        # Tăng timestep
        self.t += 1

        info = {"portfolio_value": float(portfolio_value)}
        return self.current_state.clone().numpy(), float(reward), done, False, info