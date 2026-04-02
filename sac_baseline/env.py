# env.py

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

        self.t = 0
        self.cash = float(INITIAL_CAPITAL)
        self.shares = 0.0

        return self.states[self.t].numpy(), {}

    def step(self, action):

        price = self.prices[self.t]
        next_price = self.prices[self.t + 1]

        value_before = self.cash + self.shares * price

        # ===== ACTION =====
        if action == 0:
            target_ratio = 0.0
        elif action == 1:
            target_ratio = 0.35
        elif action == 2:
            target_ratio = 0.7

        target_value = target_ratio * value_before
        current_value = self.shares * price
        delta = target_value - current_value

        # ===== TRADE =====
        if abs(delta) > 1e-8:
            if delta > 0:
                buy_value = min(delta, self.cash)
                cost = buy_value * TRANSACTION_COST
                shares = buy_value / price
                self.shares += shares
                self.cash -= (buy_value + cost)
            else:
                sell_value = abs(delta)
                cost = sell_value * TRANSACTION_COST
                shares = sell_value / price
                self.shares -= shares
                self.cash += (sell_value - cost)

        # ===== UPDATE =====
        value_after = self.cash + self.shares * next_price

        reward = (value_after - value_before) / (value_before + 1e-8)

        self.t += 1
        done = self.t >= len(self.prices) - 1

        return self.states[self.t].numpy(), reward, done, False, {
            "portfolio_value": float(value_after)
        }
        