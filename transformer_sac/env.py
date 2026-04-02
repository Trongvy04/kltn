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
            shape=(SEQ_LEN, STATE_DIM),
            dtype=np.float32
        )

    # ==========================================================
    # RESET
    # ==========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        data_len = len(self.states)
        if self.random_start:
            max_start = data_len - EPISODE_LEN - 1
            if max_start <= 0:
                start_idx = 0
            else:
                start_idx = np.random.randint(0, max_start)
            end_idx = start_idx + EPISODE_LEN
        else:
            start_idx = 0
            end_idx = data_len - 1
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.t = self.start_idx + SEQ_LEN - 1
        self.cash = float(INITIAL_CAPITAL)
        self.shares = 0.0
        self.max_value = float(INITIAL_CAPITAL)
        self.seq = torch.zeros(SEQ_LEN, STATE_DIM)
        for i in range(SEQ_LEN):
            idx = self.t - SEQ_LEN + 1 + i
            state = self.states[idx]
            portfolio = torch.tensor([1.0, 0.0])
            obs = torch.cat([state, portfolio])
            self.seq[i] = obs
        return self.seq.clone().numpy(), {}

    # ==========================================================
    # STEP
    # ==========================================================
    def step(self, action):
        if self.t >= self.end_idx - 1 or self.t + 1 >= len(self.prices):
            price = self.prices[self.t]
            portfolio_value = self.cash + self.shares * price
            done = True
            info = {"portfolio_value": float(portfolio_value)}
            return self.seq.clone().numpy(), 0.0, done, False, info
        price = self.prices[self.t]
        next_price = self.prices[self.t + 1]

        value_before = self.cash + self.shares * price
        current_position_value = self.shares * price
        if action == 0:
            target_ratio = 0.0
        elif action == 1:
            target_ratio = 0.35
        elif action == 2:
            target_ratio = 0.7
        target_position_value = target_ratio * value_before
        delta_value = target_position_value - current_position_value
        trade_ratio = abs(delta_value) / (value_before + 1e-8)
        if abs(delta_value) > 1e-8:
            if delta_value > 0:
                trade_value = min(delta_value, self.cash)
                cost = trade_value * TRANSACTION_COST
                total_spent = trade_value + cost
                if total_spent > self.cash:
                    trade_value = self.cash / (1 + TRANSACTION_COST)
                    cost = trade_value * TRANSACTION_COST
                    total_spent = trade_value + cost
                shares_delta = trade_value / price
                self.shares += shares_delta
                self.cash -= total_spent
            else:
                sell_value = abs(delta_value)
                cost = sell_value * TRANSACTION_COST
                shares_delta = sell_value / price
                self.shares -= shares_delta
                self.cash += sell_value - cost

        # ======================================================
        # PORTFOLIO VALUE AFTER PRICE MOVE
        # ======================================================

        value_after = self.cash + self.shares * next_price
        
        self.max_value = max(self.max_value, value_after)

        early_done = False

        if value_after <= 0:
            reward = -1.0
            early_done = True
        else:
            reward = (
                (value_after - value_before) / (value_before + 1e-8)
                - 0.001 * trade_ratio
            )

        
        # ======================================================
        # ADVANCE TIME
        # ======================================================

        self.t += 1
        time_done = self.t >= self.end_idx - 1
        done = early_done or time_done

        next_state = self.states[self.t]
        portfolio_value = self.cash + self.shares * next_price
        cash_ratio = self.cash / (portfolio_value + 1e-8)
        position_ratio = (self.shares * next_price) / (portfolio_value + 1e-8)
        portfolio = torch.tensor([cash_ratio, position_ratio])
        obs = torch.cat([next_state, portfolio])
        self.seq = torch.roll(self.seq, shifts=-1, dims=0)
        self.seq[-1] = obs
        info = {"portfolio_value": float(portfolio_value)}
        return self.seq.clone().numpy(), float(reward), done, False, info