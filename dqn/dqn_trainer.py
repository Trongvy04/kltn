import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import Counter
from .config import *
from .replay_buffer import ReplayBuffer
from .env import TradingEnv
from .model import DQN

class DQNTrainer:
    def __init__(self):
        # ================= Networks =================
        self.q_net = DQN().to(DEVICE)
        self.target_net = DQN().to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LR)

        # ================= Replay =================
        self.buffer = ReplayBuffer()
        self.total_steps = 0
        self.epsilon = EPSILON_START
        self.best_val_value = -1e18

    def select_action(self, state, training=True):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        if training and random.random() < self.epsilon:
            return random.randint(0, ACT_DIM - 1)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            return torch.argmax(q_values, dim=-1).item()

    def collect_episode(self, states, prices, training=True):
        env = TradingEnv(states, prices, random_start=training)
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        action_counter = Counter()

        while not done:
            action = self.select_action(state, training)
            action_counter[action] += 1
            next_state, reward, done, _, info = env.step(action)
            if training:
                self.buffer.push(
                    torch.tensor(state, dtype=torch.float32),
                    action,
                    float(reward),
                    torch.tensor(next_state, dtype=torch.float32),
                    done
                )
                self.total_steps += 1
                if len(self.buffer) >= BATCH_SIZE:
                    self.train_step()
                # decay epsilon
                self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

            state = next_state
            total_reward += reward

        final_value = info["portfolio_value"]
        return total_reward, final_value, action_counter

    def train_step(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()
        # ================= Q-target =================
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * GAMMA * next_q

        q_values = self.q_net(states).gather(1, actions)
        loss = F.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # ================= Soft Update Target =================
        for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
            tp.data.copy_(0.995 * tp.data + 0.005 * p.data)  # soft update tau=0.005

    def validate(self, val_states, val_prices):
        self.q_net.eval()
        total_value = 0.0
        with torch.no_grad():
            for states, prices in zip(val_states, val_prices):
                _, final_value, _ = self.collect_episode(states, prices, training=False)
                total_value += final_value / INITIAL_CAPITAL
        self.q_net.train()
        return total_value / len(val_states)

    def train(self, train_states, train_prices, tickers=None, val_states=None, val_prices=None):
        print("Training Started")
        for ep in range(1, EPISODES + 1):
            print("=" * 60)
            print(f"Episode {ep}/{EPISODES}")
            total_train_value = 0.0
            total_action_counter = Counter()

            # Shuffle data
            combined = list(zip(train_states, train_prices, tickers)) if tickers else list(zip(train_states, train_prices))
            random.shuffle(combined)
            if tickers:
                train_states, train_prices, tickers = zip(*combined)
            else:
                train_states, train_prices = zip(*combined)

            for i, (states, prices) in enumerate(zip(train_states, train_prices)):
                _, final_value, action_counter = self.collect_episode(states, prices, training=True)
                train_value = final_value / INITIAL_CAPITAL
                total_train_value += train_value
                total_action_counter.update(action_counter)
                asset_name = tickers[i] if tickers else f"Asset_{i}"
                print(f"  {asset_name:<10} Final Value: {train_value:.4f}")

            # Summary
            print("Average final value:", total_train_value / len(train_states))
            print("Action Distribution (Train):")
            total_actions = sum(total_action_counter.values())
            for a in range(ACT_DIM):
                count = total_action_counter.get(a, 0)
                pct = 100 * count / total_actions if total_actions > 0 else 0
                print(f"  Action {a}: {count:4d} ({pct:6.2f}%)")

            # Validation
            if val_states and val_prices:
                val_value = self.validate(val_states, val_prices)
                if val_value > self.best_val_value:
                    self.best_val_value = val_value
                    torch.save(self.q_net.state_dict(), str(MODEL_PATH))
                    print("New Best Model Saved")
                print(f"Validation Portfolio Value: {val_value:.4f}")
                print(f"Best Val Value: {self.best_val_value:.4f}")

            print("=" * 60)
        print("Training Finished")