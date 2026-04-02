import torch
import torch.nn.functional as F
from .config import *
from .model import SAC_Actor, SAC_Critic
from .replay_buffer import ReplayBuffer
from .env import TradingEnv
from collections import Counter
import random


class SACTrainer:
    def __init__(self):
        # ================= Networks =================
        self.actor = SAC_Actor().to(DEVICE)
        self.critic1 = SAC_Critic().to(DEVICE)
        self.critic2 = SAC_Critic().to(DEVICE)
        self.target1 = SAC_Critic().to(DEVICE)
        self.target2 = SAC_Critic().to(DEVICE)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        # ================= Optimizers =================
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) +
            list(self.critic2.parameters()),
            lr=LR_CRITIC
        )
        
        self.alpha = ALPHA

        # ================= Replay =================
        self.buffer = ReplayBuffer()
        
        self.total_steps = 0
        self.best_val_value = -1e18
        self.q1_running = []
        self.q2_running = []
        self.entropy_running = []

    def collect_episode(self, states, prices, training=True):
        env = TradingEnv(states, prices, random_start=training)
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        action_counter = Counter()
        while not done:
            state_tensor = torch.tensor(state,dtype=torch.float32,device=DEVICE)
            state_input = state_tensor.unsqueeze(0)
            if training and len(self.buffer) < MIN_REPLAY:
                action = torch.randint(0, ACT_DIM, (1,), device=DEVICE)
            else:
                if training:
                    action, _ = self.actor.sample(state_input)
                else:
                    with torch.no_grad():
                        logits = self.actor(state_input)
                        action = torch.argmax(logits, dim=-1)
            action_value = int(action.item())
            action_counter[action_value] += 1
            next_state, reward, done, _, info = env.step(action_value)
            if training:
                self.buffer.push(
                    state_tensor,
                    action_value,
                    float(reward),
                    torch.tensor(next_state, dtype=torch.float32),
                    bool(done)
                )
                self.total_steps += 1
                if self.total_steps % UPDATE_EVERY == 0 and len(self.buffer) >= BATCH_SIZE:
                    for _ in range(GRADIENT_STEPS):
                        self.train_step()
            state = next_state
            total_reward += reward
        final_value = info["portfolio_value"]
        return total_reward, final_value, action_counter

    def train_step(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()
        rewards = rewards.float()
        dones = dones.float()
        # ================= Critic Update =================
        with torch.no_grad():
            next_logits = self.actor(next_states)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            next_probs = next_log_probs.exp()
            q1_next = self.target1(next_states)
            q2_next = self.target2(next_states)
            min_q_next = torch.min(q1_next, q2_next)
            next_q = (next_probs * (
                min_q_next - self.alpha * next_log_probs
            )).sum(dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * GAMMA * next_q
        q1 = self.critic1(states)
        q2 = self.critic2(states)
        q1_a = q1.gather(1, actions)
        q2_a = q2.gather(1, actions)
        with torch.no_grad():
            self.q1_running.append(q1_a.mean().item())
            self.q2_running.append(q2_a.mean().item())
        critic_loss = (
            F.mse_loss(q1_a, target_q) +
            F.mse_loss(q2_a, target_q)
        )
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) +
            list(self.critic2.parameters()),
            1.0
        )
        self.critic_opt.step()
        
        # ================= Actor Update =================
        logits = self.actor(states)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=1).mean()
        q1_pi = self.critic1(states)
        q2_pi = self.critic2(states)
        min_q = torch.min(q1_pi, q2_pi)
        actor_loss = (
            (probs * (
                self.alpha * log_probs - min_q
            )).sum(dim=1)
        ).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            1.0
        )
        self.actor_opt.step()
        with torch.no_grad():
            self.entropy_running.append(entropy.item())

        # ================= Soft Update =================
        for p, tp in zip(self.critic1.parameters(), self.target1.parameters()):
            tp.data.copy_(tp.data * (1 - TAU) + p.data * TAU)
        for p, tp in zip(self.critic2.parameters(), self.target2.parameters()):
            tp.data.copy_(tp.data * (1 - TAU) + p.data * TAU)
            
    def validate(self, val_states, val_prices):
        self.actor.eval()
        total_value = 0.0
        with torch.no_grad():
            for states, prices in zip(val_states, val_prices):
                _, final_value, _ = self.collect_episode(
                    states, prices, training=False
                )
                total_value += final_value / INITIAL_CAPITAL
        self.actor.train()
        return total_value / len(val_states)

    def train(self,
              train_states,
              train_prices,
              tickers=None,
              val_states=None,
              val_prices=None):
        print("Training Started")
        for ep in range(1, EPISODES + 1):
            print("=" * 60)
            print(f"Episode {ep}/{EPISODES}")
            total_train_value = 0.0
            total_action_counter = Counter()
            if tickers is not None:
                combined = list(zip(train_states, train_prices, tickers))
                random.shuffle(combined)
                train_states, train_prices, tickers = zip(*combined)
            else:
                combined = list(zip(train_states, train_prices))
                random.shuffle(combined)
                train_states, train_prices = zip(*combined)
            for i, (states, prices) in enumerate(zip(train_states, train_prices)):
                _, final_value, action_counter = self.collect_episode(
                    states, prices, training=True
                )
                train_value = final_value / INITIAL_CAPITAL
                total_train_value += train_value  
                total_action_counter.update(action_counter)
                asset_name = tickers[i] if tickers is not None else f"Asset_{i}"
                print(f"  {asset_name:<10} Final Value: {train_value:.4f}")
                total_actions_asset = sum(action_counter.values())
                for a in range(ACT_DIM):
                    count = action_counter.get(a, 0)
                    pct = 100 * count / total_actions_asset if total_actions_asset > 0 else 0
                    print(f"    Action {a}: {pct:5.1f}%")
            print("Average final value:", total_train_value / len(train_states))
            print("Action Distribution (Train):")
            total_actions = sum(total_action_counter.values())
            for a in range(ACT_DIM):
                count = total_action_counter.get(a, 0)
                pct = 100 * count / total_actions if total_actions > 0 else 0
                print(f"  Action {a}: {count:4d} ({pct:6.2f}%)")
            avg_q1 = sum(self.q1_running) / len(self.q1_running) if self.q1_running else 0
            avg_q2 = sum(self.q2_running) / len(self.q2_running) if self.q2_running else 0
            avg_entropy = sum(self.entropy_running) / len(self.entropy_running) if self.entropy_running else 0
            print(
                f"Episode {ep} | "
                f"Avg Q1: {avg_q1:.4f} | "
                f"Avg Q2: {avg_q2:.4f} | "
                f"Entropy: {avg_entropy:.4f}"
            )
            # reset cho episode sau
            self.q1_running = []
            self.q2_running = []
            self.entropy_running = []
            val_value = None
            if val_states is not None and val_prices is not None and ep >= 10:
                val_value = self.validate(val_states, val_prices)
                if val_value > self.best_val_value:
                    self.best_val_value = val_value
                    torch.save(self.actor.state_dict(), str(MODEL_PATH))
                    print("New Best Model Saved")
            if val_value is not None:
                print(f"Validation Portfolio Value: {val_value:.4f}")
                print(f"Best Val Value: {self.best_val_value:.4f}")
            print("=" * 60)
        print("Training Finished")