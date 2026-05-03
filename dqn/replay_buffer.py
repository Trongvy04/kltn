import random
import torch
from collections import deque
from .config import *

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=REPLAY_CAPACITY)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            state.detach().cpu(),
            action,
            reward,
            next_state.detach().cpu(),
            done
        ))
    def sample(self):
        batch = random.sample(self.buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states).to(DEVICE)
        next_states = torch.stack(next_states).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        return states, actions, rewards, next_states, dones
    def __len__(self):
        return len(self.buffer)