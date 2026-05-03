# model.py
import torch
import torch.nn as nn
from .config import *

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.head = nn.Linear(128, ACT_DIM)
        
    def forward(self, x):
        return self.head(self.backbone(x))
