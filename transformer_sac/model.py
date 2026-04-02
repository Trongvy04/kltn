# model.py
import torch
import torch.nn as nn
from .config import *

class TransformerBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = nn.LayerNorm(STATE_DIM)
        self.input_proj = nn.Linear(STATE_DIM, D_MODEL)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, SEQ_LEN, D_MODEL)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DFF,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS,
            enable_nested_tensor=False
        )
    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_proj(x)
        x = x + self.pos_emb[:, :x.size(1)]
        x = self.encoder(x)
        return x[:, -1]

# ================= ACTOR =================
class SAC_Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TransformerBackbone()
        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, ACT_DIM)
        )
    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)
    def sample(self, x):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob

# ================= CRITIC =================
class SAC_Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TransformerBackbone()
        self.q = nn.Linear(D_MODEL, ACT_DIM)
    def forward(self, s):
        feat = self.backbone(s)
        return self.q(feat)