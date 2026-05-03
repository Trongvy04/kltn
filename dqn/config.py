# config_dqn.py
import torch
from pathlib import Path

# ================= Device =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= Data / Environment =================
STATE_DIM = 16
INITIAL_CAPITAL = 100000.0
TRANSACTION_COST = 0.002

# ================= Actions =================
ACT_DIM = 3
ACTION_NAMES = {
    0: "ALLOC_0",
    1: "ALLOC_35",
    2: "ALLOC_70"
}

# ================= Training =================
LR = 1e-4
GAMMA = 0.97
BATCH_SIZE = 128
REPLAY_CAPACITY = 100_000
MIN_REPLAY = 5000
EPISODES = 30
EPISODE_LEN = 1200
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# ================= Paths =================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "dqn_model.pth"
DATA_FOLDER = BASE_DIR.parent / "data"
SCALER_PATH = BASE_DIR / "scaler.joblib"
