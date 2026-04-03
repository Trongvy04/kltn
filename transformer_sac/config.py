# config.py
import torch
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#DATA
SEQ_LEN = 30
STATE_DIM = 16
TRANSACTION_COST = 0.002
INITIAL_CAPITAL = 100000.0


#Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "discrete_sac_actor.pth"
SCALER_PATH = BASE_DIR / "scaler.joblib"
DATA_FOLDER = BASE_DIR.parent / "data"

#TRANSFORMER
D_MODEL = 96
NHEAD = 4
NUM_LAYERS = 2
DFF = 192

#SAC
ACT_DIM = 3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4
GAMMA = 0.9
TAU = 0.005
ALPHA = 0.01
ACTION_NAMES = {
    0: "ALLOC_0",
    1: "ALLOC_35",
    2: "ALLOC_70"
}

BATCH_SIZE = 128
REPLAY_CAPACITY = 100_000
MIN_REPLAY = 5000
EPISODES = 30
EPISODE_LEN = 1200
UPDATE_EVERY = 1
GRADIENT_STEPS = 1