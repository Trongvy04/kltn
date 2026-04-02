import torch
import numpy as np
from transformer_sac.model import SAC_Actor

class ModelService:

    def __init__(self, model_path, device):
        self.device = device
        self.actor = SAC_Actor().to(self.device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        self.actor.load_state_dict(state_dict)
        self.actor.eval()

    def predict(self, state_seq):

        with torch.no_grad():
            state = torch.tensor(
                state_seq,
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            logits = self.actor(state)

            action = torch.argmax(logits, dim=-1).item()

        return action