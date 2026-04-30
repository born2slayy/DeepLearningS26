import torch
import torch.nn as nn
from pathlib import Path

class DeepClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)
    

    def save(self, save_dir: Path, suffix=None):
        '''
        Saves the model, adds suffix to filename if given
        '''

        ## TODO implement
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = "model.pth" if suffix is None else f"model_{suffix}.pth"
        save_path = save_dir / filename
        
        # state_dict
        torch.save(self.net.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load(self, path):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        
        ## TODO implement
        state_dict = torch.load(path, map_location=next(self.parameters()).device)
        self.net.load_state_dict(state_dict)
        print(f"Model loaded from {path}")