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
        """
        Saves the model, adds suffix to filename if given
        """

        model_name = self.net.__class__.__name__
        file_name = f"{model_name}_model.pth"
        if suffix is not None:
            file_name = f"{model_name}_model_{suffix}.pth"

        save_path = Path(save_dir) / file_name
        torch.save(self.state_dict(), save_path)

    def load(self, path):
        """
        Loads model from path
        Does not work with transfer model
        """

        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict)
