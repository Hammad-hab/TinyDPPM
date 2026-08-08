import os
import torch

class VersionManager:
    def __init__(self, model, name="model") -> None:
        self.dir = "versions/"
        self.name = name
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        self._model = model

    def save(self, epoch):
        torch.save(self._model.state_dict(), f"{self.dir}/{self.name}-{epoch}")

    def load_epoch(self, epoch, inplace=False):
        path=f"{self.dir}/{self.name}-{epoch}"
        if os.path.isfile(path):
            state=torch.load(path)
            if (inplace):
                self._model.load_state_dict(state)
            return torch.load(path)
        else:
            raise FileNotFoundError(f'Model Epoch {epoch} under the name {self.name}')
