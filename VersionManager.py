import os
import torch

class VersionManager:
    def __init__(self, model, name="model") -> None:
        self.dir = "versions/"
        self.name = name
        self._epoch = 0
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        self._model = model

    def setEpoch(self, e):
        self._epoch = e;
        
    def save(self):
        path = f"{self.dir}/{self.name}-{self._epoch}"
        if not os.path.isfile(path):
            torch.save(self._model.state_dict(), path)
        else:
            raise FileExistsError(f'Program refused to override already exisitng model file {self.name}-{self._epoch}')
            
    def load_epoch(self, epoch, inplace=False):
        path=f"{self.dir}/{self.name}-{epoch}"
        if os.path.isfile(path):
            state = torch.load(path)
            if (inplace):
                self._model.load_state_dict(state)
            return state
        else:
            raise FileNotFoundError(f'Model Epoch {epoch} under the name {self.name}')

    def _try_failed_save(self):
        while os.path.isfile(f"{self.dir}/{self.name}-{self._epoch}"):
            self._epoch += 1 # Keep tyring to find the right epoch
        self.save()
        
    def save_on_fail(self, clbck):
        def safe(*args, **kwargs):
            try:
                return clbck(*args, **kwargs)
            except (KeyboardInterrupt, Exception):
                self._try_failed_save()
                raise
    
        return safe