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
        if (self._epoch == 0):
            print('Not Saving at 0th epoch')
            return
            
        path = f"{self.dir}/{self.name}-{self._epoch}.pth"
        if not os.path.isfile(path):
            torch.save(self._model.state_dict(), path)
        else:
            raise FileExistsError(f'Program refused to override already exisitng model file {self.name}-{self._epoch}')

    def load_latest(self, inplace=False, silent=False):
            prefix = f"{self.name}-"
            epochs = []
            for f in os.listdir(self.dir):
                if f.startswith(prefix) and f.endswith(".pth"):
                    try:
                        epochs.append(int(f[len(prefix):-4]))
                    except ValueError:
                        continue
            if not epochs:
                if silent:
                    print('[WARNING]: This is the first time the model is being trained, no previous saves exist')
                    return None
                raise FileNotFoundError(f'No saved models found under the name {self.name}')
            latest = max(epochs)
            self._epoch = latest
            return self.load_epoch(latest, inplace=inplace)
        
    def load_epoch(self, epoch, inplace=False):
        path=f"{self.dir}/{self.name}-{epoch}.pth"
        if os.path.isfile(path):
            state = torch.load(path)
            if (inplace):
                self._model.load_state_dict(state)
            return state
        else:
            raise FileNotFoundError(f'Model Epoch {epoch} under the name {self.name}')

    def _try_failed_save(self):
        while os.path.isfile(f"{self.dir}/{self.name}-{self._epoch}.pth"):
            self._epoch += 1 # Keep tyring to find the right epoch
        self.save()
        
    def save_on_fail(self, clbck):
        def safe(*args, **kwargs):
            try:
                return clbck(*args, **kwargs)
            except KeyboardInterrupt:
                self._try_failed_save()
                raise
            except Exception:
                self._try_failed_save()
                raise
    
        return safe