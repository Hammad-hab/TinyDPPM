import os
import torch

class VersionManager:
    def __init__(self, model, name="model") -> None:
        self.dir = "versions/"
        self.name = name
        self.startepoch=0
        self._epoch = 0
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        self._model = model

    def setEpoch(self, e):
        self._epoch = e;

        
    def save(self, failprotocol=False):
        if self._epoch == 0:
            print('Not Saving at 0th epoch')
            return
    
        path = f"{self.dir}/{self.name}-{self._epoch}{'.f' if failprotocol else ''}.pth"
        torch.save(self._model.state_dict(), path)
    
        if not failprotocol:
            failed = f"{self.dir}/{self.name}-{self._epoch}.f.pth"
            if os.path.isfile(failed):
                print(f'[INFO] Removing Failed Epoch{failed}')
                os.remove(failed)
                

    def load_latest(self, inplace=False, silent=False):
        prefix = f"{self.name}-"
        epochs = []
    
        for f in os.listdir(self.dir):
            if not f.startswith(prefix):
                continue
    
            name = f[len(prefix):]
    
            if name.endswith(".f.pth"):
                print('[INFO] Discovered a faulty save. This means that the faulty epoch will have to be computed again.')
                name = name[:-6]  # remove ".f.pth"
            elif name.endswith(".pth"):
                name = name[:-4]
            else:
                continue
    
            try:
                epochs.append(int(name))
            except ValueError:
                continue
            
        if not epochs:
            if silent:
                print('[WARNING]: This is the first time the model is being trained, no previous saves exist')
                return None
            raise FileNotFoundError(
                f'No saved models found under the name {self.name}'
            )
    
        latest = max(epochs)
        
        path, status = self._filepth(
            f"{self.dir}/{self.name}-{latest}"
        )
        
        self._epoch = latest
        
        if status == 1:
            # Failed epoch: redo it
            self.startepoch = latest
        else:
            # Completed epoch: move to the next one
            self.startepoch = latest + 1
        
        print(f"[{self.name}] Found latest epoch {latest}")
        return self.load_epoch(latest, inplace=inplace)

    def _filepth(self, path):
        if os.path.isfile(path + '.pth'):
            return path + '.pth', 0
        elif os.path.isfile(path + '.f.pth'):
            return path + '.f.pth', 1
            
        raise FileNotFoundError(f'Could not find file {path}')
        
    def load_epoch(self, epoch, inplace=False):
        path, status = self._filepth(f"{self.dir}/{self.name}-{epoch}")
        if (status == 1):
            print(f'[WARNING] epoch {epoch} was not trained completely, it might not behave properly')
        state = torch.load(path)
        if (inplace):
                self._model.load_state_dict(state)
        return state

    def _try_failed_save(self):
        self.save(True)
        
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