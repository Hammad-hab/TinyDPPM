from torch import nn
import torch
import numpy as np
from ForwardDiffusion import ForwardDiffusion
from NoiseScheduler import NoiseScheduler
from UNET import UNET
from VersionManager import VersionManager
from util import _colorize_loss

class TrainProcess:
    def __init__(self, ns: NoiseScheduler, fd: ForwardDiffusion, unet:UNET, ds, vm: VersionManager, epochs: int) -> None:
        self.ns = ns
        self.ds = ds
        self.fd = fd
        self.vm = vm
        self.epoch = epochs
        self.current_epoch = 0
        self.model = unet
        self._loss = []
        self._valloss = []
        self._prev_loss = None
        
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.loss_criterion = nn.MSELoss()
        self._train_step = self._init_train_fn()
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.loss_criterion.to(device=device)

    def set_start_epoch(self, ep):
        self.current_epoch = ep
    
    def _init_mbgd(self, step_fn, loader):
        avg_loss = []
        mbgd_epoch = 1
        for x_batch, _ in loader:
            x0 = x_batch.to(torch.float32).to(self.device)
            t = torch.randint(0, self.ns.T, (x0.shape[0],)).to(self.device)
            x_t, eps = self.fd.getNoisyTensor(x0, t)
            mbgd_loss = step_fn(x_t, eps, t)
            print(f"[LOG] Running MGBD Epoch {mbgd_epoch} with loss {_colorize_loss(self._prev_loss, mbgd_loss)}")
            self._prev_loss = mbgd_loss
            avg_loss.append(mbgd_loss)
            mbgd_epoch += 1
    
        return np.mean(avg_loss)

    def _init_train_fn(self,):
        def step(x_t, eps, t):
            self.model.train()
            
            pred = self.model(x_t, t)
            loss = self.loss_criterion(pred, eps)
            loss.backward()
            
            self.optim.step()
            self.optim.zero_grad()

            return loss.item()
        return step

    def train(self):
        for _ in range(self.epoch):
            print(f"[LOG] Current epoch {self.current_epoch}")
            self.vm.setEpoch(self.current_epoch)
            loss = self._init_mbgd(self._train_step, self.ds.train_loader)
            self._loss.append(loss)
            self.vm.save()
            print(f'[LOG] loss: {loss}, epoch {self.current_epoch}')
            self.current_epoch += 1

    def getModel(self):
        return self.model