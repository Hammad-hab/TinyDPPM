from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.modules.loss import BCELoss
from torchvision import transforms
from Datasets import Flowers102
from VersionManager import VersionManager
import numpy as np
import os
import errno
from torch.utils.tensorboard import SummaryWriter

class ColourCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.l1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, stride=2),
            nn.SiLU()
        )
        
        self.l2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU()
        )
        
        self.l3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.SiLU()
        )
        
        self.l4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU()
        )

        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, padding=1, stride=1), # 32 → 32
            nn.SiLU()
        )
        
        self.l6 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2, output_padding=1), # 32 → 64
            nn.SiLU()
        )
        
        self.l7 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, padding=1, stride=2, output_padding=1), # 64 → 128
            nn.SiLU()
        )
        
        self.l8 = nn.ConvTranspose2d(
            16, 3, 3, padding=1, stride=2, output_padding=1  # 128 → 256
        )
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        return x
        
if __name__ == "__main__":
    ds = Flowers102()
    ...
    