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
        
    ...

if __name__ == "__main__":
    ds = Flowers102()
    ...
    