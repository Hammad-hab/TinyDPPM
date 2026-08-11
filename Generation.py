from TrainProcess import TrainProcess
from NoiseScheduler import NoiseScheduler
from ForwardDiffusion import ForwardDiffusion
from UNET import UNET
from Datasets import CIFAR
from VersionManager import VersionManager
import torch

model = UNET()
fd = ForwardDiffusion()
vm = VersionManager(model)
vm.load_latest(True, True)

x = torch.randn(
    1,       # number of images
    3,       # CIFAR channels
    32,      # height
    32       # width
)

model.eval()

with torch.no_grad():
    for t in reversed(range(1000)):
        tensor_t = torch.tensor([t])
        noise = model(x, tensor_t)
        x = fd.reverse(x, noise)
        