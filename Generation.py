from Image import DiffusionImage
from TrainProcess import TrainProcess
from NoiseScheduler import NoiseScheduler
from ForwardDiffusion import ForwardDiffusion
from UNET import UNET
from Datasets import CIFAR
from VersionManager import VersionManager
import torch

model = UNET()
ns = NoiseScheduler(10000, "cosine")
fd = ForwardDiffusion(ns)
vm = VersionManager(model, 'tinydppm')
vm.load_latest(True, True)

x = torch.randn(1, 3, 32, 32)

model.eval()

with torch.no_grad():
    for t in reversed(range(10000, ns.T)):
        tensor_t = torch.tensor([t])

        noise = model(x, tensor_t)

        x = fd.reverse(x, noise, t)

img = DiffusionImage(x[0]).getAsPIL()
img.save("output.png")