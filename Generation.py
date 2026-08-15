
from Image import DiffusionImage
from TrainProcess import TrainProcess
from NoiseScheduler import NoiseScheduler
from ForwardDiffusion import ForwardDiffusion
from UNET import UNET
from Datasets import CIFAR
from VersionManager import VersionManager
import torch
import matplotlib.pyplot as plt

model = UNET()
ns = NoiseScheduler(1200, "cosine")
fd = ForwardDiffusion(ns)
cf = CIFAR()
vm = VersionManager(model, 'tinydppm')
vm.load_latest(True, True)

x = torch.randn(1, 3, 32, 32)
# cf = CIFAR()
# cfitr = iter(cf.train_loader)
# next(cfitr)
# img, label = next(cfitr)
# x0 = img[0]
# noisy, _ = fd.getNoisyImage(x0, 1000)
# x = (noisy._raw.view(1, 3, 32, 32))
model.eval()

img = DiffusionImage(x[0]).getAsPIL()
img.save("output.png")

with torch.no_grad():
    for t in reversed(range(1,ns.T+1)):
        tensor_t = torch.tensor([t-1])
        
        noise = model(x, tensor_t)
        x = fd.reverse(x, noise, t)
        if t % 100 == 0:
            print(
                t,
                "x:",
                x.min().item(),
                x.max().item(),
                x.mean().item(),
                x.std().item(),
                "| noise:",
                noise.mean().item(),
                noise.std().item()
            )
        img = DiffusionImage(x[0]).getAsPIL()
        img.save("output.png")
