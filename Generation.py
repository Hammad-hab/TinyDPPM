
from Image import DiffusionImage
from TrainProcess import TrainProcess
from NoiseScheduler import NoiseScheduler
from ForwardDiffusion import ForwardDiffusion
from UNET import UNET
from Datasets import CIFAR
from VarationalAutoEncoder import VAE
from VersionManager import VersionManager
import torch
import matplotlib.pyplot as plt

model = UNET()
vae = VAE()
ns = NoiseScheduler(1200, "cosine")
fd = ForwardDiffusion(ns)
cf = CIFAR()

vae_vm = VersionManager(vae, 'tinyvae')
vm = VersionManager(model, 'tinydppm')

vae_vm.load_latest(True, True)
vm.load_latest(True, True)

x = torch.randn(1, 16, 32, 32)

model.eval()

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
            
    x = vae.decoder(x)  
    img = DiffusionImage(x[0]).getAsPIL()
    img.save("output.png")
