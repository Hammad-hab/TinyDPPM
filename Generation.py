
from Image import DiffusionImage
from TrainProcess import TrainProcess
from NoiseScheduler import NoiseScheduler
from ForwardDiffusion import ForwardDiffusion
from UNET import UNET
from VersionManager import VersionManager
from torchvision import transforms
import torch
from util import get_device

EPOCHS, TIME = 1000, 1000

device = get_device()
model = UNET()
vm = VersionManager(model, "tiny-dppm")
ns = NoiseScheduler(1000, "cosine")
fd = ForwardDiffusion(ns)
vm.load_latest(True, True)

x = torch.randn(1, 3, 256, 256).to(device)

ns.to(device)
fd.to(device)
model.to(device)

model.eval()
with torch.no_grad():
    for t in reversed(range(1, TIME+1)):
        tensor_t = torch.tensor([t-1], device=device)
        noise = model(x, tensor_t)
        x = fd.reverse(x, noise, t)
        
img = DiffusionImage(x[0])
img.getAsPIL().save('output.png')