from Image import DiffusionImage
from NoiseScheduler import NoiseScheduler
from ForwardDiffusion import ForwardDiffusion
from UNET import UNET
from VersionManager import VersionManager
import torch
from util import get_device

EPOCHS, TIME = 1000, 1000
N_SAMPLES = 6

device = get_device()
model = UNET()
vm = VersionManager(model, "tiny-dppm")
ns = NoiseScheduler(1000, "cosine")
fd = ForwardDiffusion(ns)
vm.load_latest(True, True)

x = torch.randn(N_SAMPLES, 3, 256, 256).to(device)
ns.to(device)
fd.to(device)
model.to(device)
model.eval()

with torch.no_grad():
    for t in reversed(range(1, TIME + 1)):
        tensor_t = torch.full((N_SAMPLES,), t, device=device)
        noise = model(x, tensor_t)
        x = fd.reverse(x, noise, t)

for i in range(N_SAMPLES):
    img = DiffusionImage(x[i])
    img.getAsPIL().save(f'output_{i}.png')
