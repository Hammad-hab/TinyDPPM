from Image import DiffusionImage
from TrainProcess import TrainProcess
from NoiseScheduler import NoiseScheduler
from ForwardDiffusion import ForwardDiffusion
from UNET import UNET
from Datasets import CIFAR
from VersionManager import VersionManager
import torch

model = UNET()
ns = NoiseScheduler(1000, "cosine")
fd = ForwardDiffusion(ns)
vm = VersionManager(model, 'tinydppm')
vm.load_epoch(196, True)

x = torch.randn(1, 3, 32, 32)


model.eval()

with torch.no_grad():
    t = 999
    tensor_t = torch.tensor([t])

    noise = model(x, tensor_t)

    x = fd.reverse(x, noise, t)
      
exit(0)
with torch.no_grad():
    for t in reversed(range(ns.T)):
        tensor_t = torch.tensor([t])
        
        noise = model(x, tensor_t)
        x = fd.reverse(x, noise, t)
        if t % 100 == 0:
                print(
                    t,
                    noise,
                    "min:", x.min().item(),
                    "max:", x.max().item(),
                    "mean:", x.mean().item(),
                    "std:", x.std().item()
                )

print('Saving image')
img = DiffusionImage(x[0]).getAsPIL()
img.save("output.png")
print(x.min().item(), x.max().item(), x.mean().item(), x.std().item())