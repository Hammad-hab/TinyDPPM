from typing import Union

import torch, math
from Datasets import CIFAR
import matplotlib.pyplot as plt
from torchvision import transforms
from Image import DiffusionImage
from NoiseScheduler import NoiseScheduler

class ForwardDiffusion:
    def __init__(self, ns: NoiseScheduler):
        self.ns = ns
        self.T = ns.T

    def getNoisyImage(self, x0: Union[DiffusionImage, torch.Tensor], t):
        t = self.getNoisyTensor(x0._raw if isinstance(x0, DiffusionImage) else x0, t)
        return DiffusionImage(t[0][0]), t[1][0]

    def getNoisyTensor(self, x0, t):
        eps = torch.randn_like(x0)     
        alpha = self.ns._alphab[t].view(-1, 1, 1, 1)        
        xt = torch.sqrt(alpha) * x0 + torch.sqrt(1-alpha)*eps
        return xt, eps

    def reverse(self, x, noise, t) -> torch.Tensor:
        a0 = ((1-self.ns._alphas[t])/(torch.sqrt(1-self.ns._alphab[t])))*noise
        a1 = (1/torch.sqrt(self.ns._alphas[t]))*(x-a0)
        z = torch.randn_like(x)
        if t > 0:
                z = torch.randn_like(x)
                return a1 + torch.sqrt(self.ns._betas[t]) * z
        return a1
        
if __name__ == "__main__":
    ns = NoiseScheduler(1000, 'linear')
    fd = ForwardDiffusion(ns)
    cf = CIFAR(transforms.ToTensor())
    cfitr = iter(cf.train_loader)
    next(cfitr)
    img, label = next(cfitr)
    x0 = img[0]
    
    fig, axes = plt.subplots(1, 9, figsize=(36, 4))
    
    for i in range(40, 50):
        
        noisy, _ = fd.getNoisyImage(x0, i)
        axes[i - 41].imshow(noisy._raw.permute(1, 2, 0))
        axes[i - 41].axis("off")
    
    plt.savefig("output.png")