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
        return DiffusionImage(t[0]), t[1]

    def getNoisyTensor(self, x0, t):
        eps = torch.randn_like(x0)     
        alpha = self.ns._alphab[t].view(-1, 1, 1, 1)        
        xt = torch.sqrt(alpha) * x0 + torch.sqrt(1-alpha)*eps
        return xt, eps

if __name__ == "__main__":
    ns = NoiseScheduler(1000, 'cosine')
    fd = ForwardDiffusion(ns)
    cf = CIFAR(transforms.ToTensor())
    cfitr = iter(cf.train_loader)
    next(cfitr)
    img, label = next(cfitr)
    img = img[0]
    fig, axes = plt.subplots(1, 9, figsize=(18, 2))
    
    for i in range(1, 10):
        img, _ = fd.getNoisyImage(img, i)
        axes[i - 1].imshow(img._raw.permute(1, 2, 0))
        axes[i - 1].axis("off")

    plt.show()