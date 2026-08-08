import torch, math
from Data import CIFAR
import matplotlib.pyplot as plt
from torchvision import transforms

from NoiseScheduler import NoiseScheduler

class ForwardDiffusion:
    def __init__(self, ns: NoiseScheduler):
        self.ns = ns
        self.T = ns.T

    def getNoisyImage(self, x0, t):
        eps = torch.randn_like(x0)                          
        xt = torch.sqrt(self.ns._alphab[t]) * x0 + torch.sqrt(1-self.ns._alphab[t])*eps
        return xt,eps