from typing import Union

import torch, math
from Datasets import CIFAR, Flowers102
import matplotlib.pyplot as plt
from torchvision import transforms
from Image import DiffusionImage
from NoiseScheduler import NoiseScheduler

class ForwardDiffusion:
    def __init__(self, ns: NoiseScheduler):
        self.ns = ns
        self.T = ns.T
        self.device = 'cpu'

    def to(self, device):
        self.device=device

    def getNoisyImage(self, x0: Union[DiffusionImage, torch.Tensor], t):
        t = self.getNoisyTensor(x0._raw if isinstance(x0, DiffusionImage) else x0, t)
        return DiffusionImage(t[0][0]), t[1][0]

    def getNoisyTensor(self, x0, t):
        eps = torch.randn_like(x0, device=self.device)
        alpha = self.ns._alphab[t].view(-1, 1, 1, 1).to(self.device)
        xt = torch.sqrt(alpha) * x0 + torch.sqrt(1-alpha)*eps
        return xt, eps

    def reverse(self, x, noise, t):
        alpha_bar_t = self.ns._alphab[t]
        alpha_bar_prev = self.ns._alphab[t - 1]
        alpha_t = self.ns._alphas[t - 1]
        beta_t = self.ns._betas[t - 1]
    
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise) / torch.sqrt(alpha_bar_t)
        # x0_pred = x0_pred.clamp(-3, 3)  # latents aren't [-1,1] like pixels, pick a range that matches your VAE's latent stats
    
        coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)
        coef_xt = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
    
        mean = coef_x0 * x0_pred + coef_xt * x
    
        if t > 1:
            posterior_var = (
                (1 - alpha_bar_prev)
                / (1 - alpha_bar_t)
                * beta_t
            )
    
            return mean + torch.sqrt(posterior_var) * torch.randn_like(x)
    
        return mean

if __name__ == "__main__":
    ns = NoiseScheduler(1000, 'cosine')
    fd = ForwardDiffusion(ns)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    cf = Flowers102(transform)
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
