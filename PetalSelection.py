import torch
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
from Datasets import Flowers102
from Image import DiffusionImage
import torch.nn.functional as F


def perlin_noise(
    height,
    width,
    scale=8,
    device=None,
):
    # Grid of random gradient vectors
    gh = height // scale + 2
    gw = width // scale + 2

    angles = torch.rand(gh, gw, device=device) * 2 * torch.pi

    gradients = torch.stack([
        torch.cos(angles),
        torch.sin(angles)
    ], dim=-1)

    # Pixel coordinates in grid space
    y = torch.arange(height, device=device) / scale
    x = torch.arange(width, device=device) / scale

    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Integer grid coordinates
    x0 = xx.floor().long()
    y0 = yy.floor().long()

    # Fractional coordinates
    xf = xx - x0
    yf = yy - y0

    # Fade curve: 6t^5 - 15t^4 + 10t^3
    u = xf * xf * xf * (xf * (xf * 6 - 15) + 10)
    v = yf * yf * yf * (yf * (yf * 6 - 15) + 10)

    # Dot products with the four surrounding gradients
    def dot(gx, gy):
        g = gradients[gy, gx]
        dx = xx - gx
        dy = yy - gy
        return g[..., 0] * dx + g[..., 1] * dy

    n00 = dot(x0,     y0)
    n10 = dot(x0 + 1, y0)
    n01 = dot(x0,     y0 + 1)
    n11 = dot(x0 + 1, y0 + 1)

    # Bilinear interpolation
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)

    return nx0 + v * (nx1 - nx0)

class PetalSelection:
    def __init__(self, thres=0.5, out_bright_mul=2.0):
        self.thres = thres
        self.out_bright_mul = out_bright_mul

    def __call__(self, img):
        noise = torch.rand_like(img)
        C, H, W = img.shape
        
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=img.device),
            torch.linspace(-1, 1, W, device=img.device),
            indexing="ij"
        )
        
        dist = torch.sqrt(x**2 + y**2)-0.5
        
        circle = (1 - dist).clamp(0, 1)
        circle = circle.unsqueeze(0).expand_as(img)

        noisy = img * circle - noise
        mask = noisy >= self.thres
        out = torch.where(mask, noisy, torch.zeros_like(img))
        out = torch.roll(out, 20) + noise*perlin_noise(256, 256, 32)
        return out*2*circle

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        PetalSelection(thres=0.25, out_bright_mul=1.5),
    ])

    ds = Flowers102(transform)
    dataset = ds.train_loader.dataset  # underlying Dataset, not just the first batch

    n_samples = 16
    indices = random.sample(range(len(dataset)), n_samples)

    imgs = []
    for idx in indices:
        img, _ = dataset[idx]
        imgs.append(img.clamp(0, 1))  # clamp in case out_bright_mul pushes values >1

    grid = make_grid(imgs, nrow=4, padding=2)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title(f"{n_samples} random petal selections from full dataset")
    plt.show()