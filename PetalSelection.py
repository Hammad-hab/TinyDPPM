import torch
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
from Datasets import Flowers102
from Image import DiffusionImage
import torch.nn.functional as F


class PetalSelection:
    def __init__(self, thres=0.5, out_bright_mul=2.0):
        self.thres = thres
        self.out_bright_mul = out_bright_mul

    def __call__(self, img):
        noise = torch.rand_like(img)
    
        noisy = img - noise
        mask = noisy >= self.thres
        out = torch.where(mask, noisy + noise, 0)
        
        return F.avg_pool2d(
            out.unsqueeze(0),
            kernel_size=5,
            stride=1,
            padding=2
        ).squeeze(0)*self.out_bright_mul
        

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        PetalSelection(thres=0.25, out_bright_mul=1.5)
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