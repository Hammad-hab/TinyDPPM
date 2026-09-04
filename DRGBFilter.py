import torch
from torchvision import transforms

from Datasets import Flowers102
from Image import DiffusionImage
import torch
import torch.nn.functional as F


class CorruptImage:
    def __init__(self, p=0.8):
        self.p = p

    def __call__(self, img):
        # img: [C, H, W]
        img = img.clone()

        if torch.rand(()) > self.p:
            return img

        _, h, w = img.shape

        # Random region
        rh = torch.randint(h // 8, h // 2, ())
        rw = torch.randint(w // 8, w // 2, ())

        y = torch.randint(0, h - rh + 1, ())
        x = torch.randint(0, w - rw + 1, ())

        region = img[:, y:y+rh, x:x+rw]

        effect = torch.randint(0, 4, ()).item()

        if effect == 0:
            # Blacken
            img[:, y:y+rh, x:x+rw] = 0

        elif effect == 1:
            # Darken
            img[:, y:y+rh, x:x+rw] *= 0.1

        elif effect == 2:
            # Add noise
            noise = torch.randn_like(region) * 0.3
            img[:, y:y+rh, x:x+rw] = torch.clamp(
                region + noise, 0, 1
            )

        elif effect == 3:
            # Blur
            img[:, y:y+rh, x:x+rw] = F.avg_pool2d(
                region.unsqueeze(0),
                kernel_size=9,
                stride=1,
                padding=4
            ).squeeze(0)

        return img
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        CorruptImage()
    ])
    ds = Flowers102(transform)
    for x,_ in ds.train_loader:
        DiffusionImage(x[1]).view(True)
        break