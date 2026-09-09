from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.modules.loss import MSELoss
from torchvision import transforms
from Datasets import Flowers102
from transforms import ImageWarp, PetalSelection
from VersionManager import VersionManager
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class DRGBAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.l1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, stride=2),
            nn.SiLU()
        )
        
        self.l2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU()
        )
        
        self.l3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.SiLU()
        )
        
        self.l4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU()
        )

        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, padding=1, stride=1), # 32 → 32
            nn.SiLU()
        )
        
        self.l6 = nn.Sequential(
            # Skip features, x5+x3, dimensions add
            nn.ConvTranspose2d(64+64, 32, 3, padding=1, stride=2, output_padding=1), # 32 → 64
            nn.SiLU()
        )
        
        self.l7 = nn.Sequential(
            # Skip features, x6+x2, dimensions add
            nn.ConvTranspose2d(32+32, 16, 3, padding=1, stride=2, output_padding=1), # 64 → 128
            nn.SiLU()
        )
        
        self.l8 = nn.ConvTranspose2d(
            # Skip features, x7+x1, dimensions add
            16+16, 3, 3, padding=1, stride=2, output_padding=1  # 128 → 256
        )
    def forward(self, x):
        x1 = self.l1(x) # 128x128 16ch
        x2 = self.l2(x1) # 64x64 32ch
        x3 = self.l3(x2) # 32x32 64ch
        
        x4 = self.l4(x3) # 32x32 64 ch
        x5 = self.l5(x4) # 32x32 64 ch
        
        x6 = self.l6(torch.cat([x5, x3], dim=1)) # 64x64 32 ch
        x7 = self.l7(torch.cat([x6, x2], dim=1)) # 128x128 16 ch 
        x8 = self.l8(torch.cat([x7, x1], dim=1)) # 256x256 3ch
        # Note: torch.cat adds the channels. so....
        return F.tanh(x8)
        
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    cf = Flowers102(transform)
    model = DRGBAutoEncoder()

    loss_criterion = MSELoss(reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    selector = PetalSelection(thres=0.25, out_bright_mul=1.5)
    warpimg = ImageWarp(strength=1.0)
    
    vm = VersionManager(model, "tiny-drgbae", dir="versions/tiny-drgbae/")
    writer = SummaryWriter("runs/tiny-drgbae")
    vm.load_latest(True, True)

    @vm.save_on_fail
    def train():
        model.train()
        for epoch in range(vm.startepoch, 1000):
            epoch_losses = []
            vm.setEpoch(epoch)
            for mbgd_step, (x, _) in enumerate(cf.train_loader):
                x0 = selector(x)
                x0 = warpimg(x0)
                
                delta_rgb = model(x0)
                reconstructed = torch.clamp(x0 + delta_rgb, 0, 1)
                loss = loss_criterion(reconstructed, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_value = loss.item()

                epoch_losses.append(loss_value)

                global_step = epoch * len(cf.train_loader) + mbgd_step

                writer.add_scalar(
                    "Loss/train-drgb",
                    loss_value,
                    global_step
                )

            # Epoch averages
            avg_loss = np.mean(epoch_losses)

            print("Epoch average:", avg_loss)

            # Epoch-level TensorBoard data
            writer.add_scalar("Loss/loss-drgb", avg_loss, epoch)
            writer.flush()

            vm.save()

    train()

    writer.close()