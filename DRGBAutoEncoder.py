from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.modules.loss import MSELoss
from torchvision import transforms
from Datasets import Flowers102
from PetalSelection import PetalSelection
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
            nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2, output_padding=1), # 32 → 64
            nn.SiLU()
        )
        
        self.l7 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, padding=1, stride=2, output_padding=1), # 64 → 128
            nn.SiLU()
        )
        
        self.l8 = nn.ConvTranspose2d(
            16, 3, 3, padding=1, stride=2, output_padding=1  # 128 → 256
        )
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        return F.tanh(x)
        
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
                delta_rgb = model(x0)
                reconstructed = x0 + delta_rgb
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

            print("epoch average:", avg_loss)

            # Epoch-level TensorBoard data
            writer.add_scalar("Loss/loss-drgb", avg_loss, epoch)
            writer.flush()

            vm.save()

    train()

    writer.close()