from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.modules.loss import BCELoss
from torchvision import transforms
from Datasets import Flowers102
from VersionManager import VersionManager
import numpy as np


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.h = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  
            nn.ReLU(),
        )
        self.mu = nn.Conv2d(64, 3, kernel_size=3, padding=1)      # narrow to 3 only HERE
        self.logvar = nn.Conv2d(64, 3, kernel_size=3, padding=1)  # same

    def forward(self, x):
        h = self.h(x)
        mu = self.mu(h)
        logvar = self.logvar(h)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, indim=256, hiddendim=128, ltdim=32) -> None:
        super().__init__()
        self.convt1 = nn.ConvTranspose2d(3, 64, 3, stride=2, padding=1, output_padding=1)
        self.bnorm = nn.BatchNorm2d(64)
        self.convt2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.bnorm2 = nn.BatchNorm2d(32)
        self.convt3 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        z = F.relu(self.bnorm(self.convt1(z)))
        z = F.relu(self.bnorm2(self.convt2(z)))
        z = self.convt3(z)
        return F.sigmoid(z)


class VAE(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparametize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + std*eps
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametize(mu, logvar)
        x = self.decoder(z)
        return x, mu, logvar


if __name__ == "__main__":
    model = VAE()
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    cf = Flowers102(transform)
    loss_criterion = BCELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    vm = VersionManager(model, "tinyvae")
    vm.load_latest(True, True)
    @vm.save_on_fail
    def train():
        for epoch in range(vm.startepoch,1000):
            vm.setEpoch(epoch)  
            print(f'[VAE_EPOCH] {epoch}')
            epoch_losses = []
            epoch_recons = []
            epoch_kls = []
            
            for x, _ in cf.train_loader:
                reconstructed, mu, logvar = model(x)
            
                recon_loss = loss_criterion(reconstructed, x)
                kld_loss = -0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp()
                )
            
                loss = recon_loss + kld_loss
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                epoch_losses.append(loss.item())
                epoch_recons.append(recon_loss.item())
                epoch_kls.append(kld_loss.item())
            
            print("epoch average:", np.mean(epoch_losses))
            print("recon average:", np.mean(epoch_recons))
            print("kld average:", np.mean(epoch_kls))
            vm.save()
            model.eval()
            with torch.no_grad():
                x, _ = next(iter(cf.train_loader))
                recon_x, _, _ = model(x)
            model.train()
            
            import torchvision.utils as vutils
            comparison = torch.cat([x[:4], recon_x[:4]])
            grid = vutils.make_grid(comparison, nrow=4)
            vutils.save_image(grid, f"vae-tests/epoch_{epoch}_check.png")

    train()