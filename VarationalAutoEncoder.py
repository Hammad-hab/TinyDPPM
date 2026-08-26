from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.modules.loss import BCELoss
from torchvision import transforms
from Datasets import Flowers102
from VersionManager import VersionManager
import numpy as np
import os
import errno
from torch.utils.tensorboard import SummaryWriter


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.h = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.mu = nn.Conv2d(64, 16, kernel_size=3, padding=1)
        self.logvar = nn.Conv2d(64, 16, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.h(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, indim=256, hiddendim=128, ltdim=32) -> None:
        super().__init__()
        self.convt1 = nn.ConvTranspose2d(
            16, 64, 3, stride=2, padding=1, output_padding=1
        )
        self.bnorm0 = nn.BatchNorm2d(64)
        self.convt12 = nn.ConvTranspose2d(
            64, 128, 3, stride=1, padding=1, output_padding=0
        )
        self.bnorm12 = nn.BatchNorm2d(128)
        self.convt21 = nn.ConvTranspose2d(
            128, 64, 3, stride=1, padding=1, output_padding=0
        )
        self.bnorm = nn.BatchNorm2d(64)
        self.convt2 = nn.ConvTranspose2d(
            64, 32, 3, stride=2, padding=1, output_padding=1
        )
        self.bnorm2 = nn.BatchNorm2d(32)
        self.convt3 = nn.ConvTranspose2d(
            32, 3, 3, stride=2, padding=1, output_padding=1
        )

    def forward(self, z):
        z = F.relu(self.bnorm0(self.convt1(z)))
        z = F.relu(self.bnorm12(self.convt12(z)))
        z = F.relu(self.bnorm(self.convt21(z)))
        z = F.relu(self.bnorm2(self.convt2(z)))
        z = self.convt3(z)
        return F.sigmoid(z)


class VAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparametize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    vm = VersionManager(model, "tinyvae")


    # TensorBoard
    writer = SummaryWriter("runs/tiny-vae")

    vm.load_latest(True, True)
    beta = 1.5

    @vm.save_on_fail
    def train():

        for epoch in range(vm.startepoch, 1000):
            with open("beta_read", "r") as f:
                beta = float(f.read().strip())
                print(f'[VAE] Received  beta={beta}')
                
            vm.setEpoch(epoch)

            print(f'[VAE_EPOCH] {epoch}')

            epoch_losses = []
            epoch_recons = []
            epoch_kls = []

            for mbgd_step, (x, _) in enumerate(cf.train_loader):

                reconstructed, mu, logvar = model(x)

                recon_loss = loss_criterion(reconstructed, x)

                kld_loss = -0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp()
                )

                loss = recon_loss + beta * kld_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_value = loss.item()

                epoch_losses.append(loss_value)
                epoch_recons.append(recon_loss.item())
                epoch_kls.append(kld_loss.item())

                # MBGD loss
                global_step = epoch * len(cf.train_loader) + mbgd_step

                writer.add_scalar(
                    "MBGD/loss",
                    loss_value,
                    global_step
                )
                writer.add_scalar(
                    "MBGD/kld_loss",
                    recon_loss.item(),
                    global_step
                )
                writer.add_scalar(
                    "MBGD/kld_loss",
                    kld_loss.item(),
                    global_step
                )

            # Epoch averages
            avg_loss = np.mean(epoch_losses)
            avg_recon = np.mean(epoch_recons)
            avg_kl = np.mean(epoch_kls)

            print("epoch average:", avg_loss)
            print("recon average:", avg_recon)
            print("kld average:", avg_kl)

            # Epoch-level TensorBoard data
            writer.add_scalar("Epoch/loss", avg_loss, epoch)
            writer.add_scalar("Epoch/Avg_reconstruction_loss", avg_recon, epoch)
            writer.add_scalar("Epoch/Avg_KL_loss", avg_kl, epoch)
            writer.add_scalar("Epoch/beta", beta, epoch)

            writer.flush()

            vm.save()

            model.eval()

            with torch.no_grad():
                x, _ = next(iter(cf.train_loader))
                recon_x, _, _ = model(x)

            model.train()

            import torchvision.utils as vutils

            comparison = torch.cat([x[:4], recon_x[:4]])
            grid = vutils.make_grid(comparison, nrow=4)

            vutils.save_image(
                grid,
                f"vae-tests/epoch_{epoch}_check.png"
            )

    train()

    writer.close()