from torch import nn
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self, indim=256, hiddendim=128, ltdim=32) -> None:
        super().__init__()
        self.h = nn.Linear(indim, hiddendim)
        self.mu = nn.Linear(hiddendim, ltdim)
        self.logvar = nn.Linear(hiddendim, ltdim)

    def forward(self, x):
        h = F.relu(self.h(x))
        mu = self.mu(h)
        logvar = self.logvar(h)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, indim=256, hiddendim=128, ltdim=32) -> None:
        super().__init__()
        self.zl = nn.Linear(ltdim, hiddendim)
        self.rst = nn.Linear(hiddendim, indim)

    def forward(self, z):
        x = F.relu(self.zl(z))
        x = self.rst(x)
        return F.sigmoid(x)


class VAE(nn.Module):
    def __init__(self, indim=256, hiddendim=128, ltdim=32) -> None:
        super().__init__()
        self.encoder = Encoder(indim, hiddendim, ltdim)
        self.decoder = Decoder(indim, hiddendim, ltdim)

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