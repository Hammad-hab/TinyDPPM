import torch
from Image import DiffusionImage
from TrainProcess import TrainProcess
from NoiseScheduler import NoiseScheduler
from ForwardDiffusion import ForwardDiffusion
from UNET import UNET
from Datasets import Flowers102
from VarationalAutoEncoder import VAE
from VersionManager import VersionManager
from torchvision import transforms

model = UNET()
vae = VAE()
ns = NoiseScheduler(1000, "cosine")   # match training T
fd = ForwardDiffusion(ns)

vae_vm = VersionManager(vae, 'tinyvae')
vm = VersionManager(model, 'tiny-latentdppm')
vae_vm.load_latest(True, True)
vm.load_latest(True, True)
model.eval()
vae.eval()

import torch
z = torch.randn(4, 16, 32, 32)  # sample directly from N(0,1), no diffusion involved
with torch.no_grad():
    out = vae.decoder(z)
for i in range(4):
    DiffusionImage(out[i]).getAsPIL().save(f"vae_prior_sample_{i}.png")
exit()
transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
cf = Flowers102(transform)
x_img, _ = next(iter(cf.train_loader))
x_img = x_img[:1]

with torch.no_grad():
    mu, _ = vae.encoder(x_img)
    print("latent stats:", mu.min().item(), mu.max().item(), mu.mean().item(), mu.std().item())

    # sanity check 1: does the VAE reconstruct cleanly with NO diffusion involved?
    recon = vae.decoder(mu)
    DiffusionImage(recon[0]).getAsPIL().save("sanity_vae_only.png")

    # sanity check 2: partial noise + partial denoise
    t_start = 999  # moderate noise, not full 999
    tt = torch.tensor([t_start])
    x_noisy, _ = fd.getNoisyTensor(mu, tt)

    x = x_noisy.clone()
    for t in reversed(range(1, t_start+1)):
        tensor_t = torch.tensor([t])
        noise = model(x, tensor_t)
        x = fd.reverse(x, noise, t)

    recon_from_partial = vae.decoder(x)
    DiffusionImage(recon_from_partial[0]).getAsPIL().save("sanity_partial_denoise.png")