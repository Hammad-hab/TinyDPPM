import torch
from NoiseScheduler import NoiseScheduler
from ForwardDiffusion import ForwardDiffusion
from UNET import UNET
from Datasets import Flowers102
from Image import DiffusionImage
from torchvision import transforms
from util import get_device

device = get_device()
STEPS = 5000

ns = NoiseScheduler(1000, "cosine")
fd = ForwardDiffusion(ns)
ns.to(device)
fd.to(device)

model = UNET().to(device)   # fresh model, patched architecture
optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

# grab ONE real image and freeze it
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
ds = Flowers102(transform, batch_size=1)
x0, _ = next(iter(ds.train_loader))
x0 = x0.to(device)   # shape [1, 3, 256, 256]

model.train()
for step in range(STEPS):
    t = torch.randint(0, ns.T, (1,), device=device)
    x_t, eps = fd.getNoisyTensor(x0, t)
    pred = model(x_t, t)
    loss = loss_fn(pred, eps)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if step % 50 == 0:
        print(f"step {step}, loss {loss.item():.4f}")

# now sample from pure noise and see if it reconstructs that ONE flower
model.eval()
x = torch.randn(1, 3, 256, 256, device=device)
with torch.no_grad():
    for t in reversed(range(1, ns.T + 1)):
        tensor_t = torch.tensor([t], device=device)
        noise = model(x, tensor_t)
        x = fd.reverse(x, noise, t)

DiffusionImage(x[0]).getAsPIL().save("overfit_test.png")