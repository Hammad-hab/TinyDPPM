from TrainProcess import TrainProcess
from NoiseScheduler import NoiseScheduler
from ForwardDiffusion import ForwardDiffusion
from UNET import UNET
from Datasets import Flowers102
from VersionManager import VersionManager
from util import get_device
from torchvision import transforms

EPOCHS, TIME = 1000, 1000

device = get_device()
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
print(f'[DEVICE] Using device {device}') 
ns = NoiseScheduler(TIME, 'cosine')
ns.to(device)
fd = ForwardDiffusion(ns)
fd.to(device)
ds = Flowers102(transform, batch_size=16)
model = UNET()
model.to(device)
vm = VersionManager(model, 'tiny-latentdppm')
vm.load_latest(True, True)
procs = TrainProcess(ns, fd, model, ds, vm, EPOCHS)
procs.to(device)
procs.set_start_epoch(vm.startepoch)

@vm.save_on_fail
def main():
    procs.train()

main()