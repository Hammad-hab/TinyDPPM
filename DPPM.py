from TrainProcess import TrainProcess
from NoiseScheduler import NoiseScheduler
from ForwardDiffusion import ForwardDiffusion
from UNET import UNET
from Datasets import CIFAR
from VersionManager import VersionManager
from util import get_device

EPOCHS, TIME = 1000, 1000

device = get_device()
    
ns = NoiseScheduler(TIME, 'cosine')
ns.to(device)
fd = ForwardDiffusion(ns)
fd.to(device)
ds = CIFAR(batch_size=32)
model = UNET()
model.to(device)
vm = VersionManager(model, 'tinydppm')
vm.load_latest(True, True)
procs = TrainProcess(ns, fd, model, ds, vm, EPOCHS)
procs.to(device)
procs.set_start_epoch(vm.startepoch)

@vm.save_on_fail
def main():
    procs.train()

main()