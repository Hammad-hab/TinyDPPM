from TrainProcess import TrainProcess
from NoiseScheduler import NoiseScheduler
from ForwardDiffusion import ForwardDiffusion
from UNET import UNET
from Datasets import CIFAR
from VersionManager import VersionManager

EPOCHS, TIME = 1000, 1000

ns = NoiseScheduler(TIME, 'cosine')
fd = ForwardDiffusion(ns)
ds = CIFAR(batch_size=32)
model = UNET()
vm = VersionManager(model, 'tinydppm')
vm.load_latest(True, True)
procs = TrainProcess(ns, fd, model, ds, vm, EPOCHS)

@vm.save_on_fail
def main():
    procs.train()

main()