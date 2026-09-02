from NoiseScheduler import NoiseScheduler
from ForwardDiffusion import ForwardDiffusion
from UNET import UNET
from VersionManager import VersionManager
import torch
from util import get_device


class Generator:
    TIME=1000

    def __init__(self, name='tiny-dppm') -> None:
        self._device = get_device()
        self.model = UNET()
        self._name = name
        self.vm = VersionManager(self.model, self._name)
        self.ns = NoiseScheduler(Generator.TIME, "cosine")
        self.fd = ForwardDiffusion(self.ns)
        self.ns.to(self._device)
        self.fd.to(self._device)
        self.model.to(self._device)
        self.loaded_epoch = None

        pass

    def load(self, epoch):
        self.vm.load_epoch(epoch, True)
        self.loaded_epoch = epoch

    def generate(self, N):
        if not self.loaded_epoch:
            raise ValueError('No Epoch Loaded, load an epoch with .load before generating')
        self.model.eval()
        x = torch.randn(N, 3, 256, 256, device=self._device)
        with torch.no_grad():
            for t in reversed(range(1, Generator.TIME + 1)):
                tensor_t = torch.full((N,), t, device=self._device)
                noise = self.model(x, tensor_t)
                x = self.fd.reverse(x, noise, t)
        return x
