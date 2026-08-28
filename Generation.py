
from Image import DiffusionImage
from TrainProcess import TrainProcess
from NoiseScheduler import NoiseScheduler
from ForwardDiffusion import ForwardDiffusion
from UNET import UNET
from Datasets import Flowers102
from VersionManager import VersionManager
from torchvision import transforms

model = UNET()
vm = VersionManager(model, "tiny-dppm")
vm.load_latest(True, True)
