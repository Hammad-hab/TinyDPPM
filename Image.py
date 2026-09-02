from PIL.Image import Image
import torch, math
from typing import Union
from Datasets import CIFAR
import matplotlib.pyplot as plt
from torchvision import transforms

    
class DiffusionImage:
    
    def __init__(self, contents: Union[torch.Tensor, Image]) -> None:
        if isinstance(contents, Image):
            contents = (transforms.ToTensor())(contents)
        self._raw = contents
        self._to_pil = transforms.ToPILImage()

    def to(self, device):
        self._raw = self._raw.to(device)
        return self._raw
        
    def getRaw(self):
        return self._raw

    @classmethod
    def generate(cls, generator, epoch=1, N=1):
        generator.load(epoch)
        tensors = generator.generate( N)
        return DiffusionImages([cls(x) for x in tensors])
   
    def getAsPIL(self):
        x = self._raw
        x = x.clamp(0, 1)   # data is already [0,1], no shift needed
        return transforms.ToPILImage()(x)
        
    def applyTransform(self, transforms):
        self._raw = transforms(self._raw)
        return self

    def __add__(self, other):
        return self._raw + other

    def __mul__(self, other):
        return self._raw * other

    def __sub__(self, other):
        return self._raw - other

    def view(self, show=False):
        _raw = self._raw.permute(1,2,0)
        plt.imshow(_raw)

        if show:
            plt.show()


class DiffusionImages(list[DiffusionImage]):
    def save_all(self, directory, prefix=""):
        for i, image in enumerate(self):
            image.getAsPIL().save(f"{directory}/{prefix}{i}.png")