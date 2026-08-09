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

    def getRaw(self):
        return self._raw

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
