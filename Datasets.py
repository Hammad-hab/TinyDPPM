import sys

import torchvision
import torch


class MNIST:
    FOLDER='./data/mnist'
    
    def __init__(self, batch_size=4) -> None:
       self._train_set = None
       self._test_set = None
       self.load_data()      
       self.test_loader = self._create_loader(self._test_set, batch_size)
       self.train_loader = self._create_loader(self._train_set, batch_size)

    def load_data(self):
        self._train_set = torchvision.datasets.MNIST(root=MNIST.FOLDER, train=True, download=True)
        self._test_set = torchvision.datasets.MNIST(root=MNIST.FOLDER, train=False, download=True,)

    def _create_loader(self, data_set, batch):
        return torch.utils.data.DataLoader(data_set, batch,)

class CIFAR:
    FOLDER='./data/cifar10'
    
    def __init__(self, transform=None, batch_size=4) -> None:
       self._train_set = None
       self._test_set = None
       if transform:
           self._transform = transform
       else:
           self._transform = torchvision.transforms.ToTensor()
           
       self.load_data()      
       self.test_loader = self._create_loader(self._test_set, batch_size)
       self.train_loader = self._create_loader(self._train_set, batch_size)

    def load_data(self):
        self._train_set = torchvision.datasets.CIFAR10(root=CIFAR.FOLDER, train=True, download=True, transform=self._transform)
        self._test_set = torchvision.datasets.CIFAR10(root=CIFAR.FOLDER, train=False, download=True, transform=self._transform)

    def _create_loader(self, data_set, batch):
        return torch.utils.data.DataLoader(data_set, batch,)

if __name__ == "__main__":
    args = sys.argv[1:]
    if (args[0] == 'cifar'):
        cf = CIFAR()
    elif (args[0] == 'mnist'):
        mns = MNIST()
    else:
        print("Help:\n\tpython3 Data.py [cifar|mnist]")