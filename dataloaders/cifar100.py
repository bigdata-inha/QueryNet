import copy
from collections.abc import Iterable

import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from dataloaders.base import BaseLoader


class Cifar100Loader(BaseLoader):
    def __init__(self, batch_size=128):
        super(Cifar100Loader, self).__init__()
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.set_default_loader(batch_size)
        self._original_train_set = copy.deepcopy(self.train_set)
        self._original_valid_set = copy.deepcopy(self.valid_set)
        self.batch_size = batch_size

    def set_default_loader(self, batch_size=None):
        self.batch_size = self.batch_size if batch_size is None else batch_size

        self.train_set = datasets.CIFAR100("./data", train=True, transform=self.train_transform, download=True)
        self.valid_set = datasets.CIFAR100("./data", train=False, transform=self.valid_transform)

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False)

        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)

    def set_subclass_loader(self, sub_classes, batch_size=None):
        assert isinstance(sub_classes, Iterable)
        self.batch_size = self.batch_size if batch_size is None else batch_size

        sub_idx = []
        for idx, classes in enumerate(self.train_set.targets):
            if classes in sub_classes:
                sub_idx.append(idx)

        self.train_set = copy.deepcopy(self._original_train_set)
        self.train_set.data = self.train_set.data[sub_idx]
        self.train_set.targets = [self.train_set.targets[i] for i in sub_idx]

        sub_idx = []
        for idx, classes in enumerate(self.valid_set.targets):
            if classes in sub_classes:
                sub_idx.append(idx)

        self.valid_set = copy.deepcopy(self._original_valid_set)
        self.valid_set.data = self.valid_set.data[sub_idx]
        self.valid_set.targets = [self.valid_set.targets[i] for i in sub_idx]

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False)

        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)
