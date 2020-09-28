import copy
import numpy as np
from collections.abc import Iterable

import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from dataloaders.base import BaseLoader


class Cifar10Loader(BaseLoader):
    def __init__(self, batch_size=128):
        super(Cifar10Loader, self).__init__()
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
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

        self.make_original_set(batch_size)

        self.batch_size = batch_size
        self.have_sub_classes = False

    def make_original_set(self, batch_size=None):
        self.batch_size = self.batch_size if batch_size is None else batch_size

        self.train_set = datasets.CIFAR10("./data", train=True, transform=self.train_transform, download=True)
        self.valid_set = datasets.CIFAR10("./data", train=False, transform=self.valid_transform)

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False)

        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)

    def make_subclass_set(self, sub_classes, batch_size=None):
        assert isinstance(sub_classes, Iterable)
        self.have_sub_classes = True
        self.batch_size = self.batch_size if batch_size is None else batch_size

        sub_idx = []
        for idx, classes in enumerate(self.train_set.targets):
            if classes in sub_classes:
                sub_idx.append(idx)

        self.sub_train_set = copy.deepcopy(self.train_set)
        self.sub_train_set.data = self.train_set.data[sub_idx]
        self.sub_train_set.targets = [self.train_set.targets[i] for i in sub_idx]

        sub_idx = []
        for idx, classes in enumerate(self.valid_set.targets):
            if classes in sub_classes:
                sub_idx.append(idx)

        self.sub_valid_set = copy.deepcopy(self.valid_set)
        self.sub_valid_set.data = self.valid_set.data[sub_idx]
        self.sub_valid_set.targets = [self.valid_set.targets[i] for i in sub_idx]

        self.sub_train_loader = torch.utils.data.DataLoader(self.sub_train_set, batch_size=self.batch_size, shuffle=True)
        self.sub_valid_loader = torch.utils.data.DataLoader(self.sub_valid_set, batch_size=self.batch_size, shuffle=False)

        self.sub_train_iterations = len(self.sub_train_loader)
        self.sub_valid_iterations = len(self.sub_valid_loader)
