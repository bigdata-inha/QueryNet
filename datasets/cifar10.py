"""
Cifar100 Dataloader implementation
"""
import logging
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader


class Cifar10DataLoader:
    def __init__(self, config, subset_labels=None):
        self.config = config
        self.logger = logging.getLogger("Cifar10DataLoader")

        if config.data_mode == "download":
            self.logger.info("Loading DATA.....")

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

            valid_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

            train_set = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
            valid_set = datasets.CIFAR10("./data", train=False, transform=valid_transform)

            if subset_labels is not None:
                # make others validation set for binary
                d = dict()
                num_sample = (100 * len(subset_labels)) // (100 - len(subset_labels))
                for val in np.unique(valid_set.targets):
                    if val in subset_labels:
                        continue
                    d[str(val)] = np.where(valid_set.targets == val)
                    d[str(val)] = np.random.choice(d[str(val)][0], num_sample, replace=False)
                ix_val = np.concatenate([values for values in d.values()])

                train_subset_indices = [i for i, e in enumerate(train_set.targets) if e in subset_labels]
                valid_subset_indices = [i for i, e in enumerate(valid_set.targets) if e in subset_labels] + list(ix_val)

                train_set = torch.utils.data.dataset.Subset(train_set, train_subset_indices)
                valid_set = torch.utils.data.dataset.Subset(valid_set, valid_subset_indices)
        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

        self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False)

        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)

    def finalize(self):
        pass
