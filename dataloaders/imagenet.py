import os
import copy
import numpy as np
from collections.abc import Iterable
from collections import Counter

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from dataloaders.base import BaseLoader


class ImageNetLoader(BaseLoader):
    def __init__(self, batch_size=32):
        super(ImageNetLoader, self).__init__()
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.root = "D:\DataSet\imagenet_256x256"

        self.train_transform = transforms.Compose([
            # transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        self.valid_transform = transforms.Compose([
            # transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        self.set_default_loader(batch_size)
        self.batch_size = batch_size

    def set_default_loader(self, batch_size=None):
        self.batch_size = self.batch_size if batch_size is None else batch_size

        self.train_set = datasets.ImageFolder(os.path.join(self.root, "train"), transform=self.train_transform)
        self.valid_set = datasets.ImageFolder(os.path.join(self.root, "val"), transform=self.valid_transform)

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False)

        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)

    def set_one_class_loader(self, sub_class, batch_size=None):
        assert isinstance(sub_class, int)
        self.batch_size = self.batch_size if batch_size is None else batch_size

        dic_class_cnt = Counter(self.train_set.targets)  # class당 데이터 갯수
        idx = 0
        for k, v in dic_class_cnt.items():
            if k == sub_class:
                break
            else:
                idx += dic_class_cnt[k]
        n_samples_in_class = dic_class_cnt[sub_class]
        new_samples = self.train_set.samples[idx:idx + n_samples_in_class]
        new_targets = self.train_set.targets[idx:idx + n_samples_in_class]

        self.sub_train_set = copy.deepcopy(self.train_set)
        self.sub_train_set.samples = self.sub_train_set.imgs = new_samples
        self.sub_train_set.targets = new_targets

        self.sub_train_loader = torch.utils.data.DataLoader(self.sub_train_set, batch_size=self.batch_size, shuffle=True)
        self.sub_train_iterations = len(self.sub_train_loader)

    def set_subclass_loader(self, sub_classes, batch_size=None):
        assert isinstance(sub_classes, Iterable)
        self.batch_size = self.batch_size if batch_size is None else batch_size

        idx_to_new_idx = {c: i + 1 for i, c in enumerate(sub_classes)}  # mapping to new target idx / 0: others
        # sub train set: half of sub classes, half of others
        dic_class_cnt = Counter(self.train_set.targets)     # class당 데이터 갯수
        new_samples, new_targets = [], []
        n_sample_sub_classes = 0
        for subcls in sub_classes:
            n_sample_sub_classes += dic_class_cnt[subcls]
        idx = 0
        while idx < len(self.train_set.targets):
            target = self.train_set.targets[idx]
            if target in sub_classes:
                n_samples_per_class = dic_class_cnt[target]
                new_samples.extend(self.train_set.samples[idx:idx + n_samples_per_class])
                new_targets.extend(self.train_set.targets[idx:idx + n_samples_per_class])
            else:
                n_samples_per_class = n_sample_sub_classes // (len(dic_class_cnt) - len(sub_classes))
                new_samples.extend(self.train_set.samples[idx:idx + n_samples_per_class])
                new_targets.extend(self.train_set.targets[idx:idx + n_samples_per_class])
            idx += dic_class_cnt[target]

        for i, (sample, target) in enumerate(zip(new_samples, new_targets)):
            if target in sub_classes:
                new_label = idx_to_new_idx[target]
                new_samples[i] = (sample[0], new_label)
                new_targets[i] = new_label
            else:
                new_label = 0
                new_samples[i] = (sample[0], new_label)
                new_targets[i] = new_label

        self.sub_train_set = copy.deepcopy(self.train_set)
        self.sub_train_set.samples = self.sub_train_set.imgs = new_samples
        self.sub_train_set.targets = new_targets

        # sub valid set: use all data, but change others to label 0
        # new_samples, new_targets = [], []
        # for i, (sample, target) in enumerate(zip(self.valid_set.samples, self.valid_set.targets)):
        #     if target in sub_classes:
        #         new_label = idx_to_new_idx[target]
        #         new_samples.append((sample[0], new_label))
        #         new_targets.append(new_label)
        #     else:
        #         new_label = 0
        #         new_samples.append((sample[0], new_label))
        #         new_targets.append(new_label)

        # half of sub classes, half of the others
        dic_class_cnt = Counter(self.valid_set.targets)
        new_samples, new_targets = [], []
        idx = 0
        while idx < len(self.valid_set.targets):
            target = self.valid_set.targets[idx]
            if target in sub_classes:
                n_samples_per_class = 50
                new_samples.extend(self.valid_set.samples[idx:idx + n_samples_per_class])
                new_targets.extend(self.valid_set.targets[idx:idx + n_samples_per_class])
            else:
                n_samples_per_class = 1
                new_samples.extend(self.valid_set.samples[idx:idx + n_samples_per_class])
                new_targets.extend(self.valid_set.targets[idx:idx + n_samples_per_class])
            idx += dic_class_cnt[target]


        self.sub_valid_set = copy.deepcopy(self.valid_set)
        self.sub_valid_set.samples = self.sub_valid_set.imgs = new_samples
        self.sub_valid_set.targets = new_targets

        self.sub_train_loader = torch.utils.data.DataLoader(self.sub_train_set, batch_size=self.batch_size, shuffle=True)
        self.sub_valid_loader = torch.utils.data.DataLoader(self.sub_valid_set, batch_size=self.batch_size, shuffle=False)

        self.sub_train_iterations = len(self.sub_train_loader)
        self.sub_valid_iterations = len(self.sub_valid_loader)