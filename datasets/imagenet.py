"""
Imagenet Dataloader implementation
"""
import os
import logging
import copy
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader


class ImagenetDataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("ImagenetDataLoader")
        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        if config.data_mode == "image_folder":
            self.logger.info("Loading DATA.....")

            train_transform = transforms.Compose([
                # transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            valid_transform = transforms.Compose([
                # transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            train_set = datasets.ImageFolder(os.path.join(self.config.data_dir, "train"), transform=train_transform)
            valid_set = datasets.ImageFolder(os.path.join(self.config.data_dir, "val"), transform=valid_transform)
        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

        self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
                                       num_workers=self.config.data_loader_workers, pin_memory=self.config.pin_memory)
        self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False,
                                       num_workers=self.config.data_loader_workers, pin_memory=self.config.pin_memory)
        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)

    def finalize(self):
        pass


class SpecializedImagenetDataLoader:
    def __init__(self, config, *subset_labels):
        self.config = config
        self.logger = logging.getLogger("ImagenetDataLoader")
        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        if config.data_mode == "image_folder":
            self.logger.info("Loading DATA.....")

            train_transform = transforms.Compose([
                # transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            valid_transform = transforms.Compose([
                # transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            train_set = datasets.ImageFolder(os.path.join(self.config.data_dir, "train"), transform=train_transform)
            valid_set = datasets.ImageFolder(os.path.join(self.config.data_dir, "val"), transform=valid_transform)

            part_and_sample_train_set = datasets.ImageFolder(os.path.join(self.config.data_dir, "train"), transform=valid_transform)
            binary_nc_valid_set = copy.deepcopy(valid_set)

            mapping_table = dict()  # 0은 others labels (dustbin class), 그 후 subset label은 1부터 순서대로 mapping
            for new_label, label in enumerate(subset_labels):
                assert isinstance(label, int)
                mapping_table[label] = new_label + 1

            subset_idx, oth_idx = [], []
            num_sample = (1000 * len(subset_labels)) // (1000 - len(subset_labels))  # 나머지 클래스당 뽑아야할 샘플 수
            sample = 0
            last_labels = 0
            oth_full = False
            for i, e in enumerate(train_set.targets):
                if last_labels != e:    # 클래스가 바뀌었다면, 다시 클래스에 대한 샘플을 num_sample 수만큼 뽑기위해 초기화
                    oth_full = False
                    sample = 0
                if e in subset_labels:  # subset에 대한 모든 샘플은 뽑음
                    subset_idx.append(i)
                else:
                    if oth_full:        # 현재의 클래스에 대한 샘플을 num_sample만큼 뽑았으면, 더이상 뽑지 않음
                        pass
                    else:
                        oth_idx.append(i)   # others 클래스에 대한 샘플을 num_sample 수만큼 뽑음
                        sample += 1
                        if sample >= num_sample:    # 현재 클래스에 대한 샘플을 다 뽑았으면 다음 클래스가 나오기 전까지 skip
                            oth_full = True
                last_labels = e         # 클래스가 바뀌는 지를 탐색하기 위해 기록

            # change label to binary label
            train_set.targets = [mapping_table[i] if i in subset_labels else 0 for i in train_set.targets]
            train_set.samples = [(x, mapping_table[y]) if y in subset_labels else (x, 0) for x, y in train_set.samples]

            binary_train_set = torch.utils.data.dataset.Subset(train_set, subset_idx + oth_idx)
            self.binary_train_loader = DataLoader(binary_train_set, batch_size=self.config.batch_size, shuffle=True)

            # make part train set
            part_train_set = torch.utils.data.dataset.Subset(part_and_sample_train_set, subset_idx)
            self.part_train_loader = DataLoader(part_train_set, batch_size=self.config.batch_size, shuffle=False)

            # make valid set
            full_idx = []
            subset_idx = [i for i, e in enumerate(valid_set.targets) if e in subset_labels]
            oth_idx = []
            num_sample = (50 * len(subset_labels)) // (1000 - len(subset_labels))
            if num_sample < 1:
                num_sample = 1
            self.binary_valid_loader = []
            sample = 0
            last_labels = 0
            oth_full = False
            for i, e in enumerate(valid_set.targets):
                if last_labels != e:
                    oth_full = False
                    sample = 0
                    if len(oth_idx) == 50 * len(subset_labels):
                        full_idx = full_idx + subset_idx + oth_idx
                        oth_idx = []
                if e in subset_labels:
                    continue
                else:
                    if oth_full:
                        pass
                    else:
                        oth_idx.append(i)
                        sample += 1
                        if sample >= num_sample:
                            oth_full = True
                last_labels = e
            else:
                full_idx = full_idx + subset_idx[:len(oth_idx)] + oth_idx
            binary_valid_set = torch.utils.data.dataset.Subset(valid_set, full_idx)
            binary_nc_valid_set = torch.utils.data.dataset.Subset(binary_nc_valid_set, full_idx)
            self.binary_valid_loader = DataLoader(binary_valid_set, batch_size=self.config.batch_size, shuffle=True)
            self.binary_nc_valid_loader = DataLoader(binary_nc_valid_set, batch_size=self.config.batch_size, shuffle=True)

            # change label to binary label
            valid_set.targets = [mapping_table[i] if i in subset_labels else 0 for i in valid_set.targets]
            valid_set.samples = [(x, mapping_table[y]) if y in subset_labels else (x, 0) for x, y in valid_set.samples]


            self.binary_train_iterations = len(self.binary_train_loader)
            self.part_train_iterations = len(self.part_train_loader)
            self.binary_valid_iterations = len(self.binary_valid_loader)
        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def finalize(self):
        pass
