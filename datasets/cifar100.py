"""
Cifar100 Dataloader implementation
"""
import os
import logging
import copy
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader


class Cifar100DataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Cifar100DataLoader")
        self.mean, self.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]

        if config.data_mode == "download":
            self.logger.info("Loading DATA.....")

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            valid_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            train_set = datasets.CIFAR100("./data", train=True, download=True, transform=train_transform)
            valid_set = datasets.CIFAR100("./data", train=False, transform=valid_transform)
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


class SpecializedCifar100DataLoader:
    def __init__(self, config, *subset_labels):
        self.config = config
        self.logger = logging.getLogger("Cifar100DataLoader")
        self.mean, self.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]

        if config.data_mode == "download":
            self.logger.info("Loading DATA.....")

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            valid_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

            train_set = datasets.CIFAR100("./data", train=True, download=True, transform=train_transform)
            valid_set = datasets.CIFAR100("./data", train=False, transform=valid_transform)

            part_train_set = copy.deepcopy(train_set)

            mapping_table = dict()  # 0은 others labels (dustbin class), 그 후 subset label은 1부터 순서대로 mapping
            for new_label, label in enumerate(subset_labels):
                assert isinstance(label, int)
                mapping_table[label] = new_label + 1

            # make binary trainset
            subset_idx, oth_idx = [], []
            num_sample = (1000 * len(subset_labels)) // (1000 - len(subset_labels))  # 나머지 클래스당 뽑아야할 샘플 수
            sample = 0
            last_labels = 0
            oth_full = False
            for i, e in enumerate(train_set.targets):
                if last_labels != e:  # 클래스가 바뀌었다면, 다시 클래스에 대한 샘플을 num_sample 수만큼 뽑기위해 초기화
                    oth_full = False
                    sample = 0
                if e in subset_labels:  # subset에 대한 모든 샘플은 뽑음
                    subset_idx.append(i)
                else:
                    if oth_full:  # 현재의 클래스에 대한 샘플을 num_sample만큼 뽑았으면, 더이상 뽑지 않음
                        pass
                    else:
                        oth_idx.append(i)  # others 클래스에 대한 샘플을 num_sample 수만큼 뽑음
                        sample += 1
                        if sample >= num_sample:  # 현재 클래스에 대한 샘플을 다 뽑았으면 다음 클래스가 나오기 전까지 skip
                            oth_full = True
                last_labels = e  # 클래스가 바뀌는 지를 탐색하기 위해 기록

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def finalize(self):
        pass
