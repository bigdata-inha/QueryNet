import copy
from collections import Counter
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

        self.set_default_loader(batch_size)
        self._original_train_set = copy.deepcopy(self.train_set)
        self._original_valid_set = copy.deepcopy(self.valid_set)
        self.batch_size = batch_size

    def set_default_loader(self, batch_size=None):
        self.batch_size = self.batch_size if batch_size is None else batch_size

        self.train_set = datasets.CIFAR10("./data", train=True, transform=self.train_transform, download=True)
        self.valid_set = datasets.CIFAR10("./data", train=False, transform=self.valid_transform)

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False)

        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)

    def set_subclass_loader(self, sub_classes, batch_size=None, use_dustbin_class=False, relabeling=False):
        """
        :param sub_classes: iterable, wanted classes to be generated by loader
        :param batch_size: int, batch size of loader
        :param use_dustbin_class: bool, whether using dustbin class or not
        :param relabeling: bool, whether changing label to new label,
        if True, label of subclasses are re-labeled from 0,
        if using dustbin, the dustbin class is labeled to the last number (e.g. len(subclasses))
        :return: None
        """
        assert isinstance(sub_classes, Iterable), "sub_classes must to be iterable"
        self.batch_size = self.batch_size if batch_size is None else batch_size
        num_subclasses, num_dustbin = len(sub_classes), (len(self.train_set.classes) - len(sub_classes))
        sub_classes = sorted(sub_classes)   # 입력된 class list를 순서대로 정리 e.g. [5,2,1] -> [1,2,5]

        # selecting train set samples
        dic_train_class_cnt = Counter(self.train_set.targets)  # class당 데이터 갯수
        total_samples_sub_classes, total_sample_dustbin = 0, 0    # sub class 총 데이터 갯수 & dustbin class 총 데이터 갯수
        for i in range(len(self.train_set.classes)):
            if i in sub_classes:
                total_samples_sub_classes += dic_train_class_cnt[i]
            else:
                total_sample_dustbin += dic_train_class_cnt[i]
        train_new_samples, train_new_targets = [], []
        idx = 0
        while idx < len(self.train_set.targets):
            target = self.train_set.targets[idx]
            if target in sub_classes:   # sub class의 데이터는 전부 뽑음
                n_samples_per_class = dic_train_class_cnt[target]
                train_new_samples.extend(self.train_set.data[idx:idx + n_samples_per_class])
                train_new_targets.extend(self.train_set.targets[idx:idx + n_samples_per_class])
            else:   # dustbin class당 데이터 갯수: 전체 subclass 데이터의 수를 dustbin 클래스의 수로 나눠줌
                n_samples_per_class = (total_samples_sub_classes // total_sample_dustbin)
                if n_samples_per_class == 0:
                    n_samples_per_class = 1
                train_new_samples.extend(self.train_set.data[idx:idx + n_samples_per_class])
                train_new_targets.extend(self.train_set.targets[idx:idx + n_samples_per_class])
            idx += dic_train_class_cnt[target]

        # selecting valid set samples
        dic_valid_class_cnt = Counter(self.valid_set.targets)
        total_samples_sub_classes, total_sample_dustbin = 0, 0    # sub class 총 데이터 갯수 & dustbin class 총 데이터 갯수
        for i in range(len(self.train_set.classes)):
            if i in sub_classes:
                total_samples_sub_classes += dic_valid_class_cnt[i]
            else:
                total_sample_dustbin += dic_valid_class_cnt[i]
        valid_new_samples, valid_new_targets = [], []
        idx = 0
        while idx < len(self.valid_set.targets):
            target = self.valid_set.targets[idx]
            if target in sub_classes:
                n_samples_per_class = dic_valid_class_cnt[target]
                valid_new_samples.extend(self.valid_set.data[idx:idx + n_samples_per_class])
                valid_new_targets.extend(self.valid_set.targets[idx:idx + n_samples_per_class])
            else:
                n_samples_per_class = (total_samples_sub_classes // total_sample_dustbin)
                if n_samples_per_class == 0:
                    n_samples_per_class = 1
                valid_new_samples.extend(self.valid_set.data[idx:idx + n_samples_per_class])
                valid_new_targets.extend(self.valid_set.targets[idx:idx + n_samples_per_class])
            idx += dic_valid_class_cnt[target]

        idx_to_new_idx = {c: i for i, c in enumerate(sub_classes)}  # mapping to new target idx / dustbin: the last num

        # re-labeling
        if relabeling is True:
            # re-labeling train set
            for i, (sample, target) in enumerate(zip(train_new_samples, train_new_targets)):
                if target in sub_classes:
                    new_label = idx_to_new_idx[target]
                    train_new_samples[i] = (sample[0], new_label)
                    train_new_targets[i] = new_label
                else:
                    new_label = len(sub_classes)    # last label
                    train_new_samples[i] = (sample[0], new_label)
                    train_new_targets[i] = new_label

            # re-labeling valid set
            for i, (sample, target) in enumerate(zip(valid_new_samples, valid_new_targets)):
                if target in sub_classes:
                    new_label = idx_to_new_idx[target]
                    valid_new_samples[i] = (sample[0], new_label)
                    valid_new_targets[i] = new_label
                else:
                    new_label = len(sub_classes)    # last label
                    valid_new_samples[i] = (sample[0], new_label)
                    valid_new_targets[i] = new_label

        self.train_set.data = train_new_samples
        self.train_set.targets = train_new_targets

        self.valid_set.data = valid_new_samples
        self.valid_set.targets = valid_new_targets

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False)

