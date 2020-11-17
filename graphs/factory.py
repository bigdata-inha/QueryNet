import torch
import torchvision.models as models

from .models import *


class ModelFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(dataset, model_name):

        if dataset == 'CIFAR10' or dataset == 'Cifar10' or dataset == 'cifar10':
            if 'VGG' in model_name:
                return VGG(model_name, num_classes=10, use_large_top=False)

        if dataset == 'CIFAR100' or dataset == 'Cifar100' or dataset == 'cifar100':
            if 'VGG' in model_name:
                return VGG(model_name, num_classes=100, use_large_top=False)

        if dataset == 'Imagenet' or dataset == 'imagenet':
            if 'VGG' in model_name:
                return VGG(model_name, num_classes=1000, use_large_top=True)

        print("Model isn't generated. Please set model name and dataset name correctly")

