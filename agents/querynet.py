import time
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.utils as vutils

from agents.base import BaseAgent
from dataloaders import *
from prune.channel import *

SEED = 42
random.seed(SEED)


class Vgg16QueryNet(BaseAgent):
    def __init__(self):
        super(Vgg16QueryNet, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(SEED)
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(SEED)

        self.epochs = 300
        self.ft_epochs = 10

        self.model = models.vgg16_bn(pretrained=True)
        self.optimaizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
        self.criterion = nn.CrossEntropyLoss()

        self.loader = ImageNetLoader(batch_size=16)

        self.history = {"train_acc": [], "train_loss": [], "valid_acc": [], "valid_loss": []}

        self.named_modules_list, self.named_conv_list = dict(), dict()
        i = 0
        for idx, m in enumerate(self.model.features):
            if isinstance(m, torch.nn.Conv2d):
                self.named_modules_list['{}.conv'.format(i)] = m
                self.named_conv_list['{}.conv'.format(i)] = m
            elif isinstance(m, torch.nn.BatchNorm2d):
                self.named_modules_list['{}.bn'.format(i)] = m
                i += 1
        self.channel_importance = {v: dict() for k, v in self.loader.train_set.class_to_idx.items()}

    def save_channel_info(self, path="checkpoints/channel_info_vgg16.pkl"):
        if os.path.exists(path):
            return
        with open(path, 'wb') as f:
            pickle.dump(self.channel_importance, f)

    def load_channel_info(self, path="checkpoints/channel_info_vgg16.pkl"):
        if os.path.exists(path):
            with open(path, 'wb') as f:
                self.channel_importance = pickle.load(f)
        else:
            print("file is not exist")

    def preprocess(self):
        """ preprocess for all classes 0 ~ 999 in imagenet """
        def save_grad(idx):
            def hook(grad):
                grads[idx] = grad
            return hook

        def cal_importance(grads_list, outputs_list):
            for n, m in self.named_modules_list.items():
                if isinstance(m, torch.nn.Conv2d):
                    grad = grads_list[n]
                    output = outputs_list[n]
                    importance = (grad * output).mean(dim=(2, 3))
                    total_importance = torch.abs(importance).sum(dim=0)
                    self.channel_importance[n] += total_importance.data.cpu()

        for query_k, _ in self.channel_importance.items():
            for name, module in self.named_conv_list.items():
                self.channel_importance[query_k][name] = torch.zeros(module.out_channels)

            self.loader.make_one_class_set(query_k, batch_size=1)
            for inputs, labels in self.loader.sub_train_loader:
                self.optimaizer.zero_grad()
                num_batch = inputs.size(0)
                outputs, grads = {}, {}

                inputs = inputs.to(self.device)
                inputs.requires_grad = True

                x = inputs
                i = 0
                for m in self.model.features:
                    x = m(x)
                    if isinstance(m, torch.nn.Conv2d):
                        outputs[f'{i}.conv'] = x
                        outputs[f'{i}.conv'].register_hook(save_grad(f'{i}.conv'))
                        i += 1
                else:
                    x = x.view(num_batch, -1)

                x = self.model.classifier(x)

                one_hot = np.zeros((x.shape[0], x.shape[-1]), dtype=np.float32)
                one_hot[:, query_k] = 1
                one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                one_hot = torch.sum(one_hot.to(self.device) * x)
                one_hot.backward(retain_graph=True)

                cal_importance(grads, outputs)

    def query_process(self, k, method='gradient'):
        if method == "gradient":
            for i, m in enumerate(self.named_conv_list.values()):
                if isinstance(m, torch.nn.Conv2d):
                    bn = self.named_modules_list[str(i) + '.bn']
                    if str(i + 1) + '.conv' in self.named_conv_list:
                        next_m = self.named_modules_list[str(i + 1) + '.conv']
                    else:
                        next_m = self.model.classifier[0]
                    channel_importance = self.channel_importance[str(i) + '.conv']
                    channel_importance = channel_importance / channel_importance.sum()
                    threshold = k / channel_importance.size(0)
                    indices_stayed = [i for i in range(len(channel_importance)) if channel_importance[i] > threshold]
                    module_surgery(m, bn, next_m, indices_stayed)

    def train(self, epochs=None):
        self.model = self.model.to(self.device)
        epochs = self.epochs if epochs is None else epochs
        for epoch in range(epochs):
            self._train_one_epoch(epoch)
            self._validate(epoch)
            self.scheduler.step()

    def _train_one_epoch(self, epoch):
        self.model.train()
        start = time.time()
        for i, (data, label) in enumerate(self.loader.train_loader):
            data, label = data.to(self.device), label.to(self.device)
            self.optimaizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimaizer.step()
            print(f"\rEpoch {epoch}/{self.epochs}\t{time.time() - start:.3f}\tTraining loss: {loss.item():.4f}", end='')

    def _validate(self, epoch):
        self.model.eval()
        n, correct, val_loss = 0, 0, 0.
        with torch.no_grad():
            for i, (data, label) in enumerate(self.loader.valid_loader):
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                n += data.size(0)
                correct += pred.eq(label).sum().item()
                val_loss += self.criterion(output, label)

        val_loss /= self.loader.valid_iterations
        print(f"\t\tValidation loss: {val_loss:.4f} Acc: {correct/n:.4f} ({correct}/{n})")
