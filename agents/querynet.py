import time
import random
import pickle
import torch.backends as backends
import torch.nn as nn
import torch.optim as optim

from agents.base import BaseAgent
from dataloaders import *
from graphs.factory import ModelFactory

SEED = 42


class Vgg16QueryNet(BaseAgent):
    def __init__(self):
        super(Vgg16QueryNet, self).__init__()
        backends.cudnn.deterministic = True
        backends.cudnn.benchmark = False
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        self.model = ModelFactory.get_model("CIFAR10", "VGG16")
        self.loader = Cifar10Loader(batch_size=128)
        self.optimaizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimaizer, milestones=[150, 225], gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

        self.named_paramed_modules = dict()
        for name, module in self.model.features.named_modules(prefix=''):
            if isinstance(module, torch.nn.Conv2d):
                self.named_paramed_modules[name] = module
            elif isinstance(module, torch.nn.BatchNorm2d):
                self.named_paramed_modules[name] = module

        self.channel_importance = {v: dict() for k, v in self.loader.train_set.class_to_idx.items()}

    def save_channel_info(self, path="checkpoints/channel_info_vgg16_cifar10.pkl"):
        if os.path.exists(path):
            return
        with open(path, 'wb') as f:
            pickle.dump(self.channel_importance, f)

    def load_channel_info(self, path="checkpoints/channel_info_vgg16_cifar10.pkl"):
        if os.path.exists(path):
            with open(path, 'wb') as f:
                self.channel_importance = pickle.load(f)
        else:
            print("file is not exist")

    def preprocess(self):
        outputs, grads = dict(), dict()

        def save_grad(module_name):
            def hook(grad):
                grads[module_name] = grad
            return hook

        for query in len(self.channel_importance):
            self.loader.set_subclass_loader()







    # def preprocess(self):
    #     """ preprocess for all classes 0 ~ 999 in imagenet """
    #     def save_grad(idx):
    #         def hook(grad):
    #             grads[idx] = grad
    #         return hook
    #
    #     def cal_importance(query_k, grads_list, outputs_list):
    #         for n, m in self.named_modules_list.items():
    #             if isinstance(m, torch.nn.Conv2d):
    #                 grad = grads_list[n]
    #                 output = outputs_list[n]
    #                 importance = (grad * output).mean(dim=(2, 3))
    #                 total_importance = torch.abs(importance).squeeze()
    #                 self.channel_importance[query_k][n] += total_importance.data.cpu()
    #
    #     self.model = self.model.to(self.device)
    #     for query_k, _ in self.channel_importance.items():
    #         for name, module in self.named_conv_list.items():
    #             self.channel_importance[query_k][name] = torch.zeros(module.out_channels)
    #
    #         self.loader.make_one_class_set(query_k, batch_size=1)
    #         for inputs, labels in self.loader.sub_train_loader:
    #             self.optimaizer.zero_grad()
    #             num_batch = inputs.size(0)
    #             outputs, grads = {}, {}
    #
    #             inputs = inputs.to(self.device)
    #             inputs.requires_grad = True
    #
    #             x = inputs
    #             i = 0
    #             for m in self.model.features:
    #                 x = m(x)
    #                 if isinstance(m, torch.nn.Conv2d):
    #                     outputs[f'{i}.conv'] = x
    #                     outputs[f'{i}.conv'].register_hook(save_grad(f'{i}.conv'))
    #                     i += 1
    #             else:
    #                 x = x.view(num_batch, -1)
    #
    #             x = self.model.classifier(x)
    #
    #             one_hot = np.zeros((x.shape[0], x.shape[-1]), dtype=np.float32)
    #             one_hot[:, query_k] = 1
    #             one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    #             one_hot = torch.sum(one_hot.to(self.device) * x)
    #             one_hot.backward(retain_graph=True)
    #
    #             cal_importance(query_k, grads, outputs)
    #     self.save_channel_info()

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

    def train(self, epochs=100, checkpoint_name="checkpoint.pt"):
        assert self.loader is not None, "please set Vgg16QueryNet.loader"
        self.model = self.model.to(self.device)
        max_acc = 0.
        for epoch in range(epochs):
            self.last_epoch += 1
            train_loss = self._train_one_epoch(epochs)
            val_loss, acc = self._validate()
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(acc)
            if acc > max_acc:
                max_acc = acc
                self.save_checkpoint(checkpoint_name=checkpoint_name)
            if self.scheduler is not None:
                self.scheduler.step()

    def _train_one_epoch(self, epochs):
        self.model.train()
        start = time.time()
        n, correct, train_loss = 0, 0, 0.
        for i, (data, label) in enumerate(self.loader.train_loader):
            data, label = data.to(self.device), label.to(self.device)
            self.optimaizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimaizer.step()
            print(f"\rEpoch {self.last_epoch}/{epochs}\t({i+1}/{self.loader.train_iterations})\tElapsed time:{time.time() - start:.3f}\tTraining loss: {loss.item():.4f}", end='')
            train_loss += loss.item()
        train_loss = train_loss / self.loader.train_iterations
        return train_loss

    def _validate(self):
        self.model.eval()
        self.model = self.model.to(self.device)
        n, correct, val_loss = 0, 0, 0.
        start = time.time()
        with torch.no_grad():
            for i, (data, label) in enumerate(self.loader.valid_loader):
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                n += data.size(0)
                correct += pred.eq(label).sum().item()
                val_loss += self.criterion(output, label)

        val_loss /= self.loader.valid_iterations
        acc = correct / n
        print(f"\tInference time: {time.time() - start}\tValidation loss: {val_loss:.4f} Acc: {acc:.4f} ({correct}/{n})")
        return val_loss, acc

    def sub_validate(self):
        self.model.eval()
        self.model = self.model.to(self.device)
        n, correct, val_loss = 0., 0., 0.
        selected_classes = [340, 386, 105, 30, 300, 985, 527, 673, 779, 526]

        def scoring(pred, label):
            correct = 0.
            for t, y in zip(pred, label):
                if t in selected_classes:
                    if t == y:
                        correct += 1
                else:
                    if t == y:
                        correct += 1
                    elif t not in selected_classes and y not in selected_classes:
                        correct += 1

            return correct

        self.loader.make_subclass_set(selected_classes)
        with torch.no_grad():
            for i, (data, label) in enumerate(self.loader.sub_valid_loader):
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                n += data.size(0)
                # correct += pred.eq(label).sum().item()
                correct += scoring(pred, label)
                val_loss += self.criterion(output, label)

        val_loss /= self.loader.valid_iterations
        print(f"\t\tValidation loss: {val_loss:.4f} Acc: {correct/n:.4f} ({correct}/{n})")


if __name__=="__main__":
    agent = Vgg16QueryNet()
    agent.sub_validate()
