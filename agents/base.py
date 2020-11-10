import os
import torch
import torch.nn as nn

class BaseAgent:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.last_epoch = -1

        self.loader = None

        self.model = None
        self.optimaizer = None
        self.scheduler = None
        self.criterion = None

        self.history = {"train_loss": [], "valid_loss": [], "valid_acc": []}

    def init_model_weights(self):
        assert self.model is not None, "please set self.model"
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def save_checkpoint(self, directory="./checkpoints", checkpoint_name="checkpoint.pt"):
        if not os.path.exists(directory):
            os.makedirs(os.path.dirname(directory))
        if self.model is None:
            print("model is not exist. Define the model graph")
            raise ValueError

        torch.save(self.model.state_dict(), os.path.join(directory, checkpoint_name))

    def load_checkpoint(self, load_path="./checkpoints/checkpoint.pt"):
        if self.model is None:
            print("model is not exist. Define the model graph")
            raise ValueError

        self.model.load_state_dict(torch.load(load_path))
