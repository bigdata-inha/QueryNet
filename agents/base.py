import os
import torch


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
