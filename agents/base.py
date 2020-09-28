import os
import torch


class BaseAgent:
    def __init__(self):
        self.loader = None

        self.model = None
        self.optimaizer = None
        self.criterion = None

    def save_checkpoint(self, directory="./checkpoint", checkpoint_name="checkpoint.pt"):
        if not os.path.exists(directory):
            os.makedirs(os.path.dirname(directory))
        if self.model is None:
            print("model is not exist. Define the model graph")
            raise ValueError

        torch.save(self.model.state_dict(), os.path.join(directory, checkpoint_name))

    def load_checkpoint(self, directory="./checkpoint", checkpoint_name="checkpoint.pt"):
        if self.model is None:
            print("model is not exist. Define the model graph")
            raise ValueError

        self.model.load_state_dict(torch.load(os.path.join(directory, checkpoint_name)))
