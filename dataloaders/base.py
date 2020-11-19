

class BaseLoader:
    def __init__(self):
        self.train_transform = None
        self.valid_transform = None

        self.train_set = None
        self.valid_set = None

        self.train_loader = None
        self.valid_loader = None

        self.train_iterations = None
        self.valid_iterations = None

    def set_default(self):
        pass
