

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

        self.sub_train_set = None
        self.sub_valid_set = None

        self.sub_train_loader = None
        self.sub_valid_loader = None

        self.sub_train_iterations = None
        self.sub_valid_iterations = None
