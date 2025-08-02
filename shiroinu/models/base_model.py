import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    data_based_hyperparams = ['mean_', 'std_']

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def create(cls, state_path=None, **kwargs):
        model = cls(**kwargs)
        if (state_path is not None) and (state_path != ''):
            model.load_state_dict(torch.load(state_path))
            model.eval()
        model.to(model.device)
        return model

    @abstractmethod
    def extract_input(self, batch):
        pass

    @abstractmethod
    def extract_target(self, batch):
        pass

    @abstractmethod
    def predict(self, batch, dataset):
        pass

    def get_loss(self, batch, criterion):
        input = self.extract_input(batch)
        target = self.extract_target(batch)
        output = self(input)
        if isinstance(output, tuple):
            loss = criterion(output[0], target)
        else:
            loss = criterion(output, target)
        return loss, input, target, output

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
