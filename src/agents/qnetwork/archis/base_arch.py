import torch.nn as nn

class BaseArch(nn.Module):
    def __init__(self):
        super(BaseArch, self).__init__()

    def forward(self, encoded_actions):
        raise NotImplementedError("Subclasses should implement method: forward")
    
    def encode(self, obs, actions, device):
        raise NotImplementedError("Subclasses should implement method: encode")
    
    @staticmethod
    def cat(encoded_actions_list):
        raise NotImplementedError("Subclasses should implement method: cat")
    