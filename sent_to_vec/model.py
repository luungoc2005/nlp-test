import torch
import torch.autograd as autograd
import torch.nn as nn
from config import START_TAG, STOP_TAG, EMBEDDING_DIM, HIDDEN_DIM

class NLINet(nn.Module):
    def __init__(self, config):
        super(NLINet, self).__init__()
        