import torch
import torch.nn as nn
from common.modules import LockedDropout, WeightDrop
from torchqrnn import QRNNLayer

class RNNModel(nn.Module):

    def __init__(self):
        super(RNNModel, self).__init__()
        