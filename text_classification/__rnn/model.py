
import torch.nn as nn
from torch.autograd import Variable

class TextRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(TextRNN, self).__init__()

