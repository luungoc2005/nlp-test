import torch
import torch.nn.functional as F
import torch.nn as nn
from config import FILTER_SIZES, KERNEL_NUM, EMBEDDING_DIM

class TextCNN(nn.Module):

    def __init__(self,
                 classes = 10,
                 filter_sizes = None,
                 embedding_dim = None,
                 kernel_num = 0,
                 dropout_keep_prob = 0.5):
        super(TextCNN, self).__init__()

        self.filter_sizes = filter_sizes or FILTER_SIZES
        self.embedding_dim = embedding_dim or EMBEDDING_DIM
        self.kernel_num = kernel_num or KERNEL_NUM
        self.dropout_keep_prob = dropout_keep_prob

        self.convolutions = nn.ModuleList([
            nn.Conv2d(1, self.kernel_num, (k_size, self.embedding_dim), padding=(k_size - 1, 0))
            for k_size in self.filter_sizes
        ])
        self.dropout = nn.AlphaDropout(1 - self.dropout_keep_prob)
        self.fc1 = nn.Linear(len(self.filter_sizes) * self.kernel_num, classes)

    def forward(self, sentence):
        # Add a channel dimension
        embed = sentence.unsqueeze(1)

        # Convolution & ReLU (or SeLU/eLU)
        output = [F.selu(conv(embed)).squeeze(3) for conv in self.convolutions]

        # Max pooling
        output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in output]

        # Concatenate & Dropout
        output = torch.cat(output, 1)
        output = self.dropout(output)

        # Fully connected layer & softmax
        output = self.fc1(output)

        return output