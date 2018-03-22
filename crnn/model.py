import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
from config import EMBEDDING_DIM, KERNEL_NUM, SENTENCE_DIM

class TextCRNN(nn.Module):

    def __init__(self,
                 classes = 10,
                 filter_sizes = None,
                 embedding_dim = None,
                 kernel_num = None,
                 filter_size = 3,
                 hidden_size = 100,
                 dropout_keep_prob = 0.5):
        super(TextCRNN, self).__init__()

        self.embedding_dim = embedding_dim or EMBEDDING_DIM
        self.dropout_keep_prob = dropout_keep_prob
        self.kernel_num = kernel_num or KERNEL_NUM
        self.filter_size = filter_size
        self.hidden_size = hidden_size

        self.conv2d = nn.Conv2d(1, self.embedding_dim, (self.filter_size, self.embedding_dim))
        self.lstm = nn.LSTM(self.embedding_dim, 
                            self.kernel_num // 2,
                            num_layers=1,
                            bidirectional=True)
        self.dropout = nn.Dropout(1 - self.dropout_keep_prob)
        self.fc1 = nn.Linear(self.kernel_num, classes)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(2, batch_size, self.kernel_num // 2)),
                autograd.Variable(torch.randn(2, batch_size, self.kernel_num // 2)))

    def forward(self, sentence):
        # Add a channel dimension
        embed = sentence.unsqueeze(1)

        # Convolution & ReLU (or SeLU/eLU)
        output = F.relu(self.conv2d(embed))

        # Remove extra dimension
        output = output.squeeze(3)
        # print(output.size())
        output = torch.transpose(output, 1, 2)
        # print(output.size())

        # Concatenate & Dropout
        output = self.dropout(output)

        # LSTM
        batch_size = output.size(1)
        self.hidden = self.init_hidden(batch_size)
        output, _ = self.lstm(output, self.hidden)
        output = F.tanh(output)
        output = self.dropout(output)
        # print(output.size())

        # Fully connected layer & softmax
        # print(output.size())
        output = output.permute(0, 2, 1)
        # print(output.size())
        output = F.max_pool1d(output, output.size(2)).squeeze(2)
        # print(output.size())
        output = self.fc1(output)
        # print(output.size())

        return output