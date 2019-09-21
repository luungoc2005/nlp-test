import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union
from config import EMBEDDING_DIM, UNK_TAG
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from common.torch_utils import to_gpu
from common.utils import letterToIndex, n_letters, prepare_vec_sequence, word_to_vec, argmax

class CNNWordEncoder(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        kernel_sizes: List[int],
        hidden_dim: Optional[int] = None,
        letters_dim: Optional[int] = None,
        *args,
        **kwargs
    ) -> None:
        super(CNNWordEncoder, self).__init__()

        self.hidden_dim = hidden_dim or EMBEDDING_DIM
        self.letters_dim = letters_dim or n_letters
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.embedding = nn.Embedding(n_letters + 1, self.hidden_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.hidden_dim, 
                    self.out_channels, 
                    K, padding=K // 2
                )
                for K in self.kernel_sizes
            ]
        )
        self.output_dim = self.out_channels * len(self.kernel_sizes)

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        batch_size, max_sent_length, max_word_length = tuple(chars.size())
        chars = chars.view(batch_size * max_sent_length, max_word_length)

        # char_embedding: (bsize * max_sent_length, max_word_length, emb_size)
        char_embedding = self.char_embed(chars)

        # conv_inp dim: (bsize * max_sent_length, emb_size, max_word_length)
        conv_inp = char_embedding.transpose(1, 2)
        char_conv_outs = [F.relu(conv(conv_inp)) for conv in self.convs]

        # Apply max pooling
        # char_pool_out[i] dims: (bsize * max_sent_length, out_channels)
        char_pool_outs = [torch.max(out, dim=2)[0] for out in char_conv_outs]

        # Concat different feature maps together
        # char_pool_out dim: (bsize * max_sent_length, out_channel * num_kernels)
        char_pool_out = torch.cat(char_pool_outs, 1)

        # Reshape to (bsize, max_sent_length, out_channel * len(self.convs))
        return char_pool_out.view(batch_size, max_sent_length, -1)
