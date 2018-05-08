import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from config import EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, MAX_NUM_WORDS
from paraphrase_vae.tokenizer import SOS_token, EOS_token

NLL = nn.NLLLoss(size_average=False)


def KLDivLoss(logp, target, mean, logv, step, k=0.0025, x0=2500):
    NLL_loss = 0.
    for idx in range(len(target)):
        NLL_loss += NLL(logp[idx].unsqueeze(0), target[idx])

    KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    KL_weight = float(1 / (1 + np.exp(-k * (step - x0))))

    return NLL_loss, KL_loss, KL_weight


class ParaphraseVAE(nn.Module):

    def __init__(self,
                 vocab_size,
                 max_length=None,
                 dropout_keep_prob=0.5,
                 hidden_size=600,
                 latent_size=100):
        super(ParaphraseVAE, self).__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length or MAX_SEQUENCE_LENGTH
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.dropout_keep_prob = dropout_keep_prob

        self.encoder = EncoderRNN(self.vocab_size, num_layers=1, hidden_size=self.hidden_size)

        self.hidden2mean = nn.Linear(self.hidden_size, latent_size)
        self.hidden2logv = nn.Linear(self.hidden_size, latent_size)

        self.latent2hidden = nn.Linear(self.latent_size, self.hidden_size)
        self.decoder = AttnDecoderRNN(self.vocab_size,
                                      num_layers=1,
                                      hidden_size=self.hidden_size,
                                      dropout_keep_prob=self.dropout_keep_prob)

    def _encode_to_latent(self, input):
        encoder_hidden = self.encoder.init_hidden()

        input_length = input.size(0)

        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        mean = self.hidden2mean(encoder_outputs)
        logv = self.hidden2logv(encoder_outputs)

        return encoder_hidden, mean, logv

    def _latent_to_output(self, z, encoder_hidden, target=None, teacher_forcing_ratio=0.5):
        if target is None:
            target_length = self.max_length
        else:
            target_length = target.size(0)
        
        hidden = self.latent2hidden(z)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_input = torch.tensor([[EOS_token]])
        decoder_hidden = encoder_hidden

        decoded = torch.zeros(target_length, self.vocab_size)
        if target is not None and use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, hidden)
                decoder_input = target[di]  # Teacher forcing
                decoded[di] = decoder_output

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, hidden)
                decoded[di] = decoder_output

                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                if decoder_input.item() == SOS_token:
                    break

        return decoded

    def forward(self, input, target = None, teacher_forcing_ratio=0.5):
        encoder_hidden, mean, logv = self._encode_to_latent(input)

        std = torch.exp(0.5 * logv)
        z = torch.randn((self.max_length, self.latent_size))
        z = z * std + mean
        
        decoded = self._latent_to_output(z, encoder_hidden, target, teacher_forcing_ratio)
        
        return decoded, mean, logv


class EncoderRNN(nn.Module):

    def __init__(self,
                 vocab_size,
                 num_layers=1,
                 hidden_size=4096):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.rnn = nn.GRU(self.hidden_size,
                          self.hidden_size // 2, 
                          self.num_layers,
                          bidirectional=True)

    def forward(self, input, hidden):
        emb = self.embedding(input).view(1, 1, -1)
        output, hidden = self.rnn(emb, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(2, 1, self.hidden_size // 2)


class AttnDecoderRNN(nn.Module):
    
    def __init__(self,
                 vocab_size,
                 max_length=None,
                 num_layers=2,
                 hidden_size=4096,
                 latent_size=1100,
                 dropout_keep_prob=0.5):
        super(AttnDecoderRNN, self).__init__()
        self.max_length = max_length or MAX_SEQUENCE_LENGTH
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.dropout_keep_prob = dropout_keep_prob
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(1 - self.dropout_keep_prob)
        self.gru = nn.GRU(self.hidden_size,
                          self.hidden_size // 2,
                          self.num_layers,
                          bidirectional=True)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, input, hidden, latent_out):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden.view(1, 1, -1)[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), latent_out.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(2, 1, self.hidden_size)