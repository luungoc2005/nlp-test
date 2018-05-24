import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import heapq
from config import MAX_SEQUENCE_LENGTH
from paraphrase_vae.tokenizer import SOS_token, EOS_token
from common.utils import argmax

NLL = nn.NLLLoss(size_average=False)


def KLDivLoss(logp, target, mean, logv, step, k=0.0025, x0=75000):
    # x0 ~ total number of sentences / 2 (midpoint of epoch)
    NLL_loss = 0.
    for idx in range(len(target)):
        NLL_loss += NLL(logp[idx].unsqueeze(0), target[idx])

    KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    KL_weight = float(1 / (1 + np.exp(-k * (step - x0))))

    return NLL_loss, KL_loss, KL_weight


# https://geekyisawesome.blogspot.com/2016/10/using-beam-search-to-generate-most.html


class Beam(object):

    def __init__(self, beam_width):
        self.heap = list()
        self.beam_width = beam_width

    def add(self, prob, complete, prefix, hidden_state):
        heapq.heappush(self.heap, ((prob, complete), (prefix, hidden_state)))
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):
        return iter(self.heap)


class ParaphraseVAE(nn.Module):

    def __init__(self,
                 vocab_size,
                 max_length=None,
                 dropout_keep_prob=0.5,
                 embedding_size=300,
                 hidden_size=600,
                 latent_size=100,
                 rnn_type='GRU'):
        super(ParaphraseVAE, self).__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length or MAX_SEQUENCE_LENGTH
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.dropout_keep_prob = dropout_keep_prob
        self.embedding_size = embedding_size
        self.rnn_type = rnn_type

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        self.encoder = EncoderRNN(self.embedding,
                                  num_layers=1,
                                  hidden_size=self.hidden_size,
                                  rnn_type=self.rnn_type)

        self.hidden2mean = nn.Linear(self.hidden_size, latent_size)
        self.hidden2logv = nn.Linear(self.hidden_size, latent_size)

        self.latent2hidden = nn.Linear(self.latent_size, self.hidden_size)
        self.decoder = AttnDecoderRNN(self.embedding,
                                      self.vocab_size,
                                      num_layers=1,
                                      hidden_size=self.hidden_size,
                                      dropout_keep_prob=self.dropout_keep_prob,
                                      rnn_type=self.rnn_type)

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

        decoder_input = torch.LongTensor([EOS_token])
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

    def generate(self, input, beam_width=0, k=5000):
        with torch.no_grad():
            if beam_width == 0:  # Eager
                decoded, _, _ = self.forward(input)
                result = []
                for idx in range(len(decoded)):
                    token = argmax(decoded[idx])
                    result.append(token)
                    if token == SOS_token:
                        break

                return result[::-1]
            else:  # Beam search
                encoder_hidden, mean, logv = self._encode_to_latent(input)

                std = torch.exp(0.5 * logv)
                z = torch.randn((self.max_length, self.latent_size))
                z = z * std + mean

                prev_beam = Beam(beam_width)
                prev_beam.add(0., False, [EOS_token], encoder_hidden)

                hidden = self.latent2hidden(z)

                while True:
                    curr_beam = Beam(beam_width)

                    for (prefix_prob, complete), (prefix, hidden_state) in prev_beam:
                        if complete:
                            curr_beam.add(prefix_prob, True, prefix, hidden_state)
                        else:
                            decoder_input = torch.LongTensor([prefix[-1]])
                            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                                decoder_input, hidden_state, hidden)

                            if k == -1:
                                k = self.vocab_size

                            topv, topi = decoder_output.topk(k)
                            topv, topi = topv.squeeze(), topi.squeeze()

                            for idx, next_prob in enumerate(topv):
                                next_prob = next_prob.detach().item()
                                token = topi[idx].detach().item()
                                complete = (token == SOS_token)

                                # Adding probs because this is actually log probs (from log softmax)
                                curr_beam.add(prefix_prob + next_prob, complete, prefix + [token], decoder_hidden)
                    (best_prob, best_complete), (best_prefix, _) = max(curr_beam)
                    if best_complete or len(best_prefix) - 1 == -1:
                        print('Found best candidate with probability: %s' % best_prob)
                        return best_prefix[1:][::-1]
                    prev_beam = curr_beam


class EncoderRNN(nn.Module):

    def __init__(self,
                 embedding_layer,
                 num_layers=1,
                 embedding_size=300,
                 hidden_size=600,
                 rnn_type='GRU'):
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.embedding_size = embedding_size

        self.embedding = embedding_layer

        rnn_func = nn.GRU if self.rnn_type == 'GRU' else nn.LSTM
        self.rnn = rnn_func(self.embedding_size,
                            self.hidden_size // 2,
                            self.num_layers,
                            bidirectional=True)

    def forward(self, input, hidden):
        emb = self.embedding(input).view(1, 1, -1)
        output, hidden = self.rnn(emb, hidden)
        return output, hidden

    def init_hidden(self):
        if self.rnn_type == 'GRU':
            return torch.zeros(2, 1, self.hidden_size // 2)
        else:
            return torch.zeros(2, 1, self.hidden_size // 2), torch.zeros(2, 1, self.hidden_size // 2)


class AttnDecoderRNN(nn.Module):
    
    def __init__(self,
                 embedding_layer,
                 vocab_size,
                 max_length=None,
                 num_layers=2,
                 embedding_size=300,
                 hidden_size=600,
                 latent_size=1100,
                 dropout_keep_prob=0.5,
                 rnn_type='GRU'):
        super(AttnDecoderRNN, self).__init__()
        self.max_length = max_length or MAX_SEQUENCE_LENGTH
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.dropout_keep_prob = dropout_keep_prob
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.embedding_size = embedding_size

        self.embedding = embedding_layer
        self.attn = nn.Linear(self.embedding_size + self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.embedding_size + self.hidden_size, self.embedding_size)
        self.dropout = nn.Dropout(1 - self.dropout_keep_prob)

        rnn_func = nn.GRU if self.rnn_type == 'GRU' else nn.LSTM
        self.gru = rnn_func(self.embedding_size,
                            self.hidden_size // 2,
                            self.num_layers,
                            bidirectional=True)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, input, hidden, latent_out):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        if self.rnn_type == 'GRU':
            attn_weights = F.softmax(
                self.attn(torch.cat((embedded[0], hidden.view(1, 1, -1)[0]), 1)), dim=1)
        else:  # if rnn_type is LSTM, hidden will be a tuple of (h0, c0)
            attn_weights = F.softmax(
                self.attn(torch.cat((embedded[0], hidden[0].view(1, 1, -1)[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0), latent_out.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        if self.rnn_type == 'GRU':
            return torch.zeros(2, 1, self.hidden_size // 2)
        else:
            return torch.zeros(2, 1, self.hidden_size // 2), torch.zeros(2, 1, self.hidden_size // 2)
