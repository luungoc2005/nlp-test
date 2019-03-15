import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from nltk.tokenize import WhitespaceTokenizer
from common.wrappers import IModel
from common.preprocessing.keras import Tokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, UNK_TAG
from featurizers.basic_featurizer import BasicFeaturizer
from common.langs.vi_VN.utils import remove_tone_marks
from common.modules import BRNNWordEncoder
from common.torch_utils import to_gpu

class VNTokenizer(nn.Module):

    def __init__(self, config):
        super(VNTokenizer, self).__init__()
        self.max_emb_words = config.get('max_emb_words')
        self.embedding_dim = config.get('embedding_dim', EMBEDDING_DIM)
        self.char_embedding_dim = config.get('char_embedding_dim', CHAR_EMBEDDING_DIM)
        self.hidden_dim = config.get('hidden_dim', 1200)
        self.num_layers = config.get('num_layers', 3)
        self.dropout_prob = config.get('dropout_prob', .2)
        self.is_cuda = is_cuda if is_cuda is not None else torch.cuda.is_available()

        self.word_encoder = to_gpu(BRNNWordEncoder(self.char_embedding_dim, rnn_type='LSTM'))
        self.dropout = nn.Dropout(self.dropout_prob))

        # 0: reserved index by Keras tokenizer
        # num_words + 1: index for oov token
        self.embedding = nn.Embedding(self.max_emb_words + 2, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim + self.char_embedding_dim,
                            self.hidden_dim // 2,
                            num_layers=self.num_layers,
                            bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, 1)

        # Set tokenizer
        self.tokenizer = tokenizer

        self.tokenize_fn = WhitespaceTokenizer().tokenize

    def forward(self, sent_batch):
        sentence = sent_batch[0]
        tokens = self.tokenizer.texts_to_sequences([sentence])

        tokens = to_gpu(torch.LongTensor(tokens))

        word_embeds = self.embedding(tokens).permute(1, 0, 2)
        # print('word_embeds: %s' % str(word_embeds.size()))

        char_embeds = self.word_encoder([
            remove_tone_marks(token) for token in sentence
        ]).unsqueeze(1)
        # print('char_embeds: %s' % str(char_embeds.size()))

        sentence_in = torch.cat((word_embeds, char_embeds), dim=-1)

        seq_len = len(sentence_in)

        # embeds = sentence_in.view(seq_len, 1, -1)  # [seq_len, batch_size, features]
        lstm_out, _ = self.lstm(sentence_in)
        lstm_out = lstm_out.view(seq_len, self.hidden_dim)
        tags = self.hidden2tag(lstm_out).squeeze(1)

        return tags

class VNTokenizerWrapper(IModel):

    def __init__(self, config={}, *args, **kwargs):
        super(VNTokenizerWrapper, self).__init__(
            model_class=VNTokenizer, 
            config=config,
            featurizer=BasicFeaturizer(config),
            *args, **kwargs
        )

    def get_layer_groups(self):
        model = self.model
        return [
            *zip(model.word_encoder.get_layer_groups()),
            model.lstm,
            model.hidden2tag
        ]