import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from sent_to_vec.masked_lm.transformer import BertPreTrainedModel, BertLayerNorm, BertEncoder, BertPooler, BertOnlyNSPHead
from common.torch_utils import to_gpu
from common.utils import word_to_vec, dotdict, pad_sequences
from common.wrappers import IModel
from config import CHAR_EMBEDDING_DIM, START_TAG, STOP_TAG
from typing import List, Optional

DEFAULT_CONFIG = dotdict({
    'char_embedding_dim': 100,
    'hidden_size': 300,
    'num_hidden_layers': 4,
    'num_attention_heads': 10,
    'intermediate_size': 1024,
    'hidden_act': 'gelu',
    'hidden_dropout_prob': 0.1,
    'type_vocab_size': 2,
    'attention_probs_dropout_prob': 0.1,
    'max_position_embeddings': 256,
    'featurizer_seq_len': 256, # same as above
    'initializer_range': 0.02,
})

class TransformerSimpleEncoder(BertPreTrainedModel):

    def __init__(self, config=DEFAULT_CONFIG):
        super(TransformerSimpleEncoder, self).__init__()
        self.config = config

        self.word_embedding_dim = config.hidden_size

        # self.word_embeddings = nn.Embedding(config.num_words, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

        self.char_encoder = BRNNWordEncoder(self.char_embedding_dim, rnn_type='LSTM')
        
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.apply(self.init_bert_weights)

    def forward(self, words_embeddings: torch.FloatTensor, token_type_ids: Optional[torch.LongTensor] = None):
        max_length = words_embeddings.size(1)

        position_ids = torch.arange(max_length, dtype=torch.long, device=words_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(words_embeddings.size(0), words_embeddings.size(1))

        position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class TransformerSimpleLM(BertPreTrainedModel):

    def __init__(self, config=DEFAULT_CONFIG):
        super(TransformerSimpleLM, self).__init__()
        self.config = config

        self.embeddings = TransformerSimpleEncoder(config)
        self.encoder = BertEncoder(config)

        self.apply(self.init_bert_weights)

    def forward(self, words_embeddings: torch.FloatTensor, token_type_ids: Optional[torch.LongTensor] = None, output_all_encoded_layers: bool = False):
        max_length = words_embeddings.size(1)

        attention_mask = torch.zeros(len(sent_batch), max_length).long()
        for ix, sent in enumerate(sent_batch):
            attention_mask[ix, :min(len(sent), max_length)] = 1

        # See BertModel class
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(sent_batch, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        return sequence_output, encoded_layers

class SimpleLMForNextSentenceClassification(BertPreTrainedModel):

    def __init__(self, config=DEFAULT_CONFIG):
        super(SimpleLMForNextSentenceClassification, self).__init__()
        self.config = config

        self.encoder = TransformerSimpleLM(config)
        self.pooler = BertPooler(config)
        self.cls = BertOnlyNSPHead(config)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
            output_all_encoded_layers=False)
        seq_relationship_score = self.cls(pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score

    
class TransformerSimpleLMWrapper(IModel):

    def __init__(self, config=DEFAULT_CONFIG, *args, **kwargs):
        super(TransformerSimpleLMWrapper, self).__init__(
            model_class=SimpleLMForNextSentenceClassification,
            config=config,
            *args, **kwargs
        )
        self.config = DEFAULT_CONFIG
        self.config.update(config)
