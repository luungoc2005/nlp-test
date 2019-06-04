import torch
import torch.nn as nn
import torch.nn.functional as F
from common.modules import BRNNWordEncoder
from sent_to_vec.masked_lm.transformer import BertLayerNorm, BertEncoder
from common.torch_utils import to_gpu
from common.utils import word_to_vec, dotdict, pad_sequences
from common.crf import CRF
from common.wrappers import IModel
from config import CHAR_EMBEDDING_DIM, START_TAG, STOP_TAG
from typing import List, Optional

DEFAULT_CONFIG = dotdict({
    'mode': 'transformer',
    'char_embedding_dim': 50,
    'hidden_size': 350,
    'num_hidden_layers': 4,
    'num_attention_heads': 7,
    'intermediate_size': 128,
    'hidden_act': 'gelu',
    'hidden_dropout_prob': 0.1,
    'attention_probs_dropout_prob': 0.1,
    'max_position_embeddings': 256,
    'featurizer_seq_len': 256, # same as above
    'initializer_range': 0.02,
})

class TransformerPretrainedDualEmbedding(nn.Module):

    def __init__(self, config=DEFAULT_CONFIG):
        super(TransformerPretrainedDualEmbedding, self).__init__()
        self.config = config

        self.char_embedding_dim = config.get('char_embedding_dim', CHAR_EMBEDDING_DIM)
        self.word_embedding_dim = config.hidden_size - self.char_embedding_dim
        self.use_position_embeddings = config.use_position_embeddings

        if self.use_position_embeddings:
            # self.word_embeddings = nn.Embedding(config.num_words, config.hidden_size, padding_idx=0)
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)

        self.char_encoder = to_gpu(BRNNWordEncoder(self.char_embedding_dim, rnn_type='LSTM'))
        
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, sent_batch: List[List[str]]):
        max_length = min(max([len(sent) for sent in sent_batch]), self.config.max_position_embeddings)

        words_embeddings = to_gpu(torch.FloatTensor(
            word_to_vec(sent_batch, pad_to_length=max_length)
        ))
        
        chars_embeddings = to_gpu(torch.stack([
            to_gpu(
                torch.cat(
                    (self.char_encoder(sent), 
                    to_gpu(torch.zeros(max_length - len(sent), self.char_embedding_dim))), 
                    dim=0
                )
                if len(sent) < max_length
                else self.char_encoder(sent)[:max_length]
                    if len(sent) > max_length
                    else self.char_encoder(sent)
            )
            for sent in sent_batch
        ], 0))

        position_embeddings = None

        embeddings = torch.cat([words_embeddings, chars_embeddings], dim=-1)
        
        if self.use_position_embeddings:
            position_ids = torch.arange(max_length, dtype=torch.long, device=words_embeddings.device)
            position_ids = position_ids.unsqueeze(0).expand(words_embeddings.size(0), words_embeddings.size(1))

            position_embeddings = self.position_embeddings(position_ids)

            embeddings = embeddings + position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class TransformerSequenceTagger(nn.Module):

    def __init__(self, config=DEFAULT_CONFIG):
        super(TransformerSequenceTagger, self).__init__()
        self.config = config
        self.use_crf = True

        assert self.config.mode.lower() in ['transformer', 'lstm']

        self.tag_to_ix = config.get('tag_to_ix', {})
        self.tagset_size = max(self.tag_to_ix.values()) + 1

        self.embeddings = TransformerPretrainedDualEmbedding(config)

        if self.config.mode == 'transformer':
            self.encoder = BertEncoder(config)
        else:
            self.encoder = nn.LSTM(
                config.hidden_size, 
                config.hidden_size // 2, 
                num_layers=config.num_hidden_layers,
                dropout=config.hidden_dropout_prob,
                bidirectional=True, batch_first=True
            )
        
        self.hidden2tag = nn.Linear(config.hidden_size, self.tagset_size)
        self.crf = to_gpu(CRF(self.tagset_size))
        
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, sent_batch: List[List[str]], output_all_encoded_layers: bool = False, decode_tags: Optional[bool] = None):
        max_length = min(max([len(sent) for sent in sent_batch]), self.config.max_position_embeddings)

        attention_mask = torch.zeros(len(sent_batch), max_length).long()
        for ix, sent in enumerate(sent_batch):
            attention_mask[ix, :min(len(sent), max_length)] = 1

        # See BertModel class
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(sent_batch)
        encoded_layers = self.encoder(embedding_output,
            to_gpu(extended_attention_mask),
            output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        tags_output = self.hidden2tag(
            sequence_output.view(sequence_output.size(0) * sequence_output.size(1), sequence_output.size(2))
        )
        tags_output = tags_output.view(sequence_output.size(0), sequence_output.size(1), -1)

        if decode_tags is None: decode_tags = not self.training
        if decode_tags:
            if self.use_crf:
                seq_lens = to_gpu(
                    torch.LongTensor([
                        len(sent) for sent in sent_batch
                    ])
                )
                tags_output = self.crf.decode(
                    tags_output, 
                    seq_lens
                )
            else:
                tags_output = torch.max(
                    tags_output, 
                    dim=-1
                )[1]

        return tags_output, encoded_layers, sent_batch
    
class TransformerSequenceTaggerWrapper(IModel):

    def __init__(self, config=DEFAULT_CONFIG, *args, **kwargs):
        super(TransformerSequenceTaggerWrapper, self).__init__(
            model_class=TransformerSequenceTagger,
            config=config,
            *args, **kwargs
        )
        self.config = DEFAULT_CONFIG
        self.config.update(config)

        self.tag_to_ix = config.get('tag_to_ix', {START_TAG: 0, STOP_TAG: 1})
        self.task = config.get('task', 'ner')
        assert self.task in ['pos', 'chunking', 'ner']
        
        # Invert the tag dictionary
        self.ix_to_tag = {value: key for key, value in self.tag_to_ix.items()}

    def load_state_dict(self, state_dict):
        config = state_dict['config']

        # re-initialize model with loaded config
        self.tag_to_ix = config.get('tag_to_ix', {START_TAG: 0, STOP_TAG: 1})
        self.ix_to_tag = {value: key for key, value in self.tag_to_ix.items()}

        self.task = config.get('task', 'ner')

    def infer_predict(self, logits, delimiter=''):
        # print(logits)
        tag_seq_batch, _, sent_batch = logits
        result = []

        # print(tag_seq_batch.size())
        # tag_seq_batch = torch.max(tag_seq_batch, 2)[1]
        # print(tag_seq_batch.size())
        for sent_ix, tokens_in in enumerate(sent_batch):
            tag_seq = [self.ix_to_tag[int(tag)] for tag in tag_seq_batch[sent_ix]]

            if self.task == 'ner':
                entities = {}
                entity_name = ''
                buffer = []

                for idx, tag_name in enumerate(tag_seq):
                    if len(tag_name) > 2 and tag_name[:2] in ['B-', 'I-']:
                        new_entity_name = tag_name[2:]
                        if entity_name != '' and \
                                (tag_name[:2] == 'B-' or entity_name != new_entity_name):
                            # Flush the previous entity
                            if entity_name not in entities:
                                entities[entity_name] = []
                                entities[entity_name].append(delimiter.join(buffer))
                                buffer = []

                        entity_name = new_entity_name

                    # If idx is currently inside a tag
                    if entity_name != '':
                        # Going outside the tag
                        if idx == len(tag_seq) - 1 or \
                                tag_name == '-' or \
                                tag_name == 'O':

                            # if end of tag sequence then append the final token
                            if idx == len(tag_seq) - 1 and tag_name != '-':
                                buffer.append(tokens_in[idx])

                            if entity_name not in entities:
                                entities[entity_name] = []
                            entities[entity_name].append(delimiter.join(buffer))
                            buffer = []
                            entity_name = ''
                        else:
                            buffer.append(tokens_in[idx])

                # return [entities]
                result.append([{ 'name': key, 'values': value } for key, value in entities.items()])
            elif self.task == 'pos':
                ret_list = [(token, tag_seq[idx]) for idx, token in tokens_in]
                result.append(ret_list)
            else:
                result.append(tag_seq)
        
        return result