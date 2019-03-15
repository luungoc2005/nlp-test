import torch
from common.torch_utils import to_gpu
from featurizers.basic_featurizer import BasicFeaturizer
from sent_to_vec.masked_lm.transformer import BertForMaskedLM
from common.wrappers import IModel
from config import LM_VOCAB_SIZE, LM_HIDDEN_DIM, LM_SEQ_LEN, LM_CHAR_SEQ_LEN, START_TAG, STOP_TAG, UNK_TAG, MASK_TAG

BERT_DEFAULT_CONFIG = {
    'num_words': 30000,
    'hidden_size': 768,
    'num_hidden_layers': 12,
    'num_attention_heads': 12,
    'intermediate_size': 3072,
    'hidden_act': 'gelu',
    'hidden_dropout_prob': 0.1,
    'attention_probs_dropout_prob': 0.1,
    'max_position_embeddings': 512,
    'type_vocab_size': 2,
    'initializer_range': 0.02
}

class BertLMWrapper(IModel):

    def __init__(self, config=dict(), *args, **kwargs):
        featurizer_config = config
        featurizer_config['append_sos_eos'] = True
        featurizer_config['featurizer_reserved_tokens'] = [START_TAG, STOP_TAG, UNK_TAG, MASK_TAG]
        # featurizer_config['return_mask'] = True # TODO: correctly implement masking

        super(BertLMWrapper, self).__init__(
            model_class=BertForMaskedLM, 
            config=config,
            featurizer=BasicFeaturizer(featurizer_config),
            *args, **kwargs
        )

        self.seq_len = config.get('seq_len', LM_SEQ_LEN)
        self.config = config