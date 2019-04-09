# from sent_to_vec.masked_lm.pervasive_model import PervasiveAttnLanguageModelWrapper
from sent_to_vec.masked_lm.bert_model import BertLMWrapper
from sent_to_vec.masked_lm.train import LanguageModelLearner
from sent_to_vec.masked_lm.data import WikiTextDataset
from common.callbacks import PrintLoggerCallback, EarlyStoppingCallback, ModelCheckpointCallback, TensorboardCallback, ReduceLROnPlateau
from os import path, listdir
from config import BASE_PATH
# from torch.optim import RMSprop
from common.modules import BertAdam
from common.utils import dotdict

# alias for old path
import sys
from common.preprocessing import keras
sys.modules['common.keras_preprocessing'] = keras

if __name__ == '__main__':
    MODEL_PATH = 'vi-masked-lm-test.bin'
    model_config = dotdict({
        'num_words': 20000,
        'hidden_size': 400,
        'num_hidden_layers': 3,
        'num_attention_heads': 8,
        'intermediate_size': 1140,
        'hidden_act': 'gelu',
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1,
        'max_position_embeddings': 128,
        'featurizer_seq_len': 128, # same as above
        'type_vocab_size': 2,
        'initializer_range': 0.02,
        'use_adasoft': False,
    })
    if path.exists(MODEL_PATH):
        print('Resuming from saved checkpoint')
        # model = PervasiveAttnLanguageModelWrapper(from_fp=MODEL_PATH)
        model = BertLMWrapper(from_fp=MODEL_PATH)
        model.init_model(update_configs=model_config)
    else:
        # model = PervasiveAttnLanguageModelWrapper({
        #     'n_layers': 6,
        #     'tie_weights': True,
        #     'embedding_dim': 300,
        #     'hidden_dim': 300,
        #     'use_adasoft': True,
        #     'num_words': 50000
        # }) # large model
        model = BertLMWrapper(model_config)

    dataset = WikiTextDataset()

    SAVE_PATH = path.join(BASE_PATH, 'wikitext-maskedlm-data.bin')
    BATCH_SIZE = 32

    if path.exists(SAVE_PATH):
        print('Loading from previously saved file')
        dataset.load(SAVE_PATH, model)
    else:
        paths = [
            # path.join(BASE_PATH, 'data/wikitext2/wiki.train.tokens'),
            path.join(BASE_PATH, 'vi-corpus/vi.all'),
            path.join(BASE_PATH, 'vi-corpus/vi_corpus_1.txt'),
            path.join(BASE_PATH, 'vi-corpus/vi_corpus_2.txt'),
        ]
        dataset.initialize(model, data_path=paths)
        dataset.save(SAVE_PATH)

    # learner = LanguageModelLearner(model, 
    #     optimizer_fn='sgd', 
    #     optimizer_kwargs={'lr': 30, 'weight_decay': 1.2e-6}
    # )
    # learner = LanguageModelLearner(model, 
    #     optimizer_fn='sgd',
    #     optimizer_kwargs={'lr': 10, 'weight_decay': 1.2e-6}
    # )
    learner = LanguageModelLearner(model,
        optimizer_fn=BertAdam,
        optimizer_kwargs={'lr': 2e-5, 'warmup': 1, 't_total': 10}
    )
    print('Dataset: {} sentences'.format(len(dataset)))
    # lr_range = list(range(25, 35))
    # losses = learner.find_lr(lr_range, {
    #     'training_data': dataset,
    #     'batch_size': BATCH_SIZE,
    #     'epochs': 1,
    #     'minibatches': 500
    # })
    # print([
    #     (lr, losses[idx]) for idx, lr in enumerate(lr_range)
    # ])
    learner.fit(
        training_data=dataset,
        batch_size=BATCH_SIZE,
        epochs=100,
        callbacks=[
            PrintLoggerCallback(log_every_batch=1000, log_every=1, metrics=['loss']),
            # TensorboardCallback(log_every_batch=100, log_every=-1, metrics=['loss']),
            ModelCheckpointCallback(metrics=['loss']),
            ReduceLROnPlateau(reduce_factor=4, patience=2)
        ],
        # gradient_accumulation_steps=3,
        # optimize_on_cpu=True,
    )
