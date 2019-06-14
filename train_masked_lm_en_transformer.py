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
    MODEL_PATH = 'en-masked-lm-test.bin'
    model_config = dotdict({
        'num_words': 30000,
        'hidden_size': 768,
        'num_hidden_layers': 12,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'hidden_act': 'gelu',
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1,
        'max_position_embeddings': 100,
        'featurizer_seq_len': 100, # same as above
        'type_vocab_size': 2,
        'initializer_range': 0.02,
        'use_adasoft': True
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
    BATCH_SIZE = 200

    if path.exists(SAVE_PATH):
        print('Loading from previously saved file')
        dataset.load(SAVE_PATH, model)
    else:
        paths = [
            # path.join(BASE_PATH, 'data/wikitext2/wiki.train.tokens'),
            path.join(BASE_PATH, 'data/wikitext103raw/wiki.train.raw')
        ]
        bookcorpus_path = path.join(BASE_PATH, 'data/bookcorpus')
        if path.exists(bookcorpus_path):
            paths.extend([
                path.join(bookcorpus_path, filename)
                for filename in listdir(bookcorpus_path)
                if filename.lower().endswith('txt')
            ])
        
        billionword_path = path.join(BASE_PATH, 'data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled')
        if path.exists(billionword_path):
            paths.extend([
                path.join(billionword_path, filename)
                for filename in listdir(billionword_path)
                # if filename.lower().endswith('txt')
            ])

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
    n_epochs=10
    learner = LanguageModelLearner(model,
        optimizer_fn=BertAdam,
        optimizer_kwargs={
            'lr': 1e-4,
            'warmup': 0.04,
            't_total':  n_epochs * (len(dataset) // BATCH_SIZE)}
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
        epochs=n_epochs,
        callbacks=[
            PrintLoggerCallback(log_every_batch=1000, log_every=1, metrics=['loss']),
            # TensorboardCallback(log_every_batch=100, log_every=-1, metrics=['loss']),
            ModelCheckpointCallback(metrics=['loss']),
            ReduceLROnPlateau(reduce_factor=4, patience=2)
        ],
        gradient_accumulation_steps=1,
        fp16=True
        # optimize_on_cpu=True,
    )
