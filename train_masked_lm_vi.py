from sent_to_vec.masked_lm.model import BiLanguageModelWrapper
from sent_to_vec.masked_lm.train import LanguageModelLearner
from sent_to_vec.masked_lm.data import WikiTextDataset
from common.callbacks import PrintLoggerCallback, EarlyStoppingCallback, ModelCheckpointCallback, TensorboardCallback, ReduceLROnPlateau
from os import path, listdir
from config import BASE_PATH
# from torch.optim import RMSprop
from common.modules import BertAdam

if __name__ == '__main__':
    MODEL_PATH = 'masked-lm-checkpoint.bin'
    if path.exists(MODEL_PATH):
        print('Resuming from saved checkpoint')
        model = BiLanguageModelWrapper(from_fp=MODEL_PATH)
    else:
        model = BiLanguageModelWrapper({
            'rnn_type': 'LSTM',
            'n_layers': 3,
            'tie_weights': True,
            'embedding_dim': 400,
            'hidden_dim': 1150,
            'alpha': 2,
            'beta': 1,
            'emb_dropout': .1,
            'h_dropout': .25,
            'w_dropout': .5,
            'rnn_dropout': 0,
            'use_adasoft': True,
            'num_words': 30000
        }) # large model

    dataset = WikiTextDataset()

    SAVE_PATH = path.join(BASE_PATH, 'vi-corpus.bin')
    BATCH_SIZE = 80

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
        optimizer_kwargs={'lr': 1e-3, 'weight_decay_rate': 1.2e-6}
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
        ]
        # gradient_accumulation_steps=3,
        # optimize_on_cpu=True,
        # fp16=True
    )
