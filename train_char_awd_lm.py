from sent_to_vec.awd_lm.model import LanguageModelWrapper
from sent_to_vec.awd_lm.train import LanguageModelLearner
from sent_to_vec.awd_lm.data import WikiTextDataset
from common.callbacks import PrintLoggerCallback, EarlyStoppingCallback, ModelCheckpointCallback, TensorboardCallback
from os import path
from config import BASE_PATH
from torch.optim import RMSprop

model = LanguageModelWrapper({
    'char_level': True,
    'rnn_type': 'LSTM',
    'embedding_dim': 100,
    'hidden_size': 2400,
    'n_layers': 3
})

dataset = WikiTextDataset()

SAVE_PATH = path.join(BASE_PATH, 'wikitext-char-data.bin')
BATCH_SIZE = 128

if path.exists(SAVE_PATH):
    print('Loading from previously saved file')
    dataset.load(SAVE_PATH, model, batch_size=BATCH_SIZE)
else:
    dataset.initialize(model, data_path=[
        # path.join(BASE_PATH, 'data/wikitext2raw/wiki.train.raw'),
        path.join(BASE_PATH, 'data/wikitext103raw/wiki.train.raw')
    ], batch_size=BATCH_SIZE)
    dataset.save()

# learner = LanguageModelLearner(model, 
#     optimizer_fn='sgd', 
#     optimizer_kwargs={'lr': 30, 'weight_decay': 1.2e-6}
# )
learner = LanguageModelLearner(model, 
    optimizer_fn='adam',
    optimizer_kwargs={'weight_decay': 1.2e-6}
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
    batch_size=1,
    epochs=1000,
    callbacks=[
        PrintLoggerCallback(log_every_batch=1000, log_every=1, metrics=['loss']),
        TensorboardCallback(log_every_batch=100, log_every=-1, metrics=['loss']),
        ModelCheckpointCallback(metrics=['loss'])
    ]
)