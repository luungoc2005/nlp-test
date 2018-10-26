from sent_to_vec.awd_lm.model import LanguageModelWrapper
from sent_to_vec.awd_lm.train import LanguageModelLearner
from sent_to_vec.awd_lm.data import WikiTextDataset
from common.callbacks import PrintLoggerCallback, EarlyStoppingCallback, ModelCheckpointCallback, TensorboardCallback
from os import path
from config import BASE_PATH
from torch.optim import RMSprop

model = LanguageModelWrapper({'embedding_dim': 400}) # small model

dataset = WikiTextDataset()

SAVE_PATH = path.join(BASE_PATH, 'wikitext-data.bin')
if path.exists(SAVE_PATH):
    print('Loading from previously saved file')
    dataset.load(SAVE_PATH, model)
else:
    dataset.initialize(model, data_path=[
        path.join(BASE_PATH, 'data/wikitext2/wiki.train.tokens'),
        path.join(BASE_PATH, 'data/wikitext103/wiki.train.tokens')
    ])
    dataset.save()

learner = LanguageModelLearner(model, optimizer_fn='adam')

print('Dataset: {} minibatches per epoch'.format(len(dataset)))
learner.find_lr(range(25, 35), {
    'training_data': dataset,
    'batch_size': 64,
    'epochs': 1,
    'minibatches': 5000
})
# learner.fit(
#     training_data=dataset,
#     batch_size=64,
#     epochs=1000,
#     callbacks=[
#         PrintLoggerCallback(log_every_batch=1000, log_every=1, metrics=['loss']),
#         TensorboardCallback(log_every_batch=100, log_every=-1, metrics=['loss']),
#         EarlyStoppingCallback(),
#         ModelCheckpointCallback(metrics=['loss'])
#     ]
# )