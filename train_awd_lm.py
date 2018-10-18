from sent_to_vec.awd_lm.model import LanguageModelWrapper
from sent_to_vec.awd_lm.train import LanguageModelLearner, WikiTextDataset
from common.callbacks import PrintLoggerCallback, EarlyStoppingCallback, ModelCheckpointCallback
from os import path
from config import BASE_PATH
from torch.optim import RMSprop

model = LanguageModelWrapper({'embedding_dim': 600}) # small model

dataset = WikiTextDataset()

SAVE_PATH = path.join(BASE_PATH, 'wikitext-data.bin')
if path.exists(SAVE_PATH):
    print('Loading from previously saved file')
    dataset.load(SAVE_PATH, model, batch_size=64)
else:
    dataset.initialize(model, data_path=[
        path.join(BASE_PATH, 'data/wikitext2/wiki.train.tokens'),
        path.join(BASE_PATH, 'data/wikitext103/wiki.train.tokens')
    ], batch_size=64)
    dataset.save()

learner = LanguageModelLearner(model, optimizer_fn='rmsprop')

print('Dataset: {} minibatches per epoch'.format(len(dataset)))
learner.fit(
    training_data=dataset,
    batch_size=1,
    epochs=1000,
    callbacks=[
        PrintLoggerCallback(log_every_batch=1000, log_every=1), 
        EarlyStoppingCallback(),
        ModelCheckpointCallback()
    ]
)