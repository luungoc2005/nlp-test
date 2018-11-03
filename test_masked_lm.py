
from sent_to_vec.masked_lm.model import BiLanguageModelWrapper
from sent_to_vec.masked_lm.data import WikiTextDataset
from config import BASE_PATH
from os import path

dataset = WikiTextDataset()

model = BiLanguageModelWrapper(from_fp='masked-lm-checkpoint.bin')
model.init_model()

SAVE_PATH = path.join(BASE_PATH, 'wikitext-masked-data.bin')

if path.exists(SAVE_PATH):
    print('Loading from previously saved file')
    dataset.load(SAVE_PATH, model)
else:
    dataset.initialize(model, data_path=[
        # path.join(BASE_PATH, 'data/wikitext2/wiki.train.tokens'),
        path.join(BASE_PATH, 'data/wikitext103/wiki.train.tokens')
    ])
    dataset.save()

