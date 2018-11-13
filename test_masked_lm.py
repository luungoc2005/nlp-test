
from sent_to_vec.masked_lm.model import BiLanguageModelWrapper
from sent_to_vec.masked_lm.data import WikiTextDataset, collate_seq_lm_fn

from torch.utils.data import DataLoader
from config import BASE_PATH
from os import path
import torch

if __name__ == '__main__':
    dataset = WikiTextDataset()

    SAVE_PATH = path.join(BASE_PATH, dataset.get_save_name())
    model = BiLanguageModelWrapper(from_fp='masked-lm-checkpoint.bin')
    # model = BiLanguageModelWrapper()
    model.init_model()

    if path.exists(SAVE_PATH):
        print('Loading from previously saved file')
        dataset.load(SAVE_PATH, model)
    else:
        dataset.initialize(model, data_path=[
            # path.join(BASE_PATH, 'data/wikitext2/wiki.train.tokens'),
            path.join(BASE_PATH, 'data/wikitext103/wiki.train.tokens')
        ])
        dataset.save()

    # print(dataset.get_sent(4))
    loader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_seq_lm_fn
    )
    inputs, outputs = next(iter(loader))

    result, hidden = model(inputs)
    result = torch.max(result, dim=1)[1].view(inputs.size(0), inputs.size(1))
    
    X_decoded = model.featurizer.inverse_transform(inputs.t().contiguous())
    y_decoded = model.featurizer.inverse_transform(result.t().contiguous())

    for ix in range(inputs.size(1)):
        print('Source: {}'.format(X_decoded[ix]))
        print('Result: {}'.format(y_decoded[ix]))
        print('---')