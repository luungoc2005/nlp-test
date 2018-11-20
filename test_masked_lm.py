
from sent_to_vec.masked_lm.model import BiLanguageModelWrapper
from sent_to_vec.masked_lm.data import WikiTextDataset, collate_seq_lm_fn
from torch.utils.data import DataLoader
from common.torch_utils import to_gpu
from common.metrics import accuracy
from config import BASE_PATH
from os import path
from tqdm import trange
import torch

def pad_sents(first_array, second_array):
    first_res = []
    second_res = []
    for ix, token in enumerate(first_array):
        token2 = second_array[ix]
        word_len = max(len(token), len(token2))

        first_res.append(token.ljust(word_len))
        second_res.append(token2.ljust(word_len))

    return first_res, second_res


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
    BATCH_SIZE = 16
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_seq_lm_fn,
        num_workers=torch.multiprocessing.cpu_count() - 1
    )

    TEST_EPOCHS = 100
    total_accuracy = 0.

    print(model)
    for _ in trange(TEST_EPOCHS):
        inputs, outputs = next(iter(loader))
        inputs, outputs = to_gpu(inputs), to_gpu(outputs)

        outputs = outputs.view(inputs.size(0), inputs.size(1))

        result, hidden = model(inputs)
        result = torch.max(result, dim=1)[1].view(inputs.size(0), inputs.size(1))
        
        total_accuracy += accuracy(result, outputs)
    
    total_accuracy /= TEST_EPOCHS
    total_accuracy = (total_accuracy - .85) / .15
    print('Accuracy over %s test sentences: %4f' % (
        TEST_EPOCHS * BATCH_SIZE,
        total_accuracy * 100
    ))

    X_decoded = model.featurizer.inverse_transform(inputs.cpu().t().contiguous())
    y_t_decoded = model.featurizer.inverse_transform(outputs.cpu().t().contiguous())
    y_decoded = model.featurizer.inverse_transform(result.cpu().t().contiguous())

    for ix in range(BATCH_SIZE):
        y_t, y = pad_sents(y_t_decoded[ix], y_decoded[ix])
        print('ST: {}'.format(' '.join(X_decoded[ix])))
        print('GT: {}'.format(' '.join(y_t)))
        print('RT: {}'.format(' '.join(y)))
        print('---')