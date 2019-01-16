
from sent_to_vec.masked_lm.model import BiLanguageModelWrapper
from sent_to_vec.masked_lm.data import WikiTextDataset, collate_seq_lm_fn
from torch.utils.data import DataLoader
from common.torch_utils import to_gpu
from common.metrics import accuracy
from config import BASE_PATH
from os import path
from tqdm import trange
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default='masked-lm-checkpoint.bin')
parser.add_argument("--show_raws", action='store_true')
parser.add_argument("--quantize", action='store_true')
parser.add_argument("--export_onnx", action='store_true')

args = parser.parse_args()

def pad_sents(first_array, second_array, third_array):
    first_res = []
    second_res = []
    third_res = []
    for ix, token in enumerate(first_array):
        token2 = second_array[ix]
        token3 = third_array[ix]

        word_len = max(len(token), len(token2), len(token3))

        first_res.append(token.ljust(word_len))
        second_res.append(token2.ljust(word_len))
        third_res.append(token3.ljust(word_len))

    return first_res, second_res, third_res


if __name__ == '__main__':
    dataset = WikiTextDataset()

    model = BiLanguageModelWrapper(from_fp=args.checkpoint)
    # patch to fix adasoft on older checkpoint file
    # model = BiLanguageModelWrapper()
    model.init_model(update_configs={'use_adasoft': False})
    # model.save(args.checkpoint)

    SAVE_PATH = path.join(BASE_PATH, dataset.get_save_name())
    if not path.exists(SAVE_PATH):
        SAVE_PATH = path.join(BASE_PATH, dataset.get_save_name(model.config['num_words']))

    if path.exists(SAVE_PATH):
        print('Loading from previously saved file at %s' % SAVE_PATH)
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
        num_workers=0
    )

    TEST_EPOCHS = 100
    # total_accuracy = 0.
    total_correct = 0
    total_count = 0

    print(model)

    if args.quantize:
        model.quantize()
        model.save('masked-lm-quantized.bin')

    if args.export_onnx:
        dummy_input = torch.LongTensor(70, 1).random_(1, 10)
        model.export_onnx(dummy_input, 'masked-lm.onnx')

    for _ in trange(TEST_EPOCHS):
        inputs, outputs = next(iter(loader))
        inputs, outputs = to_gpu(inputs), to_gpu(outputs)

        outputs = outputs.view(inputs.size(0), inputs.size(1))

        result, hidden = model(inputs)
        result = torch.max(result, dim=1)[1].view(inputs.size(0), inputs.size(1))
        
        mask = (outputs != 0)
        total_count += mask.sum().item()
        total_correct += (result.masked_select(mask) == outputs.masked_select(mask)).sum().item()
        # total_accuracy += accuracy(result.masked_select(mask), outputs.masked_select(mask))
    
    # total_accuracy /= TEST_EPOCHS
    total_accuracy = total_correct / total_count
    print('Accuracy over %s test sentences: %4f' % (
        TEST_EPOCHS * BATCH_SIZE,
        total_accuracy * 100
    ))

    X_decoded = model.featurizer.inverse_transform(inputs.cpu().t().contiguous())
    y_t_decoded = model.featurizer.inverse_transform(outputs.cpu().t().contiguous())
    y_decoded = model.featurizer.inverse_transform(result.cpu().t().contiguous())

    if args.show_raws == False:
        y_decoded = [
            [token if y_t_decoded[sent_ix][ix].strip() != '' else '' for ix, token in enumerate(sent)] 
            for sent_ix, sent in enumerate(y_decoded)
        ]

    for ix in range(BATCH_SIZE):
        x, y_t, y = pad_sents(X_decoded[ix], y_t_decoded[ix], y_decoded[ix])
        print('ST: {}'.format(' '.join(x)))
        print('GT: {}'.format(' '.join(y_t)))
        print('RT: {}'.format(' '.join(y)))
        print('---')