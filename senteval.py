from config import BASE_PATH
from os import path

MODEL_PATH = path.join(BASE_PATH, 'output/model/checkpoint_157_96.75.bin')
print(MODEL_PATH)

import torch
import numpy as np
from sent_to_vec.model import *

ENCODER_DIM = 2400
encoder = QRNNEncoder(hidden_dim=ENCODER_DIM, num_layers=3)
nli_net = NLINet(encoder=encoder, lstm_dim=ENCODER_DIM, bidirectional_encoder=False)

cp_data = torch.load(MODEL_PATH)
nli_net.load_state_dict(cp_data['nli_state'])
print(nli_net)
print('lr: %s, step: %s' % (cp_data['optimizer_state']['param_groups'][0]['lr'], cp_data['epoch']))

if torch.cuda.is_available():
    nli_net = nli_net.cuda()

import torch.nn.functional as F

def encode(sent, batch=False):
    if not batch:
        sent_input = process_input([sent])
        print(sent_input)
    else:
        sent_input = process_input(sent)
    sent_batch, sent_len = process_batch(sent_input)
    if torch.cuda.is_available():
        sent_batch = sent_batch.cuda()
    with torch.no_grad():
        encoder.eval()
        embs = encoder((sent_batch, sent_len))
        encoder.train()
    if not batch and len(embs.size()) > 1:
        embs = embs.squeeze(0)
    return embs.cpu().data.numpy()


def prepare(params, samples):
    return

def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embs = encode(batch, batch=True)
    return embs

PATH_TO_SENTEVAL = path.join(BASE_PATH, 'third_party/SentEval/')
PATH_TO_DATA = path.join(BASE_PATH, 'third_party/SentEval/data/')

import sys

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import logging
import senteval

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 25}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                    'SICKRelatedness', 'SICKEntailment', 'STSBenchmark',
                    'SNLI', 'ImageCaptionRetrieval', 'STS12', 'STS13',
                    'STS14', 'STS15', 'STS16',
                    'Length', 'WordContent', 'Depth', 'TopConstituents',
                    'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                    'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)