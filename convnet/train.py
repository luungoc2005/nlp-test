import torch
import torch.optim as optim

from tqdm import tqdm, trange
from config import START_TAG, STOP_TAG

from tensorboardX import SummaryWriter
from os import path, getcwd

from convnet.model import TextCNN
from common.utils import wordpunct_tokenize, to_categorical, get_datetime_hostname, prepare_vec_sequence, word_to_vec, timeSince

import time

BASE_PATH = path.join(getcwd(), 'bilstm/')
SAVE_PATH = path.join(BASE_PATH, 'model/model.bin')
LOG_DIR = path.join(BASE_PATH, 'logs/')

"""
Input is in the form of tuples of (class:int, sent:string)
"""
def process_input(sents):
    return [
        label, wordpunct_tokenize(sent)
        for label, sent in sents
    ]

def _train(input_variable, output_variable, model, criterion, optimizer):
    model.zero_grad()
    
    # Prepare training data
    sentence_in = prepare_vec_sequence(input_variable, word_to_vec)
    target_tensor = to_categorical(output_variable)

    # Run the forward pass
    output = model(input_variable)
    loss = criterion(output, target_tensor)

    return loss
