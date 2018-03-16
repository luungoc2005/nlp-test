import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm, trange
from config import START_TAG, STOP_TAG

from tensorboardX import SummaryWriter
from os import path, getcwd

from convnet.model import TextCNN
from common.utils import argmax, wordpunct_tokenize, to_categorical, get_datetime_hostname, prepare_vec_sequence, word_to_vec, timeSince

import time

BASE_PATH = path.join(getcwd(), 'convnet/')
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

    loss.backward()
    optimizer.step()

    return loss.data[0]

def trainIters(data,
               log_every=10,
               optimizer='adam',
               learning_rate=0.01,
               weight_decay=None,
               verbose=2)
    
    input_data = process_input(data)
    num_classes = len(set([label for label, _ in input_data]))

    model = TextCNN(classes=num_classes)
    criterion = nn.NLLLoss()

    # weight_decay = 1e-4 by default for SGD
    if optimizer == 'adam':
        weight_decay = weight_decay or 0
        model_optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay)
    else:
        weight_decay = weight_decay or 1e-4
        model_optimizer = optim.SGD(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay)

    LOSS_LOG_FILE = path.join(LOG_DIR, 'neg_loss')
    INST_LOG_DIR = path.join(LOG_DIR, get_datetime_hostname())
    writer = SummaryWriter(log_dir=INST_LOG_DIR)

    loss_total = 0
    print_loss_total = 0

    if verbose == 2:
        iterator = trange(1, n_iters + 1, desc='Epochs', leave=False)
        epoch_iterator = tqdm(input_data)
    else:
        iterator = range(1, n_iters + 1)
        epoch_iterator = input_data
    
    # For timing with verbose=1

    start = time.time()
    for epoch in iterator:
        for label, sentence in epoch_iterator:
            loss = _train(sentence, label, model, model_optimizer)
            loss_total += loss
            print_loss_total += loss
        
        writer.add_scalar(LOSS_LOG_FILE, loss_total, epoch)
        loss_total = 0

        if epoch % log_every == 0:
            accuracy = evaluate(model, data, tag_to_ix)

            if verbose == 1:
                print_loss_avg = print_loss_total / log_every
                progress = float(epoch) / float(n_iters)
                print('%s (%d %d%%) %.4f - acurracy: %.4f' % (timeSince(start, progress),
                    epoch, 
                    progress * 100, 
                    print_loss_avg,
                    accuracy))
            
            print_loss_total = 0

    torch.save(model.state_dict(), SAVE_PATH)

    LOG_JSON = path.join(LOG_DIR, 'all_scalars.json')
    writer.export_scalars_to_json(LOG_JSON)
    writer.close()

    return model

def evaluate(model, data):
    correct = 0
    total = len(data)
    input_data = process_input(data)
    for idx, (gt_class, sentence) in enumerate(input_data):
        precheck_sent = prepare_vec_sequence(sentence, word_to_vec)
        pred_class = argmax(model(precheck_sent))
        if gt_class == pred_class:
            correct += 1
    return float(correct) / float(total)