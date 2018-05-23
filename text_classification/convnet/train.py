import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm, trange
from config import SENTENCE_DIM

from tensorboardX import SummaryWriter
from os import path

from text_classification.convnet.model import TextCNN
from common.utils import argmax, to_variable, wordpunct_tokenize, get_datetime_hostname, prepare_vec_sequence, \
    word_to_vec, timeSince

import time

BASE_PATH = path.dirname(__file__)
SAVE_PATH = path.join(BASE_PATH, 'model/model.bin')
LOG_DIR = path.join(BASE_PATH, 'logs/')


def process_input(data):
    """
    Input is in the form of tuples of (class:int, sent:string)
    """
    return [
        (prepare_vec_sequence(wordpunct_tokenize(sent), word_to_vec, SENTENCE_DIM, output='tensor'),
         label)
        for sent, label in data
    ]


def _train(input_variable, output_variable, model, criterion, optimizer):
    optimizer.zero_grad()

    # Run the forward pass
    logits = model(input_variable)

    loss = criterion(logits, output_variable)

    loss.backward()
    optimizer.step()

    return loss.item()


def trainIters(data,
               classes,
               batch_size=32,
               n_iters=50,
               log_every=10,
               optimizer='adam',
               learning_rate=1e-3,
               weight_decay=None,
               verbose=2,
               save_path=None):
    save_path = save_path or SAVE_PATH

    num_classes = len(classes)
    input_data = process_input(data)

    # get class weights
    class_weights = {}
    intents_count = float(len(data))
    weights_tensor = torch.zeros(num_classes).float()
    for _, label in data:
        if label not in class_weights:
            class_weights[label] = 1.
        else:
            class_weights[label] += 1.
    for label in class_weights:
        weights_tensor[label] = intents_count / class_weights[label]

    model = TextCNN(classes=num_classes)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

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

    LOSS_LOG_FILE = path.join(LOG_DIR, 'cross_entropy_loss')
    INST_LOG_DIR = path.join(LOG_DIR, get_datetime_hostname())
    writer = SummaryWriter(log_dir=INST_LOG_DIR)

    all_losses = []
    loss_total = 0
    accuracy_total = 0
    print_loss_total = 0
    print_accuracy_total = 0
    real_batch = 0

    if verbose == 2:
        iterator = trange(1, n_iters + 1, desc='Epochs', leave=False)
    else:
        iterator = range(1, n_iters + 1)

    # For timing with verbose=1
    start = time.time()
    data_loader = DataLoader(input_data, batch_size=batch_size)

    for epoch in iterator:
        for _, data_batch in enumerate(data_loader, 0):
            sentences, labels = data_batch
            # Prepare training data
            sentence_in, target_variable = Variable(sentences), Variable(labels.type(torch.LongTensor))

            real_batch += len(sentences)  # real batch size

            # Run the training epoch
            loss = _train(sentence_in, target_variable, model, criterion, model_optimizer)

            loss_total += loss
            accuracy_total += evaluate(model, sentence_in, labels)

        loss_total = loss_total / real_batch
        accuracy_total = accuracy_total / real_batch

        print_accuracy_total += accuracy_total
        print_loss_total += loss_total

        writer.add_scalar(LOSS_LOG_FILE, loss_total, epoch)
        all_losses.append(loss_total)

        real_batch = 0
        accuracy_total = 0
        loss_total = 0

        if epoch % log_every == 0:
            print_accuracy_total = print_accuracy_total / log_every

            if verbose == 1:
                print_loss_avg = print_loss_total / log_every
                progress = float(epoch) / float(n_iters)
                print('%s (%d %d%%) %.4f - accuracy: %.4f' % (timeSince(start, progress),
                                                              epoch,
                                                              progress * 100,
                                                              print_loss_avg,
                                                              print_accuracy_total))

            print_loss_total = 0
            print_accuracy_total = 0

    torch.save({
        'classes': classes,
        'state_dict': model.state_dict()
    }, save_path)

    LOG_JSON = path.join(LOG_DIR, 'all_scalars.json')
    writer.export_scalars_to_json(LOG_JSON)
    writer.close()

    return all_losses, model


def evaluate(model, input, output):
    with torch.no_grad():
        correct = 0

        result = model(input)

        for idx, gt_class in enumerate(output):
            pred_class = argmax(result[idx])
            if gt_class == pred_class:
                correct += 1
    return float(correct)


def evaluate_all(model, data):
    with torch.no_grad():
        correct = 0
        total = len(data)
        input_data = process_input(data)
        for sentence, gt_class in input_data:
            precheck_sent = Variable(sentence)
            pred_class = argmax(model(precheck_sent.unsqueeze(0)))
            if gt_class == pred_class:
                correct += 1
    return float(correct) / float(total)
