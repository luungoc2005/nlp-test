import torch
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
# from config import SENTENCE_DIM
# import numpy as np

from tensorboardX import SummaryWriter
from os import path

from common.utils import argmax, get_datetime_hostname, timeSince
from text_classification.fast_text.model import FastText, FastTextDataset

import time

BASE_PATH = path.dirname(__file__)
SAVE_PATH = path.join(BASE_PATH, 'model/model.bin')
LOG_DIR = path.join(BASE_PATH, 'logs/')


def _train(input_variable, output_variable, model, criterion, optimizer, adam_decay=0):
    optimizer.zero_grad()

    # Run the forward pass
    logits = model(input_variable)

    loss = criterion(logits, output_variable)

    loss.backward()

    if adam_decay > 0:
        for group in optimizer.param_groups():
            for param in group['params']:
                param.data = param.data.add(-adam_decay * group['lr'], param.data)

    optimizer.step()

    return loss.item()


def trainIters(data,
               classes,
               batch_size=32,
               n_iters=50,
               log_every=10,
               optimizer='rmsprop',
               learning_rate=1e-2,
               weight_decay=None,
               verbose=2,
               patience=10,
               save_path=None):

    save_path = save_path or SAVE_PATH
    num_classes = len(classes)
    # input_data = process_input(data)
    cpu_count = mp.cpu_count()

    # Set class weights - this is kinda rough...
    weights = torch.zeros(num_classes)
    for _, class_idx in data:
        weights[int(class_idx)] += 1
    for class_idx in range(num_classes):
        weights[int(class_idx)] = 1 / weights[int(class_idx)]

    print('Training started')
    # criterion = nn.CrossEntropyLoss(weight=weights)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss(weight=weights, reduction='sum')
    model = FastText(classes=num_classes)

    # weight_decay = 1e-4 by default for SGD
    if optimizer == 'adam':
        weight_decay = weight_decay or 0
        adam_decay = weight_decay
        model_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate)
    elif optimizer == 'rmsprop':
        weight_decay = weight_decay or 1e-5
        adam_decay = 0
        model_optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay)
    else:
        weight_decay = weight_decay or 1e-4
        adam_decay = 0
        model_optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
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
    best_loss = 1e15
    wait = 0

    if verbose == 2:
        iterator = trange(1, n_iters + 1, desc='Epochs', leave=False)
    else:
        iterator = range(1, n_iters + 1)

    # For timing with verbose=1
    dataset = FastTextDataset(data, num_classes)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=cpu_count)

    start = time.time()
    for epoch in iterator:
        for _, data_batch in enumerate(data_loader, 0):
            model.train()
            sentences = data_batch['sentence']
            labels = data_batch['label']

            real_batch += len(labels)  # real batch size

            # Run the training epoch
            loss = _train(sentences, labels, model, criterion, model_optimizer, adam_decay)

            loss_total += loss

            accuracy_total += evaluate(model, sentences, labels)

            if verbose == 2:
                iterator.set_description('Minibatch: %s' % real_batch)

        loss_total = loss_total / real_batch
        accuracy_total = accuracy_total / real_batch

        print_accuracy_total += accuracy_total
        print_loss_total += loss_total

        writer.add_scalar(LOSS_LOG_FILE, loss_total, epoch)
        all_losses.append(loss_total)

        if loss_total < best_loss:
            best_loss = loss_total
            wait = 1
        else:
            if wait >= patience:
                print('Early stopping')
                break
            wait += 1

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

    print('Calibrating model')
    model._calibrate(data_loader, weights)
    print('Training completed')

    torch.save({
        'classes': classes,
        'state_dict': model.state_dict(),
    }, save_path)

    LOG_JSON = path.join(LOG_DIR, 'all_scalars.json')
    writer.export_scalars_to_json(LOG_JSON)
    writer.close()

    return all_losses, model


def evaluate(model, input, output):
    model.eval()
    with torch.no_grad():
        correct = 0
        result = model(input)
        for idx, gt_class in enumerate(output):
            pred_class = argmax(result[idx])
            gt_class = argmax(gt_class)
            if gt_class == pred_class:
                correct += 1
    return float(correct)


def evaluate_all(model, data):
    model.eval()
    with torch.no_grad():
        # Slowly evaluates sample-by-sample
        correct = 0
        total = len(data)
        for sentence, gt_class in data:
            pred_class = argmax(model([sentence]))
            gt_class = argmax(gt_class)
            if gt_class == pred_class:
                correct += 1

    return float(correct) / float(total)
