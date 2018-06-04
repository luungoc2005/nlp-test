import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm, trange
# from config import START_TAG, STOP_TAG

from tensorboardX import SummaryWriter
from os import path, getcwd

from entities_recognition.bilstm.model import BiLSTM_CRF
from common.utils import get_datetime_hostname, wordpunct_space_tokenize, timeSince
from common.torch_utils import lr_schedule_slanted_triangular

import time

BASE_PATH = path.join(getcwd(), 'bilstm/')
SAVE_PATH = path.join(BASE_PATH, 'model/model.bin')
LOG_DIR = path.join(BASE_PATH, 'logs/')

torch.manual_seed(7)


def process_input(data, tokenizer=wordpunct_space_tokenize):
    return [
        (tokenizer(sent), tags.split())
        for (sent, tags) in data
    ]


def _train(input_variable, target_variable, tag_to_ix, model, optimizer, grad_clip=5.):
    model.zero_grad()

    # Prepare training data
    targets = torch.LongTensor([tag_to_ix[t] for t in target_variable])

    # Run the forward pass.
    neg_log_likelihood = model.neg_log_likelihood(input_variable, targets)

    # Compute the loss, gradients, and update the parameters by
    # calling optimizer.step()
    neg_log_likelihood.backward()

    if grad_clip > 0:
        # clip_grad_norm_ in pytorch unstable
        nn.utils.clip_grad_norm(model.parameters(), grad_clip)

    optimizer.step()

    return neg_log_likelihood.data[0]


def trainIters(data, 
               tag_to_ix,
               n_iters=50,
               log_every=10,
               optimizer='adam',
               learning_rate=1e-3,
               weight_decay=1e-5,
               grad_clip=5.,
               gradual_unfreeze=False,
               tokenizer=wordpunct_space_tokenize,
               verbose=2,
               patience=4,
               save_path=None):
    save_path = save_path or SAVE_PATH
    # Invert the tag dictionary
    ix_to_tag = {value: key for key, value in tag_to_ix.items()}

    input_data = process_input(data, tokenizer=tokenizer)
    # Check input lengths
    for idx, (sentence, tags) in enumerate(input_data):
        if len(sentence) != len(tags):
            print('Warning: Size of sentence and tags didn\'t match')
            print('For sample: %s' % str(data[idx][0]))
            print('Lengths: %s' % str((len(sentence), len(tags))))
            return

    model = BiLSTM_CRF(tag_to_ix)

    scheduler = None
    # weight_decay = 1e-4 by default for SGD
    if optimizer == 'adam':
        weight_decay = weight_decay or 0
        model_optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay,
            amsgrad=True)
    else:
        weight_decay = weight_decay or 1e-5
        model_optimizer = optim.SGD(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.LambdaLR(
                        model_optimizer,
                        lr_lambda=lambda step: lr_schedule_slanted_triangular(step, n_iters, learning_rate)
                    )

    LOSS_LOG_FILE = path.join(LOG_DIR, 'neg_loss')
    INST_LOG_DIR = path.join(LOG_DIR, get_datetime_hostname())
    writer = SummaryWriter(log_dir=INST_LOG_DIR)

    loss_total = 0
    print_loss_total = 0
    all_losses = []

    if verbose == 2:
        iterator = trange(1, n_iters + 1, desc='Epochs', leave=False)
        epoch_iterator = tqdm(input_data)
    else:
        iterator = range(1, n_iters + 1)
        epoch_iterator = input_data
    
    # For timing with verbose=1

    start = time.time()
    layers_count = model.layers_count()

    for epoch in iterator:
        if gradual_unfreeze and epoch < layers_count:
            model.freeze_to(layers_count - epoch)

        for sentence, tags in epoch_iterator:
            loss = _train(sentence, tags, tag_to_ix, model, model_optimizer, grad_clip)
            loss_total += loss
            print_loss_total += loss

        if optimizer == 'sgd' and scheduler is not None:
            scheduler.step()
        
        all_losses.append(loss_total)
        writer.add_scalar(LOSS_LOG_FILE, loss_total, epoch)
        loss_total = 0

        if epoch % log_every == 0 and verbose != 0:
            with torch.no_grad():
                accuracy = evaluate_all(model, data, tag_to_ix, tokenizer)

                _, tag_seq = model(input_data[0][0])
                tag_interpreted = [ix_to_tag[tag] for tag in tag_seq]
                writer.add_text(
                    'Training predictions',
                    (' - Input: `%s`\r\n - Tags: `%s`\r\n - Predicted: `%s`\r\n\r\nAccuracy: %s\r\n' %
                        (str(input_data[0][0]),
                         str(input_data[0][1]),
                         str(tag_interpreted),
                         accuracy)),
                    epoch)

                if verbose == 1:
                    print_loss_avg = print_loss_total / log_every
                    progress = float(epoch) / float(n_iters)
                    print('%s (%d %d%%) %.4f' % (timeSince(start, progress),
                          epoch,
                          progress * 100,
                          print_loss_avg))

                print_loss_total = 0

        if len(all_losses) > patience > 0 and all_losses[-1] > all_losses[-patience]:
            print('Early stopping')
            break

    torch.save({
        'tag_to_ix': tag_to_ix,
        'state_dict': model.state_dict()
    }, save_path)

    LOG_JSON = path.join(LOG_DIR, 'all_scalars.json')
    writer.export_scalars_to_json(LOG_JSON)
    writer.close()

    return all_losses, model


def evaluate_all(model, data, tag_to_ix, tokenizer=wordpunct_space_tokenize):
    correct = 0
    total = 0
    input_data = process_input(data, tokenizer=tokenizer)
    with torch.no_grad():
        for idx, (sentence, tags) in enumerate(input_data):
            precheck_tags = [tag_to_ix[t] for t in tags]
            _, tag_seq = model(sentence)

            # Compare precheck_tags and tag_seq
            total += len(tag_seq)
            for idx, _ in enumerate(tag_seq):
                if tag_seq[idx] == precheck_tags[idx]:
                    correct += 1
    return float(correct) / float(total)
