import torch
import torch.optim as optim

from tqdm import tqdm, trange
from config import START_TAG, STOP_TAG

from tensorboardX import SummaryWriter
from os import path, getcwd

from bilstm.model import BiLSTM_CRF
from bilstm.utils import get_datetime_hostname, prepare_vec_sequence, process_input, word_to_vec
import time
import math

# Helper functions for time remaining
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    if percent != 0:
        es = s / (percent)
        rs = es - s
    else:
        rs = 0
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

BASE_PATH = path.join(getcwd(), 'bilstm/')
SAVE_PATH = path.join(BASE_PATH, 'model/model.bin')
LOG_DIR = path.join(BASE_PATH, 'logs/')

torch.manual_seed(7)

def _train(input_variable, target_variable, tag_to_ix, model, optimizer):
    model.zero_grad()

    # Prepare training data
    sentence_in = prepare_vec_sequence(input_variable, word_to_vec)
    targets = torch.LongTensor([tag_to_ix[t] for t in target_variable])

    # Run the forward pass.
    neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)

    # Compute the loss, gradients, and update the parameters by
    # calling optimizer.step()
    neg_log_likelihood.backward()
    optimizer.step()

    return neg_log_likelihood.data[0]

def trainIters(data, 
               tag_to_ix,
               n_iters=50,
               log_every=10,
               optimizer='adam',
               learning_rate=0.01,
               weight_decay=None,
               verbose=2):
    # Invert the tag dictionary
    ix_to_tag = {value: key for key, value in tag_to_ix.items()}

    input_data = process_input(data)
    # Check input lengths
    for idx, (sentence, tags) in enumerate(input_data):
        if len(sentence) != len(tags):
            print('Warning: Size of sentence and tags didn\'t match')
            print('For sample: %s' % str(data[idx][0]))
            print('Lengths: %s' % str((len(sentence), len(tags))))
            return

    model = BiLSTM_CRF(tag_to_ix)

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
        for sentence, tags in epoch_iterator:
            loss = _train(sentence, tags, tag_to_ix, model, model_optimizer)
            loss_total += loss
            print_loss_total += loss
        
        writer.add_scalar(LOSS_LOG_FILE, loss_total, epoch)
        loss_total = 0

        if epoch % log_every == 0:
            accuracy = evaluate(model, data, tag_to_ix)

            precheck_sent = prepare_vec_sequence(input_data[0][0], word_to_vec)
            _, tag_seq = model(precheck_sent)
            tag_interpreted = [ix_to_tag[tag] for tag in tag_seq]
            writer.add_text(
                'Training predictions',
                (' - Input: `%s`\r\n - Tags: `%s`\r\n - Predicted: `%s`\r\n\r\nAccuracy: %s\r\n' % \
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

    torch.save(model.state_dict(), SAVE_PATH)

    LOG_JSON = path.join(LOG_DIR, 'all_scalars.json')
    writer.export_scalars_to_json(LOG_JSON)
    writer.close()

    return model

def evaluate(model, data, tag_to_ix):
    correct = 0
    total = 0
    input_data = process_input(data)
    for idx, (sentence, tags) in enumerate(input_data):
        precheck_sent = prepare_vec_sequence(sentence, word_to_vec)
        precheck_tags = [tag_to_ix[t] for t in tags]
        _, tag_seq = model(precheck_sent)

        # Compare precheck_tags and tag_seq
        total += len(tag_seq)
        for idx, _ in enumerate(tag_seq):
            if tag_seq[idx] == precheck_tags[idx]:
                correct += 1
    return float(correct) / float(total)