import torch
import csv
import torch.optim as optim
import numpy as np
import time
from os import path
from paraphrase_vae.model import *
from paraphrase_vae.tokenizer import build_vocab, segment_ngrams, SEPARATOR, UNK
from config import BASE_PATH, START_TAG, STOP_TAG, QUORA_PATH, MAX_NUM_WORDS
from common.utils import get_datetime_hostname, asMinutes
from tensorboardX import SummaryWriter

np.random.seed(197)
torch.manual_seed(197)

if torch.cuda.is_available():
    torch.cuda.manual_seed(197)

# BASE_PATH = path.dirname(__file__)
SAVE_PATH = path.join(BASE_PATH, 'paraphrase_vae/model/')
LOG_DIR = path.join(BASE_PATH, 'paraphrase_vae/logs/')


def get_quora(data_path):
    question1 = []
    question2 = []
    with open(data_path, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            if row['is_duplicate']:
                question1.append(row['question1'])
                question2.append(row['question2'])
    print('Duplicate question pairs: %d' % len(question1))
    return question1, question2


def to_idxs(tokens, vocab):
    return [vocab.get(token, vocab[UNK]) for token in tokens]


def process_input(input, target):
    vocab = build_vocab(input)

    input_tokens, in_vocab = segment_ngrams(input, vocab)
    vocab = vocab + in_vocab
    vocab = list(sorted(vocab, key=lambda x: x[1], reverse=True)[:MAX_NUM_WORDS])

    target_tokens, out_vocab = segment_ngrams(target, vocab)
    vocab = vocab + out_vocab
    vocab = list(sorted(vocab, key=lambda x: x[1], reverse=True)[:MAX_NUM_WORDS])

    vocab = [(START_TAG, 0), (STOP_TAG, 0), (SEPARATOR, 0), (UNK, 0)] + vocab

    # Convert to a dictionary
    vocab = {word: idx for idx, (word, freq) in enumerate(vocab)}

    input_tokens = [[START_TAG] + to_idxs(tokens) + [STOP_TAG] for tokens in input_tokens]
    target_tokens = [[STOP_TAG] + to_idxs(tokens[::-1]) + [START_TAG] for tokens in target_tokens]

    return input_tokens, target_tokens, vocab


def trainIters(n_iters=10,
               batch_size=64,
               lr=1e-3,
               lr_decay=0.99,
               lr_shrink=5,
               min_lr=1e-5,
               checkpoint=None):
    s1, s2 = get_quora(QUORA_PATH)
    s1, s2, vocab = process_input(s1, s2)

    model = ParaphraseVAE(vocab_size=)

    criterion = nn.CrossEntropyLoss()
    criterion.size_average = False

    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    # optimizer = optim.RMSprop(nli_net.parameters())
    # optimizer = optim.SGD(nli_net.parameters(), lr=lr)
    epoch_start = 1

    if checkpoint is not None and checkpoint != '':
        checkpoint_data = torch.load(checkpoint)
        model.load_state_dict(checkpoint_data['model_state'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state'])
        epoch_start = checkpoint['epoch']
        print('Resuming from checkpoint %s (epoch %s - accuracy: %s)' % (
            checkpoint, checkpoint_data['epoch'], checkpoint_data['accuracy']))

    is_cuda = torch.cuda.is_available()

    LOSS_LOG_FILE = path.join(LOG_DIR, 'cross_entropy_loss')
    INST_LOG_DIR = path.join(LOG_DIR, get_datetime_hostname())
    writer = SummaryWriter(log_dir=INST_LOG_DIR)

    if is_cuda:
        print('Training with GPU mode')
        nli_net = model.cuda()
        criterion = criterion.cuda()
    else:
        print('Training with CPU mode')

    start_time = time.time()
    last_time = start_time
    train_acc = 0.
    accuracies = []

    for epoch in range(epoch_start, n_iters + 1):

        correct = 0.
        losses = []
        batch_idx = 0.
        total_steps = len(s1) / batch_size

        for start_idx in range(0, len(s1), batch_size):
            s1_batch, s1_len = process_batch(s1[start_idx:start_idx + batch_size])
            s2_batch, s2_len = process_batch(s2[start_idx:start_idx + batch_size])
            target_batch = torch.LongTensor(target[start_idx:start_idx + batch_size])

            s1_batch, s2_batch, target_batch = Variable(s1_batch), Variable(s2_batch), Variable(target_batch)

            batch_idx += 1.
            k = s1_batch.size(1)  # Actual batch size

            if is_cuda:
                s1_batch = s1_batch.cuda()
                s2_batch = s2_batch.cuda()
                target_batch = target_batch.cuda()

            loss, output = _train((s1_batch, s1_len), (s2_batch, s2_len), target_batch, nli_net, criterion, optimizer)

            pred = output.data.max(1)[1]
            correct += pred.long().eq(target_batch.data.long()).cpu().sum().item()

            losses.append(loss)

            # log for every 100 batches:
            if len(losses) % 100 == 0:
                loss_total = np.mean(losses)

                writer.add_scalar(LOSS_LOG_FILE, loss_total, epoch)

                # metrics for floydhub
                # print(json.dumps({
                #     'metric': 'sentence/s', 
                #     'value': batch_size * 100 / (time.time() - last_time)
                # }))
                # print(json.dumps({
                #     'metric': 'accuracy', 
                #     'value': 100. * correct / (start_idx + k)
                # }))
                # print(json.dumps({
                #     'metric': 'loss', 
                #     'value': loss_total
                # }))

                print('%s - epoch %s: loss: %s ; %s sentences/s ; Accuracy: %s (%s of epoch)' % \
                      (asMinutes(time.time() - start_time), \
                       epoch, loss_total, \
                       round(batch_size * 100 / (time.time() - last_time), 2),
                       round(100. * correct / (start_idx + k), 2),
                       round(100. * batch_idx / total_steps, 2)))
                last_time = time.time()
                losses = []

        train_acc = round(100 * correct / len(s1), 2)
        accuracies.append(train_acc)

        torch.save(nli_net.encoder.state_dict(), path.join(SAVE_PATH, 'encoder_{}_{}.bin'.format(epoch, train_acc)))
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'accuracy': train_acc
        }, path.join(SAVE_PATH, 'checkpoint_{}_{}.bin'.format(epoch, train_acc)))
        # Saving checkpoing

        # Decaying LR
        # if epoch>1:
        #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * lr_decay

        # if len(accuracies) > 5: # Minimum of 2 epochs:
        # if accuracies[-1] < accuracies[-2] and accuracies[-2] < accuracies[-3]:
        # Early stopping
        # break
        # optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / lr_shrink
        # print('Accuracy deteriorated. Shrinking lr by %s - new lr: %s', (lr_shrink, optimizer.param_groups[0]['lr']))
        # if optimizer.param_groups[0]['lr'] < min_lr:
        # Early stopping
        # break

    LOG_JSON = path.join(LOG_DIR, 'all_scalars.json')
    writer.export_scalars_to_json(LOG_JSON)
    writer.close()

    return encoder, nli_net
