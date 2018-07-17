import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from os import path
from config import BASE_PATH
from common.langs.vi_VN.tokenizer.model import BiLSTMTagger
from common.keras_preprocessing import Tokenizer
from common.utils import get_datetime_hostname
from tensorboardX import SummaryWriter


np.random.seed(197)
torch.manual_seed(197)

if torch.cuda.is_available():
    torch.cuda.manual_seed(197)

# BASE_PATH = path.dirname(__file__)
SAVE_PATH = path.join(BASE_PATH, 'common/langs/vi_VN/tokenizer/model/')
LOG_DIR = path.join(BASE_PATH, 'common/langs/vi_VN/tokenizer/logs/')

def _train(sent, target, model, optimizer, criterion):
    optimizer.zero_grad()

    output = model(sent)

    loss = criterion(output, target)
    loss.backward()

    # Gradient clipping (to prevent exploding gradients)
    nn.utils.clip_grad_norm_(model.parameters(), 5.)

    optimizer.step()

    return loss.cpu().item(), output


def trainIters(n_iters=10,
               batch_size=64,
               lr=1e-3,
               lr_decay=1e-5,
               lr_shrink=5,
               min_lr=1e-5,
               checkpoint=None):
    # Loading data


    tokenizer = Tokenizer(oov_token=0)

    model = BiLSTMTagger(max_emb_words=10000, tokenizer=tokenizer)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    epoch_start = 1

    epoch_start = 0
    checkpoint_start = 0

    if checkpoint is not None and checkpoint != '':
        checkpoint_data = torch.load(checkpoint)
        model.load_state_dict(checkpoint_data['model_state'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state'])
        epoch_start = checkpoint_data.get('epoch', 0)
        # scheduler.last_epoch = epoch_start
        checkpoint_start = checkpoint_data.get('batch_number', 0)
        print('Resuming from checkpoint %s (epoch %s - accuracy: %s)' %
              (checkpoint, checkpoint_data['epoch'], checkpoint_data['accuracy']))

    is_cuda = torch.cuda.is_available()

    LOSS_LOG_FILE = path.join(LOG_DIR, 'bce_loss')
    ACC_LOG_FILE = path.join(LOG_DIR, 'train_accuracy')
    INST_LOG_DIR = path.join(LOG_DIR, get_datetime_hostname())
    writer = SummaryWriter(log_dir=INST_LOG_DIR)

    if is_cuda:
        print('Training with GPU mode')
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        print('Training with CPU mode')

    start_time = time.time()
    last_time = start_time
    accuracies = []
    train_acc = 0
    train_best_acc = 0

    for epoch in range(epoch_start, n_iters + 1):

        correct = 0.
        losses = []
        batch_idx = 0.
        total_steps = len(s1) / batch_size

        for start_idx in range(checkpoint_start, len(s1), batch_size):
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

            # log for every 100 minibatches:
            if len(losses) % 100 == 0:
                loss_total = np.mean(losses)
                accuracy_current = round(100. * correct / (start_idx + k), 2)

                writer.add_scalar(LOSS_LOG_FILE, loss_total, epoch)
                writer.add_scalar(ACC_LOG_FILE, accuracy_current, epoch)

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

                print('%s - epoch %s: loss: %s ; %s sentences/s ; Accuracy: %s (%s of epoch)' %
                      (asMinutes(time.time() - start_time),
                       epoch, loss_total,
                       round(batch_size * 100 / (time.time() - last_time), 2),
                       accuracy_current,
                       round(100. * batch_idx / total_steps, 2)))
                last_time = time.time()
                losses = []

            # checkpoint every 5000 minibatches
            if len(losses) % 5000 == 0:
                torch.save(nli_net.encoder.state_dict(),
                            path.join(SAVE_PATH, 'encoder_{}_{}.bin'.format(epoch, train_acc)))
                torch.save({
                    'epoch': epoch,
                    'nli_state': nli_net.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'accuracy': train_acc,
                    'batch_number': start_idx
                }, path.join(SAVE_PATH, 'checkpoint_{}_{}.bin'.format(epoch, train_acc)))
                # Saving checkpoint

        train_acc = round(100 * correct / len(s1), 2)
        accuracies.append(train_acc)
        # scheduler.step()

        checkpoint_start = 0

        # Decaying LR
        if train_acc > train_best_acc:
            train_best_acc = train_acc
        else:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / lr_shrink
            print('Accuracy deteriorated. Shrinking lr by %s - new lr: %s' % (lr_shrink, optimizer.param_groups[0]['lr']))
        if optimizer.param_groups[0]['lr'] < min_lr:
            print('Early stopping')
            # Early stopping
            break

    LOG_JSON = path.join(LOG_DIR, 'all_scalars.json')
    writer.export_scalars_to_json(LOG_JSON)
    writer.close()

    return encoder, nli_net
