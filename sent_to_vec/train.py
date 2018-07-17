import json
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from os import path
from config import SNLI_PATH, MultiNLI_PATH, BASE_PATH
from sent_to_vec.model import NLINet, BiLSTMEncoder, ConvNetEncoder, QRNNEncoder, QRNNEncoderConcat, process_batch, process_input
from common.utils import get_datetime_hostname, asMinutes
from common.torch_utils import lr_schedule_slanted_triangular

np.random.seed(197)
torch.manual_seed(197)

if torch.cuda.is_available():
    torch.cuda.manual_seed(197)

# BASE_PATH = path.dirname(__file__)
SAVE_PATH = path.join(BASE_PATH, 'output/model/')
LOG_DIR = path.join(BASE_PATH, 'output/logs/')


def get_nli(data_path):
    target_dict = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    s1 = [
        line.rstrip().lower() for line in
        open(path.join(data_path, 's1.train'), 'r', encoding='utf-8')
    ]
    s2 = [
        line.rstrip().lower() for line in
        open(path.join(data_path, 's2.train'), 'r', encoding='utf-8')
    ]
    target = [
        target_dict[line.rstrip()] for line in
        open(path.join(data_path, 'labels.train'), 'r', encoding='utf-8')
    ]

    assert len(s1) == len(s2) == len(target)

    return s1, s2, target


def _train(s1_data, s2_data, target_batch, model, criterion, optimizer):
    optimizer.zero_grad()

    output = model(s1_data, s2_data)

    loss = criterion(output, target_batch)
    loss.backward()

    # Gradient clipping (to prevent exploding gradients)
    nn.utils.clip_grad_norm_(model.parameters(), 5.)

    optimizer.step()

    return loss.cpu().item(), output


def trainIters(n_iters=10,
               batch_size=64,
               lr=0.01,
               lr_decay=1e-5,
               lr_shrink=5,
               min_lr=1e-5,
               encoder_type='qrnn_mean',
               checkpoint=None):
    
    assert encoder_type in ['qrnn_mean', 'qrnn_max', 'qrnn_concat', 'bilstm', 'convnet']
    
    if encoder_type == 'qrnn_mean':
        encoder = QRNNEncoder(pool_type='mean')
    elif encoder_type == 'qrnn_max':
        encoder = QRNNEncoder(pool_type='max')
    elif encoder_type == 'qrnn_concat':
        encoder = QRNNEncoderConcat()
    elif encoder_type == 'bilstm':
        encoder = BiLSTMEncoder()
    elif encoder_type == 'convnet':
        encoder = ConvNetEncoder()
    
    # encoder = ConvNetEncoder()
    nli_net = NLINet(encoder=encoder, bidirectional_encoder=False)

    criterion = nn.CrossEntropyLoss()
    criterion.size_average = False

    # optimizer = optim.Adam(nli_net.parameters(), lr=lr)
    # optimizer = optim.RMSprop(nli_net.parameters())
    optimizer = optim.SGD(nli_net.parameters(), lr=lr, weight_decay=lr_decay)

    # scheduler = optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda step: lr_schedule_slanted_triangular(step, n_iters, lr)
    # )
    epoch_start = 0
    checkpoint_start = 0

    if checkpoint is not None and checkpoint != '':
        checkpoint_data = torch.load(checkpoint)
        nli_net.load_state_dict(checkpoint_data['nli_state'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state'])
        epoch_start = checkpoint_data.get('epoch', 0)
        # scheduler.last_epoch = epoch_start
        checkpoint_start = checkpoint_data.get('batch_number', 0)
        print('Resuming from checkpoint %s (epoch %s - accuracy: %s)' %
              (checkpoint, checkpoint_data['epoch'], checkpoint_data['accuracy']))

    s1, s2, target = get_nli(SNLI_PATH)
    m_s1, m_s2, m_target = get_nli(MultiNLI_PATH)
    s1 = s1 + m_s1
    s2 = s2 + m_s2
    target = np.array(target + m_target)

    print('Preprocessing inputs (%s sentence pairs)' % len(s1))
    s1 = process_input(s1)
    s2 = process_input(s2)
    print('Preprocessing completed')
    # permutation = np.random.permutation(len(s1))
    # s1 = s1[permutation]
    # s2 = s2[permutation]
    # target = target[permutation]

    is_cuda = torch.cuda.is_available()

    LOSS_LOG_FILE = path.join(LOG_DIR, 'cross_entropy_loss')
    ACC_LOG_FILE = path.join(LOG_DIR, 'train_accuracy')
    INST_LOG_DIR = path.join(LOG_DIR, get_datetime_hostname())
    writer = SummaryWriter(log_dir=INST_LOG_DIR)

    if is_cuda:
        print('Training with GPU mode')
        encoder = encoder.cuda()
        nli_net = nli_net.cuda()
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
