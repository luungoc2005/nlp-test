import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from os import path
from config import NLI_PATH, EMBEDDING_DIM
from sent_to_vec.model import NLINet, BiGRUEncoder
from common.glove_utils import get_word_vector
from common.utils import get_datetime_hostname, asMinutes

np.random.seed(197)
torch.manual_seed(197)
torch.cuda.manual_seed(197)

BASE_PATH = path.dirname(__file__)
SAVE_PATH = path.join(BASE_PATH, 'model/')
LOG_DIR = path.join(BASE_PATH, 'logs/')

def get_nli(data_path):
    target_dict = {'entailment': 0,  'neutral': 1, 'contradiction': 2}
    s1 = [
        line.rstrip().lower() for line in
        open(path.join(data_path, 's1.train'), 'r')
    ]
    s2 = [
        line.rstrip().lower() for line in
        open(path.join(data_path, 's2.train'), 'r')
    ]
    target = np.array([
        target_dict[line.rstrip()] for line in
        open(path.join(data_path, 'labels.train'), 'r')
    ])

    assert len(s1) == len(s2) == len(target)

    return s1, s2, target

def process_batch(batch):
    lengths = np.array([len(sent) for sent in batch])
    max_len = np.max(lengths)
    embeds = np.zeros((max_len, len(batch), EMBEDDING_DIM))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            vec = get_word_vector(batch[i][j])
            if vec is None: embeds[j, i, :] = vec

    return torch.from_numpy(embeds).float(), lengths

def _train(s1_data, s2_data, target_batch, model, criterion, optimizer):
    optimizer.zero_grad()
    
    output = model(s1_data, s2_data)
    
    loss = criterion(output, target_batch)
    loss.backward()

    # Gradient clipping (to prevent exploding gradients)
    nn.utils.clip_grad_norm(model.parameters(), 5.)

    optimizer.step()

    return loss.cpu().data[0], output

def trainIters(n_iters=20, 
               batch_size=64,
               lr=0.1,
               lr_decay=0.99,
               lr_shrink=5,
               min_lr=1e-5):
    encoder = BiGRUEncoder()
    nli_net = NLINet(encoder=encoder)

    criterion = nn.CrossEntropyLoss()
    criterion.size_average = False

    # optimizer = optim.Adam(nli_net.parameters())
    # optimizer = optim.RMSprop(nli_net.parameters())
    optimizer = optim.SGD(nli_net.parameters(), lr=lr)

    s1, s2, target = get_nli(NLI_PATH)
    # permutation = np.random.permutation(len(s1))
    # s1 = s1[permutation]
    # s2 = s2[permutation]
    # target = target[permutation]

    is_cuda = torch.cuda.is_available()

    LOSS_LOG_FILE = path.join(LOG_DIR, 'cross_entropy_loss')
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
    train_acc = 0.
    accuracies = []

    for epoch in range(1, n_iters+1):

        correct = 0.
        losses = []
        batch_idx = 0.
        total_steps = len(s1) / batch_size

        for start_idx in range(0, len(s1), batch_size):
            s1_batch, s1_len = process_batch(s1[start_idx:start_idx + batch_size])
            s2_batch, s2_len = process_batch(s2[start_idx:start_idx + batch_size])
            target_batch = torch.LongTensor(target[start_idx:start_idx+batch_size])

            s1_batch, s2_batch, target_batch = Variable(s1_batch), Variable(s2_batch), Variable(target_batch)
            
            batch_idx += 1.
            k = s1_batch.size(1) # Actual batch size

            if is_cuda:
                s1_batch = s1_batch.cuda()
                s2_batch = s2_batch.cuda()
                target_batch = target_batch.cuda()

            loss, output = _train((s1_batch, s1_len), (s2_batch, s2_len), target_batch, nli_net, criterion, optimizer)

            pred = output.data.max(1)[1]
            correct += pred.long().eq(target_batch.data.long()).cpu().sum()
        
            losses.append(loss)
        
            # log for every 100 batches:
            if len(losses) % 100 == 0:
                loss_total = np.mean(losses)

                writer.add_scalar(LOSS_LOG_FILE, loss_total, epoch)

                print('%s - epoch %s: loss: %s ; %s sentences/s ; Accuracy: %s (%s of epoch)' % \
                    (asMinutes(time.time() - start_time), \
                    epoch, loss_total, \
                    round(batch_size * 100 / (time.time() - last_time), 2),
                    round(100. * correct / (start_idx + k), 2),
                    round(100. * batch_idx / total_steps, 2)))
                last_time = time.time()
                losses = []

        train_acc = round(100 * correct/len(s1), 2)
        accuracies.append(train_acc)

        torch.save(nli_net.state_dict(), path.join(SAVE_PATH, 'nli_model_{}_{}.bin'.format(epoch, train_acc)))
        torch.save(nli_net.encoder.state_dict(), path.join(SAVE_PATH, 'encoder_{}_{}.bin'.format(epoch, train_acc)))
        
        # Decaying LR
        if epoch>1:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * lr_decay

        if len(accuracies) > 5: # Minimum of 2 epochs:
            if accuracies[-1] < accuracies[-2] and accuracies[-2] < accuracies[-3]:
                # Early stopping
                # break
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / lr_shrink
                print('Accuracy deteriorated. Shrinking lr by %s - new lr: %s', (lr_shrink, optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < min_lr:
                    # Early stopping
                    break

    LOG_JSON = path.join(LOG_DIR, 'all_scalars.json')
    writer.export_scalars_to_json(LOG_JSON)
    writer.close()

    return nli_net