import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from os import path
from config import BASE_PATH
from common.langs.vi_VN.tokenizer.model import BiLSTMTagger
from common.langs.vi_VN.tokenizer import data_utils
from common.langs.vi_VN.utils import remove_tone_marks, random_remove_marks
from common.keras_preprocessing import Tokenizer
from common.utils import get_datetime_hostname, asMinutes
from tensorboardX import SummaryWriter

np.random.seed(197)
torch.manual_seed(197)

if torch.cuda.is_available():
    torch.cuda.manual_seed(197)

# BASE_PATH = path.dirname(__file__)
SAVE_PATH = path.join(BASE_PATH, 'common/langs/vi_VN/tokenizer/model/')
LOG_DIR = path.join(BASE_PATH, 'common/langs/vi_VN/tokenizer/logs/')

def _train(sent, target, model, criterion, optimizer):
    optimizer.zero_grad()

    output = model(sent)

    loss = criterion(output, target)
    loss.backward()

    # Gradient clipping (to prevent exploding gradients)
    nn.utils.clip_grad_norm_(model.parameters(), 5.)

    optimizer.step()

    return loss.cpu().item(), output


def trainIters(n_iters=10,
               lr=1e-3,
               lr_decay=1e-5,
               lr_shrink=5,
               min_lr=1e-5,
               checkpoint=None,
               cuda=None):
    # Loading data
    print('Loading VN Treebank dataset')
    data_sents, data_tags = data_utils.load_treebank_dataset()
    assert len(data_sents) == len(data_tags)
    print('Loaded %s sentences' % len(data_sents))

    epoch_start = 0
    checkpoint_start = 0

    criterion = nn.BCEWithLogitsLoss()

    if checkpoint is not None and checkpoint != '':
        checkpoint_data = torch.load(checkpoint)
        tokenizer = checkpoint_data['tokenizer']
        
        model = BiLSTMTagger(max_emb_words=10000, tokenizer=tokenizer)
        model.load_state_dict(checkpoint_data['model_state'])

        optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint_data['optimizer_state'])

        epoch_start = checkpoint_data.get('epoch', 0)
        # scheduler.last_epoch = epoch_start
        checkpoint_start = checkpoint_data.get('batch_number', 0)
        print('Resuming from checkpoint %s (epoch %s - accuracy: %s)' %
              (checkpoint, checkpoint_data['epoch'], checkpoint_data['accuracy']))
    else:
        print('Preprocessing...')
        data_sents_no_marks = [
            [remove_tone_marks(token) for token in sent]
            for sent in data_sents
        ]
        tokenizer = Tokenizer(oov_token=1, num_words=10000)
        tokenizer.fit_on_texts(data_sents + data_sents_no_marks)
        print('Preprocessing completed')

        model = BiLSTMTagger(max_emb_words=10000, tokenizer=tokenizer)
        optimizer = optim.Adam(model.parameters(), lr=lr)

    is_cuda = cuda if cuda is not None else torch.cuda.is_available()

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

    for epoch in range(epoch_start, n_iters + 1):

        correct = 0.
        total = 0.
        losses = []
        total_steps = len(data_sents)

        for start_idx in range(checkpoint_start, len(data_sents)):
            sentence = data_sents[start_idx]
            target = data_tags[start_idx]

            # normalizing for tone marks removed tokens
            sentence = [random_remove_marks(token) for token in sentence]
            target = torch.FloatTensor(target)
            
            if is_cuda:
                target = target.cuda()

            loss, output = _train(sentence, target, model, criterion, optimizer)

            pred = F.sigmoid(output).data >= 0.5
            correct += pred.eq(target.data >= 0.5).cpu().sum().item()
            total += output.numel()

            losses.append(loss)

            # log for every 100 minibatches:
            if len(losses) % 100 == 0:
                loss_total = np.mean(losses)
                accuracy_current = round(100. * correct / total, 2)

                writer.add_scalar(LOSS_LOG_FILE, loss_total, epoch)
                writer.add_scalar(ACC_LOG_FILE, accuracy_current, epoch)

                print('%s - epoch %s: loss: %s ; %s sentences/s ; Accuracy: %s (%s of epoch)' %
                      (asMinutes(time.time() - start_time),
                       epoch, loss_total,
                       round(100 / (time.time() - last_time), 2),
                       accuracy_current,
                       round(100. * start_idx / total_steps, 2)))

                last_time = time.time()
                losses = []

            # checkpoint every 5000 minibatches
            if len(losses) % 5000 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'accuracy': train_acc,
                    'batch_number': start_idx,
                    'tokenizer': tokenizer
                }, path.join(SAVE_PATH, 'checkpoint_{}_{}.bin'.format(epoch, train_acc)))
                # Saving checkpoint

        train_acc = round(100 * correct / total, 2)
        accuracies.append(train_acc)
        # scheduler.step()

        checkpoint_start = 0

        with torch.no_grad():
            test_examples = [
                'Học sinh học sinh học'.split(' '),
                'Hoc sinh hoc sinh hoc'.split(' '),
                'Đoi tuyen U23 Việt Nam vo địch.'.split(' '),
                'Đội tuyển U23 Việt Nam vô địch.'.split(' '),
                'Hà Nội mùa này vắng những cơn mưa'.split(' '),
                'Trump - Putin trả lời câu hỏi hóc búa trong họp báo thế nào?'.split(' '),
                'Thợ lặn từng lo sợ 5 thiếu niên sẽ chết khi được cứu khỏi hang Thái Lan'.split(' ')
            ]
            log_text = ''
            for test_str in test_examples:
                test = model(test_str)
                log_text += 'Sanity test result: \nlogits: {}, sentence: {}\n\n'.format(
                    str(test),
                    str(tokenize(test_str, F.sigmoid(test).data >= 0.5))
                )
            writer.add_text(
                'Epoch ' + str(epoch),
                log_text
            )

    LOG_JSON = path.join(LOG_DIR, 'all_scalars.json')
    writer.export_scalars_to_json(LOG_JSON)
    writer.close()

    return model

def tokenize(sent, tags):
    tokens_arr = []
    running_word = []
    for idx, token in enumerate(sent):
        running_word.append(token)
        if tags[idx] == 0:
            tokens_arr.append('_'.join(running_word))
            running_word = []
    return tokens_arr