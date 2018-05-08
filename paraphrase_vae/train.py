import torch
import csv
import torch.optim as optim
import numpy as np
import time
from os import path
from paraphrase_vae.model import ParaphraseVAE, KLDivLoss
from paraphrase_vae.tokenizer import build_vocab, segment_ngrams, SOS_token, EOS_token, SEPARATOR, UNK
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
    return question1[:20000], question2[:20000]


def to_idxs(tokens, vocab):
    return [vocab.get(token, vocab[UNK]) for token in tokens]


def process_input(input, target):
    print('Building vocabulary...')
    vocab = build_vocab(input + target)

    print('Segmenting input into bigrams...')
    input_tokens, in_vocab = segment_ngrams(input, vocab)
    vocab = vocab + in_vocab
    vocab = list(sorted(vocab, key=lambda x: x[1], reverse=True)[:MAX_NUM_WORDS])

    print('Segmenting target into bigrams...')
    target_tokens, out_vocab = segment_ngrams(target, vocab)
    vocab = vocab + out_vocab
    vocab = list(sorted(vocab, key=lambda x: x[1], reverse=True)[:MAX_NUM_WORDS])

    vocab = [(UNK, 0), (START_TAG, 0), (STOP_TAG, 0), (SEPARATOR, 0)] + vocab

    vocab_size = len(vocab)
    print('Vocabulary contains %s tokens' % vocab_size)

    # Convert to a dictionary
    vocab = {vocab[idx][0]: idx for idx in range(len(vocab))}

    input_tokens =  [np.array([SOS_token] + to_idxs(tokens, vocab) + [EOS_token])
                     for tokens in input_tokens]
    target_tokens = [np.array([EOS_token] + to_idxs(tokens, vocab)[::-1] + [SOS_token])
                     for tokens in target_tokens]

    # Sort by lengths
    lengths = [len(input_tokens[idx]) + len(target_tokens[idx]) for idx in range(len(input_tokens))]
    idxs = np.argsort(lengths)

    print('Total input tokens: %s' % np.sum(lengths))

    input_tokens = np.array(input_tokens)[idxs]
    target_tokens = np.array(target_tokens)[idxs]

    return input_tokens, target_tokens, vocab, vocab_size


def _train(s1_batch, s2_batch, model, optimizer, step):
    model.zero_grad()

    decoded, mean, logv = model(s1_batch, s2_batch, teacher_forcing_ratio=0.5)

    NLL_loss, KL_loss, KL_weight = KLDivLoss(decoded, s2_batch, mean, logv, step)

    loss = NLL_loss + KL_weight * KL_loss

    loss.backward()
    optimizer.step()

    return decoded, NLL_loss.item(), loss.item()


def trainIters(n_iters=10,
               lr=1e-3,
               lr_decay=0.99,
               lr_shrink=5,
               min_lr=1e-5,
               checkpoint=None):
    s1, s2 = get_quora(QUORA_PATH)
    s1, s2, vocab, vocab_size = process_input(s1, s2)

    model = ParaphraseVAE(vocab_size)

    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    # optimizer = optim.RMSprop(nli_net.parameters())
    # optimizer = optim.SGD(nli_net.parameters(), lr=lr)
    epoch_start = 1

    if checkpoint is not None and checkpoint != '':
        checkpoint_data = torch.load(checkpoint)
        vocab = checkpoint['vocab']
        vocab_size = checkpoint['vocab_size']

        model = ParaphraseVAE(vocab_size)
        model.load_state_dict(checkpoint_data['model_state'])

        optimizer.load_state_dict(checkpoint_data['optimizer_state'])
        epoch_start = checkpoint['epoch']

        print('Resuming from checkpoint %s (epoch %s - accuracy: %s)' % (
            checkpoint, checkpoint_data['epoch'], checkpoint_data['accuracy']))

    is_cuda = torch.cuda.is_available()

    LOSS_LOG_FILE = path.join(LOG_DIR, 'cross_entropy_loss')
    KDIV_LOG_FILE = path.join(LOG_DIR, 'kl_div_loss')
    INST_LOG_DIR = path.join(LOG_DIR, get_datetime_hostname())
    writer = SummaryWriter(log_dir=INST_LOG_DIR)

    if is_cuda:
        print('Training with GPU mode')
        model = model.cuda()
    else:
        print('Training with CPU mode')

    start_time = time.time()
    last_time = start_time
    step = 0

    ix_to_word = {value: key for key, value in vocab.items()}

    for epoch in range(epoch_start, n_iters + 1):

        losses = []
        NLL_losses = []
        batch_idx = 0.
        total_steps = len(s1)

        for start_idx in range(0, len(s1)):
            s1_batch = torch.LongTensor(s1[start_idx]).view(-1, 1)
            s2_batch = torch.LongTensor(s2[start_idx]).view(-1, 1)

            batch_idx += 1.

            if is_cuda:
                s1_batch = s1_batch.cuda()
                s2_batch = s2_batch.cuda()

            output, NLL_loss, loss = _train(s1_batch, s2_batch, model, optimizer, step)

            step += 1

            NLL_losses.append(NLL_loss)
            losses.append(loss)

            # log for every 100 batches:
            if len(losses) % 100 == 0:
                loss_total = np.mean(NLL_losses)
                kl_loss_total = np.mean(losses)

                writer.add_scalar(LOSS_LOG_FILE, loss_total, step)
                writer.add_scalar(KDIV_LOG_FILE, kl_loss_total, step)

                orig_sent = ' '.join([ix_to_word[ix] for ix in s1[start_idx]])
                ref_sent = ' '.join([ix_to_word[ix] for ix in s2[start_idx]][::-1])
                pred_sent = ' '.join([ix_to_word[ix] for ix in model.generate(s1_batch)])

                writer.add_text('Step: %s' % step,
                                ' - Source: `%s`\r\n - Reference: `%s`\r\n - Predicted: `%s`' % (orig_sent, ref_sent, pred_sent),
                                step)

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

                print('%s - epoch %s: loss: %s ; %s sentences/s (%s of epoch)' %
                      (asMinutes(time.time() - start_time),
                       epoch, loss_total,
                       round(100. / (time.time() - last_time), 2),
                       round(100. * batch_idx / total_steps, 2)))
                last_time = time.time()
                losses = []
                NLL_losses = []

        torch.save(model.state_dict(), path.join(SAVE_PATH, 'quora_vae_{}.bin'.format(epoch)))
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'vocab': vocab,
            'vocab_size': vocab_size
        }, path.join(SAVE_PATH, 'checkpoint_{}.bin'.format(epoch)))
        # Saving checkpoint

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

    return model, vocab, vocab_size
