import torch
from paraphrase_vae.model import ParaphraseVAE
from paraphrase_vae.tokenizer import UNK, SOS_token, EOS_token, vocab_tokenize

MODEL = None
VOCAB = None


def load_model(model_file=None):
    global MODEL, VOCAB

    if model_file is not None:
        data = torch.load(model_file)
        vocab = data['vocab']
        vocab_size = data['vocab_size']
        step = data.get('step', None)
        rnn_type = data.get('rnn_type', 'GRU')

        print('Model RNN type: %s' % rnn_type)

        model = ParaphraseVAE(vocab_size, rnn_type=rnn_type)
        model.load_state_dict(data['model_state'])

        MODEL = model
        VOCAB = vocab

        if step is not None:
            print('Finished loading checkpoint at step %s' % step)
        else:
            print('Finished loading model')

        return model, vocab
    else:
        return MODEL, VOCAB


def predict(input_sent, beam_width=10):
    model, vocab = load_model()

    ix_to_word = {value: key for key, value in vocab.items()}

    tokens = [vocab.get(token, vocab[UNK]) for token in vocab_tokenize(input_sent, vocab)]
    tokens = [SOS_token] + tokens + [EOS_token]
    tokens = torch.LongTensor(tokens).view(-1, 1)

    model.eval()
    decoded = model.generate(tokens, beam_width=beam_width)
    model.train()

    return ' '.join([ix_to_word[ix] for ix in decoded])

