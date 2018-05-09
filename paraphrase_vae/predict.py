import torch
from nltk.tokenize import wordpunct_tokenize
from paraphrase_vae.model import ParaphraseVAE
from paraphrase_vae.tokenizer import NGRAM_SIZE, SEPARATOR, UNK

MODEL = None
VOCAB = None


def load_model(model_file=None):
    global MODEL, VOCAB

    if model_file is None:
        data = torch.load(model_file)
        vocab = data['vocab']
        vocab_size = data['vocab_size']

        model = ParaphraseVAE(vocab_size)
        model.load_state_dict(data['model_state'])

        MODEL = model
        VOCAB = vocab

        return model, vocab
    else:
        return MODEL, VOCAB


def predict(input_sent):
    model, vocab = load_model()

    ix_to_word = {value: key for key, value in vocab.items()}

    tokens = [vocab.get(token, vocab[UNK]) for token in vocab_tokenize(input_sent)]
    tokens = torch.LongTensor(tokens).view(-1, 1)

    decoded = model.generate(tokens, beam_width=10)

    return ' '.join([ix_to_word[ix] for ix in decoded])


def vocab_tokenize(sents, vocab):
    if not isinstance(vocab, dict):
        vocab = {word: idx + 1 for idx, (word, freq) in enumerate(vocab)}

    result = []
    for sent in sents:
        tokens = []
        for word in wordpunct_tokenize(sent.lower()):
            if word not in vocab:
                i = 0
                while i * NGRAM_SIZE < len(word):
                    token = word[i * NGRAM_SIZE:i * NGRAM_SIZE + NGRAM_SIZE]

                    if token in vocab:
                        tokens.append(token)
                    else:
                        tokens.append(UNK)

                    i += 1
                    if i * NGRAM_SIZE < len(word):
                        tokens.append(SEPARATOR)

            else:
                tokens.append(word)
        result.append(tokens)

    return result
