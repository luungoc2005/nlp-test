from nltk.tokenize import wordpunct_tokenize
from collections import Counter
from config import MAX_NUM_WORDS, WORDS_SHORTLIST

NGRAM_SIZE = 2
SEPARATOR = '@@'
UNK = '<UNK>'
UNK_token = 0
SOS_token = 1
EOS_token = 2
SEP_token = 3


def build_vocab(sents):
    c = Counter()
    for sent in sents:
        for word in wordpunct_tokenize(sent.lower()):
            c[word] += 1
    return list(sorted(c.items(), key=lambda x: x[1], reverse=True)[:WORDS_SHORTLIST])


def segment_ngrams(sents, vocab):
    c = Counter()

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
                    tokens.append(token)

                    if token not in vocab:
                        c[token] += 1

                    i += 1
                    if i * NGRAM_SIZE < len(word):
                        tokens.append(SEPARATOR)

            else:
                tokens.append(word)
        result.append(tokens)

    tokens_vocab = list(sorted(c.items(), key=lambda x: x[1], reverse=True)[:MAX_NUM_WORDS])
    return result, tokens_vocab
