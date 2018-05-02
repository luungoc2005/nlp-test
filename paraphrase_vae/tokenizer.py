from nltk.tokenize import word_tokenize
from collections import Counter
from config import MAX_NUM_WORDS

NGRAM_SIZE = 2
SEPARATOR = '@@'

def build_vocab(sents):
    c = Counter()
    for sent in sents:
        for word in word_tokenize(sent):
            c[word] += 1
    return list(sorted(c.items(), key=lambda x: x[1], reverse=True))[:MAX_NUM_WORDS]

def segment_ngrams(sents, vocab):
    vocab = {word: idx for idx, (word, freq) in enumerate(vocab)}
    result = []
    for sent in sents:
        tokens = []
        for word in word_tokenize(sent):
            if word not in vocab:
                i = 0
                while i * NGRAM_SIZE < len(word):
                    tokens.append(word[i*NGRAM_SIZE:i*NGRAM_SIZE+NGRAM_SIZE])
                    i += 1
                    if i * NGRAM_SIZE < len(word):
                        tokens.append(SEPARATOR)

            else:
                tokens.append(word)
        result.append(tokens)
    return result