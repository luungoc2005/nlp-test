import torch
import torch.optim as optim

from tqdm import tqdm, trange
from fastText import FastText
from nltk import wordpunct_tokenize
from config import FASTTEXT_PATH, START_TAG, STOP_TAG

from .model import BiLSTM_CRF
from .utils import prepare_vec_sequence

training_data = [(
    'My email address is at luungoc2005@gmail.com',
    '- - - - - EMAIL EMAIL EMAIL EMAIL EMAIL'
), (
    'Contact me at contact@2359media.net',
    '- - - EMAIL EMAIL EMAIL EMAIL EMAIL'
), (
    'test.email@microsoft.com is a testing email address',
    'EMAIL EMAIL EMAIL EMAIL EMAIL EMAIL EMAIL - - - - -'
), (
    'Any inquiries email thesloth_197@gmail.com for assistance',
    '- - - EMAIL EMAIL EMAIL EMAIL EMAIL - -'
)]

test_data = [
    'Contact us: hello_vietnam@yahoo.com',
    'hello.sunbox@gmail.com - drop us an email here!~'
]

tag_to_ix = {
    '-': 0,
    'EMAIL': 1,
    START_TAG: 2,
    STOP_TAG: 3
}

WORD_EMBEDDINGS = {}
fastText_model = None

def word_to_vec(word):
    global fastText_model, WORD_EMBEDDINGS
    if not fastText_model:
        print('Loading fastText model...', end='', flush=True)
        fastText_model = FastText.load_model(FASTTEXT_PATH)
        print('Done.')
    
    if word not in WORD_EMBEDDINGS:
        WORD_EMBEDDINGS[word] = fastText_model.get_word_vector(word)
    return WORD_EMBEDDINGS[word]

def process_input(data):
    return [
        (wordpunct_tokenize(sent), tags.split())
        for (sent, tags) in data
    ]

def train(data, tag_to_ix, test_data = None):
    global fastText_model

    model = BiLSTM_CRF(tag_to_ix)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    input_data = process_input(data)

    # Check input lengths
    for idx, (sentence, tags) in enumerate(input_data):
        if len(sentence) != len(tags):
            print('Warning: Size of sentence and tags didn\'t match')
            print('For sample: %s' % data[idx])
            return

    # Check pre-training predictions
    print('Pre-training results')
    precheck_sent = prepare_vec_sequence(input_data[0][0], word_to_vec)
    precheck_tags = torch.LongTensor([tag_to_ix[t] for t in input_data[0][1]])
    print(model(precheck_sent))

    last_loss = 0

    for epoch in trange(25, desc='Epochs', leave=False):
        for sentence, tags in tqdm(input_data, desc=last_loss):
            model.zero_grad()

            # Prepare training data
            sentence_in = prepare_vec_sequence(sentence, word_to_vec)
            targets = torch.LongTensor([tag_to_ix[t] for t in tags])

            # Run the forward pass.
            neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)
            last_loss += neg_log_likelihood

            # Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            neg_log_likelihood.backward()
            optimizer.step()
        last_loss = 0

    # Check post-training predictions:
    print('')
    print('Post-training results:')
    precheck_sent = prepare_vec_sequence(input_data[0][0], word_to_vec)
    precheck_tags = torch.LongTensor([tag_to_ix[t] for t in input_data[0][1]])
    print(model(precheck_sent))

    # Test the model on test data:
    if test_data:
        for sentence in test_data:
            sequence_in = wordpunct_tokenize(sentence)
            sentence_in = prepare_vec_sequence(sequence_in, word_to_vec)
            score, tag_seq = model(sentence_in)
            email_out = ''
            for idx, tag in enumerate(tag_seq):
                if tag == tag_to_ix['EMAIL']:
                    email_out += sequence_in[idx]
            print('Test sample: %s' % sentence)
            print('Result: %s (%s, %s)' % email_out)

    del fastText_model
    return model