import torch
# import numpy as np

from os import path

from text_classification.fast_text.train import trainIters
from text_classification.fast_text.train import evaluate_all
from config import BASE_PATH

AMAZON_TRAIN_PATH = path.join(BASE_PATH, 'data/amazon/train.ft.txt')
AMAZON_TEST_PATH = path.join(BASE_PATH, 'data/amazon/test.ft.txt')


def read_input_file(filename):
    classes = []
    texts = []
    with open(filename, 'r') as train_file:
        for line in train_file:
            line_class = line.split()[0][-1]
            line_text = ' '.join(line.split()[1:])
#             line_title = line_text.split(': ')[0]
            line_content = line_text.split(': ')[1]
            classes.append(line_class)
            texts.append(line_content)
    print('Read %s samples from %s' % (len(texts), filename))
    return texts, classes


(x_train, y_train) = read_input_file(AMAZON_TRAIN_PATH)
(x_test, y_test) = read_input_file(AMAZON_TEST_PATH)

classes = list(set(y_train))
training_data = [(item, classes.index(y_train[idx])) for idx, item in enumerate(x_train)]
print('Finished loading training dataset')

losses, model = trainIters(training_data, 
                           classes, 
                           n_iters=5, 
                           log_every=1, 
                           verbose=2,
                           learning_rate=1e-3, 
                           batch_size=64)

print('Saving model...')
torch.save(model.state_dict(), 'amazon_ft.bin')

test_data = [(item, classes.index(y_test[idx])) for idx, item in enumerate(x_test)]

print('Model evaluation result - Test set:')
print(evaluate_all(model, test_data))
