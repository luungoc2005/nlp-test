import csv
import random
from os import path
from config import BASE_PATH

TRAIN_PATH = path.join(BASE_PATH, 'data/kore/ML_Train.csv')
TEST_PATH = path.join(BASE_PATH, 'data/kore/ML_Test.csv')

TRAIN_SET = {}
TEST_SET = {}

count = 0
print('Reading training data')
with open(TRAIN_PATH, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        count += 1
        intent_name = row[0].lower()
        if intent_name not in TRAIN_SET:
            TRAIN_SET[intent_name] = []
        TRAIN_SET[intent_name].append(row[1])

print('Training set: %s intents, %s examples' % (str(len(TRAIN_SET.keys())), str(count)))
print('Random sentence: %s' % random.choice(random.choice(list(TRAIN_SET.values()))))
print(TRAIN_SET.keys())

count = 0
with open(TEST_PATH, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        count += 1
        intent_name = row[0].lower()
        if intent_name not in TEST_SET:
            TEST_SET[intent_name] = []
        TEST_SET[intent_name].append(row[1])

print('Testing set: %s intents, %s examples' % (str(len(TEST_SET.keys())), str(count)))
print('Random sentence: %s' % random.choice(random.choice(list(TEST_SET.values()))))
print(TEST_SET.keys())

def get_data(dataset):
    _X = []
    _y = []
    for key in dataset:
        for value in dataset[key]:
            _X.append(value)
            _y.append(key)
    return _X, _y

X_train, y_train = get_data(TRAIN_SET)
X_test, y_test = get_data(TEST_SET)

from text_classification.ensemble.model import EnsembleWrapper
from text_classification.ensemble.train import EnsembleLearner
model = EnsembleWrapper()
learner = EnsembleLearner(model)
learner.fit(training_data=(X_train, y_train))
# from text_classification.fast_text.model import FastTextWrapper
# from text_classification.fast_text.train import FastTextLearner
# from common.callbacks import PrintLoggerCallback, EarlyStoppingCallback
# model = FastTextWrapper()
# learner = FastTextLearner(model)

# learner.fit(
#     training_data=(X_train, y_train),
#     epochs=500,
#     callbacks=[
#         PrintLoggerCallback(),
#         EarlyStoppingCallback(patience=20)
#     ]
# )

test_result = model(X_test)
top1_intents = [
    item[0]['intent'] if item[0]['confidence'] > .3 else 'none'
    for item in test_result
]

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
le = LabelEncoder()
le.fit(y_test)

y_test = le.transform(y_test)
top1_intents = le.transform(top1_intents)

print('Accuracy: %s' % accuracy_score(y_test, top1_intents))
print('Recall: %s' % recall_score(y_test, top1_intents, average='macro'))
print('Precision: %s' % precision_score(y_test, top1_intents, average='macro'))
print('F1: %s' % f1_score(y_test, top1_intents, average='macro'))

