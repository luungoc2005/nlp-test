import json


def data_from_json(JSON_FILE):
    data = {}
    with open(JSON_FILE, errors='ignore') as json_file:
        json_data = json.load(json_file)
    for intent_object in json_data:
        if not data.get(intent_object['name'], None):
            data[intent_object['name']] = []
        for example in intent_object['usersays']:
            data[intent_object['name']].append(example.strip())
    return data


def get_data_pairs(test_data):
    X_train = []
    y_train = []
    for key in test_data.keys():
        for sent in test_data[key]:
            X_train.append(sent)
            y_train.append(key)
    return X_train, y_train
