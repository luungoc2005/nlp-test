import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.sparse
from torch.utils.data import Dataset
from common.utils import to_categorical
from common.word_vectors import get_emb_matrix
from config import NGRAM_BINS, EMBEDDING_DIM
from fasttext_utils import _process_sentences
from sklearn.svm import SVC


def process_sentences(sentences):
    words, ngrams = _process_sentences(list(sentences))
    return torch.from_numpy(words).float(), \
           torch.from_numpy(ngrams).long()

class FastTextDataset(Dataset):
    def __init__(self, input_array, n_classes):
        self.input_array = input_array
        self.sentences = []
        self.labels = []
        self.n_classes = n_classes

        for sent, label in input_array:
            self.sentences.append(sent)
            self.labels.append(label)
        
        self.words, self.ngrams = _process_sentences(self.sentences)

    def __len__(self):
        return len(self.input_array)

    def __getitem__(self, idx):
        return {
            'sentence': (
                torch.from_numpy(self.words[idx]).float(), 
                torch.from_numpy(self.ngrams[idx]).long()
            ),
            'label': to_categorical(int(self.labels[idx]), self.n_classes).float()
        }

class FastText(nn.Module):

    def __init__(self,
                 hidden_size=100,
                 dropout_keep_prob=0.5,
                 classes=10):
        super(FastText, self).__init__()

        self.hidden_size = hidden_size
        self.dropout_keep_prob = dropout_keep_prob
        self.classes = classes

        self.ngrams_embs = nn.Embedding(NGRAM_BINS, EMBEDDING_DIM, sparse=True, padding_idx=0)
        self.ngrams_embs.weight.requires_grad = False

        self.i2h = nn.Linear(EMBEDDING_DIM, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, self.classes)

        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        self.init_weights()

    def init_weights(self):
        # nn.init.xavier_normal_(self.i2o.weight)
        nn.init.xavier_normal_(self.i2h.weight)
        nn.init.xavier_normal_(self.h2o.weight)

    def _temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def _calibrate(self, data_loader, weight=None):
        pass
        # all_logits = []
        # all_targets = []

        # with torch.no_grad():
        #     for _, data_batch in enumerate(data_loader, 0):
        #         sentences = data_batch['sentence']
        #         labels = data_batch['label']

        #         feats = self._get_hidden_features(*sentences)
        #         all_logits.append(self._forward_alg(feats))
        #         all_targets.append(labels)
        # logits_tensor = torch.cat(all_logits, dim=0)
        # targets_tensor = torch.cat(all_targets, dim=0)

        # # Find temperature (probability calibration)
        # optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        # if weight is None:
        #     criterion = nn.CrossEntropyLoss()
        # else:
        #     criterion = nn.CrossEntropyLoss(weight=weight)

        # def _step():
        #     optimizer.zero_grad()
        #     loss = criterion(self._temperature_scale(logits_tensor), targets_tensor)
        #     loss.backward()
        #     return loss

        # optimizer.step(_step)

    def _get_hidden_features(self, embs, ngram_embs):
        # embs = self.word_embs(embs)
        ngram_embs = self.ngrams_embs(ngram_embs)

        x = torch.cat([embs, ngram_embs], dim=1)
        x = torch.mean(x, dim=1)
        x = F.dropout(x, 1 - self.dropout_keep_prob)
        # x = F.relu(x)
        x = self.i2h(x)

        return x

    def _forward_alg(self, feats):
        x = self.h2o(feats)

        return x

    def forward(self, sents, calibrated=False):
        if torch.is_tensor(sents[0]):
            feats = self._get_hidden_features(*sents)
        else:
            feats = self._get_hidden_features(*process_sentences(sents))
        logits = self._forward_alg(feats)

        if calibrated:
            # logits = self._temperature_scale(logits)
            # topk_scores, topk_idx = torch.topk(logits, 5)
            # print(topk_scores)
            # logits = F.softmax(logits, dim=-1)
            # return logits if d_results is None else logits / d_results
            return logits
        else:
            return F.sigmoid(logits)
