import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.sparse
from torch.autograd import Variable
from glove_utils import get_emb_matrix
from config import NGRAM_BINS, EMBEDDING_DIM
from fasttext_utils import _process_sentences
from sklearn.svm import SVC


def process_sentences(sentences):
    words, ngrams = _process_sentences(list(sentences))
    return Variable(torch.from_numpy(words).long(), requires_grad=False), \
        Variable(torch.from_numpy(ngrams).long(), requires_grad=False)


class FastText(nn.Module):

    def __init__(self,
                 hidden_size=100,
                 dropout_keep_prob=0.5,
                 classes=10):
        super(FastText, self).__init__()

        self.hidden_size = hidden_size
        self.dropout_keep_prob = dropout_keep_prob
        self.classes = classes

        # self.mean_embs = nn.EmbeddingBag(MAX_NUM_WORDS + 1, EMBEDDING_DIM, mode='mean', padding_idx=-1)
        emb_matrix = torch.from_numpy(get_emb_matrix()).float()
        self.word_embs = nn.Embedding.from_pretrained(emb_matrix)
        self.word_embs.padding_idx = 0
        self.word_embs.weight.requires_grad = False

        self.ngrams_embs = nn.EmbeddingBag(NGRAM_BINS, EMBEDDING_DIM,
                                           mode='mean', sparse=True)
        self.ngrams_embs.weight.requires_grad = False
        # self.highway = Highway(EMBEDDING_DIM * 2, 1, F.relu)

        self.i2h = nn.Linear(EMBEDDING_DIM * 2, self.hidden_size, bias=False)
        self.h2o = nn.Linear(self.hidden_size, self.classes)

        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        self.detector = None

        # self.i2h = nn.Linear(EMBEDDING_DIM * 2, self.hidden_size)
        # self.w2h = nn.Linear(emb_matrix.size(1), self.hidden_size // 2)
        # self.n2h = nn.Linear(EMBEDDING_DIM, self.hidden_size // 2)

        # self.h2o = nn.Linear(self.hidden_size, self.classes)

        self.init_weights()

    def init_weights(self):
        # nn.init.xavier_normal_(self.i2o.weight)
        nn.init.xavier_normal_(self.i2h.weight)
        nn.init.xavier_normal_(self.h2o.weight)

    def _temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def _calibrate(self, data_loader, weight=None):
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for _, (inputs, targets) in enumerate(data_loader, 0):
                feats = self._get_hidden_features(*process_sentences(inputs))
                all_logits.append(self._forward_alg(feats))
                all_targets.append(targets.long())
        logits_tensor = torch.cat(all_logits, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)

        # Find temperature (probability calibration)
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        if weight is None:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss(weight=weight)

        def _step():
            optimizer.zero_grad()
            loss = criterion(self._temperature_scale(logits_tensor), targets_tensor)
            loss.backward()
            return loss

        optimizer.step(_step)

        # Train outlier detector
        # self.eval()

        all_logits = []
        all_feats = []
        for _, (inputs, targets) in enumerate(data_loader, 0):
            feats = self._get_hidden_features(*process_sentences(inputs))
            all_feats.append(feats)
            # all_logits.append(self._forward_alg(feats))
        # logits_tensor = torch.cat(all_logits, dim=0)
        # feats_tensor = torch.cat(all_feats, dim=0)

        # # Scale all logits
        # logits_tensor = self._temperature_scale(logits_tensor)

        # self.detector = SVC(kernel='linear', probability=True)
        # self.detector.fit(feats_tensor.detach().numpy(), targets_tensor.detach().numpy())
        self.train()

    def _get_hidden_features(self, embs, ngram_embs):
        embs = torch.mean(self.word_embs(embs), dim=1)
        # embs = F.dropout(embs, 1 - self.dropout_keep_prob)
        # embs = F.relu(self.w2h(embs))

        ngram_embs = self.ngrams_embs(ngram_embs)
        # ngram_embs = torch.mean(self.ngrams_embs(ngram_embs), dim=1)
        # ngram_embs = F.dropout(ngram_embs, 1 - self.dropout_keep_prob)
        # ngram_embs = F.relu(self.n2h(ngram_embs))

        x = torch.cat([embs, ngram_embs], dim=1)
        x = F.dropout(x, 1 - self.dropout_keep_prob)
        x = F.relu(x)
        # x = self.highway(x)
        # x = self.i2o(x)
        x = self.i2h(x)

        return x

    def _forward_alg(self, feats):
        x = self.h2o(feats)

        return x

    def forward(self, sents, calibrated=False):
        feats = self._get_hidden_features(*process_sentences(sents))
        logits = self._forward_alg(feats)

        if calibrated:
            # d_results = None
            logits = self._temperature_scale(logits)

            # if self.detector is not None:
                # logits = self.detector.predict_proba(feats.detach().numpy())
                # logits = torch.from_numpy(logits)
                # d_results = self.detector.predict(logits.detach().numpy()).reshape(-1, 1)
                # print(self.detector.decision_function(logits.detach().numpy()).reshape(-1, 1))
                # print(d_results)
                # d_results = torch.from_numpy(d_results).float().expand(-1, logits.size(1))

            logits = F.softmax(logits, dim=-1)
            # return logits if d_results is None else logits / d_results
            return logits
        else:
            return logits
