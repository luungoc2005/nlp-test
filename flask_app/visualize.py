import torch
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from nltk.tokenize import word_tokenize
from common.utils import word_to_vec
from common.word_vectors import get_dim

from typing import List, Tuple

def transform_input(input_list: List[str], model=None) -> np.array:
    X = None
    if not torch.is_tensor(X) and model._featurizer is not None:
        X = model._featurizer.transform(input_list)
        X = model.preprocess_input(X)
        X = X.numpy()
    else: # falls back to BoW
        raw_tokens = [word_tokenize(sent) for sent in input_list]
        sent_vectors = word_to_vec(raw_tokens)
        
        sent_matrix = np.zeros(len(raw_tokens), get_dim())
        for ix in range(len(raw_tokens)):
            sent_matrix[ix] = np.mean(np.array(sent_vectors[ix]), axis=0)
    return X

    # vectorizer = TfidfVectorizer(min_df=2,
    #                              strip_accents='unicode', lowercase=True, ngram_range=(1, 2),
    #                              norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)

    # return vectorizer.fit_transform(input_list).todense()

def visualize_matrix(X: np.array, n_clusters: int = None) -> Tuple[np.array, np.array]:
    y_pred = None
    if n_clusters is not None:
        print("Clustering data...")
        y_pred = KMeans(n_clusters=n_clusters).fit_predict(X)

    print("Analysing with t-SNE...")
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)

    return X_tsne, y_pred

def visualize_inputs(input_list: List[str], model=None, n_clusters: int = None) -> List[dict]:
    if len(input_list) == 0: return []
    
    X = transform_input(input_list, model=model)
    if X is None: return None
    
    X_tsne, y_pred = visualize_matrix(X, n_clusters)

    result = []
    for i in range(X_tsne.shape[0]):
        item_value = {
            'text': input_list[i],
            'x': float(X_tsne[i, 0]),
            'y': float(X_tsne[i, 1]),
        }
        if y_pred is not None:
            item_value['group'] = int(y_pred[i])
        result.append(item_value)
    
    return result