import matplotlib
matplotlib.use('Agg')

import argparse
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib import offsetbox
from textwrap import shorten

from common.data_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input', help='Input data in JSON/TXT format'
    )
    parser.add_argument(
        'output', nargs='?', help='Path to output file in .PNG', default=''
    )
    parser.add_argument(
        '--cutoff', type=int, help='The amount of items to analyse', default=100
    )
    parser.add_argument(
        '--clusters', type=int, help='Number of clusters for KMeans', default=10
    )
    parser.add_argument(
        '--logdir', type=str, help='Tensorboard log folder'
    )
    parser.add_argument(
        '--dist', type=float, help='Min distance for annotations', default=0.02
    )
    parser.add_argument(
        '--width', type=int, help='Max text width', default=25
    )
    args = parser.parse_args()

    if not args.input or (not args.output and not args.logdir):
        parser.print_help()
        sys.exit(1)

    DATA_POINTS = 100

    print("Reading data...")
    if (len(args.input) > 5 and args.input[-5:].lower() == '.json'):
        X_data, y_data = get_data_pairs(data_from_json(args.input))
    else:
        X_data = [
            line.rstrip() for line in
            open(args.input, 'r')
        ]
    
    vectorizer = TfidfVectorizer(min_df=2,
        strip_accents = 'unicode', lowercase=True, ngram_range=(1,2),
        norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)

    print("Converting data to vectors...")
    X_data = X_data[:args.cutoff]
    X = vectorizer.fit_transform(X_data)
    # X = -(X * X.T) # Calculate dot product
    X = X.todense()

    if args.output:
        print("Clustering data...")
        y_pred = KMeans(n_clusters=args.clusters).fit_predict(X)

        print("Analysing with t-SNE...")
        tsne = TSNE(n_components=2,
            perplexity=5,
            learning_rate=10, 
            n_iter=3000)
        X_tsne = tsne.fit_transform(X)
        
        print("Ploting scatter graph...")
        x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
        X_tsne = (X_tsne - x_min) / (x_max - x_min)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(X_tsne.shape[0]):
            X_txt = shorten(X_data[i], width=args.width, placeholder='...')
            plt.text(X_tsne[i, 0], X_tsne[i, 1], X_txt,
                        color=plt.cm.Dark2(y_pred[i] / 9.),
                        fontdict={'weight': 'bold', 'size': 6})
        plt.xticks([]), plt.yticks([])
        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X_tsne.shape[0]):
                dist = np.sum((X_tsne[i] - shown_images) ** 2, 1)
                if np.min(dist) < args.dist:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X_tsne[i]]]
                X_txt = shorten(X_data[i], width=args.width, placeholder='...')
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.TextArea(X_txt, textprops={'size': 8}),
                    X_tsne[i])
                ax.add_artist(imagebox)
        fig.savefig(args.output)
    elif args.logdir:
        X = torch.from_numpy(X).float()
        print("Writing tensorboard embedding...")
        writer = SummaryWriter(log_dir=args.logdir)
        writer.add_embedding(X, global_step=0, metadata=X_data)
        writer.close()