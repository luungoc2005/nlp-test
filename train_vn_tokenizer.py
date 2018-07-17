import torch
import argparse
from common.langs.vi_VN.tokenizer.train import trainIters

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--checkpoint", type=str, default='')

args = parser.parse_args()

model = trainIters(n_iters=args.n_epochs,
                   checkpoint=args.checkpoint,
                   lr=args.lr,)
