import torch
import argparse
from paraphrase_vae.train import trainIters

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--checkpoint", type=str, default='')

args = parser.parse_args()

model, vocab = trainIters(n_iters=args.n_epochs,
                          checkpoint=args.checkpoint)
torch.save({
    'model_state': model.state_dict(),
    'vocab': vocab
}, 'paraphrase_vae.bin')