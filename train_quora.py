import torch
import argparse
from paraphrase_vae.train import trainIters

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5)
parser.add_argument("--checkpoint", type=str, default='')

args = parser.parse_args()

model, vocab, vocab_size = trainIters(n_iters=args.n_epochs,
                                      checkpoint=args.checkpoint)

torch.save({
    'model_state': model.state_dict(),
    'vocab': vocab,
    'vocab_size': vocab_size
}, 'paraphrase_vae.bin')
