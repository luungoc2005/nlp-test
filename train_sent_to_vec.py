import torch
import argparse
from sent_to_vec.train import trainIters

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--checkpoint", type=str, default='')

args = parser.parse_args()

encoder, nli_net = trainIters(n_iters=args.n_epochs,
                              batch_size=args.batch_size,
                              checkpoint=args.checkpoint)
torch.save(encoder.state_dict(), 'encoder.bin')
torch.save(nli_net.state_dict(), 'nli_net.bin')
