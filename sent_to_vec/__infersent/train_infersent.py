import torch
import argparse
from sent_to_vec.train import trainIters

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lr_shrink", type=float, default=5)
parser.add_argument("--lr_decay", type=float, default=1e-5)
parser.add_argument("--checkpoint", type=str, default='')
parser.add_argument("--encoder_type", type=str, default='qrnn_mean')

args = parser.parse_args()

encoder, nli_net = trainIters(n_iters=args.n_epochs,
                              batch_size=args.batch_size,
                              checkpoint=args.checkpoint,
                              lr=args.lr,
                              lr_shrink=args.lr_shrink,
                              lr_decay=args.lr_decay,
                              encoder_type=args.encoder_type)
torch.save(encoder.state_dict(), 'encoder.bin')
torch.save(nli_net.state_dict(), 'nli_net.bin')
