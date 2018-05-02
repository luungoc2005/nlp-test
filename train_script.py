import torch
from sent_to_vec.train import trainIters

encoder, nli_net = trainIters(batch_size=32)
torch.save(encoder.state_dict(), 'encoder.bin')
torch.save(nli_net.state_dict(), 'nli_net.bin')