import torch
import argparse
from paraphrase_vae.train import trainIters
from paraphrase_vae.predict import load_model, predict

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5)
parser.add_argument("--checkpoint", type=str, default='')
parser.add_argument("--rnn_type", type=str, default='GRU')
parser.add_argument("--interactive", type=bool, default=False)

args = parser.parse_args()

if not args.interactive:
    model, vocab, vocab_size = trainIters(n_iters=args.n_epochs,
                                          checkpoint=args.checkpoint,
                                          rnn_type=args.rnn_type)

    torch.save({
        'model_state': model.state_dict(),
        'vocab': vocab,
        'vocab_size': vocab_size,
        'rnn_type': model.rnn_type
    }, 'paraphrase_vae.bin')
else:
    if args.checkpoint == '':
        print('Interactive mode requires the --checkpoint arg')
    else:
        load_model(model_file=args.checkpoint)
        print('Interactive mode (type "exit" to quit)')

        text = input(' > ')
        while text != 'exit':
            print(' (eager) H: %s' % predict(text, beam_width=0)) # Eager attempt
            for _ in range(3):  # generate 3 times to attempt different outputs
                print(' (beam) H: %s' % predict(text, beam_width=7))
            text = input(' > ')
