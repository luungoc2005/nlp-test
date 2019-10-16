# from sent_to_vec.masked_lm.pervasive_model import PervasiveAttnLanguageModelWrapper
from sent_to_vec.masked_lm.bert_model import BertLMWrapper
from sent_to_vec.masked_lm.train import LanguageModelLearner
from sent_to_vec.masked_lm.corpus_data import LanguageModelCorpusDataset
from common.callbacks import PrintLoggerCallback, EarlyStoppingCallback, ModelCheckpointCallback, TensorboardCallback, ReduceLROnPlateau
from common.lr_schedulers import WarmupLinearSchedule
from os import path, listdir
from config import BASE_PATH
from common.modules import AdamW
from apex.optimizers import FusedAdam
from common.utils import dotdict

# alias for old path
import sys
from common.preprocessing import keras
sys.modules['common.keras_preprocessing'] = keras

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--export_vocab", default=False, action='store_true')
parser.add_argument("--from_vocab", default='', type=str)
parser.add_argument("--base_dir", default=BASE_PATH, type=str)

args = parser.parse_args()

if __name__ == '__main__':

    from common.utils import set_seed
    set_seed(42)

    MODEL_PATH = 'en-masked-lm-test.bin'
    model_config = dotdict({
        'num_words': 36000,
        'embedding_size': 128,
        'hidden_size': 576,
        'num_hidden_layers': 7, # or 6 is also fine
        'num_attention_heads': 12,
        'intermediate_size': 1200,
        'hidden_act': 'gelu',
        'hidden_dropout_prob': 0,
        'attention_probs_dropout_prob': 0,
        'positional_embedding_type': 'absolute',
        'max_position_embeddings': 256,
        'featurizer_seq_len': 256, # same as above
        'type_vocab_size': 1,
        'initializer_range': 0.025,
        'proj_share_all_but_first': True,
        'div_val': 1.0,
        'use_adasoft': True,
        'adasoft_cutoffs': [8000, 10000, 18000],
    })
    if path.exists(MODEL_PATH):
        print('Resuming from saved checkpoint')
        # model = PervasiveAttnLanguageModelWrapper(from_fp=MODEL_PATH)
        model = BertLMWrapper(from_fp=MODEL_PATH)
        model.init_model(update_configs=model_config)
    else:
        # model = PervasiveAttnLanguageModelWrapper({
        #     'n_layers': 6,
        #     'tie_weights': True,
        #     'embedding_dim': 300,
        #     'hidden_dim': 300,
        #     'use_adasoft': True,
        #     'num_words': 50000
        # }) # large model
        model = BertLMWrapper(model_config)

    dataset = LanguageModelCorpusDataset()

    SAVE_PATH = path.join(args.base_dir, 'maskedlm-data.bin')
    BATCH_SIZE = 200

    if path.exists(SAVE_PATH):
        print('Loading from previously saved file')
        dataset.load(SAVE_PATH, model, base_dir=args.base_dir)
    else:
        paths = [
            # path.join(BASE_PATH, 'data/wikitext2/wiki.train.tokens'),
            path.join(BASE_PATH, 'data/wikitext103raw/wiki.train.raw')
        ]
        def load_folder(folder_path, filter_txt=True):
            paths.extend([
                path.join(folder_path, filename)
                for filename in listdir(folder_path)
                if not filter_txt or filename.lower().endswith('txt')
            ])
        # load_folder(path.join(BASE_PATH, 'data/bookcorpus'))
        load_folder(path.join(BASE_PATH, 'data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled'), filter_txt=False)
        load_folder(path.join(BASE_PATH, 'data/stories_corpus'))

        if args.from_vocab == '':
            dataset.init_on_model(model, data_path=paths, base_dir=args.base_dir)
        else:
            with open(args.from_vocab, 'r') as vocab_fp:
                dataset.init_on_model(model, data_path=paths, vocab_fp=vocab_fp, base_dir=args.base_dir)
            # import random
            # indices = list(range(len(dataset)))
            # random.shuffle(indices)
            # print('5 sample sentences:')
            # for ix in indices[:5]:
            #     sent = dataset._get_raw_sent(ix)
            #     print(f'- #{ix}: {sent} ({len(sent)})')
        dataset.save(SAVE_PATH)

    if args.export_vocab:
        with open('vocab.json', 'w') as vocab_file:
            model.featurizer.tokenizer.export_vocab(vocab_file)
            exit()
    # learner = LanguageModelLearner(model, 
    #     optimizer_fn='sgd', 
    #     optimizer_kwargs={'lr': 30, 'weight_decay': 1.2e-6}
    # )
    # learner = LanguageModelLearner(model, 
    #     optimizer_fn='sgd',
    #     optimizer_kwargs={'lr': 10, 'weight_decay': 1.2e-6}
    # )
    n_epochs=2
    t_total = n_epochs * (len(dataset) // BATCH_SIZE)
    learner = LanguageModelLearner(model,
        optimizer_fn=FusedAdam,
        optimizer_kwargs={
            'betas': (0.9, 0.98),
            'lr': 7e-4
        }
    )
    print('Dataset: {} sentences'.format(len(dataset)))
    # lr_range = list(range(25, 35))
    # losses = learner.find_lr(lr_range, {
    #     'training_data': dataset,
    #     'batch_size': BATCH_SIZE,
    #     'epochs': 1,
    #     'minibatches': 500
    # })
    # print([
    #     (lr, losses[idx]) for idx, lr in enumerate(lr_range)
    # ])
    learner.fit(
        training_data=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        epochs=n_epochs,
        callbacks=[
            PrintLoggerCallback(log_every_batch=1000, log_every=1, metrics=['loss']),
            TensorboardCallback(log_every_batch=100, log_every=-1, metrics=['loss']),
            ModelCheckpointCallback(metrics=['loss']),
            # ReduceLROnPlateau(reduce_factor=4, patience=2)
        ],
        gradient_accumulation_steps=10,
        fp16=True,
        lr_schedulers=[
            (WarmupLinearSchedule, {
                'warmup_steps': 20000,
                't_total':  t_total
            })
        ]
        # optimize_on_cpu=True,
    )
