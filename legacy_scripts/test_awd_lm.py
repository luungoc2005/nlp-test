
from sent_to_vec.awd_lm.model import LanguageModelWrapper


if __name__ == '__main__':
    model = LanguageModelWrapper(from_fp='awd-lm-checkpoint.bin')
    model.init_model()

    # print('Featurizer: Top 10 words: {}'.format(
    #     ' '.join(list(model.featurizer.tokenizer.word_index.keys())[:10])
    # ))

    print(model.generate(50))