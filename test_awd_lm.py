
from sent_to_vec.awd_lm.model import LanguageModelWrapper

model = LanguageModelWrapper(from_fp='awd-lm-checkpoint.bin')
model.init_model()

print(model.generate(50))