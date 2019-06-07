from sent_to_vec.masked_lm.bert_model import BertLMWrapper

model = BertLMWrapper(from_fp='bert_vi_base.bin')
model.init_model()

_, seq_output = model(['Chào buổi sáng, hôm nay thật là một ngày đẹp trời'])
print(seq_output.size())