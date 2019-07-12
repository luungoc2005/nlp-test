from text_classification.with_pretrained.model import LMClassifierWrapper
import torch
test_item = ["thử nghiệm"]
classifier = LMClassifierWrapper(from_fp='bert_vi_sentiment.bin')
classifier.init_model()
torch.softmax(classifier(test_item, return_logits=True)[0], dim=1)