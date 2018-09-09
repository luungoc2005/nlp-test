import torch

def infer_classification_output(model, logits, topk=None, context=''):
    # - model: the ModelWrapper class
    # - logits: torch.Tensor of size (batch_size, n_classes)
    # - context: [optional]: string
    # requires model to have the following attributes
    # label_encoder [LabelEncoder]: The Sklearn LabelEncoder object used to transform class labels
    # topk [int]: the number of classes to get. Default is 5

    topk = topk or model.topk
    batch_size = logits.size(0)
    topk = min(topk, model.n_classes) # Maximum will be the number of classes

    if context != '' and hasattr(model, 'contexts') and model.contexts is not None:
        assert len(model.contexts) == logits.size(1), \
            'Length of contexts array must equal the number of classes'
        mul = [cls_context == context for cls_context in model.contexts]
        mul = torch.Tensor(mul).long()
        mul = torch.unsqueeze(0).expand_as(logits)
        logits = torch.mul(logits, mul)

    top_probs, top_idxs = torch.topk(logits, topk)

    top_idxs = top_idxs.numpy()
    top_classes = [
        model.label_encoder.inverse_transform(top_idxs[idx])
        for idx in range(batch_size)
    ]
    return [
        [{
            'intent': top_classes[sent_idx][idx],
            'confidence': top_probs[sent_idx][idx].item()
        } for idx in range(topk)]
        for sent_idx in range(batch_size)]