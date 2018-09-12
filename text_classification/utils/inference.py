import torch
import warnings

def infer_classification_output(model, logits, topk=None, context=''):
    # - model: the ModelWrapper class
    # - logits: torch.Tensor of size (batch_size, n_classes)
    # - context: [optional]: string
    # requires model to have the following attributes
    # label_encoder [LabelEncoder]: The Sklearn LabelEncoder object used to transform class labels
    # topk [int]: the number of classes to get. Default is 5

    # Just in case the user forgot to set topk for the model
    if not hasattr(model, 'topk') and isinstance(model.topk, int):
        warnings.warn('The model wrapper class should have a `topk` attribute. Using default value of 5')
        model_topk = 5
    else:
        model_topk = model.topk
    
    if model.is_pytorch_module():
        assert hasattr(model, 'n_classes') or hasattr(model.model, 'n_classes'), \
            "The attribute `n_classes` is required on the model wrapper class"
    else:
        assert hasattr(model, 'n_classes'), "The attribute `n_classes` is required on the model wrapper class"

    topk = topk or model_topk
    batch_size = logits.size(0)
    n_classes = model.model.n_classes if not model.is_pytorch_module() else model.n_classes
    topk = min(topk, n_classes) # Maximum will be the number of classes

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