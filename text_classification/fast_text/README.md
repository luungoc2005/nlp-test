Sample usage:

```
from text_classification.fast_text.model import FastTextWrapper
from text_classification.fast_text.train import FastTextLearner
from common.callbacks import PrintLoggerCallback

model = FastTextWrapper({'num_classes': num_classes})
learner = FastTextLearner(
    model,
    optimizer_fn='sgd',
    optimizer_kwargs={'lr': 1e-2, 'momentum': .9}
)

learner.fit(
    training_data=(X_train, y_train), 
    epochs=50, 
    callbacks=[PrintLoggerCallback(log_every=5)]
)
```
Where:
 - X_train: list of strings
 - y_train: list of classes (int)