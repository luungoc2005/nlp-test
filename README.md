botbot-nlp

# Environmental setup:

1. Download GloVE word vectors
- Download the files from ...
- Place the downloaded `glove.6B.300d.txt` file into data/glove (this path can also be changed in `./glove_utils.py`)
- The project also uses 20.000 most common words list from ...

2. Setup Python environment:
- Download Anaconda/Miniconda for Python 3.6
- Create a new environment for the project by running `conda create --name botbot-nlp python=3.6`
- Switch to the newly created environment by running `source activate botbot-nlp`
- Install dependencies by running `pip install -r requirements.txt`
- Install any missing dependencies (because of platform differences)

3. Install HDF5 for Keras model saving/loading

4. Download NLTK data:
- Run `python`
- Type the following into the prompt:
```
import nltk

```

5. Using Jupyter notebook for evaluation
- Run `jupyter notebook`. A browser tab should open and navigate to jupyter notebook at `localhost:8888` by default
- Open `entities_train.ipynb` inside the notebook
- Click Kernel > Run All to start training

4. Using GPU
- Follow Tensorflow-GPU setup instructions at ... (Including installing CUDA & CuDNN)

# Explanation


# TODOS
- Try replacing ConvNet Dense layers by AveragePooling layers for reduced number of parameters (faster training)
- Implement transfer learning (Dropping the last Dense layer & Modifying Embedding layer to reuse old weights)
- Explore RNN/LSTM-based architectures for classification