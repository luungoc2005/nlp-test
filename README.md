# botbot-nlp

## Quick start
- See the .ipynb files (viewable on Github) for instructions and demo using the script

[entities_train.ipynb](https://github.com/2359media/botbot-nlp/blob/master/entities_train.ipynb) for the intent classification model results

- Complete the environmental setup to run the .ipynb files

## Environmental setup:

1. Download the fastText English word vectors from [https://github.com/facebookresearch/fastText/blob/master/docs/pretrained-vectors.md](https://github.com/facebookresearch/fastText/blob/master/docs/pretrained-vectors.md)
(Download the `bin+text` file but only copy the `.bin` model file)

Note: Technically, the models can be generalized to use vectors from [https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md](https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md) for all other languages

Data path configurations are stored in `config.py`

(Deprecated but might have possible future uses)
GloVE word vectors can be downloaded from:
- [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) - This project uses `glove.6B.zip` but can also be changed to the other files by changing `config.py`
- Place the downloaded `glove.6B.300d.txt` file into data/glove
- The project also uses 20.000 most common words list from [https://github.com/first20hours/google-10000-english/blob/master/20k.txt](https://github.com/first20hours/google-10000-english/blob/master/20k.txt)

2. Setup Python environment:
- Download Anaconda/Miniconda for Python 3.6
- Create a new environment for the project by running `conda create --name botbot-nlp python=3.6`
- Switch to the newly created environment by running `source activate botbot-nlp` (`activate botbot-nlp` inside Anaconda Prompt on Windows)
- Install dependencies by running `pip install -r requirements.txt` or `pip install -r requirements_win.txt` for Windows
- Install any missing dependencies (because of platform differences)

3. Using Jupyter notebook for evaluation
- Activate the environment by `source activate botbot-nlp`
- Navigate to the root directory by `cd`
- Run `jupyter notebook`. A browser tab should open and navigate to jupyter notebook at `localhost:8888` by default
- Open `entities_train.ipynb` inside the notebook
- Click Kernel > Run All to start training

Note: if the progress bars doesn't show up properly during training, run `conda install -c conda-forge ipywidgets`

## Extras

1. Using GPU (TO BE IMPLEMENTED)

Using GPU will massively speed up training and inference time (brings training from hours of CPU time to about an hour or a few minutes depending on GPU spec)

e.g:
Training on CPU: Intel Core i7-4710HQ @2.7Ghz: ~45m/epoch - 12 epochs to reach 87% accuracy
Training on GPU: same machine, NVIDIA GeForce GTX 850M: ~4m/epoch 

- Follow Tensorflow-GPU setup instructions at [https://www.tensorflow.org/install/install_linux](https://www.tensorflow.org/install/install_linux) (Including installing the exact CUDA & CuDNN versions - e.g if using Tensorflow 1.5.0 then CUDA 9.0 and CuDNN v7.0 is required even if newer versions exist)
- Run `source activate botbot-nlp`
- Run `pip uninstall tensorflow` then `pip install --upgrade tensorflow-gpu`

2. Using Tensorboard
- Run `tensorboard --logdir=bilstm/logs/`
- Navigate to [localhost:6006](localhost:6006) to see training graphs

(These steps are completely optional but are used for deprecated code paths / might help with experimenting)
3. Install HDF5 for Keras model saving/loading
- Install HDF5 from [https://support.hdfgroup.org/HDF5/](https://support.hdfgroup.org/HDF5/) or by using `homebrew`:
- Instructions using `homebrew` (on UNIX):
```
brew tap homebrew/science
brew install hdf5
```

- Activate the environment by `source activate botbot-nlp`
- Install h5py by running `pip install h5py`

4. Download NLTK data:
- Activate the environment by `source activate botbot-nlp`
- Run `python -m nltk.downloader all`


## Explanation
This demonstrates the use of Facebook fastText for intent classification
Entities recognition is a Bi-directional LSTM + CRF implemented in PyTorch
This also uses fastText for word embeddings to alleviate the use of Char-CNN + GloVE embeddings for performance (should re-explore the other option some time in the future)

## TODOS

Models:
- Try using spaces as separate tokens (modify `bilstm/utils - wordpunct_tokenize` function)

Other:

- Setup a docker image for ease getting started
- Setup a small Flask server and CLI for ease in using the project