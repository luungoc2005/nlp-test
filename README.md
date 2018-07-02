# botbot-nlp

## Quick start
- See the .ipynb files (viewable on Github) for instructions and demo using the script

- Intent classification:
[intents_train.ipynb](https://github.com/2359media/botbot-nlp/blob/master/text_classification/intents_train.ipynb)
[intents_predict.ipynb](https://github.com/2359media/botbot-nlp/blob/master/text_classification/intents_predict.ipynb)

- Entities recognition:
[entities_train.ipynb](https://github.com/2359media/botbot-nlp/blob/master/entities_recognition/entities_train.ipynb)
[entities_predict.ipynb](https://github.com/2359media/botbot-nlp/blob/master/entities_recognition/entities_predict.ipynb)

- Complete the environmental setup to run the .ipynb files

## Environmental setup:

1. Run `data/get_data.bash` to download data files
Notes:
- Modify this file to selectively download only required files (usually either GLoVE/fasttext vectors)
- You might need to run `chmod u+x data/get_data.bash` as well to grant execute permissions
- Technically, the models can be generalized to use vectors from [https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md](https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md) for all other languages

Data path configurations are stored in `config.py`

- The project (might) also uses 20.000 most common words list from [https://github.com/first20hours/google-10000-english/blob/master/20k.txt](https://github.com/first20hours/google-10000-english/blob/master/20k.txt)

2. Setup Python environment:
- Download Anaconda/Miniconda for Python 3.6
- Create a new environment for the project by running `conda create --name botbot-nlp python=3.6`
- Switch to the newly created environment by running `source activate botbot-nlp` (`activate botbot-nlp` inside Anaconda Prompt on Windows)
- Install dependencies by running `pip install -r requirements.txt` or `pip install -r requirements_win.txt` for Windows
- Install any missing dependencies (because of platform differences)

3. Build Cython modules
- Run `python setup.py build_ext --inplace` (requires gcc/clang - `sudo apt-get install build-essential` on Deb/Ubuntu or Xcode+CLI tools on OSX)

4. Using Jupyter notebook for evaluation
- Activate the environment by `source activate botbot-nlp`
- Navigate to the root directory by `cd`
- Run `jupyter notebook`. A browser tab should open and navigate to jupyter notebook at `localhost:8888` by default
- Open `entities_train.ipynb` inside the notebook
- Click Kernel > Run All to start training

Note: if the progress bars doesn't show up properly during training, run `conda install -c conda-forge ipywidgets`

## Scripts
`train_quora.py` trains the paraphrasing model on the Quora duplicate questions dataset

`train_sent_to_vec.py` trains the InferSent model on NLI + SNLI

`train_amazon_sentiment.py` trains the classification model on the amazon sentiment dataset

`train_conll_eval` trains the entity recognition model on the CoNLL2003 dataset

`start_flask.py` (to be used with -debug True) is for debugging the NLU flask server

### Notes about using the Flask server
The model should be run on Gunicorn by

`gunicorn -w 1 -t 0 -b 127.0.0.1:5000 start_flask:app`

Arguments:
- `-w` number of workers
- `-t` timeout (set to a high number because loading word vectors takes a while)
- `-b` optionally binds to a different address

After running the server
1. `/upload (POST)` is used to upload a data file & train the NLU on the data file with the `file` argument - e.g: `curl -X POST -F 'file=@./francis.json' 127.0.0.1:5000`
```
curl -X POST 
-F "file=@./francis.json"
127.0.0.1:5000/upload
```

2. `/predict (POST)` is used to send a query for prediction

e.g
```
curl -X POST
-H "Content-Type: application/json"  
-d '{"query":"Hello world!"}' 
127.0.0.1:5000/predict | json_pp
```

## Extras

1. Using GPU (TO BE IMPLEMENTED)

Using GPU will massively speed up training and inference time (brings training from hours of CPU time to about an hour or a few minutes depending on GPU spec)

e.g:
- Training on CPU: Intel Core i7-4710HQ @2.7Ghz: ~45m/epoch - 12 epochs to reach 87% accuracy
- Training on GPU: same machine, NVIDIA GeForce GTX 850M: ~4m/epoch 

- Follow Tensorflow-GPU setup instructions at [https://www.tensorflow.org/install/install_linux](https://www.tensorflow.org/install/install_linux) (Including installing the exact CUDA & CuDNN versions - e.g if using Tensorflow 1.5.0 then CUDA 9.0 and CuDNN v7.0 is required even if newer versions exist)
- Run `source activate botbot-nlp`
- Use the commands for specific platforms on [http://pytorch.org/](http://pytorch.org/) to install PyTorch
- (Run `pip uninstall tensorflow` then `pip install --upgrade tensorflow-gpu`)

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