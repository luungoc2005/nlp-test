botbot-nlp

# Quick start
- See the .ipynb files (viewable on Github) for instructions and demo using the script
- Complete the environmental setup to run the .ipynb files

# Environmental setup:

1. Download GloVE word vectors
- Download the files from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) - This project uses `glove.6B.zip` but can also be changed to the other files by changing `config.py`
- Place the downloaded `glove.6B.300d.txt` file into data/glove (this path can also be changed in `config.py`)
- The project also uses 20.000 most common words list from [https://github.com/first20hours/google-10000-english/blob/master/20k.txt](https://github.com/first20hours/google-10000-english/blob/master/20k.txt)

2. Setup Python environment:
- Download Anaconda/Miniconda for Python 3.6
- Create a new environment for the project by running `conda create --name botbot-nlp python=3.6`
- Switch to the newly created environment by running `source activate botbot-nlp` (`activate botbot-nlp` inside Anaconda Prompt on Windows)
- Install dependencies by running `pip install -r requirements.txt` or `pip install -r requirements_win.txt` for Windows
- Install any missing dependencies (because of platform differences)

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

5. Using Jupyter notebook for evaluation
- Activate the environment by `source activate botbot-nlp`
- Navigate to the root directory by `cd`
- Run `jupyter notebook`. A browser tab should open and navigate to jupyter notebook at `localhost:8888` by default
- Open `entities_train.ipynb` inside the notebook
- Click Kernel > Run All to start training

Note: if the progress bars doesn't show up properly during training, run `conda install -c conda-forge ipywidgets`

4. Using GPU

Using GPU will massively speed up training and inference time (brings training from hours of CPU time to about an hour or a few minutes depending on GPU spec)

e.g:
Training on CPU: Intel Core i7-4710HQ @2.7Ghz: ~45m/epoch - 12 epochs to reach 87% accuracy
Training on GPU: same machine, NVIDIA GeForce GTX 850M: ~4m/epoch 

- Follow Tensorflow-GPU setup instructions at [https://www.tensorflow.org/install/install_linux](https://www.tensorflow.org/install/install_linux) (Including installing the exact CUDA & CuDNN versions - e.g if using Tensorflow 1.5.0 then CUDA 9.0 and CuDNN v7.0 is required even if newer versions exist)
- Run `source activate botbot-nlp`
- Run `pip uninstall tensorflow` then `pip install --upgrade tensorflow-gpu`

5. Using Tensorboard
- Navigate to the respective models' folders (`convnet` and `bilstm`)
- Run `tensorboard --logdir=./logs`
- Navigate to [localhost:6006](localhost:6006) to see training graphs

# Explanation

# TODOS

Models:

- Try replacing ConvNet Dense layers by AveragePooling layers for reduced number of parameters (faster training)
- Implement transfer learning (Dropping the last Dense layer & Modifying Embedding layer to reuse old weights - Important - this will massively speed up training when editing models)
- Explore RNN/LSTM-based architectures for classification (requires large text corpus)

Other:

- Setup a docker image for ease getting started
- Setup a small Flask server and CLI for ease in using the project