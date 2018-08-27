import torch
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm, trange
from config import SENTENCE_DIM

from tensorboardX import SummaryWriter
from os import path

from common.utils import argmax, to_variable, get_datetime_hostname, timeSince
from text_classification.fast_text.model import FastText, sentence_vector

import time

BASE_PATH = path.dirname(__file__)
SAVE_PATH = path.join(BASE_PATH, 'model/model.bin')
LOG_DIR = path.join(BASE_PATH, 'logs/')
