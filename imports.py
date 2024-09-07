import argparse
import json
import os
import random
from functools import partial
from pathlib import Path
from pprint import pprint
import pickle

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from jaxtyping import Float
from nnsight import LanguageModel
from tqdm import tqdm
# from transformer_lens import ActivationCache, HookedTransformer, utils
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

import wandb
# from e2e_sae import SAETransformer
# from sparse_autoencoder import sparse_autoencoder

from datasets import load_dataset

data = load_dataset("roneneldan/TinyStoriesInstruct")