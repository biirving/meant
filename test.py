#!/usr/bin/env python
import csv
import pandas as pd
import ast
import torch
from torch import nn, tensor
import os, sys, argparse, time, gc, socket
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

#nightly pytorch version required
#import torch._dynamo as dynamo

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from torch import nn, tensor
from torchmetrics import Accuracy, MatthewsCorrCoef, AUROC
import torchmetrics
import string
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, AutoModel
from transformers.debug_utils import DebugUnderflowOverflow
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall
import re
sys.path.append('../meant')
from meant import meant, meant_vision
from joblib import Memory

"""

all_ys = np.load('/work/nlp/b.irving/stock/complete/all_ys_5.npy')
all_xs = np.load('/work/nlp/b.irving/stock/complete/all_xs_5.npy')
print('total samples:', all_ys.shape)
print(all_xs.shape)
ones = 0
zeros = 0
for y in all_ys:
    if y.item() == 1:
        ones += 1
    else:
        zeros += 1

print('Positives:', ones)
print('Negatives: ', zeros)

y_resampled = np.load('/work/nlp/b.irving/stock/complete/y_resampled_5.npy')
print(y_resampled.shape)
ones = 0
zeros = 0
for y in y_resampled:
    if y.item() == 1:
        ones += 1
    else:
        zeros += 1

print('Positives resampled:', ones)
print('Negatives resampled: ', zeros)

"""

device = torch.device('cuda')
bertweet = AutoModel.from_pretrained("vinai/bertweet-base")

batch_size = 3 
target = torch.tensor([0, 1, 0]).to(device)

torch_dtype = torch.float16
model = meant(text_dim = 768, 
                image_dim = 768, 
                price_dim = 4, 
                height = 224, 
                width = 224, 
                patch_res = 16, 
                lag = 5, 
                num_classes = 2, 
                embedding = bertweet.embeddings, 
                num_encoders=12).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fct = nn.CrossEntropyLoss()

scaler = torch.cuda.amp.GradScaler()
for _ in range(100):  
    optimizer.zero_grad()
    sample_graphs = torch.randn(batch_size, 5, 4, 224, 224).cuda()
    sample_macds = torch.randn(batch_size, 5, 4).cuda()
    sample_tweets = torch.ones(batch_size, 5, 128).long().cuda()
    with torch.autocast(device_type="cuda", dtype=torch_dtype):
        out = model.forward(sample_tweets, sample_graphs, sample_macds)
        loss = loss_fct(out, target.long()) 
        print('out', out)
        print('loss', loss)

    scaler.scale(loss).backward()
    #scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
    #optimizer.zero_grad()
    #loss.backward()
    #optimizer.step()

