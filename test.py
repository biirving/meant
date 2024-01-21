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
from transformers import BertTokenizer, BertModel

from transformers.debug_utils import DebugUnderflowOverflow
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall
import re
sys.path.append('../meant')
from meant import meant, meant_vision, meant_tweet, temporal, meant_tweet_no_lag
from joblib import Memory
from datasets import load_dataset

from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall




class metrics():
    def __init__(self, num_classes, set_name):
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.f1_micro = MulticlassF1Score(num_classes=num_classes, average='micro')
        self.precision_macro = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.precision_micro = MulticlassPrecision(num_classes=num_classes, average='micro')
        self.recall_macro = MulticlassRecall(num_classes=num_classes, average='macro')
        self.recall_micro = MulticlassRecall(num_classes=num_classes, average='micro')
        self.set_name = set_name
    
    def update(self, pred, target):
        self.accuracy.update(pred, target) 
        self.f1_macro.update(pred, target) 
        self.f1_micro.update(pred, target)
        self.precision_macro.update(pred, target)
        self.precision_micro.update(pred, target)
        self.recall_macro.update(pred, target)
        self.recall_micro.update(pred, target)

    def compute(self):
        acc = self.accuracy.compute()
        f1_macro = self.f1_macro.compute()
        f1_micro = self.f1_micro.compute()
        precision_macro = self.precision_macro.compute()
        precision_micro = self.precision_micro.compute()
        recall_macro = self.precision_macro.compute()
        recall_micro = self.precision_micro.compute()
        return (acc, f1_macro, f1_micro, precision_macro, 
                precision_micro, recall_macro, recall_micro)

    def show(self):
        (accuracy, 
        f1_macro, 
        f1_micro, 
        precision_macro, 
        precision_micro, 
        recall_macro, 
        recall_micro) = self.compute()
        print(self.set_name + ' accuracy: ', accuracy)
        print('Macro ' + self.set_name + ' f1: ', f1_macro)
        print('Micro ' + self.set_name + ' f1: ', f1_micro)
        print('Macro ' + self.set_name + ' precision: ', precision_macro)
        print('Micro ' + self.set_name + ' precision: ', precision_micro)
        print('Macro ' + self.set_name + ' recall: ', recall_macro)
        print('Micro ' + self.set_name + ' recall: ', recall_micro)
        return f1_macro, f1_micro

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

np_dtype = np.float64
#graphs = np.memmap('/work/nlp/b.irving/stock/complete/graphs_5.npy', dtype=np_dtype, mode='r', shape=(16714, 5, 4, 224, 224))
#tweets = np.memmap('/work/nlp/b.irving/stock/complete/tweets_5.npy', dtype=np_dtype, mode='r', shape=(16714, 5, 128))
# not memmapped, so the autocast can change it?
graphs = np.load('/work/nlp/b.irving/stock/complete/graphs_5.npy')

# going for the original tweets
#tweets = np.load('/work/nlp/b.irving/stock/complete/all_original_tweets_resampled_5.npy')
#tweets = np.load('/work/nlp/b.irving/stock/complete/all_original_tweets_5.npy')
tweets = np.load('/work/nlp/b.irving/stock/complete/tweets_5.npy')
#macds = np.memmap('/work/nlp/b.irving/stock/complete/macds_5.npy', dtype=np_dtype, mode='r', shape=(16714, 5, 4))
#labels = np.memmap('/work/nlp/b.irving/stock/complete/y_resampled_5.npy', dtype=np_dtype, mode='r', shape=(16714, 1))
#labels = np.load('/work/nlp/b.irving/stock/complete/y_og_tweets_resampled_5.npy')
labels = np.load('/work/nlp/b.irving/stock/complete/y_resampled_5.npy')
#labels = np.load('/work/nlp/b.irving/stock/complete/all_ys_5.npy')
 
# Load the IMDb dataset

# we are seeing if my model works on a simple balanced dataset with the berttweet auto tokenizer
#dataset = load_dataset("imdb")

# Access the training and testing splits
#train_dataset = dataset["train"]
#test_dataset = dataset["test"]
#unsupervised = dataset["unsupervised"]

print('loaded data')

# normalize data - min max scaling. My data is NOT necessarily normalized
# should we normalize by lag data

# this could hurt the graph data, because many of the columns are just simply one all the way through
# first scale the graphs:

# lets try without scaling first?

"""
min_val = 0.0
max_val = 1.0

# Perform Min-Max scaling for each channel (color channel)
graphs = (graphs.astype(float) - 0) / (255 - 0)  # Assuming original pixel values range from 0 to 255

# Ensure the scaled values are within the specified range [0, 1]
graphs = np.clip(graphs, min_val, max_val)

# then, scale the tweets (normalize by column)
# how do you do this with the lag period
# flatten, do it column wise, then reshape again?
# how do you even multiply it after that
min_val = tweets.max(axis=0)
max_val = tweets.max(axis=0)

min_val = np.repeat(min_val[np.newaxis, :, :], tweets.shape[0], axis=0)
max_val = np.repeat(max_val[np.newaxis, :, :], tweets.shape[0], axis=0)

tweets = (tweets - min_val) / (max_val - min_val) 
"""
# load into dataloader
from torch.utils.data import Dataset, DataLoader

# load 
class customDataset(Dataset):
    def __init__(self, graphs, tweets, macds, labels):
        self.graphs = torch.tensor(graphs)
        self.tweets = torch.tensor(tweets)
        self.macds = torch.tensor(macds)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.graphs[idx], self.tweets[idx], self.macds[idx], self.labels[idx]


class TweetsData(Dataset):
    def __init__(self, tweets, labels):
        self.tweets = torch.tensor(tweets)
        self.labels = torch.tensor(labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.tweets[idx], self.labels[idx]

# generalize training loop

batch_size = 16 
#data = customDataset(graphs, tweets, macds, labels)
data = TweetsData(tweets, labels)
toProcess = DataLoader(data, shuffle=True, batch_size=batch_size, pin_memory=True)

class CustomIMDBDataset(Dataset):
    def __init__(self, dataset, split):
        self.dataset = dataset[split]
        self.texts = self.dataset["text"]
        self.labels = self.dataset["label"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label

#train_data = CustomIMDBDataset(dataset, 'train')
#test_data = DataLoader(CustomIMDBDataset(dataset, 'test'), shuffle=True, batch_size=batch_size, pin_memory=True)

device = torch.device('cuda')
# use the bertweet embeddings?
bert = AutoModel.from_pretrained("vinai/bertweet-base")

# do the graphs introduce too much noise?
# should we get more general images, of S&P 500 related images in general?
# I think so
# using the macd graphs is not very interesting now is it
model = meant_tweet(text_dim = 768, 
                price_dim = 4, 
                height = 224, 
                width = 224, 
                patch_res = 16, 
                num_classes = 2, 
                lag=5, 
                embedding = bert.embeddings, 
                num_encoders=1).cuda()

# create a function which detects vanishing or exploding gradients
def detect_vanishing_exploding(layer_name, gradient_tensor, exploding_threshold=10, vanishing_threshold=0.0000001):
    gradient_magnitude = gradient_tensor.norm()
    vanishing = False
    if gradient_magnitude > exploding_threshold:
        print("Exploding Gradients Detected in " + layer_name + "!")
        vanishing = True
    elif gradient_magnitude < vanishing_threshold:
        print("Vanishing Gradients Detected in " + layer_name + "!")
        vanishing = True
    else:
        print("Gradients within acceptable range.")
    if vanishing:
        sys.exit()

# temporal gradient?
def get_temporal_grad(module):
    for layer in module:
        if type(layer) is temporal:
            detect_vanishing_exploding('multi mad', layer.multi_mad.weight.grad)
            detect_vanishing_exploding('q', layer.q.weight.grad)
            detect_vanishing_exploding('k', layer.k.weight.grad)
            detect_vanishing_exploding('v', layer.v.weight.grad)
        else:
            detect_vanishing_exploding('linear layer', layer.weight.grad)

# custom hook to calculate the layer gradients (chat gpt generated)
def custom_hook(module, grad_input, grad_output):
    print("Layer Gradients:")
    for name, param in module.named_parameters():
        if param.grad is not None:
            print(name, "Gradient Norm:", param.grad.norm().item())


# torch datatype for casting
torch_dtype = torch.float16

# okay, lets test this with metrics. The investigation continues

# test with a simple balanced dataset
metric = metrics(2, 'train')
test_metric = metrics(2, 'test')


index = 0
# lets try to use a pretrained bert model

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class SentimentClassifier(nn.Module):
  def __init__(self, n_classes=2):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained("bert-base-uncased")
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(output.pooler_output)
    return self.out(output)

#model = SentimentClassifier()

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fct = nn.CrossEntropyLoss()

# will using the scaler mitigate the problem
#scaler = torch.cuda.amp.GradScaler()

num_epochs=1
for epoch in range(num_epochs):
    progress_bar = tqdm(toProcess, desc=f'Epoch {epoch+1}/{num_epochs}')
    for text, target in progress_bar:

        # tokenize inputs and move everything to cuda
        #tweets = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        #tweets = {key: value.to('cuda') for key, value in tweets.items()}

        #out = model(tweets['input_ids'], tweets['attention_mask'])
        # so now the tweets are out of bounds? WTF?
        # they were processed with different tokenizer
        out = model(text.long().cuda())
        
#        with torch.autocast(device_type="cuda", dtype=torch.float16):
        optimizer.zero_grad()
        loss = loss_fct(out, target.long().cuda()) 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
            #if torch.isnan(loss).any() or torch.isnan(out[1]).any():
            #    raise ValueError('Nans encountered. Training failure')
            #    sys.exit()
        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
        #scaler.update()

        out = out.detach().cpu()
        target = target.detach().cpu()
        metric.update(out, target)
        index += 1
        if index % 100 == 0:
            print('out', out)
            print('target', target)
            metric.show()

metric.show()

"""
for text, target in test_data:
    tweets = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    out = model.forward(tweets['input_ids'].cuda().long())
    out = out.detach().cpu()
    test_metric.update(out, target)

test_metric.show()
"""

    # detect vanishing or exploding gradient
    #get_temporal_grad(model.temporal_encoding[0].temp_encode)
