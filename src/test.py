#!/usr/bin/env python
import sys
from datasets import load_dataset
import torch
import csv
import pandas as pd
from transformers import ViltProcessor, ViltModel
from PIL import Image
import requests
import numpy as np
sys.path.append('..')

# prepare image and text
model = torch.load('/work/nlp/b.irving/meant_runs/models/meant_timesformer/meant_timesformer_1_TempStockLarge_0_15.pt')
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')

sys.exit()
texts = pd.read_csv('/scratch/irving.b/stock/TempStockLarge.csv')

unique_tickers = texts['ticker'].unique()
num_unique_tickers = len(unique_tickers)

# Calculate the number of rows for each unique value in the 'ticker' column
ticker_counts = texts['ticker'].value_counts()
# Prepare the output string
output = f"Number of unique tickers: {num_unique_tickers}\n\n"
output += "Number of rows for each unique ticker:\n"
output += ticker_counts.to_string()

# Save the output to a text file
with open('ticker_counts.txt', 'w') as file:
    file.write(output)

sys.exit()
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm", do_rescale=False)
model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

text = texts.iloc[0]['text_4']
image = graphs[0, 0, :, :, :]
inputs = processor(image, text, max_length=40, return_tensors="pt", truncation=True)
print(inputs.keys())
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)

sys.exit()

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
import torch
import sys



# hugging face login 
login('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')

model_checkpoint = "dandelin/vilt-b32-mlm"
dataset = load_dataset("Graphcore/vqa", split="validation[:200]")
print(dataset)

labels = [item['ids'] for item in dataset['label']]
flattened_labels = list(itertools.chain(*labels))
unique_labels = list(set(flattened_labels))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()} 

def replace_ids(inputs):
  inputs["label"]["ids"] = [label2id[x] for x in inputs["label"]["ids"]]
  return inputs

dataset = dataset.map(replace_ids)
flat_dataset = dataset.flatten()
flat_dataset.features

processor = ViltProcessor.from_pretrained(model_checkpoint)
def preprocess_data(examples):
    image_paths = examples['image_id']
    images = [Image.open(image_path) for image_path in image_paths]
    texts = examples['question']    
    encoding = processor(images, texts, padding="max_length", truncation=True, return_tensors="pt")
    print(encoding['pixel_mask'])
    for k, v in encoding.items():
          encoding[k] = v.squeeze()
    targets = []
    for labels, scores in zip(examples['label.ids'], examples['label.weights']):
        target = torch.zeros(len(id2label))
        for label, score in zip(labels, scores):
            target[label] = score
        targets.append(target)
    encoding["labels"] = targets
    return encoding

processed_dataset = flat_dataset.map(preprocess_data, batched=True, remove_columns=['question','question_type',  'question_id', 'image_id', 'answer_type', 'label.ids', 'label.weights'])
from transformers import DefaultDataCollator
data_collator = DefaultDataCollator()
from transformers import ViltForQuestionAnswering
model = ViltForQuestionAnswering.from_pretrained(model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)
from transformers import TrainingArguments
repo_id = "MariaK/vilt_finetuned_200"
training_args = TrainingArguments(
    output_dir=repo_id,
    per_device_train_batch_size=4,
    num_train_epochs=20,
    save_steps=200,
    logging_steps=50,
    learning_rate=5e-5,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=True,
)
from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=processed_dataset,
    tokenizer=processor,
)
trainer.train()
sys.exit()












bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
device = torch.device('cuda')
model = meant(text_dim = 768, 
    image_dim = 768, 
    price_dim = 4, 
    height = 224, 
    width = 224, 
    patch_res = 16, 
    lag = 5, 
    num_classes = 2, 
    embedding = bertweet.embeddings,
    flash=False,
    num_encoders=1).to(device)
tweets = torch.ones(16, 5, 128)
images = torch.ones(16,5, 4, 224, 224)
model.forward(tweets.long().cuda(), images.cuda())

sys.exit()
model_checkpoint = "dandelin/vilt-b32-mlm"
processor = ViltProcessor.from_pretrained(model_checkpoint)

dataset = load_dataset("Graphcore/vqa", split="train[0:10]")
print(dataset[0])
print(dataset)

labels = [item['ids'] for item in dataset['label']]
flattened_labels = list(itertools.chain(*labels))
unique_labels = list(set(flattened_labels))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()} 
def replace_ids(inputs):
    inputs["label"]["ids"] = [label2id[x] for x in inputs["label"]["ids"]]
    return inputs

# we want to preprocess these values
dataset = dataset.map(replace_ids)
flat_dataset = dataset.flatten()
print(flat_dataset)

# need to account for the preprocess data
def preprocess_data(examples):
    image_paths = examples['image_id']
    images = [Image.open(image_path) for image_path in image_paths]
    texts = examples['question']    
    encoding = processor(images, texts, padding="max_length", truncation=True, return_tensors="pt")
    for k, v in encoding.items():
        encoding[k] = v.squeeze()
    targets = []
    for labels, scores in zip(examples['label.ids'], examples['label.weights']):
        target = torch.zeros(len(id2label))
        for label, score in zip(labels, scores):
            target[label] = score
        targets.append(target)
    encoding["labels"] = targets
    return encoding

processed_dataset = flat_dataset.map(preprocess_data, batched=True, 
    remove_columns=['question', 'question_type', 'question_id', 'image_id', 'answer_type', 'label.ids', 'label.weights'])

# TODO: Batch extract the image and language input ids

print(processed_dataset[0].keys())
print(id2label)
print([len(label) for label in processed_dataset["labels"]])
#print(processed_dataset)

#print(flat_dataset)
#print(flat_dataset[0])
sys.exit()
device = torch.device('cuda')
bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
print(pretrained_vision)
model = meant(text_dim = 768, 
                image_dim = 768, 
                price_dim = 4, 
                height = 224, 
                width = 224, 
                patch_res = 16, 
                lag = 5, 
                num_classes = 2, 
                embedding = bertweet.embeddings,
                flash=False,
                num_encoders=12).to(device)
# using pretrained language encoder
# need to pretrain a 12 encoder setup
pretrained_vision = torch.load('/work/nlp/b.irving/meant_runs/models/meant_vision_encoder/meant_vision_encoder_Tempstock_0.pt')
language_encoders = torch.load('/work/nlp/b.irving/meant_runs/models/meant_language_encoder/meant_language_encoder_12_tempstock_0_1.pt')
model.languageEncoders = language_encoders.languageEncoders
model.visionEncoders = pretrained_vision.visionEncoders
print(model)
sys.exit()
# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters: {total_params}")
print(f"Trainable Parameters: {trainable_params}")

sys.exit()
sample_graph = torch.ones(3, 5, 4, 224, 224).half().to(device)
sample_tweet = torch.ones(3, 5, 128).long().to(device)

t0 = time.time()
for _ in range(1000):
    model.forward(sample_tweet, sample_graph)
print(time.time() - t0)
sys.exit()



adjacent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'michinaga'))

# also, want to make a stocknet dataset
# for training?
# figure out this import, and we are on the homestretch
# on god
sys.path.append('/work/nlp/b.irving/michinaga/teanet/models')
sys.path.append('/work/nlp/b.irving/michinaga/teanet/utils/')
import classicAttention
from teanet import teanet
device = torch.device('cuda')

#config = AutoConfig.from_pretrained('/work/nlp/b.irving/nlp/src/hug/configs/vilbert.json', local_files_only=True)
#config = AutoConfig.from_pretrained('/work/nlp/b.irving/nlp/src/hug/configs/' + args.model_name +'.json', local_files_only=True)

# run on teanet to compare. Make stocknet into a dataset that I can process easily?
# only contains one modality
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
macds = np.memmap('/work/nlp/b.irving/stock/complete/macds_5.npy', dtype=np_dtype, mode='r', shape=(16714, 5, 4))
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
        self.labels = torch.tensor(labels)
        self.macds = torch.tensor(macds)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.graphs[idx], self.tweets[idx], macds[idx], self.labels[idx]


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
data = customDataset(tweets, graphs, macds, labels)
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
#model = meant_tweet(text_dim = 768, 
#                price_dim = 4, 
#                num_classes = 2, 
#                lag=5, 
#                embedding = bert.embeddings, 
#                num_encoders=1).cuda()

#config = AutoConfig.from_pretrained('/work/nlp/b.irving/nlp/src/hug/configs/vl_bert.json', local_files_only=True)
#vl_bert_model = VisualBertModel._from_config(config).cuda()
# change the word embeddings to match the bertweet word embeddings
#vl_bert_model.embeddings.word_embeddings = bert.embeddings.word_embeddings
# need to use the BerTWEET embedding layer, because the number of classes is different
#model = CustomClassifier(vl_bert_model, 768, 2).cuda()

# for vilt testing

config = AutoConfig.from_pretrained('/work/nlp/b.irving/nlp/src/hug/configs/vilt.json', local_files_only=True)
vilt = ViltModel._from_config(config)
vilt.embeddings.text_embeddings.word_embeddings=bert.embeddings.word_embeddings
#model = ViltWrapper(vilt, 768, 2).to(device)
model = teanet(5, 128, 2, 5, 12, 10).cuda()

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
metric = f1_metrics(2, 'train')
test_metric = f1_metrics(2, 'test')


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
    for text, image, macd, target in progress_bar:

        # tokenize inputs and move everything to cuda
        #tweets = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        #tweets = {key: value.to('cuda') for key, value in tweets.items()}

        #out = model(tweets['input_ids'], tweets['attention_mask'])
        # so now the tweets are out of bounds? WTF?
        # they were processed with different tokenizer
        # going to throw an error
        #out = model(text[:, 4, :].squeeze(dim=1).to(torch.float32).cuda(), image[:, 4, :, :].to(torch.float32).squeeze(dim=1).cuda())
        out = model(text.float().cuda(), macd.float().cuda())
        
        
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
