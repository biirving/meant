import torch
import json
import torchvision.transforms as transforms
import torch
from transformers import AutoModel, AutoTokenizer
import time
from PIL import Image
import numpy as np 
import os
from tqdm import tqdm 

# bert handles 512? Thats pretty large though
# will start with 128
max_seq_length = 128

# we gonna need those tickers boy
sp500arr = np.loadtxt("constituents.csv",
                 delimiter=",", dtype=str)

# going to start with a slice of the dataset
sp500 = sp500arr[:, 0][1:50]

# for the test
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

"""
how should we read the tweet files to train the model
"""
index = 0
for index in tqdm(range(sp500.shape[0])):
    tick = sp500[index]
    final_path = f'/work/socialmedia/stock/tweets/' 
    #os.makedirs(final_path, exist_ok=True)

    graph_path = f'/home/irving.b/aurelian_data/graphs/' + tick + '/'
    # original tweets
    folder_path = "/home/irving.b/aurelian_data/tweets/" + tick  # replace with the path to your folder
    files = os.listdir(folder_path)
    files.sort()
    count = 0
    # so, we are going to make a condensed tensor for each ticker
    to_save = None
    for f in files:
        date = f.split('.')[0]
        # we have to check if both the graph and the tweet exist
        # the tweet file inherently exists, because BANG
        if os.path.isfile(graph_path + date + '.png'):
            tweet_file = open(folder_path + '/' + f)
            text = tweet_file.readlines()
            tokens = []
            for t in range(len(text) - 1):
                tweet_dict = json.loads(text[t])
                tokens += tokenizer.tokenize(tweet_dict['text']) + ['[SEP]']
            # then we pad or truncate the text
            if(len(tokens) > max_seq_length):
                tokens = tokens[:max_seq_length]
            else:
                tokens += ["[PAD]"] * (max_seq_length - len(tokens))
            input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
            if(to_save is None):
                to_save = input_ids
            else:
                to_save = torch.cat((to_save, input_ids), dim = 0)
    torch.save(to_save, final_path + tick + '.pt')        
    
