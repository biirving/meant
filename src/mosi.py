# Some insanity here at the end
import os
import sys
from typing import *
import pickle
import h5py
import numpy as np
from numpy.core.numeric import zeros_like
from torch.nn.functional import pad
from torch.nn import functional as F

import torch
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 


def drop_entry(dataset):
    """Drop entries where there's no text in the data."""
    drop = []
    for ind, k in tqdm(enumerate(dataset["text"])):
        if k.sum() == 0:
            drop.append(ind)
    # for ind, k in enumerate(dataset["vision"]):
    #     if k.sum() == 0:
    #         if ind not in drop:
    #             drop.append(ind)
    # for ind, k in enumerate(dataset["audio"]):
    #     if k.sum() == 0:
    #         if ind not in drop:
    #             drop.append(ind)
    
    for modality in list(dataset.keys()):
        dataset[modality] = np.delete(dataset[modality], drop, 0)
    return dataset

# Like with most things, you have to do it yourself
filepath='../aligned_50.pkl'
with open(filepath, "rb") as f:
    alldata = pickle.load(f)

processed_dataset = {'train': {}, 'test': {}, 'valid': {}}
alldata['train'] = drop_entry(alldata['train'])
alldata['valid'] = drop_entry(alldata['valid'])
alldata['test'] = drop_entry(alldata['test'])


print(alldata['train']['vision'].shape)
print(alldata['train']['text'].shape)
print(alldata['train']['text_bert'].shape)
print(alldata['train']['text_bert'][0].shape)
print(alldata['train']['text_bert'][1].shape)


print(alldata['train'].keys())

labels = pd.read_csv('../label.csv')

print(len([label for label in alldata['train']['classification_labels'] if label > 0]))




