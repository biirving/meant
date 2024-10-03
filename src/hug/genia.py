
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
import torch


genia = load_dataset('siddharthtumre/jnlpba-split')
train = genia['train']

tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
model = AutoModelForTokenClassification.from_pretrained('dmis-lab/biobert-v1.1')
print(model)
print(model.classifier)

input = tokenizer(train[0]['tokens'], padding=True, truncation=True, is_split_into_words=True, return_tensors='pt')


outputs = model(input['input_ids'])
print(outputs['logits'].shape)
