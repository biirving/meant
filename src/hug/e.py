import torch
from torch import nn, tensor
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from allennlp_light.modules import ConditionalRandomField
from allennlp_light.modules.conditional_random_field.conditional_random_field import allowed_transitions

data = load_dataset('tner/tweetner7')
train = data['train_2021']
input = train[0:1]

tokenizer = AutoTokenizer.from_pretrained('roberta-large', add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained('tner/roberta-large-tweetner7-all')
print(model)
"""
def tokenize_and_align_labels(examples):
    list_len = [len(i) for i in examples['tokens']]
    max_length = max(list_len)
    if(len(list_len) == 1):
        if(max_length <= 2):
            tokenized_inputs = tokenizer(examples['tokens'], padding='max_length', max_length=128, is_split_into_words=True, return_tensors='pt')
        else:
            tokenized_inputs = tokenizer(examples['tokens'], padding='max_length', max_length=128, truncation=True,is_split_into_words=True, return_tensors='pt')
    else:
        tokenized_inputs = tokenizer(examples['tokens'], padding='max_length', max_length=128, truncation=True, is_split_into_words=True, return_tensors='pt')
    labels = []
    accuracy_labels = []
    for i, label in enumerate(examples['tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        accuracy_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx: 
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    return tokenized_inputs, labels 


x_inputs, labels = tokenize_and_align_labels(input)
model = AutoModelForTokenClassification.from_pretrained('roberta-large')
model.classifier = nn.Linear(1024, 15)
out = model(**x_inputs)
id2label = {
    "B-corporation": 0,
    "B-creative_work": 1,
    "B-event": 2,
    "B-group": 3,
    "B-location": 4,
    "B-person": 5,
    "B-product": 6,
    "I-corporation": 7,
    "I-creative_work": 8,
    "I-event": 9,
    "I-group": 10,
    "I-location": 11,
    "I-person": 12,
    "I-product": 13,
    "O": 14
}
id2label = {v: k for k, v in id2label.items()}
crf_layer = ConditionalRandomField(
                num_tags=15,
                constraints=allowed_transitions(constraint_type="BIO", labels=id2label)
            )
crf_out = crf_layer(out['logits'], torch.tensor(labels), x_inputs['attention_mask'])
#for x in range(100):
 #  if(63 in wow['train'][x]['ner_ids']):
  #    print(63 in wow['train'][x]['ner_ids'])


"""