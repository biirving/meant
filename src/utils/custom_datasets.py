from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import emoji
from datasets import load_dataset
import itertools
from PIL import Image
import torchvision.transforms as transforms
import sys
import numpy as np
from PIL import Image
import io
import pandas as pd

def replace_emojis_with_text(text):
    return emoji.demojize(text) 

# custom datasets classes for loading different tasks
# dynamic padding for the longest sequence?
class mlm_dataset(Dataset):
    def __init__(self, dataset, split, tokenizer, max_length=256, mlm_probability=0.15):
        # Select the split of the dataset
        if split is not None:
            self.dataset = dataset[split]
        else:
            self.dataset = dataset
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.dataset)

    # this should be dynamic though
    def __getitem__(self, idx):
        # Get the text item from the dataset at the specified index
        text = self.dataset[idx]
        # Tokenize the text; this does NOT yet apply MLM
        tokens = self.tokenizer(text, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        # Prepare inputs for MLM
        inputs, labels = self.mask_tokens(tokens['input_ids'])
        return {'input_ids': inputs, 'labels': labels, 'attention_mask': tokens['attention_mask']}

    def mask_tokens(self, inputs):
        """Prepare masked tokens inputs/labels for masked language modeling."""
        # Clone the input_ids to use as labels for MLM
        labels = inputs.clone()
        # Create a mask array
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # Apply special token mask: do not mask special tokens
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        # Mask tokens with probability `mlm_probability`
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # Replace masked input tokens with tokenizer.mask_token_id
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        return inputs, labels


class clm_dataset(Dataset):
    def __init__(self, dataset, split, tokenizer, max_length=256):
        # Select the split of the dataset
        if split is not None:
            self.dataset = dataset[split]
        else:
            self.dataset = dataset
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the text item from the dataset at the specified index
        text = self.dataset[idx]
        # Tokenize the text
        tokens = self.tokenizer(text, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        
        # Prepare inputs and labels for CLM: labels are shifted by one token
        # so that the model predicts the next token for each token in the input
        inputs = tokens['input_ids']
        labels = inputs.clone()
        labels[:, :-1] = inputs[:, 1:]  # Shift input to the left for labels
        labels[:, -1] = -100  # Ignore the last token for loss calculation

        return {'input_ids': inputs.squeeze(), 'labels': labels.squeeze(), 'attention_mask': tokens['attention_mask'].squeeze()}


# our class
class mim_dataset(Dataset):
    def __init__(self, dataset, split=None, transform=None, mask_probability=0.15, mask_value=0):
        # Select the split of the dataset
        if split is not None:
            self.dataset = dataset[split]
        else:
            self.dataset = dataset
        
        self.transform = transform
        self.mask_probability = mask_probability
        self.mask_value = mask_value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the image from the dataset at the specified index
        image = self.dataset[idx]
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        # Prepare inputs for MIM by masking parts of the image
        inputs, labels = self.mask_image(image)
        return {'input_ids': inputs, 'labels': labels}

    def mask_image(self, image):
        """
        Randomly mask parts of the image.
        """
        labels = image.clone()
        mask = torch.bernoulli(torch.full(labels.shape, self.mask_probability)).bool()
        
        inputs = torch.where(mask, self.mask_value, labels)
        labels[~mask] = -100  

        return inputs, labels



# dynamic padding for our language and image inputs
# maybe I should just account for this as if they were all tensors
def vqa_collate_fn(batch):
    # pad our language inputs dynamically, according to the max length of our longest input
    max_length = max([len(item['language_input_ids']) for item in batch])
    # Pad each item in the batch to the max_length
    for item in batch:
        item['language_input_ids'] += [0] * (max_length - len(item['language_input_ids']))
    # Current image dimensions
    max_height = max(item['pixel_values'][0].shape[1] for item in batch)
    max_width = max(item['pixel_values'][0].shape[2] for item in batch)
    for item in batch:
        current_height = item['pixel_values'][0].shape[1]
        current_width = item['pixel_values'][0].shape[2]
        # Calculate required padding for height and width
        height_padding = (max_height - current_height) // 2
        width_padding = (max_width - current_width) // 2
        # For odd differences, add the extra padding to the 'after' side
        top_pad = height_padding
        bottom_pad = max_height - current_height - top_pad
        left_pad = width_padding
        right_pad = max_width - current_width - left_pad
        # Specify padding for height, width, and (optionally) channels
        if len(item['pixel_values'][0].shape) == 3:  # If the image has a channel dimension
            padding = ((0, 0), (top_pad, bottom_pad), (left_pad, right_pad))
        else:  # For grayscale images without a channel dimension
            padding = ((top_pad, bottom_pad), (left_pad, right_pad))
        # Pad the image
        padded_image = np.pad(item['pixel_values'][0], padding, mode='constant', constant_values=0)
        item['pixel_values'][0] = padded_image
    # we can return a dictionary with our stacked tensors
    padded_images = torch.stack([torch.from_numpy(item['pixel_values'][0]) for item in batch])
    padded_pixel_masks = torch.ones(padded_images.shape)
    padded_language_input_ids = torch.stack([torch.tensor(item['language_input_ids']) for item in batch])
    padded_attention_masks = torch.ones(padded_language_input_ids.shape)
    labels = torch.stack([item['labels'] for item in batch])

    encoding = {'language_input_ids':padded_language_input_ids, 'attention_mask':padded_attention_masks, 
        'pixel_values':padded_images, 'pixel_mask':padded_pixel_masks, 'labels':labels}
    return encoding



# Im proud of YOU
class vqa_dataset(Dataset):
    def __init__(self, id2label, label2id, data, tokenizer, max_length=40, split=None):
        self.id2label = id2label
        self.label2id = label2id
        self.max_length = max_length
        self.num_classes = len(self.id2label) + 1
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224
            transforms.ToTensor()  # Convert the image to a PyTorch tensor
        ])
        # we want to preprocess these values
        dataset = data.map(self.replace_ids)
        # this should work
        self.dataset = dataset.flatten()
        self.tokenizer = tokenizer
        del dataset
    # replace the input ids
    def replace_ids(self, inputs):
        inputs["label"]["ids"] = [self.label2id[x] for x in inputs["label"]["ids"]]
        return inputs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # try the vilt processor?
        #question = self.tokenizer(example['question'], padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        image = example['image_id']
        image = Image.open(image).convert("RGB")

        # we don't want to return our values as tensors before the collator
        encoding = self.tokenizer(image, example['question'])
        #image_input_ids = self.transform(image)
        #encoding = self.tokenizer(image, example['question'], padding='max_length', 
        #        max_length=self.max_length, truncation=True, return_tensors='pt')
        #   pixel mask for the pixel values IS NOT precomputed
        encoding = {'language_input_ids':encoding['input_ids'], 
            'attention_mask':encoding['attention_mask'], 'pixel_values':encoding['pixel_values']}
        target = torch.zeros(self.num_classes)
        # soft 
        for i, id in enumerate(example['label.ids']):
            target[id] = example['label.weights'][i]
        encoding['labels'] = target
        # will it stack the encodings automatically 
        return encoding
    

class FilteredDataset(Dataset):
    def __init__(self, data, good_indices):
        self.data = data[good_indices]
        self.good_indices = good_indices

    def __len__(self):
        return len(self.good_indices)

    def __getitem__(self, idx):
        # Fetch the index of the 'good' data
        return self.data[idx]


        
def lag_text_image_collator(batch):
    padded_periods = []
    max_shape = None
    # Okay, so we have a bit of a problem (Again, ultra-productive: Fix your datasets today, and start to rerun experiments)
    for period in batch:
        one_period = torch.nn.utils.rnn.pad_sequence([torch.tensor(day) for day in period['input_ids']], batch_first=True, padding_value=0)
        padded_periods.append(one_period)
        if max_shape is None:
            max_shape = one_period.shape
        elif max_shape[1] < one_period.shape[1]:
            max_shape = one_period.shape
        
    # Pad tensors to the desired shape
    padded_tensors = []
    for tensor in padded_periods:
        padding = (0, max_shape[1] - tensor.shape[1])  # Pad width dimension
        padded_tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)
        padded_tensors.append(padded_tensor)
    # Stack the padded tensors along a new dimension
    input_ids = torch.stack(padded_tensors)

    input_ids = torch.stack(padded_tensors)
    if input_ids.shape[1] == 1:
        input_ids = input_ids.squeeze(dim=1)

    attention_mask = (input_ids != 0).long()  # Creates a mask of 1s for non-padded tokens and 0s for padded        

    labels = torch.tensor([period['labels'] for period in batch]) 
    pixels = torch.stack([torch.from_numpy(period['pixels']) for period in batch])
    pixel_mask = (pixels != 0).long()
    if 'prices' in list(batch[0].keys()):
        prices = torch.stack([torch.from_numpy(period['prices']) for period in batch])
        return {'input_ids':input_ids, 'attention_mask':attention_mask, 'prices':prices,  'pixels':pixels, 'labels':labels, 'pixel_mask':pixel_mask}
    elif 'audio' in list(batch[0].keys()):
        audio = torch.stack([torch.from_numpy(period['audio']) for period in batch])
        # I don't know if this is the way to make the mask though. They are not of shape 130?
        audio_attention_mask = (audio.sum(dim=-1) != 0).long()  # Shape: (b, 50)
        return {'input_ids':input_ids, 'attention_mask':attention_mask, 'audio':audio,  'pixels':pixels, 'labels':labels, 'pixel_mask':pixel_mask, 'audio_mask':audio_attention_mask}
    else:
        return {'input_ids':input_ids, 'attention_mask':attention_mask,  'pixels':pixels, 'labels':labels, 'pixel_mask':pixel_mask}

def lag_text_image_collator_no_lag(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(lag['input_ids']) for lag in batch], batch_first=True, padding_value=0)
    if input_ids.shape[1] == 1:
        input_ids = input_ids.squeeze(dim=1)
    attention_mask = (input_ids != 0).long()  # Creates a mask of 1s for non-padded tokens and 0s for padded        
    labels = torch.tensor([period['labels'] for period in batch]) 
    pixels = torch.stack([torch.from_numpy(period['pixels']) for period in batch])
    pixel_mask = (pixels.sum(dim=(-1, -2, -3)) != 0).long()
    if 'prices' in list(batch[0].keys()):
        prices = torch.stack([torch.from_numpy(period['prices']) for period in batch])
        return {'input_ids':input_ids, 'attention_mask':attention_mask, 'prices':prices,  'pixels':pixels, 'labels':labels}
    elif 'audio' in list(batch[0].keys()):
        audio = torch.stack([torch.from_numpy(period['audio']) for period in batch])
        audio_attention_mask = (audio.sum(dim=-1) != 0).long()  # Shape: (b, 50)
        return {'input_ids':input_ids, 'attention_mask':attention_mask, 'audio':audio,  'pixels':pixels, 'labels':labels, 'pixel_mask':pixel_mask, 'audio_mask':audio_attention_mask}
    else:
        return {'input_ids':input_ids, 'attention_mask':attention_mask,  'pixels':pixels, 'labels':labels, 'pixel_mask':pixel_mask}

def lag_text_collator(batch):
    padded_periods = []
    max_shape = None
    # Okay, so we have a bit of a problem (Again, ultra-productive: Fix your datasets today, and start to rerun experiments)
    for period in batch:
        one_period = torch.nn.utils.rnn.pad_sequence([torch.tensor(day) for day in period['input_ids']], batch_first=True, padding_value=0)
        padded_periods.append(one_period)
        if max_shape is None:
            max_shape = one_period.shape
        elif max_shape[1] < one_period.shape[1]:
            max_shape = one_period.shape
        
    # Pad tensors to the desired shape
    padded_tensors = []
    for tensor in padded_periods:
        padding = (0, max_shape[1] - tensor.shape[1])  # Pad width dimension
        padded_tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)
        padded_tensors.append(padded_tensor)
    # Stack the padded tensors along a new dimension

    input_ids = torch.stack(padded_tensors)
    if input_ids.shape[1] == 1:
        input_ids = input_ids.squeeze(dim=1)

    attention_mask = (input_ids != 0).long()  # Creates a mask of 1s for non-padded tokens and 0s for padded
    labels = torch.tensor([period['labels'] for period in batch]) 
    
    if 'prices' in list(batch[0].keys()):
        prices = torch.stack([torch.from_numpy(period['prices']) for period in batch])
        to_return = {'input_ids':input_ids, 'attention_mask':attention_mask, 'prices':prices, 'labels':labels}
    else:
        to_return =  {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':labels}
    

    return to_return


def lag_price_collator(batch):
    labels = torch.tensor([period['labels'] for period in batch]) 
    prices = torch.stack([torch.from_numpy(period['prices']) for period in batch])
    return {'prices':prices, 'labels':labels}

def lag_image_collator(batch):
    labels = torch.tensor([period['labels'] for period in batch]) 
    pixels = torch.stack([torch.from_numpy(period['pixels']) for period in batch])
    if 'prices' in list(batch[0].keys()):
        prices = torch.stack([torch.from_numpy(period['prices']) for period in batch])
        return {'prices':prices, 'pixels':pixels, 'labels':labels}
    else:
        return {'pixels':pixels, 'labels':labels}

def is_nan(value):
    if isinstance(value, float):
        return math.isnan(value)  # Use numpy.isnan(value) if you prefer numpy
    return False

class djia_lag_dataset(Dataset):
    def __init__(self, **kwargs):
        data = kwargs.pop('data', None)
        if data is None:
            raise ValueError("Data must be provided")

        self.data = pd.read_csv(data)
        self.tokenizer = kwargs.get('tokenizer', None)
        self.max_length = kwargs.get('max_length', 512)
        self.lag_period = kwargs.get('lag_period', 5)
        self.num_headlines = kwargs.get('num_headlines', 25)
        self.use_headlines = kwargs.get('use_headlines', True)
        self.lag_period = kwargs.get('lag_period', 5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        So, we actually weren't thinking about this before.
        How do we want to process all of the headlines into a text? Concatenate all of them, then tokenize? Or tokenize one at a time 
        then concatenate?
        '''
        item = self.data.iloc[idx]

        headlines = []
        labels = []
        prices = []
        for i in range(self.lag_period):
            headline = '' 
            if self.use_headlines:
                for j in range(1, self.num_headlines + 1):
                    cur_headline = item['Top' + str(j) + '_' + str(i)]
                    if not is_nan(cur_headline):
                        headline += cur_headline
            else:
                cur_headline = item['Top' + str(j) + '_' + str(i)]
                headline += cur_headline

            text = self.tokenizer(headline, truncation=True, max_length=self.max_length)
            headlines.append(text['input_ids'])
            prices.append(np.array(item[['High_' + str(i), 'Low_' + str(i), 'Adj Close_' + str(i)]].tolist()))
        label = item['djia_label']
        return {'input_ids':headlines, 'labels':label, 'prices':np.stack(prices, axis=0)}
    
class stocknet_dataset(Dataset):
    def __init__(self, **kwargs):
        data = kwargs.pop('data', None)
        if data is None:
            raise ValueError("Data must be provided")

        self.tokenizer = kwargs.get('tokenizer', None)
        self.max_length = kwargs.get('max_length', 512)
        self.lag_period = kwargs.get('lag_period', 5)
        self.use_tweets = kwargs.get('use_tweets', True)
        self.use_prices = kwargs.get('use_prices', True)
        self.use_lag = kwargs.get('use_lag', True)

        # Load data
        self.data = pd.read_csv(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        So, we actually weren't thinking about this before.
        How do we want to process all of the headlines into a text? Concatenate all of them, then tokenize? Or tokenize one at a time 
        then concatenate?
        '''
        item = self.data.iloc[idx]

        headlines = []
        labels = []
        prices = []
        for i in range(self.lag_period):
            cur_headline = item['text_' + str(i)]
            text = self.tokenizer(cur_headline, truncation=True, max_length=self.max_length)
            headlines.append(text['input_ids'])
            prices.append(np.array(item[['high_price_' + str(i), 'low_price_' + str(i), 'adjust_close_price_' + str(i)]].tolist()))
        
        #prices.append(np.array(item[['adjust_close_price_' + str(0),'adjust_close_price_' + str(1), 'adjust_close_price_' + str(2), 'adjust_close_price_' + str(3), 'adjust_close_price_' + str(4)]].tolist()))
        label = item['label']
        prev_labels = np.array(item[['label_0', 'label_1', 'label_2', 'label_3']])
        return {'input_ids':headlines, 'labels':label, 'prices':np.stack(prices, axis=0), 'prev_labels':prev_labels}

# This should load the data inside
class tempstock_lag_dataset(Dataset):
    def __init__(self, **kwargs):
        data = kwargs.pop('data', None)
        if data is None:
            raise ValueError("Data must be provided")
        self.tokenizer = kwargs.get('tokenizer', None)
        self.max_length = kwargs.get('max_length', 512)
        self.num_headlines = kwargs.get('num_headlines', 25)
        self.lag_period = kwargs.get('lag_period', 5)
        self.use_images = kwargs.get('use_images', True)
        self.use_tweets = kwargs.get('use_tweets', True)
        self.use_prices = kwargs.get('use_prices', True)
        self.use_lag = kwargs.get('use_lag', True)

        # Load labels
        self.labels = pd.read_csv(data['labels'], usecols=['label'])

        # Load data
        self.data = pd.read_csv(data['data']) if data.get('data') is not None else None
        self.graphs = np.load(data['graphs']) if data.get('graphs') is not None else None


    def __len__(self):
        if self.data is not None:
            return len(self.data)
        elif self.graphs is not None:
            return self.graphs.shape[0]
        else:
            raise ValueError('No data passed to loader. Please check arguments.')

    def __getitem__(self, idx):
        if self.data is not None:
            item = self.data.iloc[idx]
        headlines = []
        labels = []
        prices = []
        prev_labels = []
        label = self.labels.iloc[idx]['label']
        if self.use_tweets and self.use_images and self.use_prices:
            if self.use_lag:
                for i in range(self.lag_period):        
                    cur_headline = item['text_' + str(i)]
                    text = self.tokenizer(cur_headline, truncation=True, max_length=self.max_length)
                    headlines.append(text['input_ids'])
                    prices.append(np.array(item[['EMA12_' + str(i),'EMA26_' + str(i), 'Signal_Line_' + str(i), 'MACD_Histogram_' + str(i), 'MACD_' + str(i)]].tolist()))
                pixels = self.graphs[idx]
            else:
                cur_headline = item['text_' + str(4)]
                text = self.tokenizer(cur_headline, truncation=True, max_length=self.max_length)
                headlines.append(text['input_ids'])
                prices.append(np.array(item[['EMA12_' + str(4),'EMA26_' + str(4), 'Signal_Line_' + str(4), 'MACD_Histogram_' + str(4), 'MACD_' + str(4)]].tolist()))
                pixels = self.graphs[idx][4, :, :, :]
            return {'input_ids':headlines, 'labels':label, 'prices':np.stack(prices, axis=0), 'pixels':pixels}
        elif self.use_tweets and self.use_images:
            if self.use_lag:
                for i in range(self.lag_period):        
                    cur_headline = item['text_' + str(i)]
                    text = self.tokenizer(cur_headline, truncation=True, max_length=self.max_length)
                    headlines.append(text['input_ids'])
                    pixels = self.graphs[idx]            
            else:
                cur_headline = item['text_' + str(4)]
                inputs = self.tokenizer(cur_headline, truncation=True, max_length=self.max_length)
                #headlines.append(text['input_ids'])
                # For VILT (We will promptly change back for the remainder of the MEANT
                # Experiments )
                pixels = self.graphs[idx][4, :, :, :]
                #inputs = self.tokenizer(pixels, cur_headline, max_length=35, truncation=True)
                prices.append(np.array(item[['EMA12_' + str(4),'EMA26_' + str(4), 'Signal_Line_' + str(4), 'MACD_Histogram_' + str(4), 'MACD_' + str(4)]].tolist()))
                headlines.append(inputs['input_ids'])
            return {'input_ids':headlines, 'prices':np.stack(prices, axis=0), 'labels':label, 'pixels':pixels}
        elif self.use_tweets and self.use_prices:
            if self.use_lag:
                for i in range(self.lag_period):        
                    cur_headline = item['text_' + str(i)]
                    text = self.tokenizer(cur_headline, truncation=True, max_length=self.max_length)
                    headlines.append(text['input_ids'])
                    # Trying to add the previous 3 labels as well
                    prices.append(np.array(item[['EMA12_' + str(i),'EMA26_' + str(i), 'Signal_Line_' + str(i), 'MACD_Histogram_' + str(i), 'MACD_' + str(i), ]].tolist()))
                    #prices.append(np.array(item['Low_' + str(i), 'High_' + str(i), 'Open_' + str(i), 'Close_' + str(i), 'Adj Close_' + str(i)]))
            else:
                cur_headline = item['text_' + str(4)]
                text = self.tokenizer(cur_headline, truncation=True, max_length=self.max_length)
                headlines.append(text['input_ids'])
                prices.append(np.array(item[['EMA12_' + str(4),'EMA26_' + str(4), 'Signal_Line_' + str(4), 'MACD_Histogram_' + str(4), 'MACD_' + str(4)]].tolist()))
            # Here we can take some inspiration from the previous paper
            return {'input_ids':headlines, 'labels':label, 'prices':np.stack(prices, axis=0)}
        elif self.use_images and self.use_prices:
            if self.use_lag:
                for i in range(self.lag_period):        
                    prices.append(np.array(item[['EMA12_' + str(i),'EMA26_' + str(i), 'Signal_Line_' + str(i), 'MACD_Histogram_' + str(i), 'MACD_' + str(i)]].tolist()))
                pixels = self.graphs[idx]
            else:
                prices.append(np.array(item[['EMA12_' + str(4),'EMA26_' + str(4), 'Signal_Line_' + str(4), 'MACD_Histogram_' + str(4), 'MACD_' + str(4)]].tolist()))
                pixels = self.graphs[idx][4, :, :, :]
            return {'labels':label, 'prices':np.stack(prices, axis=0), 'pixels':pixels}
        elif self.use_tweets:
            if self.use_lag:
                for i in range(self.lag_period):
                    cur_headline = item['text_' + str(i)]
                    text = self.tokenizer(cur_headline, truncation=True, max_length=self.max_length)
                    headlines.append(text['input_ids'])
            else:
                cur_headline = item['text_' + str(4)]
                text = self.tokenizer(cur_headline, truncation=True, max_length=self.max_length)
                headlines.append(text['input_ids'])
            return {'input_ids':headlines, 'labels':label}
        elif self.use_images:
            if self.use_lag:
                pixels = self.graphs[idx]
            else:
                pixels = self.graphs[idx][4, :, :, :]
            return {'labels':label, 'pixels':pixels}
        else:
            if self.use_lag:
                for i in range(self.lag_period):
                    prices.append(np.array(item[['EMA12_' + str(i),'EMA26_' + str(i), 'Signal_Line_' + str(i), 'MACD_Histogram_' + str(i), 'MACD_' + str(i)]].tolist()))
            else:
                #prices.append(np.array(item[['EMA12_' + str(4),'EMA26_' + str(4), 'Signal_Line_' + str(4), 'MACD_Histogram_' + str(4), 'MACD_' + str(4)]].tolist()))
                prices.append(np.array(item[['MACD_' + str(0),'MACD_' + str(1), 'MACD_' + str(2), 'MACD_' + str(3), 'MACD_' + str(4)]].tolist()))
            return {'labels':label, 'prices':np.stack(prices, axis=0)}


class CSVChunkDataset():
    def __init__(self, csv_file, start_row, end_row):
        self.csv_file = csv_file
        self.start_row = start_row
        self.end_row = end_row

    def get_chunk(self):
        df = pd.read_csv(self.csv_file, skiprows=self.start_row, nrows=self.end_row - self.start_row - 1, names=['text'], lineterminator='\n')
        return df


class mosi_dataset(Dataset):
    def __init__(self, **kwargs):
        data = kwargs.pop('data', None)
        if data is None:
            raise ValueError("Data must be provided")
        self.data = data

        self.tokenizer = kwargs.get('tokenizer', None)
        self.max_length = kwargs.get('max_length', 128)
        self.lag_period = kwargs.get('lag_period', 50)
        self.use_images = kwargs.get('use_images', True)
        self.use_text = kwargs.get('use_tweets', True)
        self.use_lag = kwargs.get('use_lag', True)

        # Load data

    def __len__(self):
        return self.data['vision'].shape[0]

    def __getitem__(self, idx):
        vision = self.data['vision'][idx]
        text = self.data['raw_text'][idx]
        audio = self.data['audio'][idx]
        spectrum_label = self.data['classification_labels'][idx]
        all_text = []

        tokenized_text = self.tokenizer(text, truncation=True, max_length=self.max_length, add_special_tokens=True)

        # We want to align the text with the video? I guess with the dual encoder stream it doesn't really matter
        if spectrum_label > 0:
            label = 1
        else:
            label = 0

        return {'input_ids':np.array(tokenized_text['input_ids']),  'audio':audio, 'pixels':vision, 'labels':label}