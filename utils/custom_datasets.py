from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import emoji
from datasets import load_dataset
import itertools
from PIL import Image
import torchvision.transforms as transforms
import sys
import numpy as np

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