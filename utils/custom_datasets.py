from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import emoji

def replace_emojis_with_text(text):
    return emoji.demojize(text) 

# custom datasets classes for loading different tasks
class mlm_dataset(Dataset):
    def __init__(self, dataset, split, tokenizer, max_length=512, mlm_probability=0.15):
        if split is not None:
            self.dataset = dataset[split]
        else:
            self.dataset = dataset
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = replace_emojis_with_text(self.dataset[idx])
        tokens = self.tokenizer(text, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
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

    