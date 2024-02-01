import torch
from torch import nn, tensor
from einops.layers.torch import Rearrange

device = torch.device('cuda')

class vl_BERT_Wrapper(nn.Module):
    def __init__(self, model, input_dim, output_dim):
        super(CustomClassifier, self).__init__()
        self.model = model.to(device)
        self.dropout = nn.Dropout(0.1)
        self.mlp_head = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())
        patch_res = 16
        channels = 4
        patch_dim = channels * patch_res * patch_res
        self.patches = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_res, p2 = patch_res),
            nn.Linear(patch_dim, 2048))


    def forward(self, tweets, images):
        images = self.patches(images)
        # this will take a lot of time maybe?
        visual_token_type_ids = torch.ones(images.shape[:-1], dtype=torch.long).to(device)
        visual_attention_mask = torch.ones(images.shape[:-1], dtype=torch.float).to(device)
        inputs = {'input_ids':tweets.long(), 'token_type_ids':torch.ones(tweets.shape).to(device).long(), 'attention_mask':torch.zeros(tweets.shape).to(device).float()}
        # so do these even need to be defined
        inputs.update(
        {
            "visual_embeds": images,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask
        }
        )
        outputs = self.model(**inputs)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.mlp_head(pooled_output)
        return logits

class ViltWrapper(nn.Module):
    def __init__(self, vilt, input_dim, output_dim):
        super(ViltWrapper, self).__init__()
        self.vilt = vilt
        # change channel dim to 4, currently only supports 3
        self.vilt.embeddings.patch_embeddings.projection = nn.Conv2d(4, 768, kernel_size=(32, 32), stride=(32, 32))
        # what is going on
        self.vilt = self.vilt.to(device)
        self.dropout = nn.Dropout(0.1)
        self.mlp_head = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())
        patch_res = 16
        channels = 4
        patch_dim = channels * patch_res * patch_res
        self.patches = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_res, p2 = patch_res),
            nn.Linear(patch_dim, 2048))

    def forward(self, tweets, images):
        #images = self.patches(images)
        visual_token_type_ids = torch.ones(images.shape[:-1], dtype=torch.long).to(device)
        inputs = {'input_ids':tweets.long(), 'token_type_ids':torch.ones(tweets.shape).to(device).long(), 'attention_mask':torch.zeros(tweets.shape).to(device).float()}
        inputs.update(
        {
            "pixel_values": images.cuda(),
            "pixel_mask": visual_token_type_ids.cuda(),
        }
        )
        outputs = self.vilt(**inputs)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.mlp_head(pooled_output)
        return logits


class roberta_mlm_wrapper(nn.Module):
    def __init__(self, roberta, input_dim=768, output_dim=512):
        """
        Simple pretraining wrapper for the roberta model
        """
        super(roberta_mlm_wrapper, self).__init__()
        self.roberta = roberta
        self.mlm_output_head = nn.Linear(input_dim, 1)

    def forward(self, **inputs):
        intermediate_val = self.roberta(**inputs)
        # project the last hidden state to the dimension of one
        outputs = self.mlm_output_head(intermediate_val['last_hidden_state'])
        return outputs.squeeze(dim=2) 