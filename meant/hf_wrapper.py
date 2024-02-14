import torch
from torch import nn, tensor
from einops.layers.torch import Rearrange

device = torch.device('cuda')

class vl_BERT_Wrapper(nn.Module):
    def __init__(self, model, input_dim, output_dim):
        super(vl_BERT_Wrapper, self).__init__()
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


class bertweet_wrapper(nn.Module):
    def __init__(self, bertweet, input_dim, output_dim):
        super(bertweet_wrapper, self).__init__()
        self.bertweet = bertweet
        self.dropout = nn.Dropout(0.1)
        self.mlp_head = nn.Sequential(nn.LayerNorm(input_dim), nn.GELU(), nn.Linear(input_dim, output_dim), nn.Sigmoid())
    def forward(self, tweets):
        # pass attention mask forward
        attention_masks = torch.tensor([[1 if token_id != 1 else 0 for token_id in seq] for seq in tweets]).cuda()
        inputs_actually = {'input_ids':tweets, 'attention_mask':attention_masks}
        outputs = self.bertweet(**inputs_actually)
        # do we want to process the pooler output, or the sequence output
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.mlp_head(pooled_output)
        return logits

# so just have to use bertweet embeddings because of pretokenization?
# YES
#class bert_wrapper(nn.Module):

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

class meant_language_pretrainer(nn.Module):
    def __init__(self, num_encoders, mlm_input_dim, embedding, lm_head, lag=5, text_dim=768, num_heads=8):
        super(meant_language_pretrainer, self).__init__()
        self.embedding = nn.ModuleList([embedding])
        self.languageEncoders = nn.ModuleList([languageEncoder(text_dim, num_heads, flash=True)])
        # my mlm head has to be the same size as the vocabulary list (come on son)
        self.mlm_head = lm_head 
        self.lag=lag

    def forward(self, words):
        for mod in self.embedding:
            words = mod(words)
        for encoder in self.languageEncoders:
            words = encoder.forward(words)
        return self.mlm_head(words)

class meant_vision_pretrainer(nn.Module):
    def __init__(self, num_encoders, decoder, mlm_input_dim, patch_res=16, channels = 4, height=224, width=224, image_dim=768, num_heads=8):
        super(meant_vision_pretrainer, self).__init__()
        self.channels = channels
        self.patch_dim = self.channels * patch_res * patch_res
        self.n = int((height * width) / (patch_res ** 2))
        self.patchEmbed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_res, p2 = patch_res),
            nn.Linear(self.patch_dim, image_dim))
        self.visionEncoders = nn.ModuleList([visionEncoder(image_dim, num_heads, flash=True)])
        # what sort of lm_head do we use
        self.decoder = decoder

    # I need to set up Annika's experiments, VQA, and some other things
    def forward(self, images):
        images = self.patchEmbed(images)
        for encoder in self.visionEncoders:
            images = encoder.forward(images)
        # Reshape to (batch_size, num_channels, height, width)
        batch_size, sequence_length, num_channels = images.shape
        height = width = math.floor(sequence_length**0.5)
        sequence_output = images.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
        return self.decoder(sequence_output)
