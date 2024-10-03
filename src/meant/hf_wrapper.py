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
        channels = 3 
        patch_dim = channels * patch_res * patch_res
        self.patches = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_res, p2 = patch_res),
            nn.Linear(patch_dim, 2048))


    def forward(self, **kwargs):
        tweets = kwargs.get('input_ids')
        images = kwargs.get('pixels')
        prices = kwargs.get('prices')

        attention_mask = kwargs.get('attention_mask')
        images = self.patches(images)
        visual_token_type_ids = torch.ones(images.shape[:-1], dtype=torch.long).to(device)
        visual_attention_mask = torch.ones(images.shape[:-1], dtype=torch.float).to(device)
        inputs = {'input_ids':tweets.long(), 'token_type_ids':torch.ones(tweets.shape).to(device).long(), 'attention_mask':torch.zeros(tweets.shape).to(device).float(), 'prices':prices}
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
        self.vilt.embeddings.patch_embeddings.projection = nn.Conv2d(3, 768, kernel_size=(32, 32), stride=(32, 32))
        self.vilt = self.vilt.to(device)
        self.dropout = nn.Dropout(0.1)
        self.mlp_head = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())
        patch_res = 16
        channels = 4
        patch_dim = channels * patch_res * patch_res
        self.patches = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_res, p2 = patch_res),
            nn.Linear(patch_dim, 2048))

    def forward(self, **kwargs):
        tweets = kwargs.get('input_ids')
        prices = kwargs.get('prices')
        images = kwargs.get('pixels')


        attention_mask = kwargs.get('attention_mask')
        inputs = {'input_ids':tweets.long(), 'token_type_ids':torch.ones(tweets.shape).to(device).long(), 'prices':prices, 'attention_mask':attention_mask}
        inputs.update(
        {
            "pixel_values": images.cuda(),
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
    def forward(self, **kwargs):
        tweets = kwargs.get('input_ids')
        attention_mask = kwargs.get('attention_mask')
        # pass attention mask forward
        inputs_actually = {'input_ids':tweets, 'attention_mask':attention_mask}
        outputs = self.bertweet(**inputs_actually)
        # do we want to process the pooler output, or the sequence output
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
        outputs = self.mlm_output_head(intermediate_val['last_hidden_state'])
        return outputs.squeeze(dim=2) 

class meant_language_pretrainer(nn.Module):
    def __init__(self, num_encoders, mlm_input_dim, embedding, lm_head, lag=5, text_dim=768, num_heads=8):
        super(meant_language_pretrainer, self).__init__()
        self.embedding = nn.ModuleList([embedding])
        self.languageEncoders = nn.ModuleList([languageEncoder(text_dim, num_heads, flash=True)])
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
        self.decoder = decoder

    def forward(self, images):
        images = self.patchEmbed(images)
        for encoder in self.visionEncoders:
            images = encoder.forward(images)
        batch_size, sequence_length, num_channels = images.shape
        height = width = math.floor(sequence_length**0.5)
        sequence_output = images.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
        return self.decoder(sequence_output)
