import sys, os
import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
from src.meant.attention import attention
from src.meant.flash_attention import flash_attention
from src.meant.xPosAttention import xPosAttention
from src.meant.xPosAttention_flash import xPosAttention_flash
from src.meant.temporal import temporal
from rotary_embedding_torch import RotaryEmbedding
import math
from transformers import AutoModel, AutoTokenizer
from src.utils.rms_norm import RMSNorm
from torch.distributions import Normal
from src.meant.temporal_new import temporal_2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# okay, lets run these experiments
# because
MAX_SEQ_LENGTH = 3333

# separate encoder module, because might make changes to structure
# should we have temporal encoding baked into each encoder?

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the name of the CUDA device
    cuda_device_name = torch.cuda.get_device_name(0)

    # Check if the device name contains "Ampere" or a later architecture
    if "Ampere" in cuda_device_name or "A100" in cuda_device_name:
        ampere = True
    else:
        ampere = False
else:
    print("CUDA is not available on this system.")
    ampere = False

class languageEncoder(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0, flash=False):
        """
        Encoder to extract language inputs. Virtually identical to visual encoder, except that it utitilizes 
        the xPos embeddings rather than the base rotary embeddings
        """
        super(languageEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.xPos = RotaryEmbedding(
            dim = 48,
            use_xpos = True,   # set this to True to make rotary embeddings extrapolate better to sequence lengths greater than the one used at training time
            #xpos_scale_base=2
        )
        if flash and ampere:
            attention = xPosAttention_flash(num_heads, dim, self.xPos) 
        else:
            if flash and not ampere and torch.cuda.is_available():
                print(f"The {cuda_device_name} GPU is not from the Ampere series or later. Flash attention not supported.")
            elif flash:
                print('Cuda not supported. Cannot use flash attention.')
            attention = xPosAttention(num_heads, dim, self.xPos)
        self.encode = nn.ModuleList([nn.LayerNorm(dim), 
                                    nn.Linear(dim, dim), 
                                    attention, 
                                    nn.LayerNorm(dim), 
                                    nn.Dropout(dropout),
                                    nn.Linear(dim, dim)])
        self.encode2 = nn.ModuleList([nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), RMSNorm(dim), nn.Dropout(), nn.Linear(dim, dim)])
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, input, attention_mask=None):
        inter = input
        for mod in self.encode:
            if type(mod).__name__ == 'xPosAttention':
                inter = mod(inter, attention_mask)
            else:
                inter = mod(inter)
        inter = inter + input
        final_resid = inter
        for mod in self.encode2:
            inter = mod(inter)
        return inter + final_resid

# how does this scale to deal with an arbitrary lag period
# lets make this multimodal temporal model, shall we?
# how does this scale to deal with an arbitrary lag period
class temporalEncoder(nn.Module):
    def __init__(self, dim, num_heads, lag, sequence_length=512, use_rot_embed=True):
        super(temporalEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        if use_rot_embed:
            self.xPos = RotaryEmbedding(
                dim = 48,
                use_xpos = True,   # set this to True to make rotary embeddings extrapolate better to sequence lengths greater than the one used at training time
                #xpos_scale_base=2
            )
        else:
            self.xPos = None
        # positional encoding


        self.layer1 = nn.Linear(dim, dim)                        
        self.temp_encode = temporal(num_heads, dim, sequence_length=sequence_length, rot_embed=self.xPos)
        self.layer2 = nn.Linear(dim, dim)
        self._initialize_weights()


    # God dammit
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attention_mask=None):
        #b, l, s, d = x.shape
        x = self.layer1(x)
        x = self.temp_encode(x, attention_mask=attention_mask)
        x = self.layer2(x)
        return x

# the meant model without image inputs
class meant_tweet(nn.Module):
    def __init__(self, text_dim, price_dim, lag, num_classes, embedding, flash=False, z_dim=4, num_heads=8, num_encoders = 1, channels=4, sequence_length=512):
        """
        Args
            dim: The dimension of the input to the encoder
            num_heads: The number of attention heads for the attention mechanism
            height: The height of the images being processed
            width: The width of the images being processed
            patch_res: The dimension of each image patch
            channels: The number of channels in the images being processed
            num_classes: The number of classes for the mlp output head
        
        returns: A classification vector, of size num_classes
        """
        super(meant_tweet, self).__init__()
        
        # concatenation strategy: A simple concatenation to feed the multimodal information into the encoder.
        self.dim = text_dim + price_dim
        self.num_heads = num_heads

        # pretrained language embedding from hugging face model
        # what if we have already used the flair embeddings
        self.embedding = nn.ModuleList([embedding])

        # classification token for the image component. Will be passed to the temporal attention mechanism
        self.languageEncoders = nn.ModuleList([languageEncoder(text_dim, num_heads, flash=flash, dropout=0.1) for i in range(num_encoders)])

        self.temporal_encoding = nn.ModuleList([temporalEncoder(self.dim, num_heads, lag, sequence_length=sequence_length)])
        self.temp_proj = nn.Linear(self.dim, 1)


        # I think this might be the way to extract our 'important' tweets
        self.lang_prep = nn.Sequential(nn.Linear(text_dim, text_dim), nn.LayerNorm(text_dim), nn.GELU(), nn.Linear(text_dim, 1), nn.Softmax(dim=2))
        self.lang_red = nn.Sequential(nn.Linear(text_dim, 5), nn.LayerNorm(5), nn.GELU())
        #self.lang_proj = nn.Sequential(nn.Linear(sequence_length, 1), nn.LayerNorm(1), nn.GELU())

        #self.lang_proj = nn.LSTM(input_size = seq_len, hidden_size = 1, num_layers = 1)

        # output head
        # Here at the end

        #self.mlpHead = nn.ModuleList([nn.LayerNorm(sequence_length), nn.Linear(sequence_length, num_classes), nn.Sigmoid()])
        #self.other_dim=1541
        self.mlpHead = nn.ModuleList([nn.LayerNorm(self.dim), nn.Linear(self.dim, num_classes), nn.Sigmoid()])
        self.lag = lag
        self.seq_len = sequence_length 


        # We use tanh because we want 0 centering for our distributions
        self.mean_weight = nn.Sequential(nn.Linear(4, 1), nn.Tanh())
        self.vars_weight = nn.Sequential(nn.Linear(4, 1), nn.Tanh())


        self.z_mat = nn.Linear(z_dim, 1)

        # Its a woozy its a wazi
        self.latent_dim = 10

    # How do we incorporate our previous price outputs
    # I don't want the muddied waters of my previous outputs
    def conditional_dist(self, x_t):
        # be simple
        # be direct
        # The inital mean is a little strange right here (we want ones)
        # Got to line up these dimensions
        b = x_t.shape[0]
        inf_score = torch.zeros(b, 1).cuda()
        for i in range(self.lag):
            print(x_t[:, i].shape)
            print(inf_score.shape)

            # Recurrently updating the informational scored based on the previous day's prices and label
            mean = self.mean_weight(torch.cat((x_t[:, i], inf_score)))
            var = self.vars_weight(torch.cat((x_t[:, i], inf_score)))
            dist = Normal(mean, var.exp())
            # Reparameterization Trick
            z_cur = dist.rsample()
            print(z_cur.shape)
            inf_score = self.z_mat(torch.cat((z_cur, x_t[:, i])))
        return inf_score



    # This is the model which gave me stocknet
    # but I can do even better
    def forward(self, **kwargs):
        tweets = kwargs.get('input_ids')
        prices = kwargs.get('prices').half()
        attention_mask = kwargs.get('attention_mask')

        _batch = tweets.shape[0]
        words = tweets.view(_batch * self.lag, tweets.shape[2])
        attention_mask = attention_mask.view(_batch * self.lag, attention_mask.shape[2])
        for mod in self.embedding:
            words = mod(words)

        for encoder in self.languageEncoders:
            words = encoder.forward(words, attention_mask=attention_mask)

        # so now, we have encoded our tweets:
        # How to we extract the inter-day dependencies from them?

        words = rearrange(words, '(b l) s d -> b l s d', b = _batch)
        # So: The language encodings I have here should work fine (Even though they are a bunch of tweets combined...)
        # There are sep tokens!
        # We should figure out which tweets are important, and which ones aren't (in the sequence, and in the lag period)
        # We should use a softmax in our lang projection!

        # This detects what we should use in a sequence...
        # But what about which lag days are important?

        # mean pooling works way better
        # Does it? Does it really?
        #words = torch.mean(words, dim=2)



        #temp_atten_mask = torch.ones((words.shape[0], words.shape[1], self.seq_len), dtype=torch.float)

        act_seq_len = words.shape[2]
        if act_seq_len < self.seq_len:
            padding = self.seq_len - act_seq_len 
            # We want to mask these 0 weights before feeding to an attention mechanism as well
            words = nn.functional.pad(words, (0, 0, 0, padding))
           #words = nn.functional.pad(words, (0, 0, 0, padding))
           #temp_atten_mask[:, :, act_seq_len:] = 0.0


        word_intelligence = self.lang_prep(words)
        words = torch.matmul(words.transpose(2, 3), word_intelligence)
        words = words.squeeze(dim=-1)

        # sequence extraction complete!

        #words = rearrange(words, '(b l) d -> b l d', b = _batch)

        # This should be the same everytime
        #temporal = torch.cat((words, repeat(prices, 'b l p -> b l s p', s=self.seq_len)), dim=3)


        # Should we feed the prices both to the temporal encoder and withdraw from the distribution
        # first off: the words matrix should be projected into a lower dimension so the prices have more weights
        ##words = self.lang_red(words)
        temporal = torch.cat((words, prices), dim = 2)
        temporal = temporal.half()

        # see if it works better to just feed the tweets forward?
        #words = tweets.view(_batch * self.lag, tweets.shape[2])
        # so this is the temporal score which finds relationships between the
        # auxiliary days and the target day
        for encoder in self.temporal_encoding:
            temporal = encoder.forward(temporal, prices)
            # then do we sample from a distribution?

        inf_score = self.conditional_dist(prices)
        print(temporal.shape)
        print(inf_score.shape)
        
        # Again: Project the temporal data do contribute a balanced amount of information
        for mod in self.mlpHead:
            temporal = mod(temporal, inf_score)

        return temporal
