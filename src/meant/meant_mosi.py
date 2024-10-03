import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
from src.meant.attention import attention
from src.meant.flash_attention import flash_attention
from src.meant.xPosAttention import xPosAttention
from src.meant.xPosAttention_flash import xPosAttention_flash
from src.meant.temporal import temporal
from src.meant.temporal_new import temporal_2
from src.meant.timesformer_pytorch import TimeSformer
from src.utils.rms_norm import RMSNorm 
from rotary_embedding_torch import RotaryEmbedding
import math
from transformers import AutoModel, AutoTokenizer
from positional_encodings.torch_encodings import PositionalEncodingPermute1D, Summer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# okay, lets run these experiments
# because
MAX_SEQ_LENGTH = 3333

# should the vision encoder encode temporal information?
class visionEncoder(nn.Module):
    # we should pretrain the patch embeddings, right?
    def __init__(self, dim, num_heads, flash=False):
        """
        The initial encoder for extracting relevant features from the multimodal input.
        """
        super(visionEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        # so, the xPos embeddings will focus on the pixel case
        self.posEmbed = RotaryEmbedding(
            dim = math.floor(dim/num_heads/2),
            freqs_for='pixel')
        
        if flash and ampere:
            atten = flash_attention(num_heads, dim, self.posEmbed) 
        else:
            if flash and not ampere and torch.cuda.is_available():
                print(f"The {cuda_device_name} GPU is not from the Ampere series or later. Flash attention not supported.")
            elif flash:
                print('Cuda not supported. Cannot use flash attention.')
            atten = attention(num_heads, dim, self.posEmbed)

        self.encode = nn.ModuleList([nn.LayerNorm(dim), 
                                    nn.Linear(dim, dim), 
                                    atten, 
                                    nn.LayerNorm(dim), 
                                    nn.Linear(dim, dim)])
        self.encode2 = nn.ModuleList([nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), RMSNorm(dim), nn.Linear(dim, dim)])

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

    def forward(self, input):
        inter = input
        for mod in self.encode:
            inter = mod(inter)
        inter = inter + input
        final_resid = inter
        for mod in self.encode2:
            inter = mod(inter)
        # then another residual connection before the output is processed
        return inter + final_resid

# separate encoder module, because might make changes to structure
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
            dim = 30,
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
class temporalEncoder(nn.Module):
    def __init__(self, dim, num_heads, lag, use_rot_embed=True):
        super(temporalEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        if use_rot_embed:
            self.xPos = RotaryEmbedding(
                dim = 37,
                use_xpos = True,   # set this to True to make rotary embeddings extrapolate better to sequence lengths greater than the one used at training time
                #xpos_scale_base=2
            )
        else:
            self.xPos = None
        # positional encoding


        #self.temp_cls_token = nn.Parameter(torch.randn)
        # Definitely matters how this is initialized
        #self.temp_embedding = nn.Parameter(torch.randn(1, lag, dim))
        self.temp_encode = nn.ModuleList([nn.LayerNorm(dim), 
                                            nn.Linear(dim, dim), 
                                            temporal(num_heads, dim, rot_embed=self.xPos), 
                                            nn.LayerNorm(dim), 
                                            nn.Linear(dim, dim)])

        
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

    def forward(self, x):
        b, l, d = x.shape
        # the repeated temporal embedding is not good
        # how does this contribute to the batch being 
        # the same value for all of them?
        #temp_embed = repeat(self.temp_embedding, '1 l d -> b l d', b = b)
        #x += temp_embed
        for mod in self.temp_encode:           
            x = mod(x)
        return x


class meant_mosi(nn.Module):
    # what happens if I try to just use the embedding layer anyways?
    def __init__(self, text_dim, image_dim, height, width, patch_res, lag, num_classes, audio_embeddings=None, embedding=None, flash=False, num_heads= 8, num_encoders = 1, channels=3, seq_len=512):
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
        super(meant_mosi, self).__init__()

        # recent additions for editing purposes
        self.lag = lag
        self.text_dim = text_dim
        self.image_dim = image_dim

        # concatenation strategy: A simple concatenation to feed the multimodal information into the encoder.
        self.dim = text_dim  
        self.num_heads = num_heads

        # for the image component of the encoder
        self.channels = channels
        self.patch_dim = self.channels * patch_res * patch_res
        self.n = int((height * width) / (patch_res ** 2))

        if embedding is not None:
            self.embedding = nn.ModuleList([embedding])
        else:
            self.embedding = None
        
        if audio_embeddings is not None:
            self.audio_embeddings = nn.ModuleList([embedding])
        else:
            self.embedding = None

        # classification token for the image component. Will be passed to the temporal attention mechanism
        #self.cls_token = nn.Parameter(torch.randn(1, lag, 1, image_dim))

        # the patch embedding for the image
        # we have to apply it to every image in the lag period
        # c = channel
        # h = height
        # w = width
        # b = batch
        # f = the number of frames we are processing
        self.patchEmbed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_res, p2 = patch_res),
            nn.Linear(self.patch_dim, image_dim))

        self.visionEncoders = nn.ModuleList([visionEncoder(image_dim, num_heads, flash=flash) for i in range(num_encoders)])
        # Instead of using a vision encoder, use TimeSFormer for your 
        # Image encoding?
        self.timesformer = TimeSformer(dim=image_dim, 
        image_size=20,
        patch_size=1,
        num_frames=lag, 
        num_classes=num_classes,
        depth=num_encoders,
        heads=8,
        dim_head=64,
        attn_dropout = 0.1,
        ff_dropout = 0.1,
        channels=1
        )

        self.languageEncoders = nn.ModuleList([languageEncoder(self.text_dim, num_heads, flash=flash) for i in range(num_encoders)])
        self.visionEncoders = nn.ModuleList([visionEncoder(image_dim, num_heads, flash=flash) for i in range(num_encoders)])
        self.patchEmbed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_res, p2 = patch_res),
            nn.Linear(self.patch_dim, image_dim))

        self.visinoEncoders = nn.ModuleList([])
    
        self.embedding = nn.ModuleList([embedding])

        self.lang_proj = nn.Sequential(nn.Linear(self.text_dim, 1), nn.LayerNorm(1), nn.GELU())
        self.image_proj = nn.Sequential(nn.Linear(981, 1), nn.LayerNorm(1), nn.GELU())
        self.temporal_encoding = nn.ModuleList([temporalEncoder(self.text_dim, num_heads, lag, use_rot_embed=False)])
        self.lang_prep = nn.Sequential(nn.Linear(self.text_dim, self.text_dim), nn.LayerNorm(self.text_dim), nn.GELU(), nn.Linear(self.text_dim, 1), nn.Softmax(dim=2))
        self.lang_red = nn.Sequential(nn.Linear(self.text_dim, 5), nn.LayerNorm(5), nn.GELU())
        self.other_dim=1536
        self.mlpHead = nn.ModuleList([nn.LayerNorm(self.other_dim), nn.Linear(self.other_dim, num_classes), nn.Sigmoid()])

        self.seq_len = seq_len

        self.audio_embedding_dim=130
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.audio_embedding_dim, nhead=2, batch_first=True, dim_feedforward=512)
        self.audio_emb = nn.Embedding(num_embeddings=1, embedding_dim=self.audio_embedding_dim)
        self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3,enable_nested_tensor=False)

    def prepend_cls(self, inputs, masks):
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = self.audio_emb(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, self.audio_embedding_dim)
        outputs = torch.cat((cls_emb, inputs), dim=1)
        
        cls_mask = torch.zeros(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks

    def forward(self, **kwargs):
        words = kwargs.get('input_ids')
        labels = kwargs.get('labels')
        prices = kwargs.get('prices')
        images = kwargs.get('pixels')
        pixel_mask = kwargs.get('pixel_mask')
        audio = kwargs.get('audio')
        audio_attention_mask = kwargs.get('audio_mask')

        attention_mask = kwargs.get('attention_mask')
        _batch = images.shape[0]

        for mod in self.embedding:
            words = mod(words, attention_mask) 
        
        audio, audio_masks = self.prepend_cls(audio, audio_attention_mask)
        pos_enc = Summer(PositionalEncodingPermute1D(audio.shape[1])).cuda()
        
        audio = pos_enc(audio)
        audio = self.audio_encoder(audio, src_key_padding_mask=audio_attention_mask.bool())
        for encoder in self.languageEncoders:
            words = encoder.forward(words, attention_mask=attention_mask)

        #words = rearrange(words, '(b l) s d-> b l d s', b = _batch)
        images = self.timesformer.meant_forward(images.unsqueeze(dim=2).unsqueeze(dim=3).half()) #, mask=pixel_mask)
        images = images[:, 1:, :] 
        images = rearrange(images, 'b (l s) d-> b l s d', l=self.lag)

        #act_seq_len = words.shape[3]
        #if act_seq_len < self.seq_len:
        #    padding = self.seq_len - act_seq_len 
        #    words = nn.functional.pad(words, (0, padding))
        # mosi mosi mosi mosi
        #words = self.lang_proj(words).squeeze(dim=2)
        temporal = torch.max(words, dim=1).values
        temporal = temporal.half()
        #for encoder in self.temporal_encoding:
        #    temporal = encoder.forward(temporal)
        temporal = temporal.squeeze(dim=1)
        temporal = torch.cat((temporal, images[:, -1, :].squeeze(dim=1)), dim=1)
        for mod in self.mlpHead:
            temporal = mod(temporal)
        return temporal