from meant import meant
import torch
import json
import torchvision.transforms as transforms

new = meant(126, 126, 4, 224, 224, 16, 3, 2)

"""
# this is our long range encoding
image = torch.randn((3, 3, 3, 224, 224))

from PIL import Image


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Open the image file
image_actual = Image.open("/home/benjamin/Desktop/ml/michinaga_extensions/src/dataUtils/graphs/macd/AAPL/1999-12-09.png")
yuh = transform(image_actual)
print(yuh.shape)

price = torch.randn((3, 3, 196, 4))
text = torch.randn((3, 3, 196, 126))
what = new.forward(text, image, price)
print(what)
"""
"""
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", return_tensors = "pt", use_fast=False)
bertweet = AutoModel.from_pretrained("vinai/bertweet-base")

tweet_file = open('/home/benjamin/Desktop/ml/michinaga_extensions/src/dataUtils/newTweets/AAPL/2022-04-10.json')
text = tweet_file.readlines()
for t in range(len(text)):
    tweet_dict = json.loads(text[t])
    input = tweet_dict['text']
    vector = tokenizer.encode(input)
    
   # print(len(vector['input_ids']))

print(bertweet.embeddings)

print(bertweet.embeddings(torch.tensor([vector])).shape)
"""

"""
import torch
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding, broadcat

pos_emb = RotaryEmbedding(
    dim = 32,
    freqs_for = 'pixel',
    max_freq = 8
)

# queries and keys for frequencies to be rotated into

q = torch.randn(1, 8, 256, 64)
k = torch.randn(1, 8, 256, 64)

# get frequencies for each axial
# -1 to 1 has been shown to be a good choice for images and audio
freqs_h = pos_emb(torch.linspace(-1, 1, steps = 256), cache_key = 256)
freqs_w = pos_emb(torch.linspace(-1, 1, steps = 256), cache_key = 256)
print('h', freqs_h.shape)
print('w', freqs_w.shape)
# concat the frequencies along each axial
# broadcat function makes this easy without a bunch of expands

freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)

# rotate in frequencies

q = apply_rotary_emb(freqs, q)
k = apply_rotary_emb(freqs, k)
"""