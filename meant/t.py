from meant import meant
import torch
import json
import torchvision.transforms as transforms
import torch
from transformers import AutoModel, AutoTokenizer
import time

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large", return_tensors = "pt", use_fast=False)
bertweet = AutoModel.from_pretrained("vinai/bertweet-large")

# is there a point to even having a text_dim?
new = meant(768, 768, 4, 224, 224, 16, 3, 2, bertweet.embeddings)

# this is our long range encoding
image = torch.randn((3, 3, 3, 224, 224))

from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Open the image file
#image_actual = Image.open("/home/benjamin/Desktop/ml/michinaga_extensions/src/dataUtils/graphs/macd/AAPL/1999-12-09.png")
#yuh = transform(image_actual)
#print(yuh.shape)

#tweet_file = open('../../michinaga_extensions/src/dataUtils/newTweets/AAPL/2022-04-10.json')
#text = tweet_file.readlines()
#for t in range(len(text)):
#    tweet_dict = json.loads(text[t])
#    input = tweet_dict['text']
#    
# should we have the lag period contained within each text line?
# so we have a single tweet from each day over the lag period?
text = ["holy function apple is so", "dub", "hallo", "what", "stop", "stop the madness how does this work akjdhladshf aljsdhf ads asd "]
text2 = ["apple sucked today", "dub", "fuck", "pattern", "p", "stop the madness how does this work"]
# so each of these are groups of tweets over a lag period?
text3 = ["apple was awesome today", "dub", "shite", "o", "whatas", "stop the madness how does this work"]

text4 = ["holy function apple is so", "dub", "hallo", "what", "stop", "stop the madness how does this work akjdhladshf aljsdhf ads asd "]
text5 = ["apple sucked today", "dub", "fuck", "pattern", "p", "stop the madness how does this work"]
# so each of these are groups of tweets over a lag period?
text6 = ["apple was awesome today", "dub", "shite", "o", "whatas", "stop the madness how does this work"]

text7 = ["holy function apple is so", "dub", "hallo", "what", "stop", "stop the madness how does this work akjdhladshf aljsdhf ads asd "]
text8 = ["apple sucked today", "dub", "fuck", "pattern", "p", "stop the madness how does this work"]
# so each of these are groups of tweets over a lag period?
text9 = ["apple was awesome today", "dub", "shite", "o", "whatas", "stop the madness how does this work"]


#pract_encode = "OPTION WATCHLIST + CHARTS \ud83c\udfaf\ud83d\udcc8\n\n\ud83d\udecd $LULU | Calls &gt; 380.35 ; Puts &lt; 368.55\n\ud83d\udc65 $FB | Calls &gt; 225 ; Puts &lt; 219\n\ud83d\udcf1 $AAPL | Calls &gt; 172 ; Puts &lt; 168.92\n\ud83d\udcc8 $SPY | Calls &gt; 450.40 ; Puts &lt; 443.50\n\n75 LIKES FOR A BONUS PLAY \u2705 https:\/\/t.co\/Br2blT8tLK"
#print('pract', torch.tensor([tokenizer.encode([pract_encode])]))

text = torch.tensor([tokenizer.encode(text)])
text2 = torch.tensor([tokenizer.encode(text2)])
text3 = torch.tensor([tokenizer.encode(text3)])
text4 = torch.tensor([tokenizer.encode(text4)])
text5 = torch.tensor([tokenizer.encode(text5)])
text6 = torch.tensor([tokenizer.encode(text6)])
text7 = torch.tensor([tokenizer.encode(text7)])
text8 = torch.tensor([tokenizer.encode(text8)])
text9 = torch.tensor([tokenizer.encode(text9)])

price = torch.randn((3, 3, 196, 4))
#print('text shape', text.shape)
#print('text shape', text2.shape)
text_input = torch.cat((text, text2, text3, text4, text5, text6, text7, text8, text9), dim = 0)
#print(text_input)
#print('close', text_input.shape)
#text_inptu = torch.randn(3, 3, )
#print(bertweet.forward(text_input))

#text_input = torch.randn((3, 5))
#text = torch.randn((1, 7)).int()

# Measure the time taken to perform a forward pass
start_time = time.time()
with torch.no_grad():
    what = new.forward(text_input.int(), image, price)
end_time = time.time()
# Print the elapsed time
print(f"Forward pass took {end_time - start_time:.4f} seconds")
print(what.shape)


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