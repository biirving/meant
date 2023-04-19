from meant import meant
import torch

new = meant(100, 100, 4, 224, 224, 16, 3, 2)

# this is our long range encoding
image = torch.randn((1, 3, 3, 224, 224))
price = torch.randn((1, 3, 4))
text = torch.randn((1, 3, 100))
what = new.forward(text, image, price)
print(what)
