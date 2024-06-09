import torch
import json
import torchvision.transforms as transforms
import torch
import time
from PIL import Image
import numpy as np
import os

# we want the images to have 3 channels
# check that this transformation is working correctly
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# a file to preprepare the images
sp500arr = np.loadtxt("constituents.csv",
                 delimiter=",", dtype=str)

# going to start with a slice of the dataset
sp500 = sp500arr[:, 0][15:37]
# rerun this to make the condensed tensors
for tick in sp500:
    final_path = f'/work/socialmedia/stock/graphs/'
    #os.makedirs(final_path, exist_ok=True)
    # then, we iterate through the processed graphs to make them into tensors
    graph_path = '/home/irving.b/aurelian_data/graphs/' + tick + '/'
    # we only iterate through the images which have tweets
    folder_path = "/home/irving.b/aurelian_data/tweets/" + tick  # replace with the path to your folder
    files = os.listdir(folder_path)
    files.sort()
    count = 0
    # now we make the condensed files
    to_save = None
    for f in files:
        date = f.split('.')[0]
        image_path = graph_path + date + '.png'
        # the graph has to exist as well (we discount weekends in this model)
        if os.path.isfile(image_path):
            image_actual = Image.open(image_path)
            image_transformed = transform(image_actual).unsqueeze(0)
            if(to_save is None):
                to_save = image_transformed
            else:
                to_save = torch.cat((to_save, image_transformed), dim = 0)
    torch.save(to_save, final_path + tick + '.pt')
