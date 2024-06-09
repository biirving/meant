import requests
import csv
import pandas as pd
import numpy as np
import torch
import json
import time


# a file for s and p 500 data collection!
# the nasdaq tickers (as of Feb 1, 2023)
nasdaq_names = np.load('nasdaq_tickers.npy')
key = 'ONRD3KINP1JRCPSC'

new_price_data = {}


# For my first forray, I will measure against the S&P500

sp500arr = np.loadtxt("constituents.csv",
                 delimiter=",", dtype=str)
sp500 = sp500arr[:, 0][1:]
sp500_data = {}


"""
There are a limited number of API calls per 5 second

"""

progress = torch.load('progress.pt')
start = progress.item()

count = 0
for name in sp500[0:]:
    # sleep for 0.3 seconds between calls, 5 per second
    time.sleep(13)
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=' + name + '&outputsize=full&apikey=' + key
    r = requests.get(url)
    data = r.json()
    price_data = None
    count = 0
    print('------------------------------------------ New Ticker ' + name + '------------------------------------------')
    key = 'Time Series (Daily)'
    # the oldest points will be at the back
    if(key in data.keys()): 
        for day in data['Time Series (Daily)']:
            today = data['Time Series (Daily)'][day]
            onThisDay = torch.tensor([float(today['1. open']), float(today['2. high']), float(today['3. low']),
            float(today['5. adjusted close']), float(today['6. volume'])])
            if(price_data is None):
                price_data = onThisDay.view(1, 5)
            else:
                price_data = torch.concat((price_data, onThisDay.view(1, 5)), 0)
        # saving all of the prices just in case
        torch.save(price_data, '/home/benjamin/Desktop/ml/michinaga/src/dataUtils/sp500/' + name + '.pt')
        sp500_data[name] = price_data
    count += 1

progress = torch.tensor([count])
torch.save(progress, 'progress.pt') 

# create json object from dictionary
json = json.dumps(new_price_data)

# open file for writing, "w" 
f = open("sp500_data.json","w")

# write json object to file
f.write(json)

# close file
f.close()


