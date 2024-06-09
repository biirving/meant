import torch
import os
import numpy as np
import datetime as dt
import sys



"""
FILE FOR SMOTE (Synthetic Minority Over-sampling Technique)
"""

# assuming this is for the 5-day lag period?
total_labels = 0
total_pos = 0

graphs_preprocessed = '/work/nlp/b.irving/stock/graphsPreprocessed'
tweets_directory = '/work/nlp/b.irving/stock/tweets' 

tickers = []

def generate_dates(start_date, end_date, interval_days):
    delta = dt.timedelta(days=interval_days)
    current_date = start_date
    dates = []
    while current_date <= end_date:
        dates.append(current_date)
        current_date += delta
    return dates

# Example usage
start_date = dt.date(2022, 4, 10)
end_date = dt.date(2023, 4, 10)
interval_days = 1
dates = generate_dates(start_date, end_date, interval_days)

sp500arr = np.loadtxt("constituents.csv",
                    delimiter=",", dtype=str)
sp500 = sp500arr[:, 0][1:37]

# iterate through each of the relevant tickers
for tick in sp500:
    print(tick)
    # Construct the full file path
    map_dates = os.path.join(graphs_preprocessed, tick)
    tweets_file_path = os.path.join(tweets_directory, tick + '.pt')
    
    valid_dates = os.listdir(map_dates)
    indices = []
    index = 0
    for day in dates:
        if (str(day) + '.pt' in valid_dates):
            indices.append(index)
        index += 1

    tweets = torch.load(tweets_file_path)
    print(tweets.shape)
    print(indices)
    try:
        tweets_2 = tweets[indices].cpu()
    
    # controversial decision - how were these offset originally? Only effects one of the tickers, so should be fine?
    # can easily extend this to some arbitrary number of stocks
    except IndexError:
        tweets_2 = tweets[:249].cpu()
    print(tweets_2.shape)
    torch.save(tweets_2, '/work/nlp/b.irving/stock/tweets_2/' + tick + '.pt')