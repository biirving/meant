import snscrape.modules.twitter as twitter
import snscrape
import pandas as pd
import datetime as dt
import numpy as np
import os
from os.path import exists

"""
Class to scrape tweets that correspond to stocks in the S&P500.
"""
# the tweets will include weekends? Better to include or not? I don't know
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
sp500 = sp500arr[:, 0][230:]

# Create directory outside the loop
if not os.path.exists('/home/benjamin/Desktop/ml/michinaga_extensions/src/dataUtils/newTweets'):
    os.mkdir('/home/benjamin/Desktop/ml/michinaga_extensions/src/dataUtils/newTweets')

count = 0
num_tweets = 10

# how many tweets to gather for each day
tweets = []


count = 0
for ticker in sp500[0:]:
    print(ticker)
    hashtag = '$' + str(ticker)
    for date in range(len(dates) - 1):
        start_date  = dates[date]
        end_date = dates[date+1]
        if(count ==0):
            count += 1
        tweets = []
        try:
            call = twitter.TwitterSearchScraper(hashtag + ' until:' + str(end_date) + ' since:' + str(start_date), mode = twitter.TwitterSearchScraperMode.TOP).get_items()
            for tweet in call:
                tweets.append([start_date, tweet.rawContent, tweet.likeCount])
                if(len(tweets) >= num_tweets):
                    break
        except: 
            continue
        tweets_pd = pd.DataFrame(tweets, columns=["date", "text", "likeCount"])
        if(not exists('/home/benjamin/Desktop/ml/michinaga_extensions/src/dataUtils/newTweets/' + ticker)):
            os.mkdir('/home/benjamin/Desktop/ml/michinaga_extensions/src/dataUtils/newTweets/' + ticker)
            print('new dir')
        tweets_pd.to_json('newTweets/' + ticker + '/' + str(dates[date]) + '.json', orient = 'records', lines=True)