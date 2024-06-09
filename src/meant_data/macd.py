"""
MACD will be an indicator that I initially focus on

MACD + RSI

MACD + MFI

Woe to the conquered.

Fix this dataset (where are my positive examples bruh)

"""

import numpy as np
import torch
from torch import nn, tensor
import matplotlib.pyplot as plt
import math
import ta
from ta.trend import MACD 
from ta.momentum import rsi
import pandas as pd
import datetime as dt
import os
from os.path import exists

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
#adjusted closing prices
#googl_data = torch.load('sp500/GOOG.pt').numpy().astype('double')[:, 3]
#macd, macd_signal, macd_hist = talib.MACD(np.flip(googl_data)[0:100], fastperiod = 12, slowperiod = 26, signalperiod = 9)  

#signal = talib.signal(googl_data)

def plot(x1, y1, x2, y2):
    plt.plot(x1, y1, label = 'MACD')
    plt.plot(x2, y2, label = 'Signal')
    plt.legend()
    plt.show()


class macdrsi:

    def __init__(self, tickers):
        self.tickers = tickers

    # it would be easier to have the dates that we are trying to access
    def generate_dates(self, start_date, end_date, interval_days):
        delta = dt.timedelta(days=interval_days)
        current_date = start_date
        dates = []
        string_dates = []
        while current_date <= end_date:
            if current_date.weekday() < 5:  # check if the current day is a weekday (0 = Monday, 1 = Tuesday, ..., 6 = Sunday)
                dates.append(current_date)
                date_str = current_date.strftime('%Y-%m-%d')
                string_dates.append(date_str)
            current_date += delta
        return dates, string_dates

   

    # for the multimodal processing, we should save the plots as well with names that
    # are identifiable and matchable to the data?
    # our long term data will be found in the plots.
    def plot(self, x1, y1, x2, y2, x3, y3, date, ticker):
        plt.plot(x1, y1, label='MACD')
        plt.plot(x2, y2, label='Signal')
        plt.legend()
        path = f'/home/benjamin/Desktop/ml/michinaga/src/dataUtils/graphs/macd/{ticker}'
        os.makedirs(path, exist_ok=True)
        filename = f'{path}/{date.split(".")[0]}.png'
        plt.savefig(filename)
        plt.clf()  # Clear the figure



    def extract_macd(self, data, fast=12, slow=26, signal=9):
        macd_2 = MACD(pd.DataFrame(np.flip(data), columns=['close'])['close'], window_fast=fast, window_slow =slow, window_sign=signal, fillna=True)
        macd_obj = macd_2.macd()
        macd_hist = macd_2.macd_diff()
        macd_sig = macd_2.macd_signal()
        index = 0 
        nans =0
        return macd_obj.values, macd_sig.values, macd_hist.values


    def get_rsi(self, data):
        # rsi
        rsi_1 = rsi(pd.DataFrame(np.flip(data), columns=['close'])['close'], window=14, fillna = True)
        return rsi_1[:].values

    """

    MACD BASIC STRATEGY
    Buy: 𝑀𝑎𝑐𝑑𝑡−1 < 𝑆𝑖𝑔𝑛𝑎𝑙𝑡−1 & (𝑀𝑎𝑐𝑑𝑡 > 𝑆𝑖𝑔𝑛𝑎𝑙𝑡 & 𝑀𝑎𝑐𝑑𝑡 > 0)
    Sell: 𝑀𝑎𝑐𝑑𝑡−1 > 𝑆𝑖𝑔𝑛𝑎𝑙𝑡−1 & (𝑀𝑎𝑐𝑑𝑡 < 𝑆𝑖𝑔𝑛𝑎𝑙𝑡 & 𝑀𝑎𝑐𝑑𝑡 < 0)

    (This data is for buy classification!)

    MACD --> signal crossover AND (∀{𝑅𝑆𝐼𝑡 , 𝑅𝑆𝐼𝑡−1 , 𝑅𝑆𝐼𝑡−2 , 𝑅𝑆𝐼𝑡−3 , 𝑅𝑆𝐼𝑡−4, 𝑅𝑆𝐼𝑡−5 } ≤ 𝐿𝑜𝑤𝑒𝑟 𝑇ℎ𝑟𝑒𝑠ℎ𝑜𝑙𝑑
    so, how would we calculate these statistics on a dataset? 

    """

    """
    args:

    macd: the macd values to be processed
    macd_signal: the macd signal values to be processed
    rsi: the rsi values to be processed

    returns:
    the prepared macd/rsi vector with the following values
    [𝑀𝑎𝑐𝑑𝑡−1, 𝑆𝑖𝑔𝑛𝑎𝑙𝑡−1, 𝑀𝑎𝑐𝑑𝑡, 𝑆𝑖𝑔𝑛𝑎𝑙𝑡, 𝑅𝑆𝐼𝑡 , 𝑅𝑆𝐼𝑡−1 , 𝑅𝑆𝐼𝑡−2 , 𝑅𝑆𝐼𝑡−3 , 𝑅𝑆𝐼𝑡−4, 𝑅𝑆𝐼𝑡−5]
    """

    def extract_macd_rsi_data(self, macd, macd_signal, rsi, tick):
        macd_rsi = None
        # macd signal crossover combined with the fact that the the 5 previous rsi's have fallen below the lower threshold
        # the lower threshold is 30
        # for many trend identifiers they use less hard-bounds, such as crossing 33 
        # as a bullish signal (which can be confirmed by volume?)
        lower_threshold = 33
        labels = None

        folder_path = "/home/benjamin/Desktop/ml/michinaga_extensions/src/dataUtils/sp500/" + tick  # replace with the path to your folder
        files = os.listdir(folder_path)
        files.sort()
        
        # we are going to start from 28, so that we are not using filled in values, and so that we can graph the rsi over
        # the slow period
        num_pos = 0
        for x in range(27, rsi.shape[0]):
            toCheck = rsi[x-6:x]
            file_name = files[x] 
            #the_plot = self.plot(np.arange(26), macd[x-26:x],  np.arange(26), macd_signal[x-26:x], np.arange(26), rsi[x-26:x], file_name, tick)
            path = f'/home/benjamin/Desktop/ml/michinaga_extensions/src/dataUtils/macd/{tick}/'
            os.makedirs(path, exist_ok=True)
            label_path = f'/home/benjamin/Desktop/ml/michinaga_extensions/src/dataUtils/macd_labels/{tick}/'
            os.makedirs(label_path, exist_ok=True)
           # print(max(toCheck))
           # print('threshold: ', max(toCheck) <= lower_threshold)
           # print('before cross: ', macd[x - 1] < macd_signal[x - 1])
           # print('after cross: ',macd[x] > macd_signal[x] and macd[x] > 0)
            #if(max(toCheck) <= lower_threshold) and (macd[x - 1] < macd_signal[x - 1]) and ((macd[x] > macd_signal[x] and macd[x] > 0)):
            # okay: we are gonna test with the MACD signal crossover w/0 the rsi  
            #if (max(toCheck) <= lower_threshold) and (macd[x - 1] < macd_signal[x - 1]) and ((macd[x] > macd_signal[x] and macd[x] > 0)):
            if (macd[x - 1] < macd_signal[x - 1]) and ((macd[x] > macd_signal[x] and macd[x] > 0)):
                num_pos += 1
            if(macd[x - 1] < macd_signal[x - 1]) and ((macd[x] > macd_signal[x] and macd[x] > 0)):
                # we will just demarcate with binary classifications?
                # the formation of the input tensors:
                # 𝑀𝑎𝑐𝑑𝑡−1, 𝑆𝑖𝑔𝑛𝑎𝑙𝑡−1, 𝑀𝑎𝑐𝑑𝑡, 𝑆𝑖𝑔𝑛𝑎𝑙𝑡, 𝑅𝑆𝐼𝑡 , 𝑅𝑆𝐼𝑡−1 , 𝑅𝑆𝐼𝑡−2 , 𝑅𝑆𝐼𝑡−3 , 𝑅𝑆𝐼𝑡−4, 𝑅𝑆𝐼𝑡−5 
                day = torch.tensor([macd[x - 1], macd_signal[x-1], macd[x], macd_signal[x]])
                label = torch.tensor([0, 1]).to(device)
                torch.save(day, path + file_name + '.pt')
                torch.save(label, label_path + file_name + '.pt')
                if(macd_rsi is None):
                    macd_rsi = day.view(1, day.shape[0])
                    labels = torch.tensor([0, 1]).to(device).view(1, 2)
                else:
                    macd_rsi = torch.cat((macd_rsi, day.view(1, day.shape[0])), axis = 0)
                    labels = torch.cat((labels, torch.tensor([0, 1]).to(device).view(1, 2)), axis = 0)
            else:
                label = torch.tensor([1, 0]).to(device)
                day = torch.tensor([macd[x - 1], macd_signal[x-1], macd[x], macd_signal[x]])
                torch.save(day, path + file_name + '.pt')
                torch.save(label, label_path + file_name + '.pt')
                if(macd_rsi is None):
                    macd_rsi = day.view(1, day.shape[0])
                    labels = torch.tensor([1, 0]).view(1, 2).to(device)
                else:
                    macd_rsi = torch.cat((macd_rsi, day.view(1, day.shape[0])), axis = 0)
                    labels = torch.cat((labels, torch.tensor([1, 0]).to(device).view(1, 2)), axis = 0)
        print(num_pos)
        return macd_rsi, labels

    """
    gather

    produces the collected macd and rsi inputs for processing

    """
    def gather(self):
        prep = None
        labels = None
        for tick in self.tickers:
            print(tick)
            if tick == 'BF.B':
                continue     
            data = torch.load('sp500/condensed/' + tick + '.pt').numpy().astype('double')
            close = data[:, 3]
            volume_data = data[:, 4]
            macd, macd_signal, macd_hist = self.extract_macd(close)
            rsi = self.get_rsi(close)
            # our inputs for the model
            macd_rsi, label = self.extract_macd_rsi_data(macd, macd_signal, rsi, tick)
            if(prep is None):
                prep = macd_rsi
                labels = label
            else:
                prep = torch.cat((prep, macd_rsi.view(macd_rsi.shape)), axis = 0)
                labels = torch.cat((labels, label), axis = 0)
        return prep, labels




sp500arr = np.loadtxt("constituents.csv",
                 delimiter=",", dtype=str)
sp500 = sp500arr[:, 0][1:]
scale = macdrsi(sp500)
x_data, y_data = scale.gather()
torch.save(x_data, 'macd_x_data.pt')
torch.save(y_data, 'macd_y_data.pt')

# another thing to consider: 
# how should volume play a role in our model?
