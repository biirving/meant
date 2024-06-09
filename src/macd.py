"""
MACD will be an indicator that I initially focus on

MACD + RSI

MACD + MFI

Woe to the conquered.

"""

import numpy as np
import torch
from torch import nn, tensor
#import talib
import matplotlib.pyplot as plt
import math
import ta
from ta.trend import MACD 
from ta.momentum import rsi
import pandas as pd

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


    # for the multimodal processing, we should save the plots as well with names that
    # are identifiable and matchable to the data?
    # our long term data will be found in the plots.
    def plot(self, x1, y1, x2, y2, x3, y3):
        plt.plot(x1, y1, label = 'MACD')
        plt.plot(x2, y2, label = 'Signal')
        #plt.plot(x3, y3, label = 'RSI')
        plt.legend()
        plt.show()

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
        lower_threshold = 30
        labels = None
        # we are going to start from 26, so that we are not using filled in values, and so that we can graph the rsi over
        # the slow period
        for x in range(27, rsi.shape[0] - 6):
            toCheck = rsi[x:x+6]
            # a BUY signal
            # should we have the rsi plot on the same graph
            the_plot = self.plot(np.arange(26), macd[x-26:x],  np.arange(26), macd_signal[x-26:x], np.arange(26), rsi[x-26:x])
            if(max(toCheck) <= lower_threshold and macd[x - 1] < macd_signal[x - 1] and (macd[x] > macd_signal[x] and macd[x] > 0)):
                # we will just demarcate with binary classifications?
                # the formation of the input tensors:
                # 𝑀𝑎𝑐𝑑𝑡−1, 𝑆𝑖𝑔𝑛𝑎𝑙𝑡−1, 𝑀𝑎𝑐𝑑𝑡, 𝑆𝑖𝑔𝑛𝑎𝑙𝑡, 𝑅𝑆𝐼𝑡 , 𝑅𝑆𝐼𝑡−1 , 𝑅𝑆𝐼𝑡−2 , 𝑅𝑆𝐼𝑡−3 , 𝑅𝑆𝐼𝑡−4, 𝑅𝑆𝐼𝑡−5 
                # volume or no volume?
                day = torch.tensor([macd[x - 1], macd_signal[x-1], macd[x], macd_signal[x], toCheck[4], toCheck[3], toCheck[2], toCheck[1], toCheck[0]])
                if(macd_rsi is None):
                    macd_rsi = day.view(1, day.shape[0])
                    labels = torch.tensor([0, 1]).to(device).view(1, 2)
                else:
                    macd_rsi = torch.cat((macd_rsi, day.view(1, day.shape[0])), axis = 0)
                    labels = torch.cat((labels, torch.tensor([0, 1]).to(device).view(1, 2)), axis = 0)
            else:
                day = torch.tensor([macd[x - 1], macd_signal[x-1], macd[x], macd_signal[x], toCheck[0], toCheck[1], toCheck[2], toCheck[3], toCheck[4]])
                if(macd_rsi is None):
                    macd_rsi = day.view(1, day.shape[0])
                    labels = torch.tensor([1, 0]).view(1, 2).to(device)
                else:
                    macd_rsi = torch.cat((macd_rsi, day.view(1, day.shape[0])), axis = 0)
                    labels = torch.cat((labels, torch.tensor([1, 0]).to(device).view(1, 2)), axis = 0)
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
            data = torch.load('sp500/' + tick + '.pt').numpy().astype('double')
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
torch.save(x_data, 'macd_rsi_x_data.pt')
torch.save(y_data, 'macd_rsi_y_data.pt')

# another thing to consider: 
# how should volume play a role in our model?


# again: This is just focusing on the buy signal.