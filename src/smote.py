import torch
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
import sys, os, time, argparse


"""
FILE FOR SMOTE (Synthetic Minority Over-sampling Technique)
"""


# assuming this is for the lag-day lag period?
total_labels = 0
total_pos = 0


#graphs_directory = '/work/nlp/b.irving/stock/graphsPreprocessed'
graphs_directory = '/work/nlp/b.irving/stock/graphs'
tweets_directory = '/work/nlp/b.irving/stock/tweets_2'
labels_directory ='/work/nlp/b.irving/stock/labels_2' 
macd_directory = '/work/nlp/b.irving/stock/macd'

tickers = []
# Iterate over the files in the specified directory


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--lag', type=int, help='Lag period', default=5)
parser.add_argument('-c', '--channels', type=int, help='Number of channels in the image', default=4)
parser.add_argument('-ih', '--image_height', type=int, help='Height of the images', default=224)
parser.add_argument('-iw', '--image_width', type=int, help='Width of the images', default=224)
parser.add_argument('-t', '--tweet_embedding_dim', type=int, help='Dimension of the tweet embedding', default=128)
parser.add_argument('-md', '--macd_dim', type=int, help='Dimension of the macd information', default=4)
args = parser.parse_args()

lag = args.lag 
channels = args.channels
image_height = args.image_height
image_width = args.image_width
image_dim = channels * image_height * image_width
tweet_embedding_dim = args.tweet_embedding_dim 
macd_dim = args.macd_dim
feature_vector_length = lag * (image_dim + tweet_embedding_dim + macd_dim)

all_positives = []
all_negatives = []
print('Gathering features...')
for filename in os.listdir(tweets_directory):
    print(filename)
    # Construct the full file path
    tweets_file_path = os.path.join(tweets_directory, filename)
    graph_file_path = os.path.join(graphs_directory, filename)
    labels_file_path = os.path.join(labels_directory, filename)
    macd_file_path = os.path.join(macd_directory, filename)

    tweets = torch.load(tweets_file_path, map_location=torch.device('cpu')).cpu()
    graphs = torch.load(graph_file_path, map_location=torch.device('cpu')).cpu()
    macds = torch.load(macd_file_path, map_location=torch.device('cpu')).cpu()
    labels = torch.load(labels_file_path, map_location=torch.device('cpu')).cpu()

    positive_indices = (labels == 1).nonzero(as_tuple=True)[0]
    negative_indices = (labels == 0).nonzero(as_tuple=True)[0]

    # refactor
    for index in range(labels.shape[0]):
        if index > lag:
            lag_graphs = graphs[index - (lag - 1):index + 1].numpy()
            lag_tweets = tweets[index - (lag - 1):index + 1].numpy()
            lag_macds = macds[index - (lag - 1):index + 1].numpy()
            new_graphs = lag_graphs.reshape(1, lag, -1)
            new_tweets = lag_tweets.reshape(1, lag, -1)
            new_macds = lag_macds.reshape(1, lag, -1)
            combined_features = np.concatenate((new_graphs, new_tweets, new_macds), axis=2)
            combined_features = combined_features.reshape(1, feature_vector_length)
            if (labels[index] == 0):
                all_negatives.append(combined_features)
            else:
                all_positives.append(combined_features)

    """
    for index in positive_indices:
        if index > lag:
            # so for the positive graphs, we convert to numpy?
            positive_graphs = graphs[index - (lag - 1):index + 1].numpy()
            positive_tweets = tweets[index - (lag - 1):index + 1].numpy()

            new_graphs = positive_graphs.reshape(1, lag, -1)
            new_tweets = positive_tweets.reshape(1, lag, -1)
            combined_features = np.concatenate((new_graphs, new_tweets), axis=2)
            combined_features = combined_features.reshape(1, feature_vector_length)
            all_positives.append(combined_features)

    for index in negative_indices:
        if index > lag:
            # so for the positive graphs, we convert to numpy?
            negative_graphs = graphs[index - (lag - 1):index + 1].numpy()
            negative_tweets = tweets[index - (lag - 1):index + 1].numpy()

            new_graphs = positive_graphs.reshape(1, lag, -1)
            new_tweets = positive_tweets.reshape(1, lag, -1)
            combined_features = np.concatenate((new_graphs, new_tweets), axis=2)
            combined_features = combined_features.reshape(1, feature_vector_length)
            all_negatives.append(combined_features)
    """


all_positives_array = np.squeeze(np.stack(all_positives, axis=0), axis=1)
all_positives_labels = np.ones(all_positives_array.shape[0])
all_negatives_array = np.squeeze(np.stack(all_negatives, axis=0), axis=1)
all_negatives_labels = np.zeros(all_negatives_array.shape[0])
all_xs, all_ys = np.concatenate((all_positives_array, all_negatives_array), axis=0), np.concatenate((all_positives_labels, all_negatives_labels), axis=0)

np.save('/work/nlp/b.irving/stock/complete/all_xs_' + str(lag) + '.npy', all_xs)
np.save('/work/nlp/b.irving/stock/complete/all_ys_' + str(lag) + '.npy', all_ys)

print('Features gathered.')

#all_xs = np.load('all_xs_10.npy')
#all_ys = np.load('all_ys_10.npy')

# Doing SMOTE on this will take a long time to run
# is any of this worth it?
# YES. Synethetic data is really important
# will this work, ah ah ahhhhh
smote = SMOTE()
X_minority_resampled, y_minority_resampled = smote.fit_resample(all_xs, all_ys)

np.save('/work/nlp/b.irving/stock/complete/x_resampled_' + str(lag) + '.npy', X_minority_resampled)
np.save('/work/nlp/b.irving/stock/complete/y_resampled_' + str(lag) + '.npy', y_minority_resampled)

print('Resampling complete.')

#X_minority_resampled = np.load('x_resampled.npy')
#y_minority_resampled = np.load('y_resampled.npy')

# Is this the way to do this...
# I have no idea. Stay the course, and submit this paper
print('Saving images and tweets...')
images = X_minority_resampled[:, :1003520]
print('images', images.shape)
images_to_save = images.reshape(-1, lag, channels, image_height, image_width)
print('images to save', images_to_save.shape)
tweets = X_minority_resampled[:, 1003520:1004160]
print('tweets', tweets.shape)
tweets_to_save = tweets.reshape(-1, lag, tweet_embedding_dim)
print('tweets to save', tweets_to_save.shape)
macds = X_minority_resampled[:, 1004160:]
print('MACD', macds.shape)
macds_to_save = macds.reshape(-1, lag, macd_dim)
print('MACDs to save', macds_to_save.shape)

np.save('/work/nlp/b.irving/stock/complete/graphs_' + str(lag) + '.npy', images_to_save)
np.save('/work/nlp/b.irving/stock/complete/tweets_' + str(lag) + '.npy', tweets_to_save)
np.save('/work/nlp/b.irving/stock/complete/macds_' + str(lag) + '.npy', macds_to_save)
print('Process complete.')
# now, reshape and save the respective numpy arrays
