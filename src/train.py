import torch
from torch import tensor, nn, autograd
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
#from random_data import random_data
from torchmetrics import Accuracy, MatthewsCorrCoef, AUROC
from torchmetrics.classification import BinaryF1Score
import argparse
import sys
sys.path.append('../meant')
from meant import meant

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

def make_plot(loss, tick):
        timesteps = np.arange(1, len(loss) + 1)
        # Plot the MSE vs timesteps
        plt.plot(timesteps, loss)
        # Add axis labels and a title
        plt.xlabel('Timestep')
        plt.ylabel('Loss')
        plt.title('Loss')
        # Show the plot
        #plt.show()
        plt.savefig('/home/irving.b/meant_dep/plots/' + tick + '.png')

def train(model, params):
    """
    A simple training loop. The aim is to train meant over a specified number of epochs, and to measure accuracy with a basic measure.

    args: 
        - model
            the teanet model to be trained

        - params 
            the parameters to construct the training process
    """
    epochs = params['epochs']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    lag = params['lag']
    train_tickers = params['train_tickers']
    test_tickers = params['test_tickers']
    model.to(device)
    adam = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # learning rate schedulers, so that a saddle point doesn't debilitate model performance
    exponential = torch.optim.lr_scheduler.ExponentialLR(adam, gamma=0.95)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(adam, epochs)

    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCELoss()
    training_loss_over_epochs = []

    """
    TRAIN
    """
    # we should measure the training accuracy
    for e in range(epochs):
        accuracy = Accuracy(task='multiclass', num_classes=2).to(device)
        mcc = MatthewsCorrCoef(task='binary').to(device)
        auroc = AUROC(task="binary")
        training_loss = []
        total_acc = 0
        total_mc = 0

        # iterate through the training tickers
        # but now everything is manageably stored, and ready to go! well almost
        for tick_index in range(len(train_tickers)):
            tick = train_tickers[tick_index]                        
            tick_index += 1
            # for each ticker, we get the graphs, the tweets, the prices, and the labels in this step
            tweets = torch.load('/work/socialmedia/stock/tweets/' + tick).to(device)
            graphs = torch.load('/work/socialmedia/stock/graphs/' + tick + '.pt').to(device)
            macd = torch.load('/work/socialmedia/stock/macd/' + tick + '.pt').to(device)
            labels = torch.load('/work/socialmedia/stock/labels_2/' + tick + '.pt').to(device)
            train_index = 0
            while(train_index < tweets.shape[0] - (batch_size * lag) - 1 and train_index < macd.shape[0] - (batch_size * lag) - 1):
                model.zero_grad()

                # I need to fix the dataset. To make these experiments more interesting
                # I need to process the positive examples multiple times
                out = model.forward(tweets[train_index:train_index + (lag * batch_size)], 
                                    graphs[train_index:train_index + (lag * batch_size)].view(batch_size, lag, 4, 224, 224), 
                                    macd[train_index:train_index + (lag * batch_size)].view(batch_size, lag, 4))

                # the labels are the tensors at the end of every lag period
		        # rewrite this training loop, with the gated script

                loss = loss_fn(out.view(batch_size, 2).float(), labels[train_index:train_index+(batch_size * lag):lag].float().to(device))
            # loss = loss_fn(out.float(), labels[train_index:train_index+(batch_size * lag):lag].float().to(device))
                training_loss.append(loss.item())

                #with autograd.detect_anomaly():
                adam.zero_grad()
                loss.backward()
                adam.step()
                train_index += 1
            progress_accuracy = accuracy.compute()
            progress_mcc = mcc.compute()
            print("accuracy for " + tick + ": ", progress_accuracy.item())
            print("mcc for " + tick + ": ", progress_mcc.item())
                                       
            # eval dataset? 

           # make_plot(training_loss, tick)
        acc = accuracy.compute()
        total_mc = mcc.compute()
        total_auroc = auroc.compute()
        print('\n')
        print('epoch: ', e)
        print('training set accuracy: ', acc.item())
        print('matthews correlation coefficient', total_mc.item())
        print('auroc: ', total_auroc.item())
        print('loss total: ', sum(training_loss))
        print('\n')
        # we should just concatenate the loss because the model is being trained
        training_loss_over_epochs.append(training_loss)
        make_plot(training_loss, "total_loss")
        #exponential.step()
        cosine.step()

    torch.save(model, 'trained_teanet.pt')
    
    """
    EVALUATE

    For the first trial, I just want to see the basic accuracy. The more acute measurements can be implemented later. 
    """

    model.eval()

    with torch.no_grad():
        num_correct = 0
        tot = 0
        actuals = []
        outputs = []
        # now we iterate through the test tickers
        for tick in test_tickers:
            test_tweets = torch.load('/work/socialmedia/stock/tweets/' + tick).to(device)
            test_graphs = torch.load('/work/socialmedia/stock/graphs/' + tick + '.pt').to(device)
            test_macd = torch.load('/work/socialmedia/stock/macd/' + tick + '.pt').to(device)
            test_labels = torch.load('/work/socialmedia/stock/labels/' + tick + '.pt').to(device)
            for y in tqdm(range(test_tweets.shape[0] - lag)):
                out = model.forward(test_tweets[y:y+lag], 
                                    test_graphs[y:y+lag].view(1, lag, 4, 224, 224), 
                                    test_macd[y:y+lag].view(1, lag, 4))

                actual = torch.max(test_labels[y + lag].to(device), dim = 0).indices
                # want it to be one big list
                out_index = torch.max(out, dim = 2).indices
                if(actual.item() == out_index.item()):
                    num_correct += 1
                tot += 1
                # mcc coefficient from these list
                actuals.append(actual.item())
                outputs.append(out_index.item())

    act = torch.tensor(actuals).float().to(device)
    outs = torch.tensor(outputs).float().to(device)
    mat = mcc(outs, act)
    accuracy = num_correct / tot
    print("Basic accuracy on test set: ", accuracy)
    print("mcc: ", mat)
    return training_loss_over_epochs, accuracy

def plot(arr_list, legend_list, color_list, ylabel, fig_title):
    """
    Args:
        arr_list (list): list of results arrays to plot
        legend_list (list): list of legends corresponding to each result array
        color_list (list): list of color corresponding to each result array
        ylabel (string): label of the Y axis

        Note that, make sure the elements in the arr_list, legend_list and color_list are associated with each other correctly.
        Do not forget to change the ylabel for different plots.
    """
    # set the figure type
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time Steps")

    # ploth results
    h_list = []
    for arr, legend, color in zip(arr_list, legend_list, color_list):
        # compute the standard error
        arr_err = arr.std(axis=0) / np.sqrt(arr.shape[0])
        # plot the mean
        h, = ax.plot(range(arr.shape[1]), arr.mean(axis=0), color=color, label=legend)
        # plot the confidence band
        arr_err *= 1.96
        ax.fill_between(range(arr.shape[1]), arr.mean(axis=0) - arr_err, arr.mean(axis=0) + arr_err, alpha=0.3,
                        color=color)
        # save the plot handle
        h_list.append(h)

    # plot legends
    ax.set_title(f"{fig_title}")
    ax.legend(handles=h_list)

    plt.show()


if __name__ == "__main__":

    sp500arr = np.loadtxt("constituents.csv",
                    delimiter=",", dtype=str)
    sp500 = sp500arr[:, 0][1:37]
    np.random.shuffle(sp500)
    print(len(sp500))
    # should it be shuffled by ticker?
    # no. But with the limited space that we have, it is the best way
    train_tickers = sp500[0:25]
    val_tickers = sp500[25:29]
    test_tickers = sp500[29:]

    parser = argparse.ArgumentParser(description="Training loop options")

    # Add command-line arguments
    parser.add_argument("--epochs", type=int, help="Number of epochs", default = 1)
    parser.add_argument("--batch_size", type=int, help="Batch size", default = 5)
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default = 1e-3)
    parser.add_argument("--lag_period", type = int, help="Lag Period", default = 3)
    # to change this argument changes the inherent structure of the code
    parser.add_argument("--num_classes", type = int, help = "Number of Classes", default = 2)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Read the training loop options from the parsed arguments
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    lag_period = args.lag_period
    num_classes = args.num_classes

    # Print the training loop options
    print("Training Loop Options:")
    print("Epochs:", epochs)
    print("Batch Size:", batch_size)
    print("Learning Rate:", learning_rate)
    print("Lag Period:", lag_period)
    print("Number of Classes", num_classes)

    #randomize = random_data(lag)

    # initialize model
    bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
    
    model = meant(text_dim =768, 
                image_dim = 768, 
                price_dim = 4, 
                height = 224, 
                width = 224, 
                patch_res = 16, 
                lag = lag_period, 
                num_classes = num_classes, 
                embedding = bertweet.embeddings)
    model.to(device)
    model.to(torch.float32)

    accuracy_over_time = []
    
    params = {
        'lag': lag_period,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'train_tickers': train_tickers,
        'test_tickers': test_tickers
    }


    training_loss, accuracy = train(model, params)
