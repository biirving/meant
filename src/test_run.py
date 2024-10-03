#!/usr/bin/env python
import os, sys, time, gc, socket, math, string, re, csv, ast, argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Memory
from torch import nn, tensor
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
    AutoModel,
    BertTokenizer,
    VisualBertModel,
    ViltModel,
    ViltProcessor,
    #DebugUnderflowOverflow,
)
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset, Dataset
sys.path.append('../meant')
from meant import meant, meant_vision, meant_tweet, temporal, meant_tweet_no_lag, vl_BERT_Wrapper, ViltWrapper
from utils import f1_metrics
from torch.utils.tensorboard import SummaryWriter
import wandb
import pandas as pd

sys.path.append('teanet/models')
sys.path.append('teanet/utils/')
import classicAttention
from teanet import teanet


# detecting where nans originate from
#torch.autograd.set_detect_anomaly(True)

torch.cuda.empty_cache()
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ensure that this datatype is the same as what the arrays you load in are saved in if doing memmap
np_dtype = np.float64

# torch datatype to used for automatic mixed precision training
torch_dtype = torch.float16

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# simple class to help load the dataset
class customDataset(Dataset):
    def __init__(self, graphs, tweets, macds, labels):
        self.graphs = torch.tensor(graphs)
        self.tweets = torch.tensor(tweets)
        self.macds = torch.tensor(macds)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.graphs[idx], self.tweets[idx], self.macds[idx], self.labels[idx]

# simple class to help load the dataset
class customDataset_stocknet(Dataset):
    def __init__(self, tweets, prices, labels):
        self.tweets = torch.tensor(tweets)
        self.prices = torch.tensor(prices)
        self.labels = torch.tensor(labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.tweets[idx], self.prices[idx], self.labels[idx]

class meant_trainer():
    def __init__(self, params):
        """
        args:
            params
                epochs: Number of times we train the model on the entire training set
                model: Model that we are training.
                optimizer: Optimizer that we use to modifiy learning rates, and backpropogate through model.
                train_batch_size: Batch size for training runs.
                debug_overflow: Flag is active if we want to see reasons behind Nans, in underflow and overflow.
                X_train: X values for training data (In this case, embeddings for related work or all references).
                y_train: y values for training data (Embeddings produced by ProNE, which we treat as a baseline).
                y_test: y values for test data. (Same as above)
                dimension: Dimension of the hidden layers of the model.
                num_layers: Number of layers in the model.
                dropout: Dropout of the model.
                pretrained_model: If we want to use object for evaluation only, we need to load a pretrained model in.
        """
        # general
        self.run_id = params['run_id']

        # batch sizes
        self.test_batch_size = params['test_batch_size']

        # for debugging
        self.debug_overflow = params['debug']

        # DATA
        self.dataset = params['dataset']
        self.test_loader = params['test_loader']
        self.model = params['model']

        self.file_path = params['file_path']
        self.model_name = params['model_name']

    def plot(self, loss, model, epoch):
        timesteps = np.arange(1, loss.shape[0] + 1)
        alpha = 0.9  # Smoothing factor (adjust as needed)
        ema = [loss[0]]  # Initialize the EMA with the first loss value
        for i in range(1, loss.shape[0]):
            ema.append(alpha * ema[-1] + (1 - alpha) * loss[i])
        ema = np.array(ema)  # Convert the EMA list to a NumPy array
        plt.plot(ema, loss)
        plt.xlabel('Timestep')
        plt.ylabel('CrossEntropyLoss')
        plt.title('Loss')
        plt.savefig(self.file_path + '/output_files/' + self.dataset + '/plots/' + model + '_' + self.run_id + '_' + str(epoch) + '.png')
        plt.close()


    def f1_plot(self, scores):
        plt.figure(figsize=(6, 4))  # Adjust the figure size as needed
        plt.scatter(np.linspace(0, len(scores) - 1, len(scores), endpoint=True, retstep=False, dtype=None, axis=0), [scores], c='blue', label='F1 Score')
        # Customize the plot
        plt.xlabel(self.model_name)
        plt.ylabel('F1 Score')
        plt.title('F1 score for model')
        plt.ylim(0, 1)  # Set the y-axis limits (F1 score ranges from 0 to 1)
        plt.xticks([1], [self.model_name])  # Label the x-axis with the model name
        plt.legend()
        # Display the plot
        plt.grid(True, linestyle='--', alpha=0.5)  # Add a grid
        plt.tight_layout()
        try:
            plt.savefig(self.file_path + '/output_files/' + self.dataset + '/plots/' + 'train_f1_' + self.model_name + '_' + self.run_id + '_' + str(self.epoch) + '.png')
        except FileNotFoundError:
            print('Invalid filepath. Check that you have created the right folders.')

    def train(self):

        if(self.debug_overflow):
            debug_overflow = DebugUnderflowOverflow(self.model)
    
        test_metrics = f1_metrics(self.num_classes, 'test') 
        self.model.eval()
        with torch.no_grad():
            if self.dataset == 'Tempstock':
                for graphs, tweets, macds, target in self.test_loader:
                    # the autocasting doesn't work with flash attention? Examine bug
                    # why doesn't this stuff work? What is the problem?????
                    # let it rip
                    with torch.autocast(device_type="cuda", dtype=torch_dtype):
                        if self.model_name == 'meant':
                            out = model.forward(tweets.long().to(device), graphs.to(torch_dtype).to(device))
                        elif self.model_name == 'meant_vision':
                            out = model.forward(graphs.to(torch_dtype).to(device))
                        elif self.model_name == 'meant_tweet':
                            out = model.forward(tweets.long().to(device))
                        elif self.model_name == 'teanet':
                            out = model.forward(tweets.to(torch_dtype).to(device), macds.to(torch_dtype).to(device))
                        else:
                            out = model(tweets[:, 4, :].squeeze(dim=1).long().cuda(), graphs[:, 4, :, :].to(torch_dtype).squeeze(dim=1).cuda())
                    test_metrics.update(out.detach().cpu(), target) 
            elif self.dataset == 'Stocknet':
                for tweets, prices, target in self.test_loader:
                    with torch.autocast(device_type="cuda", dtype=torch_dtype):
                        if self.model_name == 'meant':
                            raise ValueError('MEANT is a multimodal model, while Stocknet is a unimodal dataset. Use MEANTweet.')
                        elif self.model_name == 'meant_vision':
                            raise ValueError('MEANT_vision is a vision focused model, while Stocknet is a language focused dataset. Use MEANTweet.')
                        elif self.model_name == 'meant_tweet':
                            out = model.forward(tweets.long().to(device))
                        elif self.model_name == 'teanet':
                            out = model.forward(tweets.to(torch_dtype).to(device), prices.to(torch_dtype).to(device))
                        elif self.model_name == 'bertweet':
                            # we only process the tweet on the lag day
                            out = model(tweets[:, 4, :].squeeze(dim=1).long().cuda())
                            pass
                        else:
                            raise ValueError('Model not supported.')
                    test_metrics.update(out.detach().cpu(), target) 
        test_metrics.show()  

if __name__=='__main__':
    # nightly pytorch build required
    #torch._dynamo.config.verbose = True
    #torch._dynamo.config.suppress_errors = True

    parser = argparse.ArgumentParser()
    

    # Training loop 
    parser.add_argument('-e', '--epoch', type = int, help = 'Current epoch at start of training', default=0)
    parser.add_argument('-ne', '--num_epochs', type=int, help = 'Number of epochs to run training loop', default=1)
    parser.add_argument('-es', '--early_stopping', type=str2bool, help = 'Early stopping is active', nargs='?', const=False, default=False)
    parser.add_argument('-s', '--stoppage', type=float, help='Stoppage value', default=1e-4)
    parser.add_argument('-tesb', '--test_batch_size',type=int, help='Batch size for test step', default=1)
    parser.add_argument('-testm', '--test_model', type=str2bool, help='Whether or not to test our model', nargs='?', const=True, default=True)


    # hugging face
    parser.add_argument('-hf', '--hugging_face_model', type=str2bool, help='If we want to finetune/pretrain a model from Hugging face.', nargs='?', const=False, default=False)
    parser.add_argument('-hfd', '--hugging_face_data', type=str, help='Data set to load from Hugging Face', default=None)
    parser.add_argument('-hft', '--hugging_face_tokenizer', type=str, help='HuggingFace tokenizer', default=None)

    # Miscellaneous
    parser.add_argument('-db', '--debug', type = bool, help = 'Debug underflow and overflow', default = False)
    parser.add_argument('-fp', '--file_path', type=str, help='Path to files', default='/work/nlp/b.irving/meant_runs')
    parser.add_argument('-rid', '--run_id', type=str, help='Run identification number', required=True)
    parser.add_argument('-lag', '--lag', type=int, help='Lag period for data', default=5)
    parser.add_argument('-norm', '--normalize', type=str2bool, help='Whether or not to normalize the data', nargs='?', const=False, default=False)
    parser.add_argument('-ds', '--dataset', type=str, help='The dataset that we are training on', default='Tempstock')
    args = parser.parse_args()

    t0 = time.time()
    
    # the model most be loaded first, in order to support the instantiation of the optimizer and the learning rate scheduler

    #bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
    # this file is primarily for loading and testing fine tuned models
    #model = torch.load(args.file_path + '/models/' + args.model_name + '/' + args.model_name + '_' + args.run_id + '_' + str(args.epoch) + '.pt')
    model = torch.load('')
    
    # delete the bertweet model
    del bertweet
    gc.collect()

    if(args.hugging_face_tokenizer is not None):
        tokenizer = AutoTokenizer.from_pretrained(args.hugging_face_tokenizer)
    else:
        tokenizer = None

    print('Preparing data...')

    if args.dataset == 'Tempstock':
        assert not (args.image_only == True and args.language_only == True), 'Cannot be an image only AND a language only task'
        if args.image_only:
            graphs = np.memmap('/work/nlp/b.irving/stock/complete/graphs_5.npy', dtype=np_dtype, mode='r')
            tweets = np.ones(graphs.shape[0], 1).astype(np.float32)
        elif args.language_only:
            tweets = np.load('/work/nlp/b.irving/stock/complete/tweets_5.npy', dtype=np_dtype, mode='r')
            graphs = np.ones(tweets.shape[0], 1).astype(np.float32)
        else:
            graphs = np.load('/work/nlp/b.irving/stock/complete/graphs_5.npy')
            tweets = np.load('/work/nlp/b.irving/stock/complete/tweets_5.npy')
            macds = np.load('/work/nlp/b.irving/stock/complete/macds_5.npy')
            labels = np.load('/work/nlp/b.irving/stock/complete/y_resampled_5.npy')

        if args.normalize:
            print('Normalizing data...')
            # our memmap arrays are read-only
            graphs -= np.mean(graphs)
            graphs /= np.std(graphs)

            tweets -= np.mean(tweets)
            tweets /= np.std(tweets)

            macds -= np.mean(macds)
            macds /= np.std(macds)
            print('Data normalized.')


        # create a dataLoader object
        test_dataset = customDataset(graphs, tweets, macds, y)
        # clear up memory
        del graphs
        del tweets
        del macds
        del labels
        gc.collect()
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)

    elif args.dataset == 'Stocknet':
        tweets = np.load('/scratch/irving.b/stock/stocknet_tweets.npy')
        prices = np.load('/scratch/irving.b/stock/stocknet_prices.npy')
        labels = np.load('/scratch/irving.b/stock/stocknet_labels.npy')
        if args.normalize:
            print('Normalizing data...')
            tweets -= np.mean(tweets)
            tweets /= np.std(tweets)
            prices -= np.mean(prices)
            prices/= np.std(prices)
            print('Data normalized.')

        test_dataset = customDataset_stocknet(tweets, prices, labels)
        del tweets
        del prices 
        del labels
        gc.collect()
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)

    print('Data prepared.')

    params = {

            # DATA
            'dataset':args.dataset, 
            'test_loader':test_loader,

            'stoppage':args.stoppage,
            'lr': args.learning_rate,
            'run_id':args.run_id,
            'file_path': args.file_path,
            'pretrained_model': args.pretrained_model,

            # Epochs
            'epoch': args.epoch,
            'num_epochs' : args.num_epochs, 
            'early_stopping':args.early_stopping,

            'test_batch_size':args.test_batch_size,
            'model':model,
            'debug':args.debug,
            'model_name':args.model_name,
    }

    train = meant_trainer(params)
    train.train()
    print('Done in ' +  str(time.time() - t0) + ' seconds.')