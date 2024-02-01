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

sys.path.append('/work/nlp/b.irving/michinaga/teanet/models')
sys.path.append('/work/nlp/b.irving/michinaga/teanet/utils/')
import classicAttention
from teanet import teanet


# detecting where nans originate from
torch.autograd.set_detect_anomaly(True)

torch.cuda.empty_cache()
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ensure that this datatype is the same as what the arrays you load in are saved in if doing memmap
np_dtype = np.float64

# torch datatype to used for automatic mixed precision training
torch_dtype = torch.float32

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
        self.learning_rate = params['lr']
        self.optimizer = params['optimizer']
        self.optimizer_name = params['optimizer_name']

        # epochs
        self.epoch = params['epoch']
        self.num_epochs = params['num_epochs']
        self.early_stopping = params['early_stopping']

        # batch sizes
        self.train_batch_size = params['train_batch_size']
        self.eval_batch_size = params['eval_batch_size']
        self.test_batch_size = params['test_batch_size']

        # for debugging
        self.debug_overflow = params['debug']

        # DATA
        self.dataset = params['dataset']
        self.train_loader = params['train_loader']
        self.val_loader = params['val_loader']
        self.test_loader = params['test_loader']
        self.test_model = params['test_model']

        # model specific         
        self.model = params['model']
        self.dimension = params['dim']
        self.num_layers = params['num_layers']
        self.dropout = params['dropout']
        self.pretrained_model = params['pretrained_model']
        self.num_classes = params['classes']
        self.num_encoders = params['num_encoders']

        self.lr_scheduler = params['lr_scheduler']
        self.tokenizer = params['tokenizer']
        self.stoppage = params['stoppage']
        self.lrst = params['lrst']
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
        loss_fct = nn.CrossEntropyLoss()
        t0 = time.time()
        training_loss = []
        total_acc = 0
        self.model.train()
        prev_f1 = 0
        final_epoch = 0
    
        # for early stopping
        lost_patience = 5
        patience = 0
        prev_f1 = float('inf')

        scaler = torch.cuda.amp.GradScaler()

        for ep in range(self.num_epochs):
            final_epoch = ep
            predictions = []
            target_values = []
            train_metrics = f1_metrics(self.num_classes, 'train') 
            train_f1_scores = []
            print('Training model on epoch ' + str(self.epoch + ep))
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {ep+1}/{self.num_epochs}')
            if self.dataset == 'Tempstock':
                for graphs, tweets, macds, target in progress_bar:
                    self.optimizer.zero_grad() 
                    # should have a separate forward pass function
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
                            # we run without the lag period. Only meant supports this!
                            out = model(tweets[:, 4, :].squeeze(dim=1).to(torch.float32).cuda(), graphs[:, 4, :, :].to(torch_dtype).squeeze(dim=1).cuda())
                        loss = loss_fct(out, target.to(device).long())                
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                    out = out.detach().cpu()
                    target = target.detach().cpu()
                    train_metrics.update(out, target) 
                    del out
                    del target
                    del loss
            elif self.dataset == 'Stocknet':
                for tweets, prices, target in progress_bar:
                    self.optimizer.zero_grad() 
                    with torch.autocast(device_type="cuda", dtype=torch_dtype):
                        if self.model_name == 'meant':
                            raise ValueError('MEANT is a multimodal model, while Stocknet is a unimodal dataset. Use MEANTweet.')
                        elif self.model_name == 'meant_vision':
                            raise ValueError('MEANT_vision is a vision focused model, while Stocknet is a language focused dataset. Use MEANTweet.')
                        elif self.model_name == 'meant_tweet':
                            out = model.forward(tweets.long().to(device))
                        elif self.model_name == 'teanet':
                            out = model.forward(tweets.to(torch_dtype).to(device), prices.to(torch_dtype).to(device))
                        else:
                            raise ValueError('Model not supported.')
                        loss = loss_fct(out, target.to(device).long())                
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                    out = out.detach().cpu()
                    target = target.detach().cpu()
                    train_metrics.update(out, target) 
                    del out
                    del target
                    del loss

            print('length: ', str(time.time() - t0))
            print('loss total: ', sum(training_loss))

            train_metrics.show() 

            #self.f1_plot(np.array(train_f1_scores))
            self.lr_scheduler.step()
            val_metrics = f1_metrics(self.num_classes, 'validation') 
            self.model.eval()
            val_loss = 0
            print('Evaluating Model...')
            with torch.no_grad():
                val_progress_bar = tqdm(self.val_loader, desc=f'Epoch {ep+1}/{self.num_epochs}')
                if self.dataset == 'Tempstock':
                    for graphs, tweets, macds, target in self.val_loader:
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
                        val_metrics.update(out.detach().cpu(), target) 
                elif self.dataset == 'Stocknet':
                    for tweets, prices, target in self.val_loader:
                        with torch.autocast(device_type="cuda", dtype=torch_dtype):
                            if self.model_name == 'meant':
                                raise ValueError('MEANT is a multimodal model, while Stocknet is a unimodal dataset. Use MEANTweet.')
                            elif self.model_name == 'meant_vision':
                                out = model.forward(graphs.to(torch_dtype).to(device))
                            elif self.model_name == 'meant_tweet':
                                out = model.forward(tweets.long().to(device))
                            elif self.model_name == 'teanet':
                                out = model.forward(tweets.to(torch_dtype).to(device), macds.to(torch_dtype).to(device))
                            else:
                                raise ValueError('Model not supported.')
                        val_metrics.update(out.detach().cpu(), target) 
                else:
                    raise ValueError('Dataset not supported.')
            val_f1_macro, val_f1_micro = val_metrics.show() 

            if self.early_stopping:
                if(val_f1_macro <= prev_f1):
                    patience += 1
                    if(patience == lost_patience):
                        print('Stopped at epoch ' + str(ep + 0))
                        break
                else:
                    patience = 0
                prev_f1 = val_f1_macro
        try:
            torch.save(self.model, self.file_path + '/models/' + self.model_name + '/' + self.model_name + '_' + str(self.num_encoders) + '_' +  self.dataset + '_' + self.run_id + '_' + str(final_epoch + 1) + '.pt')
            torch.save(self.optimizer.state_dict(), self.file_path + '/optimizers/' +  self.optimizer_name + '/' + self.model_name + '_' + self.run_id + '_' + str(args.learning_rate) + '_' + str(self.epoch + 1) + '.pt')
            torch.save(self.lr_scheduler.state_dict(), self.file_path + '/lr_schedulers/' + self.lrst + '/' + self.model_name + '_' +  self.run_id + '_' + str(self.epoch + 1) + '.pt')
        except FileNotFoundError:
            print('Your filepath is invalid. Save has failed')

        if(self.test_model):
            print('Testing...') 
            test_metrics = f1_metrics(self.num_classes, 'test') 
            self.model.eval()
            f1_scores = []
            with torch.no_grad():
                if self.dataset == 'Tempstock':
                    for graphs, tweets, macds, target in self.test_loader:
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
                            else:
                                raise ValueError('Model not supported.')
                        test_metrics.update(out.detach().cpu(), target) 
            test_metrics.show()  
            #self.f1_plot(np.array(f1_scores))

if __name__=='__main__':
    # nightly pytorch build required
    #torch._dynamo.config.verbose = True
    #torch._dynamo.config.suppress_errors = True

    parser = argparse.ArgumentParser()
    
    # Learning rate scheduler
    parser.add_argument('-t0', '--t0', type = int, help = 'Number of iterations for the first restart', default = 7)
    parser.add_argument('-tm', '--tmax', type = int, help = 'The number of epochs that the cosine lr takes to complete a cycle', default = 10)
    parser.add_argument('-lrst', '--learning_rate_scheduler_type', type=str, help='The type of learning rate scheduler to use.', default='cosine_warm')

    # optimizer
    parser.add_argument('-l', '--learning_rate', type=float, help='Learning rate for the trainer', default=5e-5)
    parser.add_argument('-o', '--optimizer', type = str, help = 'Optimizer', default = 'AdamW')
    parser.add_argument('-d', '--decay', type = float, help = 'Weight decay for the optimizer', default = 0.0)
    parser.add_argument('-b1','--beta_1', type = float, help='Beta1 for the optimizer', default = 0.9)
    parser.add_argument('-b2', '--beta_2', type = float, help = 'Beta2 for the optimizer', default= 0.999)

    # Training loop 
    parser.add_argument('-e', '--epoch', type = int, help = 'Current epoch at start of training', default=0)
    parser.add_argument('-ne', '--num_epochs', type=int, help = 'Number of epochs to run training loop', default=1)
    parser.add_argument('-es', '--early_stopping', type=str2bool, help = 'Early stopping is active', nargs='?', const=False, default=False)
    parser.add_argument('-s', '--stoppage', type=float, help='Stoppage value', default=1e-4)
    parser.add_argument('-tb', '--train_batch_size', type = int, help = 'Batch size for training step', default = 16)
    parser.add_argument('-eb', '--eval_batch_size',type=int, help='Batch size for evaluation step', default=1)
    parser.add_argument('-tesb', '--test_batch_size',type=int, help='Batch size for test step', default=1)
    parser.add_argument('-testm', '--test_model', type=str2bool, help='Whether or not to test our model', nargs='?', const=True, default=True)

    # Model specific
    parser.add_argument('-mn', '--model_name', type=str, help='Model name', default='meant')
    parser.add_argument('-nc', '--num_classes', type= int, help='Number of classes', default = 2)
    parser.add_argument('-t', '--task', type = str, help = 'Task type for training loop', default = 'classification')
    parser.add_argument('-cl', '--cache_location', type = str, help = 'Location for HuggingFace files')
    parser.add_argument('-di', '--dimension', type=int, help = 'internal dimension', default = 128)
    parser.add_argument('-nl', '--num_layers', type=int, help= 'The number of layers to use in the model', default=3)
    parser.add_argument('-do', '--dropout', type=float, help='Dropout in our model', default=0.0)
    parser.add_argument('-ptm', '--pretrained_model', type=str, help='Path to model', default=None)
    parser.add_argument('-p', '--pretrained', type =str2bool, help='Load pretrained model if True. Train from scratch if false', nargs='?', const=False, default=False)
    parser.add_argument('-nec', '--num_encoders', type=int, help='The number of encoders in our model', default=12)
    parser.add_argument('-img', '--image_only', type=str2bool, help='Is our task image only or not', nargs='?', const=False, default=False)
    parser.add_argument('-lang', '--language_only', type=str2bool, help='Is our task language only or not', nargs='?', const=False, default=False)

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

    bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
    if(args.epoch == 0):
        if args.hugging_face_model is True:
            if args.pretrained is True:
                model = AutoModelForTokenClassification.from_pretrained(args.hugging_face_model).to(device)
            else: 
                print('Training model from scratch')
                config = AutoConfig.from_pretrained('/work/nlp/b.irving/nlp/src/hug/configs/' + args.model_name +'.json', local_files_only=True)
                if args.model_name == 'vl_bert':
                    # use pretrained model for TESTING 
                    #vl_bert_model = VisualBertModel._from_config(config).cuda()
                    # using pretrained VisualBERT
                    # rerun these experiments
                    vl_bert_model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
                    vl_bert_model.embeddings.word_embeddings = bertweet.embeddings.word_embeddings
                    model = vl_BERT_Wrapper(vl_bert_model, 768, 2).cuda()
                elif args.model_name == 'vilt':
                    config = AutoConfig.from_pretrained('/work/nlp/b.irving/nlp/src/hug/configs/vilt.json', local_files_only=True)
                    #vilt = ViltModel._from_config(config)
                    # using pretrained ViLT
                    # rerun these experiments
                    vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
                    vilt.embeddings.text_embeddings.word_embeddings = bertweet.embeddings.word_embeddings
                    model = ViltWrapper(vilt, 768, 2).to(device) 
        elif args.model_name == 'meant':
            # do we need the embedding layer if we have already used the flair nlp embeddings?
            model = meant(text_dim = 768, 
                image_dim = 768, 
                price_dim = 4, 
                height = 224, 
                width = 224, 
                patch_res = 16, 
                lag = args.lag, 
                num_classes = args.num_classes, 
                embedding = bertweet.embeddings,
                num_encoders=args.num_encoders).to(device)
        elif args.model_name == 'meant_vision':
            model = meant_vision(
                image_dim = 768, 
                price_dim = 4, 
                height = 224, 
                width = 224, 
                patch_res = 16, 
                lag = args.lag, 
                num_classes = args.num_classes, 
                num_encoders=args.num_encoders).to(device)
        elif args.model_name == 'meant_tweet':
            model = meant_tweet(text_dim = 768, 
                price_dim = 4, 
                lag = args.lag, 
                num_classes = args.num_classes, 
                embedding = bertweet.embeddings,
                num_encoders=args.num_encoders).to(device) 
        elif args.model_name == 'teanet':
            model = teanet(5, 128, 2, 5, 12, 10).cuda()
        else:
            raise ValueError('Pass a valid model name.')
    else:
        model = torch.load(args.file_path + '/models/' + args.model_name + '/' + args.model_name + '_' + args.run_id + '_' + str(args.epoch) + '.pt')
    
    # delete the bertweet model
    del bertweet
    gc.collect()

    if(args.optimizer == 'AdamW'):
        optimizer = torch.optim.AdamW(params = model.parameters(), lr=args.learning_rate, weight_decay=args.decay, betas=(args.beta_1, args.beta_2))
    elif(args.optimizer == 'Adam'):
        optimizer = torch.optim.Adam(params = model.parameters(), lr=args.learning_rate, weight_decay=args.decay, betas=(args.beta_1, args.beta_2))
    else: 
        raise ValueError("This type of optimizer is not supported.")

    if(args.hugging_face_tokenizer is not None):
        tokenizer = AutoTokenizer.from_pretrained(args.hugging_face_tokenizer)
    else:
        tokenizer = None

    # load incrementally
    if(args.learning_rate_scheduler_type == 'cosine_warm'):
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.t0)
    elif(args.learning_rate_scheduler_type == 'cosine'):
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.tmax)
    elif(args.learning_rate_scheduler_type == 'linear'):
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    else:
        raise ValueError('Not supported')
    
    if(args.epoch == 0):
       pass # we don't need to load in progress state dictionaries
    else:
        optimizer_state_dict = torch.load(args.file_path + '/optimizers/' + args.optimizer + '/' + args.model_name + '_' + args.run_id + '_' + str(args.learning_rate) + '_' + str(args.epoch) + '.pt')
        lr_scheduler_state_dict = torch.load(args.file_path + '/lr_schedulers/' + args.learning_rate_scheduler_type + '/' + args.model_name + '_' + args.run_id + '_' + str(args.epoch) + '.pt')
        optimizer.load_state_dict(optimizer_state_dict)
        lr_scheduler.load_state_dict(lr_scheduler_state_dict, start_factor=0.1)

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

        # First split: Separate out the test set
        graphs_train_val, graphs_test, tweets_train_val, tweets_test, macds_train_val, macds_test, y_train_val, y_test = train_test_split(
            graphs, tweets, macds, labels, test_size=0.2, random_state=42)

        # clear up memory
        del graphs
        del tweets
        del macds
        del labels
        gc.collect()

        # Second split: Split the remaining data into training and validation sets
        graphs_train, graphs_val, tweets_train, tweets_val, macds_train, macds_val, y_train, y_val= train_test_split(
            graphs_train_val, tweets_train_val, macds_train_val, y_train_val, test_size=0.25, random_state=42) 
        
        del macds_train_val
        del y_train_val
        del tweets_train_val
        del graphs_train_val
        # use here, because a signficant amount of memory can be reclaimed
        gc.collect()
        
        # create a dataLoader object
        train_dataset = customDataset(graphs_train, tweets_train, macds_train, y_train)
        val_dataset = customDataset(graphs_val, tweets_val, macds_val, y_val)
        test_dataset = customDataset(graphs_test, tweets_test, macds_test, y_test)

        # pass these to our training loop
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)

        # clear up memory again
        del graphs_train, tweets_train, macds_train, graphs_val, tweets_val, macds_val, graphs_test, tweets_test, macds_test
        gc.collect()
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
        tweets_train_val, tweets_test, prices_train_val, prices_test, y_train_val, y_test = train_test_split(
            tweets, prices, labels, test_size=0.2, random_state=42)
        del tweets
        del prices 
        del labels
        gc.collect()
        tweets_train, tweets_val, prices_train, prices_val, y_train, y_val= train_test_split(
            tweets_train_val, prices_train_val, y_train_val, test_size=0.25, random_state=42) 
        del prices_train_val 
        del y_train_val
        del tweets_train_val
        gc.collect()
        train_dataset = customDataset_stocknet(tweets_train, prices_train, y_train)
        val_dataset = customDataset_stocknet(tweets_val, prices_val, y_val)
        test_dataset = customDataset_stocknet(tweets_test, prices_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)
        del tweets_train, prices_train, prices_val, tweets_val, tweets_test, prices_test, y_train, y_val, y_test
        gc.collect()

    print('Data prepared.')

    params = {

            # DATA
            'dataset':args.dataset, 
            'train_loader':train_loader, 
            'val_loader':val_loader,
            'test_loader':test_loader,
            'test_model':args.test_model,

            'stoppage':args.stoppage,
            'lr': args.learning_rate,
            'run_id':args.run_id,
            'file_path': args.file_path,
            'pretrained_model': args.pretrained_model,

            # Epochs
            'epoch': args.epoch,
            'num_epochs' : args.num_epochs, 
            'early_stopping':args.early_stopping,

            'optimizer': optimizer,
            'optimizer_name':args.optimizer,
            'train_batch_size': args.train_batch_size,
            'eval_batch_size':args.eval_batch_size,
            'test_batch_size':args.test_batch_size,
            'model':model,
            'debug':args.debug,
            'dim':args.dimension,
            'dropout':args.dropout,
            'num_layers':args.num_layers,
            'lr_scheduler':lr_scheduler,
            'lrst':args.learning_rate_scheduler_type,
            'classes':args.num_classes,
            'tokenizer':tokenizer,
            'model_name':args.model_name,
            'num_encoders':args.num_encoders
    }

    train = meant_trainer(params)
    train.train()
    print('Done in ' +  str(time.time() - t0) + ' seconds.')