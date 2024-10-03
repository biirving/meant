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
    BertModel,
    AutoModelForAudioClassification
    #DebugUnderflowOverflow,
)
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset, Dataset

sys.path.append('..')
from src.meant.meant import meant
from src.meant.meant_vision import meant_vision
from src.meant.meant_tweet import meant_tweet
from src.meant.meant_tweet_no_lag import meant_tweet_no_lag
from src.meant.meant_tweet_price import meantTweetPrice
from src.meant.meantPrice import meant_price
from src.meant.hf_wrapper import vl_BERT_Wrapper, ViltWrapper, bertweet_wrapper
from src.utils.f1_metrics import f1_metrics
from src.utils.torchUtils import save_confusion_matrix
from src.meant.simple_mlp import mlpEncoder, LSTMEncoder
from src.meant.meant_timesformer import meant_timesformer
from src.meant.timesformer_pytorch import TimeSformer
from src.meant.meant_mean_pooling import meant_mean_pooling
from src.meant.meant_mosi import meant_mosi
from src.utils.custom_datasets import (
    djia_lag_dataset, 
    lag_text_collator, 
    lag_image_collator, 
    lag_text_image_collator, 
    tempstock_lag_dataset,
    lag_price_collator,
    stocknet_dataset, 
    mosi_dataset,
    lag_text_image_collator_no_lag)

from src.pretrain_mlm import meant_language_pretrainer
from src.pretrain_mim import meant_vision_pretrainer

from torch.utils.tensorboard import SummaryWriter
import wandb
import pandas as pd

sys.path.append('../michinaga/teanet/models')
sys.path.append('../michinaga/teanet/utils/')
from teanet import teanet

import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

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
    def __init__(self, graphs, tweets, macds, attention_masks,  labels):
        self.graphs = torch.tensor(graphs)
        self.tweets = torch.tensor(tweets)
        self.macds = torch.tensor(macds)
        self.attention_masks = torch.tensor(attention_masks)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.graphs[idx], self.tweets[idx], self.macds[idx], self.attention_masks[idx], self.labels[idx]

# simple class to help load the dataset
class customDataset_stocknet(Dataset):
    def __init__(self, tweets, prices, attention_masks, labels):
        self.tweets = torch.tensor(tweets)
        self.prices = torch.tensor(prices)
        self.labels = torch.tensor(labels)
        self.attention_masks = torch.tensor(attention_masks)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.tweets[idx], self.prices[idx], self.attention_masks[idx], self.labels[idx]

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
        self.tokenizer = params['tokenizer']
        self.use_images=params['use_images']
        self.use_tweets=params['use_tweets']
        self.use_prices=params['use_prices']
        self.use_lag=params['use_lag']
        self.collate_fn=params['collate_fn']
        self.dataset_class=params['dataset_class']

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

    def train(self):

        if(self.debug_overflow):
            debug_overflow = DebugUnderflowOverflow(self.model)
        loss_fct = nn.CrossEntropyLoss()
        t0 = time.time()
        training_loss = []
        total_acc = 0
        prev_f1 = 0
        final_epoch = 0
    
        # for early stopping
        lost_patience = 5
        patience = 0
        prev_f1 = float('inf')

        # How to load the train, val, and test partitions?
        dataset = self.dataset_class

        dataset_args = {'data':self.train_loader, 'tokenizer':self.tokenizer, 
            'use_images':self.use_images, 
            'use_tweets':self.use_tweets, 
            'use_prices':self.use_prices, 'use_lag':self.use_lag}
        train_dataset = dataset(**dataset_args)
        self.train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True, collate_fn=self.collate_fn, pin_memory=True)
        del train_dataset

        val_dataset_args = {'data':self.val_loader, 'tokenizer':self.tokenizer, 
            'use_images':self.use_images, 
            'use_tweets':self.use_tweets, 
            'use_prices':self.use_prices, 'use_lag':self.use_lag}
        val_dataset = dataset(**val_dataset_args)

        self.val_loader = DataLoader(val_dataset, batch_size=self.eval_batch_size, 
        shuffle=True, collate_fn=self.collate_fn, pin_memory=True)
        del val_dataset

        accumulation_steps=1
        accumulation_counter=0

        scaler = torch.cuda.amp.GradScaler()
        for ep in range(self.num_epochs):
            self.model.train()
            final_epoch = ep
            predictions = []
            target_values = []
            train_metrics = f1_metrics(self.num_classes, 'train', self.dataset) 
            train_f1_scores = []
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {ep+1}/{self.num_epochs}')
            for batch in progress_bar:
                accumulation_counter += 1
                batch = {key: value.to('cuda') for key, value in batch.items()}
                if 'prices' in list(batch.keys()):
                    batch['prices'] = batch['prices'].squeeze(dim=1).half()
                self.optimizer.zero_grad() 
                with torch.autocast(device_type="cuda", dtype=torch_dtype):
                    out = self.model.forward(**batch)
                    loss = loss_fct(out, batch['labels'].long())                
                # + 1 prevents update on first pass!
                if (accumulation_counter + 1) % accumulation_steps == 0:
                    accumulation_counter = 0
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                out = out.detach().cpu()
                train_metrics.update(out, batch['labels'].detach().cpu()) 
            print('length: ', str(time.time() - t0))
            print('loss total: ', sum(training_loss))
            train_metrics.show() 
            self.lr_scheduler.step()

            val_metrics = f1_metrics(self.num_classes, 'validation', self.dataset) 
            self.model.eval()
            val_loss = 0
            print('Evaluating Model...')
            with torch.no_grad():
                val_progress_bar = tqdm(self.val_loader, desc=f'Epoch {ep+1}/{self.num_epochs}')
                for batch in val_progress_bar:
                    batch = {key: value.to('cuda') for key, value in batch.items()}
                    if 'prices' in list(batch.keys()):
                        batch['prices'] = batch['prices'].squeeze(dim=1).half()
                    with torch.autocast(device_type="cuda", dtype=torch_dtype):
                        out = self.model.forward(**batch)
                    val_metrics.update(out.detach().cpu(), batch['labels'].detach().cpu()) 

            val_f1_macro, val_f1_micro, non_zero_f1 = val_metrics.show() 
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
            torch.save(self.model, self.file_path + '/models/' + self.model_name + '/' + self.model_name + '_' + str(self.num_encoders) + '_' +  self.dataset + '_' + str(self.run_id) + '_' + str(final_epoch + 1) + '.pt')
            #torch.save(self.optimizer.state_dict(), self.file_path + '/optimizers/' +  self.optimizer_name + '/' + self.model_name + '_' + self.run_id + '_' + str(args.learning_rate) + '_' + str(self.epoch + 1) + '.pt')
            #torch.save(self.lr_scheduler.state_dict(), self.file_path + '/lr_schedulers/' + self.lrst + '/' + self.model_name + '_' +  self.run_id + '_' + str(self.epoch + 1) + '.pt')
        except FileNotFoundError:
            print('Your filepath is invalid. Save has failed')
        
        del self.train_loader
        del self.val_loader

        if(self.test_model):
            print('Testing...') 
            test_metrics = f1_metrics(self.num_classes, 'test', self.dataset) 
            test_progress_bar = tqdm(self.test_loader, desc=f'Epoch {ep+1}/{self.num_epochs}')
            self.model.eval()
            f1_scores = []
            with torch.no_grad():
                # Should use kwargs here
                test_dataset_args = {'data':self.test_loader, 'tokenizer':self.tokenizer, 
                    'use_images':self.use_images, 
                    'use_tweets':self.use_tweets, 
                    'use_prices':self.use_prices, 'use_lag':self.use_lag}
                test_dataset = self.dataset_class(**test_dataset_args)

                test_loader = DataLoader(test_dataset, batch_size=self.test_batch_size, 
                shuffle=True, collate_fn=self.collate_fn, pin_memory=True)
                del test_dataset 

                all_true_labels = []
                all_predicted_labels = []
                test_progress_bar = tqdm(test_loader, desc=f'Epoch {ep+1}/{self.num_epochs}')
                for batch in test_progress_bar:
                    batch = {key: value.to('cuda') for key, value in batch.items()}
                    if 'prices' in list(batch.keys()):
                        batch['prices'] = batch['prices'].squeeze(dim=1).half()
                    with torch.autocast(device_type="cuda", dtype=torch_dtype):
                        out = self.model.forward(**batch)
                    test_metrics.update(out.detach().cpu(), batch['labels'].detach().cpu()) 
                    # Extract true labels and predicted labels for confusion matrix
                    true_labels = batch['labels'].cpu().numpy()
                    _, predicted_labels = torch.max(out, 1)
                    predicted_labels = predicted_labels.cpu().numpy()
                    all_true_labels.extend(true_labels)
                    all_predicted_labels.extend(predicted_labels)
                # Save the confusion matrix after the loop
                save_path = '../../meant/confusion_matrices/' + self.model_name + '_' + str(self.num_encoders) + '_' +  self.dataset + '_' + str(self.run_id) + '_' + str(final_epoch + 1) + '.png'
                class_names = ['Sell Signal (0)', 'Buy signal (1)']
                save_confusion_matrix(all_true_labels, all_predicted_labels, class_names, save_path)
            test_metrics.show()  

            # Assuming tempstock_lag_dataset and other necessary components are defined elsewhere



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
    parser.add_argument('-ds', '--dataset', type=str, help='The dataset that we are training on', default='TempStockLarge')
    args = parser.parse_args()

    t0 = time.time()
    
    # the model most be loaded first, in order to support the instantiation of the optimizer and the learning rate scheduler
    if args.dataset == 'Stocknet':
        price_dim=3
    elif args.dataset == 'TempStockLarge':
        price_dim=5
    elif args.dataset == 'djiaNews':
        price_dim=3
    elif args.dataset == 'mosi':
        price_dim=0


    bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
    if(args.epoch == 0):
        if args.hugging_face_model is True:
            
            use_lag = False
            if args.pretrained is True:
                model = AutoModelForTokenClassification.from_pretrained(args.hugging_face_model).to(device)
            else: 
                print('Training model from scratch')
                if args.model_name == 'vl_bert':
                    # use pretrained model for TESTING 
                    #vl_bert_model = VisualBertModel._from_config(config).cuda()
                    # using pretrained VisualBERT
                    # rerun these experiments
                    # fuck it lets move on
                    config = AutoConfig.from_pretrained('configs/' + args.model_name +'.json', local_files_only=True)
                    vl_bert_model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
                    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
                    #vl_bert_model.embeddings.word_embeddings = bertweet.embeddings.word_embeddings
                    model = vl_BERT_Wrapper(vl_bert_model, 768, 2).cuda()
                    use_images = True
                    use_tweets = True
                    use_prices = False
                    collate_fn = lag_text_image_collator
                elif args.model_name == 'vilt':
                    config = AutoConfig.from_pretrained('configs/vilt.json', local_files_only=True)
                    #vilt = ViltModel._from_config(config)
                    # using pretrained ViLT
                    # rerun these experiments
                    config = AutoConfig.from_pretrained('configs/' + args.model_name +'.json', local_files_only=True)
                    vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
                    tokenizer = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm", do_rescale=False)
                    model = ViltWrapper(vilt, 768, 2).to(device) 
                    use_images = True
                    use_tweets = True
                    use_prices = False
                    collate_fn = lag_text_image_collator
                elif args.model_name == 'bertweet':
                    config = AutoConfig.from_pretrained('configs/' + args.model_name +'.json', local_files_only=True)
                    model = bertweet_wrapper(bertweet, 768, 2).to(device)
                    tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
                    use_images = False
                    use_tweets = True
                    use_prices = False
                    collate_fn = lag_text_collator
                elif args.model_name == 'bert':
                    model = BertModel.from_pretrained("bert-base-uncased").cuda()
                    model = bertweet_wrapper(model, 768, 2)
                    #model.bertweet.embeddings = bertweet.embeddings
                    model = model.cuda()
                    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                    use_images = False
                    use_tweets = True
                    use_prices = False
                    collate_fn = lag_text_collator
                elif args.model_name == 'finbert':
                    # Also, do we want to employ comparisons to an unpretrained model?
                    # We need to pretrain our language encoder on a wider dataset
                   
                    # we use the bertweet embeddings because that is how the tweets were prepared
                    # so fuck this strategy, we might as well collect all of our runs again!
                    fin_bert = AutoModel.from_pretrained('ProsusAI/finbert')
                    model = bertweet_wrapper(fin_bert, 768, 2)
                    #model.bertweet.embeddings = bertweet.embeddings
                    model = model.cuda()
                    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
                    use_images = False
                    use_tweets = True
                    use_prices = False
                    collate_fn = lag_text_collator
                
                else:
                    print('Hugging face model not defined!')
        elif args.model_name == 'meant':
            fin_bert = AutoModel.from_pretrained('ProsusAI/finbert')
            tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
            model = meant(text_dim = 768, 
                image_dim = 768, 
                price_dim = price_dim, 
                height = 224, 
                width = 224, 
                patch_res = 16, 
                lag = args.lag, 
                num_classes = args.num_classes, 
                embedding = fin_bert.embeddings,
                flash=False,
                num_encoders=args.num_encoders).to(device)
            if args.num_encoders == 24:
                print('Meant x large')
            elif args.num_encoders == 12:
                print('meant large')
            elif args.num_encoders == 1:
                print('Meant base')

            if args.pretrained:
                print('pretrained')
                if args.num_encoders == 12:
                    print('Meant large')
                    language_encoders = torch.load('..').to(device)
                elif args.num_encoders == 24:
                    print('Meant x large')
                    language_encoders = torch.load('..').to(device)
                elif args.num_encoders == 1:
                    print('Meant base')
                    language_encoders = torch.load('..').to(device)
                pretrained_vision = torch.load('..').to(device)
                model.languageEncoders = language_encoders.languageEncoders
                model.visionEncoders = pretrained_vision.visionEncoders
                del pretrained_vision
                del language_encoders
                gc.collect()
            use_images = True
            use_tweets = True
            use_prices = True
            use_lag = True
            collate_fn = lag_text_image_collator
        elif args.model_name == 'meant_mean_pooling':
            fin_bert = AutoModel.from_pretrained('ProsusAI/finbert')
            tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
            model = meant_mean_pooling(text_dim = 768, 
                image_dim = 768, 
                price_dim = price_dim, 
                height = 224, 
                width = 224, 
                patch_res = 16, 
                lag = args.lag, 
                num_classes = args.num_classes, 
                embedding = fin_bert.embeddings,
                flash=False,
                num_encoders=args.num_encoders).to(device)
            use_images = True
            use_tweets = True 
            use_prices = True
            use_lag = True
            collate_fn = lag_text_image_collator
        elif args.model_name == 'meant_vision':
            model = meant_vision(
                image_dim = 768, 
                price_dim = price_dim, 
                height = 224, 
                width = 224, 
                patch_res = 16, 
                lag = args.lag, 
                num_classes = args.num_classes, 
                flash=True, 
                num_encoders=args.num_encoders).to(device)
            use_images = True
            use_tweets = False
            use_prices = True 
            use_lag = True
            collate_fn = lag_image_collator
            tokenizer = None
        elif args.model_name == 'meant_tweet':

            fin_bert = AutoModel.from_pretrained('ProsusAI/finbert')
            tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
            #tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
            model = meant_tweet(text_dim = 768, 
                price_dim = price_dim, 
                lag = args.lag, 
                num_classes = args.num_classes, 
                flash=True,
                embedding = fin_bert.embeddings,
                num_encoders=args.num_encoders).to(device) 
            use_images = False
            use_tweets = True
            use_prices = True 
            use_lag = True
            collate_fn = lag_text_collator
            if args.num_encoders == 12:
                language_encoders = torch.load('..').to(device)
            elif args.num_encoders == 24:
                language_encoders = torch.load('..').to(device)
            elif args.num_encoders == 1:
                language_encoders = torch.load('..').to(device)
            model.languageEncoders = language_encoders.languageEncoders
            del language_encoders 
            gc.collect()
        elif args.model_name == 'meant_tweet_no_lag':
            tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
            # do we need the embedding layer if we have already used the flair nlp embeddings?
            model = meant_tweet_no_lag(
                text_dim=768,
                num_classes = args.num_classes,
                embedding = bertweet.embeddings,
            ).to(device)
            use_images = False
            use_tweets = True
            use_lag = False
            use_prices = False
            collate_fn = lag_text_collator
        elif args.model_name == 'teanet':
            print('teanet')
            fin_bert = AutoModel.from_pretrained('ProsusAI/finbert')
            model = teanet(5, 512, 2, 5, 12, 10, embedding=fin_bert.embeddings).cuda()
            tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
            use_images = False
            use_tweets = True
            use_prices = True
            use_lag = True
            collate_fn = lag_text_collator
        elif args.model_name == 'meant_tweet_price':
            model = meantTweetPrice(text_dim=768, 
            price_dim=price_dim,
            lag=args.lag,
            num_classes=args.num_classes,
            flash=False,
            embedding = bertweet.embeddings,
            num_encoders=args.num_encoders).to(device)
            use_images = False
            use_tweets = True
            use_prices = True
            use_lag = True
            collate_fn = lag_text_collator
            tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
            if args.pretrained:
                if args.num_encoders == 12:
                    language_encoders = torch.load('..').to(device)
                elif args.num_encoders == 24:
                    language_encoders = torch.load('..').to(device)
                elif args.num_encoders == 1:
                    language_encoders = torch.load('..').to(device)
                model.languageEncoders = language_encoders.languageEncoders
        elif args.model_name == 'meant_price':
            model = meant_price(
            price_dim=5,
            lag=args.lag,
            num_classes=args.num_classes,
            num_encoders=args.num_encoders).to(device)
            use_images = False
            use_tweets = False 
            use_prices = True
            use_lag = True
            collate_fn = lag_price_collator 
            tokenizer = None
        elif args.model_name == 'simple_mlp_no_lag':
            model = mlpEncoder(
                input_dim=5,
                output_dim=args.num_classes,
                hidden_dim=64,
                num_hidden_layers=1000
            ).cuda()
            total_params = sum(p.numel() for p in model.parameters())
            print('total params', total_params)
            use_prices=True
            use_images=False
            use_lag=False
            use_tweets=False
            collate_fn = lag_price_collator
            tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
        elif args.model_name == 'lstm':
            model = LSTMEncoder(
                input_dim=5,
                output_dim=args.num_classes,
                hidden_dim=64,
                num_hidden_layers=5000
            ).cuda()
            use_prices=True
            use_images=False
            use_lag=False
            use_tweets=False
            collate_fn = lag_price_collator 
            tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
        elif args.model_name == 'meant_timesformer':
            #tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
            #bert = AutoModel.from_pretrained("roberta-large")
            #tokenizer = AutoTokenizer.from_pretrained('roberta-large')

            fin_bert = AutoModel.from_pretrained('ProsusAI/finbert')
            tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
            # do we need the embedding layer if we have already used the flair nlp embeddings?
            model = meant_timesformer(text_dim = 768, 
                image_dim = 768, 
                price_dim = price_dim, 
                height = 224, 
                width = 224, 
                patch_res = 16, 
                lag = args.lag, 
                num_classes = args.num_classes, 
                embedding = fin_bert.embeddings,
                flash=False,
                num_encoders=args.num_encoders).to(device)
            if args.num_encoders == 24:
                print('Meant x large')
            elif args.num_encoders == 12:
                print('meant large')
            elif args.num_encoders == 1:
                print('Meant base')
            if args.pretrained:
                print('pretrained')
                if args.num_encoders == 12:
                    print('Meant large')
                    language_encoders = torch.load('')
                elif args.num_encoders == 24:
                    print('Meant x large')
                    language_encoders = torch.load('')
                elif args.num_encoders == 1:
                    print('Meant base')
                    language_encoders = torch.load('')
                model.languageEncoders = language_encoders.languageEncoders
                del language_encoders
                gc.collect()
            use_images = True
            use_tweets = True
            use_prices = True
            use_lag = True
            collate_fn = lag_text_image_collator
        elif args.model_name == 'timesformer':
            model = TimeSformer(dim=768, 
            image_size=224,
            patch_size=16,
            num_frames=args.lag, 
            num_classes=args.num_classes,
            depth=1,
            heads=8,
            dim_head=64,
            attn_dropout = 0.1,
            ff_dropout = 0.1
            ).cuda()
            use_images = True
            use_tweets = False 
            use_prices = False
            use_lag = True
            collate_fn = lag_image_collator
            tokenizer = None
        elif args.model_name == 'meant_mosi':
            #audio_model = AutoModelForAudioClassification.from_pretrained(
            #"facebook/wav2vec2-base", num_labels=2)
            fin_bert = AutoModel.from_pretrained('ProsusAI/finbert')
            tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
            model = meant_mosi(text_dim = 768, 
                image_dim = 768, 
                height = 20, 
                width = 1, 
                patch_res = 1, 
                lag = 50, 
                num_classes = args.num_classes, 
                flash=False,
                embedding=fin_bert.embeddings,
                num_encoders=args.num_encoders).to(device)
                #audio_embeddings=audio_model.wav2vec2.encoder.pos_conv_embed).to(device)
            use_images = True
            use_tweets = True
            use_prices = False 
            use_lag = True
            collate_fn = lag_text_image_collator_no_lag
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

    if args.dataset == 'TempstockSmall':
        assert not (args.image_only == True and args.language_only == True), 'Cannot be an image only AND a language only task'
        if args.image_only:
            graphs = np.memmap('..', dtype=np_dtype, mode='r')
            tweets = np.ones(graphs.shape[0], 1).astype(np.float32)
        elif args.language_only:
            tweets = np.load('..', dtype=np_dtype, mode='r')
            attention_masks = np.load('..')
            graphs = np.ones(tweets.shape[0], 1).astype(np.float32)
        else:
            graphs = np.load('..')
            tweets = np.load('..')
            attention_masks = np.load('..')
            macds = np.load('..')
            labels = np.load('..')
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
        graphs_train_val, graphs_test, tweets_train_val, tweets_test, macds_train_val, macds_test, attention_masks_train_val, attention_masks_test, y_train_val, y_test = train_test_split(
            graphs, tweets, macds, attention_masks, labels, test_size=0.2, random_state=42)

        # clear up memory
        del graphs
        del tweets
        del macds
        del labels
        del attention_masks
        gc.collect()

        # Second split: Split the remaining data into training and validation sets
        graphs_train, graphs_val, tweets_train, tweets_val, macds_train, macds_val, attention_masks_train, attention_masks_val, y_train, y_val= train_test_split(
            graphs_train_val, tweets_train_val, macds_train_val, attention_masks_train_val, y_train_val, test_size=0.25, random_state=42) 
        
        del macds_train_val
        del y_train_val
        del tweets_train_val
        del graphs_train_val
        del attention_masks_train_val
        # use here, because a signficant amount of memory can be reclaimed
        gc.collect()
        
        # create a dataLoader object
        train_dataset = customDataset(graphs_train, tweets_train, macds_train, attention_masks_train, y_train)
        val_dataset = customDataset(graphs_val, tweets_val, macds_val, attention_masks_val, y_val)
        test_dataset = customDataset(graphs_test, tweets_test, macds_test, attention_masks_test, y_test)

        # pass these to our training loop
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)

        # clear up memory again
        del graphs_train, tweets_train, macds_train, graphs_val, tweets_val, macds_val, graphs_test, tweets_test, macds_test
        gc.collect()
    elif args.dataset == 'Stocknet':
        train_loader =''
        val_loader = ''
        test_loader = ''
        dataset_class = stocknet_dataset
    elif args.dataset == 'djiaNews':
        # why doesn't this work either? Worrying...

        #djia_news_df = pd.read_csv('djia_news_final.csv')

        # Split the dataset into train (80%), validation (10%), and test (10%)
        #train_df, temp_df = train_test_split(djia_news_df, test_size=0.2, random_state=42)
        #val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        # Save the datasets to separate CSV files
        #train_df.to_csv('djia_news_train.csv', index=False)
        #val_df.to_csv('djia_news_val.csv', index=False)
        #test_df.to_csv('djia_news_test.csv', index=False)
        train_loader = ''
        val_loader = ''
        test_loader = ''
        dataset_class = djia_lag_dataset

    elif args.dataset == 'TempStockLarge':
        if not use_tweets and not use_prices and not use_images:
            raise ValueError('Not passing any data forward. Please use the tweets, graphs, or prices.')
        if use_tweets or use_prices:
            train_data =''
            val_data = ''
            test_data = ''
        else:
            train_data = None
            val_data = None
            test_data = None

        if use_images:
            train_graphs =''
            val_graphs = ''
            test_graphs = ''
        else:
            train_graphs = None
            val_graphs = None
            test_graphs = None

        train_loader = {'data':train_data, 'graphs':train_graphs, 'labels':''}
        val_loader = {'data':val_data, 'graphs':val_graphs, 'labels':''}
        test_loader = {'data':test_data, 'graphs':test_graphs, 'labels':''}
        dataset_class = tempstock_lag_dataset

    # I am not like the others
    elif args.dataset == 'mosi':
        # jank preprocess
        def drop_entry(dataset):
            """Drop entries where there's no text in the data."""
            drop = []
            for ind, k in enumerate(dataset["text"]):
                if k.sum() == 0:
                    drop.append(ind)
            for modality in list(dataset.keys()):
                dataset[modality] = np.delete(dataset[modality], drop, 0)
            return dataset
        filepath=''
        with open(filepath, "rb") as f:
            alldata = pickle.load(f)
        alldata['train'] = drop_entry(alldata['train'])
        print(alldata['train'].keys())
        sys.exit()
        alldata['valid'] = drop_entry(alldata['valid'])
        alldata['test'] = drop_entry(alldata['test'])
        train_loader = alldata['train']
        val_loader = alldata['valid']
        test_loader = alldata['test']
        dataset_class = mosi_dataset

    print('Data prepared.')
    print(args.model_name)

    params = {

            # DATA
            'dataset':args.dataset, 
            'train_loader':train_loader, 
            'val_loader':val_loader,
            'test_loader':test_loader,
            'test_model':args.test_model,
            'use_images':use_images,
            'use_tweets':use_tweets,
            'use_prices':use_prices,
            'use_lag':use_lag,
            'collate_fn':collate_fn,
            'dataset_class':dataset_class,

            'stoppage':args.stoppage,
            'lr': args.learning_rate,
            'run_id':args.run_id,
            'file_path': args.file_path,
            'pretrained_model': args.pretrained_model,
            'tokenizer':tokenizer,

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