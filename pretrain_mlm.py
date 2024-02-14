#!/usr/bin/env python

from transformers import AutoTokenizer, RobertaForMaskedLM, AutoConfig, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys, os, time
import tqdm as tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
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
    RobertaForMaskedLM
    #DebugUnderflowOverflow,
)
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from einops import repeat, rearrange

sys.path.append('../meant')
from meant import languageEncoder
from utils import mlm_dataset

torch.cuda.empty_cache()
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# detecting where nans originate from
#torch.autograd.set_detect_anomaly(True)

# ensure that this datatype is the same as what the arrays you load in are saved in if doing memmap
np_dtype = np.float64

# torch datatype to used for automatic mixed precision training
torch_dtype = torch.float16

# for argument parsing
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# we are not training these models to be temporally aware
# pretrain a word embedding or not?

class meant_language_pretrainer(nn.Module):
    def __init__(self, num_encoders, mlm_input_dim, embedding, lm_head, flash=False, lag=5, text_dim=768, num_heads=8):
        super(meant_language_pretrainer, self).__init__()
        self.embedding = nn.ModuleList([embedding])
        self.languageEncoders = nn.ModuleList([languageEncoder(text_dim, num_heads, flash=flash) for _ in range(num_encoders)])
        # my mlm head has to be the same size as the vocabulary list (come on son)
        self.mlm_head = lm_head 
        self.lag=lag

    def forward(self, words, attention_mask):
        for mod in self.embedding:
            words = mod(words)
        for encoder in self.languageEncoders:
            words = encoder.forward(words, attention_mask=attention_mask)
        return self.mlm_head(words)


class mlm_pretrainer():
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
        self.train_data = params['train_data'] 
        self.val_data = params['val_data']
        self.batch_size = params['batch_size']
        self.dataset = params['dataset']

        # epochs
        self.epoch = params['epoch']
        self.num_epochs = params['num_epochs']
        self.patience = params['patience']

        # for debugging
        self.debug_overflow = params['debug']
        self.track = params['track']

        # model specific         
        self.model = params['model']
        self.config = params['config']
        self.dimension = params['dim']
        self.num_layers = params['num_layers']
        self.dropout = params['dropout']
        self.pretrained_model = params['pretrained_model']
        self.num_encoders = params['num_encoders']

        self.lr_scheduler = params['lr_scheduler']
        self.tokenizer = params['tokenizer']
        self.lrst = params['lrst']
        self.file_path = params['file_path']
        self.model_name = params['model_name']


    def train(self):
        if self.track:
            import wandb
            wandb.init(project='stmhd_mlm',entity='Aurelian',sync_tensorboard=True,config=None,name=self.model_name,save_code=True) 
        writer = SummaryWriter(f"runs/{self.model_name}")

        if(self.debug_overflow):
            debug_overflow = DebugUnderflowOverflow(self.model)

        # throwing some nans
        loss_fct = nn.CrossEntropyLoss()

        training_loss = []
        self.model.train()
    
        scaler = torch.cuda.amp.GradScaler()
        global_step=0
        prev_val_loss = float('inf')

        # number of times our model has increased in validation loss
        lost_patience = 0

        for ep in range(self.num_epochs):
            final_epoch = ep
            target_values = []
            print('Training model on epoch ' + str(self.epoch + ep))
            t0 = time.time()
            progress_bar = tqdm(self.train_data, desc=f'Epoch {ep+1}/{self.num_epochs}')
            for batch in progress_bar:
                self.optimizer.zero_grad() 
                input_ids = batch['input_ids'].squeeze(dim=1).to(device)
                attention_mask = batch['attention_mask'].squeeze(dim=1).to(device)
                labels = batch['labels'].squeeze(dim=1).to(device)  # Assuming 'labels' are the masked labels
                with torch.autocast(device_type="cuda", dtype=torch_dtype):
                    if input_ids.shape[0] % self.batch_size != 0:
                        break 
                    out = self.model(input_ids, attention_mask=attention_mask)
                    target = batch['labels'].squeeze(dim=1).cuda()
                    loss = loss_fct(out.view(-1, self.config.vocab_size), target.view(-1))
                writer.add_scalar("charts/loss", loss.item(), global_step)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()

                # take the tensors off the gpu
                out = out.detach().cpu()
                target = target.detach().cpu()
                # clean up memory
                del out
                del loss
                global_step+=self.batch_size
            
            print('epoch length:', str(time.time() - t0))
            self.lr_scheduler.step()
            val_progress_bar = tqdm(self.val_data, desc=f'Epoch {ep+1}/{self.num_epochs}')
            self.model.eval()
            val_step = 0
            val_loss = 0
            with torch.no_grad():
                for batch in val_progress_bar:
                    input_ids = batch['input_ids'].squeeze(dim=1).to(device)
                    attention_mask = batch['attention_mask'].squeeze(dim=1).to(device)
                    labels = batch['labels'].squeeze(dim=1).to(device)  # Assuming 'labels' are the masked labels
                    with torch.autocast(device_type="cuda", dtype=torch_dtype):
                        if input_ids.shape[0] % self.batch_size != 0:
                            break 
                        # we need both a casual and a attention mask
                        out = self.model(input_ids, attention_mask=attention_mask)
                        target = batch['labels'].squeeze(dim=1).cuda()
                        loss = loss_fct(out.view(-1, self.config.vocab_size), target.view(-1))
                        val_step += self.batch_size
                        # track the val loss with tensorboard
                        writer.add_scalar("charts/val_loss", loss, val_step)
                        val_loss += loss.item()

                if val_loss >= prev_val_loss:
                    lost_patience += 1
                    if lost_patience > self.patience:
                        print('Model is not improving. Exiting pretraining loop.')
                        break
                else:
                    prev_val_loss = val_loss

        torch.save(self.model, self.file_path + '/models/' + self.model_name + '/' + self.model_name + '_' + str(self.num_encoders) + '_' + self.dataset + '_' + str(self.run_id) + '_' + str(final_epoch + 1) + '.pt')
        torch.save(self.optimizer.state_dict(), self.file_path + '/optimizers/' +  self.optimizer_name + '/' + self.model_name + '_' + str(self.run_id) + '_' + str(args.learning_rate) + '_' + str(self.epoch + 1) + '.pt')
        torch.save(self.lr_scheduler.state_dict(), self.file_path + '/lr_schedulers/' + self.lrst + '/' + self.model_name + '_' +  str(self.run_id) + '_' + str(self.epoch + 1) + '.pt')

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
    parser.add_argument('-b', '--batch_size',type=int, help='Batch size for pretraining', default=16)
    parser.add_argument('-testm', '--test_model', type=str2bool, help='Whether or not to test our model', nargs='?', const=True, default=True)
    parser.add_argument('-dn', '--dataset_name', type=str, help='Name of dataset', default='tempstock')
    parser.add_argument('-tr', '--track', type=str2bool, help='Track with weights and biases', nargs='?', const=False, default=False)
    parser.add_argument('-pa', '--patience', type=int, help='Patience parameter for MLM training loop', default=3)

    # Model specific
    parser.add_argument('-mn', '--model_name', type=str, help='Model name', default='meant_language_encoder')
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
    parser.add_argument('-rid', '--run_id', type=str, help='Run identification number', default=0)
    parser.add_argument('-lag', '--lag', type=int, help='Lag period for data', default=5)
    parser.add_argument('-norm', '--normalize', type=str2bool, help='Whether or not to normalize the data', nargs='?', const=False, default=False)
    args = parser.parse_args()

    t0 = time.time()
    # the model most be loaded first, in order to support the instantiation of the optimizer and the learning rate scheduler

    # first check if we can run on multiple GPUs
    if torch.cuda.device_count() > 1:
        multi_gpu = True
    else:
        multi_gpu = False

    #bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
    bertweet_config = AutoConfig.from_pretrained('/work/nlp/b.irving/nlp/src/hug/configs/bertweet.json', local_files_only=True)
    bertweet = RobertaForMaskedLM._from_config(bertweet_config)
    if(args.epoch == 0):
        # what is the reason for this flag?
        if args.hugging_face_model is True:
            if args.pretrained is True:
                model = AutoModelForTokenClassification.from_pretrained(args.hugging_face_model).to(device)
            else: 
                print('Training model from scratch')
                config = AutoConfig.from_pretrained('/work/nlp/b.irving/nlp/src/hug/configs/' + args.model_name +'.json', local_files_only=True)
                if args.model_name == 'vl_bert':
                    vl_bert_model = VisualBertModel._from_config(config).cuda()
                elif args.model_name == 'vilt':
                    vilt = ViltModel._from_config(config)
                elif args.model_name == 'roberta_mlm':
                    config = AutoConfig.from_pretrained("/work/nlp/b.irving/nlp/src/hug/configs/roberta_mlm.json", output_hidden_states=True)
                    model = RobertaForMaskedLM._from_config(config).cuda()
        elif args.model_name == 'meant_language_encoder':
            # which are basically just robertabase embeddings
            config = bertweet_config
            embeddings = bertweet.roberta.embeddings
            lm_head = bertweet.lm_head
            # but how do you withdraw loss from frankenstein model?
            model = meant_language_pretrainer(args.num_encoders, 768, embeddings, lm_head).to(device) 
        elif args.model_name == 'meant_vision_encoder':
            raise ValueError('Unimplemented')
        else:
            raise ValueError('Pass a valid model name.')
    else:
        model = torch.load(args.file_path + '/models/' + args.model_name + '/' + args.model_name + '_' + args.run_id + '_' + str(args.epoch) + '.pt')
    
    if multi_gpu:
        model = DataParallel(model)

    
    # delete the bertweet model
    del bertweet
    gc.collect()

    if(args.optimizer == 'AdamW'):
        if multi_gpu:
            optimizer = torch.optim.AdamW(params = model.module.parameters(), lr=args.learning_rate, weight_decay=args.decay, betas=(args.beta_1, args.beta_2))
        else:
            optimizer = torch.optim.AdamW(params = model.parameters(), lr=args.learning_rate, weight_decay=args.decay, betas=(args.beta_1, args.beta_2))
    elif(args.optimizer == 'Adam'):
        if multi_gpu:
            optimizer = torch.optim.Adam(params = model.module.parameters(), lr=args.learning_rate, weight_decay=args.decay, betas=(args.beta_1, args.beta_2))
        else:
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

    print('Loading data...')

    pretrain_data_path = '/work/nlp/b.irving/stock/pretrain_dataset.parquet'
    data = pd.read_parquet(pretrain_data_path)
    data_list = data['text'].iloc[:].tolist()
    train_tweets, val_tweets = train_test_split(data_list, test_size=0.1, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    train = mlm_dataset(train_tweets, None, tokenizer, max_length=128)
    val = mlm_dataset(val_tweets, None, tokenizer, max_length=128) 
    train_loader = DataLoader(train, shuffle=True, batch_size=args.batch_size, pin_memory=True)
    val_loader = DataLoader(val, shuffle=True, batch_size=args.batch_size, pin_memory=True)
    
    # then delete the data that we don't need to pass 
    del data
    del data_list
    del train
    del val
    gc.collect()

    print('Data loaded')

    # is there a way to pretrain the temporal component of the model?

    params = {

            # DATA
            'dataset':args.dataset_name, 

            # training loop
            'lr': args.learning_rate,
            'run_id':args.run_id,
            'file_path': args.file_path,
            'pretrained_model': args.pretrained_model,
            'track':args.track,
            'patience':args.patience,

            #data 
            'train_data':train_loader,
            'val_data':val_loader,

            # Epochs
            'epoch': args.epoch,
            'num_epochs' : args.num_epochs, 

            'optimizer': optimizer,
            'optimizer_name':args.optimizer,
            'batch_size': args.batch_size,
            'model':model,
            'debug':args.debug,
            'dim':args.dimension,
            'dropout':args.dropout,
            'num_layers':args.num_layers,
            'lr_scheduler':lr_scheduler,
            'lrst':args.learning_rate_scheduler_type,
            'tokenizer':tokenizer,
            'model_name':args.model_name,
            'num_encoders':args.num_encoders,
            'config': config

    }

    train = mlm_pretrainer(params)
    train.train()
    print('Done in ' +  str(time.time() - t0) + ' seconds.')