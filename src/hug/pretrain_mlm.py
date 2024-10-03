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
    #DebugUnderflowOverflow,
)
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset, Dataset

from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from torch.nn.parallel import DataParallel


torch.autograd.set_detect_anomaly(True)
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


class CustomData(Dataset):
    def __init__(self, dataset, split, tokenizer, max_length=512, mlm_probability=0.15):
        # Select the split of the dataset
        if split is not None:
            self.dataset = dataset[split]
        else:
            self.dataset = dataset
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the text item from the dataset at the specified index
        text = self.dataset[idx]
        # Tokenize the text; this does NOT yet apply MLM
        tokens = self.tokenizer(text, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        # Prepare inputs for MLM
        inputs, labels = self.mask_tokens(tokens['input_ids'])
        return {'input_ids': inputs, 'labels': labels, 'attention_mask': tokens['attention_mask']}

    def mask_tokens(self, inputs):
        """Prepare masked tokens inputs/labels for masked language modeling."""
        # Clone the input_ids to use as labels for MLM
        labels = inputs.clone()
        # Create a mask array
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # Apply special token mask: do not mask special tokens
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        # Mask tokens with probability `mlm_probability`
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # Replace masked input tokens with tokenizer.mask_token_id
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        return inputs, labels

# simple wrapper to allow our model to pretrain on the masked language modeling schema
class roberta_mlm_wrapper(nn.Module):
    def __init__(self, roberta, input_dim=768, output_dim=512):
        """
        Simple pretraining wrapper for the roberta model
        """
        super(roberta_mlm_wrapper, self).__init__()
        self.roberta = roberta
        self.mlm_output_head = nn.Sequential(nn.GELU(), nn.Linear(input_dim, 1))

    def forward(self, inputs):
        intermediate_val = self.roberta(**inputs)
        # project the last hidden state to the dimension of one
        outputs = self.mlm_output_head(intermediate_val['last_hidden_state'])
        return outputs.squeeze(dim=2)

# generalize training loop

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
        self.data = params['data'] 
        self.batch_size = params['batch_size']

        # epochs
        self.epoch = params['epoch']
        self.num_epochs = params['num_epochs']

        # for debugging
        self.debug_overflow = params['debug']
        self.track = params['track']

        # model specific         
        self.model = params['model']
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

        loss_fct = nn.CrossEntropyLoss()

        t0 = time.time()
        training_loss = []
        self.model.train()
    
        scaler = torch.cuda.amp.GradScaler()
        global_step=0

        for ep in range(self.num_epochs):
            final_epoch = ep
            target_values = []
            print('Training model on epoch ' + str(self.epoch + ep))

            progress_bar = tqdm(self.data, desc=f'Epoch {ep+1}/{self.num_epochs}')
            for batch in progress_bar:
                self.optimizer.zero_grad() 
                with torch.autocast(device_type="cuda", dtype=torch_dtype):
                    inputs = {'input_ids':batch['input_ids'].squeeze(dim=1).to(device), 'attention_mask':batch['attention_mask'].squeeze(dim=1).to(device)}
                    out = self.model(inputs)
                    target = batch['labels'].squeeze(dim=1)
                    loss = loss_fct(out, target.float().to(device))
                print('loss', loss)
                writer.add_scalar("charts/loss", loss, global_step)
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

            print('length: ', str(time.time() - t0))
            print('loss total: ', sum(training_loss))

        torch.save(self.model, self.file_path + '/models/' + self.model_name + '/' + self.model_name + '_' +  self.dataset + '_' + self.run_id + '_' + str(final_epoch + 1) + '.pt')
        torch.save(self.optimizer.state_dict(), self.file_path + '/optimizers/' +  self.optimizer_name + '/' + self.model_name + '_' + self.run_id + '_' + str(args.learning_rate) + '_' + str(self.epoch + 1) + '.pt')
        torch.save(self.lr_scheduler.state_dict(), self.file_path + '/lr_schedulers/' + self.lrst + '/' + self.model_name + '_' +  self.run_id + '_' + str(self.epoch + 1) + '.pt')
        

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
    parser.add_argument('-ne', '--num_epochs', type=int, help = 'Number of epochs to run training loop', default=10)
    parser.add_argument('-es', '--early_stopping', type=str2bool, help = 'Early stopping is active', nargs='?', const=False, default=False)
    parser.add_argument('-s', '--stoppage', type=float, help='Stoppage value', default=1e-4)
    parser.add_argument('-b', '--batch_size',type=int, help='Batch size for pretraining', default=16)
    parser.add_argument('-testm', '--test_model', type=str2bool, help='Whether or not to test our model', nargs='?', const=True, default=True)
    parser.add_argument('-dn', '--dataset_name', type=str, help='Name of dataset', default='stmhd')
    parser.add_argument('-tr', '--track', type=str2bool, help='Track with weights and biases', nargs='?', const=False, default=False)

    # Model specific
    parser.add_argument('-mn', '--model_name', type=str, help='Model name', default='roberta_mlm')
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
    parser.add_argument('-hf', '--hugging_face_model', type=str2bool, help='If we want to finetune/pretrain a model from Hugging face.', nargs='?', const=True, default=True)
    parser.add_argument('-hfd', '--hugging_face_data', type=str, help='Data set to load from Hugging Face', default=None)
    parser.add_argument('-hft', '--hugging_face_tokenizer', type=str, help='HuggingFace tokenizer', default=None)

    # Miscellaneous
    parser.add_argument('-db', '--debug', type = bool, help = 'Debug underflow and overflow', default = False)
    parser.add_argument('-fp', '--file_path', type=str, help='Path to files', default='/work/nlp/b.irving/nlp_files')
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

    bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
    if(args.epoch == 0):
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
                    model = AutoModel.from_config(config)
                    model = roberta_mlm_wrapper(model).to(device)
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

    # here is out pretrain data
    pretrain_data_path = '/work/nlp/b.irving/annika/final_data/pretrain.parquet'
    data = pd.read_parquet(pretrain_data_path)
    data_list = data['text'].iloc[:].tolist()

    # appending anchor tweets to timeline data
    csv_file_path = '/work/nlp/b.irving/annika/anchor.csv'
    csv_data = []
    with open(csv_file_path, mode='r', newline='') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            csv_data.append(row)
    for row in csv_data:
        data_list.append(row['tweet'])

    # create custom dataset from combined data
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    to_load = CustomData(data_list, None, tokenizer)
    toProcess = DataLoader(to_load, shuffle=True, batch_size=args.batch_size, pin_memory=True)
    
    # then delete the data that we don't need to pass 
    del data
    del data_list
    del csv_data
    del to_load
    gc.collect()

    print('Data loaded')

    params = {

            # DATA
            'dataset':args.dataset_name, 

            # training loop
            'lr': args.learning_rate,
            'run_id':args.run_id,
            'file_path': args.file_path,
            'pretrained_model': args.pretrained_model,
            'track':args.track,

            #data 
            'data':toProcess,

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
            'num_encoders':args.num_encoders

    }

    train = mlm_pretrainer(params)
    train.train()
    print('Done in ' +  str(time.time() - t0) + ' seconds.')