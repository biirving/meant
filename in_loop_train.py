#!/usr/bin/env python
import csv
import pandas as pd
import ast
import torch
from torch import nn, tensor
import os, sys, argparse, time, gc, socket
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

#nightly pytorch version required
#import torch._dynamo as dynamo

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from torchmetrics import Accuracy, MatthewsCorrCoef, AUROC
import torchmetrics
import string
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, AutoModel
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall
import re
sys.path.append('../meant')
from meant import meant


torch.cuda.empty_cache()
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class metrics():
    def __init__(self, num_classes):
        accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro')
        f1_micro = MulticlassF1Score(num_classes=num_classes, average='micro')
        precision_macro = MulticlassPrecision(num_classes=num_classes, average='macro')
        precision_micro = MulticlassPrecision(num_classes=num_classes, average='micro')
        recall_macro = MulticlassRecall(num_classes=num_classes, averafe='macro')
        recall_micro = MulticlassRecall(num_classes=num_classes, average='micro')
    
    def update(self, pred, target):
        self.accuracy.update(pred, target) 
        self.f1_macro.update(pred, target) 
        self.f1_micro.update(pred, target)
        self.precision_macro.update(pred, target)
        self.precision_micro.update(pred, target)
        self.recall_macro.update(pred, target)
        self.recall_micro.update(pred, target)

    def compute(self):
        acc = self.accuracy.compute()
        f1_macro = self.f1_macro.compute()
        f1_micro = self.f1_micro.compute()
        precision_macro = self.precision_macro.compute()
        precision_micro = self.precision_micro.compute()
        recall_macro = self.precision_macro.compute()
        recall_micro = self.precision_micro.compute()
        return acc, f1_macro, f1_micro, precision_macro, precision_micro, recall_macro, recall_micro


# generalize training loop

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
        self.graphs_train = params['graphs_train']
        self.graphs_val = params['graphs_val']
        self.graphs_test = params['graphs_test']
        self.tweets_train = params['tweets_train']
        self.tweets_val = params['tweets_val']
        self.tweets_train = params['tweets_train']
        self.macds_train = params['macds_train']
        self.macds_val = params['macds_val']
        self.macds_test = params['macds_test']
        self.y_train = params['y_train']
        self.y_val = params['y_val']
        self.y_test = params['y_test']
        self.test_model = params['test_model']

        # model specific         
        self.model = params['model']
        self.dimension = params['dim']
        self.num_layers = params['num_layers']
        self.dropout = params['dropout']
        self.pretrained_model = params['pretrained_model']
        self.num_classes = params['classes']

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
        plt.savefig(self.file_path + '/output_files/' + self.dataset + '/plots/' + 'train_f1_' + self.model_name + '_' + self.run_id + '_' + str(self.epoch) + '.png')

    def train(self):
        if(self.debug_overflow):
            debug_overflow = DebugUnderflowOverflow(self.model)
        self.model = self.model.to(torch.float64)
        loss_fct = nn.CrossEntropyLoss()
        t0 = time.time()
        training_loss = []
        total_acc = 0
        f1_type = 'macro' 
        self.model.train()
        prev_f1 = 0
        final_epoch = 0
    
        # for early stopping
        lost_patience = 5
        patience = 0
        prev_f1 = float('inf')

        for ep in range(self.num_epochs):
            final_epoch = ep
            predictions = []
            target_values = []
            train_metrics = metrics(self.num_classes) 
            train_f1_scores = []

            for train_index in tqdm(range(0, self.graphs_train.shape[0], self.train_batch_size)):
                self.model.zero_grad()
                out = model.forward(torch.from_numpy(self.tweets_train[train_index:train_index + self.train_batch_size]).long().to(device), 
                        torch.from_numpy(self.graphs_train[train_index:train_index + self.train_batch_size]).long().to(device),
                        torch.from_numpy(self.macds_train[train_index:train_index + self.train_batch_size]).long().to(device))
                target = torch.from_numpy(self.macds).to(device)
                loss = loss_fct(out, target)                
                self.optimizer.zero_grad() 
                loss.backward()
                self.optimizer.step()

                out = out.detach().cpu()
                target = target.detach().cpu()
                train_metrics.update(out, target) 


                if(train_index % 10000 == 0):
                    print('loss: ',  loss.item())
            print('length: ', str(time.time() - t0))
            print('loss total: ', sum(training_loss))

            (train_accuracy, 
            train_f1_macro, 
            train_f1_micro, 
            train_precision_macro, 
            train_precision_micro, 
            train_recall_macro, 
            train_recall_micro) = train_metrics.compute()
            print('training accuracy: ', train_accuracy)
            print('Macro training f1: ', train_f1_macro)
            print('Micro training f1: ', train_f1_micro)
            print('Macro precision: ', train_precision_macro)
            print('Micro precision: ', train_precision_micro)
            print('Macro recall: ', train_recall_macro)
            print('Micro recall: ', train_recall_micro)

            #self.f1_plot(np.array(train_f1_scores))
            self.lr_scheduler.step()
            val_metrics = metrics(self.num_classes) 
            self.model.eval()
            val_loss = 0
            print('Evaluating Model...')
            with torch.no_grad():
                for val_index in range(0, len(self.graphs_val.shape[0]), self.eval_batch_size):
                    out = model.forward(torch.from_numpy(self.tweets_val[val_index:val_index + self.eval_batch_size]).to(device), 
                        torch.from_numpy(self.graphs_val[val_index:val_index + self.eval_batch_size]).to(device),
                        torch.from_numpy(self.macds_val[val_index:val_index + self.eval_batch_size]).to(device)).detach().cpu()
                    target = torch.from_numpy(self.y_val[val_index:val_index + self.eval_batch_size])
                    val_metrics.update(out, target) 
            (val_accuracy, 
            val_f1_macro, 
            val_f1_micro, 
            val_precision_macro, 
            val_precision_micro, 
            val_recall_macro, 
            val_recall_micro) = train_metrics.compute()

            print('Validation accuracy: ', val_accuracy)
            print('Macro validation f1: ', val_f1_macro)
            print('Micro validation f1: ', val_f1_micro)
            print('Macro precision: ', val_precision_macro)
            print('Micro precision: ', val_precision_micro)
            print('Macro recall: ', val_recall_macro)
            print('Micro recall: ', val_recall_micro)

            if self.early_stopping:
                if(val_f1 <= prev_f1):
                    patience += 1
                    if(patience == lost_patience):
                        print('Stopped at epoch ' + str(ep + 0))
                        break
                else:
                    patience = 0
                prev_f1 = val_f1


        torch.save(self.model, self.file_path + '/models/' + self.model_name + '/' + self.model_name + '_' + self.dataset + '_' + self.run_id + '_' + str(final_epoch + 1) + '.pt')
        #torch.save(self.model, self.file_path + '/models/' + self.model_name + '/' + self.model_name + '_' + self.run_id + '_' + str(self.epoch + 1) + '.pt')
        #torch.save(self.optimizer.state_dict(), self.file_path + '/optimizers/' +  self.optimizer_name + '/' + self.model_name + '_' + self.run_id + '_' + str(args.learning_rate) + '_' + str(self.epoch + 1) + '.pt')
        #torch.save(self.lr_scheduler.state_dict(), self.file_path + '/lr_schedulers/' + self.lrst + '/' + self.model_name + '_' +  self.run_id + '_' + str(self.epoch + 1) + '.pt')

        
        if(self.test_model):
            print('Testing...') 
            test_metrics = metrics(self.num_classes) 
            self.model.eval()
            f1_scores = []
            accuracy = Accuracy(task='multiclass', num_classes=self.num_classes).to(device)
            #i_final_input = self.tokenize_and_align_labels(self.test[0:0+self.test_batch_size])
            #self.model = torch.jit.trace(self.model, ('input_ids':x_final_input['input_ids'].to(device), 'attention_mask':x_final_input['attention_mask'].to(device)), strict=False)
            with torch.no_grad():
                for test_index in range(0, len(self.graphs_test.shape[0]), self.test_batch_size):
                    out = model.forward(torch.from_numpy(self.tweet_test[test_index:test_index + self.test_batch_size]).to(device), 
                        torch.from_numpy(self.graphs_test[test_index:test_index + self.test_batch_size]).to(device),
                        torch.from_numpy(self.macds_test[test_index:test_index + self.test_batch_size]).to(device)).detach().cpu()
                    target = torch.from_numpy(self.y_test[test_index:test_index + self.test_batch_size])
                    test_metrics.update(out, target) 
            (test_accuracy, 
            test_f1_macro, 
            test_f1_micro, 
            test_precision_macro, 
            test_precision_micro, 
            test_recall_macro, 
            test_recall_micro) = train_metrics.compute()
            print('Test accuracy: ', test_accuracy)
            print('Macro test f1: ', test_f1_macro)
            print('Micro test f1: ', test_f1_micro)
            print('Macro precision: ', test_precision_macro)
            print('Micro precision: ', test_precision_micro)
            print('Macro recall: ', test_recall_macro)
            print('Micro recall: ', test_recall_micro)   
            self.f1_plot(np.array(f1_scores))


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
    parser.add_argument('-ne', '--num_epochs', type=int, help = 'Number of epochs to run training loop', default=30)
    parser.add_argument('-es', '--early_stopping', type=str2bool, help = 'Early stopping is active', nargs='?', const=True, default=True)
    parser.add_argument('-s', '--stoppage', type=float, help='Stoppage value', default=1e-4)
    parser.add_argument('-tb', '--train_batch_size', type = int, help = 'Batch size for training step', default = 1)
    parser.add_argument('-eb', '--eval_batch_size',type=int, help='Batch size for evaluation step', default=1)
    parser.add_argument('-tesb', '--test_batch_size',type=int, help='Batch size for test step', default=1)
    parser.add_argument('-testm', '--test_model', type=str2bool, help='Whether or not to test our model', nargs='?', const=True, default=True)

    # Model specific
    parser.add_argument('-mn', '--model_name', type=str, help='Model name', default='meant')
    parser.add_argument('-nc', '--num_classes', type= int, help='Number of classes', default = 9)
    parser.add_argument('-t', '--task', type = str, help = 'Task type for training loop', default = 'classification')
    parser.add_argument('-cl', '--cache_location', type = str, help = 'Location for HuggingFace files')
    parser.add_argument('-di', '--dimension', type=int, help = 'internal dimension', default = 128)
    parser.add_argument('-nl', '--num_layers', type=int, help= 'The number of layers to use in the model', default=3)
    parser.add_argument('-do', '--dropout', type=float, help='Dropout in our model', default=0.0)
    parser.add_argument('-ptm', '--pretrained_model', type=str, help='Path to model', default=None)
    parser.add_argument('-p', '--pretrained', type = str, help='Load pretrained model if True. Train from scratch if false',default=True)

    # hugging face
    parser.add_argument('-hf', '--hugging_face_model', type=str, help='If we want to finetune/pretrain a model from Hugging face.', default=None)
    parser.add_argument('-hfd', '--hugging_face_data', type=str, help='Data set to load from Hugging Face', default=None)
    parser.add_argument('-hft', '--hugging_face_tokenizer', type=str, help='HuggingFace tokenizer', default=None)

    # Miscellaneous
    parser.add_argument('-db', '--debug', type = bool, help = 'Debug underflow and overflow', default = False)
    parser.add_argument('-fp', '--file_path', type=str, help='Path to files', default='/work/nlp/b.irving/meant_runs')
    parser.add_argument('-rid', '--run_id', type=str, help='Run identification number', required=True)
    parser.add_argument('-lag', '--lag', type=int, help='Lag period for data', default=5)
    args = parser.parse_args()

    t0 = time.time()
    
    # the model most be loaded first, in order to support the instantiation of the optimizer and the learning rate scheduler
    if(args.epoch == 0):
        if args.hugging_face_model is not None:
            if args.pretrained is True:
                model = AutoModelForTokenClassification.from_pretrained(args.hugging_face_model).to(device)
            else: 
                print('Training model from scratch')
                config = AutoConfig.from_pretrained('/work/nlp/b.irving/nlp/src/hug/configs/' + args.model_name +'.json', local_files_only=True)
                model = AutoModelForTokenClassification.from_config(config).to(device)
        elif args.model_name == 'meant':
            bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
            model = meant(text_dim = 768, 
                image_dim = 768, 
                price_dim = 4, 
                height = 224, 
                width = 224, 
                patch_res = 16, 
                lag = args.lag, 
                num_classes = args.num_classes, 
                embedding = bertweet.embeddings.double())
        else:
            raise ValueError('Pass a valid model name.')
    else:
        model = torch.load(args.file_path + '/models/' + args.model_name + '/' + args.model_name + '_' + args.run_id + '_' + str(args.epoch) + '.pt')

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

    print('Loading data...')
    """
    graphs = np.load('/work/nlp/b.irving/stock/complete/graphs_5.npy')
    tweets = np.load('/work/nlp/b.irving/stock/complete/tweets_5.npy')
    macds = np.load('/work/nlp/b.irving/stock/complete/macds_5.npy')
    labels = np.load('/work/nlp/b.irving/stock/complete/y_resampled_5.npy')  

    # First split: Separate out the test set
    graphs_train_val, graphs_test, tweets_train_val, tweets_test, macds_train_val, macds_test, y_train_val, y_test = train_test_split(
        graphs, tweets, macds, labels, test_size=0.2, random_state=42)

    # Second split: Split the remaining data into training and validation sets
    graphs_train, graphs_val, tweets_train, tweets_val, macds_train, macds_val, y_train, y_val= train_test_split(
        graphs_train_val, tweets_train_val, macds_train_val, y_train_val, test_size=0.25, random_state=42) 


    # saving for debugging
    np.save('graphs_sample.npy', graphs_train[0:100])
    np.save('tweets_sample.npy', tweets_train[0:100])
    np.save('macd_sample.npy', macds_train[0:100])
    np.save('labels_sample.npy', labels[0:100])
    """

    graphs = np.load('graphs_sample.npy')
    tweets = np.load('tweets_sample.npy')
    macds = np.load('macd_sample.npy')
    y = np.load('labels_sample.npy')

    print('Data loaded.')
    # normalize the data (compare this in runs as well)
    # lets see if we can train our transformer at allu
    """
            'graphs_train':graphs_train,
            'graphs_val':graphs_val,
            'graphs_test':graphs_test,
            'tweets_train':tweets_train,
            'tweets_val':tweets_val,
            'tweets_test':tweets_test,
            'macds_train':macds_train,
            'macds_val':macds_val,
            'macds_test':macds_test,
            'y_train':y_train,
            'y_val':y_val,
            'y_test':y_test,
    """ 
    params = {

            # DATA
            'dataset':'stocknet', 

            # for debugging
            'graphs_train':graphs,
            'graphs_val':graphs,
            'graphs_test':graphs,
            'tweets_train':tweets,
            'tweets_val':tweets,
            'tweets_test':tweets,
            'macds_train':macds,
            'macds_val':macds,
            'macds_test':macds,
            'y_train':y,
            'y_val':y,
            'y_test':y,


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
            'model':model.to(device),
            'debug':args.debug,
            'dim':args.dimension,
            'dropout':args.dropout,
            'num_layers':args.num_layers,
            'lr_scheduler':lr_scheduler,
            'lrst':args.learning_rate_scheduler_type,
            'classes':args.num_classes,
            'tokenizer':tokenizer,
            'model_name':args.model_name,
    }

    train = meant_trainer(params)
    train.train()
    print('Done in ' +  str(time.time() - t0) + ' seconds.')

