#!/usr/bin/env python
import csv
import pandas as pd
import ast
import torch
from torch import nn, tensor
import os, sys, argparse, time, gc, socket
import numpy as np
import matplotlib.pyplot as plt
import os, sys, argparse, time, gc, socket
from tqdm import tqdm
import math
import torch._dynamo as dynamo
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from torchmetrics import Accuracy, MatthewsCorrCoef, AUROC
import torchmetrics
import string
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from datasets import load_dataset
from torchmetrics.classification import MulticlassF1Score
from torch.nn.utils.rnn import pad_sequence
import re


torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class trainer():
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
        self.epoch = params['epoch']
        self.run_id = params['run_id']
        self.model = params['model']
        self.learning_rate = params['lr']
        self.optimizer = params['optimizer']
        self.optimizer_name = params['optimizer_name']
        self.train_batch_size = params['train_batch_size']
        self.debug_overflow = params['debug']
        self.X_train = params['X_train']
        self.y_train = params['y_train']
        self.X_val = params['X_val']
        self.y_val = params['y_val']
        self.train_data = params['train']
        self.val_data = params['val']
        self.dataset = params['dataset']
        self.dimension = params['dim']
        self.num_layers = params['num_layers']
        self.dropout = params['dropout']
        self.pretrained_model = params['pretrained_model']
        self.lr_scheduler = params['lr_scheduler']
        self.tokenizer = params['tokenizer']
        self.num_classes = params['classes']
        self.lrst = params['lrst']
        self.file_path = params['file_path']
        self.model_name = params['model_name']
        self.loss = params['loss']

    def make_one_hot(self, x_inputs):
        output_one_hot = [] 
        max_length = len(max(x_inputs, key=lambda x: len(x)))
        for x in x_inputs:
            padding = max_length - len(x)
            output = F.one_hot(torch.tensor(x), num_classes=self.num_classes)
            output_one_hot.append(output)
        return output_one_hot

    def plot(self, loss, model, epoch):
        timesteps = np.arange(1, loss.shape[0] + 1)
        alpha = 0.9  # Smoothing factor (adjust as needed)
        ema = [loss[0]]  # Initialize the EMA with the first loss value
        for i in range(1, loss.shape[0]):
            ema.append(alpha * ema[-1] + (1 - alpha) * loss[i])
        ema = np.array(ema)  # Convert the EMA list to a NumPy array
        plt.plot(ema, loss)
        plt.xlabel('Timestep')
        plt.ylabel(self.loss)
        plt.title('Loss')
        plt.savefig(self.file_path + '/output_files/' + self.dataset + '/plots/' + model + '_' + self.run_id + '_' + str(epoch) + '.png')
        plt.close()
    
    def merge_input(self, x_input):
        return [' '.join(x) for x in x_input]



    # Function to remove spaces between sentences and punctuation
    def remove_spaces_between_sentences(self, text):
        # Use regular expressions to remove spaces between sentences and punctuation
        cleaned_text = re.sub(r'\s+([.!?])', r'\1', text)
        return cleaned_text

    def tokenize_and_align_labels(self, examples, single_batch=False):
        list_len = [len(i) for i in examples['tokens']]
        max_length = max(list_len)
        max_length=256
        tokenized_inputs = tokenizer(examples['tokens'], truncation=True, padding='max_length', max_length=max_length, is_split_into_words=True, return_tensors='pt')
        labels = []
        accuracy_labels = []
        if(single_batch):
            examples['ner_tags'] = [examples['ner_tags']]
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            accuracy_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                    accuracy_ids.append(0)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                    accuracy_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                    accuracy_ids.append(0)
                previous_word_idx = word_idx
            labels.append(label_ids)
            accuracy_labels.append(accuracy_ids)
        tokenized_inputs["ner_tags"] = labels
        return tokenized_inputs, accuracy_labels


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

        if(self.loss is 'Cross Entropy'):
            loss_fct = nn.CrossEntropyLoss()
        elif(self.loss is 'Binary Cross Entropy'):
            loss_fct = nn.BCELoss()
        else:
            raise ValueError('Loss function not supported!')

        counter = 0
        t0 = time.time()
        training_loss = []
        total_acc = 0
        train_accuracy = Accuracy(task='multiclass', num_classes=self.num_classes).to(device)
        train_f1 = MulticlassF1Score(num_classes=self.num_classes, average='macro').to(device)
        train_f1_scores = []

        for tick in train:
            tweets = torch.load('/work/socialmedia/stock/tweets/' + tick).to(device)
            graphs = torch.load('/work/socialmedia/stock/graphs/' + tick + '.pt').to(device)
            macd = torch.load('/work/socialmedia/stock/macd/' + tick + '.pt').to(device)
            labels = torch.load('/work/socialmedia/stock/labels_2/' + tick + '.pt').to(device)

            lim = min(tweets.shape[0] - (self.train_batch_size * lag) - 1, macd.shape[0] - (self.train_batch_size * lag))
            for train_index in tqdm(range(0, lim, self.train_batch_size)):
                self.model.zero_grad()
                #x_final_input, labels = self.tokenize_and_align_labels(self.train_data[train_index:train_index+self.train_batch_size])
                #out = model.forward(x_final_input['input_ids'].to(device))
                out = model.forward(tweets[train_index:train_index + (lag * batch_size)], 
                        graphs[train_index:train_index + (lag * batch_size)].view(batch_size, lag, 4, 224, 224), 
                        macd[train_index:train_index + (lag * batch_size)].view(batch_size, lag, 4))
                batch_loss = []
                for i, vec in enumerate(out['logits']):
                    loss = loss_fct(vec.float().to(device), labels[self.train_batch_size:self.train_batch_size * lag].to(device))
                    batch_loss.append(loss)
                    training_loss.append(loss.item()) 
                    label = torch.tensor(labels[i]).to(device)
                    train_accuracy.update(vec.float().to(device), label) 
                    train_f1.update(vec.float().to(device), label) 
                    f1_score = train_f1.compute()
                    train_f1_scores.append(f1_score.item())
                avg_loss = sum(batch_loss)/self.train_batch_size
                avg_loss.backward()
                self.optimizer.zero_grad()
                self.optimizer.step()
                if(train_index % 10000 == 0):
                    print('loss: ',  loss.item())

        print('Epoch length: ', str(time.time() - t0))
        self.plot(np.array(training_loss), self.model_name, self.epoch)
        print('epoch: ', counter)
        counter += 1
        print('loss total: ', sum(training_loss))
        print('training accuracy: ', train_accuracy.compute())
        print('training f1: ', train_f1.compute())
        self.f1_plot(np.array(train_f1_scores))
        self.lr_scheduler.step()
        torch.save(self.model, self.file_path + '/models/' + self.model_name + '/' + self.model_name + '_' + self.run_id + '_' + str(self.epoch + 1) + '.pt')
        torch.save(self.optimizer.state_dict(), self.file_path + '/optimizers/' +  self.optimizer_name + '/' + self.model_name + '_' + self.run_id + '_' + str(args.learning_rate) + '_' + str(self.epoch + 1) + '.pt')
        torch.save(self.lr_scheduler.state_dict(), self.file_path + '/lr_schedulers/' + self.lrst + '/' + self.model_name + '_' +  self.run_id + '_' + str(self.epoch + 1) + '.pt')

        # validation data
        # we need to have checks for all of these, so that the script can generalize
        f1 = MulticlassF1Score(num_classes=self.num_classes, average='macro').to(device)
        accuracy = Accuracy(task='multiclass', num_classes=self.num_classes).to(device)
        with torch.no_grad():
            for y in range(len(self.y_val)):
                x_input = self.X_val[y]
                x_final_input = self.tokenizer(x_input,return_tensors="pt",padding=True,truncation=True)
                val_input, labels = self.tokenize_and_align_labels(self.val_data[y], single_batch=True)                
                out = self.model(val_input['input_ids'].to(device))
                f1.update(out['logits'].squeeze(dim=0).float().to(device), torch.tensor(labels).squeeze(dim=0).to(device))
                accuracy.update(out['logits'].squeeze(dim=0).float().to(device),
                        torch.tensor(labels).squeeze(dim=0).to(device))
        print("Accuracy on validation set: ", accuracy.compute())
        print("F1 on validation set: ", f1.compute())

if __name__=='__main__':
    torch._dynamo.config.verbose = True
    torch._dynamo.config.suppress_errors = True
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, help='Learning rate for the trainer', default=2e-5)
    parser.add_argument('-rid', '--run_id', type=str, help='Run identification number', required=True)
    parser.add_argument('-p', '--pretrained', type = str, help='Load pretrained model if True. Train from scratch if false',default=True)
    parser.add_argument('-e', '--epoch', type = int, help = 'Which epoch are we currently on', required=True)
    parser.add_argument('-t0', '--t0', type = int, help = 'Number of iterations for the first restart', default = 7)
    parser.add_argument('-tm', '--tmax', type = int, help = 'The number of epochs that the cosine lr takes to complete a cycle', default = 10)
    parser.add_argument('-o', '--optimizer', type = str, help = 'Optimizer', default = 'AdamW')
    parser.add_argument('-d', '--decay', type = float, help = 'Weight decay for the optimizer', default = 0.0)
    parser.add_argument('-b1','--beta_1', type = float, help='Beta1 for the optimizer', default = 0.9)
    parser.add_argument('-b2', '--beta_2', type = float, help = 'Beta2 for the optimizer', default= 0.999)
    parser.add_argument('-nc', '--num_classes', type= int, help='Number of classes', default = 2)
    parser.add_argument('-tb', '--train_batch_size', type = int, help = 'Batch size for training step', default = 1)
    parser.add_argument('-db', '--debug', type = bool, help = 'Debug underflow and overflow', default = False)
    parser.add_argument('-t', '--task', type = str, help = 'Task type for training loop', default = 'classification')
    parser.add_argument('-cl', '--cache_location', type = str, help = 'Location for HuggingFace files')
    parser.add_argument('-di', '--dimension', type=int, help = 'internal dimension', default = 128)
    parser.add_argument('-nl', '--num_layers', type=int, help= 'The number of layers to use in the model', default=3)
    parser.add_argument('-do', '--dropout', type=float, help='Dropout in our model', default=0.0)
    parser.add_argument('-ptm', '--pretrained_model', type=str, help='Path to model', default=None)
    parser.add_argument('-m', '--metric', type = str, help = 'Evalutation metric')
    parser.add_argument('-lrst', '--learning_rate_scheduler_type', type=str, help='The type of learning rate scheduler to use.', default='cosine_warm')
    parser.add_argument('-hf', '--hugging_face_model', type=str, help='If we want to finetune/pretrain a model from Hugging face.', default= None)
    parser.add_argument('-hfd', '--hugging_face_data', type=str, help='Data set to load from Hugging Face', default=None)
    parser.add_argument('-hft', '--hugging_face_tokenizer', type=str, help='HuggingFace tokenizer', default=None)
    parser.add_argument('-fp', '--file_path', type=str, help='Path to files',required=True)
    parser.add_argument('-mn', '--model_name', type=str, help='Model name', required=True)
    parser.add_argument('-td', '--text_dim', type=int, help='Text dimension of model', default=768)
    parser.add_argument('-id', '--image_dim', type=int, help='Image dimension of the model', default=768)
    parser.add_argument('-w', '--width', type=int, help='Width of image', default=224)
    parser.add_argument('-h', '--height', type=int, help='Height of image', default=224)
    parser.add_argument('-pr', '--patch_res', type=int, help='Height and width of the patches', default=16)
    parser.add_argument('-lp', '--lag_period', type=int, help='Lag period for a sequential model', default=5)
    parser.add_argument('-l', '--loss_function', type=str, help='Loss function we are using', default='Cross entropy')
    args = parser.parse_args()

    t0 = time.time()

    # check for the existence of the optimizer, scheduler, and model
   # if (os.path.exists(args.file_path + '/models/' + args.model_name + '/' + args.model_name + '_' + str(args.epoch + 1) + '.pt') and 
#	os.path.exists(args.file_path + '/optimizers/'+ args.optimizer + '/' + args.model_name + '_' + str(args.learning_rate) + '_' + str(args.epoch + 1)  + '.pt') and
 #       os.path.exists(args.file_path + '/lr_schedulers/' + args.learning_rate_scheduler_type + '/' + args.model_name + '_' + str(args.epoch + 1) + '.pt')):
  #      print('Files already computed, move onto next epoch')
   #     sys.exit()

    # the model most be loaded first, in order to support the instantiation of the optimizer and the learning rate scheduler
    if(args.epoch == 0):
        if args.hugging_face_model is not None:
            if args.pretrained is True:
                model = AutoModelForTokenClassification.from_pretrained(args.hugging_face_model).to(device)
            else: 
                print('Training model from scratch')
                config = AutoConfig.from_pretrained('../configs/' + args.model_name +'.json', local_files_only=True)
                model = AutoModelForTokenClassification.from_config(config).to(device)
        elif args.model_name == 'meant':
            bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
            model = meant(text_dim =768, 
                image_dim = 768, 
                price_dim = 4, 
                height = 224, 
                width = 224, 
                patch_res = 16, 
                lag = args.lag_period, 
                num_classes = args.num_classes, 
                embedding = bertweet.embeddings)
        else:
            raise ValueError('Pass a valid model name.')

    else:
        model = torch.load(args.file_path + '/models/' + args.model_name + '/' + args.model_name + '_' + args.run_id + '_' + str(args.epoch) + '.pt')

    # we declare these either way, because the objects have to be initialized
    if(args.optimizer == 'AdamW'):
        optimizer = torch.optim.AdamW(params = model.parameters(), lr=args.learning_rate, weight_decay=args.decay, betas=(args.beta_1, args.beta_2))
    elif(args.optimizer == 'Adam'):
        optimizer = torch.optim.Adam(params = model.parameters(), lr=args.learning_rate, weight_decay=args.decay)
    else: 
        raise ValueError("This type of optimizer is not supported.")

    if(args.hugging_face_tokenizer is not None):
        tokenizer = AutoTokenizer.from_pretrained(args.hugging_face_tokenizer)

    # load incrementally
    if(args.learning_rate_scheduler_type == 'cosine_warm'):
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.t0)
    elif(args.learning_rate_scheduler_type == 'cosine'):
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.tmax)
    else:
        raise ValueError('Not supported')
    
    if(args.epoch == 0):
       pass # we don't need to load in progress state dictionaries
    else:
        optimizer_state_dict = torch.load(args.file_path + '/optimizers/' + args.optimizer + '/' + args.model_name + '_' + args.run_id + '_' + str(args.learning_rate) + '_' + str(args.epoch) + '.pt')
        lr_scheduler_state_dict = torch.load(args.file_path + '/lr_schedulers/' + args.learning_rate_scheduler_type + '/' + args.model_name + '_' + args.run_id + '_' + str(args.epoch) + '.pt')
        optimizer.load_state_dict(optimizer_state_dict)
        lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    tsp500arr = np.loadtxt("/training/constituents.csv",
                    delimiter=",", dtype=str)
    sp500 = sp500arr[:, 0][1:37]
    np.random.shuffle(sp500)
    train_tickers = sp500[0:25]
    val_tickers = sp500[25:29]
    test_tickers = sp500[29:]

    params = {
            'train': train_tickers,
            'val':val_tickers,
            'lr': args.learning_rate,
            'run_id':args.run_id,
	        'file_path': args.file_path,
            'pretrained_model': args.pretrained_model,
            'epoch': args.epoch,
            'optimizer': optimizer,
            'optimizer_name':args.optimizer,
            'train_batch_size': args.train_batch_size,
            'dataset':dataset_name, 
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
            'loss':args.loss_function


    }

    train = trainer(params)
    train.train()
    print('Done in ' +  str(time.time() - t0) + ' seconds.')
