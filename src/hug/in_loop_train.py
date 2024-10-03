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
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall
import re


torch.cuda.empty_cache()
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ner_trainer():
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
        self.num_epochs = params['num_epochs']
        self.run_id = params['run_id']
        self.model = params['model']
        self.learning_rate = params['lr']
        self.optimizer = params['optimizer']
        self.optimizer_name = params['optimizer_name']
        self.train_batch_size = params['train_batch_size']
        self.test_batch_size = params['test_batch_size']
        self.debug_overflow = params['debug']
        self.X_train = params['X_train']
        self.y_train = params['y_train']
        self.X_val = params['X_val']
        self.y_val = params['y_val']
        self.train_data = params['train']
        self.val_data = params['val']
        self.test_data = params['test']
        self.dataset = params['dataset']
        self.dimension = params['dim']
        self.num_layers = params['num_layers']
        self.dropout = params['dropout']
        self.pretrained_model = params['pretrained_model']
        self.lr_scheduler = params['lr_scheduler']
        self.tokenizer = params['tokenizer']
        self.stoppage = params['stoppage']
        self.num_classes = params['classes']
        self.lrst = params['lrst']
        self.file_path = params['file_path']
        self.model_name = params['model_name']
        self.eval_batch_size = params['eval_batch_size']
        self.join_size = params['join_size']

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
    
    # Function to remove spaces between sentences and punctuation
    def remove_spaces_between_sentences(self, text):
        # Use regular expressions to remove spaces between sentences and punctuation
        cleaned_text = re.sub(r'\s+([.!?])', r'\1', text)
        return cleaned_text


    def join_examples(self, examples):
        list_len = len(examples)
        inputs = {}
        inputs['tokens'] = []
        inputs['ner_ids'] = [] 
        # join each group of 'join size' examples together
        for i in range(0, list_len, self.join_size):
            tokens = examples['tokens'][0]
            ner_tags = examples['ner_ids'][0]
            for x in range(1, self.join_size):
                tokens += examples['tokens'][x]
                ner_tags += examples['ner_ids'][x]
            inputs['tokens'].append(tokens)
            inputs['ner_ids'].append(ner_tags)
        return inputs


    # try joining them together in this method, perhaps? 
    def tokenize_and_align_labels(self, examples):
        # try increasing the number of sentences you "join" together, so that there are more non-zeroes in each training example?
        list_len = [len(i) for i in examples['tokens']]
        max_length = max(list_len)
        if(len(list_len) == 1):
            if(max_length <= 2):
               tokenized_inputs = tokenizer(examples['tokens'], padding=True,  is_split_into_words=True, return_tensors='pt')
            else:
               tokenized_inputs = tokenizer(examples['tokens'], padding=True, truncation=True,is_split_into_words=True, return_tensors='pt')
        else:
            tokenized_inputs = tokenizer(examples['tokens'], padding=True, truncation=True, is_split_into_words=True, return_tensors='pt')
        labels = []
        accuracy_labels = []
        for i, label in enumerate(examples['ner_ids']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            accuracy_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx: 
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        return tokenized_inputs, labels 


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

    #def computeMetrics(self, predictions, targets):
     #   train_accuracy = Accuracy(task='multiclass', num_classes=self.num_classes).to(device)
      #  train_f1 = MulticlassF1Score(num_classes=self.num_classes, average='macro').to(device) 
    def train(self):
        if(self.debug_overflow):
            debug_overflow = DebugUnderflowOverflow(self.model)
        self.model = self.model.to(torch.float64)
        loss_fct = nn.CrossEntropyLoss()
        counter = 0
        t0 = time.time()
        training_loss = []
        total_acc = 0
        f1_type = 'macro' 
        #train_accuracy = Accuracy(task='multiclass', num_classes=self.num_classes).to(device)
        #train_f1 = MulticlassF1Score(num_classes=self.num_classes, average=f1_type).to(device)
        #train_f1_micro = MulticlassF1Score(num_classes=self.num_classes, average='micro').to(device)
       # train_f1_scores = []
        
        self.model.train()
        prev_f1 = 0
        final_epoch = 0
        lost_patience = 5
        patience = 0
        prev_val_loss = float('inf')
        for ep in range(self.num_epochs):
            final_epoch = ep
            predictions = []
            target_values = []
            train_accuracy = Accuracy(task='multiclass', num_classes=self.num_classes).to(device)
            train_f1 = MulticlassF1Score(num_classes=self.num_classes, average=f1_type).to(device)
            train_f1_micro = MulticlassF1Score(num_classes=self.num_classes, average='micro').to(device)
            train_f1_scores = []
            for train_index in range(0, len(self.X_train) - self.train_batch_size, self.train_batch_size):
                to_tokenize = self.train_data[train_index:train_index + self.train_batch_size]
                x_final_input, labels = self.tokenize_and_align_labels(to_tokenize)
                x_final_input.to(device)
                out = model.forward(**x_final_input)
                del x_final_input
                batch_loss = []
                for i, vec in enumerate(out['logits']):
                    target = torch.tensor(labels[i]).to(device)
                    loss = loss_fct(vec.to(device), target)
                    batch_loss.append(loss)
                    training_loss.append(loss.item()) 

                    # metric computation
                    mask = (target != -100) 
                    indices = torch.nonzero(mask)
                    indices = torch.transpose(indices, 0, 1).squeeze(dim=0)
                    metric_pred = vec[indices]
                    if(metric_pred.shape[0] == 0):
                            continue
                    metric_target = target[indices]
                    train_accuracy.update(metric_pred, metric_target) 
                    train_f1.update(metric_pred, metric_target) 
                    train_f1_micro.update(metric_pred, metric_target)
                    f1_score = train_f1.compute()
                    train_f1_scores.append(f1_score.item())
                target.cpu()
                out['logits'].cpu()
                avg_loss = sum(batch_loss)/self.train_batch_size
                self.optimizer.zero_grad() 
                avg_loss.backward()
                self.optimizer.step()
                if(train_index % 10000 == 0):
                    print('loss: ',  loss.item())
            print('length: ', str(time.time() - t0))
            #self.plot(np.array(training_loss), self.model_name, self.epoch)
            counter += 1
            print('loss total: ', sum(training_loss))
            print('training accuracy: ', train_accuracy.compute())
            print(f1_type + ' training f1: ', train_f1.compute())
            print('micro training f1: ', train_f1_micro.compute())
            #self.f1_plot(np.array(train_f1_scores))
            self.lr_scheduler.step()
            f1 = MulticlassF1Score(num_classes=self.num_classes, average=f1_type).to(device)
            f1_micro = MulticlassF1Score(num_classes=self.num_classes, average='micro').to(device)
            accuracy = Accuracy(task='multiclass', num_classes=self.num_classes).to(device)
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_index in range(0, len(self.val_data) - self.eval_batch_size, self.eval_batch_size):
                    val_input, val_labels = self.tokenize_and_align_labels(self.val_data[val_index:val_index+self.eval_batch_size])                 
                    val_input = val_input.to(device)
                    out = self.model(**val_input)
                    del val_input
                    cpu_dict = {key: value.to('cpu') for key, value in val_input.items()}
                    for i, vec in enumerate(out['logits']):
                        target = torch.tensor(val_labels[i]).to(device)
                        loss = loss_fct(vec.to(device), target)
                        val_loss += loss.item() 
                        # metric computation, we are going to do a hard metric
                        # (it should be able to identify the non zero ner tags)
                        mask = (target != -100) 
                        indices = torch.nonzero(mask)
                        metric_pred = torch.transpose(vec[indices], 1, 2).squeeze(dim=2)
                        if(metric_pred.shape[0] == 0):
                            continue
                        metric_target = torch.transpose(target[indices], 0, 1).squeeze(dim=0)    
                        accuracy.update(metric_pred, metric_target) 
                        f1.update(metric_pred, metric_target) 
                        f1_micro.update(metric_pred, metric_target)
                        f1_score = f1.compute()
            val_f1 = f1.compute()
            print("Accuracy on validation set: ", accuracy.compute())
            print(f1_type + " F1 on validation set: ", f1.compute())
            print('micro f1 validation set: ', f1_micro.compute())
            print(abs(val_f1 - prev_f1))
            if(val_f1 > prev_val_loss):
                patience += 1
                if(patience == lost_patience):
                    print('Stopped at epoch ' + str(ep + 0))
            prev_val_loss = val_loss
            prev_f1 = val_f1
        torch.save(self.model, self.file_path + '/models/' + self.model_name + '/' + self.model_name + '_' + self.dataset + '_' + self.run_id + '_' + str(final_epoch + 1) + '.pt')
        #torch.save(self.model, self.file_path + '/models/' + self.model_name + '/' + self.model_name + '_' + self.run_id + '_' + str(self.epoch + 1) + '.pt')
        #torch.save(self.optimizer.state_dict(), self.file_path + '/optimizers/' +  self.optimizer_name + '/' + self.model_name + '_' + self.run_id + '_' + str(args.learning_rate) + '_' + str(self.epoch + 1) + '.pt')
        #torch.save(self.lr_scheduler.state_dict(), self.file_path + '/lr_schedulers/' + self.lrst + '/' + self.model_name + '_' +  self.run_id + '_' + str(self.epoch + 1) + '.pt')
       
        f1_type = 'micro'
        precision = MulticlassPrecision(average=f1_type,num_classes=self.num_classes).to(device)
        recall = MulticlassRecall(average=f1_type, num_classes=self.num_classes).to(device)
        f1 = MulticlassF1Score(num_classes=self.num_classes, average=f1_type).to(device)
        macro_precision = MulticlassPrecision(average='macro',num_classes=self.num_classes).to(device)
        macro_recall = MulticlassRecall(average='macro', num_classes=self.num_classes).to(device)
        macro_f1 = MulticlassF1Score(num_classes=self.num_classes, average='macro').to(device)
        
        f1_scores = []
        accuracy = Accuracy(task='multiclass', num_classes=self.num_classes).to(device)
        #i_final_input = self.tokenize_and_align_labels(self.test[0:0+self.test_batch_size])
        #self.model = torch.jit.trace(self.model, ('input_ids':x_final_input['input_ids'].to(device), 'attention_mask':x_final_input['attention_mask'].to(device)), strict=False)
        with torch.no_grad():
            for y in tqdm(range(0, len(self.test_data) - self.test_batch_size, self.test_batch_size)):
                x_final_input, test_labels = self.tokenize_and_align_labels(self.test_data[y:y+self.test_batch_size])
                x_final_input = x_final_input.to(device)
                out = self.model(**x_final_input)
                cpu_dict = {key: value.to('cpu') for key, value in x_final_input.items()}
                del x_final_input
                for i, vec in enumerate(out['logits']):
                    target = torch.tensor(test_labels[i]).to(device)
                    # metric computation
                    #mask = (target != -100) 
                    mask = (target != -100)
                    indices = torch.nonzero(mask)
                    metric_pred = torch.transpose(vec[indices], 1, 2).squeeze(dim=2)
                    if(metric_pred.shape[0] == 0):
                        continue
                    metric_target = torch.transpose(target[indices], 0, 1).squeeze(dim=0)    
                    accuracy.update(metric_pred, metric_target) 
                    precision.update(metric_pred, metric_target)
                    recall.update(metric_pred, metric_target)
                    f1.update(metric_pred, metric_target)  
                    macro_precision.update(metric_pred, metric_target)
                    macro_recall.update(metric_pred, metric_target)
                    macro_f1.update(metric_pred, metric_target) 
                    f1_score = f1.compute()
                    f1_scores.append(f1_score.item())
        print("Accuracy on test set: ", accuracy.compute())
        print("Micro F1 on test set: ", f1.compute())
        print("Micro Precision: ", precision.compute())
        print("Micro Recall: ", recall.compute())
        print("Macro F1 on test set: ", macro_f1.compute())
        print("Macro Precision: ", macro_precision.compute())
        print("Macro Recall: ", macro_recall.compute())
        self.f1_plot(np.array(f1_scores))


if __name__=='__main__':
    torch._dynamo.config.verbose = True
    torch._dynamo.config.suppress_errors = True
    parser = argparse.ArgumentParser()
    parser.add_argument('-js', '--join_size', type=int, help='Number of sentences to join together in each training example', default=1)
    parser.add_argument('-s', '--stoppage', type=float, help='Stoppage value', default=1e-4)
    parser.add_argument('-l', '--learning_rate', type=float, help='Learning rate for the trainer', default=5e-5)
    parser.add_argument('-rid', '--run_id', type=str, help='Run identification number', required=True)
    parser.add_argument('-p', '--pretrained', type = str, help='Load pretrained model if True. Train from scratch if false',default=True)
    parser.add_argument('-e', '--epoch', type = int, help = 'Which epoch are we currently on', default=0)
    parser.add_argument('-ne', '--num_epochs', type=int, help = 'Number of epochs to run training loop', default=30)
    parser.add_argument('-t0', '--t0', type = int, help = 'Number of iterations for the first restart', default = 7)
    parser.add_argument('-tm', '--tmax', type = int, help = 'The number of epochs that the cosine lr takes to complete a cycle', default = 10)
    parser.add_argument('-o', '--optimizer', type = str, help = 'Optimizer', default = 'AdamW')
    parser.add_argument('-d', '--decay', type = float, help = 'Weight decay for the optimizer', default = 0.0)
    parser.add_argument('-b1','--beta_1', type = float, help='Beta1 for the optimizer', default = 0.9)
    parser.add_argument('-b2', '--beta_2', type = float, help = 'Beta2 for the optimizer', default= 0.999)
    parser.add_argument('-nc', '--num_classes', type= int, help='Number of classes', default = 9)
    parser.add_argument('-tb', '--train_batch_size', type = int, help = 'Batch size for training step', default = 1)
    parser.add_argument('-eb', '--eval_batch_size',type=int, help='Batch size for evaluation step', default=1)
    parser.add_argument('-tesb', '--test_batch_size',type=int, help='Batch size for test step', default=1)
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
    args = parser.parse_args()

    print('hugging face dataset: ', args.hugging_face_data)
    print('hugging face model: ', args.hugging_face_model)

    t0 = time.time()
    # the model most be loaded first, in order to support the instantiation of the optimizer and the learning rate scheduler
    if(args.epoch == 0):
        if args.hugging_face_model is not None:
            if args.pretrained is True:
                model = AutoModelForTokenClassification.from_pretrained(args.hugging_face_model).to(device)
                #model.classifier = nn.Linear(768, args.num_classes)
            else: 
                print('Training model from scratch')
                config = AutoConfig.from_pretrained('/work/nlp/b.irving/nlp/src/hug/configs/' + args.model_name +'.json', local_files_only=True)
                model = AutoModelForTokenClassification.from_config(config).to(device)
                # we need to make this adaptable by getting the previous layer
                model.classifier = nn.Linear(model.classifier.in_features, args.num_classes)
    else:
        model = torch.load(args.file_path + '/models/' + args.model_name + '/' + args.model_name + '_' + args.run_id + '_' + str(args.epoch) + '.pt')

    # we declare these either way, because the objects have to be initialized
    if(args.optimizer == 'AdamW'):
        optimizer = torch.optim.AdamW(params = model.parameters(), lr=args.learning_rate, weight_decay=args.decay, betas=(args.beta_1, args.beta_2))
    elif(args.optimizer == 'Adam'):
        optimizer = torch.optim.Adam(params = model.parameters(), lr=args.learning_rate, weight_decay=args.decay, betas=(args.beta_1, args.beta_2))
    else: 
        raise ValueError("This type of optimizer is not supported.")

    if(args.hugging_face_tokenizer is not None):
        tokenizer = AutoTokenizer.from_pretrained(args.hugging_face_tokenizer)

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

    if(args.hugging_face_data is None):
        pass
    else:
        dataset_name = args.hugging_face_data
        data = load_dataset(dataset_name)
        train = data["train"]
        X_train = train["tokens"]
        y_train = train["ner_tags"]
        val = data["validation"]
        X_val = val["tokens"]
        y_val = val["ner_tags"]
        test = data["test"]
        y_test = test["ner_tags"]
        if(args.hugging_face_data == 'adsabs/WIESP2022-NER'):
            dataset_name = 'wiesp'
        elif(args.hugging_face_data == 'siddharthtumre/jnlpba-split'):
            dataset_name = 'jnlpba'
        else:
            dataset_name = args.hugging_face_data
 
    params = {
            'train': train,
            'test':test,
            'stoppage':args.stoppage,
            'lr': args.learning_rate,
            'run_id':args.run_id,
            'file_path': args.file_path,
            'pretrained_model': args.pretrained_model,
            'epoch': args.epoch,
            'num_epochs' : args.num_epochs, 
            'optimizer': optimizer,
            'optimizer_name':args.optimizer,
            'train_batch_size': args.train_batch_size,
            'eval_batch_size':args.eval_batch_size,
            'test_batch_size':args.test_batch_size,
            'dataset':dataset_name, 
            'model':model.to(device),
            'X_train':X_train,
            'y_train':y_train,
            'X_val':X_val,
            'y_val':y_val,
            'debug':args.debug,
            'dim':args.dimension,
            'dropout':args.dropout,
            'num_layers':args.num_layers,
            'lr_scheduler':lr_scheduler,
            'lrst':args.learning_rate_scheduler_type,
            'classes':args.num_classes,
            'tokenizer':tokenizer,
            'model_name':args.model_name,
            'val':val,
            'join_size':args.join_size
    }
    train = ner_trainer(params)
    train.train()
    print('Done in ' +  str(time.time() - t0) + ' seconds.')

#python checkpoint_train.py --hugging_face_model='dslim/bert-base-ner' --hugging_face_tokenizer='dslim/bert-base-ner' --hugging_face_data='conll2003' --epoch=0 --file_path='/work/nlp/b.irving/nlp_files' --model_name='bert_ner' --pretrained=True --run_id='000000' --num_classes=9 --train_batch_size=8
