from datasets import load_dataset
from transformers import ( AutoTokenizer, 
    AutoModel, 
    Trainer, 
    AutoModelForPreTraining )
from transformers.debug_utils import DebugUnderflowOverflow
import torchvision.transforms as transforms
from torchvision.io import read_image
import numpy as np
from datasets import Features, ClassLabel, Array3D, Image
import torch
from torch import nn, tensor
import os
from torchmetrics import Accuracy, MatthewsCorrCoef, AUROC
import matplotlib.pyplot as plt
from PIL import Image
import requests
from tqdm import tqdm
import math
import torch._dynamo as dynamo
import time
from sklearn.model_selection import train_test_split
import sys
import argparse


# to combat memory problems
torch.cuda.empty_cache()

# we can specifiy the cache location of hugging face files
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ClassifierModel(nn.Module):
    def __init__(self, num_classes, model):

        super(ClassifierModel, self).__init__()
        self.model = AutoModel.from_pretrained(model)
        self.dropout = nn.Dropout(0.1)

        # how to find the output dim from the model?????
        self.fc = nn.ModuleList([nn.Linear(768, num_classes), nn.GELU(), nn.Softmax(dim=1)])

    def forward(self, inputs):
        outputs = self.model(**inputs)
        pooled_output = outputs.pooler_output
        logits = self.dropout(pooled_output)
        for mod in self.fc:
            logits = mod(logits)
        return logits

# how to compute the mlm task
class mlm(nn.Module):
    def __init__(self):
        super(mlm, self).__init__()
        pass
    def forward(self, inputs):
        pass


# generalize
class CustomTrainer(Trainer):
    def __init__(self, params):
        """
        A trainer class, that is easy for me to use. Probably less efficient, and most
        certainly less generalizable than the HuggingFace Trainer class. But at least
        I understand how to use it, and perhaps others will as well.

        I think the trick is in the manner in which data is fed into the training loop.

        args:
            params
                epochs
                model
                tokenizer
                optimizer
                train_batch_size
        """
        self.epochs = params['epochs']
        self.model = params['model']
        self.tokenizer = params['tokenizer']
        self.optimizer = params['optimizer']
        self.train_batch_size = params['train_batch_size']
        self.eval_batch_size = params['eval_batch_size']
        self.debug_overflow = params['debug']
        self.X_train_texts = params['X_train_texts']
        self.y_train = params['y_train']
        self.X_test_texts = params['X_test_texts']
        self.y_test = params['y_test']

    def plot(self, loss, epoch):
        timesteps = np.arange(1, loss.shape[0] + 1)
        # Plot the MSE vs timesteps
        plt.plot(timesteps, loss)
        # Add axis labels and a title
        plt.xlabel('Timestep')
        plt.ylabel('BCE Loss')
        plt.title('Loss')
        plt.savefig('/home/irving.b/neuMultiModalWork/mentalHealth/loss/bert_loss_' + str(epoch) + '.png')
        plt.close()

    def plot_accuracy(self, loss, epoch):
        timesteps = np.arange(1, loss.shape[0] + 1)
        # Plot the MSE vs timesteps
        plt.plot(timesteps, loss)
        # Add axis labels and a title
        plt.xlabel('Timestep')
        plt.ylabel('Percentage correct')
        plt.title('Accuracy')
        plt.savefig('/home/irving.b/neuMultiModalWork/mentalHealth/accuracy/bert_accuracy_' + str(epoch) + '.png')
        plt.close()

    def plot_auroc(self, loss, epoch):
        timesteps = np.arange(1, loss.shape[0] + 1)
        # Plot the MSE vs timesteps
        plt.plot(timesteps, loss)
        # Add axis labels and a title
        plt.xlabel('Timestep')
        plt.ylabel('Percentage correct')
        plt.title('AUROC')
        plt.savefig('/home/irving.b/neuMultiModalWork/mentalHealth/auroc/bert_auroc_' + str(epoch) + '.png')
        plt.close()


    def train(self):
        if(self.debug_overflow):
            debug_overflow = DebugUnderflowOverflow(self.model)
        self.model = self.model.to(torch.float64)

        # generalize all of these arguments
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)

        loss_fct = nn.BCELoss()


        accuracy = Accuracy(task='multiclass', num_classes=2).to(device)
        auroc = AUROC(task='binary', num_classes=2).to(device)
        test_auroc = AUROC(task='binary', num_classes=2).to(device)
        mcc = MatthewsCorrCoef(task='binary').to(device)
        
        tokenizer = self.tokenizer
        counter = 0

        training_loss_over_epochs = []
        for epoch in range(self.epochs):
            accuracy = Accuracy("binary").to(device)
            training_loss = []
            train_accuracies = []
            train_auroc = []
            total_acc = 0
            #num_train_steps = len(train_x) - 1

            # optimize for speed?
            for train_index in range(0, len(self.X_train_texts) - self.train_batch_size, self.train_batch_size):
                self.model.zero_grad()
                # pad to the max length, because the longest length would exceed the input size 
                inputs_df = tokenizer(self.X_train_texts[train_index:train_index+self.train_batch_size], padding='max_length', truncation=True, max_length = 512, return_tensors='pt')
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = self.model(inputs_df.to(device))
                end.record()
                torch.cuda.synchronize()
                #print('inference time: ', start.elapsed_time(end) / 1000)

                # obviously these would have to be generalized as well to meet
                # a new task
                truth = self.y_train[train_index:train_index+self.train_batch_size].to(device)
                loss = loss_fct(out.float().to(device), truth.float().to(device))
                training_loss.append(loss.item())
                maximums = torch.argmax(out, dim = 1)
                truth_max = torch.argmax(truth, dim = 1)
                train_accuracies.append(accuracy.update(out, truth))
                train_auroc.append(auroc.update(maximums.to(device), truth_max.to(device)))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # should actually be relative to the train batch size
                if(train_index % 100 == 0):
                    prog_acc = accuracy.compute()
                    print('accuracy: ', prog_acc)
                    print('loss: ',  loss.item())

            acc = accuracy.compute()
            auroc_acc = auroc.compute()
            print("Bert Accuracy:", acc)
            print("Bert AUROC: ", auroc_acc)
            self.plot(np.array(training_loss), epoch)
            self.plot_accuracy(np.array(train_accuracies), epoch)
            self.plot_auroc(np.array(train_auroc), epoch)
            print('\n')
            print('epoch: ', counter)
            counter += 1
            print('loss total: ', sum(training_loss))
            print('\n')
            training_loss_over_epochs.append(training_loss)
            #exponential.step()
            cosine.step()
            self.model.eval()
        # save the model before the evaluation stage
        torch.save(model, 'bert.pt')
        with torch.no_grad():
            test_aurocs = []
            test_accuracy = []
           # num_test_steps = len(test_x)
            total_acc = 0
            runtime = 0
            value = 0
            accuracy_two = Accuracy("binary").to(device)
            for y in range(len(self.X_test_texts) - self.eval_batch_size):
                inputs_df = tokenizer(self.X_test_texts[y:y+self.eval_batch_size], padding='max_length', truncation=True, max_length = 512, return_tensors='pt')
                out = self.model(inputs_df.to(device))
                truth = self.y_test[y:y+eval_batch_size].to(device)
                accuracy_two.update(out.to(device),truth.unsqueeze(0).to(device))
                test_aurocs.append(test_auroc.update(out.to(device), truth.unsqueeze(0).to(device)))
        print(" accuracy on test set: ", accuracy_two.compute())
        print("AUROC on test set: ", test_auroc.compute())
       # torch.save(self.model, 'bert.pt')
        return training_loss_over_epochs, accuracy

if __name__=='__main__':
    # optimize script, will triton
    # torch.compile? This should run real fast
    # also, support for multi-gpu training?

    torch._dynamo.config.verbose = True
    torch._dynamo.config.suppress_errors = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, help='Learning rate for the trainer', default=5e-5)
    parser.add_argument('-p', '--pretrained', type = str, help='Location of a previously trained model')
    parser.add_argument('-e', '--epochs', type = int, help = 'Number of epochs', default = 1)
    parser.add_argument('-o', '--optimizer', type = str, help = 'Optimizer', default = 'AdamW')
    parser.add_argument('-d', '--decay', type = float, help = 'Weight decay for the optimizer', default = 0.0)
    parser.add_argument('-b1','--beta_1', type = float, help='Beta1 for the optimizer', default = 0.9)
    parser.add_argument('-b2', '--beta_2', type = float, help = 'Beta2 for the optimizer', default= 0.999)
    parser.add_argument('-c', '--classes', type= int, help='Number of classes', default = 2)
    parser.add_argument('-tb', '--train_batch_size', type = int, help = 'Batch size for training step', default = 8)
    parser.add_argument('-ev', '--eval_batch_size', type = int, help = 'Batch size for eval step', default = 1)
    parser.add_argument('-db', '--debug', type = bool, help = 'Debug underflow and overflow', default = False)
    parser.add_argument('-t', '--task', type = str, help = 'Task type for training loop', default = 'classification')
    parser.add_argument('-tk', '--tokenizer', type = str, help = 'Tokenizer for training loop', default = 'bert-base-uncased')
    parser.add_argument('-cl', '--cache_location', type = str, help = 'Location for HuggingFace files')
    # what are the best metrics?
    parser.add_argument('-m', '--metric', type = str, help = 'Evalutation metric')
    args = parser.parse_args()

    lr = args.learning_rate
    pretrained_model = args.pretrained
    epochs = args.epochs
    optimizer = args.optimizer
    decay = args.decay
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    num_classes = args.classes
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    task = args.task
    debug = args.debug
    tokenizer = args.tokenizer
    cache_location = args.cache_location

    # model
    # First, there is a check for the pretrained model option. If a file is specified,
    # we won't create one. 
    if(pretrained_model is not None):
        model = torch.load(pretrained_model)
    elif(task == 'classification'):
        model = ClassifierModel(num_classes, model)
    elif(task == 'mlm'):
        print('unsupported')
        sys.exit()
        pass
        #model = mlm()
    
    # on default, we will leave everything as is?
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    if(optimizer == 'AdamW'):
        optimizer = torch.optim.AdamW(params = model.parameters(), lr=lr, weight_decay=decay, betas=(beta_1, beta_2))
    elif(optimizer == 'Adam'):
        optimizer = torch.optim.Adam(params = model.parameters(), lr=lr, weight_decay=decay)
    else: 
        raise ValueError("This type of optimizer is not supported.")

    # separate classes for the DataLoaders?
    # how will you load datasets efficiently
    params = {
            'lr': lr,
            'pretrained_model': pretrained_model,
            'tokenizer':tokenizer, 
            'epochs': epochs,
            'optimizer': optimizer,
            'train_batch_size': train_batch_size,
            'eval_batch_size': eval_batch_size,
            'model':model.to(device),
            'X_train_texts':X_train_texts,
            'X_test_texts':X_test_texts,
            'y_train':y_train,
            'y_test':y_test,
            'debug':debug,
            'tokenizer':tokenizer
    }

    train = CustomTrainer(params)
    train.train()