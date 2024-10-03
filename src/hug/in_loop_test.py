#!/usr/bin/env python
import os, sys, argparse, time, gc, socket
import numpy as np
import torch
from torch import nn, tensor
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from torchmetrics import Accuracy, MatthewsCorrCoef, AUROC
from tqdm import tqdm
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
import matplotlib.pyplot as plt

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# a simple class for testing our model
class test():
    def __init__(self, params):
        self.test = params['test']
        self.X_test = params['X_test']
        self.y_test = params['y_test']
        self.num_classes = params['num_classes']
        self.model = params['model']
        self.tokenizer = params['tokenizer']
        self.file_path = params['file_path']
        self.dataset = params['dataset']
        self.run_id = params['run_id']
        self.epoch = params['epoch']
        self.model_name = params['model_name']
        self.test_batch_size = params['test_batch_size']

    def tokenize_and_align_labels(self, examples):
        list_len = [len(i) for i in examples['tokens']]
        max_length = max(list_len)
        max_length = 256
        if(len(list_len) == 1):
            tokenized_inputs = tokenizer(examples['tokens'], padding=True, is_split_into_words=True, return_tensors='pt')
        else:
            tokenized_inputs = tokenizer(examples['tokens'], padding=True, truncation=True, is_split_into_words=True, return_tensors='pt')
        labels = []
        accuracy_labels = []
        for i, label in enumerate(examples['ner_tags']):
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
        tokenized_inputs["ner_tags"] = labels
        return tokenized_inputs    

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
        plt.savefig(self.file_path + '/output_files/' + self.dataset + '/plots/' + 'f1_' + self.model_name + '_' + self.run_id + '_' + str(self.epoch) + '.png')
 
    def run_test(self):
        f1_type = 'micro'
        precision = MulticlassPrecision(average=f1_type,num_classes=self.num_classes).to(device)
        recall = MulticlassRecall(average=f1_type, num_classes=self.num_classes).to(device)
        f1 = MulticlassF1Score(num_classes=self.num_classes, average=f1_type).to(device)
        macro_precision = MulticlassPrecision(average='macro',num_classes=self.num_classes).to(device)
        macro_recall = MulticlassRecall(average='macro', num_classes=self.num_classes).to(device)
        macro_f1 = MulticlassF1Score(num_classes=self.num_classes, average='macro').to(device)
        
        f1_scores = []
        accuracy = Accuracy(task='multiclass', num_classes=self.num_classes).to(device)
        x_final_input = self.tokenize_and_align_labels(self.test[0:0+self.test_batch_size])
        #self.model = torch.jit.trace(self.model, ('input_ids':x_final_input['input_ids'].to(device), 'attention_mask':x_final_input['attention_mask'].to(device)), strict=False)
        with torch.no_grad():
            for y in tqdm(range(0, len(self.y_test) - self.test_batch_size, self.test_batch_size)):
                x_final_input = self.tokenize_and_align_labels(self.test[y:y+self.test_batch_size])
                out = self.model(x_final_input['input_ids'].to(device))
                for i, vec in enumerate(out['logits']):
                    target = torch.tensor(x_final_input['ner_tags'][i]).to(device)
                    # metric computation
                    #mask = (target != -100) 
                    mask = (target != 0) & (target != -100)
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
    torch.set_float32_matmul_precision('high')  
    parser = argparse.ArgumentParser()
    parser.add_argument('-nc', '--num_classes', type=int, help='Number of classes that we are going to use in our model.', default=2)
    parser.add_argument('-hfd', '--hugging_face_data', type=str, help='Hugging Face Dataset', default=None)
    parser.add_argument('-e', '--epoch', type=int, help='Epoch of the model to test', default=None)
    parser.add_argument('-fp', '--file_path', type=str, help='Path to files',required=True)
    parser.add_argument('-mn', '--model_name', type=str, help='Model name', required=True)
    parser.add_argument('-hft', '--hugging_face_tokenizer', type=str, help='Hugging face tokenizer', required=True)
    parser.add_argument('-rid', '--run_id', type=str, help='Run identification number', required=True)
    parser.add_argument('-b', '--test_batch_size', type=int, help='Test batch size', default=8)
    parser.add_argument('pvo', '--prev_output_file', type=str, help='Previous output file', required=True)
    args = parser.parse_args()

    if(args.hugging_face_data is not None):
        dataset_name = args.hugging_face_data
        data = load_dataset(dataset_name)
        test_data = data['test']
        X_test = test_data['tokens']
        y_test = test_data['ner_tags']
        
    tokenizer = AutoTokenizer.from_pretrained(args.hugging_face_tokenizer)
    epoch = args.epoch
    # we need to get the right epoch for the job
    if(args.epoch > 0):
    # Define the word you want to search for
        search_word = "Cancellation"
        file_path = args.prev_output_file  
        with open(file_path, 'r') as file:
            file_content = file.read()
        if search_word in file_content:
            index = file_content.index("Epoch number:")
            # Extract the number that follows "Epoch number:"
            number_start = index + len("Epoch number:")
            number_end = number_start + file_content[number_start:].find('\n')
            epoch_number_str = file_content[number_start:number_end]
            # Convert the extracted string to an integer
            epoch = int(epoch_number_str)
    model = torch.load(args.file_path + '/models/' + args.model_name + '/' + args.model_name + '_' + args.run_id + '_' + str(epoch + 1) + '.pt')
    #model = AutoModelForTokenClassification.from_pretrained('dslim/bert-base-ner')
    #model = torch.compile(model).to(device)
    #model_jit = torch.jit.trace({"model":model, example_inputs:})

    params = {
        'test':test_data,
        'X_test': X_test,
        'y_test': y_test,
        'num_classes':args.num_classes,
        'model': model.to(device),
        'tokenizer':tokenizer,
        'run_id':args.run_id,
        'dataset': dataset_name,
        'epoch':args.epoch,
        'run_id':args.run_id,
        'file_path':args.file_path,
        'model_name':args.model_name,        
        'test_batch_size':args.test_batch_size
    }

    test = test(params)
    test.run_test()

#python test.py --hugging_face_tokenizer='dslim/bert-base-ner' --hugging_face_data='conll2003' --epoch=0 --file_path='/work/nlp/b.irving/nlp_files' --model_name='bert_ner' --run_id='000000' --num_classes=9
