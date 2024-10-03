#!/bin/bash

# change file paths
output_filepath='/work/nlp/b.irving/nlp_files/output_files'
filepath='/work/nlp/b.irving/nlp_files'

# change according to model, dataset
model_name='roberta_tweet'
hfm='tner/roberta-large-tweetner7-all'
hft='tner/roberta-large-tweetner7-all'
bert_ner='dslim/bert-base-NER'
biob='dmis-lab/biobert-v1.1'
#dataset='adsabs/WIESP2022-NER'
#dataset_name='wiesp'
in_loop='in_loop'
num_classes=9

# we will be training a lot of the same models
# generate a unique id that can connect all of the output files
model_id_number=$(shuf -i 100000-999999 -n 1)
model_id="$model_id_number"
echo $model_id
jobtime='08:00:00'
output_file=$model_name'_'$model_id'_in_loop' 
sbatch -p gpu --time=08:00:00 --mem=32GB --gres=gpu:p100:1 \
--output=/work/nlp/b.irving/nlp_files/output_files/tweet7/in_loop/'roberta-tweet'-%j.out \
tweet7.py \
--hugging_face_model=$hfm \
--hugging_face_tokenizer=$hfm \
--hugging_face_data='tner/tweetner7' \
--epoch=0 \
--file_path='/work/nlp/b.irving/nlp_files' \
--model_name='bert_ner' \
--pretrained=True \
--run_id=$model_id \
--num_classes=15 \
--train_batch_size=1 \
--decay=1e-7 \
--lr_warmup_step_ratio=0.15 \
--learning_rate=1e-5 \
--num_epochs=200 
