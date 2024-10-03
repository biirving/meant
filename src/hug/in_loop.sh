#!/bin/bash

# change file paths
output_filepath='/work/nlp/b.irving/nlp_files/output_files'
filepath='/work/nlp/b.irving/nlp_files'

# change according to model, dataset
model_name='bert_ner'
hfm='dslim/bert-base-ner'
hft='dslim/bert-base-ner'

biob='dmis-lab/biobert-v1.1'
dataset='conll2003'
dataset_name='conll2003'
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
--output=/work/nlp/b.irving/nlp_files/output_files/jnlpba/in_loop/'bert_ner_jnlpba'-%j.out \
in_loop_genia.py \
--hugging_face_model=$hfm \
--hugging_face_tokenizer=$hfm \
--hugging_face_data='siddharthtumre/jnlpba-split' \
--epoch=0 \
--file_path='/work/nlp/b.irving/nlp_files' \
--model_name='bert_ner' \
--pretrained=False \
--run_id=$model_id \
--num_classes=11 \
--train_batch_size=4 \
--num_epochs=23 
