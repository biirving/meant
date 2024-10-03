#!/bin/bash

# change file paths
output_filepath='/work/nlp/b.irving/nlp_files/output_files'
elim=30
filepath='/work/nlp/b.irving/nlp_files'

# change according to model, dataset
model_name='bert_ner'
hfm='dslim/bert-base-ner'
hft='dslim/bert-base-ner'
#dataset='conll2003'
dataset='adsabs/WIESP2022-NER'
dataset_name='wiesp'
train='train'
test='test'
num_classes=63

# we will be training a lot of the same models
# generate a unique id that can connect all of the output files
model_id_number=$(shuf -i 100000-999999 -n 1)
model_id="$model_id_number"
echo $model_id
for ((epoch=0; epoch<=$elim; epoch++))
   do
      jobtime='05:00:00'
  	  output_file=$model_name'_'$model_id'_'$epoch 
      if [ $epoch == 0 ]; then
        to_check_output_file="Irrelevant"
        job=`sbatch --mem=10G --time=$jobtime -p gpu --gres=gpu:p100:1 --output=$output_filepath/$dataset_name/$train/$output_file'_'%j.out checkpoint_train.py --file_path=$filepath --model_name=$model_name --hugging_face_model=$hfm --hugging_face_tokenizer=$hft --hugging_face_data=$dataset --epoch=$epoch --pretrained=False --run_id=$model_id --num_classes=$num_classes --prev_output_file=$to_check_output_file| awk '{print $NF}'`
        jobnext=$job
        to_check_output_file=$output_filepath/$dataset_name/$train/$model_name'_'$model_id'_'$epoch'_'$job'.out'
        echo $job
      else 
        # check for early stoppage. In the case that there is minimal improvement, we want to go directly to the test step
        # we have to propogate some sort of cancellation through the uneeded jobs, and process the test on the correct epoch
        job=`sbatch --mem=11G --time=$jobtime -d afterany:$jobnext -p gpu --gres=gpu:p100:1 --output=$output_filepath/$dataset/$train/$output_file'_'%j.out checkpoint_train.py --file_path=$filepath --model_name=$model_name --hugging_face_model=$hfm --hugging_face_tokenizer=$hft --hugging_face_data=$dataset --epoch=$epoch --run_id=$model_id --num_classes=$num_classes --prev_output_file=$to_check_output_file| awk '{print $NF}'`
        jobnext=$job
        to_check_output_file=$output_filepath/$dataset_name/$train/$model_name'_'$model_id'_'$epoch'_'$job'.out'
        echo $job
	      if [ $epoch == $elim ]; then
          echo 'testing'
          job=`sbatch --mem=10G --time=$jobtime -d afterany:$jobnext -p gpu --gres=gpu:p100:1 --output=$output_filepath/$dataset/$test/$output_file'_'%j.out test.py --file_path=$filepath --model_name=$model_name --hugging_face_tokenizer=$hft --hugging_face_data=$dataset --num_classes=9 --epoch=$test_epoch --run_id=$model_id --num_classes=$num_classes --prev_output_file=$to_check_output_file| awk '{print $NF}'`
	        echo $job
	      fi
      fi
done
