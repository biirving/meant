#!/bin/bash

# change file paths
output_filepath='/work/nlp/b.irving/nlp_files/output_files'
elim=10
filepath='/work/nlp/b.irving/nlp_files'

# change according to model, dataset
model_name='bert_ner'
hfm='dslim/bert-base-ner'
hft='dslim/bert-base-ner'
dataset='conll2003'
train='train'
test='test'
num_classes=9

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
        job=`sbatch --mem=10G --time=$jobtime -p gpu --gres=gpu:p100:1 --output=$output_filepath/$dataset/$train/$output_file'_'%j.out checkpoint_train.py --file_path=$filepath --model_name=$model_name --hugging_face_model=$hfm --hugging_face_tokenizer=$hft --hugging_face_data=$dataset --epoch=$epoch --pretrained=False --run_id=$model_id --num_classes=$num_classes | awk '{print $NF}'`
        jobnext=$job
        echo $job
      else 
        job=`sbatch --mem=10G --time=$jobtime -d afterany:$jobnext -p gpu --gres=gpu:p100:1 --output=$output_filepath/$dataset/$train/$output_file'_'%j.out checkpoint_train.py --file_path=$filepath --model_name=$model_name --hugging_face_model=$hfm --hugging_face_tokenizer=$hft --hugging_face_data=$dataset --epoch=$epoch --run_id=$model_id --num_classes=$num_classes | awk '{print $NF}'`
        jobnext=$job
        echo $job
	      if [ $epoch == $elim ]; then
          echo 'testing'
          job=`sbatch --mem=10G --time=$jobtime -d afterany:$jobnext -p gpu --gres=gpu:p100:1 --output=$output_filepath/$dataset/$test/$output_file'_'%j.out test.py --file_path=$filepath --model_name=$model_name --hugging_face_tokenizer=$hft --hugging_face_data=$dataset --num_classes=9 --epoch=$epoch --run_id=$model_id --num_classes=$num_classes| awk '{print $NF}'`
	        echo $job
	      fi
      fi
done
