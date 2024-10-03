#!/bin/bash

jobtime='08:00:00'
output_filepath='..'
filepath='..'
model_id_number=$(shuf -i 100000-999999 -n 1)
model_id="$model_id_number"
num_encoders=1

optimizer='AdamW'
model_name='meant'
hug=False
dataset='Tempstock'
img=False
lang=False

job=$(sbatch --mem=50G \
             --time=$jobtime \
             -p gpu \
             --gres=gpu:a100:1 \
             --output="${output_filepath}$model_name-$dataset-$model_id-$num_encoders-%j.out" \
             in_loop_train.py \
             --num_epochs=6 \
             --normalize=False \
             --num_encoders=$num_encoders \
             --optimizer=$optimizer \
             --image_only=$img \
             --language_only=$lang \
             --hugging_face_model=$hug \
             --model_name=$model_name \
             --dataset=$dataset \
             --run_id=$model_id | awk '{print $NF}')
echo $job
