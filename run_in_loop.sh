#!/bin/bash

jobtime='08:00:00'
output_filepath='/work/nlp/b.irving/meant_runs/output_files/'
filepath='/work/nlp/b.irving/meant_runs/'
model_id_number=$(shuf -i 100000-999999 -n 1)
model_id="$model_id_number"
num_encoders=12
optimizer='Adam'
model_name='meant_vision'
img=False
lang=False
job=$(sbatch --mem=30G \
             --time=$jobtime \
             -p gpu \
             --gres=gpu:p100:1 \
             --output="${output_filepath}$model_id-$num_encoders-%j.out" \
             in_loop_train.py \
             --num_epochs=40 \
             --normalize=True \
             --num_encoders=$num_encoders \
             --optimizer=$optimizer \
             --image_only=$img \
             --language_only=$lang \
             --run_id=$model_id | awk '{print $NF}')
echo $job
