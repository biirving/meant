#!/bin/bash

hug=True
model_name=vilt
sbatch -p gpu --gres=gpu:a100:1 --time=08:00:00 --mem=28GB \
--output=/work/nlp/b.irving/meant_runs/output_files/$model_name-%j-vqa.out \
vqa.py --hugging_face_model=$hug --model_name=$model_name --num_epochs=1 --track=True --train_batch_size=64
