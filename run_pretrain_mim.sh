#!/bin/bash

model_name='meant_vision_encoder'

sbatch -p gpu --time=08:00:00 --mem=32GB --gres=gpu:a100:1 \
--output=/work/nlp/b.irving/meant_runs/output_files/$model_name-%j.out \
pretrain_mim.py \
--model_name=$model_name \
--num_encoders=12 \
--track='True' \
--batch_size=16 \