#!/bin/bash


# can pretrain with multiple models
model_name='roberta_mlm'

sbatch -p gpu --time=08:00:00 --mem=128GB --gres=gpu:a100:1 \
--output=/work/nlp/b.irving/nlp_files/output_files/stmhd/$model_name-%j.out \
pretrain_mlm.py \
--model_name=$model_name \
--track='True' \
--batch_size=16 \
