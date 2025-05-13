#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU




task=sst2
# SST-2
echo "Running distillation for ${task}..."
python -u distill.py \
    --task ${task} \
    --maxsequence 128 \
    --teacher_model_path /network/rit/lab/ziyang_lab/ziyang/github/Quantize_Tensor/finetuned_models/${task}_128 \
    |tee temp.log
