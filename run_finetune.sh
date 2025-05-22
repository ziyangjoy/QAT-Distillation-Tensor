#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=1  # Use first GPU




task=rte
maxsequence=128

# SST-2
echo "Running distillation for ${task}..."
python -u finetune.py \
    --task ${task} \
    --maxsequence ${maxsequence} | tee logs/finetune_${task}_${maxsequence}.log
