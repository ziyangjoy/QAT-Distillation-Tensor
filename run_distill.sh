#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU




task=rte
# SST-2
echo "Running distillation for ${task}..."
python -u distill.py \
    --task ${task} \
    --maxsequence 128 \
    |tee temp.log
