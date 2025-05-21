#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU




task=sst2
qat=1
maxsequence=128

# SST-2
echo "Running distillation for ${task}..."
python -u distill.py \
    --task ${task} \
    --maxsequence ${maxsequence} \
    --qat ${qat} \
    --teacher_model_path /network/rit/lab/ziyang_lab/ziyang/github/Quantize_Tensor/finetuned_models/${task}_${maxsequence} \
    |tee logs/${task}_${maxsequence}_qat${qat}.log
