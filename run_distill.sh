#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU




task=mnli
qat=1
maxsequence=128
# learning_rate_final=5e-5
# batchsize=32
# num_train_epochs=5

#  Select learning rate, batch size, and epochs based on the task
case $task in
  "sst2")
    learning_rate=1e-3
    learning_rate_final=5e-5
    batchsize=32
    num_train_epochs=10
    ;; # works well
  "mrpc")
    learning_rate=1e-4
    learning_rate_final=2e-5
    batchsize=8
    num_train_epochs=20
    ;;
  "qnli")
    learning_rate=1e-3
    learning_rate_final=5e-5
    batchsize=32
    num_train_epochs=5
    ;; # works well
  "mnli")
    learning_rate=1e-3
    learning_rate_final=5e-5
    batchsize=32
    num_train_epochs=3
    ;;
  "qqp")
    learning_rate=1e-3
    learning_rate_final=5e-5
    batchsize=32
    num_train_epochs=3
    ;;
  *)
    echo "Task $task not recognized. Please add it to the case statement."
    exit 1
    ;;
esac

echo "Running distillation for ${task}..."
python -u distill.py \
    --task ${task} \
    --maxsequence ${maxsequence} \
    --batchsize ${batchsize} \
    --learning_rate ${learning_rate} \
    --learning_rate_final ${learning_rate_final} \
    --num_train_epochs ${num_train_epochs} \
    --qat ${qat} | tee logs/${task}_${maxsequence}_qat${qat}_${batchsize}_${learning_rate_final}.log
