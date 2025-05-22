import os
import argparse
from datasets import load_dataset
import evaluate
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import torch

from huggingface_hub import login
from typing import List

from utils import Trainer_Distill, TrainingArguments_Distill


import os 

parser = argparse.ArgumentParser(description="Finetune BERT-base on a GLUE task.")
parser.add_argument('--task', type=str, required=True, help='GLUE task name (e.g., mrpc, sst2, mnli, etc.)')
parser.add_argument('--maxsequence', type=int, default=256, help='Maximum sequence length for tokenization')
parser.add_argument('--num_train_epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--batchsize', type=int, default=32, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning ratio for the last layer')
parser.add_argument('--learning_rate_final', type=float, default=5e-5, help='learning ratio for the last layer')
parser.add_argument('--teacher_model_path', type=str, help='Path to the teacher model checkpoint')
parser.add_argument('--qat', type=int, default=0, help='0: no qat, 1: qat with weights, 2: qat with weights & activation')


torch.cuda.manual_seed(42)

args = parser.parse_args()

task_name = args.task.lower()
max_seq_length = args.maxsequence


if args.qat == 0:
    use_qat = False
elif args.qat == 1:
    use_qat = True
    use_qat_activation = False
elif args.qat == 2:
    use_qat = True
    use_qat_activation = True

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import copy
tokenizer = AutoTokenizer.from_pretrained(f"JeremiahZ/bert-base-uncased-{args.task}")
teacher_model = AutoModelForSequenceClassification.from_pretrained(f"JeremiahZ/bert-base-uncased-{args.task}")
model_tensor = copy.deepcopy(teacher_model)


TTM_dims = [[16,20,10,10],[4,4,8,6]]
TTM_ranks = [1,20,20,20,1]
TT_dims_att = [24,32,32,24]
TT_ranks_att = [1,24,30,24,1]
TT_dims_ffn = [32,24,48,64]
TT_ranks_ffn = [1,30,30,30,1]

from utils_tensor_layers import get_tensor_model, set_quantization_aware_model
get_tensor_model(model_tensor,TT_dims_att,TT_ranks_att,TT_dims_ffn,TT_ranks_ffn,TTM_dims,TTM_ranks)

if use_qat:
    set_quantization_aware_model(model_tensor,bit_cores=4,bit_intermediate=8,q_activation=use_qat_activation)



# print(model_tensor)

# for n,p in model_tensor.named_parameters():
#     if 'cores' in n:
#         print(f"training {n}")
#         p.requires_grad = True
#     else:
#         p.requires_grad = False

# Test forward pass for model_tensor
# Create a dummy input batch matching the expected input shape
from transformers import BertTokenizerFast



# Set device to CUDA if available
device = 'cuda'

model_tensor.to(device)
teacher_model.to(device)

dataset = load_dataset("glue", task_name)

sentence1_key, sentence2_key = None, None
if task_name in ["cola", "sst2"]:
    sentence1_key = "sentence"
    sentence2_key = None
elif task_name in ["mrpc", "rte"]:
    sentence1_key, sentence2_key = "sentence1", "sentence2"
elif task_name == "qqp":
    sentence1_key, sentence2_key = "question1", "question2"  # Fix for QQP
elif task_name == "stsb":
    sentence1_key, sentence2_key = "sentence1", "sentence2"
elif task_name in ["mnli", "mnli-mm", "mnli-m"]:
    sentence1_key, sentence2_key = "premise", "hypothesis"
elif task_name == "qnli":
    sentence1_key, sentence2_key = "question", "sentence"  # Fix for QNLI
elif task_name == "wnli":
    sentence1_key, sentence2_key = "sentence1", "sentence2"
else:
    # Try to infer keys
    keys = list(dataset["train"].features.keys())
    keys = [k for k in keys if k != "label"]
    if len(keys) == 1:
        sentence1_key = keys[0]
    elif len(keys) >= 2:
        sentence1_key, sentence2_key = keys[:2]
    else:
        raise ValueError(f"Cannot determine input keys for task {task_name}")

# Tokenize datasets
def preprocess_function(examples):
    return tokenizer(
        examples[sentence1_key],
        examples[sentence2_key] if sentence2_key else None,
        truncation=True,
        padding="max_length",  # Pad all sequences to the same length
        max_length=128,        # Ensure all sequences are truncated to max_length
    )

encoded_dataset = dataset.map(preprocess_function, batched=True)

from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr
import numpy as np

# Compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits)  # Convert logits to a PyTorch tensor
    labels = torch.tensor(labels)

    if task_name == "cola":
        # Compute Matthews Correlation for CoLA
        predictions = torch.argmax(logits, axis=-1)
        mcc = matthews_corrcoef(labels.numpy(), predictions.numpy())
        return {"matthews_correlation": mcc}

    elif task_name in ["mrpc", "qqp"]:
        # Compute accuracy and F1 score for MRPC and QQP
        predictions = torch.argmax(logits, axis=-1)
        accuracy = (predictions == labels).float().mean().item()
        f1 = f1_score(labels.numpy(), predictions.numpy(), average="weighted")
        return {"accuracy": accuracy, "f1": f1}

    elif task_name in ["sst2", "rte", "qnli", "mnli"]:
        # Compute accuracy for other classification tasks
        predictions = torch.argmax(logits, axis=-1)
        accuracy = (predictions == labels).float().mean().item()
        return {"accuracy": accuracy}

    elif task_name == "stsb":
        # Compute Pearson Correlation and Mean Squared Error for STS-B
        predictions = logits.squeeze()
        pearson_corr, _ = pearsonr(predictions.numpy(), labels.numpy())
        mse = torch.mean((predictions - labels) ** 2).item()
        return {"pearson": pearson_corr, "mse": mse}

    else:
        raise ValueError(f"Task {task_name} not supported for metric computation.")

# evaluate the teacher model
print("Evaluating teacher model...")
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Keep evaluation strategy as "epoch"
    per_device_eval_batch_size=16,
    logging_dir="./logs",
)
# Handle MNLI validation splits
if task_name == "mnli":
    validation_key = "validation_matched"  # Use "validation_mismatched" for out-of-domain validation
else:
    validation_key = "validation"
trainer = Trainer(
    model=teacher_model,
    args=training_args,
    eval_dataset=encoded_dataset[validation_key],  # Pass the evaluation dataset here
    compute_metrics=compute_metrics,
)
# Perform evaluation
results = trainer.evaluate()
print("Teacher Model Evaluation Results:", results)
print("Finish evaluating teacher model...")

# Dynamically set metric_for_best_model based on the task
if task_name == "cola":
    metric_for_best_model = "matthews_correlation"
elif task_name in ["mrpc", "qqp"]:
    metric_for_best_model = "f1"  # You can also use "accuracy" if preferred
elif task_name == "stsb":
    metric_for_best_model = "pearson"
else:
    metric_for_best_model = "accuracy"

# Initialize distillation trainer
# Training arguments
training_args = TrainingArguments_Distill(
    output_dir=f"./results/{task_name}_distill",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=100,
    save_steps=1000000,
    save_total_limit=1,
    learning_rate=args.learning_rate,
    learning_rate_final=args.learning_rate_final,
    per_device_train_batch_size=args.batchsize,
    per_device_eval_batch_size=args.batchsize,
    num_train_epochs=args.num_train_epochs, 
    weight_decay=0,
    warmup_steps=0,
    lr_scheduler_type="constant",
    logging_dir=f'./logs/{task_name}_distill',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model=metric_for_best_model,
    fp16=False,
    bf16=False,
    max_grad_norm=10.0,
    save_safetensors=False,  # Disable safe serialization
    run_name=f"{task_name}_distill_qat{args.qat}_bs{args.batchsize}"  # Custom wandb run name
)

num_train_examples = len(encoded_dataset["train"])
training_args.max_steps = int(num_train_examples / training_args.per_device_train_batch_size * training_args.num_train_epochs)
print("Training steps: ", training_args.max_steps)
steps_per_layer = training_args.max_steps // len(teacher_model.bert.encoder.layer) // 2
print("Steps per layer: ", steps_per_layer)
training_args.steps_per_layer = steps_per_layer

trainer = Trainer_Distill(
    teacher_model=teacher_model,
    student_model=model_tensor,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()



