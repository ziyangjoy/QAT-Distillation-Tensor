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

from utils import preprocess_function, Trainer_Distill, TrainingArguments_Distill


import os 
os.environ["HF_HOME"] = '/network/rit/lab/ziyang_lab/ziyang/dataset_cache'
model_dir = "/network/rit/lab/ziyang_lab/ziyang/models_cache"
data_dir = "/network/rit/lab/ziyang_lab/ziyang/dataset_cache"

parser = argparse.ArgumentParser(description="Finetune BERT-base on a GLUE task.")
parser.add_argument('--task', type=str, required=True, help='GLUE task name (e.g., mrpc, sst2, mnli, etc.)')
parser.add_argument('--maxsequence', type=int, default=256, help='Maximum sequence length for tokenization')
parser.add_argument('--teacher_model_path', type=str, help='Path to the teacher model checkpoint')
args = parser.parse_args()

task_name = args.task.lower()
max_seq_length = args.maxsequence

# Load dataset and metric
raw_datasets = load_dataset("glue", task_name, cache_dir=data_dir)
metric = evaluate.load("glue", task_name)

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", cache_dir=model_dir)
label_list = raw_datasets["train"].features["label"].names if hasattr(raw_datasets["train"].features["label"], 'names') else None
num_labels = len(label_list) if label_list else len(set(raw_datasets["train"]["label"]))
print(f"num_labels: {num_labels}")

# Load teacher model (finetuned BERT)
if args.teacher_model_path:
    teacher_model = BertForSequenceClassification.from_pretrained(args.teacher_model_path)
    model_tensor = BertForSequenceClassification.from_pretrained(args.teacher_model_path)
else:
    teacher_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels, cache_dir=model_dir)
    model_tensor = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels, cache_dir=model_dir)
# Initialize student model (tensor model)


TTM_dims = [[16,20,10,10],[4,4,8,6]]
TTM_ranks = [1,20,20,20,1]
TT_dims_att = [24,32,32,24]
TT_ranks_att = [1,24,30,24,1]
TT_dims_ffn = [32,24,48,64]
TT_ranks_ffn = [1,30,30,30,1]

from utils_tensor_layers import get_tensor_model, set_quantization_aware_model
get_tensor_model(model_tensor,TT_dims_att,TT_ranks_att,TT_dims_ffn,TT_ranks_ffn,TTM_dims,TTM_ranks)
set_quantization_aware_model(model_tensor,bit_cores=8,bit_intermediate=8)
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

sentence1_key, sentence2_key = None, None
if task_name in ["cola", "sst2"]:
    sentence1_key = "sentence"
    sentence2_key = None
elif task_name in ["mrpc", "qqp",'rte']:
    sentence1_key, sentence2_key = "sentence1", "sentence2"
elif task_name == "stsb":
    sentence1_key, sentence2_key = "sentence1", "sentence2"
elif task_name in ["mnli", "mnli-mm", "mnli-m", "qnli", "wnli"]:
    sentence1_key, sentence2_key = "premise", "hypothesis"
else:
    # Try to infer keys
    keys = list(raw_datasets["train"].features.keys())
    keys = [k for k in keys if k != "label"]
    if len(keys) == 1:
        sentence1_key = keys[0]
    elif len(keys) >= 2:
        sentence1_key, sentence2_key = keys[:2]
    else:
        raise ValueError(f"Cannot determine input keys for task {task_name}")

# Tokenize datasets


map_fn = lambda examples: preprocess_function(examples,tokenizer,max_seq_length,sentence1_key,sentence2_key=sentence2_key)
encoded_datasets = raw_datasets.map(map_fn, batched=True)
encoded_datasets = encoded_datasets.remove_columns(
    [col for col in encoded_datasets["validation"].column_names if col not in ["input_ids", "attention_mask", "label","token_type_ids"]]
)

print(encoded_datasets['validation'][0])

from torch.nn.utils.rnn import pad_sequence
pad_token_id = tokenizer.pad_token_id
padding_side = "right"
def collate_fn(batch):
    input_ids = []
    attention_mask = []
    target_ids = []
    token_type_ids = []
    
    for b in batch:
        input_ids.append(torch.tensor(b['input_ids']))
        attention_mask.append(torch.tensor(b['attention_mask']))
        target_ids.append(torch.tensor(b['label']))
        token_type_ids.append(torch.tensor(b['token_type_ids']))
    input_ids = torch.swapaxes(pad_sequence(input_ids, padding_value=pad_token_id, padding_side=padding_side),0,1)
    attention_mask = torch.swapaxes(pad_sequence(attention_mask, padding_value=0, padding_side=padding_side),0,1)
    # target_ids = torch.swapaxes(pad_sequence(target_ids, padding_value=-100, padding_side=padding_side),0,1)
    target_ids = torch.tensor(target_ids)
    token_type_ids = torch.swapaxes(pad_sequence(token_type_ids, padding_value=0, padding_side=padding_side),0,1)
    return {'input_ids': input_ids, 'attention_mask':attention_mask,'labels':target_ids,'token_type_ids':token_type_ids}

print(encoded_datasets)

# Data collator
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
data_collator = collate_fn

# Compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # predictions = predictions[0]
    predictions = predictions.argmax(axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Training arguments
training_args = TrainingArguments_Distill(
    output_dir=f"./results/{task_name}_distill",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=100,
    save_steps=1000000,
    save_total_limit=1,
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=500,
    weight_decay=0,
    warmup_steps=0,
    lr_scheduler_type="constant",
    logging_dir=f'./logs/{task_name}_distill',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy" if task_name != "stsb" else "pearson",
    fp16=False,
    bf16=False,
    max_grad_norm=10.0,
    steps_per_layer=2000
)

# Initialize distillation trainer
trainer = Trainer_Distill(
    teacher_model=teacher_model,
    student_model=model_tensor,
    args=training_args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()



