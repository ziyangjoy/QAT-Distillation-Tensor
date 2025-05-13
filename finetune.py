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
from utils import preprocess_function


import os 
os.environ["HF_HOME"] = '/network/rit/lab/ziyang_lab/ziyang/dataset_cache'
model_dir = "/network/rit/lab/ziyang_lab/ziyang/models_cache"
data_dir = "/network/rit/lab/ziyang_lab/ziyang/dataset_cache"

def main():
    parser = argparse.ArgumentParser(description="Finetune BERT-base on a GLUE task.")
    parser.add_argument('--task', type=str, required=True, help='GLUE task name (e.g., mrpc, sst2, mnli, etc.)')
    parser.add_argument('--maxsequence', type=int, default=256, help='Maximum sequence length for tokenization')
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
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels, cache_dir=model_dir)
    print(model)
    # Set device to CUDA if available
    device = 'cuda'
    model.to(device)

    # Determine input keys for the task
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

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{task_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir=f'./logs/{task_name}',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy" if task_name != "stsb" else "pearson",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # Train
    trainer.train()

    # Save the best model
    trainer.save_model(f"./finetuned_models/{task_name}_{args.maxsequence}")

    # Evaluate
    results = trainer.evaluate()
    print(results)

if __name__ == "__main__":
    main()
